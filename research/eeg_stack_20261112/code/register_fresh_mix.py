"""Register four new same-recording-session menus and the prospective sub-04 mixture study."""

import argparse
import csv
from pathlib import Path

import numpy as np
from prepare_scale_trial import PUBLIC, metadata
from prepare_trial_object_inputs import check
from register_fresh_photo_panel import freeze, read, sha

SEED = 260626
SUB = "sub-04"
LOCK = PUBLIC / "fresh_mix_design.json"


def register(data, out):
    base, extra, _ = metadata(data)
    frame = data / "fresh-photo/registration/exclusion_sets.json"
    dbl = PUBLIC / "double_calibration_design.json"
    cohorts = PUBLIC / "cohort_session_matched_design.json"
    personal = data / "object-head-confirmation/candidate_panel_lock.json"
    schedule = data / "object-transfer/sessions.json"
    manifest = data / "official-object-readout/sub-03/atm_train_rows.csv"
    files = dict(
        exclusions=frame,
        original_lists=dbl,
        prior_cohort=cohorts,
        personal_menus=personal,
        sessions=schedule,
        original_photo_rows=manifest,
        fresh=data / "fresh-photo/references/fresh_ablation.json",
        prospective=data / "prospective-photo/references/prospective_ablation.json",
    )
    check(
        sha(frame) == read(PUBLIC / "fresh_panel_design.json")["exclusion_sets_sha256"]
        and sha(manifest)
        == read(PUBLIC / "fresh_panel_design.json")["inputs_sha256"]["row_manifest"],
        "Original sampling frame differs",
    )
    exclusions = read(frame)["excluded_concept_ids"]
    configs = read(dbl)
    blocked = set(configs["old_objects"] + extra + configs["fifth_objects"])
    for study in (read(cohorts), read(personal)):
        blocked.update(c for panel in study["panels"].values() for c in panel["selected_concepts"])
    for tag in ("fresh", "prospective"):
        blocked.update(c // 10 for c in read(files[tag])["rows"])
    for key, ids in exclusions.items():
        if key != "historical_rgb_gain_calibration":
            blocked.update(ids)
    broad = set(range(1654)) - blocked
    strict = broad - set(exclusions["historical_rgb_gain_calibration"])
    decoys = broad - strict
    check(
        len(broad) == 182
        and len(strict) == 35
        and strict <= set(configs["reserve_objects"])
        and not broad & set(base["test_classes"]),
        "Wrong conservative eligibility",
    )
    sessions = np.asarray(read(schedule)["subjects"][SUB]["session_pairs"])
    with manifest.open(encoding="utf-8-sig", newline="") as fp:
        row_names = list(csv.DictReader(fp))
    words = [v["unique_id"] for v in row_names[::10]]
    check(
        len(set(words)) == 1654
        and all(
            int(v["train_row"]) == i and v["unique_id"] == words[i // 10]
            for i, v in enumerate(row_names)
        ),
        "Invalid official row indices",
    )
    rng = np.random.default_rng(SEED)
    panels = {}
    for pair in ((1, 3), (1, 4), (2, 3), (2, 4)):
        primary = sorted(c for c in strict if tuple(sessions[c]) == pair)
        pool = sorted(c for c in decoys if tuple(sessions[c]) == pair)
        supplement = sorted(rng.choice(pool, 12 - len(primary), replace=False).tolist())
        classes = rng.permutation(primary + supplement)
        order = np.stack([10 * c + rng.permutation(10) for c in classes]).reshape(12, 5, 2)
        rows = order.transpose(1, 0, 2).ravel()
        name = f"mix{pair[0]}{pair[1]}"
        ref = dict(
            words=[words[c] for c in classes],
            rows=rows.tolist(),
            folds=np.repeat(np.arange(5), 24).tolist(),
            truth=np.tile(np.repeat(np.arange(12), 2), 5).tolist(),
            calibration_rows_by_fold_and_word=[
                order[:, np.arange(5) != f].reshape(12, 8).tolist() for f in range(5)
            ],
        )
        path = out / "references" / (name + "_ablation.json")
        freeze(path, ref)
        panels[name] = dict(
            session_pair=list(pair),
            object_ids=classes.tolist(),
            words=ref["words"],
            primary_object_ids=primary,
            supplemental_object_ids=supplement,
            registration_sha256=sha(path),
        )
    design = dict(
        subject=SUB,
        seed=SEED,
        sources={k: sha(p) for k, p in files.items()},
        previous_scale_result_sha256=sha(PUBLIC / "scale_trial.json"),
        original_validation_ids_excluded_from_weight_loss=base["test_classes"],
        object_pool=dict(
            generic_fit=1024,
            strict_primary=35,
            only_legacy_rgb_available=147,
            supplementary_selected=13,
            fifth_reserved=12,
        ),
        panels=panels,
        primary_decisions=350,
        all_decisions=480,
        scope="All primary classes exclude every traced old personal/menu confirmation, historical D/V and validation, original 1024, RGB gain calibration, generic radius and protected fifth. The 13 supplemental classes were eligible except for historical sub-03 RGB-gain calibration. All 12 are plausible candidates in every row. The 350 primary rows use only strict classes; the 480-row sensitivity includes all 48 objects. Class identity is used only to construct and score the supplied menus.",
        epochs="All added neural fits use the previously tested architecture and seed 260553, 1500 updates and same-session 12-object episodes. The old comparator is the original 512-object 750-step checkpoint and original 512-object Fisher, standardized score mix 1/2.",
        crossfit="Fit four independent encoders and Fishers. Split the 928 calibration objects (excluding the previously scored 96) by true object, stratifying within recording-session pair. All 96 may enter model training but never supply queries, candidate supports or labels for the global mix scalar. For each fold use the other objects of the 1024 for all normalizers and weights. The two held photos of each class are excluded from every support set. One ten-query-per-object same-session twelve-way OOF evaluation, totaling 9280. Fit final encoder/Fisher on 1024.",
        mixture="Use standardized twelve-candidate cosine scores from Fisher and the trial encoder. Fit exactly one common neural weight w in [0,1] by minimizing OOF mean multinomial negative log likelihood at fixed temperature 2 (multiply logits by 2), plus 0.1*(w-0.5)^2. No selection of temperature, penalty, epochs, checkpoint, fold, subgroup or decision via new panels. Freeze w before any new target inference.",
        arms="Old fixed 512+512 equal mix; new learned mix; new equal mix; new neural only; new Fisher only. Same 4 physical presentations, eight other-photo supports, words, session-pair and test menus. Exchange query EEG for each method with the next class's held photo in menu order, same fold and within-fold photo position, with supports fixed.",
        inference="Predict all 120 rows of every menu without applying the strict/supplemental indicator. Freeze each case before scoring. Retain all 12 alternatives for primary and supplemental photos.",
        inference_gate="Primary new learned mix over old model on all 350 strict photos, weight every row equally. Require positive net and strictly positive 95% lower bound on this paired difference and versus the new method's wrong-query control before formal old/new matched image generation. Equal mix and single components have descriptive contrasts; target performance cannot select a different mixture.",
        uncertainty="10000 cluster-bootstrap samples of whole truth objects independently within the four fixed session menus. Each object retains all ten photos; pool subject-level rows with denominators 350 (primary) and 480 (sensitivity). Seed 260626. One participant and four fixed menus.",
    )
    freeze(LOCK, design)
    freeze(out / "design.json", design)
    print("Locked", sha(LOCK), {k: len(v["primary_object_ids"]) for k, v in panels.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("outputs/fresh-mix"))
    a = p.parse_args()
    register(a.data, a.out)
