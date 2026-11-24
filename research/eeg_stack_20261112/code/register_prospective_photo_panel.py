"""Register a fourth 12-object panel from row metadata alone, prior to data acquisition."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from register_fresh_photo_panel import SUBJECTS, freeze, read, sha

SEED = 260513
RESULTS = Path(__file__).resolve().parents[1] / "results/object_fidelity"
PREDECESSOR_SHA = "e11f1a371894ccf3b31778b2a133aabb6065e27bd26817cd171cf38eb7c960af"


def register(args):
    old = read(args.prior)
    if (
        sha(args.prior) != PREDECESSOR_SHA
        or sha(args.frame) != old["exclusion_sets_sha256"]
        or sha(args.manifest) != old["inputs_sha256"]["row_manifest"]
    ):
        raise ValueError("Changed source registration, exclusion frame, or row metadata")
    frame = read(args.frame)
    prior_eligible = np.asarray(frame["eligible_concept_ids"])
    prior_ids = np.asarray(old["selected_concept_ids"])
    eligible = np.setdiff1d(prior_eligible, prior_ids)
    if len(np.unique(prior_eligible)) != 624 or len(prior_ids) != 12 or len(eligible) != 612:
        raise ValueError("Expected the original 624 eligible objects minus twelve used objects")
    with args.manifest.open(encoding="utf8", newline="") as stream:
        table = list(csv.DictReader(stream))
    names = np.array([r["unique_id"] for r in table])
    if [int(r["train_row"]) for r in table] != list(range(16540)) or not np.array_equal(
        names, np.repeat(names[::10], 10)
    ):
        raise ValueError("Original ten-row THINGS object ordering changed")
    excluded = set().union(*frame["excluded_concept_ids"].values(), prior_ids.tolist())
    if set(eligible) != set(range(1654)) - excluded:
        raise ValueError("Candidate frame differs from the complete set complement")
    rng = np.random.default_rng(SEED)
    ids = rng.choice(eligible, 12, replace=False)
    groups = np.stack([rng.permutation(k * 10 + np.arange(10)) for k in ids]).reshape(12, 5, 2)
    rows = groups.transpose(1, 0, 2).ravel()
    truth = np.tile(np.repeat(np.arange(12), 2), 5)
    folds = np.repeat(np.arange(5), 24)
    words = names[ids * 10]
    support = [groups[:, np.arange(5) != f].reshape(12, 8).tolist() for f in range(5)]
    if len(set(rows)) != 120 or not np.array_equal(names[rows], words[truth]):
        raise ValueError("Prospective photo identities differ from the row table")
    for f, group in enumerate(support):
        if set(np.ravel(group)) != set(rows[folds != f]) or len(np.unique(group)) != 96:
            raise ValueError("Fold contains a duplicate or a held-out query")
    reg = dict(
        words=words.tolist(),
        rows=rows.tolist(),
        truth=truth.tolist(),
        folds=folds.tolist(),
        calibration_rows_by_fold_and_word=support,
    )
    ref = args.out / "references/prospective_ablation.json"
    freeze(ref, reg)
    frame_file = args.out / "registration/eligible_concepts.json"
    freeze(
        frame_file,
        {"eligible_concept_ids": eligible.tolist(), "excluded_prior_panel": prior_ids.tolist()},
    )
    baseline = old["model"]
    full_head = read(args.full_head_design)
    if (
        full_head["subjects"] != list(SUBJECTS)
        or full_head["training_classes"] != 512
        or full_head["classifier_weights"] != {"posterior": 1, "full": 0.25, "cross": 0.25}
    ):
        raise ValueError("Expected the locked 63-channel 512-class readout design")
    design = dict(
        protocol="Prospective fourth 12-object panel. Registration reads earlier registration and eligibility metadata, the original row-to-word manifest, and the full-head candidate design. No recordings, image pixels, image features, model predictions or scores enter this draw.",
        scope="These objects and rows are disjoint from the 512 generic objects of the current posterior and full-head candidates and from the three earlier target panels. The same participants return. Historical sub-03 generic training or foundation-model pretraining may have seen these objects.",
        participants=list(SUBJECTS),
        seed=SEED,
        population=1654,
        prior_eligible=624,
        excluded_previous_draw=12,
        eligible=612,
        selected_concept_ids=ids.tolist(),
        words=words.tolist(),
        sampling="NumPy PCG64 default_rng(260513), draw 12 without replacement from sorted eligible IDs, then continue the same RNG to permute each object's ten rows in draw order. Consecutive pairs give folds 0–4. Output order is fold, object, pair member.",
        distinct_photos=120,
        queries_per_person=120,
        participant_photo_queries=360,
        eeg_trials_per_photo=4,
        folds=5,
        per_object_per_fold={"support": 8, "held_out": 2},
        heldout_rows_by_fold=rows.reshape(5, 24).tolist(),
        registration_sha256=sha(ref),
        eligible_ids_sha256=sha(frame_file),
        baseline=baseline,
        baseline_visual={
            "gain": old["gain"],
            "max_angle_radians": old["max_angle_radians"],
            "radius_sha256": old["inputs_sha256"]["radius"],
            "source_design_sha256": old["inputs_sha256"]["old_render_design"],
        },
        object_candidate=dict(
            name="512-object 63-channel combined readout",
            weights=full_head["classifier_weights"],
            development_panels=full_head["panels"],
            gate="Using all three earlier panels, sum the full-head minus posterior-512 difference in 12-way EEG first-place hits across all 1,080 participant-queries. Require at least +11 and strictly positive per-person net gains in at least two of the three. Both thresholds must hold.",
            if_pass="Run a paired test of the frozen full-head and posterior readouts on this panel. Score each readout's words and generate its own images under the same frozen photo generator, with identical row seeds. Report actual generated-image naming before claiming an image-level improvement.",
            if_fail="Retain posterior-512. The earlier visual-prototype and support-covariance candidates are ineligible for this prospective comparison.",
        ),
        generator_candidate="Any additional image condition must have its own named specification and pass its independently registered generic-held-class control criterion without data from this panel. Otherwise keep the frozen direct H/14-angle baseline. Report each readout change with generator fixed and each generator change with readout fixed.",
        controls="Own query direction, uniform mean of the eight support photographs of the predicted word, or another EEG direction from the same predicted word and fold in frozen cyclic order. A singleton keeps its own donor. The same 923761 + original row noise seed is used across arms. One 512 × 512 saved JPEG per arm and query, four SDXL-Turbo steps.",
        endpoints=dict(
            primary_photo="On all 120 generated JPEGs per person, independent DINOv2-S/14 final normalized CLS raw cosine identifies the original among the ten originals of its actual class. Include incorrectly predicted words. Chance is 10%.",
            object="Report 12-way EEG object hits and independent OpenAI CLIP-L/14 1,854-label naming of the saved JPEGs for every query. Tied labels share credit.",
            strict_pair="For each true object and fold with both held-out words predicted correctly, test one-to-one original-to-JPEG assignment by summed raw cosines. Report CLIP-L/14 and DINOv2; one observation per pair, ties half credit. Record eligible pairs per person and per readout. If comparing two readouts also use their common eligible pairs.",
            controls_and_intervals="Compare own to the uniform and wrong-EEG controls on matched rows or pairs. Report each participant separately, resampling whole object classes. Pooled intervals resample the three participants and twelve shared true object classes separately, retaining all paired arms. Report the all-query endpoints even if few strict pairs are eligible.",
        ),
        acquisition="Publish the registration before acquiring prospective EEG. Evaluation-only query photographs may be opened after query conditions and renderings are frozen. No prospective outcomes are included.",
        inputs_sha256=dict(
            preceding_public_design=sha(args.prior),
            preceding_eligibility=sha(args.frame),
            row_manifest=sha(args.manifest),
            full_head_design_lf=hashlib.sha256(
                args.full_head_design.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest(),
        ),
        reproduce="From the repository root, run python research/eeg_stack_20261112/code/register_prospective_photo_panel.py after the earlier fresh-panel registration. This writes the complete row/support registry, eligible IDs and this read-only registration to outputs/prospective-photo/.",
    )
    freeze(args.out / "panel_lock.json", design)
    freeze(args.public, design)
    print(sha(args.public), sha(ref), words.tolist(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/prospective-photo"))
    p.add_argument("--public", type=Path, default=RESULTS / "prospective_panel_design.json")
    p.add_argument("--prior", type=Path, default=RESULTS / "fresh_panel_design.json")
    p.add_argument(
        "--frame", type=Path, default=Path("outputs/fresh-photo/registration/exclusion_sets.json")
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/official-object-readout/sub-03/atm_train_rows.csv"),
    )
    p.add_argument("--full-head-design", type=Path, default=RESULTS / "full_head_512_design.json")
    register(p.parse_args())
