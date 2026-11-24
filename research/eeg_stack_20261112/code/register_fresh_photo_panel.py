"""Freeze a new 12-object panel from protocol metadata before reading its EEG or images."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 260510
SUBJECTS = ("sub-04", "sub-05", "sub-01")
CODE = Path(__file__).resolve().parent
PUBLIC = CODE.parent / "results/object_fidelity/fresh_panel_design.json"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path, data):
    raw = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Existing registration differs at {path}")
        return
    path.write_bytes(raw)


def register(out, public):
    source = Path("outputs/official-object-readout/sub-03")
    expanded = Path("outputs/photo-structure/expanded/features")
    angle = Path("outputs/photo-structure/residual_angle")
    refs = Path("outputs/object-fidelity")
    manifest = source / "atm_train_rows.csv"
    splitpath = out / "source/current_split_manifest.json"
    files = dict(row_manifest=manifest, historical_split=splitpath)
    if sha(manifest) != "dafc55af7366974feae0e11418680990ff8d74707ec5b89e398fd5a26567c473":
        raise ValueError("Unexpected source row order")
    with manifest.open(encoding="utf8", newline="") as f:
        table = list(csv.DictReader(f))
    universe = np.arange(1654)
    names = np.array([r["unique_id"] for r in table])
    objects = names[::10]
    assert len(set(objects)) == 1654 and np.array_equal(names, np.repeat(objects, 10))
    assert [int(r["train_row"]) for r in table] == list(range(16540))
    test = source / "evaluation/originals_manifest.csv"
    with test.open(encoding="utf8", newline="") as f:
        official = [r["unique_id"] for r in csv.DictReader(f)]
    assert len(set(official)) == 200 and not set(official).intersection(objects)
    files["official_test_object_manifest"] = test
    excluded = {}
    for sub in SUBJECTS:
        receipt = expanded / (sub + ".json")
        files[sub + "_training"] = receipt
        classes = read(receipt)["training_concepts"]
        assert len(set(classes)) == 512 and (
            sub == SUBJECTS[0] or classes == excluded["personal_generic_512"]
        )
        excluded["personal_generic_512"] = classes
    for panel in ("registered", "replicate"):
        src = refs / (panel + "_ablation.json")
        files[panel] = src
        reg = read(src)
        ids = np.unique(np.asarray(reg["rows"]) // 10)
        assert set(objects[ids]) == set(reg["words"]) and len(ids) == 12
        excluded[panel + "_objects"] = ids.tolist()
        excluded[panel + "_legacy_validation"] = np.sort(
            np.random.default_rng(260425).choice(np.setdiff1d(universe, ids), 120, replace=False)
        ).tolist()
    split = read(splitpath)["image_rows"]
    parts = {k: np.unique(np.array(split[k]) // 10) for k in ("T", "D", "V")}
    assert sorted(np.concatenate(list(split.values())).tolist()) == list(range(16540))
    assert [len(parts[k]) for k in ("T", "D", "V")] == [1323, 165, 166]
    excluded["historical_D_and_V"] = np.union1d(parts["D"], parts["V"]).tolist()
    layout = refs / "r4_layout.json"
    files["legacy_rgb_layout"] = layout
    pilot = read(layout)
    for k, rows in pilot["splits"].items():
        ids = np.unique(np.array(rows) // 10)
        assert len(ids) == 20 and set(ids) <= set(excluded["historical_D_and_V"])
        excluded["historical_layout_" + k] = ids.tolist()
    rgb_val = np.random.default_rng(53117).permutation(parts["T"])[::5]
    assert len(rgb_val) == pilot["calibration_concepts"] and pilot["calibration_seed"] == 53117
    excluded["historical_rgb_gain_calibration"] = np.sort(rgb_val).tolist()
    files["radius"] = angle / "radius.json"
    excluded["generic_visual_radius"] = np.unique(
        np.array(read(files["radius"])["generic_rows"]) // 10
    ).tolist()
    files["generic_residual_selection"] = Path(
        "outputs/photo-structure/within_photo/selection.json"
    )
    files["angle_scale"] = angle / "scale.json"
    files["old_render_design"] = angle / "design.json"
    prior = read(files["generic_residual_selection"])
    assert prior["selected"] == [2.0, 0.0] and read(files["angle_scale"])[
        "selection_sha256"
    ] == sha(files["generic_residual_selection"])
    eligible = np.setdiff1d(universe, np.unique(np.concatenate(list(excluded.values()))))
    rng = np.random.default_rng(SEED)
    ids = rng.choice(eligible, 12, replace=False)
    order = np.stack([rng.permutation(i * 10 + np.arange(10)) for i in ids]).reshape(12, 5, 2)
    rows = order.transpose(1, 0, 2).ravel()
    truth = np.tile(np.repeat(np.arange(12), 2), 5)
    folds = np.repeat(np.arange(5), 24)
    support = [order[:, np.arange(5) != f].reshape(12, 8).tolist() for f in range(5)]
    assert np.array_equal(names[rows], objects[ids][truth]) and set(ids) <= set(eligible)
    for f, groups in enumerate(support):
        assert set(np.ravel(groups)) == set(rows[folds != f]) and len(np.unique(groups)) == 96
    ref = out / "references/fresh_ablation.json"
    detail = out / "registration/exclusion_sets.json"
    freeze(
        ref,
        dict(
            words=objects[ids].tolist(),
            rows=rows.tolist(),
            truth=truth.tolist(),
            folds=folds.tolist(),
            calibration_rows_by_fold_and_word=support,
        ),
    )
    freeze(detail, dict(excluded_concept_ids=excluded, eligible_concept_ids=eligible.tolist()))
    design = dict(
        protocol="Third closed 12-object panel drawn before reading its EEG or scoring its pictures. All three previously studied participants see the same new objects and photographs.",
        scope="Holdout for the current personal 512-class model, generic gain selection, and the listed target panels and calibration splits. Historical sub-03 generic fitting or out-of-fold model comparisons, and foundation-model pretraining, may have used these objects.",
        participants=list(SUBJECTS),
        seed=SEED,
        population=1654,
        eligible=len(eligible),
        excluded_union=1654 - len(eligible),
        excluded_counts={k: len(v) for k, v in excluded.items()},
        official_test_concepts_disjoint=200,
        selection="NumPy PCG64 default_rng(260510), uniform choice of 12 without replacement from ascending eligible zero-based concept indices. In draw order, continue the same RNG, permuting the ten official photo rows once per concept. Consecutive pairs form folds 0 to 4.",
        selected_concept_ids=ids.tolist(),
        words=objects[ids].tolist(),
        queries_per_person=120,
        distinct_photos=120,
        folds=5,
        per_object_per_fold=dict(support=8, held_out=2),
        eeg_trials_per_photo=4,
        model="Original personal expanded_r4 Fisher classifier on 512 disjoint generic concepts, 96 dimensions. Use its predicted word to select eight labeled support images. Frozen generic H/14 residual ridge 2.0, whitening 0.0. Apply the original generic angular gain and cap, radius and diffusion schedule.",
        gain=read(files["angle_scale"])["gain"],
        max_angle_radians=read(files["angle_scale"])["max_angle"],
        controls="Own EEG, same-fold same-predicted-word cyclic donor, uniform eight-support prior. Shared row seed 923761 + source row; singleton donor groups remain unchanged.",
        endpoints="Per participant report 12-way EEG classification, full-vocabulary generated-image naming, within-true-class ten-photo retrieval on all 120 queries, and paired assignment for the two same-object held-out photos when both words are correct. Compare own against both controls. Participants share these photos.",
        registration_sha256=sha(ref),
        exclusion_sets_sha256=sha(detail),
        inputs_sha256={k: sha(v) for k, v in files.items()},
    )
    freeze(out / "panel_lock.json", design)
    freeze(public, design)
    print(
        "Locked",
        sha(public),
        "eligible",
        len(eligible),
        "objects",
        objects[ids].tolist(),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/fresh-photo"))
    p.add_argument("--public", type=Path, default=PUBLIC)
    a = p.parse_args()
    register(a.out, a.public)
