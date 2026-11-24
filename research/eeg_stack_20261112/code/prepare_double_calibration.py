"""Register the doubled posterior EEG calibration and acquire verified compact training blocks."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from decode_eeg_layout import CHANNELS
from fit_support_covariance import eeg_features
from prepare_expanded_personal import check, training_block
from prepare_registered_subjects import OFFSET, REPO, REV, ROW_BYTES, SHAPE, SIZE, axes, digest
from register_fresh_photo_panel import SUBJECTS, freeze, read, sha

RESULTS = Path(__file__).resolve().parents[1] / "results/object_fidelity"
PUBLIC = RESULTS / "double_calibration_design.json"
PANELS = ("registered", "replicate", "fresh", "prospective")


def add_paths(p):
    for name, value in dict(
        out="outputs/double-calibration",
        expanded="outputs/photo-structure/expanded",
        inputs="outputs/object-transfer",
        source="outputs/official-object-readout/sub-03",
        fresh="outputs/fresh-photo",
        prospective="outputs/prospective-photo",
        references="outputs/object-fidelity",
    ).items():
        p.add_argument("--" + name, type=Path, default=Path(value))


def regpath(a, panel):
    if panel == "fifth":
        return a.out / "fifth/references/fifth_ablation.json"
    root = (
        a.fresh / "references"
        if panel == "fresh"
        else a.prospective / "references"
        if panel == "prospective"
        else a.references
    )
    return root / (panel + "_ablation.json")


def register(a):
    source = a.source / "atm_train_rows.csv"
    official = a.source / "evaluation/originals_manifest.csv"
    with source.open(encoding="utf8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = np.array([r["unique_id"] for r in rows])
    words = names[::10]
    with official.open(encoding="utf8", newline="") as stream:
        official_ids = {r["unique_id"] for r in csv.DictReader(stream)}
    check(
        len(official_ids) == 200
        and len(set(words)) == 1654
        and not set(words) & official_ids
        and np.array_equal(names, np.repeat(words, 10))
        and [int(r["train_row"]) for r in rows] == list(range(16540)),
        "Official held-out IDs or EEG training row identity mismatch",
    )
    framepath = a.fresh / "registration/exclusion_sets.json"
    frame = read(framepath)
    frozen = read(RESULTS / "fresh_panel_design.json")
    check(
        sha(framepath) == frozen["exclusion_sets_sha256"]
        and sha(source) == frozen["inputs_sha256"]["row_manifest"]
        and sha(official) == frozen["inputs_sha256"]["official_test_object_manifest"],
        "Old conservative exclusions differ",
    )
    old = np.asarray(frame["excluded_concept_ids"]["personal_generic_512"])
    for sub in SUBJECTS:
        check(
            np.array_equal(
                read(a.expanded / "features" / (sub + ".json"))["training_concepts"], old
            ),
            "Current personal calibration is not shared across participants",
        )
    earlier = {p: np.unique(np.asarray(read(regpath(a, p))["rows"]) // 10) for p in PANELS}
    all_old = np.concatenate(list(earlier.values()))
    eligible = np.setdiff1d(frame["eligible_concept_ids"], all_old)
    check(
        len(np.unique(all_old)) == 48
        and len(eligible) == 600
        and len(old) == 512
        and not np.intersect1d(old, all_old).size,
        "Wrong extension frame",
    )
    split = read(RESULTS / "shared_photo_eeg_design.json")
    other = read(RESULTS / "ip_denoiser_gradient_design.json")
    check(
        set(split["train_classes"] + split["test_classes"]) == set(old)
        and not set(eligible) & set(other["eval_classes"]),
        "Extra classes intersect another registered generic validation",
    )
    extra = np.sort(np.random.default_rng(260536).choice(eligible, 512, replace=False))
    reserve = np.setdiff1d(eligible, extra)
    rng = np.random.default_rng(260537)
    classes = rng.choice(reserve, 12, replace=False)
    order = np.array([rng.permutation(10) + int(c) * 10 for c in classes]).reshape(12, 5, 2)
    panelrows = order.transpose(1, 0, 2).ravel()
    truth = np.tile(np.repeat(np.arange(12), 2), 5)
    check(
        len(reserve) == 88 and np.array_equal(names[panelrows], words[classes][truth]),
        "Fifth panel draw is invalid",
    )
    fifth = dict(
        words=words[classes].tolist(),
        rows=panelrows.tolist(),
        truth=truth.tolist(),
        folds=np.repeat(np.arange(5), 24).tolist(),
        calibration_rows_by_fold_and_word=[
            order[:, np.arange(5) != f].reshape(12, 8).tolist() for f in range(5)
        ],
    )
    freeze(regpath(a, "fifth"), fifth)
    out = dict(
        method="Increase only personal posterior R4 calibration from 512 to 1024 objects. Both models retain the same 1003 features, 96-dimensional Fisher fit and native cosine to eight measured supports. For generation keep the entire H14 photo predictor, training population of 512, angle, weights, recipient noise seeds and generation controls fixed across word readouts.",
        participants=list(SUBJECTS),
        old_objects=old.tolist(),
        extra_objects=extra.tolist(),
        new_object_draw_seed=260536,
        reserve_objects=reserve.tolist(),
        fifth_draw_seed=260537,
        fifth_objects=classes.tolist(),
        fifth_registration=fifth,
        fifth_registration_sha256=sha(regpath(a, "fifth")),
        official_test_names_disjoint=200,
        exclusion_sources=dict(
            conservative_frame_sha256=sha(framepath),
            row_manifest_sha256=sha(source),
            official200_manifest_sha256=sha(official),
            earlier_registrations_sha256={p: sha(regpath(a, p)) for p in PANELS},
            old_personal_feature_receipts={
                s: sha(a.expanded / "features" / (s + ".json")) for s in SUBJECTS
            },
        ),
        scope="The existing 512 classes are retained intact. The 512 additions exclude each class outside the earlier conservative 624-object sampling frame and all four previous panels; 88 untouched eligible classes remain reserved, including the fixed fifth panel. Previously studied legacy sub-03 training and model pretraining may include these classes. Three participants return.",
        acquisition=dict(
            revision=REV,
            bytes_per_class=10 * ROW_BYTES,
            extra_bytes_per_person=512 * 10 * ROW_BYTES,
            four_trials_per_photo=True,
            photos_per_training_class=10,
        ),
        gate="On all 1440 decisions from four previously used panels, compare 1024 minus 512 word hits with identical queries. Require positive lower bound of a shared class and participant two-way paired percentile 95% bootstrap and a positive net for at least two of three people. 10000 draws, seed 260538. Report per-subject and panel intervals, and class-level hit changes. A gate failure stops before fifth-panel EEG acquisition or any rendering. A passing result advances both models to the registered fifth 12-class panel, with four physical EEG trials, identical eight supports per class and comparison of actual generated JPEGs.",
        fifth_endpoints="12-way EEG words; all-query saved JPEG 1854-name classification and within-class ten-original DINOv2 identification; two-photo assignment on mutually correct word pairs; same uniform and wrong-EEG controls.",
    )
    freeze(PUBLIC, out)
    freeze(a.out / "design.json", out)


def acquire(a, sub):
    d = read(PUBLIC)
    check(sha(a.out / "design.json") == sha(PUBLIC), "Changed calibration lock")
    url = f"{REPO}/resolve/{REV}/Preprocessed_data_250Hz/{sub}/preprocessed_eeg_training.npy"
    source = read(a.inputs / (sub + ".json"))
    priorpath = a.expanded / "features" / (sub + ".json")
    prior = read(priorpath)
    check(
        source["source"] == prior["source"] == url
        and source["revision"] == REV
        and source["source_size"] == SIZE
        and source["source_full_lfs_sha256"] == prior["source_full_lfs_sha256"]
        and sha(priorpath) == d["exclusion_sources"]["old_personal_feature_receipts"][sub],
        "Pinned EEG release or prior calibration changed",
    )
    old, values = eeg_features(a.expanded, sub)
    check(
        np.array_equal(old, d["old_objects"])
        and digest(values.astype("<f4").tobytes()) == prior["train_design_sha256"],
        "Historical 512-class waveform features failed exact replay",
    )
    cache = a.inputs / "ranges" / REV / sub
    end = OFFSET + SHAPE[0] * ROW_BYTES
    primary = [cache / f"0-{OFFSET - 1}.bin", cache / f"{end}-{SIZE - 1}.bin"]
    for p in primary:
        check(
            sha(p) == p.with_suffix(".sha256").read_text().strip(),
            "Changed header or time/channel footer",
        )
    channels, times = axes(*(p.read_bytes() for p in primary))
    check(np.allclose(times[50:], np.arange(250) / 250, atol=1e-7, rtol=0), "Unknown EEG time axis")
    post = [channels.tolist().index(ch) for ch in CHANNELS]
    dest = a.out / "features" / sub
    dest.mkdir(parents=True, exist_ok=True)
    receipt_file = dest.with_suffix(".json")
    previous = read(receipt_file)["raw_sha256"] if receipt_file.is_file() else {}
    classes = d["extra_objects"]
    hashes = {}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        group = {
            pool.submit(training_block, c, dest, cache, url, post, previous.get(str(c))): c
            for c in classes
        }
        for i, job in enumerate(as_completed(group), 1):
            c = group[job]
            features, raw_sha = job.result()
            hashes[c] = raw_sha
            check(
                features.shape == (10, 1003) and features.dtype == np.float32,
                "Incorrect design block",
            )
            if i % 64 == 0:
                print(sub, i, "/512 compact blocks", flush=True)
    raw_hashes = {str(c): hashes[c] for c in classes}
    checks = {}
    for c in classes:
        with np.load(dest / f"{c:04d}.npz", allow_pickle=False) as z:
            checks[str(c)] = digest(z["features"].tobytes())
    freeze(
        receipt_file,
        dict(
            subject=sub,
            source=url,
            revision=REV,
            source_full_lfs_sha256=source["source_full_lfs_sha256"],
            source_size=SIZE,
            design_sha256=sha(PUBLIC),
            original_receipt_sha256=sha(priorpath),
            old_design_sha256=prior["train_design_sha256"],
            header_footer_sha256=[sha(p) for p in primary],
            byte_ranges=[
                [OFFSET + c * 10 * ROW_BYTES, OFFSET + (c + 1) * 10 * ROW_BYTES - 1]
                for c in classes
            ],
            range_class_order=classes,
            raw_sha256=raw_hashes,
            feature_sha256=checks,
            source_content_range_required=True,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("register", "fetch"), default="register", nargs="?")
    add_paths(p)
    p.add_argument("--subject", choices=SUBJECTS, default="sub-04")
    p.add_argument("--workers", choices=range(1, 7), type=int, default=6)
    a = p.parse_args()
    if a.stage == "register":
        register(a)
    else:
        acquire(a, a.subject)
