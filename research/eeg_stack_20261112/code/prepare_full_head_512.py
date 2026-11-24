"""Cache only the full-scalp early/late time bins from independently hashed public raw EEG."""

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from decode_eeg_layout import CHANNELS
from prepare_expanded_personal import design_block
from prepare_registered_subjects import OFFSET, REPO, REV, ROW_BYTES, SHAPE, SIZE, axes, http
from register_fresh_photo_panel import freeze, read, sha

SUBJECTS = ("sub-04", "sub-05", "sub-01")
PANELS = ("registered", "replicate", "fresh")
PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity/full_head_512_design.json"
PREVIOUS = "c60295652887a4d108c167d5dbc0277f63a68413b0425d452e3324347dce8d0c"
WEIGHTS = dict(posterior=1, full=0.25, cross=0.25)


def check(v, msg):
    if not v:
        raise ValueError(msg)


def design(a):
    prior = a.old / "selection.json"
    check(
        sha(prior) == PREVIOUS and read(prior)["selected"] == WEIGHTS,
        "Previous 128-class preregistered weights changed",
    )
    registrations = {
        p: (
            a.fresh / "references/fresh_ablation.json"
            if p == "fresh"
            else a.references / (p + "_ablation.json")
        )
        for p in PANELS
    }
    classes = read(a.expanded / "features/sub-04.json")["training_concepts"]
    for sub in SUBJECTS:
        check(
            read(a.expanded / "features" / (sub + ".json"))["training_concepts"] == classes,
            "Different participant training classes",
        )
    check(
        len(set(classes)) == 512
        and all(
            not set(classes).intersection(np.array(read(f)["rows"]) // 10)
            for f in registrations.values()
        ),
        "Target and generic objects overlap",
    )
    info = dict(
        subjects=list(SUBJECTS),
        panels=list(PANELS),
        distinct_objects=36,
        distinct_target_photos=360,
        preexisting_exposure="All three panels and participants had prior posterior-only or 128-class-model results. No image outcome establishes a benefit for the new combination.",
        training_classes=512,
        training_photos_per_person=5120,
        classifier_weights=WEIGHTS,
        feature_design="Four true trials; released 63-channel order; two float32 two-trial means using stored slots [1,3] and [0,2]; fourteen successive signed 40-ms means over 40 to 600 ms, end-exclusive. Same four physical trials as posterior-only.",
        model="Unchanged 96-D posterior Fisher, 96-D full-head Fisher, and 32-D cross-session/different-photo Fisher, using the same historical train-only whitening, session-mean subtraction, per-query score normalization and 8-shot splits.",
        comparison="Primary comparison: 512 full-head mixture versus 512 posterior-only on the same rows and class supports; report each panel, individual and acquisition-session-conditioned null. Any generated-image comparison is separate.",
        gate="First run sub-04 alone. Process sub-05 and sub-01 only if pooled correct counts on the three 120-query panels strictly exceed the unchanged 512-class posterior baseline. No further weight, window, rank or participant selection.",
        previous_weight_selection_sha256=sha(prior),
        training_receipts_sha256={
            s: sha(a.expanded / "features" / (s + ".json")) for s in SUBJECTS
        },
        registrations_sha256={p: sha(f) for p, f in registrations.items()},
    )
    freeze(a.out / "design.json", info)
    freeze(PUBLIC, info)


def acquisition(a, sub):
    lock = read(PUBLIC)
    check(sha(PUBLIC) == sha(a.out / "design.json"), "Design changed")
    if sub != "sub-04":
        gate = read(a.out / "sub-04_gate.json")
        check(
            gate["delta_hits"] > 0 and gate["design_sha256"] == sha(PUBLIC),
            "First-participant positive gate not met",
        )
    fullpath = a.expanded / "features" / (sub + ".json")
    meta = read(fullpath)
    check(
        sha(fullpath) == lock["training_receipts_sha256"][sub], "Training source differs from lock"
    )
    rawroot = a.inputs / "ranges" / REV / sub
    channel, times = axes(
        (rawroot / f"0-{OFFSET - 1}.bin").read_bytes(),
        (rawroot / f"{OFFSET + SHAPE[0] * ROW_BYTES}-{SIZE - 1}.bin").read_bytes(),
    )
    post = [channel.tolist().index(c) for c in CHANNELS]
    expected = meta["raw_block_sha256"]
    url = f"{REPO}/resolve/{REV}/Preprocessed_data_250Hz/{sub}/preprocessed_eeg_training.npy"
    check(url == meta["source"], "Unexpected raw source")
    out = a.out / "features" / sub
    out.mkdir(parents=True, exist_ok=True)

    def get(c):
        c = int(c)
        path = out / f"{c:04d}.npz"
        expected_sha = expected[str(c)]
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                f = z["half_features"]
                check(
                    str(z["raw_sha"]) == expected_sha
                    and f.shape == (2, 10, 882)
                    and f.dtype == np.float32
                    and str(z["feature_sha"]) == hashlib.sha256(f.tobytes()).hexdigest()
                    and int(z["concept"]) == c,
                    f"Existing full-scalp bank changed at {path}",
                )
                return str(z["feature_sha"])
        first = OFFSET + 10 * c * ROW_BYTES
        last = first + 10 * ROW_BYTES - 1
        key = f"{first}-{last}"
        previous = rawroot / (key + ".bin")
        buf = (
            previous.read_bytes()
            if previous.exists()
            else http(
                url + "?range=" + key,
                {"Range": "bytes=" + key},
                last - first + 1,
                f"bytes {key}/{SIZE}",
            )
        )
        check(
            len(buf) == 10 * ROW_BYTES and hashlib.sha256(buf).hexdigest() == expected_sha,
            f"Physical EEG range differs at {sub}/{c}",
        )
        with np.load(a.expanded / "features" / sub / f"{c:04d}.npz", allow_pickle=False) as z:
            check(
                np.array_equal(design_block(buf, post), z["features"])
                and str(z["raw_sha"]) == expected_sha,
                "Published posterior design could not be reproduced",
            )
        values = np.frombuffer(buf, "<f8").reshape(10, *SHAPE[1:]).astype(np.float32)
        check(np.isfinite(values).all(), "Nonfinite full-scalp physical EEG")
        features = temporal(values)
        checksum = hashlib.sha256(features.tobytes()).hexdigest()
        temporary = path.with_suffix(".partial.npz")
        np.savez(
            temporary, half_features=features, raw_sha=expected_sha, feature_sha=checksum, concept=c
        )
        temporary.replace(path)
        return checksum

    classes = meta["training_concepts"]
    checksums = {}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs = {pool.submit(get, c): c for c in classes}
        for i, f in enumerate(as_completed(jobs), 1):
            checksums[jobs[f]] = f.result()
            if i % 64 == 0:
                print(sub, i, "/512 verified", flush=True)
    freeze(
        out.with_suffix(".json"),
        dict(
            subject=sub,
            training_receipt_sha256=sha(fullpath),
            design_sha256=sha(PUBLIC),
            channels=channel.tolist(),
            window_samples=[10, 150],
            bin_samples=10,
            times_sha256=hashlib.sha256(times[50:].tobytes()).hexdigest(),
            source_full_lfs_sha256=meta["source_full_lfs_sha256"],
            source=url,
            n_blocks=512,
            feature_sha256={str(k): checksums[k] for k in sorted(checksums)},
        ),
    )


def temporal(values):
    h = []
    for u, v in ((1, 3), (0, 2)):
        half = (values[:, u] + values[:, v]) * 0.5
        h.append(half[:, :, 10:150].reshape(10, 63, 14, 10).mean(-1).reshape(10, 882))
    return np.stack(h)


def defaults(p):
    for k, d in dict(
        out="outputs/full-head-512",
        expanded="outputs/photo-structure/expanded",
        old="outputs/photo-structure/full_head",
        inputs="outputs/object-transfer",
        references="outputs/object-fidelity",
        fresh="outputs/fresh-photo",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(d))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("design", "fetch"))
    defaults(p)
    p.add_argument("--subject", choices=SUBJECTS, default="sub-04")
    p.add_argument("--workers", type=int, choices=range(1, 9), default=5)
    a = p.parse_args()
    if a.stage == "design":
        design(a)
    else:
        acquisition(a, a.subject)
