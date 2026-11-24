"""Select two full-scalp EEG score weights on sub-04 generic objects, then replay fixed target panels."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from audit_registered_sessions import audit_panel
from decode_eeg_layout import CHANNELS, eeg_design
from fit_personal_session_metric import load, object_directions, similarities
from prepare_registered_subjects import OFFSET, REV, ROW_BYTES, SHAPE, SIZE, axes
from summarize_personal_metric import paired

SEED = 260436
GRID = (0, 0.25, 0.5, 1, 2)
PANELS = ("registered", "replicate")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")


def features(root, subject, generic):
    path = root / ("generic" if generic else "halves") / f"{subject}.npz"
    rows, half, sessions, mean = load(path)
    receipt = root / ("generic" if generic else "") / f"{subject}.json"
    meta = read(receipt)
    if meta["revision"] != REV or meta["subject"] != subject:
        raise ValueError("Raw EEG revision or participant mismatch")
    if generic:
        if meta["packet_sha256"] != sha(path) or len(rows) != 1280:
            raise ValueError("Unrecognized 128-object EEG packet")
        derived = np.setdiff1d(np.arange(1654), meta["excluded_concepts"])
        draw = np.sort(np.random.default_rng(260431).choice(derived, 128, replace=False))
        if not np.array_equal(rows, (10 * draw[:, None] + np.arange(10)).ravel()):
            raise ValueError("Training objects differ from the original 128-object draw")
    cache = root / "ranges" / REV / subject
    end = OFFSET + SHAPE[0] * ROW_BYTES
    channels, times = axes(
        (cache / f"0-{OFFSET - 1}.bin").read_bytes(), (cache / f"{end}-{SIZE - 1}.bin").read_bytes()
    )
    if not np.allclose(times[50:], np.arange(250) / 250):
        raise ValueError("Unexpected scalp latency axis")
    post = [channels.tolist().index(name) for name in CHANNELS]
    posterior = eeg_design(mean, np.arange(len(rows)), np.arange(17)).astype(np.float64)
    full = np.empty((2, len(rows), 63 * 14), dtype=np.float64)
    for n, (group, rec) in enumerate(zip(rows.reshape(-1, 10), meta["row_ranges"], strict=True)):
        first, last = int(group[0]), int(group[-1])
        begin, end = OFFSET + first * ROW_BYTES, OFFSET + (last + 1) * ROW_BYTES - 1
        blob = (cache / f"{begin}-{end}.bin").read_bytes()
        if (
            rec["rows"] != [first, last]
            or rec["bytes"] != [begin, end]
            or rec["sha256"] != hashlib.sha256(blob).hexdigest()
            or not np.array_equal(group, first + np.arange(10))
        ):
            raise ValueError(f"Corrupt cached EEG range at {subject}/{first}")
        # The public preprocessing stores the four chronological trials in order [2,1,3,0].
        x = np.frombuffer(blob, dtype="<f8").reshape(10, 4, 63, 250).astype(np.float32)
        ix = slice(n * 10, n * 10 + 10)
        for j, slots in enumerate(([1, 3], [0, 2])):
            two = (x[:, slots[0]] + x[:, slots[1]]) * 0.5
            if not np.array_equal(two[:, post], half[j, ix]):
                raise ValueError(
                    f"Full-scalp EEG fails independent posterior reproduction: {subject}/{first}"
                )
            full[j, ix] = two[:, :, 10:150].reshape(10, 63, 14, 10).mean(-1).reshape(10, -1)
    return SimpleNamespace(
        rows=rows,
        p=posterior,
        f=full,
        s=sessions,
        hashes=dict(packet=sha(path), raw_range_receipt=sha(receipt)),
    )


def fisher(train, query, rank):
    mu = train.mean(0)
    sd = train.std(0).clip(1e-4)
    x = ((train - mu) / sd).reshape(-1, 10, train.shape[-1])
    c = x.mean(1)
    d = (x - c[:, None]).reshape(train.shape)
    basis = object_directions(c.T @ c / len(c), d.T @ d / len(d), rank)
    return ((query - mu) / sd) @ basis


def stable(train, query, ids):
    mu, sd = train.mean((0, 1)), train.std((0, 1)).clip(1e-4)
    x = (train - mu) / sd
    offsets = np.stack([x[j, ids[j] == s].mean(0) for j in range(2) for s in np.unique(ids[j])])
    _, scale, vt = np.linalg.svd(offsets, full_matrices=False)
    vt = vt[scale > max(scale[0] * 1e-7, 1e-12)]

    def remove(v):
        return v - (v @ vt.T) @ vt

    # Only different pictures in the same training class enter the early/late signal covariance.
    a, b = remove(x).reshape(2, -1, 10, x.shape[-1])
    c = a.sum(1).T @ b.sum(1) - a.reshape(-1, x.shape[-1]).T @ b.reshape(-1, x.shape[-1])
    by_class = np.stack([a, b])
    d = (by_class - by_class.mean(2, keepdims=True)).reshape(-1, x.shape[-1])
    basis = object_directions((c + c.T) / (2 * len(a) * 10 * 9), d.T @ d / len(d), 32)
    return remove((query - mu) / sd) @ basis


def directions(train, ti, target, qi):
    return (
        fisher(train.p[ti], target.p[qi], 96),
        fisher(train.f[:, ti].mean(0), target.f[:, qi].mean(0), 96),
        stable(train.f[:, ti], target.f[:, qi], train.s[:, ti]),
    )


def scores(streams, query, support):
    p, f, (a, b) = streams
    cross = 0.5 * (similarities(a[query], b[support]) + similarities(b[query], a[support]))
    return np.stack([similarities(p[query], p[support]), similarities(f[query], f[support]), cross])


def pick(train):
    rng = np.random.default_rng(SEED)
    splits = np.array_split(rng.permutation(128), 4)
    grid = [(float(a), float(b)) for a in GRID for b in GRID]
    weights = np.array([[1, a, b] for a, b in grid])
    fold_counts = []
    for outer, classes in enumerate(splits):
        training = np.setdiff1d(np.arange(128), classes)
        ti = (10 * training[:, None] + np.arange(10)).ravel()
        qi = (10 * classes[:, None] + np.arange(10)).ravel()
        streams = directions(train, ti, train, qi)
        counts = np.zeros(len(grid), dtype=int)
        for _ in range(8):
            sequence = rng.permutation(len(classes))
            for offset in range(0, len(sequence), 4):
                active = np.roll(sequence, offset)[:12]
                layout = 10 * active[:, None] + np.stack([rng.permutation(10) for _ in active])
                for fold in range(5):
                    mask = np.arange(10) // 2 == fold
                    s = scores(streams, layout[:, mask].ravel(), layout[:, ~mask])
                    pred = (weights @ s.reshape(3, -1)).reshape(len(grid), 24, 12).argmax(2)
                    counts += (pred == np.repeat(np.arange(12), 2)).sum(1)
        fold_counts.append(counts)
        print("generic held-class fold", outer + 1, "of 4", flush=True)
    hits = np.sum(fold_counts, axis=0)
    best = min(range(len(grid)), key=lambda i: (-int(hits[i]), sum(grid[i]), grid[i]))
    return dict(
        participant="sub-04",
        generic_source_sha256=train.hashes,
        seed=SEED,
        training_classes=128,
        distinct_generic_photos=1280,
        training_classes_per_fit=96,
        held_out_classes_per_fold=32,
        ways=12,
        support_photos=8,
        query_photos_per_class=2,
        pseudoquery_decisions=4 * 8 * 8 * 5 * 24,
        full_head_window="40 to 600 ms end-exclusive in fourteen 40 ms bins",
        grid=[
            dict(full=a, cross=b, hits=int(hits[i]), fold_hits=[int(z[i]) for z in fold_counts])
            for i, (a, b) in enumerate(grid)
        ],
        selected=dict(posterior=1, full=grid[best][0], cross=grid[best][1]),
        rule="Maximize pooled generic episodic top-1. Ties favor smaller total added weight, then lexicographic weights.",
        heldout_generic_concepts=[
            (train.rows.reshape(-1, 10)[p, 0] // 10).tolist() for p in splits
        ],
    )


def evaluate(args, sub, selected):
    train = features(args.inputs, sub, True)
    test = features(args.inputs, sub, False)
    if np.intersect1d(train.rows // 10, test.rows // 10).size or test.rows.size != 240:
        raise ValueError("Generic and registered concepts must be disjoint")
    streams = directions(train, np.arange(len(train.rows)), test, np.arange(len(test.rows)))
    pos = {r: i for i, r in enumerate(test.rows)}
    session = read(args.inputs / "sessions.json")["subjects"][sub]
    for panel in PANELS:
        original = read(args.inputs / "personal" / panel / sub / "evaluation.json")
        path = args.references / (panel + "_ablation.json")
        ref = read(path)
        for k in ("rows", "folds", "words", "truth"):
            if original[k] != ref[k]:
                raise ValueError("Target registration differs from original personal readout")
        rows, folds, truth = np.array(ref["rows"]), np.array(ref["folds"]), np.array(ref["truth"])
        cal = np.asarray(ref["calibration_rows_by_fold_and_word"])
        if cal.shape != (5, 12, 8) or len(np.unique(rows)) != 120:
            raise ValueError("Expected the original 12-object calibration")
        matrix = np.empty((3, 120, 12))
        for f, groups in enumerate(cal):
            ix = np.flatnonzero(folds == f)
            if len(ix) != 24 or set(groups.ravel()) != set(rows) - set(rows[ix]):
                raise ValueError("A query entered calibration")
            matrix[:, ix] = scores(
                streams,
                np.array([pos[r] for r in rows[ix]]),
                np.array([[pos[r] for r in g] for g in groups]),
            )
        posterior, full, cross = matrix
        all_scores = dict(
            personal_r4=posterior,
            full_mean4=full,
            full_cross=cross,
            full_head=posterior + selected["full"] * full + selected["cross"] * cross,
        )
        results = {}
        for name, value in all_scores.items():
            pred = value.argmax(1)
            results[name] = dict(n=120, hits=int((pred == truth).sum()), predicted=pred.tolist())
        if results["personal_r4"]["predicted"] != original["results"]["personal_r4"]["predicted"]:
            raise ValueError("Old posterior method fails exact per-row reproduction")
        result = {k: ref[k] for k in ("rows", "words", "folds", "truth")}
        result.update(
            subject=sub,
            results=results,
            weights=selected,
            method="full_head",
            train_sha256=train.hashes,
            target_sha256=test.hashes,
            registration_sha256=sha(path),
            selection_sha256=sha(args.out / "selection.json"),
        )
        audit = audit_panel(session, result)["methods"]
        result["session_control"] = {
            name: {
                k: audit[name][k]
                for k in ("conditional_expected_hits", "accuracy_minus_conditional_expectation")
            }
            for name in results
        }
        save(args.out / panel / sub / "evaluation.json", result)
        print(
            panel,
            sub,
            [
                (
                    k,
                    v["hits"],
                    round(
                        result["session_control"][k]["accuracy_minus_conditional_expectation"], 4
                    ),
                )
                for k, v in results.items()
            ],
            flush=True,
        )


def summary(out):
    lock = sha(out / "selection.json")
    report = {}
    for sub in ("sub-04", "sub-05"):
        report[sub] = {}
        for panel in PANELS:
            path = out / panel / sub / "evaluation.json"
            data = read(path)
            if data["selection_sha256"] != lock:
                raise ValueError("Cannot pool evaluations from different weights")
            truth = np.asarray(data["truth"])
            a, b = [
                np.asarray(data["results"][name]["predicted"]) == truth
                for name in ("full_head", "personal_r4")
            ]
            report[sub][panel] = dict(
                paired=paired(a, b, truth),
                session_control=data["session_control"],
                evaluation_sha256=sha(path),
            )
    save(
        out / "summary.json",
        dict(
            selection_sha256=lock,
            people=report,
            uncertainty="Within each participant and panel, 10,000 percentile bootstrap draws resample the 12 classes, keeping the ten photos of each class together. Rates are fractions.",
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("select", "evaluate", "summarize"))
    for k in ("inputs", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--references", type=Path)
    p.add_argument(
        "--subjects", nargs="+", choices=("sub-04", "sub-05"), default=["sub-04", "sub-05"]
    )
    a = p.parse_args()
    if a.stage == "summarize":
        summary(a.out)
    elif a.stage == "select":
        pick_result = pick(features(a.inputs, "sub-04", True))
        save(a.out / "selection.json", pick_result)
        print("Locked sub-04 generic weights:", pick_result["selected"], flush=True)
    else:
        if not a.references:
            p.error("Evaluation requires --references with both panel ablations")
        result = read(a.out / "selection.json")
        if result["participant"] != "sub-04" or result["generic_source_sha256"]["packet"] != sha(
            a.inputs / "generic/sub-04.npz"
        ):
            raise ValueError("Frozen weights do not match the development calibration packet")
        for person in a.subjects:
            evaluate(a, person, result["selected"])


if __name__ == "__main__":
    main()
