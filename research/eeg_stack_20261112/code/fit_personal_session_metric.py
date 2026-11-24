"""Learn object directions from a participant's other concepts and test them on held-out photos."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from decode_eeg_layout import CHANNELS, eeg_design
from scipy.linalg import eigh


def load(path):
    with np.load(path, allow_pickle=False) as data:
        if (
            str(data["subject"].item()) != path.stem
            or data["channels"].tolist() != CHANNELS
            or not np.allclose(data["times"], np.arange(250) / 250, rtol=0, atol=1e-7)
        ):
            raise ValueError("Participant, posterior electrodes or sample times changed")
        rows, half, ids = (data[k] for k in ("rows", "half_eeg", "session_ids"))
        if half.shape != (2, len(rows), 17, 250) or ids.shape != (2, len(rows)):
            raise ValueError("Expected two two-repeat session means per image")
        if len(set(rows)) != len(rows) or not np.isfinite(half).all():
            raise ValueError("Nonfinite waveforms or repeated rows")
        mean = data["mean_eeg"] if "mean_eeg" in data else half.mean(0)
        if mean.shape != half.shape[1:] or not np.allclose(half.mean(0), mean, atol=2e-6):
            raise ValueError("Two session means do not reconstruct the four-repeat waveform")
    return rows, half, ids, mean


def object_directions(between, within, rank):
    within.flat[:: len(within) + 1] += 0.25
    values, vectors = eigh(between, within, subset_by_index=(len(within) - rank, len(within) - 1))
    pos = np.maximum(values - 0.1, 0)
    return vectors * np.sqrt(pos / np.maximum(values + 1, 1))


def personal_r4(train, test):
    def design(v):
        return eeg_design(v, np.arange(len(v)), np.arange(17)).astype(np.float64)

    x, y = design(train), design(test)
    mean, sd = x.mean(0), x.std(0).clip(1e-4)
    grouped = ((x - mean) / sd).reshape(128, 10, -1)
    centers = grouped.mean(1)
    residual = (grouped - centers[:, None]).reshape(x.shape)
    basis = object_directions(
        centers.T @ centers / len(centers), residual.T @ residual / len(x), 96
    )
    return ((y - mean) / sd) @ basis


def cross_photo_session(train, test, ids, controls=False):
    # Use signed amplitudes from 60 to 460 ms, in 20 ms bins across 17 posterior electrodes.
    def design(v):
        return v[..., 15:115].astype(np.float64).reshape(2, v.shape[1], -1, 5).mean(-1)

    x, y = design(train), design(test)
    mean = x.mean((0, 1))
    scale = x.std((0, 1)).clip(1e-4)
    x = (x - mean) / scale
    # Remove the generic training sessions' offsets without supplying session IDs at prediction.
    offsets = np.stack([x[j, ids[j] == s].mean(0) for j in range(2) for s in np.unique(ids[j])])
    _, singular, vt = np.linalg.svd(offsets, full_matrices=False)
    nuisance = vt[singular > singular[0] * 1e-7]

    def remove(v):
        return v - (v @ nuisance.T) @ nuisance

    x = remove(x).reshape(2, 128, 10, -1)
    means = x.mean(2)
    # Sum the early/late cross-products of distinct photos in each concept.
    # Subtracting the same-photo diagonal leaves exactly 10*9 ordered pairs per object.
    a, b = x
    centered = (x - means[:, :, None]).reshape(-1, x.shape[-1])
    same_photo = a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])
    cross = a.sum(1).T @ b.sum(1) - same_photo
    contrasts = {"cross_photo_session": (cross + cross.T) / (2 * 128 * 10 * 9)}
    if controls:
        # These post-hoc diagnostics keep recordings, feature design and evaluation fixed.
        centers = means.mean(0)
        contrasts["class_means"] = centers.T @ centers / len(centers)
        contrasts["same_photo_across_sessions"] = (same_photo + same_photo.T) / (2 * 128 * 10)
    within = centered.T @ centered / len(centered)
    query = remove((y - mean) / scale)
    return {k: query @ object_directions(v, within.copy(), 32) for k, v in contrasts.items()}


def similarities(q, groups):
    centers = groups.mean(1)
    q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-8)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True).clip(1e-8)
    scores = q @ centers.T
    return (scores - scores.mean(1, keepdims=True)) / scores.std(1, keepdims=True).clip(1e-8)


def evaluate(ref, r4, halves, target_rows):
    rows, truth, folds = (np.asarray(ref[k]) for k in ("rows", "truth", "folds"))
    lookup = {int(r): i for i, r in enumerate(target_rows)}
    words = dict(zip(rows.tolist(), truth.tolist(), strict=True))
    predictions = {k: np.full(len(rows), -1) for k in ("personal_r4", *halves)}
    for fold, groups in enumerate(ref["calibration_rows_by_fold_and_word"]):
        test = np.where(folds == fold)[0]
        if set(np.ravel(groups)) != set(rows) - set(rows[test]) or len(test) != 24:
            raise ValueError("Invalid calibration fold")
        if any(words[r] != k for k, group in enumerate(groups) for r in group):
            raise ValueError("Calibration class mismatch")
        cal = np.array([[lookup[r] for r in g] for g in groups])
        q = [lookup[r] for r in rows[test]]
        predictions["personal_r4"][test] = similarities(r4[q], r4[cal]).argmax(1)
        for k, (a, b) in halves.items():
            predictions[k][test] = (similarities(a[q], b[cal]) + similarities(b[q], a[cal])).argmax(
                1
            )
    if any((v < 0).any() for v in predictions.values()):
        raise ValueError("Incomplete registration")
    return {
        k: dict(n=len(v), predicted=v.tolist(), hits=int((v == truth).sum()))
        for k, v in predictions.items()
    }


def run(args):
    references = {
        p: json.loads((args.references / (p + "_ablation.json")).read_text())
        for p in ("registered", "replicate")
    }
    all_query = np.concatenate([v["rows"] for v in references.values()])
    other = np.setdiff1d(np.arange(1654), np.unique(all_query // 10))
    expected = np.sort(np.random.default_rng(260431).choice(other, 128, replace=False))
    for sub in args.subjects:
        bg, target = (d / (sub + ".npz") for d in (args.generic, args.halves))
        tr, train, ids, mean = load(bg)
        rows, x, _, native_mean = load(target)
        if not np.array_equal(tr, (10 * expected[:, None] + np.arange(10)).ravel()):
            raise ValueError("Generic classes differ from the prespecified independent draw")
        if set(rows) != set(all_query) or np.intersect1d(rows // 10, expected).size:
            raise ValueError("Target or calibration object entered the generic model")
        if set(ids[0]) != {1, 2} or set(ids[1]) != {3, 4}:
            raise ValueError("Unexpected generic recording schedule")
        r4 = personal_r4(mean, native_mean)
        halves = cross_photo_session(train, x, ids, args.controls)
        for panel, ref in references.items():
            result = evaluate(ref, r4, halves, rows)
            out = args.out / panel / sub
            out.mkdir(parents=True, exist_ok=True)
            data = {k: ref[k] for k in ("words", "rows", "truth", "folds")}
            data.update(
                subject=sub,
                results=result,
                generic_classes=expected.tolist(),
                generic_sha256=hashlib.sha256(bg.read_bytes()).hexdigest(),
                target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            )
            (out / "evaluation.json").write_text(json.dumps(data, indent=2) + "\n")
            print(sub, panel, {k: v["hits"] for k, v in result.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("generic", "halves", "references", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument(
        "--controls", action="store_true", help="Add fixed-data covariance controls"
    )
    run(parser.parse_args())
