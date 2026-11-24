"""Screen EEG-derived spatial detail using only the prepared historical sub-03 T archive.

Fits on concept folds 1..4 and estimates gains using leave-one-training-fold-out
predictions. Fold 0 is scored once in the raw 16x16 gallery. No generator.
"""

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from probe_context_residual import (
    CHUNK,
    EEG_RIDGE,
    FOLD_SEED,
    eeg_features,
    evaluate,
    groups_by_fold,
    within_concept_derangement,
)

ORIENTATIONS = ("left_right", "top_bottom", "diagonal")
KEYS = tuple(f"{s}_{d}" for s in (8, 16) for d in ORIENTATIONS)
BOOT_SEED = 20261127
GATE_SHRINK = 0.05


def split(x):
    """Non-normalized Haar transform, so the lowpass is exactly the 2x2 area mean."""
    a, b = x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2]
    c, d = x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]
    return (a + b + c + d) * 0.25, (
        (a - b + c - d) * 0.25,
        (a + b - c - d) * 0.25,
        (a - b - c + d) * 0.25,
    )


def join(lo, detail):
    h, v, d = detail
    n, c, s, _ = lo.shape
    out = np.empty((n, c, 2 * s, 2 * s), dtype=lo.dtype)
    out[:, :, 0::2, 0::2] = lo + h + v + d
    out[:, :, 0::2, 1::2] = lo - h + v - d
    out[:, :, 1::2, 0::2] = lo + h - v - d
    out[:, :, 1::2, 1::2] = lo - h - v + d
    return out


def bands(x):
    lo8, detail16 = split(x)
    lo4, detail8 = split(lo8)
    return lo4, detail8 + detail16


def read_targets(bank, rows):
    if bank.shape[1:] != (4, 64, 64):
        raise ValueError("Expected archived four-channel 64x64 VAE modes")
    out = np.empty((len(rows), 4, 16, 16), dtype=np.float32)
    for start in range(0, len(rows), CHUNK):
        ix = rows[start : start + CHUNK]
        tile = bank[ix].astype(np.float32, copy=False)
        out[start : start + len(ix)] = tile.reshape(-1, 4, 16, 4, 16, 4).mean(axis=(3, 5))
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite VAE mode")
    return out


def fit_views(x, y, train, test):
    """Return centered full prediction, temporal-agreement scores, and training intercept."""
    target = y[train].reshape(len(train), -1)
    intercept = target.mean(axis=0, dtype=np.float64).astype(np.float32)
    target -= intercept
    xt = x[train]
    mean = xt.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = xt.std(axis=0, dtype=np.float64).clip(1e-6).astype(np.float32)
    xt = (xt - mean) / std
    xe = (x[test] - mean) / std
    columns = np.arange(17 * 25).reshape(17, 25)
    # Ten-sample bins at 250 Hz start at 0 ms; end-exclusive windows.
    views = (slice(None), columns[:, 2:4].ravel(), columns[:, 4:8].ravel())
    predictions = []
    for cols in views:
        a = np.ascontiguousarray(xt[:, cols])
        gram = (a.T @ a).astype(np.float64)
        gram.flat[:: len(gram) + 1] += EEG_RIDGE * len(train)
        weights = np.linalg.solve(gram, (a.T @ target).astype(np.float64)).astype(np.float32)
        predictions.append((xe[:, cols] @ weights).reshape(-1, 4, 16, 16))
    _, early = bands(predictions[1])
    _, late = bands(predictions[2])
    agreement = []
    for left, right in zip(early, late, strict=True):
        left = left.reshape(len(test), -1)
        right = right.reshape(len(test), -1)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        agreement.append(np.einsum("ij,ij->i", left, right) / denominator.clip(1e-12))
    return predictions[0], np.column_stack(agreement), intercept.reshape(1, 4, 16, 16)


def within(a):
    block = a.reshape(-1, 10, a.shape[-1])
    return (block - block.mean(axis=1, keepdims=True)).reshape(a.shape)


def dot(a, b):
    return float(np.einsum("ij,ij->", a, b, dtype=np.float64, optimize=False)) / a.shape[1]


def box_ridge(a, b, anchor):
    shrink = GATE_SHRINK * np.trace(a) / 2 + 1e-12
    a = a + shrink * np.eye(2)
    b = b + shrink * anchor
    candidates = [np.linalg.solve(a, b)]
    for i in range(2):
        j = 1 - i
        for edge in (0.0, 1.0):
            v = np.zeros(2)
            v[i] = edge
            v[j] = np.clip((b[j] - a[j, i] * edge) / a[j, j], 0, 1)
            candidates.append(v)
    feasible = [v for v in candidates if np.all((v >= 0) & (v <= 1))]
    return min(feasible, key=lambda v: v @ a @ v - 2 * (b @ v))


def calibrate(oof, observed, agreement):
    """Use within-concept squared error for all three gain rules; a common 16x16 metric."""
    _, prediction = bands(oof)
    _, truth = bands(observed)
    thresholds = np.median(agreement, axis=0)
    numer, denom, static, gated = [], [], [], []
    for k, (pred, target) in enumerate(zip(prediction, truth, strict=True)):
        p = pred.reshape(len(oof), -1)
        t = within(target.reshape(len(oof), -1))
        pw = within(p)
        numer.append(dot(pw, t))
        denom.append(dot(pw, pw))
        slope = np.clip(numer[-1] / max(denom[-1], 1e-12), 0, 1)
        static.append(slope)
        # Center after the per-photo gate. This equals least squares on all
        # within-concept photo differences, including pairs in different bins.
        high = (agreement[:, k] > thresholds[k])[:, None]
        low_p, high_p = within(p * ~high), within(p * high)
        cross = dot(low_p, high_p)
        gram = np.array([[dot(low_p, low_p), cross], [cross, dot(high_p, high_p)]])
        gated.append(box_ridge(gram, np.array([dot(low_p, t), dot(high_p, t)]), slope))
    # dot() normalizes each band by its own dimension; each band's mean
    # squared energy contributes once to mean squared error on the finest grid.
    uniform = np.clip(sum(numer) / max(sum(denom), 1e-12), 0, 1)
    return float(uniform), np.array(static), np.array(gated), thresholds


def render(low, details, gains, mean):
    pieces = [d * gains[:, k, None, None, None] for k, d in enumerate(details)]
    return mean + join(join(low, pieces[:3]), pieces[3:])


def summarize_differences(a, b, resamples):
    top = np.array(a["top1_by_concept"]["paired"]) - b["top1_by_concept"]["paired"]
    mse = np.array(b["mse_by_concept"]) - a["mse_by_concept"]
    return {
        "mean_top1_gain": float(top.mean()),
        "top1_paired_bootstrap_95": np.quantile(
            top[resamples].mean(axis=1), [0.025, 0.975]
        ).tolist(),
        "mean_mse_reduction": float(mse.mean()),
        "top1_gain_by_concept": top.tolist(),
        "mse_reduction_by_concept": mse.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    labels = np.load(args.prep / "train_concept.npy")
    valid = np.load(args.prep / "lat512_train_ok.npy").astype(bool)
    train_groups, eval_groups = groups_by_fold(labels, valid, 1654)
    rows = np.concatenate((train_groups.ravel(), eval_groups.ravel()))
    n = train_groups.size
    fold = np.empty(1654, dtype=int)
    fold[np.random.default_rng(FOLD_SEED).permutation(1654)] = np.arange(1654) % 5
    train_fold = fold[labels[rows[:n]]]
    bank = np.load(args.prep / "lat512_train.npy", mmap_mode="r")
    eeg = np.load(args.prep / "sub-03_train.npy", mmap_mode="r")
    if len(bank) != len(labels) or len(eeg) != len(labels):
        raise ValueError("Prepared latent, EEG and concept rows disagree")
    y = read_targets(bank, rows)
    x = eeg_features(eeg, rows, "mean4")
    oof = np.empty_like(y[:n])
    observed = np.empty_like(oof)
    agreements = np.empty((n, len(KEYS)), dtype=np.float32)
    for k in (1, 2, 3, 4):
        fit, held = np.flatnonzero(train_fold != k), np.flatnonzero(train_fold == k)
        if not len(fit) or not len(held):
            raise ValueError("Every inner fold must contain complete concepts")
        oof[held], agreements[held], mean = fit_views(x, y, fit, held)
        observed[held] = y[held] - mean
    uniform, static, gated, threshold = calibrate(oof, observed, agreements)
    full, eval_agree, mean = fit_views(x, y, np.arange(n), np.arange(n, len(rows)))
    low, detail = bands(full)
    hi = (eval_agree > threshold).astype(np.int64)
    batch_shape = hi.shape
    gain = gated[np.arange(len(KEYS)), hi]
    rng = np.random.default_rng(BOOT_SEED + 1)
    # Placebo independent of any evaluation EEG peers or concept IDs.
    random_bin = (agreements[rng.integers(0, n, size=len(full))] > threshold).astype(np.int64)
    arms = {
        "projected_4": np.zeros(batch_shape),
        "direct_16": np.ones(batch_shape),
        "uniform": np.full(batch_shape, uniform),
        "static_bands": np.broadcast_to(static, batch_shape),
        "temporal_gate": gain,
        "train_drawn_gate": gated[np.arange(len(KEYS)), random_bin],
    }
    scores = {}
    shuffle = within_concept_derangement(len(eval_groups), BOOT_SEED + 2)
    for name, coefficients in arms.items():
        pred = render(low, detail, coefficients.astype(np.float32), mean)
        if name == "direct_16":
            err = float(np.max(np.abs(pred - (full + mean))))
            if not np.allclose(pred, full + mean, atol=2e-6, rtol=1e-5):
                raise ValueError("g=1 failed direct-ridge/Haar reconstruction identity")
        scores[name] = evaluate(pred, y[n:], mean, shuffle)
        mse = (
            np.square(pred - y[n:])
            .reshape(len(eval_groups), 10, -1)
            .mean(axis=(1, 2), dtype=np.float64)
        )
        scores[name].update(mse=float(mse.mean()), mse_by_concept=mse.tolist())
        print(name, scores[name]["paired"], scores[name]["mse"], flush=True)
    resamples = np.random.default_rng(BOOT_SEED).integers(
        0, len(eval_groups), (5000, len(eval_groups))
    )
    comparisons = {
        name: summarize_differences(scores["temporal_gate"], value, resamples)
        for name, value in scores.items()
        if name != "temporal_gate"
    }
    primary = [comparisons[k] for k in ("direct_16", "static_bands")]
    advance = all(c["mean_top1_gain"] > 0 and c["mean_mse_reduction"] > 0 for c in primary)
    advance = bool(advance and comparisons["direct_16"]["top1_paired_bootstrap_95"][0] > 0)
    result = {
        "subject": "sub-03",
        "scope": "Historical T only; EEG four-trial voltage means; CPU VAE feature retrieval.",
        "train_folds": [1, 2, 3, 4],
        "fold_seed": FOLD_SEED,
        "eval_fold": 0,
        "train_photos": n,
        "eval_photos": eval_groups.size,
        "eval_concept_ids": labels[eval_groups[:, 0]].tolist(),
        "training": "OOF by concept; each inner fit standardizes its own EEG, centers targets on its own training mean.",
        "gain_objective": "Within-concept high-frequency MSE (all pairs), centered after gating. Static and temporal use the same loss.",
        "gate_inputs": "Per photo, directional cosine between 80-160 ms and 160-320 ms centered predictions.",
        "agreement_threshold": "Training OOF median for each scale and orientation, high iff above threshold.",
        "ridge_per_photo": EEG_RIDGE,
        "gate_anchor_strength_relative_to_mean_bin_energy": GATE_SHRINK,
        "orientation_order": list(KEYS),
        "time_window_unit": "ms",
        "uniform_gain": uniform,
        "static_gains": static.tolist(),
        "low_high_gains": gated.tolist(),
        "thresholds": threshold.tolist(),
        "eval_fraction_high": hi.mean(axis=0).tolist(),
        "haar_identity_max_abs_error": err,
        "projection": "All arms add identical training mean in raw 16x16 space; 4x4 projects only centered detail to zero.",
        "metrics": scores,
        "temporal_vs": comparisons,
        "bootstrap_resamples": 5000,
        "bootstrap_seed": BOOT_SEED,
        "advance_to_generation": advance,
        "decision_rule": "Positive paired Top1 and lower MSE vs both direct_16 and static_bands; two-sided concept-bootstrap Top1 lower 95% > 0 vs direct_16.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf8", delete=False, dir=args.out.parent) as f:
        json.dump(result, f, allow_nan=False, indent=1)
        f.write("\n")
    os.replace(f.name, args.out)
    print(f"advance_to_generation={advance}", flush=True)


if __name__ == "__main__":
    main()
