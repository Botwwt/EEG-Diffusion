"""Use a participant's early evoked potentials to recognize familiar objects across sessions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from decode_eeg_layout import CHANNELS


def discriminate(support, query):
    """Equal-prior linear discriminant with OAS shrinkage of the pooled within-class covariance."""
    n_classes, n_per_class, p = support.shape
    centers = support.mean(axis=1)
    resid = (support - centers[:, None]).reshape(-1, p)
    df = n_classes * (n_per_class - 1)
    scale = np.sqrt((resid**2).sum(0) / df).clip(1e-8)
    resid = resid / scale
    cov = resid.T @ resid / df
    mu = np.trace(cov) / p
    alpha = np.square(cov).mean()
    denom = (df + 1) * (alpha - mu**2 / p)
    shrink = min(1.0, (alpha + mu**2) / denom) if denom > 0 else 1.0
    precision_input = (1 - shrink) * cov
    precision_input.flat[:: p + 1] += max(shrink * mu, 1e-8)
    grand = centers.mean(0)
    center = (centers - grand) / scale
    weights = np.linalg.solve(precision_input, center.T)
    logits = ((query - grand) / scale) @ weights - (center * weights.T).sum(1) / 2
    return (logits - logits.mean(1, keepdims=True)) / logits.std(1, keepdims=True).clip(1e-8)


def read_packet(path):
    with np.load(path, allow_pickle=False) as packet:
        if packet["channels"].tolist() != CHANNELS or int(packet["rows"].size) != 240:
            raise ValueError("Expected 240 registered rows and the 17 named posterior electrodes")
        half = packet["half_eeg"]
        rows = packet["rows"]
        if len(set(rows)) != len(rows) or half.shape != (2, 240, 17, 250):
            raise ValueError("Expected two independent two-trial recording halves")
        if not np.isfinite(half).all():
            raise ValueError("Nonfinite EEG")
        # 250 Hz, [60, 360) ms, 15 bins of 20 ms. One feature per electrode and bin.
        feature = half[..., 15:90].astype(np.float64).reshape(2, 240, 17, 15, 5).mean(-1)
    return {int(row): i for i, row in enumerate(rows)}, feature.reshape(2, 240, -1)


def score(reg, row_index, x):
    rows, truth, folds = (np.asarray(reg[k]) for k in ("rows", "truth", "folds"))
    support = np.asarray(reg["calibration_rows_by_fold_and_word"])
    if support.shape != (5, 12, 8) or len(set(rows)) != 120:
        raise ValueError("Expected the frozen 5-fold 12-class registration")
    methods = ("early_to_late", "late_to_early", "cross", "same", "mean4")
    preds = {k: np.full(len(rows), -1, dtype=int) for k in methods}
    lookup = dict(zip(rows, truth, strict=True))
    for fold, cal_rows in enumerate(support):
        take = np.flatnonzero(folds == fold)
        if len(take) != 24 or set(cal_rows.ravel()) != set(rows) - set(rows[take]):
            raise ValueError("Support photographs overlap a query, or a class is incomplete")
        if any(lookup[r] != k for k, rr in enumerate(cal_rows) for r in rr):
            raise ValueError("Calibration object name mismatch")
        cal = np.array([[row_index[int(r)] for r in g] for g in cal_rows])
        q = [row_index[int(r)] for r in rows[take]]
        a, b = x
        # Direction names give the query half followed by the calibration half.
        ab, ba = discriminate(b[cal], a[q]), discriminate(a[cal], b[q])
        same = discriminate(a[cal], a[q]) + discriminate(b[cal], b[q])
        mean4 = discriminate((a[cal] + b[cal]) / 2, (a[q] + b[q]) / 2)
        for key, scores in zip(methods, (ab, ba, ab + ba, same, mean4), strict=True):
            preds[key][take] = scores.argmax(1)
    if any((p < 0).any() for p in preds.values()):
        raise ValueError("Missing held-out prediction")
    return {k: dict(predicted=p.tolist(), hits=int((p == truth).sum())) for k, p in preds.items()}


def main(args):
    for path in sorted(args.halves.glob("sub-*.npz")):
        index, x = read_packet(path)
        for panel in ("registered", "replicate"):
            ref = args.references / (panel + "_ablation.json")
            reg = json.loads(ref.read_text())
            result = score(reg, index, x)
            dest = args.out / panel / path.stem
            dest.mkdir(exist_ok=True, parents=True)
            data = {k: reg[k] for k in ("words", "rows", "folds", "truth")}
            data.update(
                subject=path.stem,
                input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                registration_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
                results=result,
            )
            (dest / "evaluation.json").write_text(json.dumps(data, indent=2) + "\n")
            print(path.stem, panel, {k: v["hits"] for k, v in result.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("halves", "references", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
