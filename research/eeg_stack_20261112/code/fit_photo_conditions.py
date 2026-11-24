"""Fit sub-03 EEG-to-Brain-HIVE semantic conditions using historical T concepts."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from predict_instance_starts import read_registration, read_signals

FOLDS = 3
SEED = 20261129
RIDGE = 0.10
DONORS = 16
TEMPERATURE = 0.10
DONOR_MIX = 0.35
METHODS = ("linear", "cone", "manifold")


def normalize(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)


def multiscale_features(x):
    # read_signals returns the 17 posterior ERPs in successive 40 ms bins.
    erp = x.reshape(-1, 17, 25)
    coarse = erp[:, :, :24].reshape(-1, 17, 12, 2).mean(axis=3)
    early = erp[:, :, 2:10]  # 80–400 ms, with a distinct early-window kernel weight
    return np.column_stack((x, coarse.reshape(len(x), -1), early.reshape(len(x), -1)))


def axis_and_radius(target):
    direction = normalize(target).mean(axis=0)
    alpha = float(np.linalg.norm(direction))
    if not 0.0 < alpha < 1.0:
        raise ValueError("Expected a nondegenerate training visual cone")
    return direction / alpha, alpha, float(np.linalg.norm(target, axis=1).mean())


def restore_cone(value, mean, axis, alpha, radius):
    centered = value - mean
    tangent = normalize(centered - (centered @ axis)[:, None] * axis)
    return radius * normalize(alpha * axis + np.sqrt(1.0 - alpha**2) * tangent)


def predict(x_train, y_train, x_query):
    mean_x = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd_x = x_train.std(axis=0, dtype=np.float64).clip(1e-6).astype(np.float32)
    train = (x_train - mean_x) / sd_x
    mean_y = y_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = y_train - mean_y
    gram = (train.T @ train).astype(np.float64)
    gram.flat[:: len(gram) + 1] += len(train) * RIDGE
    w = np.linalg.solve(gram, (train.T @ centered).astype(np.float64)).astype(np.float32)
    linear = ((x_query - mean_x) / sd_x) @ w + mean_y
    axis, alpha, radius = axis_and_radius(y_train)
    cone = restore_cone(linear, mean_y, axis, alpha, radius)
    bank = normalize(centered)
    manifold = np.empty_like(linear)
    for start in range(0, len(linear), 128):
        q = linear[start : start + 128]
        # T donors are actual fused visual embeddings; the query is EEG-only.
        similarity = normalize(q - mean_y) @ bank.T
        inds = np.argpartition(-similarity, DONORS - 1, axis=1)[:, :DONORS]
        values = np.take_along_axis(similarity, inds, axis=1)
        weights = np.exp((values - values.max(axis=1, keepdims=True)) / TEMPERATURE)
        weights /= weights.sum(axis=1, keepdims=True)
        donors = np.einsum("nk,nkf->nf", weights, bank[inds])
        combined = (1 - DONOR_MIX) * normalize(q - mean_y) + DONOR_MIX * normalize(donors)
        manifold[start : start + len(q)] = restore_cone(combined, 0.0, axis, alpha, radius)
    return {"linear": linear, "cone": cone, "manifold": manifold}, mean_y


def evaluate(pred, truth, concepts, center):
    """Centered retrieval uses only the fitting-fold mean. Cosine to paired F is uncentered."""
    order = np.argsort(concepts, kind="stable")
    labels, counts = np.unique(concepts, return_counts=True)
    if not np.all(counts == 10):
        raise ValueError("Expected complete ten-photo concepts for scoring")
    q = normalize(pred[order] - center)
    g = normalize(truth[order] - center)
    centroid = normalize(g.reshape(len(labels), 10, -1).mean(axis=1))
    ids = np.repeat(np.arange(len(labels)), 10)
    concept_scores = q @ centroid.T
    correct = concept_scores[np.arange(len(q)), ids]
    # Gallery centroids include all ten photos, identically for all methods.
    concept_hits = (correct == concept_scores.max(axis=1)) / (
        concept_scores == concept_scores.max(axis=1, keepdims=True)
    ).sum(axis=1)
    intra = q.reshape(len(labels), 10, -1) @ g.reshape(len(labels), 10, -1).transpose(0, 2, 1)
    diagonal = np.diagonal(intra, axis1=1, axis2=2)
    intra_hits = (diagonal == intra.max(axis=2)) / (intra == intra.max(axis=2, keepdims=True)).sum(
        axis=2
    )
    global_hits = []
    for i in range(0, len(q), 128):
        scores = q[i : i + 128] @ g.T
        maximum = scores.max(axis=1)
        hits = scores[np.arange(len(scores)), np.arange(i, i + len(scores))] == maximum
        global_hits.append(hits / (scores == maximum[:, None]).sum(axis=1))
    return {
        "concept_top1": float(concept_hits.mean()),
        "intra10_top1": float(intra_hits.mean()),
        "photo_gallery_top1": float(np.concatenate(global_hits).mean()),
        "teacher_cosine": float(np.einsum("ij,ij->i", normalize(pred), normalize(truth)).mean()),
        "mean_pred_norm": float(np.linalg.norm(pred, axis=1).mean()),
        "mean_target_norm": float(np.linalg.norm(truth, axis=1).mean()),
    }


def select(x, y, concepts):
    shuffled = np.random.default_rng(SEED).permutation(np.unique(concepts))
    folds = []
    for k, held in enumerate(np.array_split(shuffled, FOLDS)):
        test = np.isin(concepts, held)
        predictions, center = predict(x[~test], y[~test], x[test])
        values = {m: evaluate(predictions[m], y[test], concepts[test], center) for m in METHODS}
        folds.append(
            {"fold": k, "concepts": len(held), "photos": int(test.sum()), "scores": values}
        )
        print(f"Historical T fold {k + 1}/{FOLDS} complete", flush=True)
    weights = [f["photos"] for f in folds]
    aggregate = {
        m: {
            key: float(np.average([f["scores"][m][key] for f in folds], weights=weights))
            for key in folds[0]["scores"][m]
        }
        for m in METHODS
    }
    winner = max(
        METHODS,
        key=lambda m: (
            aggregate[m]["concept_top1"],
            aggregate[m]["intra10_top1"],
            aggregate[m]["teacher_cosine"],
        ),
    )
    return winner, folds, aggregate


def panel_rows(path, t, concepts):
    with np.load(path, allow_pickle=False) as packet:
        rows = packet["rows"]
    if (
        rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or not rows.size
        or rows.max() >= len(concepts)
        or rows.min() < 0
    ):
        raise ValueError(f"Invalid image rows in {path}")
    labels, counts = np.unique(concepts[rows], return_counts=True)
    if (
        len(np.unique(rows)) != len(rows)
        or not np.all(counts == 10)
        or np.intersect1d(labels, concepts[t]).size
    ):
        raise ValueError(f"Expected complete, unique concepts outside historical T in {path}")
    return rows


def run(args):
    import torch

    t, concepts, _, _, channels, _ = read_registration(args.archive)
    if not np.array_equal(np.load(args.train_prep / "train_concept.npy"), concepts[t]):
        raise ValueError("Prepared training rows differ from the original archive")
    rows = {
        panel: panel_rows(args.panels / f"{panel}_layout.npz", t, concepts)
        for panel in ("dev", "confirm")
    }
    if np.intersect1d(concepts[rows["dev"]], concepts[rows["confirm"]]).size:
        raise ValueError("Development and confirmation concepts overlap")
    split = json.loads(
        (args.archive / "data/recovery_20261108_v236/current_split_manifest.json").read_text()
    )
    if not np.isin(rows["confirm"], split["image_rows"]["V"]).all():
        raise ValueError("Confirmation photos must come from historical V")
    target_path = args.archive / "runs/HIVE_native_sub03_5096542/deterministic_F_all16540.pt"
    full = torch.load(target_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(full, torch.Tensor) or tuple(full.shape) != (16540, 1024):
        raise ValueError("Expected the native deterministic 16540 × 1024 image teacher tensor")
    y = full.numpy()[t].astype(np.float32)
    if not np.isfinite(y).all() or np.any(np.linalg.norm(y, axis=1) == 0):
        raise ValueError("Invalid training teacher values")
    all_rows = np.concatenate(list(rows.values()))
    train, query = read_signals(
        args.train_prep,
        args.archive / "data/recovery_20261108_v236/sub-03_train_mean4.pt",
        t,
        all_rows,
        channels,
    )
    x = multiscale_features(train)
    winner, folds, aggregate = select(x, y, concepts[t])
    print(f"Locked from historical T only: {winner}", flush=True)
    predictions, center = predict(x, y, multiscale_features(query))
    if not all(
        v.shape == (len(all_rows), 1024) and np.isfinite(v).all() for v in predictions.values()
    ):
        raise ValueError("Invalid semantic predictions")
    meta = {
        "subject": "sub-03",
        "eeg": "17 posterior channels, four physical trial mean",
        "features": "40 ms bins, adjacent 80 ms means and duplicated 80–400 ms early window",
        "teacher": str(target_path.relative_to(args.archive)),
        "train_target_sha256_f32": hashlib.sha256(y.tobytes()).hexdigest(),
        "train_rows_sha256_i64": hashlib.sha256(np.asarray(t, dtype="<i8").tobytes()).hexdigest(),
        "train_photos": len(t),
        "train_concepts": len(np.unique(concepts[t])),
        "selection_seed": SEED,
        "folds": FOLDS,
        "ridge_per_train_row": RIDGE,
        "donors": DONORS,
        "temperature": TEMPERATURE,
        "donor_mix": DONOR_MIX,
        "selection": "Photo-weighted mean held-concept Top-1, ties by Intra10 then paired teacher cosine; exact ties use method order.",
        "method_order": list(METHODS),
        "selected": winner,
        "metric": "Train-centered cosine. Concept Top-1 uses mean normalized true-photo features per concept; Intra10 uses same-concept photos; ties share credit. Teacher cosine is uncentered.",
        "inner_validation": folds,
        "inner_means": aggregate,
        "panel_scores": {},
    }
    pos = 0
    args.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out) as tmp:
        stage = Path(tmp)
        for name, indices in rows.items():
            conditions = {m: preds[pos : pos + len(indices)] for m, preds in predictions.items()}
            pos += len(indices)
            np.savez_compressed(
                stage / f"{name}_conditions.npz",
                rows=indices,
                **conditions,
                selected=conditions[winner],
                selected_name=winner,
            )
            if name in args.score_panels:
                truth = full.numpy()[indices].astype(np.float32)
                if not np.isfinite(truth).all() or np.any(np.linalg.norm(truth, axis=1) == 0):
                    raise ValueError(f"Invalid scoring targets in {name}")
                meta["panel_scores"][name] = {
                    "photos": len(indices),
                    "concepts": len(np.unique(concepts[indices])),
                    "scores": {
                        m: evaluate(v, truth, concepts[indices], center)
                        for m, v in conditions.items()
                    },
                }
        (stage / "conditions_report.json").write_text(
            json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for artifact in sorted(stage.iterdir()):
            artifact.replace(args.out / artifact.name)
    print(f"Wrote locked {winner} conditions to {args.out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--train-prep", type=Path, required=True)
    parser.add_argument(
        "--panels",
        type=Path,
        required=True,
        help="Directory of dev_layout.npz and confirm_layout.npz",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--score-panels",
        nargs="*",
        choices=("dev", "confirm"),
        default=["dev", "confirm"],
        help="Score paired teacher targets only after locking predictions; pass no names for inference only.",
    )
    run(parser.parse_args())
