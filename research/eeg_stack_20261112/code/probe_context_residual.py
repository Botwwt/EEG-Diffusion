"""CPU diagnostic of photo-specific EEG decoding with a coarse VAE-latent proxy.

Run: python probe_context_residual.py --subjects 2 5 8 9 --out probe.json
Reads aligned .npy caches under $RAD_ROOT/data/prep. No generator or I-JEPA.
The pooled latent cells are disjoint. Their underlying pixel receptive fields
can overlap. Every transformation is fitted on folds 1..4, and retrieval is
among ten photos of the same unseen fold-0 concept.
"""

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

GRID = 4
CHUNK = 128
TIME_BIN = 10
FOLD_SEED = 20261112
SHUFFLE_SEED = 20261124
EEG_RIDGE = 0.10
CONTEXT_RIDGE = 0.001


def groups_by_fold(concept, valid, num_concepts):
    if (
        concept.ndim != 1
        or valid.shape != concept.shape
        or not np.issubdtype(concept.dtype, np.integer)
    ):
        raise ValueError("Expected aligned one-dimensional integer concepts and validity flags")
    if not len(concept) or num_concepts < 5 or concept.min() < 0 or concept.max() >= num_concepts:
        raise ValueError("Concept IDs must lie in the declared original concept universe")
    # Same fixed assignment as common.concept_folds, without its GPU imports.
    permutation = np.random.default_rng(FOLD_SEED).permutation(num_concepts)
    fold = np.empty_like(permutation)
    fold[permutation] = np.arange(len(fold)) % 5
    order = np.argsort(concept, kind="stable")
    boundaries = np.flatnonzero(np.diff(concept[order])) + 1
    groups = [g for g in np.split(order, boundaries) if len(g) == 10 and valid[g].all()]
    if not groups:
        raise ValueError("No concepts with all ten valid photographs")
    groups = np.stack(groups)
    is_test = fold[concept[groups[:, 0]]] == 0
    if not is_test.any() or is_test.all():
        raise ValueError("Both train and evaluation need complete ten-photo concepts")
    return groups[~is_test], groups[is_test]


def within_concept_derangement(n_groups, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random((n_groups, 10)), axis=1)
    perm = np.empty_like(order)
    np.put_along_axis(perm, order, np.roll(order, 1, axis=1), axis=1)
    return (10 * np.arange(n_groups)[:, None] + perm).ravel()


def read_latents(latents, rows):
    if latents.ndim != 4:
        raise ValueError("Expected VAE latents in [photo, channel, height, width] order")
    _, c, h, w = latents.shape
    if h % GRID or w % GRID:
        raise ValueError("VAE dimensions must be divisible by four")
    result = np.empty((len(rows), GRID, GRID, c), dtype=np.float32)
    for start in range(0, len(rows), CHUNK):
        idx = rows[start : start + CHUNK]
        tile = np.asarray(latents[idx], dtype=np.float32)
        pooled = tile.reshape(len(idx), c, GRID, h // GRID, GRID, w // GRID).mean(axis=(3, 5))
        result[start : start + len(idx)] = pooled.transpose(0, 2, 3, 1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite pooled latents in valid photographs")
    return result


def eeg_features(cache, rows, eeg_input):
    expected = (4, 17, 250) if eeg_input == "trials4" else (17, 250)
    if cache.shape[1:] != expected:
        raise ValueError(f"{eeg_input} expects EEG [photo, {expected}], got {cache.shape}")
    times = cache.shape[-1]
    starts = np.arange(0, times, TIME_BIN)
    lengths = np.diff(np.append(starts, times))
    result = np.empty((len(rows), cache.shape[-2] * len(starts)), dtype=np.float32)
    for first in range(0, len(rows), CHUNK):
        idx = rows[first : first + CHUNK]
        tile = cache[idx]
        erp = (
            np.mean(tile, axis=1, dtype=np.float32)
            if eeg_input == "trials4"
            else tile.astype(np.float32)
        )
        pooled = np.add.reduceat(erp, starts, axis=2) / lengths
        result[first : first + len(idx)] = pooled.reshape(len(idx), -1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite EEG features")
    return result


def context_design(raw, mean, scale):
    x = (raw - mean) / scale
    n, h, w, _ = x.shape
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    neighbors = [
        padded[:, :h, 1 : w + 1],
        padded[:, 2 : h + 2, 1 : w + 1],
        padded[:, 1 : h + 1, :w],
        padded[:, 1 : h + 1, 2 : w + 2],
    ]
    around = np.concatenate(neighbors, axis=-1)
    positions = np.indices((h, w))
    boundaries = np.stack(
        [positions[0] == 0, positions[0] == h - 1, positions[1] == 0, positions[1] == w - 1],
        axis=-1,
    )
    edge = np.broadcast_to(boundaries, (n, h, w, 4))
    # A fixed nonlinear basis plus an unpenalized intercept. No target cell input.
    design = np.concatenate((around, np.tanh(around), edge, np.ones_like(edge[..., :1])), axis=-1)
    return design.reshape(n * h * w, -1)


def fit_context(raw):
    mean = raw.mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    scale = raw.std(axis=(0, 1, 2), dtype=np.float64).clip(1e-6).astype(np.float32)
    c = raw.shape[-1]
    p = 8 * c + 5
    gram = np.zeros((p, p), dtype=np.float64)
    cross = np.zeros((p, c), dtype=np.float64)
    for start in range(0, len(raw), CHUNK):
        tile = raw[start : start + CHUNK]
        design = context_design(tile, mean, scale).astype(np.float64)
        gram += design.T @ design
        cross += design.T @ tile.reshape(-1, c)
    gram.flat[:: p + 1] += CONTEXT_RIDGE * len(raw) * GRID**2
    gram[-1, -1] -= CONTEXT_RIDGE * len(raw) * GRID**2
    weight = np.linalg.solve(gram, cross).astype(np.float32)
    return mean, scale, weight


def apply_context(raw, model):
    mean, scale, weight = model
    pred = np.empty_like(raw)
    for start in range(0, len(raw), CHUNK):
        tile = raw[start : start + CHUNK]
        pred[start : start + len(tile)] = (context_design(tile, mean, scale) @ weight).reshape(
            tile.shape
        )
    return pred


def decode(x_train, x_eval, targets):
    # One factorization and identical features and penalties for all three arms.
    mean = x_train.mean(axis=0, dtype=np.float64)
    scale = x_train.std(axis=0, dtype=np.float64).clip(1e-6)
    x = (x_train - mean) / scale
    y = np.concatenate([target.reshape(len(x), -1) for target in targets], axis=1)
    center = y.mean(axis=0, dtype=np.float64)
    gram = x.T @ x
    gram.flat[:: len(gram) + 1] += EEG_RIDGE * len(x)
    weight = np.linalg.solve(gram, x.T @ (y - center))
    pred = (((x_eval - mean) / scale) @ weight + center).astype(np.float32)
    shape = (len(x_eval),) + targets[0].shape[1:]
    return [part.reshape(shape) for part in np.hsplit(pred, len(targets))]


def evaluate(pred, truth, train_center, perm):
    def unit_rows(x):
        flat = (x - train_center).reshape(len(x), -1)
        return (flat / np.linalg.norm(flat, axis=1, keepdims=True).clip(1e-9)).reshape(
            -1, 10, flat.shape[1]
        )

    q, g = unit_rows(pred), unit_rows(truth)
    output = {}
    per_concept = {}
    for name, queries in (
        ("paired", q),
        ("shuffled_eeg", q.reshape(-1, q.shape[-1])[perm].reshape(q.shape)),
    ):
        score = queries @ g.transpose(0, 2, 1)
        correct = np.diagonal(score, axis1=1, axis2=2)
        maximum = score.max(axis=2, keepdims=True)
        top = (correct == maximum[..., 0]) / (score == maximum).sum(axis=2)
        i, j = np.triu_indices(10, 1)
        margin = correct[:, i] + correct[:, j] - score[:, i, j] - score[:, j, i]
        top_by_concept = top.mean(axis=1)
        output[name] = {
            "top1": float(top_by_concept.mean()),
            "swap_win": float(np.mean((margin > 0) + 0.5 * (margin == 0))),
        }
        per_concept[name] = top_by_concept.tolist()
    output["paired_r2_vs_train_mean"] = float(
        1
        - np.square(pred - truth).sum(dtype=np.float64)
        / max(np.square(truth - train_center).sum(dtype=np.float64), 1e-12)
    )
    output["top1_by_concept"] = per_concept
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", type=int, default=[2, 5, 8, 9])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prep", type=Path, help="Override $RAD_ROOT/data/prep")
    parser.add_argument(
        "--eeg-input",
        choices=("trials4", "mean4"),
        default="trials4",
        help="Use separate voltage trials or a cached genuine four-trial voltage mean",
    )
    parser.add_argument("--protocol", default=None, help="Label for the provenance of this run")
    parser.add_argument(
        "--num-concepts",
        type=int,
        default=1654,
        help="Original fold universe, even for sparse caches",
    )
    args = parser.parse_args()
    if args.prep is None and not os.environ.get("RAD_ROOT"):
        parser.error("Provide RAD_ROOT or --prep")
    prep = args.prep or Path(os.environ["RAD_ROOT"]) / "data/prep"
    concept = np.load(prep / "train_concept.npy")
    valid = np.load(prep / "lat512_train_ok.npy").astype(bool)
    train, test = groups_by_fold(concept, valid, args.num_concepts)
    n = train.size
    rows = np.concatenate((train.ravel(), test.ravel()))
    cache = np.load(prep / "lat512_train.npy", mmap_mode="r")
    if len(cache) != len(concept):
        raise ValueError("VAE cache and concepts differ in length")
    raw = read_latents(cache, rows)
    model = fit_context(raw[:n])
    visual_context = apply_context(raw, model)
    residual = raw - visual_context
    raw_center = raw[:n].mean(axis=0)
    residual_center = residual[:n].mean(axis=0)
    mse_context = np.square(residual[n:]).mean(dtype=np.float64)
    mse_mean = np.square(raw[n:] - raw_center).mean(dtype=np.float64)
    train_perm = within_concept_derangement(len(train), SHUFFLE_SEED)
    test_perm = within_concept_derangement(len(test), SHUFFLE_SEED + 1)
    result = {
        "description": "Development EEG retrieval diagnostic in pooled VAE space. No I-JEPA or image generation.",
        "protocol": {
            "label": args.protocol or f"vae-context-proxy-{args.eeg_input}",
            "eeg_input": args.eeg_input,
            "original_concept_universe": args.num_concepts,
            "fold_seed": FOLD_SEED,
            "shuffle_seed_train": SHUFFLE_SEED,
            "shuffle_seed_test": SHUFFLE_SEED + 1,
            "grid": GRID,
            "eeg_bin_samples": TIME_BIN,
            "eeg_trials": 4,
            "eeg_ridge_per_row": EEG_RIDGE,
            "context_ridge_per_cell": CONTEXT_RIDGE,
            "train_concepts": len(train),
            "eval_concepts": len(test),
            "photos_per_concept": 10,
            "eval_concept_ids": concept[test[:, 0]].tolist(),
            "subjects": args.subjects,
        },
        "definitions": {
            "raw": "4x4 disjoint mean-pooled VAE target cells, all channels",
            "context": "Shared train-fitted ridge from adjacent cells and tanh(adjacent cells) to held-out cell",
            "residual": "Signed raw cell values minus their context prediction",
            "raw_baseline": "EEG->raw ridge; in residual gallery projected as raw_pred - context(raw_pred)",
            "residual_head": "EEG->residual ridge; in raw gallery added to context(raw_pred)",
            "deranged_residual_head": "Same residual ridge with training targets deranged within concept",
            "metrics": "Cosine after per-coordinate train gallery centering; ties receive fractional credit. Rates in [0,1].",
        },
        "visual_context_r2_vs_train_mean": float(1 - mse_context / max(mse_mean, 1e-12)),
        "subjects": {},
    }
    for subject in args.subjects:
        key = f"sub-{subject:02d}"
        eeg = np.load(prep / f"{key}_train.npy", mmap_mode="r")
        if len(eeg) != len(concept):
            raise ValueError(f"{key} EEG cache and concepts differ in length")
        x = eeg_features(eeg, rows, args.eeg_input)
        q_raw, q_res, q_wrong = decode(x[:n], x[n:], (raw[:n], residual[:n], residual[train_perm]))
        q_context = apply_context(q_raw, model)
        scores = {}
        for space, truth, center, predictions in (
            ("raw", raw[n:], raw_center, (q_raw, q_context + q_res, q_context + q_wrong)),
            ("residual", residual[n:], residual_center, (q_raw - q_context, q_res, q_wrong)),
        ):
            scores[space] = {
                name: evaluate(q, truth, center, test_perm)
                for name, q in zip(
                    ("raw_baseline", "residual_head", "deranged_residual_head"),
                    predictions,
                    strict=True,
                )
            }
        result["subjects"][key] = scores
        summary = [
            round(scores["residual"][arm]["paired"]["top1"], 4) for arm in scores["residual"]
        ]
        print(
            f"{key} residual-gallery Top1 (raw baseline, residual head, deranged head): {summary}",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=args.out.parent, delete=False, encoding="utf-8") as f:
        json.dump(result, f, indent=1, allow_nan=False)
        f.write("\n")
    os.replace(f.name, args.out)


if __name__ == "__main__":
    main()
