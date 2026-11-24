"""Calibrated, single-person recognition of unseen photos of 12 registered objects, from raw R4 EEG."""

import argparse
import json
from pathlib import Path

import numpy as np
from decode_eeg_layout import eeg_design, registration
from fit_official_continuous import regress
from fit_photo_conditions import axis_and_radius, normalize, restore_cone
from scipy.linalg import eigh

WORDS = (
    "airplane",
    "apple",
    "car",
    "chair",
    "clock",
    "dog",
    "guitar",
    "mug",
    "scissors",
    "shoe",
    "toothbrush",
    "umbrella",
)
SEED = 260425


def eeg_projection(x, train):
    """Between-concept covariance relative to within-concept photo covariance."""
    mu = x[train].mean(0)
    sd = x[train].std(0).clip(1e-4)
    x = (x - mu) / sd
    by_class = x[train].reshape(-1, 10, x.shape[1])
    means = by_class.mean(1)
    residual = (by_class - means[:, None]).reshape(-1, x.shape[1])
    within = residual.T @ residual / len(residual)
    within.flat[:: len(within) + 1] += 0.25
    between = means.T @ means / len(means)
    vals, vecs = eigh(between, within, subset_by_index=(len(mu) - 96, len(mu) - 1))
    weights = np.maximum(vals - 0.1, 0) / np.maximum(vals + 1.0, 1.0)
    return (x @ vecs * np.sqrt(weights)).astype(np.float32)


def panel(x, h, hpred, f, fpred, groups, rng, center_h, global_f, cone, mixes):
    n = len(groups)
    order = np.stack([rng.permutation(g) for g in groups])
    hits = {k: 0 for k in mixes}
    exports = []
    for fold in range(5):
        test_pos = np.arange(10) // 2 == fold
        cal, query = order[:, ~test_pos], order[:, test_pos].ravel()
        y = np.repeat(np.arange(n), 2)
        if np.intersect1d(cal, query).size:
            raise ValueError("Reconstruction query leaked into calibration exemplars")
        xe = x[cal].mean(1)
        xp = normalize(x[query]) @ normalize(xe).T
        hp = normalize(hpred[query] - center_h) @ normalize(h[cal].mean(1) - center_h).T

        # Equalize class-score scale independently for each incoming EEG.
        def standard(v):
            return (v - v.mean(1, keepdims=True)) / v.std(1, keepdims=True).clip(1e-6)

        xp, hp = standard(xp), standard(hp)
        for mix in mixes:
            prediction = ((1 - mix) * xp + mix * hp).argmax(1)
            hits[mix] += int((prediction == y).sum())
        if len(mixes) == 1:
            cl = f[cal].mean(1)[prediction]
            residual = fpred[query] - fpred[cal].mean(1)[prediction]
            calibrated = restore_cone(cl + residual, global_f, *cone)
            category = restore_cone(cl, global_f, *cone)
            exports.append((query, prediction, y, calibrated, category, np.full(len(y), fold)))
    return hits, exports


def registered_ablation(
    eeg, channels, projected, h, hpred, center_h, mix, rows, folds, truth, pred, words
):
    """Replay exactly the exported five folds, independent of the fit/selection RNG."""
    n = len(words)
    order = np.arange(len(rows)).reshape(5, n, 2)
    if not (
        np.array_equal(folds[order], np.broadcast_to(np.arange(5)[:, None, None], order.shape))
        and np.array_equal(truth[order], np.broadcast_to(np.arange(n)[None, :, None], order.shape))
    ):
        raise ValueError("Exported query row order no longer identifies the five calibration folds")
    # Compare the native, four-repeat-averaged voltage waveform without any
    # fitted transform to the same eight exemplars per class used by the heads.
    raw = eeg[rows][:, channels, :].reshape(len(rows), -1)
    predictions = {
        k: np.empty_like(pred) for k in ("raw_eeg", "projected_eeg", "visual_h14", "fused")
    }
    supports = []
    for fold in range(5):
        local_query = order[fold].ravel()
        local_cal = order[np.arange(5) != fold].transpose(1, 0, 2).reshape(n, 8)
        query, cal = rows[local_query], rows[local_cal]
        if np.intersect1d(query, cal).size:
            raise ValueError("A query appeared in its own calibration panel")
        supports.append(cal.tolist())
        raw_scores = normalize(raw[local_query]) @ normalize(raw[local_cal].mean(1)).T
        xp = normalize(projected[query]) @ normalize(projected[cal].mean(1)).T
        hp = normalize(hpred[query] - center_h) @ normalize(h[cal].mean(1) - center_h).T
        xp = (xp - xp.mean(1, keepdims=True)) / xp.std(1, keepdims=True).clip(1e-6)
        hp = (hp - hp.mean(1, keepdims=True)) / hp.std(1, keepdims=True).clip(1e-6)
        for arm, score in (
            ("raw_eeg", raw_scores),
            ("projected_eeg", xp),
            ("visual_h14", hp),
            ("fused", (1 - mix) * xp + mix * hp),
        ):
            predictions[arm][local_query] = score.argmax(1)
    if not np.array_equal(predictions["fused"], pred):
        raise ValueError("Fold replay disagrees with the frozen generation labels")
    seed = 260427
    bootstrap = np.random.default_rng(seed).integers(n, size=(10000, n))
    hit = {k: v == truth for k, v in predictions.items()}
    by_class = {k: np.bincount(truth, weights=v, minlength=n).astype(int) for k, v in hit.items()}
    counts = np.bincount(truth, minlength=n)
    paired = {}
    for key in ("raw_eeg", "projected_eeg", "visual_h14"):
        diff = (by_class["fused"] - by_class[key]) / counts
        paired[key] = dict(
            delta_fraction=float(diff.mean()),
            ci95=np.quantile(diff[bootstrap].mean(1), [0.025, 0.975]).tolist(),
            fused_only_correct=int(np.sum(hit["fused"] & ~hit[key])),
            other_only_correct=int(np.sum(hit[key] & ~hit["fused"])),
        )
    return dict(
        rows=rows.tolist(),
        folds=folds.tolist(),
        words=words.tolist(),
        truth=truth.tolist(),
        weights_projected_h14=[1 - mix, mix],
        calibration_rows_by_fold_and_word=supports,
        methods=dict(
            raw_eeg="Cosine to eight-ERP centroid, 17 posterior channels x 250 native samples; no fitted transform",
            projected_eeg="Cosine to eight-EEG centroid in learned between/within-covariance projection",
            visual_h14="EEG-predicted H/14 versus mean real H/14 of eight labeled calibration photos",
            fused="Frozen weighted sum of within-query standardized projected EEG and visual scores",
        ),
        results={
            k: dict(
                predicted=v.tolist(),
                hits=int(hit[k].sum()),
                total=len(rows),
                hits_by_word=by_class[k].tolist(),
            )
            for k, v in predictions.items()
        },
        fused_minus=paired,
        bootstrap_unit="12 registered concepts, paired percentile interval",
        bootstrap_seed=seed,
        bootstrap_samples=len(bootstrap),
    )


def main(args):
    import torch

    root = args.archive
    records, labels, split, channels = registration(root)
    subjects = [r["concept_id"].split("_", 1)[1] for r in records[::10]]
    ids = np.array([subjects.index(w) for w in WORDS])
    if args.replication:
        eligible = np.setdiff1d(np.arange(len(subjects)), ids)
        ids = np.random.default_rng(260426).choice(eligible, len(WORDS), replace=False)
    words = np.asarray(subjects)[ids]
    generic = np.setdiff1d(np.unique(labels), ids)
    rng = np.random.default_rng(SEED)
    validation = rng.choice(generic, 120, replace=False)
    train = np.flatnonzero(np.isin(labels, np.setdiff1d(generic, validation)))
    if np.intersect1d(labels[train], ids).size or np.intersect1d(labels[train], validation).size:
        raise ValueError("Registered/validation object entered global fitting")
    eeg = torch.load(
        root / "data/recovery_20261108_v236/sub-03_train_mean4.pt",
        mmap=True,
        weights_only=True,
        map_location="cpu",
    ).numpy()
    x = eeg_design(eeg, np.arange(len(labels)), channels)
    visual = torch.load(
        root / "data/native_training_v239/train_visual_embeddings.pt",
        mmap=True,
        weights_only=True,
        map_location="cpu",
    )
    h = normalize(visual["CLIP-ViT-H-14-laion2B-s32B-b79K"].numpy().astype(np.float32))
    f = (
        torch.load(
            root / "runs/HIVE_native_sub03_5096542/deterministic_F_all16540.pt",
            mmap=True,
            weights_only=True,
            map_location="cpu",
        )
        .numpy()
        .astype(np.float32)
    )
    vf = regress(x[train], np.column_stack((h[train], f[train])), x, 0.3)
    hp, fp = np.split(vf, 2, axis=1)
    xp = eeg_projection(x, train)
    hc, fc, cone = h[train].mean(0), f[train].mean(0), axis_and_radius(f[train])
    mixes = (0.0, 0.25, 0.5, 0.75, 1.0)
    counts = np.zeros(len(mixes), dtype=int)
    for cluster in validation.reshape(-1, 12):
        sets = np.stack([np.flatnonzero(labels == c) for c in cluster])
        hits, _ = panel(xp, h, hp, f, fp, sets, rng, hc, fc, cone, mixes)
        counts += [hits[k] for k in mixes]
    best = 0.25 if args.replication else mixes[int(counts.argmax())]
    groups = np.stack([np.flatnonzero(labels == c) for c in ids])
    hits, export = panel(xp, h, hp, f, fp, groups, rng, hc, fc, cone, (best,))
    rows, pred, truth, cond, class_only, folds = [
        np.concatenate(a) for a in zip(*export, strict=True)
    ]
    # All measured photo codes in source training are excluded from any registered class.
    t = np.array(split["T"])
    allowed = np.flatnonzero(np.isin(t, train))
    lat = np.load(root / "data/native_T_latents_v256/native_VAE_T_modes.npy", mmap_mode="r")
    coarse = np.empty((len(allowed), 4, 8, 8), np.float32)
    for lo in range(0, len(allowed), 64):
        ix = allowed[lo : lo + 64]
        coarse[lo : lo + len(ix)] = (
            lat[ix].astype(np.float32).reshape(-1, 4, 8, 8, 8, 8).mean((3, 5))
        )
    starts = regress(x[t[allowed]], coarse.reshape(len(allowed), -1), x[rows], 0.3).reshape(
        -1, 4, 8, 8
    )
    if len(np.unique(rows)) != 120:
        raise ValueError("A photograph must appear as a reconstruction query exactly once")
    args.out.mkdir(exist_ok=True, parents=True)
    np.savez(
        args.out / "conditions.npz",
        rows=rows,
        words=words[pred],
        condition=cond,
        class_only=class_only,
        start=starts,
        folds=folds,
    )
    (args.out / "evaluation.json").write_text(
        json.dumps(
            dict(
                words=words.tolist(),
                trial_count=4,
                rows=rows.tolist(),
                replication_seed=260426 if args.replication else None,
                folds=folds.tolist(),
                predicted=pred.tolist(),
                truth=truth.tolist(),
                exact_hits=hits[best],
                total=len(truth),
                n_calibration_photos_per_object=8,
                generic_train_photos=len(train),
                mix=best,
                mix_grid=mixes,
                validation_correct=counts.tolist(),
                validation_total=1200,
            ),
            indent=2,
        )
        + "\n"
    )
    diagnostics = registered_ablation(
        eeg, channels, xp, h, hp, hc, best, rows, folds, truth, pred, words
    )
    (args.out / "ablation.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(
        f"Registered objects, {hits[best]}/120, training-only validation {counts.tolist()}, mix={best}",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--replication",
        action="store_true",
        help="Test the frozen mix on twelve uniformly drawn additional concepts (seed 260426)",
    )
    main(p.parse_args())
