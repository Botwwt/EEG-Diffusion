"""Fit four fixed sub-03 EEG-to-VAE start decoders on historical T only.

Run with --prep PREP --archive LEGACY_ROOT --out OUTPUT [--n-concepts 20].
instance_starts.npz contains original train-manifest row IDs and one float32
NCHW array per arm. The starts are 4x4 area means in *scaled* VAE mode units.
Held-out images, latents, ranks and concept peers are unnecessary at inference.
"""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from probe_context_residual import (
    CHUNK,
    CONTEXT_RIDGE,
    EEG_RIDGE,
    GRID,
    TIME_BIN,
    apply_context,
    eeg_features,
    fit_context,
    read_latents,
)

SEED = 20261126
CHANNELS = "P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2".split()
ARMS = {
    "raw_baseline": "Global EEG ridge to raw pooled VAE grid.",
    "residual_head": "Frozen visual context(raw_baseline) plus global EEG ridge to signed residual.",
    "within_raw": "Raw global ridge plus anchored correction estimated from within-concept photo differences.",
    "within_residual": "Same frozen context(raw_baseline), plus anchored within-concept correction to the residual head.",
}


def read_registration(archive):
    data = archive / "data"
    eeg_dir = data / "recovery_20261108_v236"
    visual = data / "native_T_latents_v256"
    manifest = data / "full_T_recovery_20261108_v238/image_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        stimuli = sorted(
            (r for r in csv.DictReader(f) if r["dataset_split"] == "train"),
            key=lambda r: int(r["array_row"]),
        )
    image_ids = json.loads((data / "native_training_v239/train_image_ids.json").read_text())
    layout = json.loads((eeg_dir / "sub-03_layout.json").read_text())
    if (
        [int(r["array_row"]) for r in stimuli] != list(range(16540))
        or [Path(r["stimulus_id"]).stem for r in stimuli] != image_ids
        or len(set(image_ids)) != 16540
        or layout["selected_image_indices"] != list(range(16540))
    ):
        raise ValueError("Image manifest, stimulus IDs, and EEG row identities disagree")
    labels = [r["concept_id"] for r in stimuli]
    lookup = {label: i for i, label in enumerate(dict.fromkeys(labels))}
    concept = np.array([lookup[label] for label in labels])
    if len(lookup) != 1654 or not np.array_equal(concept, np.arange(16540) // 10):
        raise ValueError("Expected 1654 ordered ten-photo concepts")
    channels = layout["channels"]
    if len(channels) != 63 or len(set(channels)) != 63 or not set(CHANNELS).issubset(channels):
        raise ValueError("Posterior EEG channel identities missing or ambiguous")
    registration = json.loads((visual / "registration.json").read_text())
    parts = json.loads((eeg_dir / "current_split_manifest.json").read_text())["image_rows"]
    t = np.asarray(parts["T"])
    if (
        not np.issubdtype(t.dtype, np.integer)
        or t.shape != (13230,)
        or len(np.unique(t)) != len(t)
        or t.min() < 0
        or t.max() >= len(concept)
        or t.tolist() != registration["array_rows"]
        or np.intersect1d(t, parts["D"] + parts["V"]).size
    ):
        raise ValueError("Historical T rows disagree with visual registration or split manifest")
    counts = np.bincount(concept[t], minlength=1654)
    if not np.isin(counts, (0, 10)).all():
        raise ValueError("Historical T does not consist of complete ten-photo concepts")
    factor = float(registration["VAE_scaling_factor"])
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Missing positive VAE scaling factor in original registration")
    return t, concept, image_ids, labels, [channels.index(c) for c in CHANNELS], factor


def choose_eval_rows(t, concept, n_concepts):
    pool = np.setdiff1d(np.unique(concept), np.unique(concept[t]))
    if not 1 <= n_concepts <= len(pool):
        raise ValueError(f"--n-concepts must be 1..{len(pool)}")
    chosen = np.sort(np.random.default_rng(SEED).choice(pool, n_concepts, replace=False))
    rows = np.flatnonzero(np.isin(concept, chosen))
    if not np.array_equal(concept[rows], np.repeat(chosen, 10)):
        raise ValueError("Evaluation concepts must contain exactly ten ordered photos")
    return chosen, rows


def read_signals(prep, eeg_path, t, rows, channels):
    import torch

    archive = torch.load(eeg_path, weights_only=True, map_location="cpu", mmap=True).numpy()
    prepared = np.load(prep / "sub-03_train.npy", mmap_mode="r")
    if archive.shape != (16540, 63, 250) or prepared.shape != (len(t), 17, 250):
        raise ValueError("Expected archived 63-channel and prepared 17-channel four-trial means")
    for start in range(0, len(t), CHUNK):
        idx = t[start : start + CHUNK]
        if not np.array_equal(prepared[start : start + len(idx)], archive[idx][:, channels, :]):
            raise ValueError(
                f"Prepared EEG disagrees with original row/channel mapping at position {start}"
            )
    train_x = eeg_features(prepared, np.arange(len(t)), "mean4")
    selected = np.empty((len(rows), 17, 250), dtype=np.float32)
    for start in range(0, len(rows), CHUNK):
        idx = rows[start : start + CHUNK]
        selected[start : start + len(idx)] = archive[idx][:, channels, :]
    return train_x, eeg_features(selected, np.arange(len(rows)), "mean4")


def group_center(a, groups, group_count):
    sums = np.zeros((len(group_count), a.shape[1]), dtype=np.float64)
    np.add.at(sums, groups, a)
    return a - (sums / group_count[:, None])[groups]


def global_and_within_predictions(train_x, eval_x, y, concept):
    """Shared standardized design, identical raw/residual penalties, one anchored update."""
    mean = train_x.mean(axis=0, dtype=np.float64)
    std = train_x.std(axis=0, dtype=np.float64).clip(1e-6)
    x = (train_x - mean) / std
    target_mean = y.mean(axis=0, dtype=np.float64)
    ridge = EEG_RIDGE * len(x)
    gram = x.T @ x
    gram.flat[:: len(gram) + 1] += ridge
    base = np.linalg.solve(gram, x.T @ (y - target_mean))
    _, groups, counts = np.unique(concept, return_inverse=True, return_counts=True)
    xw = group_center(x, groups, counts)
    yw = group_center(y, groups, counts)
    within_gram = xw.T @ xw
    within_gram.flat[:: len(within_gram) + 1] += ridge
    delta = np.linalg.solve(within_gram, xw.T @ (yw - xw @ base))
    xe = (eval_x - mean) / std
    base_pred = target_mean + xe @ base
    return base_pred.astype(np.float32), (base_pred + xe @ delta).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--n-concepts",
        type=int,
        default=20,
        help="Number of concepts to predict. The fixed generation comparison uses 20.",
    )
    args = parser.parse_args()
    data = args.archive / "data"
    t, concept, image_ids, labels, channels, scale = read_registration(args.archive)
    chosen, rows = choose_eval_rows(t, concept, args.n_concepts)
    prep_labels = np.load(args.prep / "train_concept.npy")
    valid = np.load(args.prep / "lat512_train_ok.npy")
    if not np.array_equal(prep_labels, concept[t]) or not np.array_equal(
        valid, np.ones(len(t), bool)
    ):
        raise ValueError("Prepared training concept/validity arrays disagree with historical T")
    latent_file = args.prep / "lat512_train.npy"
    original = data / "native_T_latents_v256/native_VAE_T_modes.npy"
    bank = np.load(latent_file, mmap_mode="r")
    if bank.shape != (len(t), 4, 64, 64):
        raise ValueError("Expected original 64x64 four-channel VAE modes")
    if not latent_file.samefile(original):
        source = np.load(original, mmap_mode="r")
        if bank.shape != source.shape or bank.dtype != source.dtype:
            raise ValueError("Prepared VAE bank shape or dtype differs from original")
        for start in range(0, len(t), CHUNK):
            if not np.array_equal(bank[start : start + CHUNK], source[start : start + CHUNK]):
                raise ValueError(f"Prepared VAE bank differs from the original at position {start}")
    train_x, eval_x = read_signals(
        args.prep, data / "recovery_20261108_v236/sub-03_train_mean4.pt", t, rows, channels
    )
    raw = read_latents(bank, np.arange(len(t)))
    model = fit_context(raw)
    residual = raw - apply_context(raw, model)
    y = np.concatenate((raw.reshape(len(t), -1), residual.reshape(len(t), -1)), axis=1)
    global_pred, within_pred = global_and_within_predictions(train_x, eval_x, y, concept[t])
    shape = (len(rows), *raw.shape[1:])
    raw_base, res_base = [p.reshape(shape) for p in np.hsplit(global_pred, 2)]
    raw_within, res_within = [p.reshape(shape) for p in np.hsplit(within_pred, 2)]
    fixed_context = apply_context(raw_base, model)
    arms = dict(
        zip(
            ARMS,
            (raw_base, fixed_context + res_base, raw_within, fixed_context + res_within),
            strict=True,
        )
    )
    starts = {k: np.ascontiguousarray(v.transpose(0, 3, 1, 2)) for k, v in arms.items()}
    if not all(np.isfinite(v).all() for v in starts.values()):
        raise ValueError("Nonfinite EEG-predicted VAE starts")
    metadata = {
        "subject": "sub-03",
        "eeg": "archived four-physical-trial voltage mean",
        "channels": CHANNELS,
        "train_partition": "all historical T",
        "train_images": len(t),
        "train_concepts": len(np.unique(concept[t])),
        "selection_seed": SEED,
        "selection": "Sorted sample without replacement from full concept complement of T",
        "selected_concepts": chosen.tolist(),
        "selected_concept_labels": [labels[r] for r in rows[::10]],
        "rows": rows.tolist(),
        "image_ids": [image_ids[i] for i in rows],
        "arm_names": list(ARMS),
        "arm_definitions": ARMS,
        "grid": GRID,
        "axis_order": "NCHW",
        "units": "native VAE mode times scaling factor",
        "archive_vae_scaling_factor": scale,
        "additional_scaling_applied": False,
        "eeg_features": train_x.shape[1],
        "eeg_time_bin_samples": TIME_BIN,
        "ridge_per_train_image_global_and_anchored": EEG_RIDGE,
        "within_intercept": "Fixed global target mean, centered global train EEG features at application",
        "context_ridge_per_train_cell": CONTEXT_RIDGE,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out) as tmp:
        staging = Path(tmp)
        np.savez(
            staging / "instance_starts.npz",
            rows=rows,
            concept_ids=concept[rows],
            arm_names=np.asarray(list(ARMS)),
            **starts,
        )
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
        for name in ("metadata.json", "instance_starts.npz"):
            (staging / name).replace(args.out / name)
    print(
        f"Saved four aligned sets of {len(rows)} sub-03 starts from {len(t)} training rows.",
        flush=True,
    )


if __name__ == "__main__":
    main()
