"""Map published single-participant ATM features to a Brain-HIVE condition and VAE layout.

The full training stimulus registration is checked by filename. Only the historical
T subset is used so every target has a cached VAE mode. Test inputs are EEG vectors.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from fit_photo_conditions import axis_and_radius, normalize, restore_cone


def regress(train, target, query, lam):
    mean = train.mean(axis=0)
    sd = train.std(axis=0).clip(1e-5)
    x = (train - mean) / sd
    ymean = target.mean(axis=0)
    gram = (x.T @ x).astype(np.float64)
    gram.flat[:: len(gram) + 1] += lam * len(x)
    w = np.linalg.solve(gram, x.T @ (target - ymean)).astype(np.float32)
    return ((query - mean) / sd) @ w + ymean


def score(prediction, target, ids, mean):
    q = normalize(prediction - mean)
    y = normalize(target - mean)
    unique, label = np.unique(ids, return_inverse=True)
    centers = normalize(np.stack([y[label == k].mean(0) for k in range(len(unique))]))
    return float(np.mean((q @ centers.T).argmax(axis=1) == label))


def main(args):
    import torch

    root = args.archive
    registered = json.loads((root / "data/native_training_v239/train_image_ids.json").read_text())
    ordered = args.train_names.read_text(encoding="utf-8").splitlines()
    if ordered != registered or len(set(ordered)) != 16540:
        raise ValueError(
            "Official training EEG feature identities and archived stimulus table disagree"
        )
    split = json.loads(
        (root / "data/recovery_20261108_v236/current_split_manifest.json").read_text()
    )
    rows = np.array(split["image_rows"]["T"])
    eeg = np.load(args.train, mmap_mode="r", allow_pickle=False)
    query = np.load(args.query, allow_pickle=False)
    if eeg.shape != (16540, 1024) or query.shape != (200, 1024):
        raise ValueError(
            "Expected per-photograph ATM EEG embeddings in the same 1024-coordinate space"
        )
    x = normalize(np.asarray(eeg[rows], dtype=np.float32))
    query = normalize(query.astype(np.float32))
    teacher = (
        torch.load(
            root / "runs/HIVE_native_sub03_5096542/deterministic_F_all16540.pt",
            weights_only=True,
            map_location="cpu",
            mmap=True,
        )
        .numpy()[rows]
        .astype(np.float32)
    )
    image = np.load(root / "data/native_T_latents_v256/native_VAE_T_modes.npy", mmap_mode="r")
    if image.shape != (len(rows), 4, 64, 64):
        raise ValueError("Unexpected T-only latent layout")
    pooled = np.empty((len(rows), 4, 8, 8), dtype=np.float32)
    for start in range(0, len(rows), 64):
        block = image[start : start + 64].astype(np.float32)
        pooled[start : start + len(block)] = block.reshape(-1, 4, 8, 8, 8, 8).mean((3, 5))
    target = np.column_stack([teacher, pooled.reshape(len(rows), -1)])
    if not all(np.isfinite(v).all() for v in (target, x, query)):
        raise ValueError("Nonfinite training or inference input")
    ids = rows // 10
    hold = np.isin(ids, np.random.default_rng(531466).permutation(np.unique(ids))[::5])
    trial = {}
    for lam in (0.1, 1.0, 5.0):
        yh = regress(x[~hold], target[~hold], x[hold], lam)
        trial[lam] = score(yh[:, :1024], teacher[hold], ids[hold], teacher[~hold].mean(0))
    lam = max(trial, key=trial.get)
    pred = regress(x, target, query, lam)
    cone = restore_cone(pred[:, :1024], teacher.mean(0), *axis_and_radius(teacher))
    args.out.parent.mkdir(exist_ok=True, parents=True)
    np.savez(
        args.out,
        rows=np.arange(len(query)),
        cone=cone,
        linear=pred[:, :1024],
        layout=pred[:, 1024:].reshape(-1, 4, 8, 8),
        mean_condition=teacher.mean(0),
        mean_layout=pooled.mean(0),
    )
    print(
        json.dumps(
            dict(
                train=len(rows),
                inner_held_concepts=len(np.unique(ids[hold])),
                validation_category_accuracy=trial,
                selected_ridge=lam,
                source="published ATM embedding averaged per training photo; official single-subject R80 test EEG",
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "train", "train-names", "query", "out"):
        parser.add_argument("--" + arg, required=True, type=Path)
    main(parser.parse_args())
