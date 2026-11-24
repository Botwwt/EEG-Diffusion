"""Decode 32x32 original-photo RGB layouts from sub-03 four-repeat EEG on the CPU.

The development gallery reproduces the 20-concept VAE-start pilot. Confirmation
uses disjoint complete concepts from historical V. All coefficients and gains
are fit exclusively on historical T. No evaluation photograph enters decoding.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np

GRID = 32
PILOT_SEED = 20261126
CONFIRM_SEED = 20261125
CALIBRATION_SEED = 53117
EEG_SHA = "85a25128f6b94e3739478c60e089d33f437a8dcbeb31cf136ce9772d2cc1615d"
CHANNELS = "P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2".split()
RIDGE = 0.3


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 << 20), b""):
            h.update(data)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def registration(root):
    recovery = root / "data/recovery_20261108_v236"
    manifest = root / "data/full_T_recovery_20261108_v238/image_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        records = sorted(
            (r for r in csv.DictReader(stream) if r["dataset_split"] == "train"),
            key=lambda r: int(r["array_row"]),
        )
    layout = load_json(recovery / "sub-03_layout.json")
    names = load_json(root / "data/native_training_v239/train_image_ids.json")
    n = len(records)
    if (
        n != 16540
        or [int(r["array_row"]) for r in records] != list(range(n))
        or [Path(r["stimulus_id"]).stem for r in records] != names
        or len(set(names)) != n
        or layout["selected_image_indices"] != list(range(n))
        or len({r["image_member"] for r in records}) != n
    ):
        raise ValueError("The EEG row IDs, stimulus IDs or original-photo manifest do not agree")
    groups = {}
    concept = np.array([groups.setdefault(r["concept_id"], len(groups)) for r in records])
    if not np.array_equal(concept, np.arange(n) // 10):
        raise ValueError("Expected 1,654 complete ten-photo concepts in historical row order")
    split = {
        k: np.asarray(v, dtype=np.int64)
        for k, v in load_json(recovery / "current_split_manifest.json")["image_rows"].items()
        if k in ("T", "D", "V")
    }
    if (
        len(split["T"]) != 13230
        or not np.array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(n))
        or any(np.any(np.unique(concept[v], return_counts=True)[1] != 10) for v in split.values())
    ):
        raise ValueError("Historical T/D/V must partition the manifest by complete concepts")
    channels = layout["channels"]
    if len(channels) != 63 or len(set(channels)) != 63:
        raise ValueError("Expected a unique 63-channel EEG layout")
    return records, concept, split, [channels.index(c) for c in CHANNELS]


def evaluation_rows(concept, split, dev_starts, n_confirm):
    pool = np.setdiff1d(np.unique(concept), np.unique(concept[split["T"]]))
    selected = np.random.default_rng(PILOT_SEED).choice(pool, 20, replace=False)
    dev = np.flatnonzero(np.isin(concept, selected))
    if dev_starts is not None:
        with np.load(dev_starts, allow_pickle=False) as inputs:
            if not np.array_equal(inputs["rows"], dev):
                raise ValueError(
                    "--dev-starts differs from the original 20-concept development pilot"
                )
    eligible = np.setdiff1d(np.unique(concept[split["V"]]), selected)
    if not 1 <= n_confirm <= len(eligible):
        raise ValueError("Requested confirmation concept count exceeds the eligible V pool")
    confirm = np.random.default_rng(CONFIRM_SEED).choice(eligible, n_confirm, replace=False)
    return {"dev": dev, "confirm": np.flatnonzero(np.isin(concept, confirm))}


def read_rgb(archive, records, ids, label):
    from PIL import Image

    # Match torchvision's PIL Resize(512, BILINEAR), CenterCrop(512), then area-pool RGB.
    resize = getattr(Image, "Resampling", Image)
    result = np.empty((len(ids), 3, GRID, GRID), np.float32)
    with zipfile.ZipFile(archive) as source:
        for i, idx in enumerate(ids):
            row = records[idx]
            blob = source.read(row["image_member"])
            if (
                len(blob) != int(row["image_bytes"])
                or hashlib.sha256(blob).hexdigest() != row["image_sha256"]
            ):
                raise ValueError(f"Photo identity mismatch for original row {idx}")
            with Image.open(io.BytesIO(blob)) as photo:
                rgb = photo.convert("RGB")
                w, h = rgb.size
                size = (512, int(512 * h / w)) if w <= h else (int(512 * w / h), 512)
                rgb = rgb.resize(size, resize.BILINEAR)
                left, top = round((size[0] - 512) / 2), round((size[1] - 512) / 2)
                crop = np.asarray(rgb.crop((left, top, left + 512, top + 512)), dtype=np.uint8)
                result[i] = (
                    crop.reshape(GRID, 512 // GRID, GRID, 512 // GRID, 3)
                    .mean((1, 3))
                    .transpose(2, 0, 1)
                    / 255
                )
            if (i + 1) % 1000 == 0:
                print(
                    f"{label}: verified and pooled {i + 1}/{len(ids)} original photos", flush=True
                )
    return result


def eeg_design(eeg, ids, channels):
    parts = []
    for lo in range(0, len(ids), 128):
        sample = eeg[ids[lo : lo + 128]][:, channels, :]
        if not np.isfinite(sample).all():
            raise ValueError("Nonfinite mean-four EEG")
        # Raw baseline uses 40 ms signed voltage means over the full epoch.
        full = sample.reshape(len(sample), 17, 25, 10).mean(-1)
        # A 20 ms early-window average and non-phase-locked local variation supply extra features.
        early = sample[:, :, 10:130].reshape(len(sample), 17, 24, 5).mean(-1)
        blocks = sample.reshape(len(sample), 17, 10, 25)
        power = np.log1p(np.square(blocks - blocks.mean(-1, keepdims=True)).mean(-1))
        parts.append(
            np.concatenate([v.reshape(len(sample), -1) for v in (full, early, power)], axis=1)
        )
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)


def fit_predict(train_x, targets, test_x, robust):
    from scipy.linalg import cho_factor, cho_solve

    dim = train_x.shape[1] if robust else 425
    x, xe = train_x[:, :dim], test_x[:, :dim]
    mu = x.mean(0, dtype=np.float64).astype(np.float32)
    sd = x.std(0, dtype=np.float64).clip(1e-6).astype(np.float32)
    x, xe = (x - mu) / sd, (xe - mu) / sd
    if robust:
        np.clip(x, -5, 5, out=x)
        np.clip(xe, -5, 5, out=xe)
    xbar = x.mean(0, dtype=np.float64).astype(np.float32)
    x -= xbar
    mean = targets.mean(0, dtype=np.float64).astype(np.float32)
    gram = (x.T @ x).astype(np.float64)
    gram.flat[:: len(gram) + 1] += RIDGE * len(x)
    cross = np.zeros((dim, mean.size), np.float64)
    for lo in range(0, len(x), 512):
        y = (targets[lo : lo + 512] - mean).reshape(len(targets[lo : lo + 512]), -1)
        cross += x[lo : lo + 512].T @ y
    w = cho_solve(cho_factor(gram, check_finite=False), cross, check_finite=False).astype(
        np.float32
    )
    pred = (xe - xbar) @ w
    return mean, mean + pred.reshape(len(xe), *mean.shape)


def bands(x):
    # Nested, orthogonal block averages divide a layout into color, coarse structure, and detail.
    dc = x.mean((-1, -2), keepdims=True)
    coarse = x.reshape(len(x), 3, 4, GRID // 4, 4, GRID // 4).mean((3, 5))
    coarse = np.repeat(np.repeat(coarse, GRID // 4, axis=2), GRID // 4, axis=3)
    return dc, coarse - dc, x - coarse


def calibrate(mean, prediction, target):
    gains, boosts = [], []
    for p, y in zip(bands(prediction - mean), bands(target - mean), strict=True):
        energy = np.mean(p * p, axis=(0, 2, 3), dtype=np.float64).clip(1e-12)
        cross = np.mean(y * p, axis=(0, 2, 3), dtype=np.float64)
        target_energy = np.mean(y * y, axis=(0, 2, 3), dtype=np.float64)
        gain = np.clip(cross / energy, 0, 2)
        # Optional display arm partially restores training-target RMS on a separate scale.
        extra = np.clip((target_energy / (energy * gain**2).clip(1e-12)) ** 0.25, 1, 2)
        gains.append(gain.astype(np.float32))
        boosts.append((gain * extra).astype(np.float32))
    return np.stack(gains), np.stack(boosts)


def apply_gains(prediction, mean, gains):
    raw = mean + sum(
        p * gain[None, :, None, None]
        for p, gain in zip(bands(prediction - mean), gains, strict=True)
    )
    return np.ascontiguousarray(np.clip(raw, 0, 1), dtype=np.float32)


def shuffled_indices(labels, seed):
    rng = np.random.default_rng(seed)
    shuffle = np.arange(len(labels))
    for group in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == group))
        shuffle[indices] = np.roll(indices, 1)
    if np.any(shuffle == np.arange(len(labels))):
        raise ValueError("The EEG control needs a distinct image in every pairing")
    return shuffle


def cosine(a, b, center=False):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    if center:
        a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    product = np.sum(a * b, axis=1, dtype=np.float64)
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(product, norm, out=np.zeros_like(product), where=norm > 1e-10)


def metrics(p, y, mean):
    from scipy.ndimage import gaussian_filter

    # Gaussian-window RGB SSIM on the 32x32 layouts. Remove filter boundary pixels.
    def blur(v):
        return gaussian_filter(v, sigma=(0, 0, 1.5, 1.5), truncate=3.5)

    xp, yp = blur(p), blur(y)
    cov = blur(p * y) - xp * yp
    vx, vy = np.maximum(blur(p * p) - xp * xp, 0), np.maximum(blur(y * y) - yp * yp, 0)
    ssim = ((2 * xp * yp + 0.01**2) * (2 * cov + 0.03**2)) / (
        (xp * xp + yp * yp + 0.01**2) * (vx + vy + 0.03**2)
    )
    residual, truth = p - mean, y - mean
    pooled_p = p.reshape(len(p), 3, 4, GRID // 4, 4, GRID // 4).mean((3, 5))
    pooled_y = y.reshape(len(y), 3, 4, GRID // 4, 4, GRID // 4).mean((3, 5))
    center = slice(GRID // 4, 3 * GRID // 4)
    return {
        "rgb_pixel_corr": cosine(p, y, center=True),
        "rgb_rmse": np.sqrt(((p - y) ** 2).mean((1, 2, 3), dtype=np.float64)),
        "rgb_ssim32": ssim[:, :, 5:-5, 5:-5].mean((1, 2, 3), dtype=np.float64),
        "center_residual_cos": cosine(residual[:, :, center, center], truth[:, :, center, center]),
        "full_residual_cos": cosine(residual, truth),
        "coarse4_rgb_corr": cosine(pooled_p, pooled_y, center=True),
    }


def evaluate(arms, y, labels, rows, mean):
    values = {name: metrics(p, y, mean) for name, p in arms.items()}
    ids = np.unique(labels)
    index = np.stack([np.flatnonzero(labels == g) for g in ids])
    boot = np.random.default_rng(85031).integers(len(ids), size=(10000, len(ids)))
    paired = {}
    for name, base in (
        ("linear_rgb", "mean_rgb"),
        ("layout_rgb", "mean_rgb"),
        ("layout_rgb", "shuffled_rgb"),
        ("layout_rgb", "linear_rgb"),
        ("contrast_rgb", "layout_rgb"),
    ):
        item = {}
        for key in values[name]:
            d = (values[name][key] - values[base][key])[index].mean(1)
            item[key] = dict(
                delta=float(d.mean()), ci95=np.quantile(d[boot].mean(1), [0.025, 0.975]).tolist()
            )
        paired[name + " minus " + base] = item
    return {
        "means": {a: {k: float(v.mean()) for k, v in m.items()} for a, m in values.items()},
        "paired": paired,
        "rows": rows.tolist(),
        "concept_ids": labels.tolist(),
        "per_photo": {a: {k: v.tolist() for k, v in m.items()} for a, m in values.items()},
    }


def gallery(path, target, arms, rows, labels):
    from PIL import Image, ImageDraw

    cols = {
        "Original (32x32)": target,
        "T mean": arms["mean_rgb"],
        "Linear EEG": arms["linear_rgb"],
        "Multiscale EEG": arms["layout_rgb"],
        "Contrast EEG": arms["contrast_rgb"],
        "Shuffled EEG": arms["shuffled_rgb"],
    }
    examples = [i for label in np.unique(labels)[:2] for i in np.flatnonzero(labels == label)[:2]]
    tile, gap, left, top = 144, 6, 88, 30
    canvas = Image.new(
        "RGB", (left + len(cols) * (tile + gap), top + len(examples) * (tile + gap)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for c, (name, values) in enumerate(cols.items()):
        x = left + c * (tile + gap)
        draw.text((x, 8), name, fill="black")
        for r, i in enumerate(examples):
            y = top + r * (tile + gap)
            if c == 0:
                draw.text((4, y + 62), f"row {rows[i]}", fill="black")
            raw = np.rint(255 * values[i].transpose(1, 2, 0)).clip(0, 255).astype(np.uint8)
            canvas.paste(
                Image.fromarray(raw).resize(
                    (tile, tile), getattr(Image, "Resampling", Image).BILINEAR
                ),
                (x, y),
            )
    temp = path.with_suffix(".tmp")
    canvas.save(temp, format="JPEG", quality=94)
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dev-starts", type=Path)
    parser.add_argument("--n-confirm", type=int, default=20)
    args = parser.parse_args()
    import torch

    started = time.monotonic()
    root, out = args.archive, args.out
    out.mkdir(exist_ok=True, parents=True)
    records, concept, split, channels = registration(root)
    train = split["T"]
    evals = evaluation_rows(concept, split, args.dev_starts, args.n_confirm)
    eeg_path = root / "data/recovery_20261108_v236/sub-03_train_mean4.pt"
    if digest(eeg_path) != EEG_SHA:
        raise ValueError("Unexpected archived sub-03 EEG fingerprint")
    eeg = torch.load(eeg_path, map_location="cpu", weights_only=True, mmap=True).numpy()
    if eeg.shape != (16540, 63, 250):
        raise ValueError("Unexpected sub-03 mean-four EEG shape")
    x = eeg_design(eeg, train, channels)
    xe = eeg_design(eeg, np.concatenate(list(evals.values())), channels)
    del eeg
    archive = root / "assets/things_images_v254/training_images.zip"
    y = read_rgb(archive, records, train, "historical T")
    train_concepts = np.unique(concept[train])
    val_concepts = np.random.default_rng(CALIBRATION_SEED).permutation(train_concepts)[::5]
    val = np.isin(concept[train], val_concepts)
    print("Fitting train-only calibration and final RGB readouts", flush=True)
    mval, provisional = fit_predict(x[~val], y[~val], x[val], True)
    gain, contrast = calibrate(mval, provisional, y[val])
    mean, linear = fit_predict(x, y, xe, False)
    _, raw = fit_predict(x, y, xe, True)
    adjusted = apply_gains(raw, mean, gain)
    boosted = apply_gains(raw, mean, contrast)
    del y, x, xe
    offset, partitions = 0, {}
    for label, idx in evals.items():
        s = slice(offset, offset + len(idx))
        shuffle = shuffled_indices(concept[idx], 41017 if label == "dev" else 41018)
        arms = dict(
            mean_rgb=np.broadcast_to(mean, (len(idx), *mean.shape)),
            linear_rgb=np.clip(linear[s], 0, 1),
            layout_rgb=adjusted[s],
            contrast_rgb=boosted[s],
            shuffled_rgb=adjusted[s][shuffle],
        )
        if any(not np.isfinite(a).all() for a in arms.values()):
            raise ValueError("Nonfinite predicted RGB")
        partitions[label] = arms
        with (out / f"{label}_layout.tmp").open("wb") as stream:
            np.savez(stream, rows=idx, concept_ids=concept[idx], shuffle_index=shuffle, **arms)
        os.replace(out / f"{label}_layout.tmp", out / f"{label}_layout.npz")
        offset += len(idx)
    print("Both EEG-only prediction bundles saved; starting original-photo evaluation", flush=True)
    summary = {}
    for label, idx in evals.items():
        target = read_rgb(archive, records, idx, label)
        result = evaluate(partitions[label], target, concept[idx], idx, mean)
        gallery(out / f"{label}_layout_gallery.jpg", target, partitions[label], idx, concept[idx])
        save_json(out / f"{label}_fidelity.json", result)
        summary[label] = {k: result[k] for k in ("means", "paired")}
    save_json(
        out / "layout_result.json",
        dict(
            subject="sub-03",
            eeg="archived mean of four trials",
            eeg_sha256=EEG_SHA,
            script_sha256=digest(Path(__file__)),
            grid=GRID,
            inputs="posterior 17 channels, 250 samples, per-channel means plus early bins and log local power",
            target="original RGB, PIL Resize512 bilinear/CenterCrop512, then exact 16x16 area means",
            train_images=len(train),
            train_concepts=len(train_concepts),
            calibration_concepts=len(val_concepts),
            ridge_per_row=RIDGE,
            calibration_seed=CALIBRATION_SEED,
            gain=gain.tolist(),
            contrast_gain=contrast.tolist(),
            dev_seed=PILOT_SEED,
            confirm_seed=CONFIRM_SEED,
            confirm_pool="historical V excluding development concepts",
            splits={k: v.tolist() for k, v in evals.items()},
            arms=list(partitions["dev"]),
            units="NCHW float32 RGB [0,1]",
            paired_unit="concept; fixed participant; 10000 percentile bootstrap draws",
            center_residual_definition="Cosine after per-pixel T mean subtraction in fixed central half of image; zero prediction scores zero.",
            results=summary,
            seconds=round(time.monotonic() - started, 2),
        ),
    )
    print(json.dumps({k: r["means"] for k, r in summary.items()}), flush=True)


if __name__ == "__main__":
    main()
