"""Pair all 262,144 pixels of frozen personal12 pose renders with the registered photograph."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fit_spatial_photo import save
from score_personal12_pose import (
    ARMS,
    DESIGN,
    LOCK,
    PEOPLE,
    check,
    edited_settings,
    image_list,
    read,
    reference_list,
    setup,
)
from score_within_photo_dino import reference, sha

NAMES = ("source", *ARMS)
KEYS = ("rgb_rmse512", "luma_pearson512")


def channels(im):
    rgb = np.asarray(im, dtype=np.float32) / 255
    check(rgb.shape == (512, 512, 3), "Expected registered 512x512 RGB")
    luma = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    return rgb, luma - luma.mean(dtype=np.float64)


def score(actual, truth):
    a, al = channels(actual)
    b, bl = truth
    rgb = float(np.sqrt(np.mean((a - b) ** 2, dtype=np.float64)))
    den = np.sqrt(np.sum(al * al, dtype=np.float64) * np.sum(bl * bl, dtype=np.float64))
    corr = float(np.sum(al * bl, dtype=np.float64) / den) if den > 0 else 0.0
    return rgb, corr


def summarize(values, labels):
    cats, indices = np.unique(labels, return_inverse=True)
    result = dict(
        n=len(labels),
        distinct_objects=len(cats),
        means={k: dict(zip(KEYS, v.mean(0).tolist(), strict=True)) for k, v in values.items()},
    )
    one = np.bincount(indices, minlength=len(cats))
    rng = np.random.default_rng(260556)
    draws = rng.integers(len(cats), size=(10000, len(cats)))
    denom = one[draws].sum(1)
    paired = {}
    for left, right in (
        ("own", "prior"),
        ("own", "wrong"),
        ("own", "neutral"),
        ("own", "source"),
        ("prior", "neutral"),
    ):
        delta = (values[left] - values[right]) * np.array([-1, 1])
        sums = np.zeros((len(cats), 2))
        np.add.at(sums, indices, delta)
        ci = np.quantile(sums[draws].sum(1) / denom[:, None], [0.025, 0.975], axis=0)
        paired[f"{left}_over_{right}"] = {
            k: dict(favorable_delta=float(delta[:, j].mean()), ci95=ci[:, j].tolist())
            for j, k in enumerate(KEYS)
        }
    return result | dict(paired=paired)


def run(a):
    check(sha(DESIGN) == LOCK, "Independent registration mismatch")
    seal = a.renders / "blind_images.json"
    expected = read(seal)["cases"]
    check(
        set(expected) == set(PEOPLE)
        and all(
            sha(a.renders / "personal12" / s / "settings.json") == v for s, v in expected.items()
        ),
        "All three image sets must be sealed before opening true photographs",
    )
    a.originals = a.data / "photo-structure/training_images.zip"
    a.manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    table = reference(a)
    all_values = {k: [] for k in NAMES}
    all_labels = []
    all_rows = []
    individual = {}
    with ZipFile(a.originals) as z:
        for sub in PEOPLE:
            reg, source, orig, _ = setup(a, sub, read(DESIGN))
            folder, cfg = edited_settings(a, sub, orig)
            imgs, _ = reference_list(a, table, z, sub, reg)
            truths = [channels(im) for im in imgs]
            del imgs
            values = {}
            for arm in NAMES:
                f, k, m = (source, "own", orig) if arm == "source" else (folder, arm, cfg)
                values[arm] = np.array(
                    [
                        score(im, target)
                        for im, target in zip(
                            image_list(f, k, m["jpeg_sha256"][k]), truths, strict=True
                        )
                    ]
                )
                all_values[arm].append(values[arm])
            individual[sub] = summarize(values, reg["labels"])
            all_labels.extend(reg["labels"])
            all_rows.extend(reg["rows"])
            d = a.out / sub
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / "pixel_rows.npz", rows=reg["rows"], metrics=KEYS, **values)
    report = dict(
        evaluation_design_sha256=LOCK,
        image_seal_sha256=sha(seal),
        scorer_sha256=sha(Path(__file__)),
        original_zip_sha256=sha(a.originals),
        row_map_sha256=sha(a.manifest),
        distinct_photographs=len(set(all_rows)),
        pixels_per_image=512 * 512,
        methods="Secondary full-resolution pixel audit. Score the registered square crop in sRGB [0,1] after reopening saved JPEGs. Pearson uses 0.299 R + 0.587 G + 0.114 B; zero variance receives zero correlation. Each participant-photo has equal weight. Positive paired effects favor the named first arm. Resample the same true category across people, 10000 paired percentile draws, seed 260556.",
        individual=individual,
        pooled=summarize({k: np.concatenate(v) for k, v in all_values.items()}, all_labels),
    )
    save(a.out / "pixel512.json", report)
    if a.publish:
        save(DESIGN.parent / "personal12_pose_pixel512.json", report)
    print(report["pooled"]["means"], report["pooled"]["paired"]["own_over_prior"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--renders", type=Path, default=Path("outputs/personal-pose/renders"))
    p.add_argument("--out", type=Path, default=Path("outputs/personal-pose/evaluation"))
    p.add_argument("--publish", action="store_true")
    run(p.parse_args())
