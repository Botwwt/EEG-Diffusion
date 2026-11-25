"""Calibrate the amplitude of within-photo regional updates on disjoint training objects."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fit_regional_h14 import (
    DESIGN,
    PUBLIC,
    crop_features,
    load_fields,
    relative_to_support,
    rotated,
    split,
)
from fit_repeat_photo_subspace import centered, ridge
from fit_within_photo_embedding import images, training
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import check

PLAN = PUBLIC / "regional_h14_calibration_design.json"


def eeg(a, ids):
    classes, features, _ = training(
        SimpleNamespace(expanded=a.data / "photo-structure/expanded"),
        "sub-04",
        512,
        images(a.data / "official-object-readout/sub-03"),
    )
    ix = {c: k for k, c in enumerate(classes)}
    return features[[ix[c] for c in ids]]


def fit(a, ids):
    receipts = read(a.root / "generic_regions.json")
    check(receipts["design_sha256"] == sha(DESIGN), "Unexpected regional cache")
    y = load_fields(a, ids, receipts["files"])
    x = eeg(a, ids)
    folds = np.array_split(np.random.default_rng(260561).permutation(len(ids)), 4)
    cross, norm = 0.0, 0.0
    for val in folds:
        train = np.setdiff1d(np.arange(len(ids)), val)
        pred = relative_to_support(ridge(x[train], centered(y[train]), x[val]))
        cross += np.sum(pred * relative_to_support(y[val]))
        norm += np.sum(pred**2)
    check(norm > 0, "Constant regional estimates")
    freeze(
        a.root / "gain.json",
        dict(
            design_sha256=sha(PLAN),
            source_sha256=sha(Path(__file__)),
            gain=float(np.clip(cross / norm, 0, 1)),
            train_objects=ids.tolist(),
            folds=[ids[f].tolist() for f in folds],
            numerator=float(cross),
            denominator=float(norm),
        ),
    )


def evaluate(a, train, test, gain_file):
    reg = read(gain_file)
    check(
        reg["source_sha256"] == sha(Path(__file__))
        and reg["design_sha256"] == sha(PLAN)
        and reg["train_objects"] == train.tolist(),
        "Calibration changed",
    )
    extra = read(a.root / "additional_regions.json")
    check(extra["gain_sha256"] == sha(gain_file), "Extraction preceded gain registration")
    ids = np.r_[train, test]
    files = {**read(a.root / "generic_regions.json")["files"], **extra["files"]}
    y = load_fields(a, ids, files)
    x = eeg(a, ids)
    n = len(train)
    target = relative_to_support(y[n:])
    pred = reg["gain"] * relative_to_support(ridge(x[:n], centered(y[:n]), x[n:]))
    fields = dict(
        matched=pred,
        mean=np.zeros_like(pred),
        rotated=rotated(pred),
        exchanged=pred.reshape(len(test), 5, 2, -1)[:, :, ::-1].reshape(pred.shape),
    )
    err = {k: np.square(v - target).sum(-1).mean(1) / 4 for k, v in fields.items()}
    draws = np.random.default_rng(260562).integers(len(test), size=(10000, len(test)))
    effects = {}
    for key in ("rotated", "mean", "exchanged"):
        d = err[key] - err["matched"]
        effects[key] = dict(
            mean=float(d.mean()), ci95=np.quantile(d[draws].mean(1), [0.025, 0.975]).tolist()
        )
    pairs = {}
    for name, v in fields.items():
        margin = ((v[:, ::2] - v[:, 1::2]) * (target[:, ::2] - target[:, 1::2])).sum(-1)
        pairs[name] = float((margin > 1e-12).sum() + 0.5 * (np.abs(margin) <= 1e-12).sum())
    freeze(
        PUBLIC / "regional_h14_calibration.json",
        dict(
            design_sha256=sha(PLAN),
            source_sha256=sha(Path(__file__)),
            gain_sha256=sha(gain_file),
            gain=reg["gain"],
            additional_cache_sha256=sha(a.root / "additional_regions.json"),
            train_objects=train.tolist(),
            additional_objects=test.tolist(),
            n_photos=len(test) * 10,
            n_pairs=len(test) * 5,
            mean_squared_field_error={k: float(v.mean()) for k, v in err.items()},
            control_error_minus_matched=effects,
            pair_credit=pairs,
            advance=all(effects[k]["ci95"][0] > 0 for k in ("rotated", "mean")),
        ),
    )
    print(reg["gain"], effects, pairs, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fit", "extract", "evaluate"))
    p.add_argument("--root", type=Path, default=Path("outputs/regional-h14"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    train, old = split(a)
    test = np.setdiff1d(
        read(a.data / "trial-object-encoder/generic/receipt.json")["test_classes"], old
    )
    check(len(test) == 64 and not np.intersect1d(train, test).size, "Wrong independent groups")
    gain_file = a.root / "gain.json"
    if a.stage == "fit":
        fit(a, train)
    elif a.stage == "extract":
        gain_sha = sha(gain_file)
        a.images = a.data / "photo-structure/training_images.zip"
        a.manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
        files = crop_features(a, test)
        freeze(a.root / "additional_regions.json", dict(gain_sha256=gain_sha, files=files))
    else:
        evaluate(a, train, test, gain_file)
