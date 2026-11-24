"""Measure individual residual cosine and restore generic within-object angles for generation."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from fit_within_photo_embedding import (
    bases,
    images,
    predict,
    read,
    sha,
    training,
    uncertainty,
    unit,
)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data, allow_nan=False, indent=2) + "\n").encode())


def angles(x, center):
    return np.arccos(np.sum(x * center, axis=-1).clip(-1, 1))


def calibrate(a):
    config = read(a.within / "selection.json")
    classes, x, y = training(a, "sub-04", 128, images(a.source))
    measured, estimated = [], []
    for folds in config["generic_class_folds"]:
        test = np.isin(classes, folds)
        basis = bases(x[~test], y[~test], x[test])
        features = predict(basis, config["selected"]) @ basis[2].T
        for k in range(0, 10, 2):
            hold = np.isin(np.arange(10), [k, k + 1])
            mean = y[test][:, ~hold].mean(1, keepdims=True)
            center = unit(mean)
            preds = unit(mean + features[:, hold] - features[:, ~hold].mean(1, keepdims=True))
            measured.extend(angles(y[test][:, hold], center).ravel())
            estimated.extend(angles(preds, center).ravel())
    if len(measured) != 1280 or min(estimated) <= 0:
        raise ValueError("Expected one nondegenerate held-out angle per generic photograph")
    observed = float(np.median(measured))
    fitted = float(np.median(estimated))
    write(
        a.out / "scale.json",
        dict(
            n=len(estimated),
            generic_observed_median=observed,
            eeg_predicted_median=fitted,
            gain=observed / fitted,
            max_angle=np.quantile(measured, 0.9),
            selection_sha256=sha(a.within / "selection.json"),
            units="radians",
        ),
    )


def rotate(prior, matched, scale):
    prior, matched = prior.astype(float), matched.astype(float)
    cosine = np.sum(prior * matched, axis=-1, keepdims=True).clip(-1, 1)
    tangent = matched - cosine * prior
    theta = np.minimum(
        scale["gain"] * np.arctan2(np.linalg.norm(tangent, axis=-1, keepdims=True), cosine),
        scale["max_angle"],
    )
    return unit(np.cos(theta) * prior + np.sin(theta) * unit(tangent)).astype(np.float32)


def package(a):
    scale = read(a.out / "scale.json")
    source = read(a.base / "design.json")
    mapping = {}
    for s in a.subjects:
        for p in source["panels"]:
            rel = Path(s) / p / "conditions_512.npz"
            original = a.within / rel
            with np.load(original, allow_pickle=False) as z:
                data = dict(z)
            pos = {r: i for i, r in enumerate(data["rows"])}
            ix = [pos[r] for r in data["donor_rows"]]
            if not np.array_equal(data["uniform"], data["uniform"][ix]):
                raise ValueError("Donors must use an identical support prior")
            own = rotate(data["uniform"], data["own"], scale)
            dest = a.out / "conditions" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dest, **(data | dict(own=own, wrong=own[ix], source_sha256=sha(original))))
            mapping[f"{s}/{p}"] = sha(dest)
    write(
        a.out / "design.json",
        source
        | dict(
            packets_sha256=mapping,
            cohort=a.subjects,
            original_design_sha256=sha(a.base / "design.json"),
            angular_calibration=scale,
            arms={
                "own": "Generic-calibrated angle from support prior along current EEG direction",
                "uniform": "Original eight-support prior",
                "wrong": "Same fold/word donor direction",
            },
        ),
    )
    shutil.copyfile(a.base / "radius.json", a.out / "radius.json")


def score(a):
    bank = images(a.source)
    report = {}
    for s in a.subjects:
        entries = []
        for p in ("registered", "replicate"):
            reg = read(a.references / (p + "_ablation.json"))
            truth = np.array(reg["words"])[reg["truth"]]
            with np.load(a.within / s / p / "conditions_512.npz", allow_pickle=False) as z:
                rows, folds, words = (z[k] for k in ("rows", "folds", "predicted_words"))
                if rows.tolist() != reg["rows"] or folds.tolist() != reg["folds"]:
                    raise ValueError("Scoring registration mismatch")
                for w in np.unique(truth):
                    for f in range(5):
                        i = np.flatnonzero((truth == w) & (folds == f) & (words == w))
                        if len(i) != 2:
                            continue
                        prior = z["uniform"][i].astype(float)
                        gallery = unit(bank[rows[i]].astype(float))
                        if not np.array_equal(prior[0], prior[1]):
                            raise ValueError("The two photos must have the same calibration images")
                        targets = unit(gallery - prior)
                        entry = {"panel": p, "word": w}
                        for arm in ("own", "wrong", "uniform"):
                            delta = unit(z[arm][i].astype(float) - prior) @ targets.T
                            margin = np.diag(delta) - np.diag(delta[:, ::-1])
                            entry[arm] = float(
                                np.mean((margin > 1e-12) + 0.5 * (abs(margin) <= 1e-12))
                            )
                        entries.append(entry)
        labels = [r["panel"] + "/" + r["word"] for r in entries]
        report[s] = {}
        for arm in ("own", "wrong", "uniform"):
            v = np.array([r[arm] for r in entries])
            q = uncertainty(v - 0.5, labels)
            report[s][arm] = dict(
                n_queries=2 * len(v),
                credit=float(2 * v.sum()),
                rate=v.mean(),
                ci95=(0.5 + np.array(q["ci95"])).tolist(),
                p=q["cluster_signflip_one_sided_p"],
            )
    write(a.out / "single_query.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("calibrate", "package", "score"))
    defaults = {
        "source": "outputs/official-object-readout/sub-03",
        "references": "outputs/object-fidelity",
        "expanded": "outputs/photo-structure/expanded",
        "within": "outputs/photo-structure/within_photo",
        "out": "outputs/photo-structure/residual_angle",
        "base": "outputs/photo-structure/h14_residual_render",
    }
    for name, path in defaults.items():
        p.add_argument("--" + name, type=Path, default=Path(path))
    p.add_argument("--subjects", nargs="+", default=["sub-04", "sub-05", "sub-01"])
    args = p.parse_args()
    {"calibrate": calibrate, "package": package, "score": score}[args.stage](args)
