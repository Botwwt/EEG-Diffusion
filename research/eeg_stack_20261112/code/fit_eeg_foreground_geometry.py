"""Predict an individual photo's foreground location and extent using four EEG trials."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from decode_eeg_layout import CHANNELS
from measure_foreground_geometry import MODEL_SHA, ZIP_SHA
from scipy.linalg import eigh, solve
from scipy.optimize import nnls

TEACHERS = (1, 2, 6, 7, 8, 9, 10)
RIDGE = 0.5
RANK = 48
GENERIC_FOLDS = 4


def design(path, rows, mean=True):
    with np.load(path, allow_pickle=False) as d:
        if (
            not np.array_equal(d["rows"], rows)
            or d["channels"].tolist() != CHANNELS
            or str(d["subject"]) != path.stem
            or not np.allclose(d["times"], np.arange(250) / 250, rtol=0, atol=1e-7)
        ):
            raise ValueError("EEG participant, electrode, time or image registration differs")
        x = d["mean_eeg"] if mean else d["half_eeg"].mean(0)
        if x.shape != (len(rows), 17, 250) or not np.isfinite(x).all():
            raise ValueError("Expected finite four-repeat posterior EEG")
        # 40-520 ms in 20 ms signed ERP bins.
        return (
            x[:, :, 10:130]
            .reshape(len(rows), 17, 24, 5)
            .mean(-1, dtype=np.float64)
            .reshape(len(rows), -1)
        )


def centered(x):
    a = x.reshape(-1, 10, x.shape[-1])
    return (a - a.mean(1, keepdims=True)).reshape(x.shape)


def fit(x, teacher, y):
    x, z, y = centered(x), centered(teacher), centered(y)
    scale = x.std(0).clip(1e-8)
    x = x / scale
    z = z / z.std(0).clip(1e-8)
    n = len(y)
    cx, cz, cross = x.T @ x / n, z.T @ z / n, x.T @ z / n
    cx.flat[:: len(cx) + 1] += RIDGE
    cz.flat[:: len(cz) + 1] += RIDGE
    xy = x.T @ y / n
    direct = solve(cx, xy, assume_a="pos")
    stable = cross @ solve(cz, cross.T, assume_a="pos")
    _, v = eigh(stable, cx, subset_by_index=(len(cx) - RANK, len(cx) - 1))
    shared = v @ (v.T @ xy)
    return np.stack((direct / scale[:, None], shared / scale[:, None]))


def unseen_differences(x):
    groups = x.reshape(-1, 10, x.shape[-1])
    out = np.empty_like(groups)
    for fold in range(5):
        test = np.arange(10) // 2 == fold
        out[:, test] = groups[:, test] - groups[:, ~test].mean(1, keepdims=True)
    return out.reshape(x.shape)


def fit_generic(x, z, y):
    order = np.random.default_rng(260501).permutation(128)
    prediction = np.empty((2, *y.shape))
    for hold in np.array_split(order, GENERIC_FOLDS):
        test = np.isin(np.arange(len(x)) // 10, hold)
        w = fit(x[~test], z[~test], y[~test])
        prediction[:, test] = unseen_differences(x[test]) @ w
    target = unseen_differences(y)
    scale = centered(y).std(0).clip(1e-8)
    predicted = (prediction / scale).reshape(2, -1).T
    actual = (target / scale).ravel()
    coefficients, _ = nnls(predicted, actual)
    w = fit(x, z, y)
    return (
        np.tensordot(coefficients, w, axes=(0, 0)),
        scale,
        dict(
            nonnegative_weights=coefficients.tolist(),
            generic_oof_mse_zero=float(np.mean(actual**2)),
            generic_oof_mse_fitted_gain=float(np.mean((actual - predicted @ coefficients) ** 2)),
            generic_targets=1280,
        ),
    )


def interval(delta, truth):
    ids = np.unique(truth)
    if not len(delta):
        return {"n": 0, "mean": None, "ci95": None}
    totals = np.array([delta[truth == i].sum() for i in ids])
    sizes = np.array([(truth == i).sum() for i in ids])
    b = np.random.default_rng(260502).integers(len(ids), size=(10000, len(ids)))
    ci = np.quantile(totals[b].sum(1) / sizes[b].sum(1), [0.025, 0.975]) if len(ids) > 1 else None
    return dict(n=len(delta), mean=float(delta.mean()), ci95=None if ci is None else ci.tolist())


def panel(root, panel_path, subject, tx, trows, geom, lookup, w, scale, out):
    reg = json.loads(panel_path.read_text())
    ordered = np.array(reg["rows"])
    pred = json.loads(
        (
            root / "personal" / panel_path.name.split("_ablation")[0] / subject / "evaluation.json"
        ).read_text()
    )
    if any(reg[k] != pred[k] for k in ("rows", "words", "folds", "truth")):
        raise ValueError("Personal words and panel registration differ")
    words = np.asarray(pred["results"]["personal_r4"]["predicted"])
    folds, truth = map(np.array, (reg["folds"], reg["truth"]))
    pos = {r: i for i, r in enumerate(trows)}
    prior, signal = np.empty((len(ordered), 4)), np.empty((len(ordered), 4))
    for f, groups in enumerate(reg["calibration_rows_by_fold_and_word"]):
        query = ordered[folds == f]
        if (
            len(query) != 24
            or set(query) & set(np.ravel(groups))
            or set(np.ravel(groups)) != set(ordered) - set(query)
        ):
            raise ValueError("Query photo entered its own support fold")
        for i in np.where(folds == f)[0]:
            supports = groups[words[i]]
            if len(set(supports)) != 8:
                raise ValueError("Expected eight supports")
            prior[i] = geom[[lookup[r] for r in supports]].mean(0)
            signal[i] = (tx[pos[ordered[i]]] - tx[[pos[r] for r in supports]].mean(0)) @ w
    donor = np.arange(len(ordered))
    for f in np.unique(folds):
        for word in np.unique(words):
            idx = np.where((folds == f) & (words == word))[0]
            donor[idx] = np.roll(idx, 1)
    out.mkdir(exist_ok=True, parents=True)
    estimates = out / "geometry.npz"
    np.savez(
        estimates,
        rows=ordered,
        prior=prior,
        query=prior + signal,
        swapped=prior + signal[donor],
        donor_rows=ordered[donor],
    )
    # The target now enters scoring. The same image may be an allowed support in a different fold.
    actual = geom[[lookup[r] for r in ordered]]
    arms = {
        k: np.mean(((v - actual) / scale) ** 2, axis=1)
        for k, v in (("mean", prior), ("query", prior + signal), ("donor", prior + signal[donor]))
    }
    eligible = (truth == truth[donor]) & (donor != np.arange(len(donor)))
    subsets = {
        "all": np.ones(len(ordered), bool),
        "correct_word": words == truth,
        "same_truth_word_fold_donor": eligible,
    }
    return dict(
        prediction_sha256=hashlib.sha256(estimates.read_bytes()).hexdigest(),
        scores={
            name: dict(
                mean_mse={
                    a: float(arr[mask].mean()) if mask.any() else None for a, arr in arms.items()
                },
                against_mean=interval((arms["query"] - arms["mean"])[mask], truth[mask]),
                against_donor=interval((arms["query"] - arms["donor"])[mask], truth[mask]),
            )
            for name, mask in subsets.items()
        },
    )


def run(a):
    with np.load(a.geometry, allow_pickle=False) as d:
        rows, geometry = d["rows"], d["geometry"]
        if (
            str(d["model_sha256"]) != MODEL_SHA
            or str(d["images_sha256"]) != ZIP_SHA
            or geometry.shape != (len(rows), 4)
            or not np.isfinite(geometry).all()
        ):
            raise ValueError("Invalid foreground geometry or input provenance")
    lookup = {r: i for i, r in enumerate(rows)}
    with np.load(a.inputs / "generic/sub-04.npz", allow_pickle=False) as d:
        gr = d["rows"]
    with np.load(a.inputs / "halves/sub-04.npz", allow_pickle=False) as d:
        tr = d["rows"]
    if (
        len(gr) != 1280
        or len(tr) != 240
        or np.intersect1d(gr // 10, tr // 10).size
        or not np.array_equal(rows, np.union1d(gr, tr))
    ):
        raise ValueError("Unexpected training or evaluation row bank")
    if not np.all(gr.reshape(128, 10) == 10 * (gr[::10, None] // 10) + np.arange(10)):
        raise ValueError("Training rows must group ten distinct photos per object")
    teachers = []
    for person in TEACHERS:
        v = centered(design(a.inputs / "generic" / f"sub-{person:02d}.npz", gr))
        teachers.append(v / v.std(0).clip(1e-8))
    teacher = np.mean(teachers, 0)
    report = {}
    for s in a.subjects:
        x = design(a.inputs / "generic" / (s + ".npz"), gr)
        w, scale, fit_score = fit_generic(x, teacher, geometry[[lookup[r] for r in gr]])
        test = design(a.inputs / "halves" / (s + ".npz"), tr, mean=False)
        report[s] = dict(generic=fit_score, panels={})
        for name in ("registered", "replicate"):
            report[s]["panels"][name] = panel(
                a.inputs,
                a.references / (name + "_ablation.json"),
                s,
                test,
                tr,
                geometry,
                lookup,
                w,
                scale,
                a.out / s / name,
            )
    dest = a.out / "foreground_prediction.json"
    dest.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "references", "geometry", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--subjects", nargs="+", choices=("sub-04", "sub-05"), default=["sub-04", "sub-05"]
    )
    run(p.parse_args())
