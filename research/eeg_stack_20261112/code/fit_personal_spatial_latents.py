"""Fit within-object EEG to continuous 8x8 VAE layout and freeze matched render starts."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from fit_personal_session_metric import load
from prepare_expanded_personal import extract
from prepare_personal_spatial_latents import FULL_SHA, check, read, save, sha
from render_within_photo_ip import packet
from scipy.linalg import eigh

SEED = 260512
RIDGES = (0.5, 2.0, 8.0)


def bank(a):
    root = a.out / "source"
    reg = read(root / "registration.json")
    complete = read(root / "completion.json")
    check(
        sha(root / "registration.json") == complete["registration_sha256"]
        and complete["sha256"] == FULL_SHA,
        "Wrong historical bank",
    )
    t = root / "historical_t8.npz"
    ext = root / "extension8.npz"
    receipt = read(root / "extension8.json")
    req = read(root / "extension_rows.json")
    check(
        sha(ext) == receipt["output_sha256"]
        and receipt["new_rows_sha256"] == sha(root / "extension_rows.json"),
        "Wrong source extension",
    )
    data = np.full((16540, 4, 8, 8), np.nan, np.float32)
    with np.load(t, allow_pickle=False) as z:
        ids = z["rows"]
        check(
            ids.tolist() == reg["array_rows"]
            and z["latents"].shape == (13230, 4, 8, 8)
            and np.isfinite(z["latents"]).all(),
            "Historical pooled mode row mismatch",
        )
        data[ids] = z["latents"]
    with np.load(ext, allow_pickle=False) as z:
        ids = z["rows"]
        check(
            ids.tolist() == req["rows"] + req["parity_rows"]
            and str(z["request_sha256"]) == sha(root / "extension_rows.json")
            and str(z["registration_sha256"]) == sha(root / "registration.json")
            and z["latents"].shape == (len(ids), 4, 8, 8)
            and np.isfinite(z["latents"]).all(),
            "Physical extension mismatch",
        )
        new = len(req["rows"])
        check(
            np.isnan(data[ids[:new]]).all()
            and np.allclose(data[ids[new:]], z["latents"][new:], rtol=2e-4, atol=2e-4),
            "Extension overlaps unknown or changed historical rows",
        )
        data[ids[:new]] = z["latents"][:new]
    return data, dict(
        historical_full_sha256=FULL_SHA, historical_pooled_sha256=sha(t), extension_sha256=sha(ext)
    )


def training(a, sub, data):
    root = a.expanded / "features"
    receipt = read(root / (sub + ".json"))
    classes = np.asarray(receipt["training_concepts"])
    x = []
    for c in classes:
        with np.load(root / sub / f"{c:04d}.npz", allow_pickle=False) as z:
            features = z["features"]
            check(
                int(z["start"]) == 10 * c
                and str(z["raw_sha"]) == receipt["raw_block_sha256"][str(c)]
                and hashlib.sha256(features.tobytes()).hexdigest() == str(z["feature_sha"]),
                "Training EEG mismatch",
            )
            x.append(features)
    x = np.stack(x)
    check(
        len(set(classes)) == 512
        and x.shape == (512, 10, 1003)
        and hashlib.sha256(x.tobytes()).hexdigest() == receipt["train_design_sha256"],
        "Wrong 512-concept training examples",
    )
    y = data[classes[:, None] * 10 + np.arange(10)].astype(np.float64)
    check(np.isfinite(y).all(), "Incomplete generic latent targets")
    # Removing each image's spatial DC per latent channel isolates within-picture layout.
    y -= y.mean((-2, -1), keepdims=True)
    return classes, x.astype(np.float64), y.reshape(512, 10, -1)


def model(x, y):
    xx = (x - x.mean(1, keepdims=True)).reshape(-1, x.shape[-1])
    yy = (y - y.mean(1, keepdims=True)).reshape(-1, y.shape[-1])
    scale = xx.std(0).clip(1e-6)
    xx /= scale
    values, axes = eigh(xx.T @ xx / len(xx), check_finite=False)
    return scale, values.clip(0), axes, axes.T @ (xx.T @ yy / len(xx))


def projection(params, q, ridge):
    scale, values, axes, cross = params
    # The intercept cancels analytically when removing the mean of the eight support EEGs.
    return ((q / scale) @ axes / (values + ridge)) @ cross


def relative(v):
    """Each class contributes five disjoint query pairs, each centered on its other eight."""
    out = np.empty_like(v)
    for pair in range(5):
        q = np.arange(10) // 2 == pair
        out[:, q] = v[:, q] - v[:, ~q].mean(1, keepdims=True)
    return out


def metrics(y, p):
    ref = relative(y)
    guess = relative(p)
    d = guess.reshape(len(guess), 5, 2, -1)
    t = ref.reshape(d.shape)
    margin = ((d[:, :, 0] - d[:, :, 1]) * (t[:, :, 0] - t[:, :, 1])).sum(-1)
    wins = (margin > 1e-12) + 0.5 * (abs(margin) <= 1e-12)
    n = float((guess * ref).sum())
    p2 = float(np.square(guess).sum())
    t2 = float(np.square(ref).sum())
    return dict(
        pairs=margin.size,
        pair_credit=float(wins.sum()),
        pair_rate=float(wins.mean()),
        r2_vs_support_prior=float((2 * n - p2) / t2),
        pooled_residual_cosine=float(n / np.sqrt(t2 * p2)),
        mse_optimal_gain=float(max(0, n / p2)),
        amplitude_matching_gain=float(np.sqrt(t2 / p2)),
        pair_concept_ci95=np.quantile(
            wins.mean(1)[np.random.default_rng(SEED).integers(len(y), size=(10000, len(y)))].mean(
                1
            ),
            [0.025, 0.975],
        ).tolist(),
    )


def calibration(a, data, source):
    sub = "sub-04"
    classes, x, y = training(a, sub, data)
    folds = np.array_split(np.random.default_rng(SEED).permutation(len(x)), 4)
    oof = {r: np.empty_like(y) for r in RIDGES}
    for held in folds:
        tr = np.setdiff1d(np.arange(len(x)), held)
        fitted = model(x[tr], y[tr])
        for r in RIDGES:
            oof[r][held] = projection(fitted, x[held], r)
    scores = [metrics(y, oof[r]) for r in RIDGES]
    best = int(np.argmax([s["r2_vs_support_prior"] for s in scores]))
    chosen = scores[best]
    # Use a partial dispersion correction for generation, bounded before panel images are evaluated.
    gain = float(min(2.5, np.sqrt(chosen["mse_optimal_gain"] * chosen["amplitude_matching_gain"])))
    result = dict(
        protocol="512 generic objects only. Four held-concept folds fixed on sub-04. Select ridge by unamplified pooled cross-validated residual MSE. The eight support EEGs and modes center each query. Remove each image's spatial DC per channel from the EEG term.",
        source=source,
        selection_participant=sub,
        classes=classes.tolist(),
        fold_concepts=[classes[f].tolist() for f in folds],
        ridges=RIDGES,
        validation=scores,
        ridge=RIDGES[best],
        gain=gain,
        gain_rule="min(2.5,sqrt(MSE-optimal nonnegative gain * measured/predicted RMS)); fixed using generic out-of-fold cases only",
        training_receipt_sha256=sha(a.expanded / "features" / (sub + ".json")),
        seed=SEED,
    )
    save(a.out / "selection.json", result)
    print(
        "Generic held-out classes, ridge/gain",
        result["ridge"],
        gain,
        "paired",
        chosen["pair_credit"],
        "/",
        chosen["pairs"],
        "R2",
        chosen["r2_vs_support_prior"],
        flush=True,
    )


def package(a, data, source):
    lock = read(a.out / "selection.json")
    check(
        lock["source"] == source
        and lock["selection_participant"] == "sub-04"
        and lock["training_receipt_sha256"] == sha(a.expanded / "features/sub-04.json"),
        "Changed training selection",
    )
    for sub in a.subjects:
        classes, x, y = training(a, sub, data)
        check(classes.tolist() == lock["classes"], "Generic cohort or order changed")
        params = model(x, y)
        for panel in a.panels:
            hroot = a.fresh if panel == "fresh" else a.angle
            paths = (
                a.fresh / "eeg" / (sub + ".npz")
                if panel == "fresh"
                else a.inputs / "halves" / (sub + ".npz")
            )
            source_rows, _, _, mean = load(paths)
            rows, folds, words, donors, support, _, _, hsha = packet(hroot, sub, panel)
            with np.load(
                hroot / "conditions" / sub / panel / "conditions_512.npz", allow_pickle=False
            ) as h:
                check(
                    str(h["target_eeg_sha256"]) == sha(paths), "H/14 uses a different physical EEG"
                )
            check(not np.intersect1d(classes, np.unique(rows // 10)).size, "Generic/target leakage")
            z = projection(params, extract(mean).astype(np.float64), lock["ridge"]).reshape(
                -1, 4, 8, 8
            )
            pos = {r: i for i, r in enumerate(source_rows)}
            out = np.empty((len(rows), 4, 8, 8), np.float32)
            uniform = np.empty_like(out)
            for f in range(5):
                ids = np.flatnonzero(folds == f)
                groups = support[ids]
                check(
                    not np.intersect1d(rows[ids], groups).size,
                    "Held-out target image in its own support pool",
                )
                cal = data[groups]
                check(np.isfinite(cal).all(), "A permitted support is missing its VAE target")
                uniform[ids] = cal.astype(np.float64).mean(1)
                corrected = z[[pos[r] for r in rows[ids]]] - z[
                    [[pos[r] for r in group] for group in groups]
                ].mean(1)
                out[ids] = lock["gain"] * (corrected - corrected.mean((-1, -2), keepdims=True))
            lookup = {r: i for i, r in enumerate(rows)}
            donor = np.array([lookup[r] for r in donors])
            check(
                np.array_equal(uniform, uniform[donor]) and np.max(abs(out.mean((-2, -1)))) < 1e-5,
                "Wrong donor support or nonzero DC",
            )
            dst = a.out / "conditions" / sub / panel / "spatial.npz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                dst,
                rows=rows,
                folds=folds,
                predicted_words=words,
                donor_rows=donors,
                support_rows=support,
                prior=uniform,
                residual=out,
                h14_packet_sha256=hsha,
                spatial_selection_sha256=sha(a.out / "selection.json"),
                eeg_sha256=sha(paths),
                subject=sub,
                panel=panel,
            )
            print("Frozen", sub, panel, sha(dst), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("all", "select", "package"))
    for k, d in dict(
        out="outputs/latent-spatial",
        expanded="outputs/photo-structure/expanded",
        fresh="outputs/fresh-photo",
        angle="outputs/photo-structure/residual_angle",
        inputs="outputs/object-transfer",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(d))
    p.add_argument(
        "--subjects",
        nargs="+",
        choices=("sub-04", "sub-05", "sub-01"),
        default=["sub-04", "sub-05", "sub-01"],
    )
    p.add_argument(
        "--panels",
        nargs="+",
        choices=("registered", "replicate", "fresh"),
        default=["registered", "replicate", "fresh"],
    )
    a = p.parse_args()
    images, meta = bank(a)
    if a.stage in ("all", "select"):
        calibration(a, images, meta)
    if a.stage in ("all", "package"):
        package(a, images, meta)
