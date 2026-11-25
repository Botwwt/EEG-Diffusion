"""Calibrate native H/14 angular spread only along EEG-predictable visual directions."""

import argparse
from pathlib import Path

import numpy as np
from fit_official_repeat_objects import LOCK, S, dump, fit, read, require, sha, training
from official_object_readout import unit

SEED = 260546
BANDS = ((0, 4), (4, 12), (12, 32))


def register(a):
    original = a.out / "design.json"
    require(sha(original) == LOCK, "Prior generic selection registry differs")
    base = read(original)
    held = np.array(base["held_generic_classes"])
    perm = np.random.default_rng(SEED).permutation(held)
    design = dict(
        subject=S,
        source_design_sha256=LOCK,
        source_visual_prediction_sha256=read(a.out / "inference_receipt.json")[
            "coordinates_sha256"
        ]["1012"],
        physical_training_classes=884,
        source_prior_visual_rank=32,
        angle_calibration_classes=np.sort(perm[:64]).tolist(),
        angle_validation_classes=np.sort(perm[64:]).tolist(),
        seed=SEED,
        bands=[list(k) for k in BANDS],
        provenance="The 128 generic objects were excluded from the 884-object model. The earlier word-readout rank was selected using these 128 generic objects before this angular experiment. The new gain fit and its 64-object test use disjoint classes. The earlier official 200 results were already known, and their labels or images never enter gain calibration or the gate.",
        transform="Use the spherical logarithm around the mean training image direction. Map CCA image covectors to native H14 loadings with their regularized training covariance, project orthogonal to the mean and QR in canonical rank order. For each band, winsorize norms at 95% on the 640 calibration photographs. Multiply angular coefficients by 1 + rho^2/(rho^2+0.04)*(clip(real/pred RMS ratio,1,8)-1), rho=max(0,uncentered paired cosine across all calibration samples/band coefficients). Scale the complementary angular residual by its robust paired least-squares slope constrained to [0,1]. Apply spherical exponential at the original mean, with angular magnitude capped at the calibration real-image 95th percentile.",
        controls="Native unit(mean image + learned 1024-dimensional residual); global theta gain median(true angle)/median(predicted angle), with same 95% angle cap; fixed classwise wrong-photo control rotates the ten generics by one within each class.",
        primary="For each held generic photograph, compare its raw H14 cosine to the predicted unit direction with each of the nine other real photographs in its own category; average nine binary comparisons, tie 0.5. Count single-condition top1 among these ten separately. Per-picture CSI observations are averaged once per generic class when resampling.",
        gate="Advance if proposed minus native mean within-class pairwise credit has a positive lower endpoint of a two-sided 95% paired class bootstrap (10000 draws, seed 260547), proposed point estimate exceeds the single global angle-gain comparator, and proposed within-ten top1 credit is at least native. No threshold or variant changed according to official outcomes.",
        export="Only after passing, refit the CCA/native ridge on all eligible 1012 generic objects. Apply the fixed rank-band gains, complementary slope and cap to R4/R80 native H14. Retain rows, words, donor and radius from the existing packet. No new target-class visual support.",
    )
    dst = a.out / "angles"
    dst.mkdir(exist_ok=True, parents=True)
    p = dst / "design.json"
    if p.exists():
        require(read(p) == design, "Incompatible angle study already locked")
    else:
        dump(p, design)
    return design


def frame(vectors, images):
    mean = vectors["mean"].astype(float)
    axis = mean / np.linalg.norm(mean)
    y = images.astype(float) - mean
    cov = (y.T @ y) / len(y)
    cov.flat[:: len(cov) + 1] += 0.25 * np.trace(cov) / len(cov)
    loadings = cov @ vectors["py"][:, :32]
    loadings -= np.outer(axis, axis @ loadings)
    q, r = np.linalg.qr(loadings)
    require(
        np.min(abs(r.diagonal())) > 1e-8 and np.max(abs(axis @ q)) < 1e-8,
        "Degenerate predictable visual subspace",
    )
    return axis, q


def tangent(v, axis):
    z = unit(np.asarray(v, dtype=float))
    cos = z @ axis
    t = z - cos[:, None] * axis
    n = np.linalg.norm(t, axis=1)
    return t * (np.arctan2(n, cos) / n.clip(1e-12))[:, None]


def expmap(t, axis, cap):
    norm = np.linalg.norm(t, axis=1)
    theta = np.minimum(norm, cap)
    return (np.cos(theta)[:, None] * axis + (np.sin(theta) / norm.clip(1e-12))[:, None] * t).astype(
        np.float32
    )


def robust(v):
    size = np.linalg.norm(v, axis=1)
    return v * np.minimum(1, np.quantile(size, 0.95) / size.clip(1e-12))[:, None]


def calibrate(predicted, real, axis, q):
    a, b = (tangent(v, axis) for v in (predicted, real))
    ac, bc = a @ q, b @ q
    gain = []
    stats = []
    for start, stop in BANDS:
        x, y = robust(ac[:, start:stop]), robust(bc[:, start:stop])
        sx, sy = np.linalg.norm(x), np.linalg.norm(y)
        rho = max(0, float(np.sum(x * y) / (sx * sy)))
        ratio = float(np.clip(sy / sx, 1, 8))
        w = rho**2 / (rho**2 + 0.04)
        gain.append(1 + w * (ratio - 1))
        stats.append(
            dict(
                ranks=[start + 1, stop],
                robust_real_to_predicted_rms=ratio,
                paired_correlation=rho,
                gain=gain[-1],
            )
        )
    x, y = robust(a - ac @ q.T), robust(b - bc @ q.T)
    return dict(
        band_gains=gain,
        details=stats,
        orthogonal_gain=float(np.clip(np.sum(x * y) / np.sum(x * x), 0, 1)),
        global_gain=float(
            np.median(np.linalg.norm(b, axis=1)) / np.median(np.linalg.norm(a, axis=1))
        ),
        real_angle_cap_radians=float(np.quantile(np.linalg.norm(b, axis=1), 0.95)),
    )


def conditions(predicted, frame, params):
    axis, q = frame
    t = tangent(predicted, axis)
    scores = t @ q
    changed = (t - scores @ q.T) * params["orthogonal_gain"]
    for (start, stop), gain in zip(BANDS, params["band_gains"], strict=True):
        changed += gain * scores[:, start:stop] @ q[:, start:stop].T
    cap = params["real_angle_cap_radians"]
    return {
        "native": unit(predicted),
        "global": expmap(t * params["global_gain"], axis, cap),
        "predictable": expmap(changed, axis, cap),
    }


def rows(predicted, real):
    y = real.reshape(-1, 10, 1024)
    x = predicted.reshape(y.shape)
    sims = np.einsum("cif,cjf->cij", x.astype(float), y.astype(float))
    truth = np.diagonal(sims, axis1=1, axis2=2)
    margin = truth[:, :, None] - sims
    credit = (margin > 1e-12) + 0.5 * (abs(margin) <= 1e-12)
    within = (credit.sum(2) - 0.5) / 9
    best = sims.max(2)
    top = (truth == best) / np.sum(sims == best[:, :, None], axis=2)
    return dict(
        within_pair=within,
        top1=top,
        raw_cosine=truth,
        wrong_same_class_cosine=sims[:, np.arange(10), (np.arange(10) + 1) % 10],
    )


def difference(x, y):
    d = (x - y).mean(1)
    j = np.random.default_rng(SEED + 1).integers(len(d), size=(10000, len(d)))
    return dict(delta=float(d.mean()), ci95=np.quantile(d[j].mean(1), [0.025, 0.975]).tolist())


def run(a):
    registered = register(a)
    dest = a.out / "angles"
    base = read(a.out / "design.json")
    data = training(a, base)
    ac, av = (
        np.array(registered[k]) for k in ("angle_calibration_classes", "angle_validation_classes")
    )
    held = np.r_[ac, av]
    fitting = np.setdiff1d(base["extended_classes"], held)
    require(len(fitting) == 884 and len(set(held)) == 128, "Generic class split overlaps")
    x = np.concatenate([data[0][c] for c in held])
    real = unit(np.concatenate([data[1][c] for c in held]))
    _, _, v = fit(fitting, data, x)
    training_images = unit(np.concatenate([data[1][c] for c in fitting]))
    axis, q = frame(v, training_images)
    raw = v["ridge"] + v["mean"]
    n = len(ac) * 10
    params = calibrate(raw[:n], real[:n], axis, q)
    locked = dest / "calibration.json"
    dump(locked, dict(design_sha256=sha(dest / "design.json"), parameters=params))
    use = conditions(raw[n:], (axis, q), params)
    scored = {k: rows(w, real[n:]) for k, w in use.items()}
    summary = {
        k: dict(
            within_class_pairwise_win=float(r["within_pair"].mean()),
            within_class_top1=float(r["top1"].sum()),
            raw_target_cosine=float(r["raw_cosine"].mean()),
            wrong_same_class_cosine=float(r["wrong_same_class_cosine"].mean()),
            median_angle_radians=float(np.median(np.linalg.norm(tangent(use[k], axis), axis=1))),
        )
        for k, r in scored.items()
    }
    contrasts = {
        k: {
            metric: difference(scored["predictable"][metric], scored[k][metric])
            for metric in ("within_pair", "top1", "raw_cosine")
        }
        for k in ("native", "global")
    }
    advance = bool(
        contrasts["native"]["within_pair"]["ci95"][0] > 0
        and contrasts["global"]["within_pair"]["delta"] > 0
        and summary["predictable"]["within_class_top1"] >= summary["native"]["within_class_top1"]
    )
    result = dict(
        n_photos=640,
        n_held_classes=64,
        eligible_distractors_per_query=9,
        generic_validation_real_median_angle_radians=float(
            np.median(np.linalg.norm(tangent(real[n:], axis), axis=1))
        ),
        design_sha256=sha(dest / "design.json"),
        calibration_sha256=sha(locked),
        source_sha256=sha(Path(__file__)),
        arms=summary,
        predictable_minus=contrasts,
        advance=advance,
    )
    dump(dest / "validation.json", result)
    print("Calibration", params, "validation", summary, "gate", advance, flush=True)
    if not advance:
        return
    final = np.asarray(base["extended_classes"])
    with np.load(a.out / S / "test_budgets.npz", allow_pickle=False) as p:
        test = p["mean_four_features"][[0, 4]].reshape(400, 1003)
    _, _, v = fit(final, data, test)
    final_frame = frame(v, unit(np.concatenate([data[1][c] for c in final])))
    native_path = a.out / S / "render_packet.npz"
    with np.load(native_path, allow_pickle=False) as z:
        original = dict(z)
    variants = conditions(v["mean"] + v["ridge"], final_frame, params)
    require(
        np.allclose(
            variants["native"], np.concatenate([original["r4"], original["r80"]]), atol=2e-6
        ),
        "Final model did not reproduce frozen native directions",
    )
    for j, b in enumerate(("r4", "r80")):
        original[b + "_global"] = variants["global"][j * 200 : (j + 1) * 200]
        original[b + "_predictable"] = variants["predictable"][j * 200 : (j + 1) * 200]
    out = dest / "render_angles.npz"
    np.savez(out, **original)
    dump(
        dest / "render_receipt.json",
        dict(
            packet_sha256=sha(out),
            validation_sha256=sha(dest / "validation.json"),
            native_packet_sha256=sha(native_path),
        ),
    )
    print("Render packet", out, sha(out), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("register", "validate"))
    for k, v in dict(
        out="outputs/official-repeat-object",
        source="outputs/official-object-readout/sub-03",
        expanded="outputs/photo-structure/expanded",
        extra="outputs/double-calibration",
        external="outputs/external-object-anchors/sub-03/external_anchors.npz",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    a = p.parse_args()
    (register if a.stage == "register" else run)(a)
