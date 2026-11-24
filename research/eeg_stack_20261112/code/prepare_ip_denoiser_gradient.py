"""Freeze sub-04 generic pairs and cross-fit the established H14 base for a denoiser-gradient pilot."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from calibrate_photo_angles import rotate
from fit_support_structure import H14_SHA, unit
from fit_within_photo_embedding import images, training
from prepare_personal_spatial_latents import FULL_SHA
from render_within_photo_ip import ADAPTER_SHA, check, read, sha
from scipy.linalg import cho_factor, cho_solve

SEED = 260523
TURBO = "e869ac7d6942cb327d68d5ed83a40447aadf20e0c3358d98b2cc9e270db0da26"


def write(path, data):
    raw = (json.dumps(data, allow_nan=False, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    check(not path.exists() or path.read_bytes() == raw, "Conflicting locked record")
    path.write_bytes(raw)


def base(x, y, dx):
    a = (x - x.mean(1, keepdims=True)).reshape(-1, x.shape[-1])
    b = (y - y.mean(1, keepdims=True)).reshape(-1, y.shape[-1])
    sd = a.std(0).clip(1e-6)
    a /= sd
    gram = a.T @ a
    gram.flat[:: len(sd) + 1] += 2.0 * len(a)
    return (dx / sd) @ cho_solve(cho_factor(gram, check_finite=False), a.T @ b, check_finite=False)


def run(a):
    feat = a.expanded / "features/sub-04.json"
    m = read(feat)
    c = np.asarray(m["training_concepts"])
    old = np.asarray(m["original_concepts"])
    mapping = a.source_latents / "registration.json"
    done = read(a.source_latents / "completion.json")
    t = read(mapping)
    original_rows = np.asarray(t["array_rows"])
    check(
        done["sha256"] == FULL_SHA
        and done["registration_sha256"] == sha(mapping)
        and len(c) == 512
        and t["VAE_scaling_factor"] == 0.13025
        and np.array_equal(np.sort(original_rows) % 10, np.tile(np.arange(10), 1323)),
        "Wrong historical source",
    )
    historical = np.intersect1d(c, np.unique(original_rows // 10))
    scale_file = a.angle / "scale.json"
    scale = read(scale_file)
    rr = read(a.angle / "radius.json")
    sel = a.within / "selection.json"
    oldhp = read(sel)
    check(
        oldhp["selected"] == [2.0, 0.0] and scale["selection_sha256"] == sha(sel),
        "Wrong fixed baseline",
    )
    possible = np.setdiff1d(historical, np.r_[old, np.asarray(rr["generic_rows"]) // 10])
    g = np.random.default_rng(SEED)
    valid = np.sort(g.choice(possible, 48, replace=False))
    train = np.sort(g.choice(np.setdiff1d(historical, valid), 128, replace=False))
    chosen = np.r_[train, valid]
    a.out.mkdir(parents=True, exist_ok=True)
    ordering = np.stack([g.permutation(np.arange(10)) for _ in chosen])
    seed_list = g.integers(0, 1 << 30, size=len(chosen))
    steps = np.r_[
        g.permutation(np.repeat([499, 749], 64)), g.permutation(np.repeat([499, 749], 24))
    ]
    plan = dict(
        seed=SEED,
        subject="sub-04",
        scope="Generic concepts only. Shared known generic class prompt and eight-photo calibration prior isolate within-class sensitivity. No target panel input. Not object decoding or full-image reconstruction.",
        historical_generic_classes=historical.tolist(),
        train_classes=train.tolist(),
        eval_classes=valid.tolist(),
        pair_rows=(10 * chosen[:, None] + ordering[:, :2]).tolist(),
        support_rows=(10 * chosen[:, None] + ordering[:, 2:]).tolist(),
        noise_seeds=seed_list.tolist(),
        timesteps=steps.tolist(),
        separation="Reserve 48 generic classes outside the original 128 hyperparameter-training classes and all visual-radius classes. Exclude them from every EEG-to-H14 refit. Cross-fit the 128 adapter-training classes in 4 object folds, removing the active fold from the remaining 464 base classes; the evaluation base fits all 464. One class, one pixel-independent random held-out pair. Keep the existing ridge=2, whitening=0, 3.5244 calibrated gain/cap and native IP-Adapter radius.",
        adapter="At the common eight-support H14 direction, differentiate MSE_A - MSE_B through a frozen UNet using the same Gaussian field for both true VAE modes. Negate, project onto the unit-sphere tangent and normalize. Fit signed (own_A_unit_tangent - own_B_unit_tangent) to that target with rank-16 training SVD and ridge 0.25 times the mean retained Gram eigenvalue. Mix each individual's original unit tangent and the learned unit tangent with coefficients 1 and 0.5, then renormalize at exactly the original EEG-predicted angle. The sham head reverses exactly 64 of the 128 training signs using seed 260524. No grid search.",
        scoring="For 48 untouched generic classes, evaluate paired raw epsilon MSE on true 64x64 modes against common diffusion noise using native, learned and sign-sham matched vs exchanged EEG. An evaluation-only true H14 tangent oracle uses the original EEG angular magnitudes. The uniform center ties by construction. Report paired mean CE advantage (swapped minus matched), absolute matched MSE, sign credit and 95% class bootstrap with 10000 draws seed 260525.",
        advance="Propose further image generation only if both paired mean improvements in denoising contrast (learned minus native and learned minus sham) have 95% lower bound >0, learned mean matched MSE is no worse than native, and the evaluation-only true-H14 tangent oracle has positive mean contrast. Otherwise stop this candidate. No target outcomes.",
        h14_sha256=H14_SHA,
        t_sha256=FULL_SHA,
        registration_sha256=sha(mapping),
        feature_receipt_sha256=sha(feat),
        selection_sha256=sha(sel),
        angular_scale_sha256=sha(scale_file),
        radius=float(rr["median_norm"]),
        ip_adapter_sha256=ADAPTER_SHA,
        turbo_sha256=TURBO,
    )
    check(
        len(historical) == 415
        and not np.intersect1d(valid, np.r_[old, np.asarray(rr["generic_rows"]) // 10, train]).size,
        "Bad adaptation split",
    )
    write(a.out / "design.json", plan)
    bank = images(a.source)
    classes, x, y = training(a, "sub-04", 512, bank)
    check(np.array_equal(classes, c), "EEG class order changed")
    ix = {v: i for i, v in enumerate(c)}
    index = np.asarray([ix[v] for v in chosen])
    b = np.arange(len(chosen))[:, None]
    xx = x[index][b, ordering]
    yy = y[index][b, ordering]
    delta = xx[:, :2] - xx[:, 2:].mean(1, keepdims=True)
    prior = unit(yy[:, 2:].mean(1))
    actual = yy[:, :2]
    base_own = np.empty_like(actual)
    tr_idx = np.arange(128)
    assigned = np.array_split(np.random.default_rng(SEED + 3).permutation(tr_idx), 4)
    for loc in [*assigned, np.arange(128, 176)]:
        use = ~np.isin(c, np.r_[valid, chosen[loc]])
        check(
            not np.intersect1d(c[use], chosen[loc]).size and not np.intersect1d(valid, c[use]).size,
            "Cross-fit leakage",
        )
        residual = base(x[use], y[use], delta[loc])
        native = unit(yy[loc, 2:].mean(1, keepdims=True) + residual)
        base_own[loc] = rotate(
            np.broadcast_to(prior[loc, None], native.shape).reshape(-1, 1024),
            native.reshape(-1, 1024),
            scale,
        ).reshape(native.shape)
    with (a.source / "atm_train_rows.csv").open(encoding="utf8", newline="") as f:
        words = [r["unique_id"] for r in csv.DictReader(f)]
    with (a.source / "dictionary.csv").open(encoding="utf-8-sig", newline="") as f:
        common = {r["unique_id"]: r["word"] for r in csv.DictReader(f)}
    file = a.out / "generic_pairs.npz"
    # Actual directions available solely to the evaluation-only capability check.
    np.savez(
        file,
        classes=chosen,
        prior=prior.astype(np.float32),
        own=base_own.astype(np.float32),
        oracle_eval_only=actual[128:].astype(np.float32),
        words=np.asarray([common[words[10 * v]] for v in chosen]),
        rows=10 * chosen[:, None] + ordering[:, :2],
        seeds=seed_list,
        t=steps,
    )
    write(
        a.out / "inputs.json",
        dict(design_sha256=sha(a.out / "design.json"), packet_sha256=sha(file)),
    )
    print("cross-fitted", len(train), len(valid), sha(file), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in dict(
        out="outputs/ip-denoiser-gradient",
        expanded="outputs/photo-structure/expanded",
        source="outputs/official-object-readout/sub-03",
        angle="outputs/photo-structure/residual_angle",
        within="outputs/photo-structure/within_photo",
        source_latents="outputs/latent-spatial/source",
    ).items():
        p.add_argument("--" + k.replace("_", "-"), type=Path, default=Path(v))
    run(p.parse_args())
