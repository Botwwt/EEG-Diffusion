"""Evaluate a repeat-noise-calibrated VAE-to-EEG observation operator on held-out objects."""

import argparse
from pathlib import Path

import numpy as np
from register_fresh_photo_panel import freeze, read, sha
from run_prospective_readout import frozen_npz
from scipy.linalg import eigh, solve, svd

DESIGN = (
    Path(__file__).resolve().parents[1]
    / "results/object_fidelity/vae_forward_measurement_design.json"
)
LOCK = "5ff2ad75dacb8e438915afcffed8cb7f34d52ed825fef727a7cee3d08475059f"
N_PHOTOS = 10
SUPPORT = np.stack(
    [np.array([j for j in range(N_PHOTOS) if j // 2 != i // 2]) for i in range(N_PHOTOS)]
)
METHODS = (
    "start",
    "matched",
    "mean",
    "wrong",
    "candidate_blind",
    "inverse",
    "inverse_mean",
    "oracle35",
    "oracle",
)


def require(x, message):
    if not x:
        raise ValueError(message)


def paths(d):
    base = d / "latent-spatial/source"
    return dict(
        stored_repeats=d / "object-transfer/halves/sub-04.npz",
        physical_trials=d / "trial-object-inputs/sub-04/generic512.json",
        generic_split=d / "episodic-object/sub-04/model.npz",
        historical_latents=base / "historical_t8.npz",
        extension_latents=base / "extension8.npz",
        extension_receipt=base / "extension8.json",
        historical_registration=base / "registration.json",
        historical_completion=base / "completion.json",
    )


def data(a, d, ids):
    files = paths(a.data)
    require(
        {k: sha(v) for k, v in files.items()} == d["sources"],
        "Changed raw source or generic partition",
    )
    receipt = read(files["physical_trials"])
    ext = read(files["extension_receipt"])
    with np.load(files["stored_repeats"], allow_pickle=False) as z:
        slots = z["stored_slots"]
        names = z["channels"].tolist()
    pick = [receipt["channel_names"].index(s) for s in names]
    require(
        len(pick) == 17
        and np.array_equal(slots, [[1, 3], [0, 2]])
        and receipt["subject"] == "sub-04",
        "Wrong repeated-trial ordering",
    )
    lookup = {}
    for tag in ("historical_latents", "extension_latents"):
        with np.load(files[tag], allow_pickle=False) as z:
            for row, v in zip(z["rows"], z["latents"], strict=True):
                if row in lookup:
                    require(
                        np.allclose(lookup[row], v, rtol=2e-4, atol=2e-4),
                        "VAE extension failed parity",
                    )
                else:
                    lookup[int(row)] = v
    require(ext["n_overlap"] == 24 and ext["n_new"] == 1020, "Different latent coverage")
    eeg = []
    latents = []
    for obj in ids:
        p = files["physical_trials"].parent / f"{obj:04}.npy"
        rec = receipt["blocks"][str(obj)]
        require(
            sha(p) == rec["output_sha256"] and rec["official_row_start"] == 10 * obj,
            "Physical trial changed",
        )
        raw = np.load(p, allow_pickle=False)
        require(raw.shape == (10, 4, 63, 125) and raw.dtype == np.float32, "Unexpected EEG shape")
        # Original sampling is 250 Hz. Bins 15:115 are the half-open 60–460 ms interval.
        x = np.take(raw, pick, axis=2)[..., 15:115].astype(float).reshape(10, 4, 17, 20, 5).mean(-1)
        eeg.append(np.stack([x[:, ix].mean(1).reshape(10, 340) for ix in slots]))
        latents.append(np.stack([lookup[10 * obj + i].reshape(256) for i in range(10)]))
    return np.stack(eeg), np.stack(latents).astype(float)


def centered(v):
    return v - v.mean(axis=-2, keepdims=True)


def relative(v):
    return v - v[..., SUPPORT, :].mean(-2)


def matrix(v):
    return v.reshape(-1, v.shape[-1])


def sqrt_mats(c):
    vals, vec = eigh(c, check_finite=False)
    vals = vals.clip(1e-10)
    return (vec * np.sqrt(vals)) @ vec.T, (vec / np.sqrt(vals)) @ vec.T


def ridge(x, y):
    n = len(x)
    m = x.T @ x
    m.flat[:: len(m) + 1] += 0.3 * n
    return solve(m, x.T @ y, assume_a="pos", check_finite=False)


def fit(a, d):
    train = np.array(d["fit_classes"])
    cal = np.array(d["calibration_classes"])
    require(
        len(set(train)) == 320 and len(set(cal)) == 96 and not set(train) & set(cal),
        "Overlapping training stages",
    )
    e, z = data(a, d, train)
    zf = matrix(centered(z))
    ey = matrix(centered(e.mean(1)))
    ntrain = len(zf)
    c = zf.T @ zf / ntrain
    zroot, zw = sqrt_mats(0.9 * c + 0.1 * np.trace(c) / 256 * np.eye(256))
    noise = matrix(centered((e[:, 0] - e[:, 1]) / 2))
    rc = noise.T @ noise / ntrain
    rot = 0.5 * rc + 0.5 * np.diag(np.diag(rc)) + 1e-8 * np.trace(rc) / 340 * np.eye(340)
    _, ew = sqrt_mats(rot)
    zs = zf @ zw
    ys = ey @ ew
    A = ridge(zs, ys)
    inv = ridge(ys, zs) @ zroot
    u, s, vt = svd(A, full_matrices=False, check_finite=False)
    U, V = u[:, :16], vt[:16].T
    ecal, zcal = data(a, d, cal)
    zr = relative(zcal) @ zw @ U
    er = relative(ecal) @ ew @ V
    # Inner-test split was locked before measuring either validation half.
    cross = (zr[:, None] * er).mean(2)
    count = np.random.default_rng(260572).multinomial(96, np.repeat(1 / 96, 96), size=20000)
    draws = (count @ cross.reshape(96, -1) / 96).reshape(-1, 2, 16)
    bound = np.quantile(draws, 0.05 / (2 * 16), axis=0)
    keep = np.all(bound > 0, axis=0)
    y = er.mean(1)
    repeat = (er[:, 0] - er[:, 1]) / 2
    vz = (zr**2).mean((0, 1)).clip(1e-12)
    beta = (zr * y).mean((0, 1)) / vz
    r = np.maximum(((y - zr * beta) ** 2).mean((0, 1)), (repeat**2).mean((0, 1))).clip(1e-8)
    gain = beta * vz / (beta * beta * vz + r)
    p = a.out / "model.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    frozen_npz(
        p,
        dict(
            zroot=zroot,
            zw=zw,
            ew=ew,
            U=U[:, keep],
            V=V[:, keep],
            beta=beta[keep],
            noise=r[keep],
            gain=gain[keep],
            inverse=inv,
        ),
    )
    freeze(
        a.out / "fit.json",
        dict(
            design_sha256=LOCK,
            model_sha256=sha(p),
            code_sha256=sha(Path(__file__)),
            fit_ids=train.tolist(),
            calibration_ids=cal.tolist(),
            leading_singular_values=s[:16].tolist(),
            inner_lower_crossproduct=bound.tolist(),
            retained_mode_indices=np.flatnonzero(keep).tolist(),
            calibration_repeat_correlation=[
                float(np.corrcoef(er[:, 0, :, j].ravel(), er[:, 1, :, j].ravel())[0, 1])
                for j in range(16)
            ],
            retained_noise=r[keep].tolist(),
            retained_gain=gain[keep].tolist(),
        ),
    )


def predict(a, d):
    fit_meta = read(a.out / "fit.json")
    check_model(a, fit_meta)
    ids = np.asarray(d["evaluation_classes"])
    require(
        len(set(ids)) == 96
        and not set(ids) & set(fit_meta["fit_ids"] + fit_meta["calibration_ids"]),
        "Non-independent evaluation objects",
    )
    eeg, lat = data(a, d, ids)
    with np.load(a.out / "model.npz", allow_pickle=False) as f:
        zroot, zw, ew, U, V, beta, gain, inv = [
            f[k] for k in ("zroot", "zw", "ew", "U", "V", "beta", "gain", "inverse")
        ]
    earlier, later = relative(eeg)[:, 0] @ ew, relative(eeg)[:, 1] @ ew
    y = (earlier + later) / 2
    observed = y @ V
    wrong = observed[:, np.arange(10) ^ 1]
    basis = U.T @ zroot
    # Each query is evaluated in its own fold; the proposal uses exactly its eight listed supports.
    allowed = lat[:, SUPPORT]
    prior = allowed.mean(2)
    closest = ((allowed - prior[:, :, None]) ** 2).sum(-1).argmin(2)
    medoid = np.take_along_axis(allowed, closest[:, :, None, None], axis=2)[:, :, 0, :]
    pred = {}
    for key, start in (("mean", prior), ("medoid", medoid)):
        zc = (start - prior) @ zw @ U
        for arm, ob in (("matched", observed), ("wrong", wrong), ("mean", np.zeros_like(observed))):
            pred[key + "_" + arm] = start + 0.35 * ((ob - zc * beta) * gain) @ basis
        pred[key + "_start"] = start
        pred[key + "_candidate_blind"] = start + 0.35 * (observed * gain) @ basis
        pred[key + "_inverse"] = start + 0.35 * (prior + y @ inv - start)
        pred[key + "_inverse_mean"] = start + 0.35 * (prior - start)
    file = a.out / "heldout_predictions.npz"
    frozen_npz(
        file,
        dict(
            objects=ids,
            observed=observed,
            early=earlier @ V,
            late=later @ V,
            medoid_support_ordinal=SUPPORT[np.arange(10)[None, :], closest],
            **pred,
        ),
    )
    freeze(
        a.out / "predictions.json",
        dict(
            design_sha256=LOCK,
            model_sha256=fit_meta["model_sha256"],
            prediction_sha256=sha(file),
            code_sha256=sha(Path(__file__)),
            objects=ids.tolist(),
            n=960,
            arms=list(METHODS[:-2]),
        ),
    )


def check_model(a, m):
    require(
        m["design_sha256"] == LOCK
        and m["code_sha256"] == sha(Path(__file__))
        and sha(a.out / "model.npz") == m["model_sha256"],
        "Changed fit or scorer",
    )


def evaluate(a, d):
    fit_meta = read(a.out / "fit.json")
    check_model(a, fit_meta)
    packet = a.out / "heldout_predictions.npz"
    seal = read(a.out / "predictions.json")
    require(
        sha(packet) == seal["prediction_sha256"]
        and seal["model_sha256"] == fit_meta["model_sha256"],
        "Held-out predictions not sealed",
    )
    with np.load(packet, allow_pickle=False) as z:
        pred = dict(z)
    ids = pred["objects"]
    _, truth = data(a, d, ids)
    require(np.array_equal(ids, d["evaluation_classes"]), "Wrong final objects")
    with np.load(a.out / "model.npz", allow_pickle=False) as f:
        U, b, beta, noise = [f[k] for k in ("U", "zroot", "beta", "noise")]
        coordinates = relative(truth) @ f["zw"] @ U
    rank = len(beta)
    response = coordinates * beta
    zero = np.zeros((96, 10))
    other = response[:, np.arange(10) ^ 1]
    boot = np.random.default_rng(260573).integers(96, size=(10000, 96))

    def est(v):
        cs = v.reshape(96, -1).mean(1)
        return dict(
            mean=float(cs.mean()), ci95=np.quantile(cs[boot].mean(1), [0.025, 0.975]).tolist()
        )

    forward = {}
    for k, y in (("early", pred["early"]), ("late", pred["late"]), ("r4", pred["observed"])):
        if rank:
            raw = (y**2).mean(-1)
            actual = ((y - response) ** 2).mean(-1)
            wrong = ((y - other) ** 2).mean(-1)
            margin = (((y - other) ** 2 - (y - response) ** 2) / noise).mean(-1)
        else:
            raw = actual = wrong = margin = zero
        forward[k] = dict(
            benefit_vs_zero=est(raw - actual),
            benefit_vs_wrong=est(wrong - actual),
            fraction_explained=float((raw - actual).mean() / raw.mean()) if rank else 0.0,
            per_query_two_choice_credit=float(((margin > 0) + 0.5 * (margin == 0)).sum()),
            paired_assignment_credit=float(
                (
                    (margin.reshape(96, 5, 2).sum(-1) > 0)
                    + 0.5 * (margin.reshape(96, 5, 2).sum(-1) == 0)
                ).sum()
            ),
        )
    if rank:
        q, _ = np.linalg.qr((U.T @ b).T, mode="reduced")
    else:
        q = np.empty((256, 0))
    outcomes = {}
    medoid_gate = None
    for proposal in ("mean", "medoid"):
        start = pred[proposal + "_start"]
        projected = ((truth - start) @ q) @ q.T
        vals = {k: pred[proposal + "_" + k] for k in METHODS[:-2]}
        vals["oracle35"] = start + 0.35 * projected
        vals["oracle"] = start + projected
        err = {k: ((v - truth) ** 2).mean(-1) for k, v in vals.items()}
        benefit = {
            k: est(err[k] - err["matched"])
            for k in ("start", "mean", "wrong", "candidate_blind", "inverse")
        }
        increment = (err["mean"] - err["matched"]) - (err["inverse_mean"] - err["inverse"])
        outcomes[proposal] = dict(
            mean_squared_raw_latent_error={k: est(v) for k, v in err.items()},
            benefit_of_matched=benefit,
            forward_minus_inverse_eeg_benefit=est(increment),
            oracle_fraction_recoverable=float(
                (err["start"] - err["oracle"]).mean() / err["start"].mean()
            ),
        )
        if proposal == "medoid":
            medoid_gate = (
                all(benefit[k]["ci95"][0] > 0 for k in ("start", "mean", "wrong"))
                and benefit["candidate_blind"]["mean"] > 0
                and increment.mean() >= 0
            )
    passed = (
        rank > 0
        and all(forward[k]["benefit_vs_zero"]["ci95"][0] > 0 for k in ("early", "late"))
        and medoid_gate
    )
    result = dict(
        design_sha256=LOCK,
        source_sha256=sha(Path(__file__)),
        fit_receipt_sha256=sha(a.out / "fit.json"),
        prediction_receipt_sha256=sha(a.out / "predictions.json"),
        subject="sub-04",
        training_objects=320,
        calibration_objects=96,
        heldout_objects=96,
        heldout_photos=960,
        heldout_pairs=480,
        rank=rank,
        mode_indices=fit_meta["retained_mode_indices"],
        calibration_repeat_correlation=fit_meta["calibration_repeat_correlation"],
        forward=forward,
        proposals=outcomes,
        gate_passed=bool(passed),
        generated_images=0,
    )
    freeze(a.out / "summary.json", result)
    if a.release:
        freeze(a.release, result)
    print(
        "Rank",
        rank,
        "medoid benefit",
        outcomes["medoid"]["benefit_of_matched"],
        "gate",
        bool(passed),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["fit", "predict", "evaluate"])
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("outputs/vae-forward-measurement"))
    p.add_argument("--release", type=Path)
    a = p.parse_args()
    require(sha(DESIGN) == LOCK, "Protocol changed")
    {"fit": fit, "predict": predict, "evaluate": evaluate}[a.stage](a, read(DESIGN))
