"""Cross-fit a direct EEG-to-foreground-outline estimate and test its fixed shrinkage."""

import argparse

import numpy as np
from fit_repeat_photo_subspace import centered, ridge
from fit_support_contour import (
    DESIGN,
    HERE,
    PEOPLE,
    PUBLIC,
    check,
    generic_features,
    read,
    save,
    sha,
    split,
)

LOCK = "2e12e67fab09fc0df1282acbd6227752d2c81e1fee44f15837822140aacd825e"


def zero_mean(x):
    return x - x.mean(-1, keepdims=True)


def other_eight(n):
    ij = np.arange(n)
    return 10 * (ij // 10)[:, None] + np.array(
        [[i for i in range(10) if i // 2 != (j % 10) // 2] for j in ij]
    )


def datum():
    check(sha(PUBLIC / "direct_contour_design.json") == LOCK, "Changed direct contour design")
    classes, train, test = split()
    with np.load(HERE / "contours.npz", allow_pickle=False) as s:
        rows = s["rows"]
        contours = s["values"].astype(np.float64)
        check(
            str(s["design_sha256"]) == DESIGN
            and np.array_equal(rows, (classes[:, None] * 10 + np.arange(10)).ravel())
            and contours.shape == (1280, 24)
            and np.isfinite(contours).all()
            and np.max(abs(contours.mean(1))) < 1e-6,
            "Incorrect normalized generic outlines",
        )
    data = {}
    signatures = {}
    for s in PEOPLE:
        data[s], signatures[s] = generic_features(s, classes)
    return (
        classes,
        np.searchsorted(classes, train),
        np.searchsorted(classes, test),
        contours.reshape(128, 10, 24),
        data,
        signatures,
    )


def residual(data, y, fit, test):
    targets = centered(y[fit])
    support = other_eight(10 * len(test))
    preds = []
    for x in data.values():
        g = ridge(x[fit], targets, x[test].reshape(10 * len(test), -1))
        preds.append(g - np.median(g[support], axis=1))
    return zero_mean(np.mean(preds, axis=0))


def median_prior(bank):
    return zero_mean(np.median(bank, axis=1))


def slope(target, estimated):
    keep = np.abs(estimated) > 1e-12
    ratios = target[keep] / estimated[keep]
    weights = np.abs(estimated[keep])
    ix = np.argsort(ratios)
    j = np.searchsorted(np.cumsum(weights[ix]), weights.sum() / 2)
    return float(np.clip(ratios[ix[j]], 0, 1))


def calibrate():
    classes, train, _, y, data, signatures = datum()
    folds = np.array_split(np.random.default_rng(260555).permutation(train), 4)
    oof = np.full((len(train), 10, 24), np.nan)
    for held in folds:
        fit = np.setdiff1d(train, held)
        oof[np.searchsorted(train, held)] = residual(data, y, fit, held).reshape(-1, 10, 24)
    check(np.isfinite(oof).all(), "Missing classwise OOF calibration")
    oof = oof.reshape(-1, 24)
    actual = y[train].reshape(-1, 24)
    prior = median_prior(actual[other_eight(len(actual))])
    coefficient = slope(actual - prior, oof)
    limits = (0.0, coefficient, 1.0)
    losses = [float(np.mean(np.abs(actual - prior - w * oof))) for w in limits]
    check(losses[1] <= min(losses[0], losses[2]) + 1e-12, "L1 shrinkage optimum inconsistent")
    save(
        "direct_calibration.json",
        dict(
            design_sha256=LOCK,
            contours_sha256=sha(HERE / "contours.npz"),
            eeg_packet_sha256=signatures,
            classes=classes[train].tolist(),
            held_by_fit=[classes[h].tolist() for h in folds],
            n=960,
            coefficient=coefficient,
            prior_loss=losses[0],
            calibrated_oof_loss=losses[1],
            unit_slope_loss=losses[2],
        ),
    )
    print("OOF shrinkage", coefficient, "loss prior/fitted/unity", losses, flush=True)


def predict():
    classes, train, test, y, data, signatures = datum()
    fit = read(HERE / "direct_calibration.json")
    check(
        fit["design_sha256"] == LOCK
        and np.array_equal(fit["classes"], classes[train])
        and fit["contours_sha256"] == sha(HERE / "contours.npz")
        and fit["eeg_packet_sha256"] == signatures,
        "Calibration changed",
    )
    n = len(test) * 10
    support = other_eight(n)
    donor = np.arange(n) ^ 1
    rows = (classes[test, None] * 10 + np.arange(10)).ravel()
    allowed = rows[support]
    check(
        not np.any(allowed // 2 == rows[:, None] // 2) and np.array_equal(allowed, allowed[donor]),
        "Support leak",
    )
    continuous = residual(data, y, train, test)
    bank = y.reshape(-1, 24)[(test[:, None] * 10 + np.arange(10)).ravel()[support]]
    prior = median_prior(bank)
    correction = fit["coefficient"] * continuous
    np.savez(
        HERE / "direct_conditions.npz",
        rows=rows,
        support_rows=allowed,
        donor=donor,
        prior=prior,
        own=prior + correction,
        wrong=prior + correction[donor],
        support=bank,
        design_sha256=LOCK,
    )
    save(
        "direct_predictions.json",
        dict(
            design_sha256=LOCK,
            condition_sha256=sha(HERE / "direct_conditions.npz"),
            training_classes=classes[train].tolist(),
            held_classes=classes[test].tolist(),
            calibration_sha256=sha(HERE / "direct_calibration.json"),
        ),
    )
    print("Frozen", n, "generic predictions", flush=True)


def score(publish):
    classes, _, held = split()
    rec = read(HERE / "direct_predictions.json")
    check(
        rec["condition_sha256"] == sha(HERE / "direct_conditions.npz")
        and rec["design_sha256"] == LOCK,
        "Blind output changed",
    )
    with (
        np.load(HERE / "direct_conditions.npz", allow_pickle=False) as p,
        np.load(HERE / "contours.npz", allow_pickle=False) as original,
    ):
        rows = p["rows"]
        q = np.arange(len(rows))
        check(
            np.array_equal(rows, (held[:, None] * 10 + np.arange(10)).ravel())
            and np.array_equal(original["rows"], (classes[:, None] * 10 + np.arange(10)).ravel()),
            "Different evaluation rows",
        )
        actual = original["values"][np.searchsorted(original["rows"], rows)].astype(float)
        bank = p["support"]
        emitted = {k: p[k] for k in ("own", "prior", "wrong")}
        actual_distance = np.abs(bank - actual[:, None, :]).mean(2)
        hard = {k: np.abs(bank - v[:, None, :]).mean(2).argmin(1) for k, v in emitted.items()}
        mean_loss = {k: np.abs(actual - v).mean(1) for k, v in emitted.items()}
        mean_loss.update({"hard_" + k: actual_distance[q, j] for k, j in hard.items()})
        medoid = np.abs(bank[:, :, None, :] - bank[:, None, :, :]).mean(3).sum(2).argmin(1)
        mean_loss["medoid"] = actual_distance[q, medoid]
        mean_loss["legal_oracle"] = actual_distance.min(axis=1)
    rng = np.random.default_rng(260554)
    boot = rng.integers(32, size=(10000, 32))

    def difference(a, b):
        d = (a - b).reshape(32, 10).mean(1)
        return dict(
            delta=float(d.mean()),
            ci95=np.quantile(d[boot].mean(1), [0.025, 0.975]).tolist(),
            n_better=int((a > b).sum()),
            n_tied=int((a == b).sum()),
            n_worse=int((a < b).sum()),
        )

    diff = {k: difference(v, mean_loss["own"]) for k, v in mean_loss.items() if k != "own"}
    gate = all(diff[k]["ci95"][0] > 0 for k in ("prior", "wrong"))
    result = dict(
        n=320,
        held_objects=32,
        fit_objects=96,
        named_people=list(PEOPLE),
        physical_observations_per_query=36,
        design_sha256=LOCK,
        selection_sha256=rec["condition_sha256"],
        calibration_sha256=rec["calibration_sha256"],
        shape_sha256=sha(HERE / "contours.npz"),
        development="The previous DINO selector had been evaluated on these same 32 classes. All shrinkage fitting used only class-wise out-of-fold predictions on the other 96.",
        slope=read(HERE / "direct_calibration.json")["coefficient"],
        loss={k: float(v.mean()) for k, v in mean_loss.items()},
        control_minus_own=diff,
        hard_prior_minus_hard_own=difference(mean_loss["hard_prior"], mean_loss["hard_own"]),
        advance=bool(gate),
        uncertainty="Mean absolute normalized log-radius error. Positive control-minus-own favors own. 10000 paired 32-object bootstrap replicates.",
    )
    save("direct_result.json", result)
    np.savez(HERE / "direct_evaluation_rows.npz", rows=rows, **mean_loss)
    if publish:
        (PUBLIC / "direct_contour.json").write_bytes((HERE / "direct_result.json").read_bytes())
    print(
        "Direct contour",
        result["loss"],
        "prior",
        diff["prior"],
        "wrong",
        diff["wrong"],
        "advance",
        gate,
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("calibrate", "predict", "score"))
    p.add_argument("--publish", action="store_true")
    a = p.parse_args()
    if a.stage == "calibrate":
        calibrate()
    elif a.stage == "predict":
        predict()
    else:
        score(a.publish)
