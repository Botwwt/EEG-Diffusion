"""Cross-fit single-participant foreground pose residuals, evaluate generic objects and freeze personal12 packets."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fit_shared_photo_eeg import load_photos
from measure_personal_pose import KEYS, LOCK
from prepare_expanded_personal import extract
from prepare_trial_object_inputs import check
from register_fresh_photo_panel import freeze, read, sha
from run_prospective_readout import frozen_npz
from scipy.linalg import solve
from scipy.optimize import nnls

ARMS = ("own", "uniform", "wrong")
METRICS = (
    "orientation",
    "center",
    "log_radius",
    "five_coordinate",
    "anisotropy_xx_yy",
    "anisotropy_xy",
    "center_x",
    "center_y",
)


def center(x):
    return x - x.mean(1, keepdims=True)


def fit(x, y):
    xx = center(x).reshape(-1, x.shape[-1])
    sd = xx.std(0).clip(1e-5)
    xx = xx / sd
    yy = center(y).reshape(-1, 5)
    matrix = xx.T @ xx
    matrix.flat[:: len(sd) + 1] += 0.5 * len(xx)
    return solve(matrix, xx.T @ yy, assume_a="pos") / sd[:, None]


def eight_mean(x):
    # Each consecutive pair is held out together; all eight other photos are supports.
    return np.repeat((x.sum(1)[:, None, :] - x.reshape(len(x), 5, 2, -1).sum(2)) / 8, 2, axis=1)


def relative(x):
    return x - eight_mean(x)


def scale_for(y):
    sd = center(y).reshape(-1, 5).std(0).clip(1e-8)
    sd[3:] = np.sqrt(np.mean(sd[3:] ** 2))
    return sd


def constrain(v):
    v = v.copy()
    v[..., :2] = v[..., :2].clip(0, 1)
    v[..., 2] = v[..., 2].clip(np.log(1 / 320), np.log(np.sqrt(0.5)))
    magnitude = np.linalg.norm(v[..., 3:], axis=-1, keepdims=True)
    v[..., 3:] *= np.minimum(1, 0.99 / magnitude.clip(1e-12))
    return v


def calibrate(x, d, y, fold):
    pred = np.empty((2, *y.shape))
    for f in range(4):
        test = fold == f
        check(test.sum() == 104, "Unexpected inner fold size")
        for j, head in enumerate((x, d[f].astype(float))):
            pred[j, test] = relative(head[test]) @ fit(head[~test], y[~test])
    obs = relative(y)
    gains = np.array(
        [nnls(pred[:, :, :, j].reshape(2, -1).T, obs[:, :, j].ravel())[0] for j in range(5)]
    ).T.clip(0, 2)
    return gains


def loss(v, y, sd):
    err = ((v - y) / sd) ** 2
    return np.stack(
        [
            err[..., 3:].mean(-1),
            err[..., :2].mean(-1),
            err[..., 2],
            err.mean(-1),
            err[..., 3],
            err[..., 4],
            err[..., 0],
            err[..., 1],
        ],
        axis=-1,
    )


def evaluate_case(a, sub, design, rows, pose):
    root = a.out / "features" / sub
    info = read(root / "receipt.json")
    check(
        info["subject"] == sub
        and info["registration_sha256"] == LOCK
        and info["packet_sha256"] == sha(root / "features.npz"),
        "Unregistered EEG features",
    )
    with np.load(root / "features.npz", allow_pickle=False) as z:
        eeg = dict(z)
    ids, train, held = eeg["ids"], eeg["training"], eeg["held"]
    ti, ei = np.searchsorted(ids, train), np.searchsorted(ids, held)
    lookup = {int(r): i for i, r in enumerate(rows)}
    index = np.array([[lookup[int(c) * 10 + i] for i in range(10)] for c in ids])
    y = pose[index]
    expanded = a.data / "photo-structure/expanded"
    check(
        sha(expanded / "features" / (sub + ".json")) == design["people"][sub]["generic_source"],
        "Changed posterior data",
    )
    post = load_photos(SimpleNamespace(expanded=expanded), sub, ids)
    gains = calibrate(post[ti], eeg["deep_oof"], y[ti], eeg["inner_folds"])
    sd = scale_for(y[ti])
    v = np.zeros_like(y[ei])
    for j, head in enumerate((post, eeg["deep416"].astype(float))):
        v += (relative(head[ei]) @ fit(head[ti], y[ti])) * gains[j]
    prior = eight_mean(y[ei])
    own = constrain(prior + v)
    wrong = own[:, np.arange(10) ^ 1]
    arms = (own, constrain(prior), wrong)
    err = np.stack([loss(t, y[ei], sd) for t in arms])
    eval_path = a.out / "generic" / sub / "evaluation.npz"
    frozen_npz(
        eval_path,
        dict(
            objects=held,
            truth=y[ei],
            own=own,
            uniform=arms[1],
            wrong=wrong,
            scale=sd,
            gains=gains,
            keys=np.array(KEYS),
            metrics=np.array(METRICS),
            errors=err,
        ),
    )
    model_dir = a.out / "models" / sub
    model_file = model_dir / "model.npz"
    frozen_npz(
        model_file,
        dict(
            posterior=fit(post, y),
            deep=fit(eeg["deep512"].astype(float), y),
            scale=scale_for(y),
            gains=gains,
            fit_classes=ids,
            calibration_classes=train,
        ),
    )
    modelmeta = dict(
        subject=sub,
        design_sha256=LOCK,
        visual_targets_sha256=sha(a.out / "sources.npz"),
        feature_receipt_sha256=sha(root / "receipt.json"),
        model_sha256=sha(model_file),
        generic_results_sha256=sha(eval_path),
        joint_calibration=gains.tolist(),
        fitting_rows=4160,
        validation_rows=960,
        full_fit_rows=5120,
    )
    freeze(model_dir / "receipt.json", modelmeta)
    return err, modelmeta


def packet(a, sub, design, rows, pose):
    mix = a.data / (
        "trial-object-encoder/personal12" if sub == "sub-04" else f"trial-object-replication/{sub}"
    )
    source = mix / "render_inputs/conditions" / sub / "personal12/conditions_512.npz"
    ref = a.data / "object-head-confirmation/references" / f"personal12_{sub}_ablation.json"
    post = a.data / "object-head-confirmation/eeg" / (sub + ".npz")
    deep = a.out / "features" / sub / "features.npz"
    model = a.out / "models" / sub / "model.npz"
    cfg = design["people"][sub]
    check(
        sha(source) == cfg["mixed_conditions"]
        and sha(post) == cfg["target_posterior"]
        and sha(ref) == cfg["registration"]
        and sha(model) == read(model.parent / "receipt.json")["model_sha256"],
        "Changed packet/targets",
    )
    with np.load(source, allow_pickle=False) as z:
        ids, folds, words, supports, donors = [
            z[k] for k in ("rows", "folds", "predicted_words", "support_rows", "donor_rows")
        ]
    with np.load(post, allow_pickle=False) as z:
        sr = z["rows"]
        x = extract(z["half_eeg"].mean(0)).astype(float)
    with np.load(deep, allow_pickle=False) as z:
        check(np.array_equal(sr, z["target_rows"]), "Wrong personal row order")
        u = z["target_deep"].astype(float)
    with np.load(model, allow_pickle=False) as z:
        weights = dict(z)
    check(
        np.array_equal(np.sort(ids), sr) and not set(sr // 10) & set(weights["fit_classes"]),
        "Test class in fit",
    )
    reg = read(ref)
    check(
        np.array_equal(ids, reg["rows"]) and np.array_equal(folds, reg["folds"]),
        "Wrong per-person panel",
    )
    menus = np.asarray(reg["calibration_rows_by_fold_and_word"])
    vocabulary = np.array(reg["words"])
    wi = np.array([np.flatnonzero(vocabulary == w).item() for w in words])
    check(
        np.array_equal(supports, menus[folds, wi])
        and all(set(supports[folds == f].ravel()).isdisjoint(ids[folds == f]) for f in range(5)),
        "Unregistered visual support",
    )
    qi, si = np.searchsorted(sr, ids), np.searchsorted(sr, supports)
    vi = np.searchsorted(rows, supports)
    check(
        np.array_equal(rows[vi], supports) and np.array_equal(sr[si], supports),
        "Missing permitted input",
    )
    # The foreground source is indexed exclusively by the eight registered calibration row IDs.
    prior = constrain(pose[vi].mean(1))
    delta = sum(
        ((v[qi] - v[si].mean(1)) @ weights[name]) * weights["gains"][j]
        for j, (name, v) in enumerate((("posterior", x), ("deep", u)))
    )
    own = constrain(prior + delta)
    lookup = {int(r): i for i, r in enumerate(ids)}
    di = np.array([lookup[int(r)] for r in donors])
    check(
        np.array_equal(words, words[di])
        and np.array_equal(folds, folds[di])
        and np.array_equal(supports, supports[di]),
        "Wrong donor conditional stratum",
    )
    dest = a.out / "personal12" / sub
    target = dest / "pose.npz"
    frozen_npz(
        target,
        dict(
            subject=sub,
            rows=ids,
            folds=folds,
            predicted_words=words,
            support_rows=supports,
            donor_rows=donors,
            own=own,
            uniform=prior,
            wrong=own[di],
            keys=np.array(KEYS),
            scale=weights["scale"],
            model_sha256=sha(model),
            source_packet_sha256=sha(source),
            design_sha256=LOCK,
        ),
    )
    rec = dict(
        subject=sub,
        protocol_sha256=LOCK,
        registration_sha256=sha(ref),
        mixed_condition_sha256=sha(source),
        packet_sha256=sha(target),
        model_sha256=sha(model),
        keys=list(KEYS),
        scale=weights["scale"].tolist(),
        n=120,
        available_distinct_donors=int((ids != donors).sum()),
        effective_angular_displacements=float(
            np.linalg.norm(own[:, 3:] - prior[:, 3:], axis=1).mean()
        ),
    )
    freeze(dest / "receipt.json", rec)
    return rec


def summary(errors):
    # [people, arm, object, photo, endpoint] for the same 96 held-out objects.
    e = np.asarray(errors)
    check(e.shape[1:] == (3, 96, 10, len(METRICS)), "Unexpected assessment rows")
    draws = np.random.default_rng(260552).integers(96, size=(10000, 96))
    m = e.mean((0, 3))
    stats = {}
    for j, name in enumerate(METRICS):
        arm_mean = m[:, :, j].mean(1)
        comparisons = {}
        for k in (1, 2):
            diff = m[k, :, j] - m[0, :, j]
            ci = np.quantile(diff[draws].mean(1), [0.025, 0.975])
            comparisons[ARMS[k]] = dict(reduction=float(diff.mean()), ci95=ci.tolist())
        stats[name] = dict(
            mse=dict(zip(ARMS, arm_mean.tolist(), strict=True)), own_error_reduction=comparisons
        )
    return stats


def run(a):
    design = read(a.out / "design.json")
    check(sha(a.out / "design.json") == LOCK, "Unexpected pose registration")
    with np.load(a.out / "sources.npz", allow_pickle=False) as z:
        rows, pose = z["rows"], z["pose"]
        check(
            str(z["registration_sha256"]) == LOCK
            and z["keys"].tolist() == list(KEYS)
            and np.isfinite(pose).all()
            and np.all(np.diff(rows) > 0),
            "Invalid foreground targets",
        )
        avail = dict(
            n_unique_masks=len(rows),
            generic_rows=len(z["generic_rows"]),
            allowed_personal_support_rows=len(z["allowed_support_rows"]),
            legacy_overlap=int(z["prior_mask_overlap"]),
            max_abs_historical_moment_replay=float(z["max_prior_moment_error"]),
        )
    selected = [a.subject] if a.subject else design["subjects"]
    for sub in selected:
        if a.stage in ("fit", "all"):
            evaluate_case(a, sub, design, rows, pose)
        if a.stage in ("package", "all"):
            print(sub, packet(a, sub, design, rows, pose), flush=True)
    if a.stage in ("summarize", "all") and len(selected) == 3:
        e = []
        reports = {}
        objects = None
        for sub in selected:
            path = a.out / "generic" / sub / "evaluation.npz"
            with np.load(path, allow_pickle=False) as z:
                if objects is None:
                    objects = z["objects"]
                check(np.array_equal(objects, z["objects"]), "Inconsistent held-out object order")
                e.append(z["errors"])
            reports[sub] = dict(
                metrics=summary(e[-1:]),
                fit=read(a.out / "models" / sub / "receipt.json"),
                personal_packet=read(a.out / "personal12" / sub / "receipt.json"),
            )
        stats = summary(e)
        gate = all(c["ci95"][0] > 0 for c in stats["orientation"]["own_error_reduction"].values())
        result = dict(
            design_sha256=LOCK,
            source_sha256=sha(a.out / "sources.npz"),
            availability=avail,
            primary="orientation",
            orientation_gate=gate,
            n_held_objects=96,
            n_held_photos=960,
            n_person_photo=2880,
            metrics=stats,
            individual=reports,
        )
        freeze(a.out / "result.json", result)
        print(
            "Orientation", gate, stats["orientation"], "five", stats["five_coordinate"], flush=True
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["fit", "package", "summarize", "all"])
    p.add_argument("--subject", choices=["sub-04", "sub-05", "sub-01"])
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("outputs/personal-pose"))
    run(p.parse_args())
