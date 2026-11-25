"""Predict a soft category average and decode the trial's residual around its expected EEG."""

import argparse
from pathlib import Path

import numpy as np
from calibrate_official_repeat_angles import expmap, tangent
from fit_official_repeat_objects import fit, standard, training
from fit_official_visual_anchors import prototype_fits
from fit_shared_photo_eeg import read, require, unit, write
from scipy.linalg import cho_factor, cho_solve
from scipy.special import log_softmax
from score_object_panel import score_object_panel
from score_within_photo_dino import sha

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"


def spherical(estimate, center):
    v = unit(estimate)
    axis = unit(center)
    cosine = (v * axis).sum(1, keepdims=True)
    direction = v - cosine * axis
    length = np.linalg.norm(direction, axis=1, keepdims=True)
    return axis, direction / length.clip(1e-12), np.arctan2(length, cosine)


def expand(estimate, center, gain, cap):
    axis, direction, theta = spherical(estimate, center)
    theta = (theta * gain).clip(max=cap)
    return unit(np.cos(theta) * axis + np.sin(theta) * direction)


def within_map(x, y):
    dx = (x - x.mean(1, keepdims=True)).reshape(-1, x.shape[-1])
    dy = (y - y.mean(1, keepdims=True)).reshape(-1, y.shape[-1])
    scale = dx.std(0).clip(1e-8)
    dx /= scale
    gram = dx.T @ dx
    gram.flat[:: len(gram) + 1] += 2 * len(dx)
    return cho_solve(cho_factor(gram), dx.T @ dy) / scale[:, None]


def assess(pred, real):
    count = len(real)
    sim = pred @ real.T
    categories = np.repeat(np.arange(count // 10), 10)
    report = score_object_panel(sim, np.arange(count), categories)
    truth = sim.diagonal()
    in_class = sim.reshape(count, -1, 10)[np.arange(count), categories]
    wins = (
        (in_class < truth[:, None]).sum(1) + 0.5 * ((in_class == truth[:, None]).sum(1) - 1)
    ) / 9
    measures = {
        key: np.array([r[key] for r in report["per_photo"]])
        for key in ("object_top1", "photo_top1", "intra10_top1")
    }
    return measures | dict(within_pair_win=wins, target_cosine=truth)


def run(a):
    protocol = read(a.base / "design.json")
    split = read(a.base / "angles/design.json")
    cal, val = [
        np.asarray(split[k]) for k in ("angle_calibration_classes", "angle_validation_classes")
    ]
    classes = np.setdiff1d(protocol["extended_classes"], np.r_[cal, val])
    require(
        len(classes) == 884 and len(cal) == len(val) == 64 and not np.intersect1d(cal, val).size,
        "Expected disjoint 884/64/64 development objects",
    )
    data = training(a, protocol)
    blocks, visual, names, _, _, text, _ = data
    objects = np.r_[cal, val]
    x = np.stack([blocks[c] for c in classes]).astype("float64")
    y = unit(visual[classes].astype("float64"))
    query = np.concatenate([blocks[c] for c in objects]).astype("float64")
    real = unit(visual[objects].reshape(-1, 1024).astype("float64"))
    scores, _, native = fit(classes, data, query)
    logits = standard(scores["text_32"]).astype("float64")
    labels = np.repeat(names[objects], 10)
    nc = 10 * len(cal)
    grid = [0.25, 0.5, 1.0, 2.0]
    losses = [
        -log_softmax(logits[:nc] / t, axis=1)[np.arange(nc), labels[:nc]].mean() for t in grid
    ]
    temp = grid[np.argmin(losses)]
    posterior = np.exp(log_softmax(logits / temp, axis=1))
    gram = text @ text.T
    means = np.column_stack((x.mean(1), y.mean(1)))
    _, prototypes = next(prototype_fits(0.5 * (gram + gram**3), names[classes], means, [0.5]))
    expected = posterior @ prototypes
    eeg_mean = expected[:, : x.shape[-1]]
    center = expected[:, x.shape[-1] :]
    residual = (query - eeg_mean) @ within_map(x, y)
    estimate = center + residual
    _, _, actual_angle = spherical(real[:nc], center[:nc])
    _, _, pred_angle = spherical(estimate[:nc], center[:nc])
    gain = float(np.median(actual_angle) / np.median(pred_angle))
    cap = float(np.quantile(actual_angle, 0.95))
    official_gain = read(PUBLIC / "official_repeat_global_angle_packet.json")
    global_params = read(a.base / "angles/calibration.json")["parameters"]
    global_axis = unit(native["mean"])
    native_pred = unit(native["mean"] + native["ridge"])
    # Reuse the global transform fitted on the same 64 calibration objects.
    global_pred = expmap(
        tangent(native_pred, global_axis) * global_params["global_gain"],
        global_axis,
        global_params["real_angle_cap_radians"],
    )
    ev = np.arange(nc, len(query))
    donor = np.roll(ev.reshape(-1, 10), 1, axis=1).ravel()
    cols = {
        "native": native_pred[ev],
        "global": global_pred[ev],
        "class_mean": unit(center[ev]),
        "matched": expand(estimate[ev], center[ev], gain, cap),
        "wrong_residual": expand(center[ev] + residual[donor], center[ev], gain, cap),
    }
    observations = {k: assess(v.astype("float64"), real[ev]) for k, v in cols.items()}
    draw = np.random.default_rng(260548).integers(64, size=(10000, 64))
    stats = {}
    for metric in observations["matched"]:
        v = {k: m[metric].reshape(64, 10).mean(1) for k, m in observations.items()}
        diffs = {}
        for k in ("class_mean", "global", "wrong_residual"):
            delta = v["matched"] - v[k]
            diffs[k] = dict(
                mean=float(delta.mean()),
                ci95=np.quantile(delta[draw].mean(1), [0.025, 0.975]).tolist(),
            )
        stats[metric] = dict(
            means={k: float(row.mean()) for k, row in v.items()}, matched_minus=diffs
        )
    gate = (
        stats["object_top1"]["matched_minus"]["global"]["mean"] >= 0
        and stats["within_pair_win"]["matched_minus"]["global"]["ci95"][0] > 0
        and stats["within_pair_win"]["matched_minus"]["wrong_residual"]["mean"] > 0
    )
    report = dict(
        design_sha256=sha(PUBLIC / "soft_class_residual_design.json"),
        generic_split_sha256=sha(a.base / "angles/design.json"),
        original_global_packet_sha256=official_gain["packet_sha256"],
        temperature=temp,
        calibration_nll=[float(v) for v in losses],
        angle_gain=gain,
        angle_cap=cap,
        objects=64,
        photos=640,
        metrics=stats,
        advance=bool(gate),
    )
    write(a.out, report)
    print(
        "Generic matched vs global",
        {
            k: (round(v["means"]["matched"], 4), round(v["means"]["global"], 4))
            for k, v in stats.items()
        },
        "gate",
        gate,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key, path in dict(
        base="outputs/official-repeat-object",
        source="outputs/official-object-readout/sub-03",
        expanded="outputs/photo-structure/expanded",
        extra="outputs/double-calibration",
        external="outputs/external-object-anchors/sub-03/external_anchors.npz",
        out="outputs/soft-class-residual/result.json",
    ).items():
        p.add_argument("--" + key, type=Path, default=Path(path))
    run(p.parse_args())
