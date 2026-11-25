"""Test a photograph-specific EEG update with protected object-related visual coordinates."""

import argparse
from pathlib import Path

import numpy as np
from calibrate_official_repeat_angles import tangent
from calibrate_photo_angles import rotate
from fit_canonical_neighbor import (
    angle_set,
    metrics,
    model,
    native,
    nhood,
    photo_bank,
    physical,
    sources,
)
from fit_repeat_photo_subspace import centered, ridge
from global_photo_conditions import unit
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import check
from run_prospective_readout import frozen_npz
from scipy.linalg import svd

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
DESIGN = PUBLIC / "protected_photo_design.json"
LOCK = "9382b8d641db0b02ed84fe17fb2c7e00e5d4d3fc5ef296389c0f95a83f77bf48"


def semantic_axes(bank):
    centers = bank.reshape(-1, 10, 1024).astype(float).mean(1)
    mean = unit(centers.mean(0))
    orth = centers - np.outer(centers @ mean, mean)
    _, s, vt = svd(orth, full_matrices=False, check_finite=False)
    q = np.column_stack((mean, vt[:96].T))
    check(
        s[95] > 1e-7 and np.max(np.abs(q.T @ q - np.eye(97))) < 1e-9, "Degenerate visual subspace"
    )
    return q


def alter(base, prior, eeg, wrong, true, q):
    def outside_global(x):
        return x - (x @ q) @ q.T

    extra = unit(outside_global(prior))

    def remaining(x):
        v = outside_global(x)
        return v - (v * extra).sum(1)[:, None] * extra

    rem = remaining(base)
    fixed = base - rem
    magnitude = np.linalg.norm(rem, axis=1, keepdims=True)
    original = unit(rem)
    result = {"native": base}
    for k, delta in (("matched", eeg), ("wrong", wrong), ("reference", true)):
        v = remaining(delta)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        proposal = np.where(norm > 1e-9, v / norm.clip(1e-9), original)
        if k != "reference":
            proposal = unit(original + proposal)
        result[k] = fixed + magnitude * proposal
    dev = max(float(np.max(np.abs((v - base) @ q))) for v in result.values())
    prior_dev = max(float(np.max(np.abs(((v - base) * prior).sum(1)))) for v in result.values())
    check(
        max(dev, prior_dev) < 2e-6
        and all(np.max(abs(np.linalg.norm(v, axis=1) - 1)) < 2e-6 for v in result.values()),
        "Protected cosine or norm changed",
    )
    return result, dev, prior_dev


def run(a):
    spec = read(DESIGN)
    files = {
        "parent_image_study": PUBLIC / "global_photo_images.json",
        "canonical_protocol": PUBLIC / "canonical_neighbor_design.json",
        "generic_splits": a.data / "official-repeat-object/design.json",
        "angle": a.data / "photo-structure/residual_angle/scale.json",
        "estimator": a.data / "canonical-neighbor/selection.json",
    }
    check(
        sha(DESIGN) == LOCK and {k: sha(p) for k, p in files.items()} == spec["sources"],
        "Changed generic design or fit",
    )
    p, author, tr, held = sources(a)
    x, y = physical(p, tr), photo_bank(author, tr)
    xx = physical(p, held)
    m = model(x, y)
    q = semantic_axes(y)
    sel = read(files["estimator"])
    check(sel["training_classes"] == tr.tolist(), "Canonical fit class mismatch")
    vg, coord = native(m, xx)
    neigh = nhood(m, coord, y, (sel["k"],))[sel["k"]]
    absolute = angle_set(
        tangent(vg, m["axis"]), neigh, sel["mix"], m["axis"], sel["gain"], sel["cap"]
    ).astype(float)
    displacements = ridge(
        x, centered(y.reshape(len(tr), 10, -1).astype(float)), xx.reshape(-1, 1003)
    )
    gallery = photo_bank(author, held).astype(float)
    ix = np.arange(len(gallery))
    support = np.array(
        [[10 * (i // 10) + j for j in range(10) if j // 2 != (i % 10) // 2] for i in ix]
    )
    donor = ix ^ 1
    check(
        np.all((ix // 10) == donor // 10)
        and np.all((ix[:, None] != support) & (donor[:, None] != support)),
        "Support fold leakage",
    )
    mean = gallery[support].mean(1)
    prior = unit(mean)
    base = unit(
        rotate(
            prior, unit(mean + displacements - displacements[support].mean(1)), read(files["angle"])
        ).astype(float)
    )
    delta = absolute - absolute[support].mean(1)
    estimates, gerr, perr = alter(base, prior, delta, delta[donor], gallery, q)
    # Only this scoring block and the explicitly labeled reference condition receive the target.
    vals = {}
    for arm, v in estimates.items():
        mm = metrics(v, gallery)
        c1 = (v * gallery).sum(1)
        c2 = (v * gallery[donor]).sum(1)
        vals[arm] = mm | dict(
            held_margin=c1 - c2, held_pair=((c1 > c2) + 0.5 * (c1 == c2)).astype(float)
        )
    file = a.out / "scores.npz"
    file.parent.mkdir(exist_ok=True, parents=True)
    frozen_npz(
        file,
        {"rows": (10 * held[:, None] + np.arange(10)).ravel()}
        | {arm + "_" + k: v for arm, ms in vals.items() for k, v in ms.items()},
    )
    counts = {
        k: {arm: float(v[k].sum()) for arm, v in vals.items()}
        for k in ("top1", "top5", "within10", "held_pair")
    }
    means = {k: {arm: float(v[k].mean()) for arm, v in vals.items()} for k in vals["native"]}
    draws = np.random.default_rng(spec["seed"]).integers(len(held), size=(10000, len(held)))
    diffs = {}
    for aa, bb in (("matched", "native"), ("matched", "wrong"), ("reference", "native")):
        diffs[aa + "_minus_" + bb] = {}
        for k in vals[aa]:
            v = (vals[aa][k] - vals[bb][k]).reshape(len(held), 10).mean(1)
            diffs[aa + "_minus_" + bb][k] = dict(
                delta=float(v.mean()), ci95=np.quantile(v[draws].mean(1), [0.025, 0.975]).tolist()
            )
    gate = (
        all(diffs["matched_minus_" + k]["held_margin"]["ci95"][0] > 0 for k in ("native", "wrong"))
        and counts["within10"]["matched"] > counts["within10"]["native"]
        and diffs["reference_minus_native"]["held_margin"]["delta"] > 0
    )
    result = dict(
        design_sha256=LOCK,
        driver_sha256=sha(Path(__file__)),
        per_row_sha256=sha(file),
        objects=len(held),
        queries=len(gallery),
        protected_global_directions=q.shape[1],
        max_global_coordinate_error=gerr,
        max_support_cosine_error=perr,
        counts=counts,
        means=means,
        paired=diffs,
        advance=bool(gate),
    )
    freeze(a.release, result)
    print(counts, "held_margin", diffs["matched_minus_native"]["held_margin"], "advance", gate)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("outputs/protected-photo"))
    p.add_argument("--release", type=Path, default=PUBLIC / "protected_photo.json")
    run(p.parse_args())
