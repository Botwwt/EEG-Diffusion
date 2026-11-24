"""Fit photo-reproducible EEG directions using matched recording halves from one person."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from fit_full_head_512 import training
from fit_support_structure import unit
from fit_within_photo_embedding import images
from prepare_full_head_512 import check, defaults
from register_fresh_photo_panel import freeze, read, sha
from scipy.linalg import eigh, solve

RESULTS = Path(__file__).resolve().parents[1] / "results/object_fidelity"
METHODS = ("posterior", "all_features", "shifted", "repeat")


def centered(x):
    return (x - x.mean(1, keepdims=True)).reshape(-1, x.shape[-1])


def ridge(x, y, test):
    x = centered(x)
    scale = x.std(0).clip(1e-8)
    x /= scale
    gram = x.T @ x
    gram.flat[:: len(gram) + 1] += len(x) * 2.0
    return (test / scale) @ solve(gram, x.T @ y, assume_a="pos")


def subspace(halves, shuffle):
    early, late = halves
    if shuffle:
        rng = np.random.default_rng(260528)
        late = np.stack([np.roll(x, rng.integers(1, 10), axis=0) for x in late])
    x, z = centered(early), centered(late)
    n = len(x)
    scale = np.sqrt((x.var(0) + z.var(0)) * 0.5).clip(1e-8)
    x /= scale
    z /= scale
    cross = x.T @ z / n
    total = (x.T @ x + z.T @ z) / (2 * n)
    total.flat[:: len(total) + 1] += 0.5
    _, directions = eigh(
        (cross + cross.T) * 0.5, total, subset_by_index=(len(total) - 64, len(total) - 1)
    )
    return directions / scale[:, None]


def pair_hits(y, pred):
    hits = np.zeros(len(y))
    for i, j in combinations(range(10), 2):
        allowed = [k for k in range(10) if k not in (i, j)]
        mean = y[:, allowed].mean(1)
        calibration = pred[:, allowed].mean(1)
        first = unit(mean + pred[:, i] - calibration)
        second = unit(mean + pred[:, j] - calibration)
        margin = np.sum((first - second) * (y[:, i] - y[:, j]), axis=1)
        hits += (margin > 1e-12) + 0.5 * (abs(margin) <= 1e-12)
    return hits


def run(a):
    plan = read(a.design)
    check(sha(a.split) == plan["split_source_sha256"], "Generic split changed")
    split = read(a.split)
    fit_ids, test_ids = (np.asarray(split[k]) for k in ("train_classes", "test_classes"))
    check(len(fit_ids) == 416 and len(test_ids) == 96, "Unexpected generic population")
    check(not set(fit_ids) & set(test_ids), "Generic groups overlap")
    bank = images(a.source)
    checks = {}
    per_object = {}
    for sub in split["subjects"]:
        data, _ = training(a, sub)
        order = data.rows.reshape(-1, 10)
        objs = order[:, 0] // 10
        check(
            set(objs) == set(np.r_[fit_ids, test_ids])
            and np.array_equal(order, objs[:, None] * 10 + np.arange(10)),
            "Physical EEG rows differ from the generic split",
        )
        idx = {c: i for i, c in enumerate(objs)}
        train, test = ([idx[c] for c in group] for group in (fit_ids, test_ids))
        y = unit(bank[order].astype(float))
        target = centered(y[train])
        p = data.p.reshape(512, 10, -1)
        h = data.f.reshape(2, 512, 10, -1)
        full = h.mean(0)
        base = ridge(p[train], target, p[test])
        predictions = {
            "posterior": base,
            "all_features": (base + ridge(full[train], target, full[test])) / 2,
        }
        for name in ("shifted", "repeat"):
            directions = subspace(h[:, train], shuffle=name == "shifted")
            shared = ridge(full[train] @ directions, target, full[test] @ directions)
            predictions[name] = (base + shared) / 2
        per_object[sub] = {
            name: pair_hits(y[test], pred).tolist() for name, pred in predictions.items()
        }
        checks[sub] = sha(a.out / "features" / (sub + ".json"))

    pooled = {m: np.array([d[m] for d in per_object.values()]).sum(0) for m in METHODS}
    draws = np.random.default_rng(260528).integers(len(test_ids), size=(10000, len(test_ids)))
    stats = {}
    for m in METHODS[:-1]:
        delta = (pooled["repeat"] - pooled[m]) / (45 * len(per_object))
        stats[m] = dict(
            delta=float(delta.mean()),
            ci95=np.quantile(delta[draws].mean(1), [0.025, 0.975]).tolist(),
        )
    counts = {s: {m: sum(v) for m, v in ms.items()} for s, ms in per_object.items()}
    advance = (
        all(v["ci95"][0] > 0 for v in stats.values())
        and sum(v["repeat"] > v["posterior"] for v in counts.values()) >= 2
    )
    freeze(
        a.release,
        dict(
            design_sha256=sha(a.design),
            split_sha256=sha(a.split),
            full_feature_receipts=checks,
            evaluation_objects=test_ids.tolist(),
            people=split["subjects"],
            pairs_per_person=45 * len(test_ids),
            correct_pair_assignments=counts,
            repeat_minus_control=stats,
            class_pair_hits=per_object,
            advance=bool(advance),
        ),
    )
    print(counts, stats, "advance", advance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    defaults(parser)
    parser.add_argument(
        "--design", type=Path, default=RESULTS / "repeat_photo_subspace_design.json"
    )
    parser.add_argument("--split", type=Path, default=RESULTS / "shared_photo_eeg_design.json")
    parser.add_argument(
        "--source", type=Path, default=Path("outputs/official-object-readout/sub-03")
    )
    parser.add_argument("--release", type=Path, default=RESULTS / "repeat_photo_subspace.json")
    run(parser.parse_args())
