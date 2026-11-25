"""Test products of independent repeated EEG trials in a same-session object menu."""

import argparse
from pathlib import Path

import numpy as np
from decode_eeg_layout import CHANNELS
from fit_full_head_objects import fisher
from fit_personal_session_metric import similarities
from prepare_expanded_personal import design_block
from prepare_registered_subjects import OFFSET, REV, ROW_BYTES, SHAPE, SIZE, axes, digest
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import check

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
DESIGN = PUBLIC / "repeat_cross_moment_design.json"
METHODS = ("fisher", "distinct_trials", "mean_products", "distinct_only", "wrong_query")


def moments(trials):
    """Upper triangle of the cross-trial and R4-mean spatial Gram matrices."""
    x = trials[:, :, :, 10:130].reshape(-1, 4, 17, 6, 20).transpose(0, 3, 1, 2, 4)
    x = x - x.mean(axis=-1, keepdims=True)
    summed = x.sum(axis=2)
    outer = summed @ summed.swapaxes(-1, -2)
    self_products = (x @ x.swapaxes(-1, -2)).sum(axis=2)
    i, j = np.triu_indices(17)
    denominator = x.shape[-1]
    return [
        v[..., i, j].reshape(len(x), -1)
        for v in ((outer - self_products) / (12 * denominator), outer / (16 * denominator))
    ]


def features(a, subject):
    cache = a.raw / REV / subject
    head = (cache / f"0-{OFFSET - 1}.bin").read_bytes()
    tail = (cache / f"{OFFSET + SHAPE[0] * ROW_BYTES}-{SIZE - 1}.bin").read_bytes()
    labels, _ = axes(head, tail)
    electrodes = [labels.tolist().index(x) for x in CHANNELS]
    source = a.expanded / (subject + ".json")
    meta = read(source)
    ids = np.sort(np.asarray(meta["original_concepts"], dtype=np.int64))
    check(len(np.unique(ids)) == len(ids) == 128, "Unexpected original calibration objects")
    original, cross, average = [], [], []
    for cid in ids:
        lo = OFFSET + 10 * int(cid) * ROW_BYTES
        raw = (cache / f"{lo}-{lo + 10 * ROW_BYTES - 1}.bin").read_bytes()
        check(digest(raw) == meta["raw_block_sha256"][str(cid)], "EEG source block differs")
        baseline = design_block(raw, electrodes)
        with np.load(a.expanded / subject / f"{cid:04d}.npz", allow_pickle=False) as z:
            check(np.array_equal(baseline, z["features"]), "Historical R4 features differ")
        physical = np.frombuffer(raw, dtype="<f8").reshape(10, *SHAPE[1:])
        c, m = moments(np.take(physical, electrodes, axis=2))
        original.append(baseline)
        cross.append(c)
        average.append(m)
    arrays = [np.asarray(v, dtype=np.float64) for v in (original, cross, average)]
    return ids, arrays, sha(source)


def projection(features, fit, test, rank):
    training = features[fit].reshape(-1, features.shape[-1])
    held = features[test].reshape(-1, features.shape[-1])
    return fisher(training, held, rank).reshape(len(test), 10, rank)


def validate_subject(a, subject, subject_index, schedule, seed):
    ids, arrays, receipt = features(a, subject)
    times = np.array(schedule["subjects"][subject]["session_pairs"])[ids]
    cells, groups = np.unique(times, axis=0, return_inverse=True)
    check(len(cells) == 4, "Expected four recording-session pairs")
    predictions = {k: np.full((128, 10), -1, np.int64) for k in METHODS}
    rng = np.random.default_rng(seed + subject_index)
    order = np.stack([rng.permutation(10) for _ in ids])
    fold_counts = []
    for cell, pair in enumerate(cells):
        held = np.flatnonzero(groups == cell)
        fit = np.flatnonzero(groups != cell)
        check(len(held) >= 12 and not np.intersect1d(held, fit).size, "Invalid held-out menu")
        projected = [
            projection(x, fit, held, rank) for x, rank in zip(arrays, (96, 48, 48), strict=True)
        ]
        fold_counts.append(dict(session_pair=pair.tolist(), fit=len(fit), test=len(held)))
        for fold in range(5):
            slots = np.arange(10) // 2 == fold
            support = order[held][:, ~slots]
            for local, focal in enumerate(held):
                peers = np.delete(np.arange(len(held)), local)
                menu = rng.permutation(np.r_[local, rng.choice(peers, 11, replace=False)])
                other = int(rng.choice(menu[menu != local]))
                query_slots = order[focal, slots]
                check(
                    not np.intersect1d(support[local], query_slots).size, "Query entered supports"
                )
                sims = []
                for v in projected:
                    bank = v[menu[:, None], support[menu]]
                    query = np.concatenate(
                        (v[local, query_slots], v[other, order[held[other], slots]])
                    )
                    sims.append(similarities(query, bank))
                base, distinct, averaged = sims
                mixed = 0.75 * base + 0.25 * distinct
                scores = (
                    base[:2],
                    mixed[:2],
                    (0.75 * base + 0.25 * averaged)[:2],
                    distinct[:2],
                    mixed[2:],
                )
                for key, s in zip(METHODS, scores, strict=True):
                    predictions[key][focal, query_slots] = ids[held[menu[s.argmax(axis=1)]]]
    check(all((v >= 0).all() for v in predictions.values()), "Incomplete subject evaluation")
    a.out.mkdir(parents=True, exist_ok=True)
    file = a.out / (subject + ".npz")
    np.savez(file, ids=ids, **predictions)
    counts = {key: (v == ids[:, None]).sum(axis=1) for key, v in predictions.items()}
    print(subject, {key: int(v.sum()) for key, v in counts.items()}, flush=True)
    return (
        ids,
        counts,
        dict(source_receipt_sha256=receipt, predictions_sha256=sha(file), folds=fold_counts),
    )


def run(a):
    spec, schedule = read(DESIGN), read(a.sessions)
    data, receipts, objects = [], {}, None
    for i, subject in enumerate(spec["participants"]):
        ids, counts, receipts[subject] = validate_subject(a, subject, i, schedule, spec["seed"])
        check(objects is None or np.array_equal(ids, objects), "Different object population")
        objects = ids
        data.append(counts)
    pooled = {k: sum(part[k] for part in data) for k in METHODS}
    boot = np.random.default_rng(spec["seed"] + 50).integers(128, size=(10000, 128))
    comparisons = {}
    for k in ("fisher", "mean_products", "wrong_query"):
        delta = (pooled["distinct_trials"] - pooled[k]) / 30
        comparisons[k] = dict(
            delta=float(delta.mean()),
            ci95=np.quantile(delta[boot].mean(axis=1), [0.025, 0.975]).tolist(),
        )
    passed = comparisons["fisher"]["ci95"][0] > 0 and comparisons["mean_products"]["delta"] > 0
    report = dict(
        design_sha256=sha(DESIGN),
        source_sha256=sha(Path(__file__)),
        schedule_sha256=sha(a.sessions),
        participants=spec["participants"],
        unique_objects=128,
        unique_photographs=1280,
        queries=3840,
        counts_by_person={
            s: {k: int(v.sum()) for k, v in p.items()}
            for s, p in zip(spec["participants"], data, strict=True)
        },
        hits={k: int(v.sum()) for k, v in pooled.items()},
        distinct_trials_minus=comparisons,
        advance=bool(passed),
        receipts=receipts,
    )
    freeze(a.out / "summary.json", report)
    if a.publish:
        freeze(PUBLIC / "repeat_cross_moments.json", report)
    print(report["hits"], comparisons, "advance", passed, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/repeat-cross-moments"))
    p.add_argument(
        "--expanded", type=Path, default=Path("outputs/photo-structure/expanded/features")
    )
    p.add_argument("--raw", type=Path, default=Path("outputs/object-transfer/ranges"))
    p.add_argument("--sessions", type=Path, default=Path("outputs/object-transfer/sessions.json"))
    p.add_argument("--publish", action="store_true")
    run(p.parse_args())
