"""Extend the registered personal R4 Fisher metric from 128 to 512 non-panel concepts."""

import argparse
import hashlib
import itertools
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from audit_registered_sessions import audit_panel
from decode_eeg_layout import CHANNELS, eeg_design
from fit_personal_session_metric import evaluate, load, object_directions
from prepare_registered_subjects import (
    OFFSET,
    REPO,
    REV,
    ROW_BYTES,
    SHAPE,
    SIZE,
    axes,
    digest,
    http,
    registered_rows,
    store_json,
)

SEED = 260508
PANELS = ("registered", "replicate")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def check(valid, msg):
    if not valid:
        raise ValueError(msg)


def extract(mean):
    return eeg_design(mean, np.arange(len(mean)), np.arange(17))


def predict(train, queries, rows, refs):
    x, q = train.astype(np.float64), queries.astype(np.float64)
    mu, sd = x.mean(0), x.std(0).clip(1e-4)
    grouped = ((x - mu) / sd).reshape(-1, 10, x.shape[1])
    centers = grouped.mean(1)
    residual = (grouped - centers[:, None]).reshape(x.shape)
    basis = object_directions(
        centers.T @ centers / len(centers), residual.T @ residual / len(x), 96
    )
    projected = ((q - mu) / sd) @ basis
    return {p: evaluate(ref, projected, {}, rows)["personal_r4"] for p, ref in refs.items()}


def design_block(raw, post):
    check(len(raw) == 10 * ROW_BYTES, "Truncated physical EEG block")
    x = np.frombuffer(raw, dtype="<f8").reshape(10, *SHAPE[1:])
    check(np.isfinite(x).all(), "Nonfinite source EEG")
    return extract(np.take(x, post, axis=2).astype(np.float32).mean(axis=1, dtype=np.float32))


def training_block(concept, out, cache, url, post, pinned_hash=None):
    start = 10 * int(concept)
    first, last = OFFSET + start * ROW_BYTES, OFFSET + (start + 10) * ROW_BYTES - 1
    key = f"{first}-{last}"
    dest = out / f"{concept:04d}.npz"
    if dest.exists():
        with np.load(dest, allow_pickle=False) as z:
            features = z["features"]
            if (
                str(z["source"]) == url
                and int(z["start"]) == start
                and features.shape == (10, 1003)
                and features.dtype == np.float32
                and str(z["feature_sha"]) == digest(features.tobytes())
                and (pinned_hash is None or str(z["raw_sha"]) == pinned_hash)
            ):
                return features, str(z["raw_sha"])
    physical = cache / (key + ".bin")
    if physical.is_file():
        raw = physical.read_bytes()
        check(
            digest(raw) == (cache / (key + ".sha256")).read_text().strip(),
            "Cached range SHA mismatch",
        )
    else:
        check(pinned_hash is None, "Missing previously audited physical EEG range")
        raw = http(
            url + "?range=" + key,
            {"Range": "bytes=" + key},
            last - first + 1,
            f"bytes {key}/{SIZE}",
        )
    raw_sha = digest(raw)
    check(
        pinned_hash is None or raw_sha == pinned_hash,
        "Source differs from historical 128-class receipt",
    )
    features = design_block(raw, post)
    tmp = dest.with_suffix(".partial.npz")
    np.savez(
        tmp,
        features=features,
        feature_sha=digest(features.tobytes()),
        raw_sha=raw_sha,
        start=start,
        source=url,
    )
    tmp.replace(dest)
    return features, raw_sha


def paired_audit(sessions, ref, old, new):
    key0, key1 = "personal_r4", "expanded_r4"
    rec = {**ref, "results": {key0: old, key1: new}}
    audit = audit_panel(sessions, rec)
    truth = np.asarray(ref["truth"])
    rows = np.asarray(ref["rows"])
    a, b = np.asarray(old["predicted"]), np.asarray(new["predicted"])
    methods = {}
    for k, v in audit["methods"].items():
        methods[k] = {
            f: v[f]
            for f in (
                "correct",
                "n",
                "conditional_expected_hits",
                "conditional_expected_accuracy",
                "accuracy_minus_conditional_expectation",
                "permutation_one_sided_p",
            )
        }
    groups = {}
    for k in range(12):
        cid = np.unique(rows[truth == k] // 10).item()
        pair = tuple(sessions["session_pairs"][cid])
        groups.setdefault(pair, []).append(k)
    delta = np.zeros((12, 12), int)
    np.add.at(delta, (truth, b), 1)
    np.add.at(delta, (truth, a), -1)
    hist = Counter({0: 1})
    for group in groups.values():
        local = Counter(int(delta[group, list(p)].sum()) for p in itertools.permutations(group))
        total = Counter()
        for i, n in hist.items():
            for j, m in local.items():
                total[i + j] += n * m
        hist = total
    denom = sum(hist.values())
    expectation = sum(k * v for k, v in hist.items()) / denom
    observed = int(np.trace(delta))
    check(
        np.isclose(
            expectation,
            methods[key1]["conditional_expected_hits"] - methods[key0]["conditional_expected_hits"],
        ),
        "Conditional expectation of paired change disagrees",
    )
    null_p = (
        sum(
            v
            for k, v in hist.items()
            if abs(k - expectation) >= abs(observed - expectation) - 1e-12
        )
        / denom
    )
    d = (b == truth).astype(float) - (a == truth)
    per_class = np.array([d[truth == k].mean() for k in range(12)])
    excess = np.subtract(*(audit["methods"][m]["excess_by_word"] for m in (key1, key0)))
    boot = np.random.default_rng(SEED).integers(12, size=(10000, 12))
    return dict(
        methods=methods,
        gains=int(((b == truth) & (a != truth)).sum()),
        losses=int(((b != truth) & (a == truth)).sum()),
        delta_hits=observed,
        conditional_delta_expected_hits=expectation,
        delta_excess_hits=observed - expectation,
        delta_accuracy_ci95=np.quantile(per_class[boot].mean(1), [0.025, 0.975]).tolist(),
        delta_excess_rate_ci95=np.quantile(excess[boot].mean(1), [0.025, 0.975]).tolist(),
        exact_paired_two_sided_p=null_p,
        within_pair_whole_class_permutations=denom,
    )


def subject_run(sub, args, refs, panel_rows, old_classes, extra):
    source = read(args.inputs / f"{sub}.json")
    url = f"{REPO}/resolve/{REV}/Preprocessed_data_250Hz/{sub}/preprocessed_eeg_training.npy"
    check(
        source["source"] == url
        and source["revision"] == REV
        and source["source_size"] == SIZE
        and source["subject"] == sub,
        "Incorrect archive source",
    )
    cache = args.inputs / "ranges" / REV / sub
    tail = OFFSET + SHAPE[0] * ROW_BYTES
    channels, times = axes(
        (cache / f"0-{OFFSET - 1}.bin").read_bytes(),
        (cache / f"{tail}-{SIZE - 1}.bin").read_bytes(),
    )
    check(
        np.allclose(times[50:], np.arange(250) / 250, rtol=0, atol=1e-7),
        "Wrong poststimulus time axis",
    )
    post = [channels.tolist().index(c) for c in CHANNELS]
    work = args.out / "features" / sub
    work.mkdir(parents=True, exist_ok=True)
    generic = args.inputs / "generic" / f"{sub}.npz"
    target = args.inputs / "halves" / f"{sub}.npz"
    old_rows, _, _, x0 = load(generic)
    target_rows, _, _, test = load(target)
    check(
        np.array_equal(old_rows, (old_classes[:, None] * 10 + np.arange(10)).ravel())
        and np.array_equal(target_rows, panel_rows),
        "Historical generic or query rows differ",
    )
    old_features, q = extract(x0), extract(test)
    baseline = predict(old_features, q, target_rows, refs)
    hist_receipt = read(generic.with_suffix(".json"))
    check(
        hist_receipt["packet_sha256"] == sha(generic)
        and hist_receipt["source_full_lfs_sha256"] == source["source_full_lfs_sha256"],
        "Generic packet not tied to same physical recordings",
    )
    old_refs = {}
    for p in PANELS:
        prior = args.inputs / "personal" / p / sub / "evaluation.json"
        old_refs[p] = read(prior)
        check(
            all(old_refs[p][k] == refs[p][k] for k in ("rows", "truth", "words", "folds"))
            and baseline[p] == old_refs[p]["results"]["personal_r4"],
            "128-class reference failed exact replay",
        )
    # Verify the legacy design physically rather than trusting features extracted from the old packet.
    original_hash = {rec["rows"][0] // 10: rec["sha256"] for rec in hist_receipt["row_ranges"]}
    all_x = {}
    sourced = {}
    for k, c in enumerate(old_classes):
        feats, h = training_block(c, work, cache, url, post, original_hash[int(c)])
        check(
            np.array_equal(feats, old_features[10 * k : 10 * k + 10]),
            "Raw 63-channel range differs from historical design",
        )
        all_x[int(c)] = feats
        sourced[int(c)] = h
    print(
        sub,
        "128-class parity 240/240; all physical design values verified. Fetching 384 nonpanel concepts.",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(training_block, c, work, cache, url, post): int(c) for c in extra}
        for i, future in enumerate(as_completed(tasks), 1):
            c = tasks[future]
            all_x[c], sourced[c] = future.result()
            if i % 32 == 0:
                print(sub, i, "/384 ready", flush=True)
    classes = np.sort(np.r_[old_classes, extra])
    features = np.concatenate([all_x[c] for c in classes])
    pred = predict(features, q, target_rows, refs)
    info = args.out / "features" / f"{sub}.json"
    store_json(
        info,
        dict(
            subject=sub,
            source=url,
            source_full_lfs_sha256=source["source_full_lfs_sha256"],
            selection_seed=SEED,
            training_concepts=classes.tolist(),
            original_concepts=old_classes.tolist(),
            added_concepts=extra.tolist(),
            design=[
                "17 posterior named channels",
                "425 full-epoch + 408 early + 170 local power features",
            ],
            train_design_sha256=digest(features.tobytes()),
            raw_block_sha256={str(k): v for k, v in sorted(sourced.items())},
            old_240_predictions_equal=True,
            old_128_design_replay_exact=True,
        ),
    )
    sess_path = args.inputs / "sessions.json"
    sessions = read(sess_path)["subjects"][sub]
    for p in PANELS:
        dest = args.out / p / sub
        dest.mkdir(parents=True, exist_ok=True)
        audit = paired_audit(sessions, refs[p], baseline[p], pred[p])
        data = {k: refs[p][k] for k in ("rows", "words", "truth", "folds")}
        data.update(
            subject=sub,
            results={"expanded_r4": pred[p]},
            generic_classes=classes.tolist(),
            training_receipt_sha256=sha(info),
            target_sha256=sha(target),
            reference_128_sha256=sha(args.inputs / "personal" / p / sub / "evaluation.json"),
            session_audit=audit,
            sessions_sha256=sha(sess_path),
        )
        store_json(dest / "evaluation.json", data)
        print(sub, p, "READY", baseline[p]["hits"], "->", pred[p]["hits"], "of 120", flush=True)


def run(args):
    refs = {p: read(args.references / f"{p}_ablation.json") for p in PANELS}
    args.registered = args.references / "registered_ablation.json"
    args.replication = args.references / "replicate_ablation.json"
    target, _, _ = registered_rows(args)
    exclude = np.unique(target // 10)
    eligible = np.setdiff1d(np.arange(1654), exclude)
    original = np.sort(np.random.default_rng(260431).choice(eligible, 128, replace=False))
    extra = np.sort(
        np.random.default_rng(SEED).choice(np.setdiff1d(eligible, original), 384, replace=False)
    )
    check(
        len(np.intersect1d(np.r_[original, extra], exclude)) == 0
        and len(np.unique(np.r_[original, extra])) == 512,
        "Invalid calibration expansion",
    )
    with ThreadPoolExecutor(max_workers=len(args.subjects)) as pool:
        cases = [
            pool.submit(subject_run, s, args, refs, target, original, extra) for s in args.subjects
        ]
        for case in as_completed(cases):
            case.result()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=Path("outputs/object-transfer"))
    p.add_argument("--out", type=Path, default=Path("outputs/photo-structure/expanded"))
    p.add_argument("--references", type=Path, default=Path("outputs/object-fidelity"))
    p.add_argument(
        "--train-rows",
        type=Path,
        default=Path("outputs/official-object-readout/sub-03/atm_train_rows.csv"),
    )
    p.add_argument(
        "--subjects", nargs="+", choices=("sub-04", "sub-05"), default=["sub-04", "sub-05"]
    )
    p.add_argument("--workers", type=int, choices=range(1, 6), default=5)
    a = p.parse_args()
    if len(set(a.subjects)) != len(a.subjects):
        p.error("Repeated participant")
    run(a)
