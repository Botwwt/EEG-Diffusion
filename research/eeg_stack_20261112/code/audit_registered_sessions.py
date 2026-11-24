"""Reconstruct THINGS-EEG2 session pairs and audit already frozen registered predictions.

python audit_registered_sessions.py --out outputs/object-transfer
Add --runs DIR for registered/sub-XX/evaluation.json and replicate/sub-XX/evaluation.json.
Add --source-exports DIR for the two flat sub-03 ablation exports.
"""

import argparse
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from scipy.io import loadmat

REPO = "gifale95/eeg_encoding"
REV = "101997d4f14c41b615bcaf8c0907740106d260c5"
TRAIN = "00_data_collection/train"
RAW = f"https://raw.githubusercontent.com/{REPO}/{REV}"
METHODS = ("raw_eeg", "projected_eeg", "visual_h14", "fused")
ORDER_SCRIPT_BLOB = "232a2a1e8cc9cab7a99829df1369e4d0f67b4702"
# Git tree entries for participants 1..10 (rows), sessions 1..4 (columns).
# Git blob IDs verify cached and freshly downloaded bytes without GitHub API calls.
ORDER_BLOBS = """
4f5a6dca370014ce88e9bde7c0f706833b036b2c d3f6eb3ef1fe3a58e869020f018034941b62ff85 56e220d27da36eb932a7accbf60d959f29ad2245 45b16538743213fd1090289568878852167d8afd
0dc031400ca806c2e5f36625e58d064bbbcd3bef 4a6a0fbc5de134280fe7707bd5e3e5aaca6a2c99 5dc1a96dc0fdcc4443fe7a4013eb5ef02b51d9cf 9fc9ddd64a7c5f4db75dbf87ac3fa5a543808168
ae7ec110142eda68bd0ab42c0874ed1d5a139ca7 c68de9fd4532d293148ba6600b2581551cedae34 6dddaf2d792050432a5c449b8d361dc89e10d9aa 865e42e7302846ca21474dc5f7424363002f7f06
9c630000af38e371c5ba3e3fad7e82a8eafdb8c1 4766a143e3d05181739873064dfa2ffc0d681388 b11e59c5f6254a29c4e00ad0daebeea1c74a2fee f0ce3f80ccba1f0d253df2a7b8c43a83b23a5557
eafbb5ab5d02643831f4cbe8bc84ecbf6bf54f3e 65ecd38a15786ba036524516129078b240a4031d 07ee5310e5f1465d5a0b739dbdf29f2139e401b8 510a62f7c0a391b92585344453d915a0995a215b
f553cd9ecec64a49a5dfff0ab9f60a9887cfb90b 247a48f3116922f1db863a2f021de6cec0de8c12 d183d43e4854906a341491e2615dd66417b6cbe0 9962c15a16db6a6ede3016a803cde5864bee8ba8
54525d2f732d69ce2649b23c5ade45d2497aa649 7229cf7886c66a5e14c56423f9d97868c4c3f44b 49fb55af65f00dbb6840a239d20683736f50083b 17cbd4f3927c2be777b6a830c303bb48aca1c104
f143994874e487400bff4dea5f2a1ad46d2c362e 8dc8a5f65a6c33b6152306c5964860aa37ed4b99 e0b53b560af2ea8ee511197729084e306d578533 9e55c5b74b301e0ca1c238c3bfd9f59a22a2c54f
9ecd4b8a2cc0b5f35a0dc30c658122c1c54e5134 851ad0dd614bc44ad77f7a1581716d907fddff47 a188af03015a1c77bae6f1716945d613f02c5f65 5d39056d97e2f2e932fd1789a0f30c771a180654
fe1d5f0d21c31cb914c3631d979f5259ce4d7508 58b3833fe044c62dc120727583430de20363400c d9651d34ebbe975adf532b59179f64511667768e 350687404c609d8847bd744904b36780563ff340
""".split()


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, data):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(path)


def frozen_asset(path, blob_id, cache):
    local = cache / Path(path).name
    raw = local.read_bytes() if local.exists() else None

    def valid(data):
        return (
            data is not None
            and hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == blob_id
        )

    if not valid(raw):
        with urlopen(f"{RAW}/{path}", timeout=30) as response:
            raw = response.read(128_000)
        if not valid(raw):
            raise ValueError(f"Unexpected Git blob for {path}")
        local.write_bytes(raw)
    return raw, {"path": path, "bytes": len(raw), "git_blob": blob_id, "sha256": sha256(raw)}


def sessions(out):
    cache = out / "session_order_assets" / REV
    cache.mkdir(parents=True, exist_ok=True)
    script = f"{TRAIN}/defineStimuliOrder.m"
    _, source = frozen_asset(script, ORDER_SCRIPT_BLOB, cache)

    def load(item):
        person, session = item
        path = f"{TRAIN}/train_order/stimOrder_sub-{person}_sess-{session}.mat"
        raw, receipt = frozen_asset(path, ORDER_BLOBS[4 * (person - 1) + session - 1], cache)
        order = loadmat(io.BytesIO(raw), variable_names=["stimOrder"])["stimOrder"]
        if (
            order.shape != (20, 56, 3, 5)
            or order.dtype.kind not in "iuf"
            or not np.isfinite(order).all()
            or np.any(order != np.floor(order))
            or order.min() != 0
            or order.max() > 16540
        ):
            raise ValueError(f"Unexpected stimulus order for {path}")
        count = np.bincount(order.astype(int).ravel(), minlength=16541)
        if (
            int(count[0]) != 90
            or np.sum(count[1:] > 0) != 8270
            or np.min(count[1:][count[1:] > 0]) < 2
        ):
            raise ValueError(f"Unexpected planned catches, repetitions or session size for {path}")
        return person, session, count[1:] > 0, receipt

    membership = np.zeros((10, 16540, 4), dtype=bool)
    sources = []
    with ThreadPoolExecutor(8) as pool:
        for person, session, present, receipt in pool.map(
            load, [(p, s) for p in range(1, 11) for s in range(1, 5)]
        ):
            membership[person - 1, :, session - 1] = present
            sources.append(receipt)
    if not np.all(membership.sum(2) == 2):
        raise ValueError("A photo does not occur in exactly two planned sessions")
    groups = membership.reshape(10, 1654, 10, 4)
    if not np.all(groups == groups[:, :, :1, :]):
        raise ValueError("A concept's ten photographs have different session pairs")
    result = {}
    for p, present in enumerate(groups[:, :, 0, :], 1):
        pairs = np.where(present)[1].reshape(1654, 2) + 1
        unique, ns = np.unique(pairs, axis=0, return_counts=True)
        result[f"sub-{p:02}"] = {
            "session_pairs": pairs.tolist(),
            "counts_by_pair": {
                "-".join(map(str, pair)): int(n) for pair, n in zip(unique, ns, strict=True)
            },
        }
    obj = {
        "revision": REV,
        "repository": "https://github.com/" + REPO,
        "order_source": source,
        "concept_index": "zero-based original training image array_row // 10",
        "session_index": "one-based planned experiment session from the archived presentation order",
        "subjects": result,
        "verified_order_files": sources,
    }
    save(out / "sessions.json", obj)
    print(f"Verified {len(sources)} stimulus orders; wrote {out / 'sessions.json'}", flush=True)
    return obj


def conditional_null(truth, pred, cluster):
    """Count every within-pair permutation of whole category identities exactly."""
    matrix = np.zeros((12, 12), dtype=int)
    np.add.at(matrix, (truth, pred), 1)
    histogram = np.ones(1, dtype=np.int64)
    for cl in cluster:
        size = len(cl)
        # A state records labels already assigned within this pair and the
        # number of permutations attaining each hit count for that assignment.
        limit = int(np.isin(truth, cl).sum())
        states = np.zeros((1 << size, limit + 1), dtype=np.int64)
        states[0, 0] = 1
        for mask in range((1 << size) - 1):
            source = cl[mask.bit_count()]
            for j, target in enumerate(cl):
                if mask & (1 << j):
                    continue
                hits = matrix[source, target]
                states[mask | (1 << j), hits:] += states[mask, : limit + 1 - hits]
        histogram = np.convolve(histogram, states[-1])
    return histogram


def audit_panel(participant, record):
    rows = np.asarray(record["rows"], dtype=int)
    truth = np.asarray(record["truth"], dtype=int)
    unique = np.unique(truth)
    if (
        rows.shape != (120,)
        or truth.shape != rows.shape
        or len(np.unique(rows)) != 120
        or rows.min() < 0
        or rows.max() >= 16540
        or not np.array_equal(unique, np.arange(12))
        or not np.all(np.bincount(truth) == 10)
    ):
        raise ValueError("Expected 12 balanced objects and 120 distinct training stimuli")
    category = rows // 10
    concept = np.array([np.unique(category[truth == k]).item() for k in unique])
    if len(np.unique(concept)) != 12:
        raise ValueError("Frozen classifier labels are inconsistent with original training order")
    pair = np.asarray(participant["session_pairs"])[concept]
    groups, group = np.unique(pair, axis=0, return_inverse=True)
    cluster = [np.flatnonzero(group == g) for g in range(len(groups))]
    names = ["-".join(map(str, pr)) for pr in groups]
    methods = {}
    for method in record["results"]:
        pred = np.asarray(record["results"][method]["predicted"], dtype=int)
        if pred.shape != rows.shape or not np.isin(pred, unique).all():
            raise ValueError(f"Invalid frozen predictions for {method}")
        correct = pred == truth
        if int(correct.sum()) != record["results"][method]["hits"]:
            raise ValueError(f"Recorded correct count disagrees for {method}")
        same_group = group[pred] == group[truth]
        group_size = np.array([len(cl) for cl in cluster])
        expected_row = same_group / group_size[group[truth]]
        obs, expected = float(correct.mean()), float(expected_row.mean())
        group_records = {}
        for i, cl in enumerate(cluster):
            ix = group[truth] == i
            group_records[names[i]] = dict(
                concepts=len(cl),
                photos=int(ix.sum()),
                correct=int(correct[ix].sum()),
                expected_hits=float(expected_row[ix].sum()),
                accuracy=float(correct[ix].mean()),
                expected_accuracy=float(expected_row[ix].mean()),
                excess=float(correct[ix].mean() - expected_row[ix].mean()),
            )
        hints = group_size[group[truth]] > 1
        null_hits = conditional_null(truth, pred, cluster)
        num_permutations = int(null_hits.sum())
        grid = np.arange(len(null_hits))
        if not np.isclose(grid @ null_hits / num_permutations, expected_row.sum()):
            raise ValueError("Analytic expected hits disagrees with exact conditional distribution")
        interval = np.searchsorted(np.cumsum(null_hits) / num_permutations, [0.025, 0.975])
        methods[method] = dict(
            correct=int(correct.sum()),
            n=len(rows),
            accuracy=obs,
            predicted_pair_correct=int(same_group.sum()),
            conditional_expected_hits=float(expected_row.sum()),
            conditional_expected_accuracy=expected,
            accuracy_minus_conditional_expectation=obs - expected,
            excess_by_word=[float(np.mean((correct - expected_row)[truth == k])) for k in unique],
            macro_pair_excess=float(np.mean([v["excess"] for v in group_records.values()])),
            nonsingleton_rows=int(hints.sum()),
            nonsingleton_excess=float((correct[hints] - expected_row[hints]).mean()),
            permutation_null_hits_ci95=interval.tolist(),
            permutation_one_sided_p=float(null_hits[correct.sum() :].sum() / num_permutations),
            exact_permutation_count=num_permutations,
            by_session_pair=group_records,
        )
    return dict(
        n_photos=120,
        n_objects=12,
        words=record["words"],
        uniform_12way_accuracy=1 / 12,
        uniform_12way_expected_hits=10,
        pair_only_oracle_accuracy=len(groups) / 12,
        pair_only_oracle_expected_hits=10 * len(groups),
        by_session_pair={
            name: [record["words"][k] for k in cl] for name, cl in zip(names, cluster, strict=True)
        },
        null_distribution="Exact over all whole-category permutations within each occupied session pair",
        methods=methods,
    )


def evaluate(args, info):
    results = {name: {} for name in ("registered", "replicate")}
    for panel, subjects in results.items():
        paths = {}
        if args.source_exports:
            paths["sub-03"] = args.source_exports / (panel + "_ablation.json")
        if args.runs:
            for path in sorted((args.runs / panel).glob("sub-*/evaluation.json")):
                participant = path.parent.name
                if participant in paths:
                    original = json.loads(paths[participant].read_bytes())
                    replay = json.loads(path.read_bytes())
                    if any(
                        original[k] != replay[k] for k in ("rows", "truth", "folds", "words")
                    ) or any(
                        original["results"][k]["predicted"] != replay["results"][k]["predicted"]
                        for k in METHODS
                    ):
                        raise ValueError("The source ablation and source transfer replay disagree")
                paths[participant] = path
        if not paths:
            raise ValueError(f"No frozen predictions supplied for {panel}")
        for sub, path in sorted(paths.items()):
            raw = path.read_bytes()
            rec = json.loads(raw)
            if sub != rec.get("subject", sub):
                raise ValueError("Prediction participant ID disagrees with path")
            result = audit_panel(info["subjects"][sub], rec)
            result["predictions_sha256"] = sha256(raw)
            subjects[sub] = result
    new_people = {}
    for panel, subjects in results.items():
        ids = [k for k in subjects if k != "sub-03"]
        if not ids:
            continue
        means = {}
        methods = list(subjects[ids[0]]["methods"])
        reference_words = subjects[ids[0]]["words"]
        if any(
            set(subjects[k]["methods"]) != set(methods) or subjects[k]["words"] != reference_words
            for k in ids
        ):
            raise ValueError("Compared participants have different methods or word order")
        rng = np.random.default_rng(260429)
        bootstrap_subjects = rng.integers(len(ids), size=(10000, len(ids)))
        bootstrap_words = rng.integers(12, size=(10000, 12))
        for method in methods:
            fields = (
                "accuracy",
                "conditional_expected_accuracy",
                "accuracy_minus_conditional_expectation",
                "macro_pair_excess",
            )
            means[method] = {
                f: float(np.mean([subjects[k]["methods"][method][f] for k in ids])) for f in fields
            }
            by_word = np.array([subjects[k]["methods"][method]["excess_by_word"] for k in ids])
            boot = by_word[bootstrap_subjects[:, :, None], bootstrap_words[:, None, :]].mean((1, 2))
            means[method]["crossed_bootstrap_excess_ci95"] = np.quantile(
                boot, [0.025, 0.975]
            ).tolist()
        new_people[panel] = dict(
            subjects=ids,
            participant_means=means,
            mean_pair_only_oracle_accuracy=float(
                np.mean([subjects[k]["pair_only_oracle_accuracy"] for k in ids])
            ),
        )
    result = dict(
        sessions_sha256=sha256((args.out / "sessions.json").read_bytes()),
        conditioning="Keep EEG predictions fixed. Permute category identities within each participant's planned session pair, keeping ten photos together.",
        exact_expectation="A prediction in the true session pair matches with probability 1 / size of that pair's closed class set. A prediction outside it earns zero.",
        oracle="Optimal expected closed-set accuracy from the known session pair alone under the panel's equal category frequencies.",
        weighting="Headline excess weights each photo equally. Macro pair excess gives each occupied pair equal weight. Participant means exclude sub-03.",
        panels=results,
        new_participants=new_people,
    )
    report = args.report or (args.out / "registered_session_audit.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    save(report, result)
    print(f"Audited frozen predictions; wrote {report}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/object-transfer"))
    p.add_argument("--runs", type=Path)
    p.add_argument("--source-exports", type=Path)
    p.add_argument(
        "--report",
        type=Path,
        help="Prediction audit JSON, defaults to --out/registered_session_audit.json",
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    info = sessions(args.out)
    if args.runs or args.source_exports:
        evaluate(args, info)


if __name__ == "__main__":
    main()
