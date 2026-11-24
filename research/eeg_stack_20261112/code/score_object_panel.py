"""Score EEG reconstructions on object identity and exact-photo retrieval.

The input matrix contains generated-query by original-photo CLIP similarities.
For each query, exclude its photo ordinal from *every* candidate object, then
average the nine remaining similarities per object. This equals a dot product
with the arithmetic centroid of the nine unit reference embeddings; centroid
norms cannot be reconstructed from the cross-similarity matrix alone.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

METRICS = ("object_top1", "photo_top1", "intra10_top1")


def _top1(scores, positives):
    correct = scores[np.arange(len(positives)), positives]
    best = scores.max(axis=1)
    tied = (scores == best[:, None]).sum(axis=1)
    return np.where(correct == best, 1.0 / tied, 0.0)


def score_object_panel(scores, rows, concept_ids, ordinals=None):
    """Return JSON-ready hit rates, query-level credits, ranks, and concept means.

    rows indexes both axes in the same order. With no explicit ordinals, rank
    the ten original manifest row IDs within each concept. Ties split credit.
    """
    scores = np.asarray(scores)
    rows = np.asarray(rows)
    labels = np.asarray(concept_ids)
    if (
        rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or len(np.unique(rows)) != len(rows)
        or labels.shape != rows.shape
        or scores.shape != (len(rows), len(rows))
        or not np.isfinite(scores).all()
    ):
        raise ValueError(
            "Expected unique integer row IDs, aligned concepts and a finite square score matrix"
        )
    classes, group, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if len(classes) < 2 or not np.all(counts == 10):
        raise ValueError("Each of at least two concepts must contain exactly ten photos")
    positions = [np.flatnonzero(group == j) for j in range(len(classes))]
    if ordinals is None:
        ordinal = np.empty(len(rows), dtype=np.int64)
        for idx in positions:
            ordinal[idx[np.argsort(rows[idx])]] = np.arange(10)
    else:
        ordinal = np.asarray(ordinals)
        if ordinal.shape != rows.shape or ordinal.dtype.kind not in "iu":
            raise ValueError("Expected one integer photo ordinal per row")
        expected = np.sort(ordinal[positions[0]])
        if len(np.unique(expected)) != 10 or any(
            not np.array_equal(np.sort(ordinal[idx]), expected) for idx in positions
        ):
            raise ValueError("Each concept must supply the same ten distinct photo ordinals")

    same = group[:, None] == group[None, :]
    within = np.where(same, scores, -np.inf)
    prototype = np.empty((len(rows), len(classes)), dtype=np.float64)
    for j, idx in enumerate(positions):
        allowed = ordinal[:, None] != ordinal[idx][None, :]
        prototype[:, j] = np.where(allowed, scores[:, idx], 0.0).sum(axis=1, dtype=np.float64) / 9
    index = np.arange(len(rows))
    credits = dict(
        object_top1=_top1(prototype, group),
        photo_top1=_top1(scores, index),
        intra10_top1=_top1(within, index),
    )
    # Stable rank and margin reproduce the archived raw within-concept readout.
    # Use a float array so integer-valued synthetic matrices also support -inf.
    rivals = within.copy()
    np.fill_diagonal(rivals, -np.inf)
    margin = np.diag(scores) - rivals.max(axis=1)
    rank = (
        1
        + (within > np.diag(scores)[:, None]).sum(1)
        + ((within == np.diag(scores)[:, None]) & (index[None, :] < index[:, None])).sum(1)
    )
    return {
        "queries": len(rows),
        "n_concepts": len(classes),
        "examples_per_concept": 10,
        "mean": {k: float(v.mean()) for k, v in credits.items()},
        "hits": {k: float(v.sum()) for k, v in credits.items()},
        "chance": {
            "object_top1": 1 / len(classes),
            "photo_top1": 1 / len(rows),
            "intra10_top1": 0.1,
        },
        "per_concept": [
            dict(concept=classes[j].item(), **{k: float(v[idx].mean()) for k, v in credits.items()})
            for j, idx in enumerate(positions)
        ],
        "per_photo": [
            dict(
                row=int(rows[i]),
                concept=labels[i].item(),
                ordinal=int(ordinal[i]),
                intra10_rank=int(rank[i]),
                intra10_margin=float(margin[i]),
                **{k: float(v[i]) for k, v in credits.items()},
            )
            for i in index
        ],
    }


def paired_object_panel(baseline, candidate, seed=20261125, resamples=10000):
    """Candidate minus baseline, with participant and reference gallery fixed.

    Sample complete concepts with replacement and retain their ten paired
    photos together. Report rates in [0,1] and deltas in the same units.
    """

    def indexed(panel):
        photos = panel["per_photo"]
        result = {p["row"]: p for p in photos}
        if len(result) != len(photos):
            raise ValueError("Duplicate row in paired panel")
        return result

    a, b = indexed(baseline), indexed(candidate)
    if set(a) != set(b) or any(
        (a[i]["concept"], a[i]["ordinal"]) != (b[i]["concept"], b[i]["ordinal"]) for i in a
    ):
        raise ValueError("Paired panels must contain identical row, concept and ordinal identities")
    rows = sorted(a)
    labels = [a[i]["concept"] for i in rows]
    groups = [np.flatnonzero(np.asarray(labels) == c) for c in np.unique(labels)]
    bootstrap = np.random.default_rng(seed).integers(len(groups), size=(resamples, len(groups)))
    result = {}
    for metric in METRICS:
        d = np.array([b[i][metric] - a[i][metric] for i in rows])
        by_concept = np.array([d[g].mean() for g in groups])
        ci = np.quantile(by_concept[bootstrap].mean(axis=1), [0.025, 0.975])
        result[metric] = dict(
            delta=float(d.mean()),
            ci95=ci.tolist(),
            query_credits_gained=float(d[d > 0].sum()),
            query_credits_lost=float(-d[d < 0].sum()),
            concepts_improved=int((by_concept > 0).sum()),
            concepts_tied=int((by_concept == 0).sum()),
        )
    return result


def archived_pair(directory, baseline, candidate):
    def read(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))

    metadata = read("metadata.json")
    settings = read("settings.json")
    rows = np.asarray(metadata["rows"])
    with np.load(directory / "instance_starts.npz", allow_pickle=False) as starts:
        if not np.array_equal(rows, starts["rows"]) or not np.array_equal(rows, settings["rows"]):
            raise ValueError("Row order differs across settings, metadata and starts archive")
        concepts = starts["concept_ids"]
    starts_sha = hashlib.sha256((directory / "instance_starts.npz").read_bytes()).hexdigest()
    if starts_sha != settings["starts_sha256"]:
        raise ValueError("Starts file differs from the render registration")
    names = dict(
        zip(metadata["selected_concepts"], metadata["selected_concept_labels"], strict=True)
    )
    if not np.array_equal(rows // 10, concepts) or set(names) != set(concepts.tolist()):
        raise ValueError("Numerical concept IDs differ from the original row registration")
    labels = [names[int(c)] for c in concepts]
    images = metadata["image_ids"]
    if len(images) != len(rows) or len(set(images)) != len(rows):
        raise ValueError("Incomplete or duplicated stimulus IDs")
    order = []
    for i, image in enumerate(images):
        match = re.fullmatch(r"(.+)_(\d+)[a-z]*", image)
        if (
            not match
            or match.group(1) != labels[i].split("_", 1)[1]
            or int(match.group(2)) - 1 != rows[i] % 10
        ):
            raise ValueError(f"Photo identity or ordinal differs for row {rows[i]}")
        order.append(int(match.group(2)) - 1)
    reports = {}
    hashes = {}
    for arm in (baseline, candidate):
        if arm not in settings["arms"]:
            raise ValueError(f"Unregistered arm {arm}")
        path = directory / f"{arm}_scores.npy"
        hashes[arm] = hashlib.sha256(path.read_bytes()).hexdigest()
        report = score_object_panel(np.load(path, allow_pickle=False), rows, labels, order)
        prior = read(f"{arm}_ranks.json")["photos"]
        if len(prior) != len(rows):
            raise ValueError("Incomplete archived per-photo readout")
        for p, old in zip(report["per_photo"], prior, strict=True):
            if (
                p["row"] != old["row"]
                or p["concept"] != old["concept"]
                or p["intra10_rank"] != old["rank"]
                or p["intra10_top1"] != float(old["hit"])
                or not np.isclose(p["intra10_margin"], old["margin"], rtol=0, atol=1e-10)
            ):
                raise ValueError(
                    f"Matrix orientation or stimulus alignment disagrees for {arm}/{p['row']}"
                )
        reports[arm] = report
    return {
        "subject": metadata["subject"],
        "starts_sha256": starts_sha,
        "score_sha256": hashes,
        "prototype": "Mean similarity to nine references, excluding the query ordinal from every class.",
        "uncertainty": "Paired concept-cluster bootstrap, 10000 draws, seed 20261125, fixed gallery and sub-03.",
        "arms": reports,
        "candidate_minus_baseline": paired_object_panel(reports[baseline], reports[candidate]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/context-residual-sub03"))
    parser.add_argument("--baseline", default="raw_baseline")
    parser.add_argument("--candidate", default="within_residual")
    parser.add_argument("--out", type=Path, help="Save full per-query and paired results")
    args = parser.parse_args()
    result = archived_pair(args.input, args.baseline, args.candidate)
    if args.out:
        args.out.parent.mkdir(exist_ok=True, parents=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "arms": {name: v["mean"] for name, v in result["arms"].items()},
                "candidate_minus_baseline": result["candidate_minus_baseline"],
            },
            indent=2,
        )
    )
