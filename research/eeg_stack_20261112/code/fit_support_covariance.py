"""Test a support-estimated Mahalanobis distance in the locked personal Fisher space."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from fit_full_head_objects import fisher
from fit_personal_session_metric import load, similarities
from fit_visual_eeg_prototypes import SEED, generic_report
from fit_within_photo_embedding import read, sha
from prepare_expanded_personal import extract
from scipy.linalg import cho_factor, cho_solve
from score_reconstruction_jpegs import registered
from score_within_photo_dino import save

BASE = ("cosine", 0.0, 0.0)
GRID = [BASE] + [
    (target, sample, mix)
    for mix in (0.25, 0.50, 1.0)
    for sample in (0.0, 0.25, 0.50)
    for target in ("isotropic", "generic")
]


def split_digest(path):
    # Check the registered JSON bytes after normalizing text checkout line endings.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def eeg_features(expanded, subject):
    root = expanded / "features"
    meta = read(root / (subject + ".json"))
    classes = np.asarray(meta["training_concepts"])
    if meta["subject"] != subject or len(set(classes)) != 512:
        raise ValueError("Incorrect generic training registry")
    features = []
    for concept in classes:
        with np.load(root / subject / f"{concept:04d}.npz", allow_pickle=False) as block:
            x = block["features"]
            if (
                int(block["start"]) != 10 * concept
                or str(block["raw_sha"]) != meta["raw_block_sha256"][str(concept)]
                or x.shape != (10, 1003)
                or hashlib.sha256(x.tobytes()).hexdigest() != str(block["feature_sha"])
            ):
                raise ValueError("Incorrect EEG training block")
            features.append(x)
    x = np.stack(features)
    if hashlib.sha256(x.tobytes()).hexdigest() != meta["train_design_sha256"]:
        raise ValueError("Generic EEG design differs from the frozen 512-object baseline")
    return classes, x.astype(np.float64)


def project(train, queries):
    flat = train.reshape(-1, train.shape[-1])
    combined = np.concatenate((flat, queries.reshape(-1, flat.shape[-1])))
    directions = fisher(flat, combined, 96)
    z = directions[: len(flat)].reshape(len(train), 10, 96)
    grouped = (z - z.mean(1, keepdims=True)).reshape(-1, 96)
    generic = grouped.T @ grouped / (len(train) * 9)
    return directions[len(flat) :].reshape(*queries.shape[:-1], 96), generic


def predict(query, supports, generic, candidates):
    """Use only the eight labeled EEGs per object to adapt to the current episode."""
    means = supports.mean(1)
    residual = (supports - means[:, None]).reshape(-1, 96)
    sample = residual.T @ residual / (len(supports) * (supports.shape[1] - 1))
    scale = np.trace(sample) / len(sample)
    unit_cov = np.eye(len(sample)) * scale
    # Match trace so the blend controls shape without changing the distance scale.
    targets = {
        "isotropic": unit_cov,
        "generic": generic * (scale * len(sample) / np.trace(generic)),
    }
    old = similarities(query, supports)
    scores, predictions = {}, []
    for target, weight, mix in candidates:
        if target == "cosine":
            predictions.append(old.argmax(1))
            continue
        key = (target, weight)
        if key not in scores:
            covariance = (1 - weight) * targets[target] + weight * sample
            covariance.flat[:: len(covariance) + 1] += max(scale, 1e-12) * 1e-6
            # The query quadratic is the same for every candidate class and cancels.
            precision_means = cho_solve(cho_factor(covariance, lower=True), means.T)
            sim = query @ precision_means - 0.5 * (means * precision_means.T).sum(1)
            scores[key] = (sim - sim.mean(1, keepdims=True)) / sim.std(1, keepdims=True).clip(1e-8)
        predictions.append(((1 - mix) * old + mix * scores[key]).argmax(1))
    return np.stack(predictions)


def episodes(x, train, test, candidates, seed):
    if set(train) & set(test) or len(test) % 4:
        raise ValueError("Source classes and generic episode classes overlap")
    eeg, covariance = project(x[train], x[test])
    truth = np.repeat(np.arange(12), 2)
    rng = np.random.default_rng(seed)
    counts = np.zeros((len(test), len(candidates)), dtype=np.int64)
    for _ in range(3):
        order = rng.permutation(len(test))
        for start in range(0, len(test), 4):
            active = np.roll(order, -start)[:12]
            schedule = np.stack([rng.permutation(10) for _ in active])
            for fold in range(5):
                held = np.arange(10) // 2 == fold
                query = eeg[active[:, None], schedule[:, held]].reshape(24, 96)
                support = eeg[active[:, None], schedule[:, ~held]]
                hits = predict(query, support, covariance, candidates) == truth
                counts[active] += hits.reshape(len(candidates), 12, 2).sum(2).T
    return counts


def describe(counts, candidates, chosen):
    r = generic_report(counts, candidates, chosen)
    r["candidates_in_order"] = r.pop("weights")
    r["chosen"] = r.pop("chosen_weight")
    return r


def index_classes(classes, split):
    idx = {int(k): i for i, k in enumerate(classes)}
    folds = [np.array([idx[k] for k in group]) for group in split["heldout_folds"]]
    held = np.array([idx[k] for k in split["generic_check_classes"]])
    development = np.concatenate(folds)
    if len(held) != 128 or len(development) != 384 or len(set(np.r_[development, held])) != 512:
        raise ValueError("Incorrect classwise generic partition")
    return folds, development, held


def generic(args):
    subject = "sub-04"
    classes, x = eeg_features(args.expanded, subject)
    split = read(args.split)["generic"]["selection"]
    folds, dev, held = index_classes(classes, split)
    counts = []
    for f, group in enumerate(folds):
        counts.append(episodes(x, np.setdiff1d(dev, group), group, GRID, SEED + 11 + f))
        print("Generic selection fold", f + 1, counts[-1].sum(0).tolist(), flush=True)
    totals = np.concatenate(counts)
    if int(totals[:, 0].sum()) != int(split["selection"]["hits"][0]):
        raise ValueError("Historical generic cosine baseline failed to replay")
    best = int(np.argmax(totals.sum(0)))
    lock = dict(
        subject=subject,
        selection=describe(totals, GRID, best),
        candidate_fields=["covariance_prior", "support_covariance_weight", "distance_score_weight"],
        rule="Most OOF hits on 384 sub-04 generic objects; ties favor the earliest candidate. Check the chosen method once on 128 separate generic objects. Continue only for strictly more correct decisions than the original cosine model.",
        seed=SEED,
        grid=GRID,
        generic_split_lf_sha256=split_digest(args.split),
        source_sha256=sha(args.expanded / "features/sub-04.json"),
        code_sha256=sha(Path(__file__)),
    )
    save(args.out / "selection.json", lock)
    result = None
    proceed = False
    if best:
        chosen = [BASE, GRID[best]]
        c = episodes(x, dev, held, chosen, SEED + 90)
        result = describe(c, chosen, 1)
        if result["hits"][0] != read(args.split)["generic"]["generic_check"]["hits"][0]:
            raise ValueError("Generic check did not replay the original cosine baseline")
        proceed = result["hits"][1] > result["hits"][0]
    report = dict(
        selection=lock,
        selection_sha256=sha(args.out / "selection.json"),
        generic_check=result,
        proceed=proceed,
        definition="Four-repeat EEG, 96 Fisher directions fitted on external classes only. The episode covariance uses within-object deviations of 12×8 support EEGs (84 residual degrees of freedom), with a trace-matched isotropic or external within-class prior. A 1e-6 trace-relative diagonal ridge stabilizes factorization. Class-score vectors are centered and rescaled separately for each query before mixing. No target image or query EEG is used to estimate covariance.",
    )
    save(args.out / "generic.json", report)
    print("Checked generic selection", GRID[best], result, "Proceed:", proceed, flush=True)


def panel_result(args, sub, panel, classes, training, candidates):
    fresh = panel == "fresh"
    reg = registered(args.fresh / "references" if fresh else args.references, panel)
    source = (args.fresh / "eeg" if fresh else args.inputs / "halves") / (sub + ".npz")
    rows, _, _, signals = load(source)
    if np.intersect1d(classes, reg["rows"] // 10).size:
        raise ValueError("Target object found in generic EEG")
    test, prior = project(training, extract(signals).astype(np.float64))
    lookup = {r: i for i, r in enumerate(rows)}
    pred = np.full((len(candidates), 120), -1)
    for f, support in enumerate(reg["calibration"]):
        active = np.flatnonzero(reg["folds"] == f)
        cal = np.array([[lookup[r] for r in g] for g in support])
        query = np.array([lookup[r] for r in reg["rows"][active]])
        pred[:, active] = predict(test[query], test[cal], prior, candidates)
    prev = (
        (args.fresh / "evaluations" / sub / panel) if fresh else (args.expanded / panel / sub)
    ) / "evaluation.json"
    if (
        not np.array_equal(pred[0], read(prev)["results"]["expanded_r4"]["predicted"])
        or (pred < 0).any()
    ):
        raise ValueError("Frozen cosine object predictions differ")
    hit = pred == reg["truth"]
    delta = hit[1].astype(float) - hit[0]
    v = np.array([delta[reg["truth"] == t].mean() for t in range(12)])
    draws = np.random.default_rng(SEED).integers(12, size=(10000, 12))
    save(
        args.out / "predictions" / panel / (sub + ".json"),
        dict(
            rows=reg["rows"].tolist(),
            predicted=pred.tolist(),
            registration_sha256=reg["sha"],
            eeg_sha256=sha(source),
            original_prediction_file_sha256=sha(prev),
        ),
    )
    return dict(
        n=120,
        hits=hit.sum(1).tolist(),
        own_minus_cosine_ci95=np.quantile(v[draws].mean(1), [0.025, 0.975]).tolist(),
    )


def evaluate(args):
    report = read(args.out / "generic.json")
    s = report["selection"]
    if (
        not report["proceed"]
        or sha(args.out / "selection.json") != report["selection_sha256"]
        or s["code_sha256"] != sha(Path(__file__))
        or s["generic_split_lf_sha256"] != split_digest(args.split)
    ):
        raise ValueError(
            "Need a valid locked generic winner and an independently improved generic check"
        )
    chosen = [BASE, tuple(s["selection"]["chosen"])]
    split = read(args.split)["generic"]["selection"]
    cases = {}
    for sub in ("sub-04", "sub-05", "sub-01"):
        classes, x = eeg_features(args.expanded, sub)
        _, dev, held = index_classes(classes, split)
        check = (
            report["generic_check"]
            if sub == "sub-04"
            else describe(episodes(x, dev, held, chosen, SEED + 90), chosen, 1)
        )
        cases[sub] = dict(
            generic_check=check,
            panels={
                p: panel_result(args, sub, p, classes, x, chosen)
                for p in ("registered", "replicate", "fresh")
            },
        )
        print(sub, cases[sub], flush=True)
    save(
        args.out / "evaluation.json",
        dict(
            generic=report,
            cases=cases,
            scope="All three already observed target panels are developmental. The fixed readout needs separate confirmation before any image claim.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["generic", "evaluate"])
    for k, loc in dict(
        expanded="outputs/photo-structure/expanded",
        inputs="outputs/object-transfer",
        fresh="outputs/fresh-photo",
        references="outputs/object-fidelity",
        out="outputs/support-covariance",
        split="research/eeg_stack_20261112/results/object_fidelity/visual_eeg_prototypes.json",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(loc))
    a = p.parse_args()
    (generic if a.stage == "generic" else evaluate)(a)
