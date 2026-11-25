"""Calibrate a two-finalist EEG rule on generic objects and score its frozen held-object run."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
from fit_episodic_object import SEED, apply_projection, projection, unit
from fit_hard_object_pairs import episodes
from fit_personal_session_metric import similarities
from fit_shared_photo_eeg import load_photos, require
from fit_trial_top_two import DESIGN, PAIRS, PUBLIC, context, paths, splits
from register_fresh_photo_panel import freeze, read, sha
from scipy.optimize import minimize
from scipy.special import expit

FEATURES = (
    "fisher_gap", "full_neural_gap", "single_mean", "single_concordance", "single_dispersion",
    "pair_mean", "complement_mean", "complement_concordance", "complement_dispersion",
    "early_late_cross", "early_late_same", "support_superiority", "support_standardized",
    "support_density",
)
COMPLEMENT = [5 + PAIRS.index(tuple(k for k in range(4) if k not in pair)) for pair in PAIRS]
SESSIONS = [5 + PAIRS.index((1, 3)), 5 + PAIRS.index((0, 2))]


def zscores(scores):
    return (scores - scores.mean(-1, keepdims=True)) / scores.std(-1, keepdims=True).clip(1e-8)


def per_view_scores(views, support):
    # 11 query views against a class centroid from 8 separate support photographs.
    centers = unit(support.mean(1))
    all_scores = np.einsum("vd,kud->vuk", views, centers)
    return zscores(all_scores)


def additional(views, support, runner, leader):
    scores = per_view_scores(views, support)
    gap = scores[..., runner] - scores[..., leader]
    singles, pairs = gap[1:5, 0], gap[5:, 0]
    complementary = gap[np.arange(5, 11), COMPLEMENT]
    early, late = SESSIONS

    # In each of six complementary pairs, compare the eight support photographs from
    # each finalist. The support-only axis and its within-class dispersion are independent
    # of the query. This accounts for heterogeneous support reliability without test labels.
    left = support[runner][:, COMPLEMENT].transpose(1, 0, 2)
    right = support[leader][:, COMPLEMENT].transpose(1, 0, 2)
    qs = views[5:]
    cl = np.einsum("vd,vsd->vs", qs, left)
    cr = np.einsum("vd,vsd->vs", qs, right)
    superiority = np.sign(cl[:, :, None] - cr[:, None, :]).mean()
    axis = unit(left.mean(1) - right.mean(1))
    a = np.einsum("vd,vsd->vs", axis, left)
    b = np.einsum("vd,vsd->vs", axis, right)
    q = (axis * qs).sum(-1)
    ma, mb = a.mean(1), b.mean(1)
    va, vb = a.var(1, ddof=1), b.var(1, ddof=1)
    pooled = ((va + vb) / 2).clip(1e-6)
    va, vb = (va + pooled) * 9 / 16, (vb + pooled) * 9 / 16
    linear = ((q - (ma + mb) / 2) / np.sqrt(pooled * 9 / 8)).clip(-6, 6)
    density = (np.log(vb / va) + (q - mb) ** 2 / vb - (q - ma) ** 2 / va) / 2
    return [
        singles.mean(), np.sign(singles).mean(), singles.std(),
        pairs.mean(), complementary.mean(), np.sign(complementary).mean(), complementary.std(),
        (gap[early, late] + gap[late, early]) / 2,
        (gap[early, early] + gap[late, late]) / 2,
        superiority, linear.mean(), density.clip(-6, 6).mean(),
    ]


def episode_data(a, name):
    spec, source_ids, held96, sessions = context(a)
    folder = a.out / name
    rec = read(folder / "receipt.json")
    require(rec["packet_sha256"] == sha(folder / "views.npz") and rec["design_sha256"] == sha(DESIGN)
            and rec["training_source_sha256"] == sha(Path(__file__).with_name("fit_trial_top_two.py")),
            "Trial views have changed")
    train, ids = (np.array(rec[k]) for k in ("train_classes", "test_classes"))
    if name == "held":
        require(np.array_equal(train, source_ids) and np.array_equal(ids, held96), "Wrong held split")
        seed = SEED + 1012
    else:
        j = int(name[-1])
        expected = splits(source_ids, sessions[source_ids], spec["seed"])[j]
        require(np.array_equal(ids, expected) and np.array_equal(train, np.setdiff1d(source_ids, ids)),
                "Fold mixes fitting or evaluation objects")
        seed = spec["seed"] + 100 * (j + 1) + 12
    with np.load(folder / "views.npz", allow_pickle=False) as packet:
        require(np.array_equal(ids, packet["ids"]), "Displaced neural rows")
        neural = packet["values"].astype(float).reshape(-1, 11, 96)
    raw = load_photos(a, "sub-04", train)
    fisher = apply_projection(load_photos(a, "sub-04", ids), projection(raw, sessions[train]))
    fisher = fisher[..., :96].reshape(-1, 96)
    rows = ids[:, None] * 10 + np.arange(10)
    max_photo_error = 0.
    if name == "held":
        original = read(a.base / "receipt.json")
        require(sha(a.base / "embeddings.npz") == original["packet_sha256"], "Original embedding changed")
        with np.load(a.base / "embeddings.npz", allow_pickle=False) as source:
            require(np.array_equal(source["ids"], ids), "Original embedding order changed")
            max_photo_error = float(np.abs(neural[:, 0] - source["features"].reshape(-1, 96)).max())
        require(max_photo_error < 2e-6, "Full neural embeddings no longer replay")
    data = {k: [] for k in ("rows", "donor_rows", "candidate_ids", "support_rows", "mixed_scores",
                            "leader", "runner", "original_features", "control_features")}
    for _, _, _, _, qr, dr, sr in episodes(rows, ids, sessions[ids], 12, seed):
        qi, di, si = (np.searchsorted(rows.ravel(), k) for k in (qr, dr, sr))
        require(np.array_equal(rows.ravel()[si], sr) and np.array_equal(rows.ravel()[di], dr)
                and np.array_equal(rows.ravel()[qi], qr) and np.all(qr // 10 != dr // 10)
                and np.isin(dr // 10, sr[:, 0] // 10).all()
                and len(np.unique(sessions[sr[:, 0] // 10])) == 1, "Wrong same-menu control/support")
        s = neural[si]
        fs = similarities(fisher[qi], fisher[si])
        ds = similarities(neural[qi, 0], s[:, :, 0])
        mixed = (fs + ds) / 2
        finalists = np.argsort(-mixed, kind="stable", axis=1)[:, :2]
        groups = sr[:, 0] // 10
        for i, (leader, runner) in enumerate(finalists):
            base = [fs[i, runner] - fs[i, leader], ds[i, runner] - ds[i, leader]]
            own = base + additional(neural[qi[i]], s, runner, leader)
            sham = base + additional(neural[di[i]], s, runner, leader)
            values = (qr[i], dr[i], groups, sr, mixed[i], groups[leader], groups[runner], own, sham)
            for key, value in zip(data, values, strict=True):
                data[key].append(value)
    data = {k: np.array(v) for k, v in data.items()}
    if name == "held":
        reg = read(a.base / "score.json")
        require(sha(a.base / "score.json") == spec["base_score_sha256"]
                and sha(a.base / "generic_12.npz") == reg["predictions_sha256"]["generic_12"],
                "Original decisions changed")
        with np.load(a.base / "generic_12.npz", allow_pickle=False) as archived:
            require(all(np.array_equal(archived[k], data[k]) for k in
                        ("rows", "donor_rows", "candidate_ids", "support_rows")), "Different menu")
            err = float(np.abs(archived["mixed_scores"] - data["mixed_scores"]).max())
            require(err < 2e-5 and np.array_equal(archived["mixed"], data["leader"]), "Baseline mismatch")
            require((data["leader"] == data["rows"] // 10).sum() == 514, "Expected 514/960")
        data["max_archived_score_error"] = err
        data["max_archived_full_embedding_error"] = max_photo_error
    return data


def fit_rule(x, labels):
    mean, scale = x.mean(0), x.std(0).clip(1e-6)
    x = (x - mean) / scale

    def objective(w):
        logits = x @ w[:-1] + w[-1]
        diff = expit(logits) - labels
        loss = np.logaddexp(0., logits).sum() - (labels @ logits) + 5 * (w[:-1] @ w[:-1])
        return loss, np.r_[x.T @ diff + 10 * w[:-1], diff.sum()]

    fit = minimize(objective, np.zeros(x.shape[1] + 1), method="L-BFGS-B", jac=True,
                   options={"maxiter": 300, "ftol": 1e-13})
    require(fit.success and np.isfinite(fit.x).all(), "Conditional logistic fit failed")
    return dict(mean=mean.tolist(), scale=scale.tolist(), coef=fit.x[:-1].tolist(),
                intercept=float(fit.x[-1]))


def probability(x, rule):
    return expit(((x - rule["mean"]) / rule["scale"]) @ rule["coef"] + rule["intercept"])


def flips(p, threshold):
    return p >= threshold if threshold is not None else np.zeros(len(p), dtype=bool)


def turns(original, new, truth):
    base, alt = original == truth, new == truth
    change = original != new
    return dict(flips=int(change.sum()), corrected=int((alt & ~base).sum()),
                newly_wrong=int((base & ~alt).sum()), wrong_to_wrong=int((change & ~alt & ~base).sum()),
                net=int(alt.sum() - base.sum()))


def calibrate(a):
    spec, source_ids, _, _ = context(a)
    folds = [episode_data(a, name) for name in ("fold0", "fold1")]
    data = {k: np.concatenate([v[k] for v in folds]) for k in folds[0]}
    rows, x = data["rows"], data["original_features"]
    truth = rows // 10
    require(np.array_equal(np.sort(rows), (source_ids[:, None] * 10 + np.arange(10)).ravel()),
            "Training queries do not cover exactly the 416 fit objects")
    eligible = (truth == data["leader"]) | (truth == data["runner"])
    y = (truth == data["runner"]).astype(float)
    p = np.full(len(y), np.nan)
    for objs in np.array_split(np.random.default_rng(spec["seed"] + 1).permutation(source_ids), 4):
        test = np.isin(truth, objs)
        train = eligible & ~test
        p[test] = probability(x[test], fit_rule(x[train], y[train]))
    require(np.isfinite(p).all(), "Incomplete calibration")
    choices = []
    for cutoff in (.5, .55, .6, .65, .7, None):
        pred = np.where(flips(p, cutoff), data["runner"], data["leader"])
        choices.append(dict(threshold=cutoff, **turns(data["leader"], pred, truth)))
    selected = max(choices, key=lambda d: (d["net"], -d["flips"]))
    freeze(a.out / "locked_rule.json", dict(design_sha256=sha(DESIGN), source_sha256=sha(Path(__file__)),
           extractor_sha256=sha(Path(__file__).with_name("fit_trial_top_two.py")),
           folds={k: sha(a.out / k / "receipt.json") for k in ("fold0", "fold1")},
           feature_names=FEATURES, objects=416, rows=len(rows), binary_conditional_rows=int(eligible.sum()),
           cutoff_crossfit=choices, selected=selected, rule=fit_rule(x[eligible], y[eligible])))
    print("Locked calibration", selected, flush=True)


def contrast(a, b, truth, draws):
    objects = np.unique(truth)
    rate = np.array([((a[truth == c] == c).sum() - (b[truth == c] == c).sum()) / 10 for c in objects])
    ci = np.quantile(rate[draws].mean(1), [.025, .975])
    return dict(net=int((a == truth).sum() - (b == truth).sum()), delta_pp=float(100 * rate.mean()),
                ci95_pp=(100 * ci).tolist())


def score(a):
    spec, _, _, _ = context(a)
    path = a.out / "locked_rule.json"
    meta = read(path)
    require(meta["design_sha256"] == sha(DESIGN) and meta["source_sha256"] == sha(Path(__file__))
            and meta["extractor_sha256"] == sha(Path(__file__).with_name("fit_trial_top_two.py"))
            and meta["folds"] == {k: sha(a.out / k / "receipt.json") for k in ("fold0", "fold1")},
            "Locked rule differs")
    data = episode_data(a, "held")
    truth = data["rows"] // 10
    base, runner = data["leader"], data["runner"]
    preds = dict(baseline=base)
    for arm, key in (("own", "original_features"), ("shuffled", "control_features")):
        x = data[key]
        p = probability(x, meta["rule"])
        data[arm + "_probability"] = p
        preds[arm] = np.where(flips(p, meta["selected"]["threshold"]), runner, base)
        fixed = x[:, 0] + x[:, 1] + x[:, FEATURES.index("complement_mean")]
        preds["fixed_" + arm] = np.where(fixed > 0, runner, base)
    require(all(((pred == base) | (pred == runner)).all() for pred in preds.values())
            and np.array_equal(data["original_features"][:, :2], data["control_features"][:, :2]),
            "A finalist or base score changed")
    objects, counts = np.unique(truth, return_counts=True)
    require(len(objects) == 96 and (counts == 10).all(), "Wrong cluster denominator")
    draws = np.random.default_rng(spec["seed"]).integers(96, size=(10000, 96))
    compare = {f"{a1}_minus_{a2}": contrast(preds[a1], preds[a2], truth, draws) for a1, a2 in
               (("own", "baseline"), ("own", "shuffled"), ("fixed_own", "baseline"),
                ("fixed_own", "fixed_shuffled"))}
    passed = (compare["own_minus_baseline"]["net"] >= 10 and
              all(compare[k]["ci95_pp"][0] > 0 for k in ("own_minus_baseline", "own_minus_shuffled")))
    output = a.out / "per_query.npz"
    np.savez(output, **data, **preds)
    report = dict(design_sha256=sha(DESIGN), source_sha256=sha(Path(__file__)),
                  rule_sha256=sha(path), per_query_sha256=sha(output),
                  held_embedding_receipt_sha256=sha(a.out / "held/receipt.json"),
                  held_historical_decisions_sha256=sha(a.base / "generic_12.npz"),
                  fraction_unit="percentage points for paired differences", subject="sub-04",
                  training_objects=416, held_objects=96, n=960, candidates=12, support_photos=8, trials=4,
                  top_two_contains_truth_diagnostic_only=int(((base == truth) | (runner == truth)).sum()),
                  baseline_reproduced=int((base == truth).sum()), calibration=meta["selected"],
                  calibration_conditional_rows=meta["binary_conditional_rows"],
                  baseline_replay_max_score_error=data["max_archived_score_error"],
                  full_embedding_replay_max_error=data["max_archived_full_embedding_error"],
                  hits={k: int((v == truth).sum()) for k, v in preds.items()},
                  changes_from_baseline={k: turns(base, v, truth) for k, v in preds.items() if k != "baseline"},
                  comparisons=compare, advance_to_target=bool(passed))
    freeze(a.out / "summary.json", report)
    freeze(PUBLIC / "trial_top_two.json", report)
    print(report["hits"], compare, "advance", passed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "score"))
    parser.add_argument("--expanded", type=Path, default=Path("outputs/photo-structure/expanded"))
    paths(parser)
    args = parser.parse_args()
    (calibrate if args.stage == "calibrate" else score)(args)
