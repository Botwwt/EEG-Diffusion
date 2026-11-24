"""Estimate an eight-shot EEG object prototype with independent calibration photographs."""

import argparse
from pathlib import Path

import numpy as np
from fit_full_head_objects import fisher
from fit_personal_session_metric import load, similarities
from fit_support_structure import H14_SHA, unit
from fit_within_photo_embedding import images, read, sha, training
from prepare_expanded_personal import extract
from scipy.linalg import solve
from score_reconstruction_jpegs import registered
from score_within_photo_dino import save

SEED = 260511
GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
RIDGE = 0.3
FIGURE = "384 selection objects in four disjoint folds; 128 separate generic check objects. Three fixed balanced assignments of classes to 12-way episodes, each with five two-photo holdouts, eight supports and four-trial mean EEG. Each generic photo appears as a query nine times across alternative object sets."


def project(train_eeg, train_h14, query_eeg):
    flat = train_eeg.reshape(-1, train_eeg.shape[-1])
    q = query_eeg.reshape(-1, flat.shape[-1])
    # All normalization, between/within covariance, and Fisher directions use flat alone.
    projected = fisher(flat, np.concatenate((flat, q)), 96)
    target = projected[: len(flat)].reshape(len(train_eeg), 10, 96).mean(1)
    eeg = projected[len(flat) :].reshape(*query_eeg.shape[:-1], 96)
    visual = train_h14.mean(1)
    center, intercept = visual.mean(0), target.mean(0)
    features = visual - center
    gram = features @ features.T
    gram.flat[:: len(gram) + 1] += RIDGE * np.trace(gram) / len(gram)
    mapping = features.T @ solve(gram, target - intercept, assume_a="pos")
    return eeg, (center, mapping, intercept)


def decisions(eeg, support_eeg, support_visual, mapping, weights):
    """Inference receives query EEG and eight support recordings and images for each class."""
    center, matrix, intercept = mapping
    measured = unit(support_eeg.mean(1))
    visual_prediction = unit((support_visual.mean(1) - center) @ matrix + intercept)
    directions = unit((1 - weights) * measured + weights * visual_prediction)
    return (unit(eeg) @ directions.transpose(0, 2, 1)).argmax(axis=-1)


def episodes(x, v, train, test, weights, seed):
    if set(train) & set(test) or len(test) % 4:
        raise ValueError("Generic fitting and held-out classes must be disjoint")
    query, mapping = project(x[train], v[train], x[test])
    truth = np.repeat(np.arange(12), 2)
    mix = np.asarray(weights)[:, None, None]
    rng = np.random.default_rng(seed)
    counts = np.zeros((len(test), len(weights)), dtype=np.int64)
    for _ in range(3):
        order = rng.permutation(len(test))
        for start in range(0, len(test), 4):
            active = np.roll(order, -start)[:12]
            schedule = np.stack([rng.permutation(10) for _ in active])
            for fold in range(5):
                held = np.arange(10) // 2 == fold
                support = schedule[:, ~held]
                eeg = query[active[:, None], schedule[:, held]].reshape(24, 96)
                calibrated = query[active[:, None], support]
                pred = decisions(eeg, calibrated, v[test[active, None], support], mapping, mix)
                if weights[0] == 0 and not np.array_equal(
                    pred[0], similarities(eeg, calibrated).argmax(1)
                ):
                    raise ValueError("Measured EEG prototype differs from the frozen cosine rule")
                counts[active] += (pred == truth).reshape(len(weights), 12, 2).sum(2).T
    return counts


def generic_report(counts, weights, selected):
    # The bootstrap unit is the entire generic object, including all repeated episodes.
    diff = counts[:, selected] - counts[:, 0]
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(counts), size=(10000, len(counts)))
    return dict(
        training_domain="external generic objects",
        heldout_objects=len(counts),
        distinct_queries=10 * len(counts),
        episodic_decisions=90 * len(counts),
        weights=list(weights),
        hits=counts.sum(0).tolist(),
        chosen_weight=weights[selected],
        chosen_minus_measured=dict(
            delta=float(diff.mean() / 90),
            class_bootstrap_ci95=(np.quantile(diff[draws].mean(1), [0.025, 0.975]) / 90).tolist(),
        ),
    )


def generic(args, bank):
    subject = "sub-04"
    classes, eeg, visual = training(args, subject, 512, bank)
    perm = np.random.default_rng(SEED).permutation(512)
    development, confirmation = perm[:384], perm[384:]
    folds = np.array_split(development, 4)
    all_counts = []
    for f, hold in enumerate(folds):
        counts = episodes(eeg, visual, np.setdiff1d(development, hold), hold, GRID, SEED + 11 + f)
        all_counts.append(counts)
        print("Sub-04 generic selection fold", f + 1, counts.sum(0).tolist(), flush=True)
    counts = np.concatenate(all_counts)
    winner = int(np.argmax(counts.sum(0)))  # Grid order breaks ties toward the measured prototype.
    selection = dict(
        protocol=FIGURE,
        subject=subject,
        seed=SEED,
        ridge_ratio=RIDGE,
        source_sha256=sha(args.expanded / "features" / (subject + ".json")),
        visual_sha256=H14_SHA,
        code_sha256=sha(Path(__file__)),
        heldout_folds=[classes[f].tolist() for f in folds],
        generic_check_classes=classes[confirmation].tolist(),
        gate="Proceed only for a nonzero OOF winner with strictly more decisions correct on the separate 128 generic objects.",
        selection=generic_report(counts, GRID, winner),
    )
    selection_path = args.out / "selection.json"
    save(selection_path, selection)
    if winner:
        chosen = [0.0, GRID[winner]]
        check = episodes(eeg, visual, development, confirmation, chosen, SEED + 90)
        confirmation_result = generic_report(check, chosen, 1)
        proceed = check[:, 1].sum() > check[:, 0].sum()
    else:
        confirmation_result, proceed = None, False
    report = dict(
        model="Posterior R4 Fisher space with 96 directions; linearly map H/14 visual class means to EEG class means with a ridge of 0.3 × the average centered class Gram diagonal. At inference, mix unit measured and visual-predicted EEG class directions using only each fold's eight supports. Query image and true query word are excluded.",
        selection=selection,
        selection_sha256=sha(selection_path),
        generic_check=confirmation_result,
        proceed=bool(proceed),
        other_participants_or_target_panels_scored=False,
    )
    save(args.out / "generic.json", report)
    print(
        "Generic decision:",
        report["proceed"],
        "chosen weight",
        GRID[winner],
        confirmation_result,
        flush=True,
    )


def target_panel(args, sub, panel, bank, classes, x, v, mix):
    fresh = panel == "fresh"
    reg = registered(args.fresh / "references" if fresh else args.references, panel)
    source = (args.fresh / "eeg" if fresh else args.inputs / "halves") / (sub + ".npz")
    rows, _, _, raw = load(source)
    if np.intersect1d(classes, reg["rows"] // 10).size:
        raise ValueError("Generic training class appears in target panel")
    q, mapper = project(x, v, extract(raw).astype(np.float64))
    pred = np.full((2, 120), -1)
    lookup = {int(r): i for i, r in enumerate(rows)}
    for fold, groups in enumerate(reg["calibration"]):
        use = np.flatnonzero(reg["folds"] == fold)
        support = np.array([[lookup[int(r)] for r in g] for g in groups])
        idx = [lookup[int(r)] for r in reg["rows"][use]]
        # Only the eight permitted images for each candidate class enter this fold.
        legal = unit(bank[groups].astype(np.float64))
        pred[:, use] = decisions(q[idx], q[support], legal, mapper, mix)
    baseline_path = (
        (args.fresh / "evaluations" / sub / panel) if fresh else (args.expanded / panel / sub)
    ) / "evaluation.json"
    baseline = read(baseline_path)["results"]["expanded_r4"]["predicted"]
    if not np.array_equal(pred[0], baseline) or (pred < 0).any():
        raise ValueError("Zero-weight predictions did not replay the original model")
    correct = pred == reg["truth"]
    delta = correct[1].astype(float) - correct[0]
    by_class = np.array([delta[reg["truth"] == c].mean() for c in range(12)])
    draws = np.random.default_rng(SEED).integers(12, size=(10000, 12))
    save(
        args.out / "predictions" / panel / (sub + ".json"),
        dict(
            rows=reg["rows"].tolist(),
            predicted=pred.tolist(),
            registration_sha256=reg["sha"],
            target_eeg_sha256=sha(source),
            source_baseline_sha256=sha(baseline_path),
        ),
    )
    return dict(
        n=120,
        hits=correct.sum(1).tolist(),
        paired_ci95=np.quantile(by_class[draws].mean(1), [0.025, 0.975]).tolist(),
    )


def evaluate(args, bank):
    r = read(args.out / "generic.json")
    s = r["selection"]
    if not r["proceed"] or sha(args.out / "selection.json") != r["selection_sha256"]:
        raise ValueError("Generic confirmation did not release a nonzero readout")
    if s["code_sha256"] != sha(Path(__file__)):
        raise ValueError("Candidate source differs from the generic-selection lock")
    chosen = [0.0, s["selection"]["chosen_weight"]]
    cases = {}
    for sub in ("sub-04", "sub-05", "sub-01"):
        classes, x, v = training(args, sub, 512, bank)
        ids = {int(k): i for i, k in enumerate(classes)}
        hold = np.array([ids[k] for k in s["generic_check_classes"]])
        dev = np.array([ids[k] for f in s["heldout_folds"] for k in f])
        check = (
            r["generic_check"]
            if sub == "sub-04"
            else generic_report(episodes(x, v, dev, hold, chosen, SEED + 90), chosen, 1)
        )
        cases[sub] = dict(
            generic_check=check,
            panels={
                panel: target_panel(
                    args, sub, panel, bank, classes, x, v, np.asarray(chosen)[:, None, None]
                )
                for panel in ("registered", "replicate", "fresh")
            },
        )
        print(sub, cases[sub], flush=True)
    save(
        args.out / "evaluation.json",
        dict(
            generic=r,
            cases=cases,
            scope="Participant metrics and the three previously observed target panels are developmental. A separately locked new panel is required for confirmation. No images generated.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["generic", "evaluate"])
    for key, loc in dict(
        source="outputs/official-object-readout/sub-03",
        expanded="outputs/photo-structure/expanded",
        inputs="outputs/object-transfer",
        references="outputs/object-fidelity",
        fresh="outputs/fresh-photo",
        out="outputs/visual-eeg-prototypes",
    ).items():
        p.add_argument("--" + key, type=Path, default=Path(loc))
    a = p.parse_args()
    (generic if a.stage == "generic" else evaluate)(a, images(a.source))
