"""Train a single-person object decoder using two other people's generic EEG class means."""

import argparse
from pathlib import Path

import numpy as np
from fit_full_head_objects import fisher
from fit_personal_session_metric import load, similarities
from fit_support_covariance import eeg_features
from prepare_expanded_personal import extract
from register_fresh_photo_panel import freeze, read, sha
from scipy.linalg import cholesky, solve_triangular, svd
from score_reconstruction_jpegs import registered

DIR = Path(__file__).resolve().parents[1] / "results/object_fidelity"
ARMS = ("native", "self", "sham", "teacher")


def derange(pairs, seed):
    rng = np.random.default_rng(seed)
    groups, codes = np.unique(pairs, axis=0, return_inverse=True)
    order = np.arange(len(codes))
    for i in range(len(groups)):
        ids = rng.permutation(np.flatnonzero(codes == i))
        if len(ids) < 2:
            raise ValueError("A joint training session stratum cannot be deranged")
        order[ids] = np.roll(ids, rng.integers(1, len(ids)))
    if (
        len(set(order)) != len(codes)
        or np.any(order == np.arange(len(order)))
        or not np.array_equal(pairs, pairs[order])
    ):
        raise ValueError("Incorrect session-matched teacher permutation")
    return order


def teacher_targets(centers, sessions):
    result = centers.copy()
    for group in np.unique(sessions, axis=0):
        keep = (sessions == group).all(1)
        result[keep] -= centers[keep].mean(0)
    return result / result.std(0).clip(1e-8)


def stream(centers, query, teachers):
    mean = centers.mean(0)
    scale = centers.std(0).clip(1e-8)
    x = (centers - mean) / scale
    gram = x.T @ x
    gram.flat[:: len(gram) + 1] += 0.5 * np.trace(gram) / len(gram)
    chol = cholesky(gram, lower=True)
    y = np.concatenate(teachers, axis=1) / np.sqrt(len(teachers))
    cross = solve_triangular(chol, x.T @ y, lower=True)
    u, s, _ = svd(cross, full_matrices=False)
    directions = solve_triangular(chol.T, u[:, :64] * s[:64], lower=False)
    return ((query - mean) / scale) @ directions


def projections(sub, subjects, centers, groups, query):
    donors = [s for s in subjects if s != sub]
    teacher = {s: teacher_targets(centers[s], groups[s]) for s in subjects}
    shuffled = []
    for s in donors:
        joint = np.column_stack((groups[sub], groups[s]))
        indices = derange(joint, 260532 + 100 * subjects.index(sub) + subjects.index(s))
        shuffled.append(teacher[s][indices])
    return [
        query,
        stream(centers[sub], query, [teacher[sub]]),
        stream(centers[sub], query, shuffled),
        stream(centers[sub], query, [teacher[s] for s in donors]),
    ]


def scores(streams, q, s):
    base = similarities(streams[0][q], streams[0][s])
    return np.stack([base] + [(base + similarities(x[q], x[s])) / 2 for x in streams[1:]]).argmax(2)


def evaluate_episode(streams, active, schedule, counts, first_only=False):
    ids = active[:, None] * 10 + schedule
    truth = np.repeat(np.arange(12), 2)
    for fold in range(5):
        held = np.arange(10) // 2 == fold
        chosen = scores(streams, ids[:, held].ravel(), ids[:, ~held])
        hits = (chosen == truth).reshape(len(ARMS), 12, 2).sum(2).T
        if first_only:
            counts[active[0]] += hits[0]
        else:
            counts[active] += hits


def episodes(streams, groups, same_session):
    n = len(groups)
    counts = np.zeros((n, len(ARMS)), np.int64)
    rng = np.random.default_rng(260535 if same_session else 260533)
    for _ in range(3):
        if same_session:
            for i in range(n):
                pool = np.flatnonzero(np.all(groups == groups[i], axis=1) & (np.arange(n) != i))
                active = np.r_[i, rng.choice(pool, 11, replace=False)]
                evaluate_episode(
                    streams, active, np.stack([rng.permutation(10) for _ in active]), counts, True
                )
        else:
            order = rng.permutation(n)
            for start in range(0, n, 4):
                active = np.roll(order, -start)[:12]
                evaluate_episode(
                    streams, active, np.stack([rng.permutation(10) for _ in active]), counts
                )
    return counts


def prepare(a, ids, test, panel=False):
    centers, queries, groups, sessions, features = {}, {}, {}, {}, {}
    registry = read(a.inputs / "sessions.json")["subjects"]
    for sub in a.subjects:
        classes, data = eeg_features(a.expanded, sub)
        by_id = {int(c): i for i, c in enumerate(classes)}
        x = data[[by_id[k] for k in ids]].reshape(10 * len(ids), -1)
        if panel:
            # Only this recipient's physical EEG packet supplies queries and supports.
            path = a.prospective / "eeg" / (sub + ".npz")
            if sha(path) != read(path.with_suffix(".json"))["packet_sha256"]:
                raise ValueError("Changed registered EEG source")
            rows, _, _, mean = load(path)
            q = extract(mean).astype(float)
            sessions[sub] = rows
        else:
            q = data[[by_id[k] for k in test]].reshape(10 * len(test), -1)
            sessions[sub] = np.asarray(registry[sub]["session_pairs"])[test]
        projected = fisher(x, np.concatenate((x, q)), 96)
        centers[sub] = projected[: len(x)].reshape(len(ids), 10, 96).mean(1)
        queries[sub] = projected[len(x) :]
        groups[sub] = np.asarray(registry[sub]["session_pairs"])[ids]
        features[sub] = sha(a.expanded / "features" / (sub + ".json"))
    return centers, queries, groups, sessions, features


def summary(cases, n):
    values = np.stack(list(cases.values()))
    by_class = values.sum(0)
    draws = np.random.default_rng(260534).integers(len(by_class), size=(10000, len(by_class)))
    stats = {}
    for j, arm in enumerate(ARMS[:-1]):
        delta = (by_class[:, -1] - by_class[:, j]) / (len(cases) * n)
        stats[arm] = dict(
            delta=float(delta.mean()),
            ci95=np.quantile(delta[draws].mean(1), [0.025, 0.975]).tolist(),
        )
    return dict(
        decisions_per_object_per_person=n,
        hits={s: dict(zip(ARMS, v.sum(0).tolist(), strict=True)) for s, v in cases.items()},
        n_total=n * len(cases) * len(by_class),
        class_hits={s: v.tolist() for s, v in cases.items()},
        arms_in_order=list(ARMS),
        teacher_minus=stats,
    )


def run(a):
    design = read(a.design)
    split = read(a.split)
    if sha(a.split) != design["split_sha256"] or split["subjects"] != design["subjects"]:
        raise ValueError("Changed generic registration")
    a.subjects = design["subjects"]
    train, test = (np.array(split[k]) for k in ("train_classes", "test_classes"))
    if len(train) != 416 or len(test) != 96 or len(set(np.r_[train, test])) != 512:
        raise ValueError("Invalid generic split")
    if a.stage == "panel":
        outcome = read(a.release)
        if outcome["design_sha256"] != sha(a.design) or not outcome["advance"]:
            raise ValueError("Generic teacher gate did not qualify for panel inference")
        reg = registered(a.prospective / "references", "prospective")
        ids = np.sort(np.r_[train, test])
        if np.intersect1d(ids, reg["rows"] // 10).size:
            raise ValueError("Fourth panel intersects generic training")
        c, q, g, rows, _ = prepare(a, ids, None, True)
        case = {}
        for sub in a.subjects:
            streams = projections(sub, a.subjects, c, g, q[sub])
            lookup = {r: i for i, r in enumerate(rows[sub])}
            pred = np.full((len(ARMS), 120), -1)
            for f, cal in enumerate(reg["calibration"]):
                sel = np.flatnonzero(reg["folds"] == f)
                ix = [lookup[r] for r in reg["rows"][sel]]
                support = np.asarray([[lookup[r] for r in cl] for cl in cal])
                pred[:, sel] = scores(streams, ix, support)
            old = read(
                a.prospective
                / "render_inputs/expanded_r4/evaluations"
                / sub
                / "prospective/word_predictions.json"
            )
            if not np.array_equal(pred[0], old["predicted"]) or not np.array_equal(
                old["rows"], reg["rows"]
            ):
                raise ValueError("Panel baseline word parity failure")
            hits = pred == reg["truth"]
            case[sub] = np.stack([hits[:, reg["truth"] == i].sum(1) for i in range(12)])
        freeze(
            a.panel_release,
            dict(
                scope="Previously evaluated fourth panel, developmental",
                generic_sha256=sha(a.release),
                registration_sha256=reg["sha"],
                summary=summary(case, 10),
            ),
        )
        return
    c, q, g, s, features = prepare(a, train, test)
    main, paired = {}, {}
    for sub in a.subjects:
        streams = projections(sub, a.subjects, c, g, q[sub])
        main[sub] = episodes(streams, s[sub], False)
        paired[sub] = episodes(streams, s[sub], True)
    primary = summary(main, 90)
    gate = all(primary["teacher_minus"][k]["ci95"][0] > 0 for k in ("native", "sham"))
    freeze(
        a.release,
        dict(
            design_sha256=sha(a.design),
            split_sha256=sha(a.split),
            script_sha256=sha(Path(__file__)),
            session_registry_sha256=sha(a.inputs / "sessions.json"),
            eeg_receipt_sha256=features,
            object_ids_in_order=test.tolist(),
            primary=primary,
            same_session=summary(paired, 30),
            advance=gate,
        ),
    )
    print(primary["hits"], primary["teacher_minus"], "advance", gate, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("generic", "panel"), nargs="?", default="generic")
    for key, default in dict(
        design=DIR / "object_center_teacher_design.json",
        split=DIR / "shared_photo_eeg_design.json",
        expanded=Path("outputs/photo-structure/expanded"),
        inputs=Path("outputs/object-transfer"),
        prospective=Path("outputs/prospective-photo"),
        release=DIR / "object_center_teacher.json",
        panel_release=DIR / "object_center_teacher_prospective.json",
    ).items():
        p.add_argument("--" + key.replace("_", "-"), type=Path, default=default)
    run(p.parse_args())
