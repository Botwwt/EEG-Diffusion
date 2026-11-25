"""Test a single-person EEG object matcher on hard same-session candidate pairs."""

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fit_episodic_object import (
    PANELS,
    PEOPLE,
    PUBLIC,
    SEED,
    apply_projection,
    fit_weights,
    matching_features,
    projection,
    unit,
)
from fit_shared_photo_eeg import load_photos, read, require, sha
from register_fresh_photo_panel import freeze
from replay_cohort_candidate_menus import MENUS, canonical_json
from scipy.optimize import minimize
from scipy.special import expit
from test_cohort_session_matched import compact

DESIGN = PUBLIC / "hard_object_pairs_design.json"
DESIGN_SHA = "7efc5a919234e7d38279bfe1bdff1ffcb6d4d2cc5c2087d355316f9b0d187cf3"
ARMS = ("native", "episodic", "hard_pairs", "orientation_sham", "wrong_eeg")


def pair_features(q, s, focal=None):
    """Signed evidence for a candidate over each of its two support-defined closest rivals."""
    k = len(s)
    means = s.mean(1)
    similarity = unit(means[:, :96]) @ unit(means[:, :96]).T
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.argsort(-similarity, axis=1, kind="stable")[:, :2]
    candidates = np.arange(k) if focal is None else np.array([focal])
    left = np.repeat(candidates, 2)
    right = nearest[candidates].ravel()
    center = means.reshape(k, 6, 32)
    noise = s.reshape(k, 8, 6, 32) - center[:, None]
    var = noise.var(1, ddof=1)
    pooled = (var[left] + var[right]) / 2
    delta = center[left] - center[right]
    inv_weighted = delta / ((pooled + pooled.mean(-1, keepdims=True)) / 2).clip(1e-10)
    direction = unit(inv_weighted)
    u = np.einsum("qbd,pbd->qpb", q.reshape(-1, 6, 32), direction)
    u -= np.sum((center[left] + center[right]) * direction, axis=-1) / 2
    h = np.sum(delta * direction, axis=-1) / 2
    va = np.einsum("psbd,pbd->psb", noise[left], direction).var(1, ddof=1)
    vb = np.einsum("psbd,pbd->psb", noise[right], direction).var(1, ddof=1)
    vp = ((va + vb) / 2).clip(1e-10)
    linear = u / np.sqrt(vp * (9 / 8))
    va, vb = ((v + vp) * (9 / 16) for v in (va, vb))
    gaussian = (np.log(vb / va) + (u + h) ** 2 / vb - (u - h) ** 2 / va) / 2
    phi = np.stack((linear, gaussian), axis=-1).clip(-4, 4)
    return phi.reshape(len(q), len(candidates), 2, 12), nearest[candidates]


def episodes(z, ids, sessions, k, seed):
    """Exact original RNG sequence for menus, held query slots, and calibration slots."""
    rng = np.random.default_rng(seed)
    n = len(ids)
    order = np.stack([rng.permutation(10) for _ in ids])
    for fold in range(5):
        held = slice(2 * fold, 2 * fold + 2)
        support = np.delete(order, np.s_[held], axis=1)
        for focal in range(n):
            peers = np.flatnonzero((sessions == sessions[focal]) & (np.arange(n) != focal))
            require(len(peers) >= k - 1, "Insufficient same-session peers")
            menu = rng.permutation(np.r_[focal, rng.choice(peers, size=k - 1, replace=False)])
            label = int(np.flatnonzero(menu == focal)[0])
            donor = menu[(label + 1) % k]
            qr, dr = 10 * ids[focal] + order[focal, held], 10 * ids[donor] + order[donor, held]
            sr = 10 * ids[menu, None] + support[menu]
            require(not np.isin(np.r_[qr, dr], sr).any(), "An active query entered support")
            yield (
                z[focal, order[focal, held]],
                z[donor, order[donor, held]],
                z[menu[:, None], support[menu]],
                label,
                qr,
                dr,
                sr,
            )


def fit_pairs(parts, sham):
    training = []
    for k, (offset, features) in parts.items():
        # Each focal episode supplies two held photos and the same two rivals.
        rng = np.random.default_rng(260550 + k)
        signs = rng.choice([-1, 1], size=len(offset) // 4).repeat(4) if sham else 1
        training.append((offset, features, signs))

    def objective(w):
        loss = 0.05 * (w @ w)
        grad = 0.1 * w
        for offset, x, signs in training:
            pos = signs * (offset + x @ w)
            loss += np.logaddexp(0, -pos).mean() / len(training)
            grad += (-(signs * expit(-pos)) @ x) / (len(x) * len(training))
        return loss, grad

    opt = minimize(
        objective,
        np.zeros(12),
        method="L-BFGS-B",
        jac=True,
        options=dict(maxiter=150, ftol=1e-12, gtol=1e-7),
    )
    require(opt.success and np.isfinite(opt.x).all(), "Hard-pair fit failed")
    return opt.x


def train_matchers(train, ids, sessions):
    old, pairs = {4: [], 12: []}, {4: [], 12: []}
    folds = np.array_split(np.random.default_rng(SEED + 1).permutation(len(ids)), 4)
    for fold, held in enumerate(folds):
        fit = np.setdiff1d(np.arange(len(ids)), held)
        z = apply_projection(train[held], projection(train[fit], sessions[fit]))
        for k in old:
            for q, _, s, label, qr, _, _ in episodes(
                z, ids[held], sessions[held], k, SEED + 100 * (fold + 1) + k
            ):
                x = matching_features(q, s)
                phi, rivals = pair_features(q, s, label)
                gap = x[:, label, 0, None] - x[:, rivals[0], 0]
                pairs[k].append((gap.ravel(), phi.reshape(-1, 12)))
                old[k].append((x, np.full(len(q), label), qr // 10))

    def join(parts):
        return {k: tuple(np.concatenate(x) for x in zip(*v, strict=True)) for k, v in parts.items()}

    original = fit_weights(join(old), False)
    training = join(pairs)
    return original, fit_pairs(training, False), fit_pairs(training, True)


def prediction(q, s, weights):
    x = matching_features(q, s)
    prior = x @ weights[0]
    normalized = (prior - prior.mean(1, keepdims=True)) / prior.std(1, keepdims=True).clip(1e-8)
    phi, _ = pair_features(q, s)
    return dict(
        native=x[:, :, 0].argmax(1),
        episodic=prior.argmax(1),
        hard_pairs=(normalized + 0.5 * (phi @ weights[1]).mean(2)).argmax(1),
        orientation_sham=(normalized + 0.5 * (phi @ weights[2]).mean(2)).argmax(1),
    )


def evaluate_generic(z, ids, sessions, k, w):
    records = []
    for q, wrong, s, label, qr, dr, sr in episodes(z, ids, sessions, k, SEED + 1000 + k):
        predictions = prediction(np.r_[q, wrong], s, w)
        records.append(
            dict(
                rows=qr,
                donor_rows=dr,
                truth=np.full(2, label),
                concept=qr // 10,
                support_rows=np.repeat(sr[None], 2, axis=0),
                wrong_eeg=predictions["hard_pairs"][2:],
                **{v: predictions[v][:2] for v in predictions},
            )
        )
    return {k: np.concatenate([v[k] for v in records]) for k in records[0]}


def panel(a, sub, name, spec, menus, transform, w, historical):
    reg = spec["registration"]

    require(
        hashlib.sha256(canonical_json(reg)).hexdigest() == menus["registration_sha256"],
        "Registration differs from ordinal menu lock",
    )
    option = "ordinary_12way" if name == "stress12" else "within_matched_four"
    menu = menus["decisions"][option]
    alternatives, assigned = np.array(menu["alternatives"]), np.array(menu["menu_at_ordinal"])
    concepts = np.array(spec["selected_concepts"])
    x, receipt = compact(SimpleNamespace(out=a.panels, double=a.double), sub, name, concepts)
    require(
        receipt == historical["panels"][name]["compact_sha256"], "Physical target receipt differs"
    )
    raw = (10 * concepts[:, None] + np.arange(10)).ravel()
    rr = np.sort(raw)
    z = apply_projection(x[np.argsort(raw)], transform)
    rows, folds, cal = (
        np.array(reg[k]) for k in ("rows", "folds", "calibration_rows_by_fold_and_word")
    )
    require(
        all(np.array_equal(np.sort(cal[f].ravel()), np.sort(rows[folds != f])) for f in range(5)),
        "Each support pool must contain exactly the other four folds",
    )
    values = {k: np.full(len(rows), -1) for k in ARMS}
    donors = np.full(len(rows), -1)
    support_rows = np.empty((len(rows), alternatives.shape[1], 8), dtype=np.int64)
    rng = np.random.default_rng(SEED + 2000 + alternatives.shape[1])
    for f in range(5):
        for group, choices in enumerate(alternatives):
            ix = np.flatnonzero((folds == f) & (assigned == group))
            ix = ix[np.argsort(rows[ix])]
            sr = cal[f, choices]
            sidx, qidx = np.searchsorted(rr, sr), np.searchsorted(rr, rows[ix])
            require(
                len(ix) == 2 * len(choices)
                and np.array_equal(rr[sidx], sr)
                and np.array_equal(rr[qidx], rows[ix])
                and not np.intersect1d(sr, rows[ix]).size,
                "Invalid recipient ordinal or support row",
            )
            data = prediction(z[qidx], z[sidx], w)
            donor = np.roll(np.arange(len(ix)), rng.integers(1, len(ix)))
            donors[ix] = rows[ix[donor]]
            support_rows[ix] = sr
            for key, pred in data.items():
                values[key][ix] = choices[pred]
            values["wrong_eeg"][ix] = choices[data["hard_pairs"][donor]]
    require(
        all((v >= 0).all() for v in values.values()) and (donors != rows).all(), "Unresolved query"
    )
    return dict(
        rows=rows,
        folds=folds,
        concept=rows // 10,
        truth=np.asarray(reg["truth"]),
        support_rows=support_rows,
        donor_rows=donors,
        **values,
    )


def check_and_save(a, sub, name, data, prev):
    old = a.previous / sub / (name + ".npz")
    require(sha(old) == prev["prediction_sha256"][name], "Historical packet differs")
    with np.load(old, allow_pickle=False) as z:
        for k in (
            "truth",
            "concept",
            "native",
            "episodic",
            *(() if name.startswith("generic") else ("rows", "folds", "donor_rows")),
        ):
            require(np.array_equal(z[k], data[k]), f"{sub} {name}: original {k} not reproduced")
    dest = a.out / sub / (name + ".npz")
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **data)
    return sha(dest)


def fit(a, sub, ids, registered, menus, history):
    prev = history["receipts"][sub]
    meta = a.expanded / "features" / (sub + ".json")
    require(
        sha(meta) == prev["training_receipt_sha256"]
        and np.array_equal(read(meta)["training_concepts"], ids),
        "Training receipt or IDs differ",
    )
    train = load_photos(a, sub, ids)
    ss = np.asarray(read(a.sessions)["subjects"][sub]["session_pairs"])
    _, sessions = np.unique(ss[ids], axis=0, return_inverse=True)
    held = np.isin(ids, np.random.default_rng(SEED).choice(ids, 96, replace=False))
    hashes = {}
    for subset in ("generic", "panels"):
        use = ~held if subset == "generic" else np.ones(len(ids), dtype=bool)
        weights = train_matchers(train[use], ids[use], sessions[use])
        base = projection(train[use], sessions[use])
        if subset == "generic":
            z = apply_projection(train[held], base)
            for k in (4, 12):
                name = f"generic_{k}"
                data = evaluate_generic(z, ids[held], sessions[held], k, weights)
                hashes[name] = check_and_save(a, sub, name, data, prev)
        else:
            for name in PANELS:
                data = panel(
                    a,
                    sub,
                    name,
                    registered["panels"][name],
                    menus["panels"][name],
                    base,
                    weights,
                    prev,
                )
                hashes[name] = check_and_save(a, sub, name, data, prev)
        print(sub, subset, "validated", flush=True)
    freeze(
        a.out / sub / "receipt.json",
        dict(
            subject=sub,
            design_sha256=sha(DESIGN),
            source_sha256=sha(Path(__file__)),
            historical_episodic_model_sha256=prev["model_sha256"],
            predictions_sha256=hashes,
        ),
    )


def paired_report(by_object, boot):
    """Lists hold independent panel strata; columns align object IDs across people and menus."""
    hit = sum(v.sum((1, 2)) for v in by_object)
    denominator = sum(v.shape[2] * len(PEOPLE) * 10 for v in by_object)
    primary = ARMS.index("hard_pairs")
    contrasts = {}
    for i, name in enumerate(ARMS):
        if i == primary:
            continue
        differences = [v[primary].sum(0) - v[i].sum(0) for v in by_object]
        draws = sum(d[b].sum(1) for d, b in zip(differences, boot, strict=True))
        contrasts[name] = dict(
            delta=float((hit[primary] - hit[i]) / denominator),
            ci95=np.quantile(draws / denominator, [0.025, 0.975]).tolist(),
        )
    return dict(
        n=denominator,
        hits=dict(zip(ARMS, hit.astype(int).tolist(), strict=True)),
        hard_pairs_minus=contrasts,
    )


def summarize(a):
    cases = ("generic_4", "generic_12", *PANELS)
    receipts = {s: read(a.out / s / "receipt.json") for s in PEOPLE}
    rng = np.random.default_rng(260551)
    tally, report, objects = {}, {}, {}
    for name in cases:
        people = []
        for s in PEOPLE:
            path = a.out / s / (name + ".npz")
            receipt = receipts[s]
            require(
                receipt["source_sha256"] == sha(Path(__file__))
                and receipt["design_sha256"] == sha(DESIGN)
                and receipt["predictions_sha256"][name] == sha(path),
                "Different program/design/packet",
            )
            with np.load(path, allow_pickle=False) as z:
                ids, counts = np.unique(z["concept"], return_counts=True)
                require(
                    (counts == 10).all()
                    and (name not in objects or np.array_equal(objects[name], ids)),
                    "Incomplete object",
                )
                objects[name] = ids
                hit = np.stack([z[key] == z["truth"] for key in ARMS])
                people.append(np.stack([hit[:, z["concept"] == c].sum(1) for c in ids], axis=1))
        tally[name] = np.stack(people, axis=1)
        n = len(objects[name])
        report[name] = paired_report([tally[name]], [rng.integers(n, size=(10000, n))])
        report[name]["by_subject"] = {
            s: dict(n=n * 10, hits=dict(zip(ARMS, v.sum(1).astype(int).tolist(), strict=True)))
            for s, v in zip(PEOPLE, people, strict=True)
        }
    require(
        np.array_equal(objects["generic_4"], objects["generic_12"])
        and not set(objects[PANELS[0]]) & set(objects[PANELS[1]]),
        "Invalid shared object strata",
    )
    # Use the same draw at both menu widths for the identical 96 test objects.
    same = rng.integers(96, size=(10000, 96))
    generic = paired_report([tally[k] for k in cases[:2]], [same, same])
    panels = paired_report(
        [tally[k] for k in PANELS], [rng.integers(12, size=(10000, 12)) for _ in PANELS]
    )
    gates = dict(
        generic_positive_lower_vs_native_episodic_and_sham=all(
            generic["hard_pairs_minus"][k]["ci95"][0] > 0
            for k in ("native", "episodic", "orientation_sham")
        ),
        generic_each_width_nonnegative=all(
            report[k]["hard_pairs_minus"]["episodic"]["delta"] >= 0 for k in cases[:2]
        ),
        target_each_panel_nonnegative=all(
            report[k]["hard_pairs_minus"]["episodic"]["delta"] >= 0 for k in PANELS
        ),
        target_positive_lower_vs_native_and_episodic=all(
            panels["hard_pairs_minus"][k]["ci95"][0] > 0 for k in ("native", "episodic")
        ),
    )
    result = dict(
        design_sha256=sha(DESIGN),
        source_sha256=sha(Path(__file__)),
        participants=list(PEOPLE),
        uncertainty="Synchronized whole-object paired percentage differences (fraction units); three fixed participants. Same generic classes resampled together for both menu widths; independent panel strata.",
        checks="All native and incumbent episodic predictions replayed at every row. Wrong EEG from same supplied menu and fold. Target 4-way vocabulary provided by registered ordinal. All panels and generic check split previously studied.",
        results=report,
        generic_pooled=generic,
        panels_pooled=panels,
        gate=gates,
        advance=all(gates.values()),
        receipts=receipts,
    )
    freeze(a.out / "result.json", result)
    freeze(PUBLIC / "hard_object_pairs.json", result)
    print("Advance", result["advance"], "generic", generic, "panels", panels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["fit", "summarize"])
    p.add_argument("--subject", choices=PEOPLE)
    for key, default in dict(
        out="outputs/hard-object-pairs",
        previous="outputs/episodic-object",
        panels="outputs/session-matched-cohort",
        double="outputs/double-calibration",
        expanded="outputs/photo-structure/expanded",
        sessions="outputs/object-transfer/sessions.json",
    ).items():
        p.add_argument("--" + key, type=Path, default=Path(default))
    a = p.parse_args()
    require(sha(DESIGN) == DESIGN_SHA, "Different hard-pair registration")
    design = read(DESIGN)
    files = dict(
        previous_design=PUBLIC / "episodic_object_design.json",
        previous_program=Path(__file__).with_name("fit_episodic_object.py"),
        previous_aggregates_and_receipts=PUBLIC / "episodic_object.json",
        ordinal_menus=MENUS,
        panels_sha256=a.panels / "design.json",
        training_universe_sha256=a.double / "design.json",
        session_schedule_sha256=a.sessions,
    )
    require(
        all(sha(f) == design["sources"][k] for k, f in files.items()), "Registered inputs differ"
    )
    if a.stage == "summarize":
        summarize(a)
        return
    reg = read(a.panels / "design.json")
    ids = np.asarray(read(a.double / "design.json")["old_objects"])
    target = [set(reg["panels"][name]["selected_concepts"]) for name in PANELS]
    require(
        len(ids) == len(set(ids)) == 512
        and np.all(np.diff(ids) > 0)
        and all(not set(ids) & k for k in target)
        and not target[0] & target[1],
        "Target object leaked into training",
    )
    for s in (a.subject,) if a.subject else PEOPLE:
        fit(a, s, ids, reg, read(MENUS), read(PUBLIC / "episodic_object.json"))


if __name__ == "__main__":
    main()
