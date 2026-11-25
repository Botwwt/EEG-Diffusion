"""Train a personal eight-support EEG matcher on same-session generic object episodes."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fit_personal_session_metric import object_directions
from fit_shared_photo_eeg import load_photos, read, require, sha, write
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from test_cohort_session_matched import compact

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
SEED = 260545
ARMS = ("native", "centered", "episodic", "training_sham", "wrong_eeg")
PEOPLE = ("sub-04", "sub-05", "sub-01")
PANELS = ("stress12", "new_three4")


def projection(train, sessions):
    x = train.reshape(-1, train.shape[-1])
    mean, scale = x.mean(0), x.std(0).clip(1e-4)
    grouped = ((x - mean) / scale).reshape(train.shape)
    centers = grouped.mean(1)
    residual = (grouped - centers[:, None]).reshape(x.shape)
    within = residual.T @ residual / len(x)
    centered = centers.copy()
    for session in np.unique(sessions):
        mask = sessions == session
        centered[mask] -= centers[mask].mean(0)
    bases = [object_directions(c.T @ c / len(c), within.copy(), 96) for c in (centers, centered)]
    return mean, scale, np.concatenate(bases, axis=1)


def apply_projection(raw, transform):
    mean, scale, basis = transform
    return ((raw - mean) / scale) @ basis


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)


def matching_features(query, support):
    """Output query × candidate × feature. Each query uses only itself and labeled supports."""
    features = []
    for q, s in zip(np.split(query, 2, axis=-1), np.split(support, 2, axis=-1), strict=True):
        center = s.mean(1)
        features.append(unit(q) @ unit(center).T)
        menu = center.mean(0)
        q, c = q - menu, center - menu
        features.append(unit(q) @ unit(c).T)
        individual = unit(q) @ unit(s - menu).reshape(-1, 96).T
        individual = individual.reshape(len(q), len(s), 8)
        features.append(np.partition(individual, -2, axis=2)[:, :, -2:].mean(2))
        delta2 = (q[:, None] - c[None]) ** 2
        features.append(-delta2.mean(2))
        local = s.var(1, ddof=1)
        var = (0.5 * local + 0.5 * local.mean(0)).clip(1e-8)
        features.append(-0.5 * (delta2 / var + np.log(var)).mean(2))
        for b in range(3):
            band = slice(b * 32, (b + 1) * 32)
            features.append(unit(q[:, band]) @ unit(c[:, band]).T)
    stacked = np.stack(features, axis=2)
    return (stacked - stacked.mean(1, keepdims=True)) / stacked.std(1, keepdims=True).clip(1e-8)


def episodes(projected, ids, sessions, k, seed):
    rng = np.random.default_rng(seed)
    n = len(ids)
    photo_order = np.stack([rng.permutation(10) for _ in ids])
    features, truth, concepts = [], [], []
    for fold in range(5):
        held = slice(2 * fold, 2 * fold + 2)
        support_slots = np.delete(photo_order, np.s_[held], axis=1)
        supports = projected[np.arange(n)[:, None], support_slots]
        for focal in range(n):
            peers = np.flatnonzero((sessions == sessions[focal]) & (np.arange(n) != focal))
            require(len(peers) >= k - 1, "Insufficient same-session peers for fixed menu")
            menu = rng.permutation(np.r_[focal, rng.choice(peers, size=k - 1, replace=False)])
            label = int(np.flatnonzero(menu == focal)[0])
            features.append(
                matching_features(projected[focal, photo_order[focal, held]], supports[menu])
            )
            truth.extend([label, label])
            concepts.extend([ids[focal], ids[focal]])
    return np.concatenate(features), np.array(truth), np.array(concepts)


def oof(train, ids, sessions):
    folds = np.array_split(np.random.default_rng(SEED + 1).permutation(len(ids)), 4)
    data = {4: [], 12: []}
    for fold, held in enumerate(folds):
        fit = np.setdiff1d(np.arange(len(ids)), held)
        transform = projection(train[fit], sessions[fit])
        projected = apply_projection(train[held], transform)
        for k in data:
            data[k].append(
                episodes(projected, ids[held], sessions[held], k, SEED + 100 * (fold + 1) + k)
            )
    return {
        k: tuple(np.concatenate(v) for v in zip(*parts, strict=True)) for k, parts in data.items()
    }


def fit_weights(data, sham):
    prior = np.zeros(16)
    prior[0] = 1
    training = []
    for k, (x, y, _) in data.items():
        if sham:
            shift = np.random.default_rng(SEED + 500 + k).integers(1, k, size=len(y) // 2)
            y = (y + np.repeat(shift, 2)) % k
        training.append((x, y))

    def objective(w):
        delta = w - prior
        loss, grad = 0.025 * (delta @ delta), 0.05 * delta
        for x, y in training:
            logits = x @ w
            ix = np.arange(len(y))
            loss += (logsumexp(logits, axis=1) - logits[ix, y]).mean() / len(training)
            p = softmax(logits, axis=1)
            p[ix, y] -= 1
            grad += np.einsum("ij,ijd->d", p, x) / (len(y) * len(training))
        return loss, grad

    res = minimize(
        objective,
        prior,
        method="L-BFGS-B",
        jac=True,
        options=dict(maxiter=150, ftol=1e-12, gtol=1e-7),
    )
    require(res.success and np.isfinite(res.x).all(), "Conditional-softmax optimization failed")
    return res.x


def readouts(x, w, sham):
    return {
        "native": x[:, :, 0],
        "centered": x[:, :, 1],
        "episodic": x @ w,
        "training_sham": x @ sham,
    }


def held_validation(train, ids, sessions, held):
    train_idx = np.flatnonzero(~held)
    data = oof(train[train_idx], ids[train_idx], sessions[train_idx])
    w, sham = (fit_weights(data, flag) for flag in (False, True))
    transform = projection(train[train_idx], sessions[train_idx])
    z = apply_projection(train[held], transform)
    out = {}
    for k in (4, 12):
        x, y, c = episodes(z, ids[held], sessions[held], k, SEED + 1000 + k)
        out[k] = dict(
            truth=y,
            concept=c,
            **{key: scores.argmax(1) for key, scores in readouts(x, w, sham).items()},
        )
    return out


def panel_decisions(a, sub, transform, w, sham, name, sample, sessions_all):
    concepts = np.array(sample["selected_concepts"])
    reg = sample["registration"]
    ref = SimpleNamespace(out=a.panels, double=a.double)
    physical, receipt = compact(ref, sub, name, concepts)
    rr = (10 * concepts[:, None] + np.arange(10)).ravel()
    order = np.argsort(rr)
    rr, p = rr[order], apply_projection(physical[order], transform)
    rows, folds, cal = (
        np.array(reg[key]) for key in ("rows", "folds", "calibration_rows_by_fold_and_word")
    )
    k = 12 if name == "stress12" else 4
    predictions = {key: np.full(len(rows), -1) for key in ARMS}
    donor_rows = np.full(len(rows), -1)
    rng = np.random.default_rng(SEED + 2000 + k)
    for f in range(5):
        for block in np.arange(12).reshape(-1, k):
            support_ids = cal[f, block]
            menu_concepts = np.unique(support_ids // 10)
            require(
                len(np.unique(sessions_all[menu_concepts], axis=0)) == 1
                and len(menu_concepts) == k,
                "Menu no longer contains the registered same-session concepts",
            )
            # Only the supplied menu and fold route a query. The scoring function never receives truth.
            ix = np.flatnonzero((folds == f) & np.isin(rows // 10, menu_concepts))
            ix = ix[np.argsort(rows[ix])]
            si, qi = np.searchsorted(rr, support_ids), np.searchsorted(rr, rows[ix])
            require(
                len(ix) == 2 * k
                and np.array_equal(rr[si], support_ids)
                and np.array_equal(rr[qi], rows[ix])
                and not np.intersect1d(support_ids, rows[ix]).size
                and len(np.unique(support_ids)) == 8 * k,
                "Invalid query/support identity",
            )
            x = matching_features(p[qi], p[si])
            scores = readouts(x, w, sham)
            donor = np.roll(np.arange(len(ix)), rng.integers(1, len(ix)))
            donor_rows[ix] = rows[ix[donor]]
            scores["wrong_eeg"] = scores["episodic"][donor]
            for key, value in scores.items():
                predictions[key][ix] = block[value.argmax(1)]
    require(
        all(np.all(v >= 0) for v in predictions.values()) and (rows != donor_rows).all(),
        "Missing prediction or sham fixed point",
    )
    old = a.panels / name / "predictions.npz"
    legacy = ("ordinary_12way" if k == 12 else "within_matched_four") + "_" + sub
    with np.load(old, allow_pickle=False) as z:
        require(
            np.array_equal(z["rows"], rows)
            and np.array_equal(z["folds"], folds)
            and np.array_equal(z[legacy], predictions["native"]),
            "Legacy Fisher replay differs",
        )
    return dict(
        rows=rows,
        truth=np.array(reg["truth"]),
        concept=rows // 10,
        folds=folds,
        donor_rows=donor_rows,
        **predictions,
    ), dict(
        compact_sha256=receipt,
        original_predictions_sha256=sha(old),
        exact_original_decisions=len(rows),
    )


def fit(a, sub, lock, registered, ids):
    subject_dir = a.out / sub
    subject_dir.mkdir(parents=True, exist_ok=True)
    meta = a.expanded / "features" / (sub + ".json")
    require(np.array_equal(read(meta)["training_concepts"], ids), "Training concept list changed")
    train = load_photos(a, sub, ids)
    ss = np.asarray(read(a.sessions)["subjects"][sub]["session_pairs"])
    _, sessions = np.unique(ss[ids], axis=0, return_inverse=True)
    held = np.isin(ids, np.random.default_rng(SEED).choice(ids, 96, replace=False))
    val = held_validation(train, ids, sessions, held)
    for k, data in val.items():
        np.savez(subject_dir / f"generic_{k}.npz", **data)
    print(sub, "generic validation complete", flush=True)
    d = oof(train, ids, sessions)
    w, sham = (fit_weights(d, flag) for flag in (False, True))
    transform = projection(train, sessions)
    np.savez(
        subject_dir / "model.npz",
        classifier=w,
        sham=sham,
        mean=transform[0],
        scale=transform[1],
        basis=transform[2],
        train_ids=ids,
        generic_heldout_ids=ids[held],
    )
    receipts = {}
    for name in PANELS:
        data, receipts[name] = panel_decisions(
            a, sub, transform, w, sham, name, registered["panels"][name], ss
        )
        np.savez(subject_dir / (name + ".npz"), **data)
        print(sub, name, {key: int((data[key] == data["truth"]).sum()) for key in ARMS}, flush=True)
    write(
        subject_dir / "receipt.json",
        dict(
            subject=sub,
            design_sha256=sha(lock),
            source_sha256=sha(Path(__file__)),
            training_receipt_sha256=sha(meta),
            model_sha256=sha(subject_dir / "model.npz"),
            panels=receipts,
            prediction_sha256={
                key: sha(subject_dir / (key + ".npz"))
                for key in (*PANELS, "generic_4", "generic_12")
            },
        ),
    )


def interval(diff, boot):
    denominator = len(diff) * len(PEOPLE) * 10
    return dict(
        delta_accuracy=float(diff.sum() / denominator),
        object_bootstrap_ci95=np.quantile(diff[boot].sum(1) / denominator, [0.025, 0.975]).tolist(),
    )


def summarize(a, design_path):
    results, object_diffs, receipts = {}, {}, {}
    for sub in PEOPLE:
        receipts[sub] = read(a.out / sub / "receipt.json")
        require(
            receipts[sub]["source_sha256"] == sha(Path(__file__))
            and receipts[sub]["design_sha256"] == sha(design_path),
            "Design or program changed between scoring and summary",
        )
    for case in ("generic_4", "generic_12", *PANELS):
        masks, counts = {}, {}
        generic = case.startswith("generic_")
        arms = ARMS[:-1] if generic else ARMS
        ids = None
        for sub in PEOPLE:
            file = a.out / sub / (case + ".npz")
            require(
                sha(file) == receipts[sub]["prediction_sha256"][case],
                "Prediction packet has changed",
            )
            with np.load(file, allow_pickle=False) as z:
                concepts = z["concept"]
                sorted_ids, n = np.unique(concepts, return_counts=True)
                require(
                    np.all(n == 10) and (ids is None or np.array_equal(ids, sorted_ids)),
                    "Object coverage differs",
                )
                ids = sorted_ids
                masks[sub] = {
                    key: np.array(
                        [(z[key][concepts == c] == z["truth"][concepts == c]).sum() for c in ids]
                    )
                    for key in arms
                }
                counts[sub] = {key: int((z[key] == z["truth"]).sum()) for key in arms}
        hits_by_object = {key: np.stack([masks[s][key] for s in PEOPLE]).sum(0) for key in arms}
        boot = np.random.default_rng(SEED + 3000).integers(len(ids), size=(10000, len(ids)))
        object_diffs[case] = {
            key: hits_by_object["episodic"] - hits_by_object[key]
            for key in arms
            if key != "episodic"
        }
        results[case] = dict(
            objects=len(ids),
            queries_per_person=int(n.sum()),
            people=list(PEOPLE),
            candidates=(4 if case in ("generic_4", "new_three4") else 12),
            hits=counts,
            sums={k: sum(counts[s][k] for s in PEOPLE) for k in arms},
            paired_episodic_minus={key: interval(v, boot) for key, v in object_diffs[case].items()},
        )
    rng = np.random.default_rng(SEED + 4000)
    bootstrap = [rng.integers(12, size=(10000, 12)) for _ in PANELS]
    combined = {}
    for key in object_diffs[PANELS[0]]:
        diffs = [object_diffs[case][key] for case in PANELS]
        # Sum hits as integers before dividing so a bootstrap draw with zero net hits is exactly zero.
        denominator = sum(len(v) for v in diffs) * len(PEOPLE) * 10
        bootstrap_means = (
            sum(v[b].sum(1) for v, b in zip(diffs, bootstrap, strict=True)) / denominator
        )
        combined[key] = dict(
            delta_accuracy=float(sum(v.sum() for v in diffs) / denominator),
            panel_stratified_object_bootstrap_ci95=np.quantile(
                bootstrap_means, [0.025, 0.975]
            ).tolist(),
        )
    gate = (
        all(object_diffs[p]["native"].mean() >= 0 for p in PANELS)
        and all(
            combined[key]["panel_stratified_object_bootstrap_ci95"][0] > 0
            for key in ("native", "training_sham")
        )
        and combined["centered"]["delta_accuracy"] > 0
    )
    result = dict(
        design_sha256=sha(design_path),
        source_sha256=sha(Path(__file__)),
        scope="Developmental reuse of two previously scored menus, three individually trained and tested R4 participants. Bootstrap draws resample object identities shared across participants and preserve both menu-size strata.",
        checks="Exact historical single-participant Fisher decisions for all 720 participant-photo queries; generic and both panel concept sets disjoint.",
        results=results,
        combined_episodic_minus=combined,
        advance=bool(gate),
        receipts=receipts,
    )
    write(a.out / "result.json", result)
    write(PUBLIC / "episodic_object.json", result)
    print("Combined", combined, "advance", bool(gate), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fit", "summarize"))
    p.add_argument("--subject", choices=PEOPLE)
    for key, default in dict(
        out="outputs/episodic-object",
        panels="outputs/session-matched-cohort",
        double="outputs/double-calibration",
        expanded="outputs/photo-structure/expanded",
        sessions="outputs/object-transfer/sessions.json",
    ).items():
        p.add_argument("--" + key, type=Path, default=Path(default))
    a = p.parse_args()
    lock = PUBLIC / "episodic_object_design.json"
    design = read(lock)
    require(
        sha(a.panels / "design.json") == design["panels_sha256"]
        and sha(a.double / "design.json") == design["training_universe_sha256"]
        and sha(a.sessions) == design["session_schedule_sha256"],
        "Registered sources changed",
    )
    a.out.mkdir(exist_ok=True, parents=True)
    snapshot = a.out / "design.json"
    if not snapshot.exists():
        snapshot.write_bytes(lock.read_bytes())
    require(sha(snapshot) == sha(lock), "Private registration has changed")
    registered = read(a.panels / "design.json")
    ids = np.asarray(read(a.double / "design.json")["old_objects"])
    require(
        len(np.unique(ids)) == 512
        and np.array_equal(ids, np.sort(ids))
        and all(
            not np.intersect1d(ids, registered["panels"][n]["selected_concepts"]).size
            for n in PANELS
        ),
        "Evaluation class entered training",
    )
    if a.stage == "fit":
        for sub in (a.subject,) if a.subject else PEOPLE:
            fit(a, sub, lock, registered, ids)
    else:
        summarize(a, lock)


if __name__ == "__main__":
    main()
