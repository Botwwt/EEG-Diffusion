"""Score fixed single-person trial-encoder EEG embeddings on registered same-session object menus."""

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fit_episodic_object import (
    PANELS,
    PUBLIC,
    SEED,
    apply_projection,
    fit_weights,
    matching_features,
    oof,
    projection,
)
from fit_hard_object_pairs import episodes
from fit_personal_session_metric import similarities
from fit_shared_photo_eeg import load_photos, read, require, sha
from register_fresh_photo_panel import freeze
from replay_cohort_candidate_menus import MENUS, canonical_json
from run_prospective_readout import frozen_npz
from test_cohort_session_matched import compact

SUB = "sub-04"
ARMS = ("fisher", "episodic", "encoder", "mixed", "wrong_eeg")
BOOT_SEED = 260552


def embedded(a, stage, training, held):
    folder = a.root / stage
    receipt_path, packet = folder / "receipt.json", folder / "embeddings.npz"
    meta = read(receipt_path)
    require(
        meta["subject"] == SUB
        and set(meta["train_classes"]) == set(training)
        and len(meta["train_classes"]) == len(training)
        and set(meta["test_classes"]) == set(held)
        and len(meta["test_classes"]) == len(held)
        and not set(training) & set(held),
        "The trial encoder must exclude every evaluation object from fitting",
    )
    require(
        meta["packet_sha256"] == sha(packet)
        and meta["source_sha256"] == sha(a.training_source)
        and meta["design_sha256"] == sha(a.design),
        "Trial encoder receipt differs from the packet, driver or design",
    )
    with np.load(packet, allow_pickle=False) as z:
        require(set(z.files) == {"ids", "features"}, "Unexpected embedding file members")
        ids, x = z["ids"], z["features"]
    require(np.array_equal(ids, meta["test_classes"]), "Packet order differs from its receipt")
    if stage == "generic":
        require(np.array_equal(ids, held), "Generic IDs must follow the archived calibration order")
    if "weights_sha256" in meta:
        require(
            sha(folder / "model.safetensors") == meta["weights_sha256"], "Changed model weights"
        )
    require(
        ids.ndim == 1
        and ids.dtype.kind in "iu"
        and len(ids) == len(held)
        and np.array_equal(np.sort(ids), np.sort(held))
        and x.dtype == np.float32
        and x.shape == (len(ids), 10, 96)
        and np.isfinite(x).all()
        and np.allclose(np.linalg.norm(x, axis=-1), 1, rtol=0, atol=1e-5),
        "Invalid trial embedding identities, shape, values, or norm",
    )
    return x[np.argsort(ids)].astype(float), dict(
        encoder_receipt_sha256=sha(receipt_path),
        encoder_sha256=meta["source_sha256"],
        design_sha256=sha(a.design),
        packet_sha256=sha(packet),
    )


def scores(q, s, deep_q, deep_s, w):
    native = similarities(q[:, :96], s[:, :, :96])
    deep = similarities(deep_q, deep_s)
    old = matching_features(q, s) @ w
    return dict(fisher=native, episodic=old, encoder=deep, mixed=(native + deep) / 2)


def episode_record(ordinal, rows, donor, supports, slots, fold, raw_rows, original, deep, w):
    rows, donor = np.asarray(rows), np.asarray(donor)
    n = len(rows)
    require(
        supports.shape == (len(slots), 8)
        and np.unique(supports).size == supports.size
        and np.all(supports // 10 == supports[:, :1] // 10)
        and len(set(supports[:, 0] // 10)) == len(slots)
        and not np.isin(np.r_[rows, donor], supports).any()
        and (rows != donor).all(),
        "Active EEG or duplicate support in the calibration menu",
    )

    def index(v):
        i = np.searchsorted(raw_rows, v)
        require(np.all(i < len(raw_rows)) and np.array_equal(raw_rows[i], v), "Missing EEG row")
        return i

    qi, si = index(np.r_[rows, donor]), index(supports)
    # Fully replace the query waveform in both branches, preserving its original supports and menu.
    v = scores(original[qi], original[si], deep[qi], deep[si], w)
    groups = supports[:, 0] // 10
    result = dict(
        ordinal=np.asarray(ordinal),
        rows=rows,
        donor_rows=donor,
        folds=np.full(n, fold),
        candidate_ids=np.broadcast_to(groups, (n, len(groups))),
        support_rows=np.broadcast_to(supports, (n, *supports.shape)),
    )
    for name, z in {**{k: val[:n] for k, val in v.items()}, "wrong_eeg": v["mixed"][n:]}.items():
        pick = z.argmax(1)
        result[name] = groups[pick]
        result[name + "_slot"] = slots[pick]
        result[name + "_scores"] = z
    return result


def combine(records):
    data = {k: np.concatenate([v[k] for v in records]) for k in records[0]}
    order = np.argsort(data["ordinal"])
    require(np.array_equal(data["ordinal"][order], np.arange(len(order))), "Missing ordinal")
    data = {k: v[order] for k, v in data.items()}
    data["truth_id"] = data["rows"] // 10
    data["predicted_ids"] = data["mixed"]
    chosen = data["candidate_ids"] == data["predicted_ids"][:, None]
    require(np.all(chosen.sum(1) == 1), "Mixed prediction outside supplied menu")
    data["predicted_support_rows"] = data["support_rows"][chosen]
    require(
        len(np.unique(data["rows"])) == len(order)
        and (data["candidate_ids"] == data["truth_id"][:, None]).sum(1).min() == 1,
        "Duplicate photo or truth is absent from the scored candidate set",
    )
    return data


def legacy_check(a, name, data, subject_receipt, truth_slot):
    path = a.previous / SUB / (name + ".npz")
    require(
        sha(path) == subject_receipt["prediction_sha256"][name], "Historical predictions changed"
    )
    with np.load(path, allow_pickle=False) as old:
        for oldkey, value in (
            ("concept", data["truth_id"]),
            ("truth", truth_slot),
            ("native", data["fisher_slot"]),
            ("episodic", data["episodic_slot"]),
        ):
            require(np.array_equal(old[oldkey], value), f"{name}: historical {oldkey} differs")
        if name in PANELS:
            for key in ("rows", "folds", "donor_rows"):
                require(np.array_equal(old[key], data[key]), f"{name}: historical {key} differs")


def context(a):
    hist = read(PUBLIC / "episodic_object.json")
    baseline = read(PUBLIC / "episodic_object_design.json")
    require(
        sha(PUBLIC / "episodic_object_design.json") == hist["design_sha256"]
        and sha(a.sessions) == baseline["session_schedule_sha256"]
        and sha(a.double / "design.json") == baseline["training_universe_sha256"]
        and sha(a.panels / "design.json") == baseline["panels_sha256"],
        "Different legacy design or sources",
    )
    prev = hist["receipts"][SUB]
    meta = a.expanded / "features" / (SUB + ".json")
    require(sha(meta) == prev["training_receipt_sha256"], "Historical calibration receipt differs")
    ids = np.asarray(read(a.double / "design.json")["old_objects"])
    require(
        len(ids) == 512
        and np.all(np.diff(ids) > 0)
        and np.array_equal(read(meta)["training_concepts"], ids),
        "Different calibration subjects or objects",
    )
    held = np.isin(ids, np.random.default_rng(SEED).choice(ids, 96, replace=False))
    return prev, ids, held


def generic(a, previous, ids, held, selftest):
    selected = ids[held]
    if selftest:
        rng = np.random.default_rng(75801)
        mock = rng.normal(size=(96, 10, 96))
        deep = mock / np.linalg.norm(mock, axis=-1, keepdims=True)
        receipt = {}
    else:
        deep, receipt = embedded(a, "generic", ids[~held], selected)
    full = load_photos(a, SUB, ids)
    ss = np.asarray(read(a.sessions)["subjects"][SUB]["session_pairs"])
    _, session = np.unique(ss[ids], axis=0, return_inverse=True)
    w = fit_weights(oof(full[~held], ids[~held], session[~held]), False)
    projected = apply_projection(full[held], projection(full[~held], session[~held]))
    raw = selected[:, None] * 10 + np.arange(10)
    out = {}
    for k in (4, 12):
        rec = []
        truth = []
        for j, (_, _, _, label, rr, dr, sr) in enumerate(
            episodes(raw, selected, session[held], k, SEED + 1000 + k)
        ):
            rec.append(
                episode_record(
                    np.arange(2 * j, 2 * j + 2),
                    rr,
                    dr,
                    sr,
                    np.arange(k),
                    j // len(selected),
                    raw.ravel(),
                    projected.reshape(-1, 192),
                    deep.reshape(-1, 96),
                    w,
                )
            )
            truth.extend([label, label])
        name = f"generic_{k}"
        data = combine(rec)
        legacy_check(a, name, data, previous, np.array(truth))
        out[name] = data
    return out, receipt


def target(a, previous, ids):
    panel_design = read(a.panels / "design.json")["panels"]
    classes = np.sort(
        np.concatenate([np.array(panel_design[name]["selected_concepts"]) for name in PANELS])
    )
    require(
        len(set(classes)) == 24 and not set(classes) & set(ids), "Target/calibration object overlap"
    )
    deep, receipt = embedded(a, "target", ids, classes)
    old_model = a.previous / SUB / "model.npz"
    require(sha(old_model) == previous["model_sha256"], "Legacy full training model changed")
    with np.load(old_model, allow_pickle=False) as z:
        require(np.array_equal(z["train_ids"], ids), "Legacy model training objects differ")
        transform = z["mean"], z["scale"], z["basis"]
        w = z["classifier"]
    registry = read(MENUS)["panels"]
    require(
        sha(MENUS) == read(PUBLIC / "cohort_session_menu_replay.json")["menu_sha256"],
        "Ordinal menu file changed",
    )
    out = {}
    for name in PANELS:
        spec = panel_design[name]
        reg = spec["registration"]
        menu = registry[name]
        require(
            hashlib.sha256(canonical_json(reg)).hexdigest() == menu["registration_sha256"],
            "Screened menu registration differs",
        )
        choices = menu["decisions"][
            "ordinary_12way" if name == "stress12" else "within_matched_four"
        ]
        alternatives, assigned = (
            np.asarray(choices["alternatives"]),
            np.asarray(choices["menu_at_ordinal"]),
        )
        concepts = np.asarray(spec["selected_concepts"])
        x, signature = compact(SimpleNamespace(out=a.panels, double=a.double), SUB, name, concepts)
        require(signature == previous["panels"][name]["compact_sha256"], "Different held EEG")
        rr = (concepts[:, None] * 10 + np.arange(10)).ravel()
        zz = apply_projection(x[np.argsort(rr)], transform)
        dd = deep[np.searchsorted(classes, np.sort(concepts))].reshape(-1, 96)
        raw = np.sort(rr)
        rows, folds, cal = (
            np.asarray(reg[k]) for k in ("rows", "folds", "calibration_rows_by_fold_and_word")
        )
        require(
            all(
                np.array_equal(np.sort(cal[f].ravel()), np.sort(rows[folds != f])) for f in range(5)
            ),
            "Calibration contains withheld photos",
        )
        rng = np.random.default_rng(SEED + 2000 + alternatives.shape[1])
        records = []
        for f in range(5):
            for index, slots in enumerate(alternatives):
                ix = np.flatnonzero((folds == f) & (assigned == index))
                ix = ix[np.argsort(rows[ix])]
                require(len(ix) == 2 * len(slots), "Given menu or fold has unexpected row count")
                donor = np.roll(np.arange(len(ix)), rng.integers(1, len(ix)))
                records.append(
                    episode_record(
                        ix, rows[ix], rows[ix[donor]], cal[f, slots], slots, f, raw, zz, dd, w
                    )
                )
        data = combine(records)
        require(
            np.array_equal(cal[0, :, 0][np.asarray(reg["truth"])] // 10, data["truth_id"]),
            "Mismatched real concept IDs",
        )
        legacy_check(a, name, data, previous, np.asarray(reg["truth"]))
        words = np.asarray(reg["words"])
        data["words"] = words
        data["truth"] = np.asarray(reg["truth"])
        data["predicted_words"] = words[data["mixed_slot"]]
        out[name] = data
    return out, receipt


def object_report(data):
    concepts = np.unique(data["truth_id"])
    require(
        np.array_equal(
            np.unique(data["truth_id"], return_counts=True)[1], np.full(len(concepts), 10)
        ),
        "Each object needs ten independent query photos",
    )
    return np.stack(
        [[(data[key][data["truth_id"] == c] == c).sum() for c in concepts] for key in ARMS]
    )


def metrics(groups):
    """Resample paired object counts, drawing each supplied panel stratum independently."""
    rng = np.random.default_rng(BOOT_SEED)
    grouped = [object_report(v) for v in groups]
    n = sum(v.shape[1] * 10 for v in grouped)
    draws = [rng.integers(v.shape[1], size=(10000, v.shape[1])) for v in grouped]
    hard = ARMS.index("mixed")
    result = {}
    for i, name in enumerate(ARMS):
        if name == "mixed":
            continue
        d = [v[hard] - v[i] for v in grouped]
        b = sum(x[y].sum(1) for x, y in zip(d, draws, strict=True)) / n
        result[name] = dict(
            delta=float(sum(x.sum() for x in d) / n), ci95=np.quantile(b, [0.025, 0.975]).tolist()
        )
    return dict(
        n=n,
        hits=dict(zip(ARMS, sum(g.sum(1) for g in grouped).tolist(), strict=True)),
        mixed_minus=result,
    )


def primary_gate(result):
    return (
        result["mixed_minus"]["fisher"]["ci95"][0] > 0
        and result["mixed_minus"]["episodic"]["delta"] >= 0
    )


def save_decisions(path, data):
    if not path.exists():
        frozen_npz(path, data)
        return
    # Eigenvectors and their continuous scores can vary with the BLAS reduction order.
    # Every categorical decision, row and selected calibration set must replay exactly.
    with np.load(path, allow_pickle=False) as prev:
        require(set(prev.files) == set(data), "Existing decision packet schema differs")
        for k, value in data.items():
            if k.endswith("_scores"):
                require(np.allclose(prev[k], value, atol=1e-6, rtol=0), f"Different {k}")
            else:
                require(np.array_equal(prev[k], value), f"Different {k}")


def gate_replay(a):
    g = a.root / "generic"
    r = read(g / "score.json")
    require(
        r["stage"] == "generic"
        and r["design_sha256"] == sha(a.design)
        and r["packet_sha256"] == sha(g / "embeddings.npz")
        and r["encoder_receipt_sha256"] == sha(g / "receipt.json")
        and r["encoder_sha256"] == sha(a.training_source)
        and r["scorer_sha256"] == sha(Path(__file__)),
        "Generic gate belongs to another run",
    )
    path = g / "generic_12.npz"
    require(sha(path) == r["predictions_sha256"]["generic_12"], "Modified generic decisions")
    with np.load(path, allow_pickle=False) as z:
        decision = metrics([dict(z)])
    require(
        decision == r["cases"]["generic_12"] and primary_gate(decision) and r["advance"],
        "Generic gate has not passed",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["selftest", "generic", "target"])
    for key, default in dict(
        root=Path("outputs/trial-object-encoder"),
        design=PUBLIC / "trial_object_encoder_design.json",
        training_source=Path(__file__).with_name("fit_trial_object_encoder.py"),
        expanded=Path("outputs/photo-structure/expanded"),
        previous=Path("outputs/episodic-object"),
        panels=Path("outputs/session-matched-cohort"),
        double=Path("outputs/double-calibration"),
        sessions=Path("outputs/object-transfer/sessions.json"),
    ).items():
        p.add_argument("--" + key.replace("_", "-"), type=Path, default=default)
    a = p.parse_args()
    if a.stage != "selftest":
        design = read(a.design)
        require(
            design["subject"] == SUB
            and design["bootstrap_seed"] == BOOT_SEED
            and design["bootstrap_draws"] == 10000,
            "Score settings differ from the registered ones",
        )
    previous, ids, held = context(a)
    if a.stage == "target":
        gate_replay(a)
        cases, receipt = target(a, previous, ids)
    else:
        cases, receipt = generic(a, previous, ids, held, a.stage == "selftest")
    if a.stage == "selftest":
        print(
            "Synthetic embeddings exercised independently. Historical Fisher and episodic matched on every one of 1920 generic decisions."
        )
        return
    files = {}
    for name, v in cases.items():
        file = a.root / a.stage / (name + ".npz")
        save_decisions(file, v)
        files[name] = sha(file)
    report = dict(
        stage=a.stage,
        subject=SUB,
        scorer_sha256=sha(Path(__file__)),
        **receipt,
        predictions_sha256=files,
        cases={k: metrics([v]) for k, v in cases.items()},
        bootstrap_seed=BOOT_SEED,
        bootstrap_draws=10000,
    )
    if a.stage == "generic":
        report["advance"] = primary_gate(report["cases"]["generic_12"])
    else:
        report["panel_stratified"] = metrics(list(cases.values()))
        report["generic_score_sha256"] = sha(a.root / "generic" / "score.json")
    freeze(a.root / a.stage / "score.json", report)
    print(report)


if __name__ == "__main__":
    main()
