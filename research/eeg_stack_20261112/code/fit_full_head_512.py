"""Compare full-scalp and posterior-only 512-object Fisher models using frozen weights."""

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from audit_registered_sessions import audit_panel
from decode_eeg_layout import CHANNELS
from fit_full_head_objects import directions, features, scores
from fit_personal_session_metric import load
from prepare_expanded_personal import extract
from prepare_full_head_512 import PANELS, PUBLIC, SUBJECTS, WEIGHTS, check, defaults, temporal
from prepare_registered_subjects import OFFSET, REV, ROW_BYTES, SHAPE, SIZE, axes
from register_fresh_photo_panel import freeze, read, sha
from summarize_personal_metric import paired


def training(a, sub):
    root = a.out / "features" / sub
    meta = read(a.expanded / "features" / (sub + ".json"))
    acquired = read(root.with_suffix(".json"))
    check(
        sha(a.expanded / "features" / (sub + ".json")) == acquired["training_receipt_sha256"]
        and acquired["design_sha256"] == sha(PUBLIC),
        "Training receipts differ",
    )
    classes = np.asarray(meta["training_concepts"])
    rows = (classes[:, None] * 10 + np.arange(10)).ravel()
    post = []
    full = []
    for c in classes:
        with np.load(a.expanded / "features" / sub / f"{c:04d}.npz", allow_pickle=False) as z:
            check(
                int(z["start"]) == 10 * c and str(z["raw_sha"]) == meta["raw_block_sha256"][str(c)],
                "Unregistered posterior training block",
            )
            post.append(z["features"])
        with np.load(root / f"{c:04d}.npz", allow_pickle=False) as z:
            f = z["half_features"]
            check(
                f.shape == (2, 10, 882)
                and str(z["raw_sha"]) == meta["raw_block_sha256"][str(c)]
                and hashlib.sha256(f.tobytes()).hexdigest() == acquired["feature_sha256"][str(c)],
                "Unregistered full-head training block",
            )
            full.append(f)
    p = np.concatenate(post)
    f = np.concatenate(full, axis=1)
    check(
        p.shape == (5120, 1003)
        and hashlib.sha256(p.tobytes()).hexdigest() == meta["train_design_sha256"],
        "Historical compact posterior model changed",
    )
    sessions = read(a.inputs / "sessions.json")["subjects"][sub]
    return SimpleNamespace(
        rows=rows,
        p=p.astype(float),
        f=f.astype(float),
        s=np.asarray(sessions["session_pairs"])[rows // 10].T,
    ), sessions


def target(a, sub, root, legacy):
    path = (a.inputs / "halves" if legacy else root / "eeg") / (sub + ".npz")
    receipt = (a.inputs if legacy else root / "eeg") / (sub + ".json")
    rows, half, _, mean = load(path)
    reg = read(receipt)
    original = read(a.expanded / "features" / (sub + ".json"))
    check(
        reg["source_full_lfs_sha256"] == original["source_full_lfs_sha256"]
        and reg["source"] == original["source"]
        and reg.get("source_revision", reg.get("revision")) == REV,
        "Target belongs to a different published EEG recording",
    )
    cache = (a.inputs if legacy else root) / "ranges" / REV / sub
    blocks = reg["row_ranges"] if legacy else reg["blocks"]
    abc = a.inputs / "ranges" / REV / sub
    end = OFFSET + SHAPE[0] * ROW_BYTES
    channels, _ = axes(
        (abc / f"0-{OFFSET - 1}.bin").read_bytes(), (abc / f"{end}-{SIZE - 1}.bin").read_bytes()
    )
    post = [channels.tolist().index(ch) for ch in CHANNELS]
    hs = []
    check(
        reg["subject"] == sub and len(rows) == 10 * len(blocks),
        "Target subject or number of physical ranges differs",
    )
    if not legacy:
        check(sha(path) == reg["packet_sha256"], "Target EEG packet digest mismatch")
    for i, b in enumerate(blocks):
        group = rows[10 * i : 10 * i + 10]
        start = int(group[0])
        low = OFFSET + start * ROW_BYTES
        high = low + 10 * ROW_BYTES - 1
        check(
            np.array_equal(group, start + np.arange(10))
            and (b["bytes"] if legacy else [b["byte_start"], b["byte_end"]]) == [low, high],
            "Target row range mismatch",
        )
        file = cache / f"{low}-{high}.bin"
        if not file.is_file():
            file = abc / file.name
        buf = file.read_bytes()
        check(hashlib.sha256(buf).hexdigest() == b["sha256"], "Target physical range failed SHA")
        x = np.frombuffer(buf, "<f8").reshape(10, *SHAPE[1:]).astype(np.float32)
        for k, (first, second) in enumerate(((1, 3), (0, 2))):
            check(
                np.array_equal(
                    ((x[:, first] + x[:, second]) * 0.5)[:, post], half[k, 10 * i : 10 * i + 10]
                ),
                "Full-head acquisition did not reproduce recorded posterior halves",
            )
        hs.append(temporal(x))
    return SimpleNamespace(
        rows=rows,
        p=extract(mean).astype(float),
        f=np.concatenate(hs, axis=1).astype(float),
        source_sha256=sha(path),
        receipt_sha256=sha(receipt),
    )


def evaluate_panel(a, sub, tag, registration, baseline, train, target, streams, session):
    reg = read(registration)
    rows, folds, vocab, groups = [
        np.asarray(reg[k]) for k in ("rows", "folds", "words", "calibration_rows_by_fold_and_word")
    ]
    check(
        len(rows) == len(set(rows)) == 120
        and len(np.unique(rows // 10)) == 12
        and len(vocab) == 12
        and groups.shape == (5, 12, 8)
        and not np.intersect1d(rows // 10, train.rows // 10).size,
        "Invalid target or overlap with generic training",
    )
    pos = {int(r): i for i, r in enumerate(target.rows)}
    out = np.zeros((3, 120, 12), float)
    for fold, cal in enumerate(groups):
        ix = np.flatnonzero(folds == fold)
        check(
            len(ix) == 24
            and set(cal.ravel()) == set(rows[folds != fold])
            and len(np.unique(cal)) == 96,
            "Target support intersects query",
        )
        out[:, ix] = scores(
            streams, [pos[r] for r in rows[ix]], np.array([[pos[r] for r in g] for g in cal])
        )
    p, f, c = out
    pred = {
        name: s.argmax(1)
        for name, s in dict(
            expanded_r4=p,
            full_mean4=f,
            full_cross=c,
            full_head_512=p + WEIGHTS["full"] * f + WEIGHTS["cross"] * c,
        ).items()
    }
    # Use true query labels only after the scores and complete predictions are fixed.
    truth = np.array(reg["truth"])
    results = {
        m: dict(predicted=v.tolist(), hits=int((v == truth).sum()), n=120) for m, v in pred.items()
    }
    if baseline:
        old = read(baseline)
        check(
            all(reg[k] == old[k] for k in ("rows", "folds", "words", "truth"))
            and old["results"]["expanded_r4"] == results["expanded_r4"],
            "512-posterior arm must replay every baseline word",
        )
    data = dict(
        subject=sub,
        panel=tag,
        rows=reg["rows"],
        folds=reg["folds"],
        words=reg["words"],
        truth=reg["truth"],
        results=results,
        design_sha256=sha(PUBLIC),
        registration_sha256=sha(registration),
        target_eeg_sha256=target.source_sha256,
        raw_range_receipt_sha256=target.receipt_sha256,
        generic_full_receipt_sha256=sha((a.out / "features" / sub).with_suffix(".json")),
        posterior_baseline_sha256=sha(baseline) if baseline else None,
    )
    stats = audit_panel(session, data)["methods"]
    data["session_control"] = {
        k: {
            key: ss[key]
            for key in (
                "conditional_expected_hits",
                "accuracy_minus_conditional_expectation",
                "permutation_one_sided_p",
            )
        }
        for k, ss in stats.items()
    }
    data["paired_versus_512_posterior"] = paired(
        pred["full_head_512"] == truth, pred["expanded_r4"] == truth, truth
    )
    path = a.out / "predictions" / tag / sub / "evaluation.json"
    freeze(path, data)
    print(
        sub, tag, results["expanded_r4"]["hits"], "->", results["full_head_512"]["hits"], flush=True
    )
    return path


def evaluate(a, sub):
    train, session = training(a, sub)
    if a.confirmation_root:
        tag = a.confirmation_panel
        check(
            tag and tag not in PANELS and tag.isidentifier(),
            "Independent confirmation requires a new panel identifier",
        )
        location = a.confirmation_root
        reg = location / "references" / (tag + "_ablation.json")
        old = set()
        for p in PANELS:
            source = (
                a.fresh / "references/fresh_ablation.json"
                if p == "fresh"
                else a.references / (p + "_ablation.json")
            )
            old.update(np.array(read(source)["rows"]) // 10)
        check(
            not old.intersection(np.array(read(reg)["rows"]) // 10),
            "Confirmation concepts overlap an earlier target",
        )
        targetset = target(a, sub, location, False)
        streams = directions(
            train, np.arange(len(train.rows)), targetset, np.arange(len(targetset.rows))
        )
        evaluate_panel(a, sub, tag, reg, None, train, targetset, streams, session)
        return
    paths = []
    for legacy, panels in ((True, ("registered", "replicate")), (False, ("fresh",))):
        targetset = target(a, sub, a.fresh, legacy)
        if sub == "sub-04" and legacy:
            # The previous model used these same full-scalp, posterior and target feature values.
            for compressed, historical in (
                (train, features(a.inputs, sub, True)),
                (targetset, features(a.inputs, sub, False)),
            ):
                indices = np.searchsorted(compressed.rows, historical.rows)
                check(
                    np.array_equal(compressed.rows[indices], historical.rows)
                    and np.array_equal(compressed.p[indices], historical.p)
                    and np.array_equal(compressed.f[:, indices], historical.f),
                    "Historical 128-object or old target feature matrix fails bitwise replay",
                )
        streams = directions(
            train, np.arange(len(train.rows)), targetset, np.arange(len(targetset.rows))
        )
        for p in panels:
            reg = (
                a.references / (p + "_ablation.json")
                if legacy
                else a.fresh / "references/fresh_ablation.json"
            )
            base = (
                a.expanded / p / sub if legacy else a.fresh / "evaluations" / sub / p
            ) / "evaluation.json"
            paths.append(evaluate_panel(a, sub, p, reg, base, train, targetset, streams, session))
    if sub == "sub-04":
        docs = [read(p) for p in paths]
        counts = {
            k: sum(d["results"][k]["hits"] for d in docs) for k in ("expanded_r4", "full_head_512")
        }
        delta = counts["full_head_512"] - counts["expanded_r4"]
        freeze(
            a.out / "sub-04_gate.json",
            dict(
                design_sha256=sha(PUBLIC),
                n=360,
                counts=counts,
                delta_hits=delta,
                passed=delta > 0,
                predicted_sha256={p.parent.parent.name: sha(p) for p in paths},
            ),
        )


def summarize(a):
    gate = read(a.out / "sub-04_gate.json")
    requested = SUBJECTS if gate["passed"] else ("sub-04",)
    summaries = {}
    for sub in requested:
        panels = {}
        all_rows = []
        all_truth = []
        hit_a = []
        hit_b = []
        for panel in PANELS:
            path = a.out / "predictions" / panel / sub / "evaluation.json"
            doc = read(path)
            check(
                doc["subject"] == sub and doc["design_sha256"] == sha(PUBLIC),
                "Changed individual fit",
            )
            rows = np.array(doc["rows"])
            y = np.array(doc["truth"])
            p, q = (
                np.array(doc["results"][key]["predicted"])
                for key in ("full_head_512", "expanded_r4")
            )
            scores = paired(p == y, q == y, y)
            session = doc["session_control"]
            expected = {
                k: session[k]["conditional_expected_hits"] for k in ("full_head_512", "expanded_r4")
            }
            panels[panel] = dict(
                n=120,
                posterior_hits=int((q == y).sum()),
                combined_hits=int((p == y).sum()),
                full_mean4_only_hits=doc["results"]["full_mean4"]["hits"],
                cross_session_only_hits=doc["results"]["full_cross"]["hits"],
                changed_predictions=int((p != q).sum()),
                gains=scores["improved_rows"],
                losses=scores["worsened_rows"],
                paired_delta=scores["delta_accuracy"],
                delta_ci95=scores["class_bootstrap_ci95"],
                conditional_expected_hits=expected,
                paired_delta_session_excess_hits=scores["delta_credit"]
                - expected["full_head_512"]
                + expected["expanded_r4"],
                sha256=sha(path),
            )
            all_rows.extend(rows)
            all_truth.extend(rows // 10)
            hit_a.extend(p == y)
            hit_b.extend(q == y)
        check(
            len(np.unique(all_rows)) == 360 and len(np.unique(all_truth)) == 36,
            "Panels reuse an object",
        )
        total = paired(hit_a, hit_b, np.asarray(all_truth))
        summaries[sub] = dict(
            panels=panels,
            combined=dict(
                n=360,
                posterior_hits=int(sum(hit_b)),
                combined_hits=int(sum(hit_a)),
                delta_hits=int(total["delta_credit"]),
                gains=total["improved_rows"],
                losses=total["worsened_rows"],
                paired_delta=total["delta_accuracy"],
                delta_ci95=total["class_bootstrap_ci95"],
                paired_delta_session_excess_hits=sum(
                    v["paired_delta_session_excess_hits"] for v in panels.values()
                ),
            ),
        )
    report = dict(
        protocol_sha256=sha(PUBLIC),
        weights=WEIGHTS,
        training_objects=512,
        training_photos_per_person=5120,
        unique_target_photos=360,
        unique_target_objects=36,
        classification_choices=12,
        development_subject="sub-04",
        passed_development_gate=gate["passed"],
        scope="Closed 12-way EEG word readout. All 360 physical query images are shared across participants. These objects were previously used to evaluate other models. Image naming is a separate measurement.",
        baseline="The 512-class posterior-only comparator reproduces every saved old prediction. The combination adds two frozen full-scalp vote scores with weights originally selected in sub-04's 128 generic objects.",
        uncertainty="Per participant, paired differences are fractions. Percentile 95% intervals resample entire objects 10,000 times within each 12-object panel or across all 36 panel-specific objects. Recorded-session baselines use within-session-pair permutations of object labels.",
        participants=summaries,
    )
    freeze(a.out / "summary.json", report)
    freeze(a.release_file, report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    defaults(p)
    p.add_argument("--subject", choices=SUBJECTS, default="sub-04")
    p.add_argument("--confirmation-root", type=Path)
    p.add_argument("--confirmation-panel")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--release-file", type=Path, default=PUBLIC.with_name("full_head_512_eeg.json"))
    a = p.parse_args()
    check(sha(PUBLIC) == sha(a.out / "design.json"), "Frozen lock differs")
    if a.subject != "sub-04":
        check(read(a.out / "sub-04_gate.json")["passed"], "Development gate not met")
    if a.summarize:
        summarize(a)
    else:
        evaluate(a, a.subject)
