"""Recompute the prespecified seven-person confirmation from participant-level paired renders."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_registered_sessions import REV, audit_panel
from summarize_registered_transfer import crossed

PANELS = ("registered", "replicate")
ARMS = ("frozen", "personal")
METRICS = ("eeg_object", "generated_full1854", "generated_closed12", "within_class_photo")
DESIGN = (
    Path(__file__).resolve().parents[1]
    / "results/object_fidelity/personal_confirmation_design.json"
)


def read(path, hashes, root):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if hashes.setdefault(path.relative_to(root).as_posix(), digest) != digest:
        raise ValueError(f"Input changed during summary: {path}")
    return json.loads(raw)


def session_controls(root, people, hashes):
    schedules = read(root / "sessions.json", hashes, root)
    pairs = np.asarray(
        [schedules["subjects"][f"sub-{i:02}"]["session_pairs"] for i in range(1, 11)]
    )
    # Participant-major uint8 matrix independently replayed from the forty pinned .mat files.
    expected = "bb8394fc49425a9cacca632dcd2942ec0ba8ab03273be83fdcbf46fb0cb4d69b"
    if (
        schedules["revision"] != REV
        or pairs.shape != (10, 1654, 2)
        or pairs.dtype.kind not in "iu"
        or not np.all((pairs >= 1) & (pairs <= 4))
        or hashlib.sha256(pairs.astype(np.uint8).tobytes()).hexdigest() != expected
    ):
        raise ValueError("Wrong participant/concept session-pair schedule")
    panels, panel_rows = {}, set()
    for panel in PANELS:
        participants, oracle, by_class = {}, 0, {arm: [] for arm in ARMS}
        signature = None
        for subject in people:
            records = {}
            for arm, folder, method in (
                ("frozen", "transfer", "fused"),
                ("personal", "personal", "personal_r4"),
            ):
                rec = read(root / folder / panel / subject / "evaluation.json", hashes, root)
                registration = tuple(rec[k] for k in ("rows", "truth", "words", "folds"))
                if rec["subject"] != subject or (
                    signature is not None and signature != registration
                ):
                    raise ValueError("Frozen readouts do not share the panel registration")
                signature = registration
                records[arm] = rec["results"][method]
            rows, truth, words, folds = signature
            if len(set(words)) != 12 or sorted(zip(truth, folds, strict=True)) != [
                (c, f) for c in range(12) for f in range(5) for _ in range(2)
            ]:
                raise ValueError("Incorrect object/fold balance")
            audit = audit_panel(
                schedules["subjects"][subject],
                {"rows": rows, "truth": truth, "words": words, "results": records},
            )
            oracle += audit["pair_only_oracle_expected_hits"]
            parts = {}
            for arm, m in audit["methods"].items():
                observed, null = m["correct"], m["conditional_expected_hits"]
                parts[arm] = dict(
                    observed_hits=observed, expected_hits=null, excess_hits=observed - null
                )
                by_class[arm].append(m["excess_by_word"])
            participants[subject] = parts
        if panel_rows.intersection(rows):
            raise ValueError("Session audit panels overlap")
        panel_rows.update(rows)
        n = len(people) * len(rows)
        cohort = {}
        for arm in ARMS:
            totals = {
                k: sum(r[arm][k] for r in participants.values())
                for k in ("observed_hits", "expected_hits", "excess_hits")
            }
            rate = crossed(np.array(by_class[arm]))
            if not np.isclose(rate["mean"], totals["excess_hits"] / n):
                raise ValueError("Per-class excess and pooled hit excess differ")
            cohort[arm] = dict(
                **totals,
                observed_rate=totals["observed_hits"] / n,
                expected_rate=totals["expected_hits"] / n,
                excess_rate=rate,
            )
        difference = np.array(by_class["personal"]) - np.array(by_class["frozen"])
        panels[panel] = dict(
            n_queries=n,
            uniform_expected_hits=n / 12,
            pair_only_oracle_expected_hits=oracle,
            people=participants,
            cohort=cohort,
            paired_personal_minus_frozen_excess_rate=crossed(difference),
        )
    return dict(
        schedule_revision=REV,
        sessions_sha256=hashes["sessions.json"],
        role="Additional EEG-only confound diagnostic. Both readouts use the participant's eight calibration photos per object.",
        null="Freeze both readouts. Within each participant and panel, exchange whole class identities within their planned session pair, keeping all ten photos and five folds together. The expectation counts chance matches; an out-of-pair prediction contributes zero.",
        uncertainty="Crossed participant/object bootstrap of observed minus exact expected rates. The oracle uses only the known session pair. Session labels are used only for evaluation.",
        panels=panels,
    )


def case(root, panel, subject, hashes):
    inputs = {
        "frozen": read(root / "transfer" / panel / subject / "evaluation.json", hashes, root),
        "personal": read(root / "personal" / panel / subject / "evaluation.json", hashes, root),
    }
    reference = inputs["frozen"]
    receipt = read(root / "personal_conditions" / panel / subject / "evaluation.json", hashes, root)
    if (
        receipt["subject"] != subject
        or receipt["source_method"] != "personal_r4"
        or receipt["baseline_evaluation_sha256"]
        != hashes[f"transfer/{panel}/{subject}/evaluation.json"]
        or receipt["source_evaluation_sha256"]
        != hashes[f"personal/{panel}/{subject}/evaluation.json"]
        or receipt["baseline_conditions_sha256"] != reference["conditions_sha256"]
    ):
        raise ValueError("Render-condition receipt differs from the paired EEG readouts")
    truth = np.array(reference["truth"])
    words = np.array(reference["words"])
    folds = np.array(reference["folds"])
    if (
        len(words) != 12
        or len(set(reference["rows"])) != 120
        or not np.array_equal(np.bincount(truth), np.full(12, 10))
        or not np.all([sum((folds == f) & (truth == i)) == 2 for f in range(5) for i in range(12)])
    ):
        raise ValueError("Expected the registered balanced twelve-object, five-fold panel")
    raw, summaries, evaluator, orig = {}, {}, None, None
    for arm, method in (("frozen", "fused"), ("personal", "personal_r4")):
        inp = inputs[arm]
        image = read(root / "confirm_render" / arm / panel / subject / "results.json", hashes, root)
        if inp["subject"] != subject or any(
            inp[k] != reference[k] for k in ("words", "truth", "folds", "rows")
        ):
            raise ValueError("Recognition registrations differ")
        prediction = np.asarray(inp["results"][method]["predicted"])
        if prediction.shape != truth.shape or not np.isin(prediction, np.arange(12)).all():
            raise ValueError("Invalid recognition prediction")
        if (
            image["rows"] != reference["rows"]
            or image["labels"] != words[truth].tolist()
            or image["predicted_words"] != words[prediction].tolist()
            or image["condition_sha"]
            != (receipt if arm == "personal" else reference)["conditions_sha256"]
        ):
            raise ValueError("Generator inputs differ from the selected readout")
        src = image["results"]["aligned"]
        ref_img = image["results"]["original"]
        if evaluator is not None and (evaluator != image["clip_sha"] or orig != ref_img):
            raise ValueError("Reference or scorer differs between paired runs")
        evaluator, orig = image["clip_sha"], ref_img
        scores = {
            "eeg_object": (prediction == truth).astype(float),
            "generated_full1854": np.asarray(src["per_photo_full"]),
            "generated_closed12": np.asarray(src["per_photo_closed"]),
            "within_class_photo": np.asarray(src["per_photo_intra10"]),
        }
        if any(
            a.shape != truth.shape or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any()
            for a in scores.values()
        ):
            raise ValueError("Invalid per-photo credit")
        counts = {k: float(a.sum()) for k, a in scores.items()}
        expected = [
            inp["results"][method]["hits"],
            src["full_top1"],
            src["closed_top1"],
            src["original_photo"]["intra10_top1"],
        ]
        if src["n"] != 120 or not np.allclose(list(counts.values()), expected):
            raise ValueError("Row credits and reported totals differ")
        raw[arm] = {k: [a[truth == i].mean() for i in range(12)] for k, a in scores.items()}
        summaries[arm] = dict(
            **counts,
            full1854_credit_by_object=[
                float(scores["generated_full1854"][truth == i].sum()) for i in range(12)
            ],
            condition_sha256=image["condition_sha"],
        )
    return reference["rows"], reference["words"], evaluator, orig["per_photo_full"], summaries, raw


def run(root, sessions_only=False):
    design = json.loads(DESIGN.read_text())
    people = design["additional_participants"]
    if len(set(people)) != 7 or set(people) & set(design["excluded_from_confirmation_mean"]):
        raise ValueError("Invalid held-out participant set")
    reports, hashes, all_rows = {}, {}, []
    session_audit = session_controls(root, people, hashes)
    if sessions_only:
        return dict(people=people, eeg_session_control=session_audit, evaluation_file_sha256=hashes)
    for panel in PANELS:
        subject_info = {}
        by_class = {arm: {m: [] for m in METRICS} for arm in ARMS}
        original_rows, original_words, scorer, original_credits = None, None, None, None
        for s in people:
            rows, words, clip, reference_credits, sums, raw = case(root, panel, s, hashes)
            if original_rows is not None and (
                rows != original_rows
                or words != original_words
                or clip != scorer
                or original_credits != reference_credits
            ):
                raise ValueError("Participants saw different photos or used different evaluators")
            original_rows, original_words, scorer = rows, words, clip
            original_credits = reference_credits
            subject_info[s] = sums
            for a in ARMS:
                for m in METRICS:
                    by_class[a][m].append(raw[a][m])
        all_rows.extend(original_rows)
        performance = {}
        for m in METRICS:
            f, p = (np.array(by_class[a][m]) for a in ARMS)
            performance[m] = {
                a: dict(credit=sum(v[a][m] for v in subject_info.values()), **crossed(x))
                for a, x in zip(ARMS, (f, p), strict=True)
            }
            performance[m]["paired_personal_minus_frozen"] = crossed(p - f)
        reports[panel] = dict(
            words=original_words,
            clip_sha256=scorer,
            original_full1854_credit_per_120=sum(original_credits),
            people=subject_info,
            n_queries=len(original_rows) * len(people),
            metrics=performance,
        )
    if len(set(all_rows)) != 240:
        raise ValueError("Expected two non-overlapping panels")
    excluded_classes = np.unique(np.asarray(all_rows) // 10)
    eligible = np.setdiff1d(np.arange(1654), excluded_classes)
    calibration = np.sort(np.random.default_rng(260431).choice(eligible, 128, replace=False))
    packets = {}
    for subject in people:
        for panel in PANELS:
            rec = read(root / "personal" / panel / subject / "evaluation.json", hashes, root)
            signatures = {key: rec[key] for key in ("generic_sha256", "target_sha256")}
            if (
                len(excluded_classes) != 24
                or not np.array_equal(rec["generic_classes"], calibration)
                or packets.setdefault(subject, signatures) != signatures
            ):
                raise ValueError(
                    "Personal calibration differs between panels or from the fixed draw"
                )
    return dict(
        preregistered_protocol_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),
        frozen_model_commit=design["frozen_model_commit"],
        people=people,
        unique_stimuli=240,
        participant_photo_queries=1680,
        calibration_packet_sha256=packets,
        primary="Top-1 full 1854-name credit on one generated image per participant and photo",
        uncertainty="10000 paired crossed participant/class percentile bootstrap samples per panel. Each object contributes ten photos. Intervals are fractions.",
        descriptive_combined={
            a: {m: sum(reports[p]["metrics"][m][a]["credit"] for p in PANELS) for m in METRICS}
            for a in ARMS
        },
        panels=reports,
        eeg_session_control=session_audit,
        evaluation_file_sha256=hashes,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--sessions-only",
        action="store_true",
        help="Audit both EEG readouts using the seven participants' evaluation files before renders arrive",
    )
    a = p.parse_args()
    result = run(a.root, a.sessions_only)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n"
    )
