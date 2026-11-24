"""Summarize registered transfers, or compare all four completed session protocols.

python summarize_registered_transfer.py --session-controls outputs/object-transfer \
    --references outputs/object-fidelity --out research/eeg_stack_20261112/results/object_fidelity/session_controls.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = ("raw_eeg", "projected_eeg", "visual_h14", "fused")


def crossed(values, seed=260429):
    """Crossed participant and class resampling keeps the ten photographs in a class together."""
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Cross-participant intervals require at least two held-out people")
    rng = np.random.default_rng(seed)
    participants = rng.integers(len(values), size=(10000, len(values)))
    classes = rng.integers(values.shape[1], size=(10000, values.shape[1]))
    b = values[participants[:, :, None], classes[:, None, :]].mean((1, 2))
    return {"mean": float(values.mean()), "ci95": np.quantile(b, [0.025, 0.975]).tolist()}


def summarize(root, reference, source):
    truth = np.asarray(reference["truth"])
    rows = np.asarray(reference["rows"])
    classes = np.unique(truth)
    if len(rows) != 120 or len(classes) != 12 or not np.all(np.bincount(truth) == 10):
        raise ValueError("Expected twelve balanced object concepts")
    results = {}
    matrices = {m: [] for m in METHODS}
    for path in sorted(root.glob("sub-*/evaluation.json")):
        subject = path.parent.name
        d = json.loads(path.read_text())
        if (
            not np.array_equal(d["rows"], rows)
            or not np.array_equal(d["truth"], truth)
            or not np.array_equal(d["folds"], reference["folds"])
        ):
            raise ValueError(f"Photo or fold mismatch for {subject}")
        counts = {}
        for m in METHODS:
            record = d["results"][m]
            pred = np.asarray(record["predicted"])
            if pred.shape != truth.shape or not np.isin(pred, classes).all():
                raise ValueError("Predicted an object outside the calibration panel")
            correct = pred == truth
            counts[m] = int(correct.sum())
            if counts[m] != record["hits"]:
                raise ValueError(f"Saved hit count differs for {subject}/{m}")
            if subject != source:
                matrices[m].append([float(correct[truth == k].mean()) for k in classes])
        results[subject] = dict(
            hits=counts,
            rows=len(truth),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    if source not in results:
        raise ValueError("Missing source-participant replay")
    matrices = {m: np.asarray(v) for m, v in matrices.items()}
    if results[source]["hits"]["fused"] != reference["exact_hits"]:
        raise ValueError("Source-participant replay differs from the original experiment")
    return dict(
        words=reference["words"],
        people=results,
        new_people=len(matrices["fused"]),
        denominator_per_person=len(truth),
        new_participant_performance={k: crossed(v) for k, v in matrices.items()},
        new_participant_fused_minus={
            k: crossed(matrices["fused"] - v) for k, v in matrices.items() if k != "fused"
        },
    )


PROTOCOLS = {
    "transfer": METHODS,
    "source_proto": ("projected_eeg", "fused"),
    "cross_session": tuple(
        f"{half}_{method}"
        for half in (
            "early_to_late",
            "late_to_early",
            "early_to_early",
            "late_to_late",
            "crossed",
            "matched",
        )
        for method in ("projected", "fused")
    ),
    "erp": ("early_to_late", "late_to_early", "cross", "same", "mean4"),
}


def original_render(record, fit):
    words = np.asarray(fit["words"])
    labels = np.asarray(record["labels"])
    if (
        record["rows"] != fit["rows"]
        or not np.array_equal(labels, words[fit["truth"]])
        or record["condition_sha"] != fit["conditions_sha256"]
        or not np.array_equal(
            record["predicted_words"], words[fit["results"]["fused"]["predicted"]]
        )
    ):
        raise ValueError("sub-04 render differs from original R4 registration or conditioned words")
    metrics = {}
    arms = record["results"]
    for k, v in arms.items():
        if v["n"] != 120 or any(
            not np.isclose(np.sum(v["per_photo_" + field]), v[field + "_top1"])
            for field in ("full", "closed")
        ):
            raise ValueError("Render hit counts disagree with per-photo credit")
        metrics[k] = dict(
            full1854_top1_credit=v["full_top1"],
            closed12_top1_credit=v["closed_top1"],
            original_photo_gallery_credit=v["original_photo"],
        )
    pairs = {}
    rng = np.random.default_rng(260429)
    concepts = np.unique(labels)
    sample = rng.integers(len(concepts), size=(10000, len(concepts)))
    for field in ("full", "closed"):
        aligned = np.asarray(arms["aligned"]["per_photo_" + field])
        shuffled = np.asarray(arms["shuffled"]["per_photo_" + field])
        delta = aligned - shuffled
        d = np.array([delta[labels == c].mean() for c in concepts])
        pairs[field + "1854" if field == "full" else "closed12"] = dict(
            delta=float(delta.mean()), ci95=np.quantile(d[sample].mean(1), [0.025, 0.975]).tolist()
        )
    return dict(
        participant="sub-04",
        n=120,
        means=metrics,
        aligned_minus_shuffled=pairs,
        uncertainty="Fixed participant, 10000 paired resamples of the 12 classes; rates and interval endpoints are fractions.",
        condition_sha256=record["condition_sha"],
        clip_l14_sha256=record["clip_sha"],
    )


def control_panel(root, refs, name):
    hashes = {}

    def load(path, label):
        blob = path.read_bytes()
        hashes[label] = hashlib.sha256(blob).hexdigest()
        return json.loads(blob)

    source = "sub-03"
    reference = load(refs / (name + "_eeg.json"), "reference_eeg")
    ablation = load(refs / (name + "_ablation.json"), "reference_ablation")
    truth = np.array(reference["truth"])
    subjects = [f"sub-{i:02}" for i in range(1, 11)]
    if len(set(reference["rows"])) != 120 or not np.array_equal(
        np.bincount(truth), np.full(12, 10)
    ):
        raise ValueError("Expected a balanced twelve-class panel")
    columns = [f"{g}/{m}" for g, methods in PROTOCOLS.items() for m in methods]
    mat = {k: [] for k in columns}
    counts = {}
    four = None
    for sub in subjects:
        d = {g: load(root / g / name / sub / "evaluation.json", f"{g}/{sub}") for g in PROTOCOLS}
        native, source_proto, cross, erp = (d[g] for g in PROTOCOLS)
        if any(
            v.get("subject") != sub
            or any(v[k] != reference[k] for k in ("words", "rows", "folds", "truth"))
            for v in d.values()
        ):
            raise ValueError(f"Subject or registration mismatch for {sub}/{name}")
        if (
            native["model_signature"] != source_proto["transfer_model_signature"]
            or native["model_signature"] != cross["model_signature"]
            or native["packet_sha256"] != source_proto["target_packet_sha256"]
            or native["packet_sha256"] != cross["full_packet_sha256"]
            or cross["half_packet_sha256"] != erp["input_sha256"]
            or hashes["reference_ablation"] != erp["registration_sha256"]
        ):
            raise ValueError(
                "Comparison source model, waveforms or calibration-fold receipt differs"
            )
        for m in ("projected_eeg", "fused", "visual_h14"):
            counterpart = "r4_projected" if m == "projected_eeg" else "r4_" + m
            if native["results"][m]["predicted"] != cross["results"][counterpart]["predicted"]:
                raise ValueError("Session scorer did not reproduce original R4 decisions")
        if sub == source:
            for m in METHODS:
                if native["results"][m]["predicted"] != ablation["results"][m]["predicted"]:
                    raise ValueError("Source replay differs from original ablation")
            if any(
                source_proto["results"][m]["predicted"] != native["results"][m]["predicted"]
                for m in PROTOCOLS["source_proto"]
            ):
                raise ValueError("Source-only control failed source replay")
        if sub == "sub-04":
            four = native
        subject_counts = []
        for spec in columns:
            group, method = spec.split("/")
            result = d[group]["results"][method]
            pred = np.asarray(result["predicted"])
            if (
                pred.shape != truth.shape
                or not np.issubdtype(pred.dtype, np.integer)
                or np.any((pred < 0) | (pred >= 12))
                or result.get("n", 120) != 120
                or int(np.sum(pred == truth)) != result["hits"]
            ):
                raise ValueError(f"Invalid predictions or recorded count: {name}/{sub}/{spec}")
            hit = pred == truth
            subject_counts.append(int(hit.sum()))
            if sub != source:
                mat[spec].append([hit[truth == k].mean() for k in range(12)])
        counts[sub] = subject_counts
    mat = {k: np.array(v) for k, v in mat.items()}
    performance = {
        k: dict(hits=sum(counts[s][j] for s in subjects if s != source), **crossed(mat[k]))
        for j, k in enumerate(columns)
    }
    contrasts = [
        ("transfer/fused", "source_proto/fused"),
        ("transfer/projected_eeg", "source_proto/projected_eeg"),
        ("cross_session/crossed_fused", "cross_session/matched_fused"),
        ("cross_session/crossed_fused", "transfer/fused"),
        ("cross_session/crossed_projected", "cross_session/matched_projected"),
        ("cross_session/crossed_projected", "transfer/projected_eeg"),
        ("erp/cross", "erp/same"),
        ("erp/cross", "erp/mean4"),
        ("erp/cross", "cross_session/crossed_projected"),
        ("erp/cross", "cross_session/crossed_fused"),
    ]
    paired = {f"{a} minus {b}": crossed(mat[a] - mat[b]) for a, b in contrasts}
    rendered = original_render(load(root / f"sub04_{name}_render.json", "sub04_render"), four)
    return dict(
        words=reference["words"],
        method_order=columns,
        per_person_hits=counts,
        heldout_performance=performance,
        heldout_paired_differences=paired,
        sub04_original_r4_render=rendered,
        input_sha256=hashes,
    )


def session_controls(root, references):
    gate_path = root / "transfer/source_verification.json"
    gate = json.loads(gate_path.read_bytes())
    if gate["subject"] != "sub-03" or gate["verified_panels"] != {
        "registered": 85,
        "replicate": 92,
    }:
        raise ValueError("Missing original sub-03 source verification")
    return dict(
        source_replay_excluded_from_averages="sub-03",
        heldout_people=9,
        queries_per_person_per_panel=120,
        unique_photos_per_panel=120,
        heldout_prediction_denominator_per_panel=1080,
        classes_per_panel=12,
        chance=1 / 12,
        uncertainty="10000 crossed participant-and-class percentile bootstrap samples, seed 260429. Ten photos within each class remain together. All aggregate means, deltas and CI endpoints are fractions.",
        protocols=dict(
            transfer="Frozen sub-03 generic R4 heads. Eight target-person calibration photos per class.",
            source_proto="The same folds and visual references; eight other sub-03 photos of each class provide EEG prototypes. New participants supply query EEG only.",
            cross_session="Frozen projection using two early and two late trials. Direction names denote query then support. Crossed and matched pool standardized direction scores. Fused always uses the original four-repeat visual H14 readout and the 0.75/0.25 mix.",
            erp="Separate 60–360 ms ERP shrinkage LDA. Cross and same pool early and late directions; mean4 uses four-repeat waveforms.",
        ),
        source_waveforms=gate["waveform"],
        source_gate_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        panels={g: control_panel(root, references, g) for g in ("registered", "replicate")},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--runs", type=Path)
    mode.add_argument(
        "--session-controls",
        type=Path,
        help="Root containing transfer, source_proto, cross_session, erp and the sub-04 renders",
    )
    p.add_argument(
        "--references",
        type=Path,
        required=True,
        help="Directory of registered_eeg.json and replicate_eeg.json",
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    report = (
        session_controls(a.session_controls, a.references)
        if a.session_controls
        else dict(
            source="sub-03",
            calibration_photos_per_class=8,
            repeat_count=4,
            uncertainty="10000 crossed bootstrap resamples of held-out people and object classes; photos retained together",
            panels={
                k: summarize(
                    a.runs / k, json.loads((a.references / (k + "_eeg.json")).read_text()), "sub-03"
                )
                for k in ("registered", "replicate")
            },
        )
    )
    a.out.parent.mkdir(exist_ok=True, parents=True)
    a.out.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
