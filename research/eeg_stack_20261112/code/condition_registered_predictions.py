"""Package externally predicted registered object IDs with frozen R4 generation features."""

import argparse
import re
from pathlib import Path

import numpy as np
from decode_eeg_layout import eeg_design, registration
from fit_registered_transfer import apply_ridge, digest, packet, read, registered_panel, save, sha
from score_cross_session_objects import recondition


def validate_result(record, panel, subject, method):
    if record.get("subject") != subject:
        raise ValueError(f"Participant mismatch for {panel['name']}/{subject}")
    for key in ("rows", "folds", "truth", "words"):
        if record.get(key) != panel["ev"][key]:
            raise ValueError(f"{panel['name']}/{subject}: {key} differs from original registration")
    if method not in record.get("results", {}):
        raise ValueError(f"{panel['name']}/{subject}: missing method {method}")
    data = record["results"][method]
    ids = data["predicted"]
    if (
        not isinstance(ids, list)
        or len(ids) != len(panel["rows"])
        or any(type(i) is not int or not 0 <= i < len(panel["words"]) for i in ids)
    ):
        raise ValueError(f"{method} must have one integer concept index per registered query")
    pred = np.asarray(ids, dtype=np.int64)
    n, hits = len(pred), int((pred == panel["truth"]).sum())
    if (
        type(data.get("n")) is not int
        or type(data.get("hits")) is not int
        or (data["n"], data["hits"]) != (n, hits)
    ):
        raise ValueError(f"{method} count mismatch for {panel['name']}/{subject}")
    return pred, dict(predicted=ids, n=n, hits=hits)


def run(args):
    if args.out.resolve() in {args.transfer.resolve(), args.prediction_runs.resolve()}:
        raise ValueError("Write regenerated conditions to a separate output directory")
    gate = read(args.transfer / "source_verification.json")
    records, labels, _, posterior = registration(args.archive)
    layout_path = args.archive / "data/recovery_20261108_v236/sub-03_layout.json"
    if sha(layout_path) != gate["archive_sha256"]["data/recovery_20261108_v236/sub-03_layout.json"]:
        raise ValueError("Archive channel order differs from the gated R4 fit")
    channels = read(layout_path)["channels"]
    class_words = [r["concept_id"].split("_", 1)[1] for r in records[::10]]
    panels = [registered_panel(path, labels, class_words) for path in args.panel_dirs]
    required = np.unique(np.concatenate([p["rows"] for p in panels]))
    if (
        len(required) != 240
        or {p["name"]: p["ev"]["exact_hits"] for p in panels} != gate["verified_panels"]
    ):
        raise ValueError("Expected both original 12-way panels from the gate")
    supplied = {
        f.parent.name
        for f in args.prediction_runs.glob(f"{panels[0]['name']}/sub-*/evaluation.json")
    }
    subjects = list(dict.fromkeys(args.subjects or sorted(supplied)))
    if not subjects or any(not re.fullmatch(r"sub-(0[1-9]|10)", s) for s in subjects):
        raise ValueError("Expected at least one participant with external predictions")
    for p in panels:
        cache = args.transfer / "cache" / p["name"]
        model_path = cache / "generic_model.npz"
        model_receipt = read(cache / "generic_model.json")
        sig = digest(
            dict(archive=gate["archive_sha256"], code=gate["code_sha256"], panel=p["receipt"])
        )
        if sig != model_receipt["signature"] or sha(model_path) != model_receipt["sha256"]:
            raise ValueError("Generic R4 coefficients differ from the gated panel registration")
        with np.load(model_path, allow_pickle=False) as z:
            model = {
                k: z[k]
                for k in (
                    "visual_mu",
                    "visual_sd",
                    "visual_w",
                    "visual_bias",
                    "cal_f",
                    "global_f",
                    "axis",
                    "alpha",
                    "radius",
                )
            }
        local = {int(row): i for i, row in enumerate(p["rows"])}
        support = np.array(
            [[[local[int(row)] for row in group] for group in fold] for fold in p["support"]]
        )
        for subject in subjects:
            source_path = args.prediction_runs / p["name"] / subject / "evaluation.json"
            incoming = read(source_path)
            pred, score = validate_result(incoming, p, subject, args.prediction_method)
            base = args.transfer / p["name"] / subject
            original_path = base / "evaluation.json"
            original = read(original_path)
            old_pred, old_score = validate_result(original, p, subject, "fused")
            eeg_path = args.subjects_dir / (subject + ".npz")
            eeg_sha = sha(eeg_path)
            if (
                original["packet_sha256"] != eeg_sha
                or original["model_signature"] != sig
                or (subject == "sub-03" and gate["source_packet_sha256"] != eeg_sha)
            ):
                raise ValueError(
                    "Original R4 evaluation does not correspond to this waveform and model"
                )
            if incoming.get("target_r4_packet_sha256", eeg_sha) != eeg_sha:
                raise ValueError("External run's declared target R4 packet checksum differs")
            cond_path = base / "conditions.npz"
            if sha(cond_path) != original["conditions_sha256"]:
                raise ValueError("Original R4 generator packet failed its checksum")
            raw, _ = packet(eeg_path, required, channels)
            design = eeg_design(raw, np.searchsorted(required, p["rows"]), posterior)
            fp = apply_ridge(model, design, "visual")[:, 1024:]
            with np.load(cond_path, allow_pickle=False) as frozen:
                vals = recondition(frozen, p, model, pred, old_pred, fp, support)
                if not np.array_equal(vals["start"], frozen["start"]):
                    raise ValueError("R4 layout was altered")
            out = args.out / p["name"] / subject
            save(out / "conditions.npz", vals)
            save(
                out / "evaluation.json",
                dict(
                    subject=subject,
                    primary=args.prediction_method,
                    rows=p["ev"]["rows"],
                    folds=p["ev"]["folds"],
                    words=p["ev"]["words"],
                    truth=p["ev"]["truth"],
                    results={args.prediction_method: score},
                    baseline={"method": "fused", "protocol": "registered R4", **old_score},
                    changed_labels=int((pred != old_pred).sum()),
                    model_signature=sig,
                    r4_packet_sha256=eeg_sha,
                    source_evaluation_sha256=sha(source_path),
                    source_method=args.prediction_method,
                    baseline_evaluation_sha256=sha(original_path),
                    baseline_conditions_sha256=original["conditions_sha256"],
                    panel_registration_sha256=p["receipt"],
                    conditions_sha256=sha(out / "conditions.npz"),
                ),
            )
            print(
                subject,
                p["name"],
                args.prediction_method,
                f"{score['hits']}/{score['n']}",
                "changed predictions",
                int((pred != old_pred).sum()),
                flush=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "transfer", "subjects-dir", "prediction-runs", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--panel-dirs", nargs=2, type=Path, required=True)
    p.add_argument("--prediction-method", required=True)
    p.add_argument("--subjects", nargs="+")
    run(p.parse_args())
