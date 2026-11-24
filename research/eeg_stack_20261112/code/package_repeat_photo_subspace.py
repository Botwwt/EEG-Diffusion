"""Package held-out photo residual models at the frozen posterior readout's per-row angles."""

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
from fit_full_head_512 import target, training
from fit_repeat_photo_subspace import centered, ridge, subspace
from fit_support_structure import unit
from fit_within_photo_embedding import images
from package_full_head_512 import directions
from prepare_full_head_512 import check, defaults
from register_fresh_photo_panel import SUBJECTS, freeze, read, sha
from render_within_photo_ip import packet
from run_prospective_readout import RESULTS, frozen_npz, visual_model

PANEL = "prospective"


def add_paths(p):
    defaults(p)
    paths = dict(
        prospective="outputs/prospective-photo",
        source="outputs/official-object-readout/sub-03",
        within="outputs/photo-structure/within_photo",
        angle="outputs/photo-structure/residual_angle",
    )
    for name, value in paths.items():
        p.add_argument("--" + name, type=Path, default=Path(value))


def repeat_prediction(a, sub, target_data, bank):
    train, _ = training(a, sub)
    rows = train.rows.reshape(512, 10)
    classes = rows[:, 0] // 10
    check(
        np.array_equal(rows, classes[:, None] * 10 + np.arange(10))
        and not np.intersect1d(classes, target_data.rows // 10).size,
        "Invalid generic EEG population",
    )
    y = centered(unit(bank[rows].astype(float)))
    posterior = ridge(train.p.reshape(512, 10, -1), y, target_data.p)
    halves = train.f.reshape(2, 512, 10, -1)
    axes = subspace(halves, shuffle=False)
    full = ridge(halves.mean(0) @ axes, y, target_data.f.mean(0) @ axes)
    return (
        (posterior + full) / 2,
        posterior,
        dict(
            training_classes_sha256=hashlib.sha256(classes.astype("<i8").tobytes()).hexdigest(),
            training_receipt_sha256=sha(a.out / "features" / (sub + ".json")),
            repeated_axes=64,
            generic_classes=len(classes),
        ),
    )


def angle_matched(prior, old_own, residual, donor):
    axis = unit(prior.astype(float))
    old = unit(old_own.astype(float))
    dot = (axis * old).sum(1, keepdims=True)
    angles = np.arctan2(np.linalg.norm(old - dot * axis, axis=1, keepdims=True), dot)
    tangent = residual - (residual * axis).sum(1, keepdims=True) * axis
    check(
        np.all(np.linalg.norm(tangent, axis=1) > 1e-10) and np.array_equal(prior, prior[donor]),
        "Undefined residual direction or donor has a different support prototype",
    )
    tangent = unit(tangent)
    own = (np.cos(angles) * axis + np.sin(angles) * tangent).astype(np.float32)
    wrong = (np.cos(angles) * axis + np.sin(angles) * tangent[donor]).astype(np.float32)
    return own, wrong, angles[:, 0]


def refinement_packets(a, predictor, method, method_design, method_evidence, destination):
    """predictor(sub, target_data, H14_bank) returns candidate, direct-posterior baseline, provenance."""
    gate = read(method_evidence)
    check(
        gate["design_sha256"] == sha(method_design)
        and (gate.get("advance") is True or gate.get("passed_generation_gate") is True),
        "Generic model gate not verified",
    )
    baseline = a.prospective / "render_inputs/expanded_r4"
    baseline_design = read(baseline / "design.json")
    published = read(RESULTS / "prospective_render_inputs.json")
    regpath = a.prospective / "references/prospective_ablation.json"
    reg = read(regpath)
    check(
        published["registration_sha256"] == sha(regpath)
        and published["render_designs_sha256"]["expanded_r4"] == sha(baseline / "design.json"),
        "Unregistered posterior render design",
    )
    rows, folds, vocab, cal = (
        np.asarray(reg[k]) for k in ("rows", "folds", "words", "calibration_rows_by_fold_and_word")
    )
    hashes, checks = {}, {}
    bank = images(a.source)
    for sub in SUBJECTS:
        original_path = baseline / "conditions" / sub / PANEL / "conditions_512.npz"
        old_rows, old_folds, words, donors, allowed, native, _, base_sha = packet(
            baseline, sub, PANEL
        )
        word_file = baseline / "evaluations" / sub / PANEL / "word_predictions.json"
        word_indices = np.asarray(read(word_file)["predicted"])
        check(
            np.array_equal(rows, old_rows)
            and np.array_equal(folds, old_folds)
            and np.array_equal(words, vocab[word_indices])
            and np.array_equal(allowed, cal[folds, word_indices])
            and base_sha == published["cases"][sub]["methods"]["expanded_r4"]["conditions_sha256"]
            and sha(word_file) == published["cases"][sub]["methods"]["expanded_r4"]["words_sha256"],
            "Frozen posterior word or calibration rows changed",
        )
        eeg = a.prospective / "eeg" / (sub + ".npz")
        observed = target(a, sub, a.prospective, False)
        classes, old_model = visual_model(a, sub, reg, eeg, bank)
        replayed = directions(rows, folds, vocab, cal, word_indices, *old_model)
        check(
            all(np.array_equal(replayed[k], native[k]) for k in native)
            and np.array_equal(replayed["donor_rows"], donors),
            "Frozen H/14 packet replay differs",
        )
        predicted, posterior, receipt = predictor(sub, observed, bank)
        check(
            predicted.shape == posterior.shape == (120, 1024)
            and np.isfinite(predicted).all()
            and np.isfinite(posterior).all(),
            "Missing finite H14 readout",
        )
        direct = directions(
            rows, folds, vocab, cal, word_indices, posterior, observed.rows, bank, old_model[-1]
        )
        error = max(float(np.max(np.abs(direct[k] - native[k]))) for k in native)
        check(error < 2e-5, f"Historical direct posterior readout changed: {error}")
        q = np.searchsorted(observed.rows, rows)
        support = np.searchsorted(observed.rows, allowed)
        check(
            np.array_equal(observed.rows[q], rows)
            and np.array_equal(observed.rows[support], allowed),
            "Residual EEG query or support identity changed",
        )
        residual = predicted[q] - predicted[support].mean(1)
        lookup = {r: i for i, r in enumerate(rows)}
        donor = np.asarray([lookup[r] for r in donors])
        own, wrong, angle = angle_matched(native["uniform"], native["own"], residual, donor)
        check(
            np.max(abs(own - native["own"])) > 1e-5,
            "Refinement unexpectedly has only native directions",
        )
        with np.load(original_path, allow_pickle=False) as z:
            data = dict(z)
        check(np.array_equal(data["training_classes"], classes), "Changed training class set")
        dest = destination / "conditions" / sub / PANEL / "conditions_512.npz"
        frozen_npz(
            dest,
            data
            | dict(
                own=own,
                wrong=wrong,
                recipient_angle_radians=angle,
                baseline_packet_sha256=base_sha,
                method_design_sha256=sha(method_design),
                method_evidence_sha256=sha(method_evidence),
            ),
        )
        word_out = destination / "evaluations" / sub / PANEL / word_file.name
        freeze(word_out, read(word_file))
        check(sha(word_out) == sha(word_file), "Word receipt changed on export")
        hashes[f"{sub}/{PANEL}"] = sha(dest)
        checks[sub] = dict(
            baseline_packet_sha256=base_sha,
            historical_replay_bit_exact=True,
            direct_ridge_max_abs_replay_error=error,
            target_packet_sha256=sha(eeg),
            uniform_bit_exact=True,
            model_receipt=receipt,
        )
    design = baseline_design | dict(
        packets_sha256=hashes,
        wrong_direction_mode="recipient_native_angle",
        method=method,
        method_design_sha256=sha(method_design),
        method_evidence_sha256=sha(method_evidence),
        baseline_design_sha256=sha(baseline / "design.json"),
        protocol="Previously frozen posterior words, supports, donor EEG and row seeds. Both residual directions use the recipient's original own angle from its unchanged uniform visual prototype.",
    )
    freeze(destination / "design.json", design)
    radius = destination / "radius.json"
    if not radius.exists():
        shutil.copyfile(baseline / "radius.json", radius)
    check(sha(radius) == published["radius_sha256"], "Original radius changed")
    for sub in SUBJECTS:
        packet(destination, sub, PANEL)
    freeze(destination / "parity.json", checks)
    return hashes


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_paths(p)
    p.add_argument(
        "--method-design", type=Path, default=RESULTS / "repeat_photo_subspace_design.json"
    )
    p.add_argument("--method-evidence", type=Path, default=RESULTS / "repeat_photo_subspace.json")
    p.add_argument(
        "--destination", type=Path, default=Path("outputs/prospective-photo/refinements/repeat")
    )
    args = p.parse_args()
    refinement_packets(
        args,
        lambda sub, data, bank: repeat_prediction(args, sub, data, bank),
        "repeat",
        args.method_design,
        args.method_evidence,
        args.destination,
    )
