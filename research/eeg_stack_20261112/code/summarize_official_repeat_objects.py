"""Verify and summarize the single-subject repeat readout and its generic angle study."""

import argparse
from pathlib import Path

import numpy as np
from calibrate_official_repeat_angles import conditions, difference, frame, rows
from export_official_global_angle import PUBLIC, digest
from fit_official_repeat_objects import S, dump, fit, read, require, sha, training, unit


def verify_packet(root, design_path):
    receipt = read(root / S / "global_angle_packet.json")
    changed = root / S / "global_angle_packet.npz"
    native = root / S / "render_packet.npz"
    config = read(design_path)
    require(
        sha(changed) == receipt["packet_sha256"]
        and sha(native) == receipt["original_packet_sha256"] == config["native_packet_sha256"]
        and sha(design_path) == receipt["design_sha256"],
        "Packet provenance differs",
    )
    with np.load(root / S / "1012/visual_coordinates.npz", allow_pickle=False) as data:
        center = data["mean"].astype(np.float64)
    center /= np.linalg.norm(center)
    require(digest(center, "<f8") == receipt["unit_center_sha256"], "Changed training center")
    with (
        np.load(changed, allow_pickle=False) as after,
        np.load(native, allow_pickle=False) as before,
    ):
        same = {"rows", "r4_words", "r80_words", "donor", "radius"}
        require(set(after) == set(before) == same | {"r4", "r80"}, "Wrong field set")
        require(
            all(np.array_equal(after[k], before[k]) for k in same),
            "Words, rows, donor or radius changed",
        )
        require(
            np.array_equal(after["rows"], np.arange(200))
            and np.array_equal(after["donor"], np.roll(np.arange(200), 67)),
            "Row/donor registration changed",
        )
        for k in ("r4", "r80"):
            x = before[k].astype(np.float64)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            cos = np.clip(x @ center, -1, 1)
            axis = x - cos[:, None] * center
            axis /= np.linalg.norm(axis, axis=1, keepdims=True)
            theta = np.minimum(
                np.arccos(cos) * config["calibrated_gain"], config["calibrated_cap_radians"]
            )
            expected = np.cos(theta)[:, None] * center + np.sin(theta)[:, None] * axis
            actual = after[k]
            require(
                actual.shape == (200, 1024)
                and np.max(abs(expected - actual)) < 1e-7
                and np.max(abs(np.linalg.norm(actual, axis=1) - 1)) < 2e-7
                and digest(actual, "<f4")
                == receipt["directions"][k]["output_unit_direction_sha256"],
                f"Independent {k} angle replay failed",
            )
    return receipt


def paired_generic_angle(a, registration, angle_registration, validation):
    calibration = np.array(angle_registration["angle_calibration_classes"])
    held = np.array(angle_registration["angle_validation_classes"])
    trained = np.setdiff1d(registration["extended_classes"], np.r_[held, calibration])
    require(len(trained) == 884 and len(set(held) | set(calibration)) == 128, "Unexpected split")
    data = training(a, registration)
    # Match the original batched EEG projection; score only the disjoint validation classes.
    ordered = np.r_[calibration, held]
    x = np.concatenate([data[0][k] for k in ordered])
    _, _, native = fit(trained, data, x)
    axis, q = frame(native, unit(np.concatenate([data[1][k] for k in trained])))
    truth = unit(np.concatenate([data[1][k] for k in held]))
    params = read(a.root / "angles/calibration.json")["parameters"]
    directions = conditions(
        (native["mean"] + native["ridge"])[10 * len(calibration) :], (axis, q), params
    )
    arm_scores = {k: rows(v, truth) for k, v in directions.items()}
    for k, scores in arm_scores.items():
        old = validation["arms"][k]
        require(
            scores["top1"].sum() == old["within_class_top1"]
            and np.isclose(scores["within_pair"].mean(), old["within_class_pairwise_win"]),
            "Mismatch with the original generic ranking",
        )
    return {
        k: difference(arm_scores["global"][k], arm_scores["native"][k])
        for k in ("within_pair", "top1", "raw_cosine")
    }


def run(a):
    root = a.root
    report = read(root / "evaluation.json")
    inference = read(root / "inference_receipt.json")
    design = read(root / "design.json")
    selection = read(root / "selection.json")
    selection_key = selection["choice"]
    angles = read(root / "angles/validation.json")
    angle_registration = read(root / "angles/design.json")
    packet = verify_packet(root, a.angle_design)
    require(
        sha(root / "design.json")
        == inference["design_sha256"]
        == selection["selection_design_sha256"]
        and sha(root / "selection.json") == inference["selection_sha256"]
        and sha(root / "inference_receipt.json") == report["inference_receipt_sha256"]
        and sha(root / "angles/calibration.json") == angles["calibration_sha256"]
        and sha(root / "angles/validation.json")
        == read(a.angle_design)["source_generic_validation_sha256"]
        and sha(root / S / "test_budgets.npz") == inference["physical_sha256"],
        "Changed source reports",
    )
    budgets = design["budgets"]
    fields = {
        "all1854_top1": ("full1854", "top1"),
        "all1854_top5": ("full1854", "top5"),
        "wrong1854_top1": ("wrong_eeg_1854", "top1"),
        "wrong1854_top5": ("wrong_eeg_1854", "top5"),
        "names200_top1": ("official200_names", "top1"),
        "photo200_top1": ("eval_only_actual_image_200", "top1"),
        "wrong_photo200_top1": ("wrong_eeg_image_200", "top1"),
        "grand_voltage_mean1854_top1": ("features_of_mean_voltage_1854", "top1"),
    }
    counts = {}
    for n, family in report["by_calibration"].items():
        chosen = family["chosen"]
        counts[n] = {
            name: [chosen[str(b)][source][metric] for b in budgets]
            for name, (source, metric) in fields.items()
        }
    r80 = report["by_calibration"]["1012"]["standard_ridge"]["80"]
    broad = report["exploratory_broadly_related_official_objects"]
    source_reports = (
        "design",
        "selection",
        "native_generic_validation",
        "inference_receipt",
        "evaluation",
        "angles/design",
        "angles/calibration",
        "angles/validation",
        "sub-04/render_packet",
        "sub-04/global_angle_packet",
    )
    output = dict(
        subject=S,
        official_photos=200,
        official_photos_per_object=1,
        budgets=budgets,
        interpretation="Word predictions search all 1854 categories. Official 200-name and actual-image galleries enter only scoring. One picture per official object cannot isolate within-object photo identity. Image-rendering outcomes are reported separately.",
        generic_selection=dict(
            held_objects=128,
            photos=1280,
            fitted_objects=884,
            choice=selection_key,
            chosen=selection["validation"][selection_key],
            standard_ridge=selection["baseline"],
        ),
        official_top_k_counts_by_generic_training_objects=counts,
        denominator_per_budget=200,
        r80_released_sub04_atm=report["released_sub04_atm_r80"],
        r80_standard_ridge={
            k: v
            for k, v in r80.items()
            if k in ("full1854", "official200_names", "eval_only_actual_image_200")
        },
        r80_related_category_exploratory=dict(
            definition=broad["definition"],
            chosen=broad["calibration_1012"]["80"],
            released_atm=broad["released_atm_r80"],
        ),
        official_paired_differences=report["paired_comparisons"],
        native_h14_generic=read(root / "native_generic_validation.json"),
        native_h14_official_r80=report["render_direction_r80_diagnostic"],
        generic_angle=dict(
            fitted_objects=884,
            calibration_objects=64,
            validation_objects=64,
            photos_per_object=10,
            arms=angles["arms"],
            predictable_minus=angles["predictable_minus"],
            predictable_passed_gate=angles["advance"],
            global_minus_native=paired_generic_angle(a, design, angle_registration, angles),
            interval_unit="Paired whole-object bootstrap, 10000 draws of 64 generic classes. Rank differences are fractions, raw cosine has its own units.",
            selection_history="The generic 128 had previously been used for choosing the original semantic model rank. Gain and cap use only 64 calibration objects. The decision to render global gain followed inspection of the disjoint 64-object generic comparison.",
        ),
        global_packet=dict(
            sha256=packet["packet_sha256"],
            gain=packet["gain"],
            cap_radians=packet["cap_radians"],
            design_sha256=packet["design_sha256"],
            unit_center_sha256=packet["unit_center_sha256"],
        ),
        source_hashes=dict(
            original_physical_test_sha256=design["source_pinned_sha256"],
            extracted_eeg_sha256=inference["physical_sha256"],
            source_reports_sha256={p: sha(root / (p + ".json")) for p in source_reports},
        ),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump(a.out, output)
    print(a.out, sha(a.out), output["generic_angle"]["global_minus_native"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, path in dict(
        root="outputs/official-repeat-object",
        source="outputs/official-object-readout/sub-03",
        expanded="outputs/photo-structure/expanded",
        extra="outputs/double-calibration",
        external="outputs/external-object-anchors/sub-03/external_anchors.npz",
        angle_design=PUBLIC / "official_repeat_global_angle_design.json",
        out=PUBLIC / "official_repeat_object_evidence.json",
    ).items():
        p.add_argument("--" + k.replace("_", "-"), type=Path, default=Path(path))
    run(p.parse_args())
