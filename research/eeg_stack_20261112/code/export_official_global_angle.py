"""Register and export the follow-up global-angle comparator without opening any official target."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from calibrate_official_repeat_angles import expmap, tangent
from fit_official_repeat_objects import S, dump, read, require, sha

PUBLIC = Path(__file__).resolve().parents[1] / "results" / "object_fidelity"
ANGLE_DESIGN = "30e37b34380c205335ec31a76192b081b5f015c1bfe4634b02710348c481609f"
NATIVE = "214c21ee3c1c53e7d40927a536d9684b164e11fe9a94ec1c9e8d6ed1257a1983"


def digest(x, dtype):
    return hashlib.sha256(np.ascontiguousarray(x, dtype=dtype).tobytes()).hexdigest()


def freeze(path, value):
    if path.exists():
        require(read(path) == value, "Refusing to change an existing study receipt")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(path, value)


def register(a):
    angles = a.root / "angles"
    fit = read(angles / "calibration.json")
    previous = read(angles / "validation.json")
    require(
        sha(angles / "design.json")
        == ANGLE_DESIGN
        == fit["design_sha256"]
        == previous["design_sha256"]
        and sha(angles / "calibration.json") == previous["calibration_sha256"]
        and previous["advance"] is False,
        "Earlier subspace study or its outcome changed",
    )
    require(sha(a.root / S / "render_packet.npz") == NATIVE, "Native comparison was not recovered")
    config = dict(
        protocol="Secondary follow-up of a named comparator in the prior subspace experiment. The 64 validation classes were examined before selecting the comparator for generation. The same 64 validation classes cannot provide an independent confirmation of this follow-up.",
        population="Single participant sub-04; 200 official test objects with one distinct photo per object; compare 4 versus 80 measured EEG presentations, with a sample-matched 80-repeat wrong-EEG control.",
        source_angle_design_sha256=ANGLE_DESIGN,
        source_angle_fit_sha256=sha(angles / "calibration.json"),
        source_generic_validation_sha256=sha(angles / "validation.json"),
        native_packet_sha256=NATIVE,
        original_object_design_sha256=sha(a.root / "design.json"),
        native_model_sha256=read(a.root / "inference_receipt.json")["coordinates_sha256"]["1012"],
        calibrated_gain=fit["parameters"]["global_gain"],
        calibrated_cap_radians=fit["parameters"]["real_angle_cap_radians"],
        fit="Scalar gain is median real H14 angle / median predicted H14 angle across the 64 generic calibration classes, using ten generic photos each. The cap is the 95th percentile of true calibration angles. The failed direction-dependent candidate remains a separate experiment.",
        export="Use the unit direction of the 1012-class training image mean as sphere center. For each original native unit estimate retain its tangent direction and set theta_new=min(gain*theta, cap). Return a unit vector to the identical ViT-H IP-Adapter. Original donor permutation, full-1854 word predictions, scalar radius and row order are copied exactly.",
        generation="Compare native and globally scaled R4, R80 and deranged R80 directions under the identical generic prompt, noise seed, recipient order, text, adapter radius and diffusion schedule. No target labels, target image pixels or target image embeddings are used to form these conditions.",
        outcomes="Reopen actual JPEGs and score independent full-1854 object names and official 200-photo retrieval, including wrong-EEG and R4 controls. The true 200 names or images may enter scoring only. One photo per object precludes within-object photograph discrimination.",
        no_tuning="No adjustment to gain, cap, H14 readout, amplitudes or sampling using official labels, images or rendered outcomes.",
    )
    freeze(a.design, config)
    print("Registered", a.design, sha(a.design), flush=True)
    return config


def export(a):
    require(a.design.is_file(), "Register the follow-up before creating its test directions")
    config = register(a)
    native = a.root / S / "render_packet.npz"
    model = a.root / S / "1012/visual_coordinates.npz"
    require(sha(model) == config["native_model_sha256"], "Incorrect native H14 training center")
    with np.load(model, allow_pickle=False) as z:
        mean = z["mean"].astype(np.float64)
    center = mean / np.linalg.norm(mean)
    require(center.shape == (1024,) and np.isfinite(center).all(), "Invalid native training center")
    with np.load(native, allow_pickle=False) as z:
        original = dict(z)
    changed = original.copy()
    metadata = {}
    for tag in ("r4", "r80"):
        value = original[tag]
        delta = tangent(value, center)
        old = np.linalg.norm(delta, axis=1)
        new = np.minimum(old * config["calibrated_gain"], config["calibrated_cap_radians"])
        enlarged = expmap(
            delta * config["calibrated_gain"], center, config["calibrated_cap_radians"]
        )
        actual = tangent(enlarged, center)
        require(
            np.max(abs(np.linalg.norm(actual, axis=1) - new)) < 2e-6
            and np.allclose(
                actual / np.linalg.norm(actual, axis=1)[:, None], delta / old[:, None], atol=2e-6
            )
            and np.allclose(np.linalg.norm(enlarged, axis=1), 1, atol=2e-6),
            "Changed orientation or angle",
        )
        changed[tag] = enlarged
        metadata[tag] = dict(
            input_unit_direction_sha256=digest(value, "<f4"),
            output_unit_direction_sha256=digest(enlarged, "<f4"),
            original_median_theta=float(np.median(old)),
            adjusted_median_theta=float(np.median(new)),
            n_at_cap=int(np.sum(new == config["calibrated_cap_radians"])),
        )
    out = a.root / S / "global_angle_packet.npz"
    if not out.exists():
        np.savez(out, **changed)
    with np.load(out, allow_pickle=False) as z:
        require(
            set(z) == set(original) and all(np.array_equal(z[k], changed[k]) for k in original),
            "Global packet differs from registered conditions",
        )
    receipt = dict(
        design_sha256=sha(a.design),
        original_packet_sha256=sha(native),
        packet_sha256=sha(out),
        unit_center_sha256=digest(center, "<f8"),
        center_encoding="C-order float64 little endian, 1024 values",
        original_mean_sha256=digest(mean, "<f4"),
        direction_encoding="C-order float32 little endian, 200 by 1024",
        scalar_radius=float(original["radius"]),
        gain=config["calibrated_gain"],
        cap_radians=config["calibrated_cap_radians"],
        directions=metadata,
    )
    freeze(a.root / S / "global_angle_packet.json", receipt)
    freeze(a.receipt, receipt)
    print("Exported", out, sha(out), "center", receipt["unit_center_sha256"], metadata, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("register", "export"))
    p.add_argument("--root", type=Path, default=Path("outputs/official-repeat-object"))
    p.add_argument(
        "--design", type=Path, default=PUBLIC / "official_repeat_global_angle_design.json"
    )
    p.add_argument(
        "--receipt", type=Path, default=PUBLIC / "official_repeat_global_angle_packet.json"
    )
    a = p.parse_args()
    (register if a.stage == "register" else export)(a)
