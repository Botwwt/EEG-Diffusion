"""Refit the passed shared-response model on 512 generic classes for the locked fourth-panel render."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fit_shared_photo_eeg import SUBJECTS, fit_all, read, require, sha, write
from fit_within_photo_embedding import images


def lock_application(a):
    root = a.shared_root
    design = a.public / "shared_photo_eeg_design.json"
    evidence = a.public / "shared_photo_eeg.json"
    reg, report = read(design), read(evidence)
    require(
        report["passed_generation_gate"]
        and sha(design) == report["design_sha256"]
        and report["subjects"] == list(SUBJECTS)
        and sha(root / "result.json") == sha(evidence),
        "Shared-direction generic gate was not verified",
    )
    base = a.prospective / "render_inputs/expanded_r4"
    manifest = read(base / "design.json")
    classes = sorted(reg["train_classes"] + reg["test_classes"])
    require(
        len(set(classes)) == 512
        and not set(classes).intersection(reg["excluded_panels"]["prospective"]),
        "Wrong full generic fit",
    )
    data = dict(
        study="Fourth-panel development comparison after the panel EEG object-word readout; no independent panel confirmation.",
        generic_design_sha256=sha(design),
        generic_result_sha256=sha(evidence),
        generic_classes=classes,
        participants=list(SUBJECTS),
        source_panel="prospective",
        public_panel_sha256=sha(a.public / "prospective_panel_design.json"),
        baseline_design_sha256=sha(base / "design.json"),
        baseline_radius_sha256=sha(base / "radius.json"),
        baseline_packets_sha256=manifest["packets_sha256"],
        model="Refit the passed unaltered 50:50 direct+rank64 cross-participant ridge on all 512 generic classes. Only current participant's query/support EEG enters inference.",
        intervention="Keep each recipient's frozen expanded_r4 predicted word, eight eligible support rows, support direction, original angular amplitude, noise seed, Turbo/IP settings and visual radius. Replace the tangent with the new EEG direction. For the wrong condition use the donor's new tangent at the recipient's original amplitude. Within a word/fold the eight supports and center are identical; singleton uses own. Uniform is copied unchanged from the baseline.",
    )
    path = root / "final/application_lock.json"
    if path.exists():
        require(read(path) == data, "Application lock differs")
    write(path, data)
    return path


def fit_final(a):
    application = lock_application(a)
    plan = read(application)
    local = SimpleNamespace(expanded=a.expanded, out=application.parent)
    fit_all(local, dict(train_classes=plan["generic_classes"]), images(a.source))
    write(
        local.out / "fit.json",
        dict(
            application_lock_sha256=sha(application),
            generic_fit_driver_sha256=sha(Path(__file__).with_name("fit_shared_photo_eeg.py")),
            heads_sha256={s: sha(local.out / (s + ".npz")) for s in SUBJECTS},
        ),
    )


def package(a):
    from package_repeat_photo_subspace import refinement_packets

    registration = lock_application(a)
    local = registration.parent
    fit = read(local / "fit.json")
    require(fit["application_lock_sha256"] == sha(registration), "Different fitted model contract")

    def predictor(s, target_data, _bank):
        path = local / (s + ".npz")
        require(sha(path) == fit["heads_sha256"][s], "Wrong participant's fitted heads")
        with np.load(path, allow_pickle=False) as z:
            x = np.asarray(target_data.p, dtype=np.float64)
            require(
                x.shape == (120, 1003) and z["shared"].shape == (1003, 1024),
                "Invalid participant feature array",
            )
            return (
                (x / z["scale"]) @ z["shared"],
                (x / z["scale"]) @ z["direct"],
                dict(
                    subject=s,
                    head_sha256=sha(path),
                    weight_source="Three participants' generic calibration data, current participant's query EEG",
                    application_lock_sha256=sha(registration),
                ),
            )

    destination = a.prospective / "refinements/shared"
    hashes = refinement_packets(
        a,
        predictor,
        "shared",
        a.public / "shared_photo_eeg_design.json",
        a.public / "shared_photo_eeg.json",
        destination,
    )
    parity = read(destination / "parity.json")
    write(
        a.public / "shared_photo_eeg_application.json",
        dict(
            protocol=read(registration)["study"],
            intervention=read(registration)["intervention"],
            generic_design_sha256=sha(a.public / "shared_photo_eeg_design.json"),
            generic_result_sha256=sha(a.public / "shared_photo_eeg.json"),
            input_lock_sha256=sha(registration),
            refit_driver_sha256=fit["generic_fit_driver_sha256"],
            model_sha256=fit["heads_sha256"],
            condition_packets_sha256=hashes,
            renderer_input_design_sha256=sha(destination / "design.json"),
            baseline_design_sha256=read(destination / "design.json")["baseline_design_sha256"],
            radius_sha256=sha(destination / "radius.json"),
            per_subject={
                s: {
                    k: parity[s][k]
                    for k in (
                        "historical_replay_bit_exact",
                        "direct_ridge_max_abs_replay_error",
                        "uniform_bit_exact",
                        "baseline_packet_sha256",
                        "target_packet_sha256",
                    )
                }
                for s in SUBJECTS
            },
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fit", "package"))
    p.add_argument("--shared-root", type=Path, default=Path("outputs/shared-photo-eeg"))
    p.add_argument(
        "--public", type=Path, default=Path("research/eeg_stack_20261112/results/object_fidelity")
    )
    # Common refinement packager uses the same paths as the posterior baseline replay.
    for k, v in dict(
        expanded="outputs/photo-structure/expanded",
        source="outputs/official-object-readout/sub-03",
        prospective="outputs/prospective-photo",
        out="outputs/full-head-512",
        inputs="outputs/object-transfer",
        angle="outputs/photo-structure/residual_angle",
        within="outputs/photo-structure/within_photo",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    a = p.parse_args()
    (fit_final if a.stage == "fit" else package)(a)


if __name__ == "__main__":
    main()
