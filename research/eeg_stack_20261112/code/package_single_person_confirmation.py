"""Acquire registered R4 EEG and package the previously selected personal 512-class decoder.

Run fetch, then prepare. The transfer tar contains generator code, three
condition packets, their routing information and the frozen scalar radius.
Stimulus references, row labels and scoring data stay in the evaluation tree.
"""

import argparse
import io
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from decode_eeg_layout import CHANNELS
from fit_fresh_photo_panel import apply, physical
from fit_within_photo_embedding import images
from register_fresh_photo_panel import freeze, read, sha
from register_single_person_confirmation import LOCK_SHA, register
from render_within_photo_ip import packet
from run_prospective_readout import frozen_npz

CODE = Path(__file__).resolve().parent
RESULTS = CODE.parent / "results/object_fidelity"
PROTOCOL = RESULTS / "single_person_confirmation_image_design.json"
EXPECTED = "57900a06ac4b385bc8cf05aa1ac50fe530dd147acb2a6993a9f973b6589f0670"
PANEL = "personal12"


def check(ok, message):
    if not ok:
        raise ValueError(message)


def references(a):
    check(sha(PROTOCOL) == EXPECTED, "Different image protocol")
    spec = read(PROTOCOL)
    lock = a.out / "candidate_panel_lock.json"
    register(lock, None)
    check(sha(lock) == spec["panel_sha256"] == LOCK_SHA, "Different participants, classes or folds")
    inputs = dict(
        selection=a.within / "selection.json",
        angular_scale=a.angle / "scale.json",
        radius=a.angle / "radius.json",
        render_base=a.angle / "design.json",
        registered_native_readout=CODE / "fit_fresh_photo_panel.py",
        registered_visual_embedding=CODE / "fit_within_photo_embedding.py",
        renderer=CODE / "render_within_photo_ip.py",
    )
    check(
        {k: sha(p) for k, p in inputs.items()} == spec["source_sha256"],
        "Previously selected readout, radius or renderer changed",
    )
    study = read(lock)
    registries = {}
    for s in spec["participant_order"]:
        local = a.out / "references" / f"{PANEL}_{s}_ablation.json"
        registration = study["panels"][s]["registration"]
        freeze(local, registration)
        freeze(
            a.out / "references" / f"{PANEL}_{s}_lock.json",
            dict(
                participants=[s],
                words=registration["words"],
                registration_sha256=sha(local),
                session_pair=study["panels"][s]["session_pair"],
                physical_trials_per_photo=4,
                panel_sha256=LOCK_SHA,
                protocol_sha256=EXPECTED,
            ),
        )
        registries[s] = local
    return registries


def prepare(a, refs):
    bank = images(a.source)
    dest = a.out / "render_inputs"
    signed = {}
    for s, ref in refs.items():
        eeg = a.out / "eeg" / (s + ".npz")
        receipt = read(eeg.with_suffix(".json"))
        reg = read(ref)
        physical_input = read(a.inputs / (s + ".json"))
        check(
            receipt["packet_sha256"] == sha(eeg)
            and receipt["registration_sha256"] == sha(ref)
            and receipt["subject"] == s
            and receipt["four_physical_repeats"]
            and receipt["source_full_lfs_sha256"] == physical_input["source_full_lfs_sha256"]
            and len(receipt["blocks"]) == 12,
            "Incorrect physical source",
        )
        with np.load(eeg, allow_pickle=False) as z:
            sr = z["rows"]
            check(
                np.array_equal(sr, sorted(reg["rows"]))
                and z["channels"].tolist() == CHANNELS
                and z["half_eeg"].shape == (2, 120, 17, 250)
                and int(z["repeats_per_half"]) == 2,
                "Invalid R4 source rows, trial count or axes",
            )
            average = z["half_eeg"].mean(0)
        # Only the allowed support index array and its word vocabulary are passed
        # to the prediction functions; truth remains confined to the evaluation registry.
        safe = {k: reg[k] for k in ("rows", "folds", "words", "calibration_rows_by_fold_and_word")}
        idx, data, _ = apply(a, s, safe, bank, sr, average)
        wordfile = dest / "evaluations" / s / PANEL / "word_predictions.json"
        freeze(
            wordfile,
            dict(
                subject=s,
                panel=PANEL,
                physical_trials_per_photo=4,
                rows=reg["rows"],
                folds=reg["folds"],
                words=reg["words"],
                predicted=idx.tolist(),
                registration_sha256=sha(ref),
                protocol_sha256=EXPECTED,
            ),
        )
        save = dest / "conditions" / s / PANEL / "conditions_512.npz"
        frozen_npz(
            save,
            data
            | dict(
                subject=s,
                physical_trials_per_photo=4,
                target_eeg_sha256=sha(eeg),
                word_prediction_sha256=sha(wordfile),
                registration_sha256=sha(ref),
                angular_scale_sha256=sha(a.angle / "scale.json"),
                prospective_protocol_sha256=EXPECTED,
            ),
        )
        signed[f"{s}/{PANEL}"] = sha(save)
    base = read(a.angle / "design.json")
    config = {
        k: base[k]
        for k in (
            "visual",
            "generic_radius_rows",
            "radius_seed",
            "space_check",
            "model",
            "adapter_sha256",
            "visual_sha256",
            "angular_calibration",
        )
    }
    freeze(
        dest / "design.json",
        config
        | dict(
            panels=[PANEL],
            cohort=list(refs),
            physical_trials_per_photo=4,
            wrong_direction_mode="donor_own",
            packets_sha256=signed,
            protocol_sha256=EXPECTED,
            registration_sha256={f"{s}/{PANEL}": sha(r) for s, r in refs.items()},
        ),
    )
    shutil.copyfile(a.angle / "radius.json", dest / "radius.json")
    for s in refs:
        packet(dest, s, PANEL)
    tar = a.out / "personal12-input.tar"
    members = {"design.json": dest / "design.json", "radius.json": dest / "radius.json"}
    for s in refs:
        name = f"conditions/{s}/{PANEL}/conditions_512.npz"
        members[name] = dest / name
    for file in (
        "render_within_photo_ip.py",
        "render_official_ip_adapter.py",
        "render_official_words.py",
    ):
        members["code/" + file] = CODE / file
    with tarfile.open(tar, "w") as archive:
        for name, p in sorted(members.items()):
            raw = p.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    summary = dict(
        study_sha256=LOCK_SHA,
        protocol_sha256=EXPECTED,
        packets_sha256=signed,
        render_design_sha256=sha(dest / "design.json"),
        radius_sha256=sha(dest / "radius.json"),
        target_eeg_sha256={s: sha(a.out / "eeg" / (s + ".npz")) for s in refs},
        registrations_sha256={s: sha(p) for s, p in refs.items()},
        transfer_bytes=tar.stat().st_size,
        transfer_sha256=sha(tar),
    )
    freeze(a.out / "render_packets.json", summary)
    print("Verified three 120-row render packets", summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fetch", "prepare", "all"))
    parser.add_argument("--out", type=Path, default=Path("outputs/object-head-confirmation"))
    parser.add_argument("--data", type=Path, default=Path("outputs"))
    parser.add_argument("--workers", type=int, choices=range(1, 6), default=4)
    user = parser.parse_args()
    a = SimpleNamespace(
        out=user.out,
        workers=user.workers,
        inputs=user.data / "object-transfer",
        expanded=user.data / "photo-structure/expanded",
        source=user.data / "official-object-readout/sub-03",
        angle=user.data / "photo-structure/residual_angle",
        within=user.data / "photo-structure/within_photo",
    )
    refs = references(a)
    if user.stage in ("fetch", "all"):
        for sub, ref in refs.items():
            physical(a, sub, np.asarray(read(ref)["rows"]), False, ref)
    if user.stage in ("prepare", "all"):
        prepare(a, refs)
