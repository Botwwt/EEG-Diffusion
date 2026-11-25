"""Freeze both gated visual-input roots for the four new sub-04 menus and build a small Slurm input tar."""

import argparse
import io
import os
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
from fit_within_photo_embedding import H14_SHA, bases, images, predict, training
from package_full_head_512 import directions
from prepare_scale_trial import PUBLIC
from prepare_trial_object_inputs import check
from register_fresh_mix import LOCK, SUB
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import packet
from run_prospective_readout import frozen_npz


def package(a):
    design = read(LOCK)
    score = read(a.out / "summary.json")
    check(
        sha(LOCK) == sha(a.out / "design.json") == score["design_sha256"]
        and score["images_authorized_by_registered_gate"]
        and sha(a.out / "summary.json") == sha(PUBLIC / "fresh_mix.json"),
        "Prospective photo gate not passed",
    )
    baseline = read(PUBLIC / "trial_object_personal12.json")
    original = a.data / "trial-object-encoder/personal12/render_inputs"
    check(
        sha(original / "design.json") == baseline["render_design_sha256"]
        and sha(original / "radius.json") == baseline["radius_sha256"],
        "Frozen generator or visual radius changed",
    )
    source = a.out / "eeg/sub-04.npz"
    check(sha(source) == score["target_sha256"], "New target EEG changed")
    with np.load(source, allow_pickle=False) as eeg:
        rows0 = eeg["rows"]
        physical = eeg["features"].astype(float)
    bank = images(a.data / "official-object-readout/sub-03")
    classes, x, y = training(
        SimpleNamespace(expanded=a.data / "photo-structure/expanded"), SUB, 512, bank
    )
    check(not np.intersect1d(classes, rows0 // 10).size, "Target enters generic visual fit")
    select = a.data / "photo-structure/within_photo/selection.json"
    angle = a.data / "photo-structure/residual_angle/scale.json"
    old = original / "conditions/sub-04/personal12/conditions_512.npz"
    check(sha(old) == baseline["mixed_packet_sha256"], "Old packet changed")
    with np.load(old, allow_pickle=False) as z:
        check(
            str(z["selection_sha256"]) == sha(select)
            and str(z["angular_scale_sha256"]) == sha(angle)
            and np.array_equal(z["training_classes"], classes)
            and str(z["visual_sha256"]) == H14_SHA,
            "Visual decoder inputs differ from the legacy model",
        )
    basis = bases(x, y, physical)
    residual = predict(basis, read(select)["selected"]) @ basis[2].T
    records = {k: {} for k in ("legacy", "learned")}
    root = a.out / "render_inputs"
    settings = {}
    for name, menu in design["panels"].items():
        regpath = a.out / "references" / (name + "_ablation.json")
        dpath = a.out / "decisions" / (name + ".npz")
        check(
            sha(regpath) == menu["registration_sha256"]
            and sha(dpath) == score["decision_sha256"][name],
            "Registration or sealed words changed",
        )
        reg = read(regpath)
        rows, words, folds, cal = (
            np.array(reg[k])
            for k in ("rows", "words", "folds", "calibration_rows_by_fold_and_word")
        )
        lock = a.out / "references" / (name + "_sub-04_lock.json")
        freeze(
            lock,
            dict(
                participants=[SUB],
                words=words.tolist(),
                registration_sha256=sha(regpath),
                study_sha256=sha(LOCK),
            ),
        )
        panel = {}
        with np.load(dpath, allow_pickle=False) as d:
            check(
                np.array_equal(d["rows"], rows)
                and np.array_equal(d["folds"], folds)
                and np.array_equal(d["words"], words),
                "Row order differs",
            )
            for arm in records:
                word = root / arm / "evaluations" / SUB / name / "word_predictions.json"
                r = read(word)
                check(
                    r["predicted"] == d[arm].tolist()
                    and r["registration_sha256"] == sha(regpath)
                    and r["rows"] == reg["rows"]
                    and r["words"] == reg["words"],
                    "Sealed classifier choices differ",
                )
                conditions = directions(
                    rows, folds, words, cal, d[arm], residual, rows0, bank, read(angle)
                )
                path = root / arm / "conditions" / SUB / name / "conditions_512.npz"
                frozen_npz(
                    path,
                    conditions
                    | dict(
                        subject=SUB,
                        physical_trials_per_photo=4,
                        training_classes=classes,
                        visual_sha256=H14_SHA,
                        target_eeg_sha256=sha(source),
                        word_prediction_sha256=sha(word),
                        registration_sha256=sha(regpath),
                        selection_sha256=sha(select),
                        angular_scale_sha256=sha(angle),
                        trial_readout_design_sha256=sha(LOCK),
                        classifier=arm,
                    ),
                )
                records[arm][f"{SUB}/{name}"] = sha(path)
                panel[arm] = conditions
        same = panel["legacy"]["predicted_words"] == panel["learned"]["predicted_words"]
        check(
            all(
                np.array_equal(panel["legacy"][k][same], panel["learned"][k][same])
                for k in ("own", "uniform", "support_rows")
            ),
            "Same word changed expected H14 or support photos",
        )
        settings[name] = dict(
            same_word_count=int(same.sum()),
            distinct_queries=120,
            strict_query_count=len(menu["primary_object_ids"]) * 10,
            same_word_own_max_difference=0 if same.any() else None,
            scorer_lock_sha256=sha(lock),
        )
    regs = {f"{SUB}/{k}": v["registration_sha256"] for k, v in design["panels"].items()}
    for arm, packets in records.items():
        dest = root / arm
        freeze(
            dest / "design.json",
            read(original / "design.json")
            | dict(
                cohort=[SUB],
                panels=list(design["panels"]),
                packets_sha256=packets,
                protocol_sha256=sha(LOCK),
                registration_sha256=regs,
            ),
        )
        shutil.copyfile(original / "radius.json", dest / "radius.json")
        for name in design["panels"]:
            packet(dest, SUB, name)
    receipt = dict(
        study_sha256=sha(LOCK),
        word_score_sha256=sha(a.out / "summary.json"),
        radius_sha256=sha(original / "radius.json"),
        classifier_sha256={k: sha(root / k / "design.json") for k in records},
        packet_sha256=records,
        menus=settings,
        seed_rule="923761 + original training photo row, repeated across arms and classifiers",
    )
    freeze(a.out / "render_packets.json", receipt)
    archive = a.out / "cluster/fresh_mix_render_inputs.tar"
    archive.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        a.out / "render_packets.json",
        a.out / "design.json",
        *sorted((a.out / "references").glob("*.json")),
    ]
    for arm in records:
        d = root / arm
        paths.extend(
            [
                d / "design.json",
                d / "radius.json",
                *sorted((d / "conditions").rglob("*.npz")),
                *sorted((d / "evaluations").rglob("*.json")),
            ]
        )
    blind = archive.with_name("fresh_mix_blind_inputs.tar")
    for dest, members in (
        (archive, [(p, p.relative_to(a.out).as_posix()) for p in paths]),
        (
            blind,
            [
                (p, "inputs/" + p.relative_to(root).as_posix())
                for p in paths
                if p.is_relative_to(root)
            ],
        ),
    ):
        with tarfile.open(dest, "w", format=tarfile.USTAR_FORMAT) as tar:
            for path, name in members:
                payload = path.read_bytes()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(payload))
        print(dest, sha(dest), len(members), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/fresh-mix"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    package(p.parse_args())
