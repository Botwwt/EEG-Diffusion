"""Prepare gated legacy and expanded image conditions for the two replication participants."""

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
from prepare_fresh_people import DESIGN, PUBLIC, SUBJECTS
from prepare_trial_object_inputs import check
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import packet
from run_prospective_readout import frozen_npz


def subject_inputs(a, sub, spec, result, bank):
    root = a.out / sub
    old = a.data / "trial-object-replication" / sub / "render_inputs"
    first = read(a.data / "trial-object-replication" / sub / "receipt.json")
    info = read(PUBLIC / "trial_object_personal12_replication.json")
    check(
        sha(root / "fit/receipt.json") == result["trained"][sub]
        and sha(old / "design.json") == first["render_design_sha256"]
        and sha(old / "radius.json") == first["radius_sha256"]
        and info["participant_results"][sub] == first,
        "Changed model or old generator registration",
    )
    target = root / "eeg" / (sub + ".npz")
    physical_receipt = read(target.with_suffix(".json"))
    check(
        sha(target) == physical_receipt["packet_sha256"] and physical_receipt["subject"] == sub,
        "Target EEG mismatch",
    )
    with np.load(target, allow_pickle=False) as data:
        rr = data["rows"]
        q = data["features"].astype(float)
    classes, x, y = training(
        SimpleNamespace(expanded=a.data / "photo-structure/expanded"), sub, 512, bank
    )
    check(not np.intersect1d(classes, rr // 10).size, "Target enters photo conditioning fit")
    old_packet = old / "conditions" / sub / "personal12/conditions_512.npz"
    selection = a.data / "photo-structure/within_photo/selection.json"
    angle = a.data / "photo-structure/residual_angle/scale.json"
    with np.load(old_packet, allow_pickle=False) as ref:
        check(
            sha(old_packet) == first["mixed_packet_sha256"]
            and np.array_equal(ref["training_classes"], classes)
            and str(ref["selection_sha256"]) == sha(selection)
            and str(ref["angular_scale_sha256"]) == sha(angle),
            "Visual model changed",
        )
    components = bases(x, y, q)
    fitted = predict(components, read(selection)["selected"]) @ components[2].T
    regs = {}
    packets = {arm: {} for arm in ("legacy", "learned")}
    matches = {}
    for name, menu in spec["panels"][sub].items():
        ref = root / "references" / (name + "_ablation.json")
        readout = root / "decisions" / (name + ".npz")
        check(
            sha(ref) == menu["registration_sha256"]
            and sha(readout) == result["decisions"][sub][name],
            "Fixed decisions or supports changed",
        )
        r = read(ref)
        rows, folds, words, cal = (
            np.array(r[k]) for k in ("rows", "folds", "words", "calibration_rows_by_fold_and_word")
        )
        key = f"{sub}/{name}"
        regs[key] = sha(ref)
        freeze(
            root / "references" / (name + "_" + sub + "_lock.json"),
            dict(
                participants=[sub],
                words=r["words"],
                registration_sha256=sha(ref),
                study_sha256=sha(DESIGN),
            ),
        )
        both = {}
        with np.load(readout, allow_pickle=False) as rec:
            check(
                np.array_equal(rec["rows"], rows)
                and np.array_equal(rec["folds"], folds)
                and np.array_equal(rec["words"], words),
                "Displaced sealed prediction",
            )
            for arm in packets:
                dest = root / "render_inputs" / arm
                labels = rec[arm]
                v = directions(rows, folds, words, cal, labels, fitted, rr, bank, read(angle))
                word = dest / "evaluations" / sub / name / "word_predictions.json"
                freeze(
                    word,
                    dict(
                        subject=sub,
                        panel=name,
                        physical_trials_per_photo=4,
                        rows=r["rows"],
                        folds=r["folds"],
                        words=r["words"],
                        predicted=labels.tolist(),
                        registration_sha256=sha(ref),
                        protocol_sha256=sha(DESIGN),
                    ),
                )
                path = dest / "conditions" / sub / name / "conditions_512.npz"
                frozen_npz(
                    path,
                    v
                    | dict(
                        subject=sub,
                        physical_trials_per_photo=4,
                        training_classes=classes,
                        visual_sha256=H14_SHA,
                        target_eeg_sha256=sha(target),
                        word_prediction_sha256=sha(word),
                        registration_sha256=sha(ref),
                        selection_sha256=sha(selection),
                        angular_scale_sha256=sha(angle),
                        word_design_sha256=sha(DESIGN),
                        classifier=arm,
                    ),
                )
                packets[arm][key] = sha(path)
                both[arm] = v
        same = both["legacy"]["predicted_words"] == both["learned"]["predicted_words"]
        check(
            all(
                np.array_equal(both["legacy"][k][same], both["learned"][k][same])
                for k in ("own", "uniform", "support_rows")
            ),
            "Visual condition changed on equal word",
        )
        matches[name] = int(same.sum())
    for arm, checksums in packets.items():
        dest = root / "render_inputs" / arm
        freeze(
            dest / "design.json",
            read(old / "design.json")
            | dict(
                cohort=[sub],
                panels=list(spec["panels"][sub]),
                packets_sha256=checksums,
                registration_sha256=regs,
                protocol_sha256=sha(DESIGN),
            ),
        )
        shutil.copyfile(old / "radius.json", dest / "radius.json")
        for name in spec["panels"][sub]:
            packet(dest, sub, name)
    return dict(
        packets=packets,
        same_word_rows=matches,
        target_eeg_sha256=sha(target),
        radius_sha256=sha(old / "radius.json"),
    )


def package(a):
    spec = read(DESIGN)
    words = PUBLIC / "fresh_people_words.json"
    evidence = read(words)
    check(
        evidence["advance"] and evidence["design_sha256"] == sha(DESIGN), "Replication gate not met"
    )
    bank = images(a.data / "official-object-readout/sub-03")
    report = {s: subject_inputs(a, s, spec, evidence, bank) for s in SUBJECTS}
    freeze(
        a.out / "render_packets.json",
        dict(
            design_sha256=sha(DESIGN),
            words_sha256=sha(words),
            packaged=report,
            items=16,
            contributing_participants=list(SUBJECTS),
        ),
    )
    out = a.out / "cluster/fresh_people_blind_inputs.tar"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w", format=tarfile.USTAR_FORMAT) as tar:
        for s in SUBJECTS:
            root = a.out / s / "render_inputs"
            for arm in ("legacy", "learned"):
                d = root / arm
                paths = [
                    d / "design.json",
                    d / "radius.json",
                    *sorted((d / "conditions").rglob("*.npz")),
                    *sorted((d / "evaluations").rglob("*.json")),
                ]
                for path in paths:
                    payload = path.read_bytes()
                    ent = tarfile.TarInfo("inputs/" + s + "/" + path.relative_to(root).as_posix())
                    ent.size = len(payload)
                    tar.addfile(ent, io.BytesIO(payload))
    print(out, sha(out), report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/fresh-people"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    package(p.parse_args())
