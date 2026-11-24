"""Register by default; replay, acquire, classify or package a released prospective experiment."""

import argparse
import hashlib
import io
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fit_fresh_photo_panel import physical, require
from fit_full_head_512 import evaluate
from fit_personal_session_metric import load
from fit_within_photo_embedding import H14_SHA, bases, images, predict, training
from package_full_head_512 import directions
from prepare_expanded_personal import extract
from register_fresh_photo_panel import SUBJECTS, freeze, read, sha
from register_prospective_photo_panel import RESULTS, register
from render_within_photo_ip import packet

PANEL = "prospective"
READOUTS = ("expanded_r4", "full_head_512")
PUBLIC_LOCK = RESULTS / "prospective_panel_design.json"
FULL_DESIGN = RESULTS / "full_head_512_design.json"


def validate_lock(a):
    lock = read(PUBLIC_LOCK)
    reg = a.out / "references/prospective_ablation.json"
    require(
        sha(a.out / "panel_lock.json") == sha(PUBLIC_LOCK)
        and sha(reg) == lock["registration_sha256"]
        and lock["participants"] == list(SUBJECTS)
        and hashlib.sha256(FULL_DESIGN.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        == lock["inputs_sha256"]["full_head_design_lf"],
        "Changed prospective registration or readout design",
    )
    return reg


def promotion(a):
    counts, hashes = {}, {}
    for sub in SUBJECTS:
        net = 0
        for panel in ("registered", "replicate", "fresh"):
            file = a.full / "predictions" / panel / sub / "evaluation.json"
            r = read(file)
            original = (a.fresh / "references" if panel == "fresh" else a.references) / (
                panel + "_ablation.json"
            )
            truth = np.asarray(read(original)["truth"])
            require(
                r["design_sha256"] == sha(FULL_DESIGN)
                and r["registration_sha256"] == sha(original)
                and r["subject"] == sub
                and r["panel"] == panel
                and len(truth) == 120,
                "Wrong developmental comparison inputs",
            )
            hits = []
            for name in READOUTS:
                chosen = np.asarray(r["results"][name]["predicted"])
                require(
                    chosen.shape == (120,)
                    and chosen.dtype.kind in "iu"
                    and np.isin(chosen, np.arange(12)).all(),
                    "Incomplete developmental word predictions",
                )
                hits.append(int(np.sum(chosen == truth)))
            net += hits[1] - hits[0]
            hashes[sub + "/" + panel] = sha(file)
        counts[sub] = net
    require(
        sum(counts.values()) >= 11 and sum(v > 0 for v in counts.values()) >= 2,
        f"Full-head candidate failed locked promotion criterion: {counts}",
    )
    freeze(
        a.out / "development_gate.json",
        dict(
            delta_hits=counts,
            n=1080,
            passed=True,
            public_lock_sha256=sha(PUBLIC_LOCK),
            inputs_sha256=hashes,
        ),
    )


def visual_configuration(a):
    previous = read(RESULTS / "fresh_panel_design.json")["inputs_sha256"]
    configs = {
        "generic_residual_selection": a.within / "selection.json",
        "angle_scale": a.angle / "scale.json",
        "radius": a.angle / "radius.json",
        "old_render_design": a.angle / "design.json",
    }
    require(
        all(sha(path) == previous[key] for key, path in configs.items()),
        "Historical H/14 model inputs changed",
    )
    return read(configs["angle_scale"]), read(configs["generic_residual_selection"])["selected"]


def visual_model(a, sub, reg, source, bank):
    classes, eeg, visual = training(a, sub, 512, bank)
    rows, _, _, mean = load(source)
    require(
        set(rows.tolist()) == set(reg["rows"]) and not np.intersect1d(classes, rows // 10).size,
        "Target packet or generic classes do not match the registered split",
    )
    scale, selected = visual_configuration(a)
    basis = bases(eeg, visual, extract(mean).astype(np.float64))
    fitted = predict(basis, selected) @ basis[2].T
    return classes, (fitted, rows, bank, scale)


def conditions(reg, index, model):
    args = [
        np.asarray(reg[k]) for k in ("rows", "folds", "words", "calibration_rows_by_fold_and_word")
    ]
    ix = np.asarray(index)
    require(
        ix.shape == (120,) and ix.dtype.kind in "iu" and np.isin(ix, np.arange(12)).all(),
        "Invalid frozen word indices",
    )
    return directions(*args, ix, *model)


def replay(a):
    reg = read(a.fresh / "references/fresh_ablation.json")
    bank = images(a.source)
    receipts = {}
    for sub in SUBJECTS:
        source = a.fresh / "eeg" / (sub + ".npz")
        classes, model = visual_model(a, sub, reg, source, bank)
        word_file = a.fresh / "evaluations" / sub / "fresh/word_predictions.json"
        known = a.fresh / "conditions" / sub / "fresh/conditions_512.npz"
        data = conditions(reg, read(word_file)["predicted"], model)
        with np.load(known, allow_pickle=False) as z:
            require(
                str(z["word_prediction_sha256"]) == sha(word_file)
                and str(z["target_eeg_sha256"]) == sha(source)
                and np.array_equal(classes, z["training_classes"])
                and all(np.array_equal(z[k], v) for k, v in data.items()),
                "Old fresh posterior H/14 replay differs on a word, support, donor or float32 value",
            )
        receipts[sub] = dict(
            legacy_packet_sha256=sha(known),
            words_sha256=sha(word_file),
            queries=120,
            vectors_bit_exact=["own", "uniform", "wrong"],
            all_discrete_fields_equal=True,
        )
    freeze(
        a.replay_file,
        dict(
            source_panel="fresh",
            source_registration_sha256=sha(a.fresh / "references/fresh_ablation.json"),
            subjects=receipts,
        ),
    )


def frozen_npz(path, data):
    buf = io.BytesIO()
    np.savez(buf, **data)
    payload = buf.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        require(path.read_bytes() == payload, "Existing condition packet differs")
    else:
        temp = path.with_suffix(".partial")
        temp.write_bytes(payload)
        temp.replace(path)


def package(a, regpath):
    reg = read(regpath)
    bank = images(a.source)
    receipts = {name: {} for name in READOUTS}
    for sub in SUBJECTS:
        source = a.out / "eeg" / (sub + ".npz")
        pred_file = a.full / "predictions" / PANEL / sub / "evaluation.json"
        result = read(pred_file)
        raw = read(source.with_suffix(".json"))
        require(
            result["subject"] == sub
            and result["panel"] == PANEL
            and result["registration_sha256"] == raw["registration_sha256"] == sha(regpath)
            and result["design_sha256"] == sha(FULL_DESIGN)
            and result["target_eeg_sha256"] == raw["packet_sha256"] == sha(source)
            and all(reg[k] == result[k] for k in ("rows", "folds", "words")),
            "Frozen word registry differs from EEG",
        )
        classes, model = visual_model(a, sub, reg, source, bank)
        for name in READOUTS:
            root = a.out / "render_inputs" / name
            indices = result["results"][name]["predicted"]
            word_file = root / "evaluations" / sub / PANEL / "word_predictions.json"
            freeze(
                word_file,
                dict(
                    subject=sub,
                    rows=reg["rows"],
                    folds=reg["folds"],
                    words=reg["words"],
                    predicted=indices,
                    readout=name,
                    prediction_registry_sha256=sha(pred_file),
                    target_sha256=sha(source),
                ),
            )
            data = conditions(reg, indices, model)
            dest = root / "conditions" / sub / PANEL / "conditions_512.npz"
            frozen_npz(
                dest,
                data
                | dict(
                    training_classes=classes,
                    visual_sha256=H14_SHA,
                    target_eeg_sha256=sha(source),
                    registration_sha256=sha(regpath),
                    word_prediction_sha256=sha(word_file),
                    angular_scale_sha256=sha(a.angle / "scale.json"),
                    selection_sha256=sha(a.within / "selection.json"),
                    public_lock_sha256=sha(PUBLIC_LOCK),
                    full_head_design_sha256=sha(FULL_DESIGN),
                ),
            )
            receipts[name][f"{sub}/{PANEL}"] = sha(dest)
    original = read(a.angle / "design.json")
    for name in READOUTS:
        root = a.out / "render_inputs" / name
        keep = (
            "visual",
            "generic_radius_rows",
            "radius_seed",
            "space_check",
            "model",
            "arms",
            "endpoints",
            "adapter_sha256",
            "visual_sha256",
            "angular_calibration",
        )
        design = {k: original[k] for k in keep}
        freeze(
            root / "design.json",
            design
            | dict(
                cohort=list(SUBJECTS),
                panels=[PANEL],
                packets_sha256=receipts[name],
                public_lock_sha256=sha(PUBLIC_LOCK),
                protocol=f"Frozen {name} words, same direct H/14 four-step generator, own/uniform/wrong, row seed 923761 + row.",
            ),
        )
        dest = root / "radius.json"
        if not dest.exists():
            shutil.copyfile(a.angle / "radius.json", dest)
        require(sha(dest) == sha(a.angle / "radius.json"), "Incorrect historical image radius")
        for sub in SUBJECTS:
            packet(root, sub, PANEL)
    freeze(
        a.out / "render_packets.json",
        dict(
            packets=receipts,
            public_lock_sha256=sha(PUBLIC_LOCK),
            development_gate_sha256=sha(a.out / "development_gate.json"),
            historical_replay_sha256=sha(a.replay_file),
        ),
    )


def release_inputs(a, reg):
    """Publish aggregate word results and the input hashes required to reproduce both draws."""
    registration = read(reg)
    expected = np.asarray(registration["truth"])
    packets = read(a.out / "render_packets.json")
    cases = {}
    for sub in SUBJECTS:
        pred = a.full / "predictions" / PANEL / sub / "evaluation.json"
        result = read(pred)
        require(result["truth"] == registration["truth"], "Registration class order changed")
        records = {}
        for name in READOUTS:
            root = a.out / "render_inputs" / name
            words = root / "evaluations" / sub / PANEL / "word_predictions.json"
            selected = np.asarray(read(words)["predicted"])
            hits = int(np.sum(selected == expected))
            require(
                hits == result["results"][name]["hits"]
                and np.array_equal(selected, result["results"][name]["predicted"]),
                "Published words differ from independently scored classifications",
            )
            records[name] = dict(
                hits=hits,
                n=len(selected),
                words_sha256=sha(words),
                conditions_sha256=packets["packets"][name][f"{sub}/{PANEL}"],
            )
        cases[sub] = dict(
            eeg_sha256=result["target_eeg_sha256"],
            physical_receipt_sha256=result["raw_range_receipt_sha256"],
            readout_receipt_sha256=sha(pred),
            methods=records,
            paired_full_minus_posterior=result["paired_versus_512_posterior"],
        )
    freeze(
        a.release_file,
        dict(
            panel=PANEL,
            registration_sha256=sha(reg),
            lock_sha256=sha(PUBLIC_LOCK),
            development_gate=read(a.out / "development_gate.json"),
            historical_condition_replay=read(a.replay_file),
            packets_manifest_sha256=sha(a.out / "render_packets.json"),
            render_designs_sha256={
                n: sha(a.out / "render_inputs" / n / "design.json") for n in READOUTS
            },
            radius_sha256=sha(a.angle / "radius.json"),
            seed_per_query="923761 + original zero-based stimulus row",
            n_distinct_stimuli=120,
            n_query_decisions=360,
            cases=cases,
            pooled_word_hits={
                n: sum(c["methods"][n]["hits"] for c in cases.values()) for n in READOUTS
            },
        ),
    )


def run(a):
    if a.stage == "register":
        register(
            SimpleNamespace(
                out=a.out,
                public=PUBLIC_LOCK,
                prior=RESULTS / "fresh_panel_design.json",
                frame=a.fresh / "registration/exclusion_sets.json",
                manifest=a.source / "atm_train_rows.csv",
                full_head_design=FULL_DESIGN,
            )
        )
        return
    if a.stage == "replay":
        replay(a)
        return
    reg = validate_lock(a)
    promotion(a)
    evidence = read(a.replay_file)
    require(
        evidence["source_registration_sha256"] == sha(a.fresh / "references/fresh_ablation.json")
        and all(
            evidence["subjects"][s]["legacy_packet_sha256"]
            == sha(a.fresh / "conditions" / s / "fresh/conditions_512.npz")
            for s in SUBJECTS
        ),
        "Replay the earlier fresh posterior conditions first",
    )
    if a.stage == "fetch":
        rows = np.asarray(read(reg)["rows"])
        for sub in SUBJECTS:
            physical(a, sub, rows, False, registration=reg)
    elif a.stage == "classify":
        for sub in SUBJECTS:
            evaluate(
                SimpleNamespace(
                    **(
                        vars(a)
                        | dict(out=a.full, confirmation_root=a.out, confirmation_panel=PANEL)
                    )
                ),
                sub,
            )
    else:
        package(a, reg)
        release_inputs(a, reg)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        nargs="?",
        default="register",
        choices=("register", "replay", "fetch", "classify", "conditions"),
    )
    for k, v in dict(
        out="outputs/prospective-photo",
        full="outputs/full-head-512",
        fresh="outputs/fresh-photo",
        within="outputs/photo-structure/within_photo",
        expanded="outputs/photo-structure/expanded",
        angle="outputs/photo-structure/residual_angle",
        inputs="outputs/object-transfer",
        references="outputs/object-fidelity",
        source="outputs/official-object-readout/sub-03",
        replay_file="outputs/fresh-photo/prospective_condition_replay.json",
        release_file=str(RESULTS / "prospective_render_inputs.json"),
    ).items():
        p.add_argument("--" + k.replace("_", "-"), type=Path, default=Path(v))
    p.add_argument("--workers", type=int, choices=range(1, 7), default=5)
    run(p.parse_args())
