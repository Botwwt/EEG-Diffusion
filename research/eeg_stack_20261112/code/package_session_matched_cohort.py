"""Prepare matched-backend single-person and cohort images on both session-controlled panels."""

import argparse
import shutil
from pathlib import Path

import numpy as np
from fit_shared_photo_eeg import read, require, sha, unit, write
from fit_support_structure import H14_SHA
from fit_within_photo_embedding import bases, images, predict, training
from fuse_same_stimulus_r4 import PEOPLE, geometry, parts
from package_full_head_512 import directions
from render_within_photo_ip import packet
from test_cohort_session_matched import NAMES, compact


def combine(per):
    template = per[0]
    for x in per[1:]:
        require(
            all(
                np.array_equal(x[k], template[k])
                for k in (
                    "rows",
                    "folds",
                    "support_rows",
                    "donor_rows",
                    "uniform",
                    "predicted_words",
                )
            ),
            "Cohort supports or words differ",
        )
    t, angle = zip(*(geometry(x["uniform"], x["own"]) for x in per), strict=True)
    theta = np.mean(angle, axis=0)
    v = unit(
        np.cos(theta) * unit(template["uniform"].astype(float))
        + np.sin(theta) * unit(np.sum(t, axis=0))
    ).astype(np.float32)
    require(
        float(np.abs(geometry(template["uniform"], v)[1] - theta).max()) < 1e-6,
        "Incorrect cohort angle",
    )
    pos = {r: i for i, r in enumerate(template["rows"])}
    return template | dict(own=v, wrong=v[[pos[r] for r in template["donor_rows"]]])


def run(a):
    design = read(a.out / "design.json")
    bank = images(a.source)
    refout = a.out / "references"
    dest = a.out / "render_inputs"
    cfg = read(a.within / "selection.json")
    scale = read(a.angle / "scale.json")
    require(
        cfg["selected"] == [2.0, 0.0]
        and scale["selection_sha256"] == sha(a.within / "selection.json"),
        "Changed native H14 settings",
    )
    fitted = {}
    rowsets = {}
    for s in PEOPLE:
        classes, x, y = training(a, s, 512, bank)
        xq = []
        for name in NAMES:
            concepts = np.asarray(design["panels"][name]["selected_concepts"])
            require(not np.intersect1d(classes, concepts).size, "Test class in training EEG or H14")
            feats, sig = compact(a, s, name, concepts)
            require(
                read(a.out / name / "result.json")["target_feature_receipts"][s] == sig,
                "Target EEG features changed",
            )
            rows = (concepts[:, None] * 10 + np.arange(10)).ravel()
            order = np.argsort(rows)
            rowsets[name] = rows[order]
            xq.append(feats[order])
        b = bases(x, y, np.concatenate(xq))
        proj = predict(b, cfg["selected"]) @ b[2].T
        fitted[s] = dict(zip(NAMES, np.split(proj, len(NAMES)), strict=True))
    signatures = {}
    references = {}
    checks = {}
    for name in NAMES:
        panel = design["panels"][name]
        reg = panel["registration"]
        rfile = refout / (name + "_ablation.json")
        write(rfile, reg)
        references[name] = sha(rfile)
        src = a.out / name / "result.json"
        metrics = read(src)
        primary = metrics["primary"]
        pf = a.out / name / "predictions.npz"
        require(sha(pf) == metrics["predictions_sha256"], "Different locked word decisions")
        with np.load(pf, allow_pickle=False) as z:
            require(
                np.array_equal(z["rows"], reg["rows"]) and np.array_equal(z["folds"], reg["folds"]),
                "Misaligned words",
            )
            choices = {
                tag: z[primary + "_" + ("cohort" if tag == "cohort-3" else tag)]
                for tag in ("cohort-3", "sub-04")
            }
        checks[name] = dict(
            n=120,
            matched_predicted_words=int(np.sum(choices["sub-04"] == choices["cohort-3"])),
            candidates_per_primary_word=panel["n_words_per_primary_decision"],
            source_result_sha256=sha(src),
        )
        rows, folds, vocab, cal = parts(reg)
        sets = {}
        for who, idx in choices.items():
            contributors = list(PEOPLE) if who == "cohort-3" else ["sub-04"]
            obs = dest / "observations" / who / (name + ".json")
            write(
                obs,
                dict(
                    subject=who,
                    panel=name,
                    contributing_subjects=contributors,
                    physical_trials_per_photo=4 * len(contributors),
                    source_feature_receipts={
                        s: metrics["target_feature_receipts"][s] for s in contributors
                    },
                    same_original_photo_rows=rows.tolist(),
                ),
            )
            wordfile = dest / "evaluations" / who / name / "word_predictions.json"
            write(
                wordfile,
                dict(
                    subject=who,
                    contributing_subjects=contributors,
                    physical_trials_per_photo=4 * len(contributors),
                    rows=rows.tolist(),
                    folds=folds.tolist(),
                    words=vocab.tolist(),
                    predicted=idx.tolist(),
                    candidate_group_size=panel["n_words_per_primary_decision"],
                    registration_sha256=sha(rfile),
                    original_word_choices_sha256=sha(pf),
                ),
            )
            per = [
                directions(
                    rows, folds, vocab, cal, idx, fitted[s][name], rowsets[name], bank, scale
                )
                for s in contributors
            ]
            dirs = per[0] if len(per) == 1 else combine(per)
            sets[who] = dirs
            packetfile = dest / "conditions" / who / name / "conditions_512.npz"
            packetfile.parent.mkdir(exist_ok=True, parents=True)
            np.savez(
                packetfile,
                **dirs,
                subject=who,
                contributing_subjects=np.array(contributors),
                physical_trials_per_photo=4 * len(contributors),
                training_classes=classes,
                visual_sha256=H14_SHA,
                target_eeg_sha256=sha(obs),
                word_prediction_sha256=sha(wordfile),
                registration_sha256=sha(rfile),
                selection_sha256=sha(a.within / "selection.json"),
                angular_scale_sha256=sha(a.angle / "scale.json"),
                session_matched_design_sha256=sha(a.out / "design.json"),
            )
            signatures[who + "/" + name] = sha(packetfile)
            write(
                refout / (name + "_" + who + "_lock.json"),
                dict(
                    participants=[who],
                    contributing_subjects=contributors,
                    physical_trials_per_photo=4 * len(contributors),
                    words=vocab.tolist(),
                    registration_sha256=sha(rfile),
                    candidate_group_size=panel["n_words_per_primary_decision"],
                    study_design_sha256=sha(a.out / "design.json"),
                ),
            )
        hit = choices["sub-04"] == choices["cohort-3"]
        for k in ("uniform", "support_rows"):
            require(
                np.array_equal(sets["sub-04"][k][hit], sets["cohort-3"][k][hit]),
                "Equal words should reproduce identical reference supports",
            )
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
    write(
        dest / "design.json",
        config
        | dict(
            panels=list(NAMES),
            cohort=["sub-04", "cohort-3"],
            contributing_subjects=list(PEOPLE),
            physical_trials_per_photo=12,
            acquisition_by_output={
                "cohort-3": {
                    "contributing_subjects": list(PEOPLE),
                    "physical_trials_per_photo": 12,
                },
                "sub-04": {"contributing_subjects": ["sub-04"], "physical_trials_per_photo": 4},
            },
            packets_sha256=signatures,
            registration_sha256=references,
            session_matched_design_sha256=sha(a.out / "design.json"),
            comparisons=checks,
            protocol="One natural end-to-end word and continuous condition per acquisition, same eight supports when predicted words coincide, identical row-specific seed 923761+row across both acquisitions/three arms. stress12 is one supplied set of 12 same-session alternatives; new_three4 supplies the registered group of four alternatives for each query.",
            arms={
                "own": "Individual native direction/angle or equal-weight cohort tangent and mean native angle",
                "uniform": "Predicted word's fixed eight-support H14 direction",
                "wrong": "Whole donor condition, including its angle, from the same predicted word and fold, singleton unchanged",
            },
        ),
    )
    shutil.copyfile(a.angle / "radius.json", dest / "radius.json")
    for who in ("sub-04", "cohort-3"):
        for name in NAMES:
            packet(dest, who, name)
    write(
        a.out / "render_packets.json",
        dict(
            design_sha256=sha(dest / "design.json"),
            radius_sha256=sha(dest / "radius.json"),
            reference_sha256=references,
            packets_sha256=signatures,
            comparisons=checks,
        ),
    )
    print("Verified four conditions; input design", sha(dest / "design.json"), checks, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in dict(
        out="outputs/session-matched-cohort",
        double="outputs/double-calibration",
        expanded="outputs/photo-structure/expanded",
        source="outputs/official-object-readout/sub-03",
        angle="outputs/photo-structure/residual_angle",
        within="outputs/photo-structure/within_photo",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    run(p.parse_args())
