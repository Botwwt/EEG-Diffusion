"""Export fresh H/14 angle packets for the full-head word readout, verifying baseline parity."""

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
from calibrate_photo_angles import rotate
from fit_personal_session_metric import load
from fit_within_photo_embedding import H14_SHA, bases, generative, images, predict, training, unit
from prepare_expanded_personal import extract
from prepare_full_head_512 import PANELS, PUBLIC, SUBJECTS, check, defaults
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import packet


def directions(rows, folds, vocab, cal, indices, fitted, source_rows, bank, scale):
    n = len(rows)
    prior = np.empty((n, 1024), np.float32)
    native = np.empty_like(prior)
    support = cal[folds, indices]
    for f in range(5):
        ix = np.flatnonzero(folds == f)
        allowed = support[ix]
        check(
            len(ix) == 24
            and set(cal[f].ravel()) == set(rows[folds != f])
            and not np.intersect1d(allowed, rows[ix]).size,
            "Query photograph in active fold's support",
        )
        q = np.searchsorted(source_rows, rows[ix])
        k = np.searchsorted(source_rows, allowed)
        check(
            np.array_equal(source_rows[q], rows[ix]) and np.array_equal(source_rows[k], allowed),
            "EEG source row mismatch",
        )
        mean = unit(bank[allowed].astype(float)).mean(1)
        prior[ix] = unit(mean)
        native[ix] = generative(mean, fitted[q] - fitted[k].mean(1))
    words = vocab[indices]
    donor = np.arange(n)
    for fold in range(5):
        for word in np.unique(words[folds == fold]):
            group = np.flatnonzero((words == word) & (folds == fold))
            donor[group] = np.roll(group, 1)
    own = rotate(prior, native, scale)
    return dict(
        rows=rows,
        folds=folds,
        support_rows=support,
        predicted_words=words,
        donor_rows=rows[donor],
        own=own,
        uniform=prior,
        wrong=own[donor],
    )


def run(a):
    check(sha(PUBLIC) == sha(a.out / "design.json"), "Changed protocol")
    original = read(a.angle / "design.json")
    selected = read(a.within / "selection.json")
    scale = read(a.angle / "scale.json")
    check(
        selected["selected"] == [2.0, 0.0]
        and scale == original["angular_calibration"]
        and scale["selection_sha256"] == sha(a.within / "selection.json"),
        "Historical H/14 or angular gain changed",
    )
    bank = images(a.source)
    root = a.package_out or a.out / (
        "h14" if tuple(a.subjects) == SUBJECTS else "h14_" + "_".join(a.subjects)
    )
    root.mkdir(exist_ok=True, parents=True)
    hashes = {}
    supplied = {}
    for sub in a.subjects:
        classes, x, y = training(a, sub, 512, bank)
        for historical, panels in ((True, ("registered", "replicate")), (False, ("fresh",))):
            hg = a.angle if historical else a.fresh
            eeg = (a.inputs / "halves" if historical else a.fresh / "eeg") / (sub + ".npz")
            er, _, _, mean = load(eeg)
            b = bases(x, y, extract(mean).astype(float))
            fitted = predict(b, selected["selected"]) @ b[2].T
            for p in panels:
                file = a.out / "predictions" / p / sub / "evaluation.json"
                evals = read(file)
                refpath = (
                    a.references / (p + "_ablation.json")
                    if historical
                    else a.fresh / "references/fresh_ablation.json"
                )
                ref = read(refpath)
                baseline = hg / "conditions" / sub / p / "conditions_512.npz"
                check(
                    evals["design_sha256"] == sha(PUBLIC)
                    and sha(refpath) == evals["registration_sha256"],
                    "Different frozen classifier or support set",
                )
                ir, ifold, vocab, cal = [
                    np.asarray(ref[k])
                    for k in ("rows", "folds", "words", "calibration_rows_by_fold_and_word")
                ]
                check(
                    not np.intersect1d(classes, ir // 10).size,
                    "Target objects entered generic continuous model",
                )
                pred0 = np.array(evals["results"]["expanded_r4"]["predicted"])
                pred = np.array(evals["results"]["full_head_512"]["predicted"])
                old = directions(ir, ifold, vocab, cal, pred0, fitted, er, bank, scale)
                new = directions(ir, ifold, vocab, cal, pred, fitted, er, bank, scale)
                with np.load(baseline, allow_pickle=False) as q:
                    check(
                        all(
                            np.allclose(old[k], q[k], rtol=0, atol=2e-6)
                            if k in ("own", "wrong")
                            else np.array_equal(old[k], q[k])
                            for k in old
                        )
                        and str(q["target_eeg_sha256"]) == sha(eeg),
                        "Original directions, supports, words or donors differ from the frozen model",
                    )
                    # Last-bit BLAS differences should never change an existing same-word image condition.
                    same = pred0 == pred
                    old["own"] = q["own"]
                    new["own"][same] = old["own"][same]
                    pos = {r: i for i, r in enumerate(ir)}
                    new["wrong"] = new["own"][[pos[r] for r in new["donor_rows"]]]
                reuse = np.all(old["own"] == new["own"], axis=1) & (
                    old["predicted_words"] == new["predicted_words"]
                )
                check(
                    np.array_equal(reuse, pred0 == pred),
                    "Unexpected changed-word/changed-condition mismatch",
                )
                path = root / "conditions" / sub / p / "conditions_512.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = dict(
                    **new,
                    training_classes=classes,
                    visual_sha256=H14_SHA,
                    selected=selected["selected"],
                    selection_sha256=sha(a.within / "selection.json"),
                    angular_scale_sha256=sha(a.angle / "scale.json"),
                    target_eeg_sha256=sha(eeg),
                    registration_sha256=sha(refpath),
                    word_prediction_sha256=sha(file),
                    full_head_design_sha256=sha(PUBLIC),
                    original_condition_sha256=sha(baseline),
                )
                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        check(
                            set(saved.files) == set(payload)
                            and all(
                                np.allclose(saved[k], v, rtol=0, atol=2e-6)
                                if k in ("own", "wrong")
                                else np.array_equal(saved[k], v)
                                for k, v in payload.items()
                            )
                            and np.array_equal(saved["own"][same], old["own"][same]),
                            "Existing frozen generator inputs differ",
                        )
                else:
                    np.savez(path, **payload)
                hashes[sub + "/" + p] = sha(path)
                supplied[sub + "/" + p] = dict(
                    n=120,
                    identical_own_directions_and_words=int(reuse.sum()),
                    changed_own_ordinals=np.flatnonzero(~reuse).tolist(),
                    baseline_own_direction_sha256=hashlib.sha256(old["own"].tobytes()).hexdigest(),
                    baseline_packet_sha256=sha(baseline),
                    new_packet_sha256=sha(path),
                )
    freeze(
        root / "design.json",
        {
            k: original[k]
            for k in (
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
        }
        | dict(
            cohort=list(a.subjects),
            panels=list(PANELS),
            packets_sha256=hashes,
            full_head_design_sha256=sha(PUBLIC),
            protocol="The same H/14 angular ridge, class supports, diffusion weights, prompt, and row seed as in the matched posterior baseline, given the full-head 512-class predicted word.",
        ),
    )
    radius = root / "radius.json"
    if not radius.exists():
        shutil.copyfile(a.angle / "radius.json", radius)
    check(sha(radius) == sha(a.angle / "radius.json"), "Wrong H/14 prior radius")
    for sub in a.subjects:
        for p in PANELS:
            packet(root, sub, p)
    freeze(root / "baseline_parity.json", supplied)
    print(
        "Generator packets verified. Changed matched rows",
        {k: len(v["changed_own_ordinals"]) for k, v in supplied.items()},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    defaults(p)
    for k, d in dict(
        source="outputs/official-object-readout/sub-03",
        within="outputs/photo-structure/within_photo",
        angle="outputs/photo-structure/residual_angle",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(d))
    p.add_argument("--subjects", nargs="+", choices=SUBJECTS, default=list(SUBJECTS))
    p.add_argument(
        "--package-out", type=Path, help="Write an independent render contract to this directory"
    )
    run(p.parse_args())
