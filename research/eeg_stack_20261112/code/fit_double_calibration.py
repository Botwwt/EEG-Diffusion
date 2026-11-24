"""Compare equal-architecture 512/1024-class posterior EEG decoders and package an eligible fifth panel."""

import argparse
import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fit_fresh_photo_panel import physical
from fit_personal_session_metric import load
from fit_support_covariance import eeg_features
from fit_within_photo_embedding import H14_SHA, images
from package_full_head_512 import directions
from prepare_double_calibration import PANELS, PUBLIC, RESULTS, SUBJECTS, add_paths, regpath
from prepare_expanded_personal import check, extract, predict
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import packet
from run_prospective_readout import frozen_npz, visual_model

ARMS = ("posterior_512", "posterior_1024")
SEED = 260538


def training(a, sub):
    lock = read(PUBLIC)
    old, x = eeg_features(a.expanded, sub)
    path = a.out / "features" / (sub + ".json")
    r = read(path)
    check(
        r["design_sha256"] == sha(PUBLIC)
        and r["subject"] == sub
        and np.array_equal(old, lock["old_objects"])
        and r["range_class_order"] == lock["extra_objects"]
        and r["old_design_sha256"] == hashlib.sha256(x.astype("<f4").tobytes()).hexdigest(),
        "Changed 512 training registry or original features",
    )
    more = []
    for concept in lock["extra_objects"]:
        with np.load(a.out / "features" / sub / f"{concept:04d}.npz", allow_pickle=False) as z:
            value = z["features"]
            check(
                str(z["source"]) == r["source"]
                and int(z["start"]) == 10 * concept
                and str(z["raw_sha"]) == r["raw_sha256"][str(concept)]
                and hashlib.sha256(value.tobytes()).hexdigest()
                == r["feature_sha256"][str(concept)],
                "Changed compact feature or range identity",
            )
            more.append(value)
    classes = np.r_[old, lock["extra_objects"]]
    check(len(set(classes)) == 1024, "Repeated added class")
    both = np.concatenate((x, np.stack(more)))[np.argsort(classes)]
    return x.reshape(-1, 1003), both.reshape(-1, 1003)


def evaluate(a, sub, panels):
    small, large = training(a, sub)
    regs = {p: read(regpath(a, p)) for p in panels}
    locations = dict(
        registered=a.inputs / "halves",
        replicate=a.inputs / "halves",
        fresh=a.fresh / "eeg",
        prospective=a.prospective / "eeg",
        fifth=a.out / "fifth/eeg",
    )
    src = {p: locations[p] / (sub + ".npz") for p in panels}
    xs, rr = [], []
    for file in dict.fromkeys(src.values()):
        rows, _, _, mean = load(file)
        check(read(file.with_suffix(".json"))["packet_sha256"] == sha(file), "Changed target EEG")
        rr.extend(rows)
        xs.append(extract(mean))
    rr = np.asarray(rr)
    q = np.concatenate(xs)
    check(
        len(set(rr)) == len(rr)
        and not np.intersect1d(
            np.asarray(read(PUBLIC)["old_objects"] + read(PUBLIC)["extra_objects"]), rr // 10
        ).size,
        "Overlapping train or test sources",
    )
    old = predict(small, q, rr, regs)
    new = predict(large, q, rr, regs)
    for p, reg in regs.items():
        if p != "fifth":
            prior = a.full / "predictions" / p / sub / "evaluation.json"
            known = read(prior)
            check(
                old[p] == known["results"]["expanded_r4"]
                and all(reg[k] == known[k] for k in ("words", "rows", "folds", "truth"))
                and known["registration_sha256"] == sha(regpath(a, p)),
                "512-class predictions failed exact replay",
            )
        freeze(
            a.out / "predictions" / p / sub / "evaluation.json",
            dict(
                subject=sub,
                panel=p,
                **{k: reg[k] for k in ("rows", "truth", "words", "folds")},
                results=dict(zip(ARMS, (old[p], new[p]), strict=True)),
                design_sha256=sha(PUBLIC),
                registration_sha256=sha(regpath(a, p)),
                target_eeg_sha256=sha(src[p]),
                training_receipt_sha256=sha(a.out / "features" / (sub + ".json")),
                previous_512_sha256=sha(prior) if p != "fifth" else None,
            ),
        )
        print(sub, p, old[p]["hits"], "->", new[p]["hits"], flush=True)


def interval(x):
    return np.quantile(x, [0.025, 0.975]).tolist()


def contrast(hits):
    # participant x object x photograph x model; sample crossed people and shared objects.
    delta = hits[:, :, :, 1].astype(float) - hits[:, :, :, 0]
    rng = np.random.default_rng(SEED)
    p, c, n = delta.shape
    people = rng.integers(p, size=(10000, p, 1))
    cats = rng.integers(c, size=(10000, 1, c))
    av = delta.mean(2)
    by_sub = {}
    for i, sub in enumerate(SUBJECTS):
        by_sub[sub] = dict(
            hits=hits[i].sum((0, 1)).tolist(),
            n=c * n,
            delta=float(av[i].mean()),
            object_ci95=interval(av[i, cats[:, 0]].mean(1)),
        )
    return dict(
        hits=hits.sum((0, 1, 2)).tolist(),
        n=p * c * n,
        delta=float(delta.mean()),
        people_and_object_ci95=interval(av[people, cats].mean((1, 2))),
        object_only_ci95=interval(av.mean(0)[cats[:, 0]].mean(1)),
        gain=int(((hits[:, :, :, 0] == 0) & (hits[:, :, :, 1] == 1)).sum()),
        loss=int(((hits[:, :, :, 0] == 1) & (hits[:, :, :, 1] == 0)).sum()),
        by_subject=by_sub,
    )


def summarize(a, panels, public):
    all_h = []
    panel_results = {}
    # One object identifier per class and strict original-photo order inside each class.
    individuals = {}
    receipts = {}
    for p in panels:
        reg = read(regpath(a, p))
        rows = np.asarray(reg["rows"])
        truth = np.asarray(reg["truth"])
        order = np.stack(
            [np.flatnonzero(truth == i)[np.argsort(rows[truth == i])] for i in range(12)]
        )
        hits = []
        for sub in SUBJECTS:
            path = a.out / "predictions" / p / sub / "evaluation.json"
            r = read(path)
            check(
                r["subject"] == sub
                and r["design_sha256"] == sha(PUBLIC)
                and r["registration_sha256"] == sha(regpath(a, p))
                and r["truth"] == reg["truth"]
                and r["rows"] == reg["rows"],
                "Unregistered result",
            )
            correct = np.stack(
                [np.asarray(r["results"][k]["predicted"]) == truth for k in ARMS], axis=-1
            )
            check(
                np.array_equal(correct.sum(0), [r["results"][k]["hits"] for k in ARMS]),
                "Word count discrepancy",
            )
            hits.append(correct[order])
            receipts[f"{sub}/{p}"] = sha(path)
        h = np.asarray(hits)
        panel_results[p] = contrast(h)
        rng = np.random.default_rng(SEED)
        people = rng.integers(len(SUBJECTS), size=(10000, len(SUBJECTS), 1))
        photos = rng.integers(10, size=(10000, 1, 10))
        for i, word in enumerate(reg["words"]):
            check(word not in individuals, "Overlapping target panels")
            delta = h[:, i, :, 1].astype(float) - h[:, i, :, 0]
            individuals[word] = dict(
                panel=p,
                by_person_hits=h[:, i].sum(1).tolist(),
                delta=float(delta.mean()),
                people_and_photo_ci95=interval(delta[people, photos].mean((1, 2))),
            )
        all_h.append(h)
    pooled = contrast(np.concatenate(all_h, axis=1))
    sources = {}
    for sub in SUBJECTS:
        receipt = read(a.out / "features" / (sub + ".json"))
        classes = receipt["range_class_order"]
        sources[sub] = {
            k: receipt[k]
            for k in ("source", "revision", "source_full_lfs_sha256", "old_design_sha256")
        }
        for key in ("raw_sha256", "feature_sha256"):
            digest_list = b"".join(bytes.fromhex(receipt[key][str(c)]) for c in classes)
            sources[sub][key + "_ordered_digest"] = hashlib.sha256(digest_list).hexdigest()
    advance = (
        pooled["people_and_object_ci95"][0] > 0
        and sum(x["delta"] > 0 for x in pooled["by_subject"].values()) >= 2
    )
    freeze(
        public,
        dict(
            design_sha256=sha(PUBLIC),
            subjects=list(SUBJECTS),
            arms=list(ARMS),
            panels=panel_results,
            pooled=pooled,
            class_results=individuals,
            predictions_sha256=receipts,
            acquisition_receipts_sha256={
                s: sha(a.out / "features" / (s + ".json")) for s in SUBJECTS
            },
            calibration_provenance=sources,
            interval_units="Paired participants and shared true objects. Per-person intervals resample objects; within one object resample people and its ten shared photos. 10,000 draws; rates are fractions.",
            advance_to_fifth=bool(advance) if len(panels) == 4 else None,
        ),
    )
    print(pooled["hits"], pooled["people_and_object_ci95"], "development gate", advance, flush=True)


def package(a):
    work = a.out / "fifth"
    reg_file = regpath(a, "fifth")
    reg = read(reg_file)
    original = a.prospective / "render_inputs/expanded_r4"
    native = read(original / "design.json")
    require_sha = read(RESULTS / "prospective_render_inputs.json")["render_designs_sha256"][
        "expanded_r4"
    ]
    check(sha(original / "design.json") == require_sha, "Historical visual recipe changed")
    keys = [
        np.asarray(reg[k]) for k in ("rows", "folds", "words", "calibration_rows_by_fold_and_word")
    ]
    hashes = {m: {} for m in ARMS}
    bank = images(a.source)
    for sub in SUBJECTS:
        eeg = work / "eeg" / (sub + ".npz")
        prediction = a.out / "predictions/fifth" / sub / "evaluation.json"
        res = read(prediction)
        check(
            res["registration_sha256"] == sha(reg_file) and res["target_eeg_sha256"] == sha(eeg),
            "Fourth/fifth source mismatch",
        )
        # Reuse exactly the 512-object historical H14 fit for both object word predictions.
        classes, model = visual_model(a, sub, reg, eeg, bank)
        for method in ARMS:
            dest = work / "render_inputs" / method
            preds = res["results"][method]["predicted"]
            word_file = dest / "evaluations" / sub / "fifth/word_predictions.json"
            freeze(
                word_file,
                dict(
                    subject=sub,
                    rows=reg["rows"],
                    folds=reg["folds"],
                    words=reg["words"],
                    predicted=preds,
                    method=method,
                    source_sha256=sha(prediction),
                ),
            )
            data = directions(*keys, np.asarray(preds), *model)
            out = dest / "conditions" / sub / "fifth/conditions_512.npz"
            frozen_npz(
                out,
                data
                | dict(
                    training_classes=classes,
                    visual_sha256=H14_SHA,
                    registration_sha256=sha(reg_file),
                    target_eeg_sha256=sha(eeg),
                    word_prediction_sha256=sha(word_file),
                    classifier_design_sha256=sha(PUBLIC),
                    visual_calibration_count=512,
                ),
            )
            hashes[method][f"{sub}/fifth"] = sha(out)
    for method in ARMS:
        dest = work / "render_inputs" / method
        freeze(
            dest / "design.json",
            native
            | dict(
                panels=["fifth"],
                packets_sha256=hashes[method],
                public_lock_sha256=sha(PUBLIC),
                protocol=f"Fifth panel, {method} word decoder and the fixed 512-object visual decoder, matched own/uniform/wrong with historical row seeds.",
            ),
        )
        rad = dest / "radius.json"
        if not rad.is_file():
            shutil.copyfile(original / "radius.json", rad)
        check(sha(rad) == sha(original / "radius.json"), "Radius changed")
        for sub in SUBJECTS:
            packet(dest, sub, "fifth")
    identical = {}
    for sub in SUBJECTS:
        left, right = (packet(work / "render_inputs" / m, sub, "fifth") for m in ARMS)
        same = left[2] == right[2]
        check(
            np.array_equal(left[0], right[0])
            and np.array_equal(left[1], right[1])
            and np.array_equal(left[4][same], right[4][same])
            and all(
                np.array_equal(left[5][arm][same], right[5][arm][same])
                for arm in ("own", "uniform")
            ),
            "Unchanged words must reproduce identical own and uniform directions",
        )
        identical[sub] = int(same.sum())
    freeze(
        RESULTS / "double_calibration_fifth_inputs.json",
        dict(
            design_sha256=sha(PUBLIC),
            packets=hashes,
            same_word_rows_with_bit_exact_own_and_uniform=identical,
            render_designs_sha256={
                m: sha(work / "render_inputs" / m / "design.json") for m in ARMS
            },
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("evaluate", "summary", "fifth"))
    add_paths(p)
    p.add_argument("--subject", choices=SUBJECTS, default="sub-04")
    p.add_argument("--full", type=Path, default=Path("outputs/full-head-512"))
    p.add_argument("--within", type=Path, default=Path("outputs/photo-structure/within_photo"))
    p.add_argument("--angle", type=Path, default=Path("outputs/photo-structure/residual_angle"))
    a = p.parse_args()
    check(sha(a.out / "design.json") == sha(PUBLIC), "Changed locked study")
    release = RESULTS / "double_calibration.json"
    if a.stage == "evaluate":
        evaluate(a, a.subject, PANELS)
    elif a.stage == "summary":
        summarize(a, PANELS, release)
    else:
        check(
            read(release)["advance_to_fifth"] is True
            and read(release)["design_sha256"] == sha(PUBLIC),
            "Four-panel development gate did not qualify",
        )
        reg = regpath(a, "fifth")
        check(sha(reg) == read(PUBLIC)["fifth_registration_sha256"], "Changed fifth panel")
        freeze(
            a.out / "fifth/panel_lock.json",
            dict(
                registration_sha256=sha(reg),
                participants=list(SUBJECTS),
                words=read(reg)["words"],
                source_design_sha256=sha(PUBLIC),
            ),
        )
        for sub in SUBJECTS:
            physical(
                SimpleNamespace(**(vars(a) | dict(out=a.out / "fifth", workers=5))),
                sub,
                np.asarray(read(reg)["rows"]),
                False,
                reg,
            )
            evaluate(a, sub, ("fifth",))
        summarize(a, ("fifth",), RESULTS / "double_calibration_fifth.json")
        package(a)
