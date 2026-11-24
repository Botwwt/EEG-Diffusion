"""Pool three synchronized same-stimulus R4 measurements and export one cohort condition."""

import argparse
import shutil
from pathlib import Path

import numpy as np
from fit_fresh_photo_panel import fisher
from fit_personal_session_metric import load, similarities
from fit_shared_photo_eeg import read, require, sha, unit, write
from fit_support_structure import H14_SHA
from fit_within_photo_embedding import bases, images, predict, training
from package_full_head_512 import directions
from prepare_expanded_personal import extract

PEOPLE = ("sub-04", "sub-05", "sub-01")
PANELS = ("registered", "replicate", "fresh", "prospective")


def paths(a, p, s):
    if p in ("registered", "replicate"):
        return (
            a.references / (p + "_ablation.json"),
            a.inputs / "halves" / (s + ".npz"),
            a.angle / "conditions" / s / p / "conditions_512.npz",
        )
    root = a.fresh if p == "fresh" else a.prospective
    b = root if p == "fresh" else root / "render_inputs/expanded_r4"
    return (
        root / "references" / (p + "_ablation.json"),
        root / "eeg" / (s + ".npz"),
        b / "conditions" / s / p / "conditions_512.npz",
    )


def plan(a):
    return dict(
        protocol="Same-stimulus cohort-3, three participants with four measured presentations each, one decision per photograph. Optional higher acquisition budget. The participants and all four development panels were previously studied.",
        contributing_subjects=list(PEOPLE),
        physical_trials_per_photo=12,
        individual_trials_per_photo=4,
        panels=list(PANELS),
        object="Each participant uses their frozen 512-generic-object posterior 17-channel Fisher96 model. For each query, cosine to each object's eight-support EEG mean, subtract across the 12 scores and divide by their population SD. Sum standardized scores equally across people observing the identical original row; largest wins, first class on a tie.",
        controls="Compare paired on identical stimuli to the mean correctness of the three individual R4 predictions and to fixed sub-04. Additional negative control holds sub-04 aligned and shifts sub-05 and sub-01 score vectors by two distinct nonzero cyclic offsets within each of five folds. Row order is sorted by original stimulus ID; offsets fixed by seed 260531 before scoring. This preserves supports/fold/marginals and breaks photo alignment, sometimes retaining an object by chance.",
        visual="Use the fused predicted word and its eight shared support image/EEG IDs. Recompute each participant's original 512-generic H14 direction and generically calibrated angle for that fused word. Normalize the three tangents independently and normalize their equal-weight sum. The output angular amplitude is the arithmetic mean of those same three angles. Uniform equals their common eight-support prior. Wrong rotates the entire fused direction including its own amplitude from the preceding query within identical fold and predicted word, singleton uses own. Verify per-subject projections and individual-baseline target-word predictions.",
        seed="Previously used four-step Turbo, native IP-Adapter radius, same word and seed 923761 + global row. Cohort produces one image per query and arm; prospective panel would have 360 outputs total.",
        metrics="Each of the four panels contains 120 stimuli (12 objects). Report overall 480 stimulus-level decisions and same-row paired hit differences vs mean individual R4 and sub-04. Resample 12 objects within panel, or all 48 objects together, 10000 times seed 260532; participants fixed. Descriptive shuffled co-viewers control.",
        gate="Prepare fourth-panel cohort image experiment only if cohort has at least 5 more correct words than sub-04 on the prospective panel and strictly more than sub-04 over all four panels. This is development evidence, not separate confirmation.",
        generic_inputs={s: sha(a.expanded / "features" / (s + ".json")) for s in PEOPLE},
        visual_sha256=H14_SHA,
        selection_sha256=sha(a.within / "selection.json"),
        angle_sha256=sha(a.angle / "scale.json"),
        source_registration_sha256={p: sha(paths(a, p, PEOPLE[0])[0]) for p in PANELS},
        frozen_single_conditions_sha256={
            p: {s: sha(paths(a, p, s)[2]) for s in PEOPLE} for p in PANELS
        },
    )


def parts(reg):
    return tuple(
        np.asarray(reg[k]) for k in ("rows", "folds", "words", "calibration_rows_by_fold_and_word")
    )


def individual(a, bank, s, reg, scale):
    classes, x, y = training(a, s, 512, bank)
    files = list(dict.fromkeys(paths(a, p, s)[1] for p in PANELS))
    raw = {}
    for file in files:
        rr, _, _, mean = load(file)
        require(
            np.all(np.diff(rr) > 0) and not np.intersect1d(classes, rr // 10).size,
            "Unknown or overlapping target EEG",
        )
        raw[file] = (rr, extract(mean).astype(float))
    query = np.concatenate([raw[k][1] for k in files])
    obj = fisher(x, query)
    b = bases(x, y, query)
    visual = predict(b, (2.0, 0.0)) @ b[2].T
    offsets = np.cumsum([0, *[len(raw[k][0]) for k in files]])
    source = {
        k: (raw[k][0], obj[offsets[i] : offsets[i + 1]], visual[offsets[i] : offsets[i + 1]])
        for i, k in enumerate(files)
    }
    out = {}
    for panel in PANELS:
        _, eeg, packet = paths(a, panel, s)
        rr, proj, v = source[eeg]
        rows, folds, vocab, cal = parts(reg[panel])
        scores = np.empty((120, 12))
        for f in range(5):
            q = np.flatnonzero(folds == f)
            ids = np.searchsorted(rr, rows[q])
            sup = np.searchsorted(rr, cal[f])
            require(
                np.array_equal(rr[ids], rows[q])
                and np.array_equal(rr[sup], cal[f])
                and set(cal[f].ravel()) == set(rows[folds != f]),
                "Support mismatch",
            )
            scores[q] = similarities(proj[ids], proj[sup])
        require(
            np.max(abs(scores.mean(1))) < 1e-6 and np.max(abs(scores.std(1) - 1)) < 1e-6,
            "Nonstandardized participant scores",
        )
        idx = scores.argmax(1)
        pure = directions(rows, folds, vocab, cal, idx, v, rr, bank, scale)
        with np.load(packet, allow_pickle=False) as z:
            error = max(float(np.max(abs(pure[k] - z[k]))) for k in ("own", "uniform", "wrong"))
            require(
                np.array_equal(vocab[idx], z["predicted_words"])
                and np.array_equal(z["rows"], rows)
                and np.array_equal(z["support_rows"], pure["support_rows"])
                and sha(eeg) == str(z["target_eeg_sha256"])
                and set(z["training_classes"]) == set(classes)
                and error < 2e-5,
                "Historical participant replay failed",
            )
        out[panel] = dict(
            scores=scores,
            fitted=v,
            source_rows=rr,
            baseline_max_error=error,
            source_eeg_sha256=sha(eeg),
        )
    return classes, out


def geometry(prior, own):
    c = unit(prior.astype(float))
    v = unit(own.astype(float))
    d = (c * v).sum(1, keepdims=True)
    r = v - d * c
    return unit(r), np.arctan2(np.linalg.norm(r, axis=1, keepdims=True), d)


def export(a, reg, scale, bank, members, idx, classes, summary):
    p = "prospective"
    rows, folds, vocab, cal = parts(reg[p])
    source = a.prospective / "render_inputs/expanded_r4"
    out = a.prospective / "refinements/cohort"
    per = []
    for s in PEOPLE:
        m = members[s][p]
        per.append(
            directions(rows, folds, vocab, cal, idx, m["fitted"], m["source_rows"], bank, scale)
        )
    template = per[0]
    prior = template["uniform"]
    require(
        all(
            all(
                np.array_equal(t[k], template[k])
                for k in (
                    "uniform",
                    "rows",
                    "predicted_words",
                    "support_rows",
                    "donor_rows",
                    "folds",
                )
            )
            for t in per
        ),
        "Cohort support or ordering inconsistent",
    )
    tangents, angles = zip(*(geometry(prior, t["own"]) for t in per), strict=True)
    theta = np.mean(angles, axis=0)
    tangent = unit(np.sum(tangents, axis=0))
    own = unit(unit(prior.astype(float)) * np.cos(theta) + tangent * np.sin(theta)).astype(
        np.float32
    )
    require(
        np.max(abs(geometry(prior, own)[1] - theta)) < 1e-6,
        "Mean cohort angular magnitude not preserved",
    )
    lookup = {r: i for i, r in enumerate(rows)}
    di = np.array([lookup[r] for r in template["donor_rows"]])
    observations = out / "observations.json"
    write(
        observations,
        dict(
            subject="cohort-3",
            physical_trials_per_photo=12,
            contributing_subjects=list(PEOPLE),
            individual_r4_sha256={s: members[s][p]["source_eeg_sha256"] for s in PEOPLE},
            same_original_row_order=rows.tolist(),
        ),
    )
    words_file = out / "evaluations/cohort-3/prospective/word_predictions.json"
    write(
        words_file,
        dict(
            subject="cohort-3",
            contributing_subjects=list(PEOPLE),
            physical_trials_per_photo=12,
            rows=rows.tolist(),
            folds=folds.tolist(),
            words=vocab.tolist(),
            predicted=idx.tolist(),
            cohort_lock_sha256=sha(a.out / "design.json"),
            target_sha256=sha(observations),
        ),
    )
    packet_file = out / "conditions/cohort-3/prospective/conditions_512.npz"
    packet_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        packet_file,
        **(
            template
            | dict(
                own=own,
                wrong=own[di],
                training_classes=classes,
                visual_sha256=H14_SHA,
                contributing_subjects=np.array(PEOPLE),
                physical_trials_per_photo=12,
                subject="cohort-3",
                registration_sha256=sha(paths(a, p, PEOPLE[0])[0]),
                word_prediction_sha256=sha(words_file),
                target_eeg_sha256=sha(observations),
                selection_sha256=sha(a.within / "selection.json"),
                angular_scale_sha256=sha(a.angle / "scale.json"),
                cohort_lock_sha256=sha(a.out / "design.json"),
            )
        ),
    )
    baseline = read(source / "design.json")
    signature = sha(packet_file)
    write(
        out / "design.json",
        baseline
        | dict(
            cohort=["cohort-3"],
            contributing_subjects=list(PEOPLE),
            physical_trials_per_photo=12,
            individual_trials_per_photo=4,
            packets_sha256={"cohort-3/prospective": signature},
            cohort_lock_sha256=sha(a.out / "design.json"),
            protocol="Cohort-3: three people's synchronized four-repeat EEG, twelve physical presentations of the same target photo. One cohort output per arm; same word, eight supports and row seed.",
            arms=dict(
                own="Equal-weight pooled tangent and mean of three native angles",
                uniform="Common eight-support H14 prior",
                wrong="Same-word/same-fold exchanged whole pooled condition, including donor angular amplitude",
            ),
        ),
    )
    shutil.copyfile(source / "radius.json", out / "radius.json")
    from render_within_photo_ip import packet

    packet(out, "cohort-3", p)
    summary["render_packet"] = dict(
        subject="cohort-3",
        physical_trials_per_photo=12,
        contributing_subjects=list(PEOPLE),
        packet_sha256=signature,
        design_sha256=sha(out / "design.json"),
        radius_sha256=sha(out / "radius.json"),
        word_predictions_sha256=sha(words_file),
        source_eeg_receipt_sha256=sha(observations),
    )


def run(a, lock):
    require(read(a.within / "selection.json")["selected"] == [2.0, 0.0], "Wrong individual head")
    bank = images(a.source)
    scale = read(a.angle / "scale.json")
    reg = {p: read(paths(a, p, PEOPLE[0])[0]) for p in PANELS}
    members = {}
    classset = None
    for s in PEOPLE:
        classes, members[s] = individual(a, bank, s, reg, scale)
        require(
            classset is None or np.array_equal(classes, classset),
            "Generic classes differ between people",
        )
        classset = classes
    pred = {}
    rng = np.random.default_rng(260531)
    for p in PANELS:
        rows, folds, _, _ = parts(reg[p])
        scores = np.stack([members[s][p]["scores"] for s in PEOPLE])
        mixed = scores.copy()
        for f in range(5):
            ix = np.flatnonzero(folds == f)
            ix = ix[np.argsort(rows[ix])]
            shifts = rng.choice(np.arange(1, len(ix)), 2, replace=False)
            for i, k in enumerate(shifts, start=1):
                mixed[i, ix] = scores[i, np.roll(ix, k)]
        chosen = scores.sum(0).argmax(1)
        pred[p] = dict(
            cohort=chosen,
            misaligned=mixed.sum(0).argmax(1),
            **{s: v.argmax(1) for s, v in zip(PEOPLE, scores, strict=True)},
        )
        dest = a.out / p / "predictions.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dest, rows=rows, folds=folds, individual_scores=scores, **pred[p])
    # All four panels' matched and negative-control predictions are frozen before reading query truth for scoring.
    summary = dict(
        subject="cohort-3",
        contributing_subjects=list(PEOPLE),
        physical_trials_per_photo=12,
        individual_trials_per_photo=4,
        design_sha256=sha(a.out / "design.json"),
        driver_sha256=sha(Path(__file__)),
        panels={},
        within_person_h14_replay_max_abs={
            p: {s: members[s][p]["baseline_max_error"] for s in PEOPLE} for p in PANELS
        },
    )
    all_rows = []
    cluster = []

    def calc(x, labels):
        obj = np.unique(labels)
        g = np.stack([x[labels == w].sum(0) for w in obj])
        den = np.array([(labels == w).sum() for w in obj])
        ix = np.random.default_rng(260532).integers(len(obj), size=(10000, len(obj)))

        def diff(i):
            delta = g[:, 0] - g[:, i]
            q = delta[ix].sum(1) / den[ix].sum(1)
            return dict(
                delta_hits=float(delta.sum()),
                difference=float(delta.sum() / len(x)),
                ci95=np.percentile(q, [2.5, 97.5]).tolist(),
            )

        keys = ("cohort", "sub-04", "sub-05", "sub-01", "misaligned", "mean_single")
        return dict(
            n=len(x),
            objects=len(obj),
            hits={k: float(v) for k, v in zip(keys, x.sum(0), strict=True)},
            cohort_minus_sub04=diff(1),
            cohort_minus_mean_individual=diff(5),
            cohort_minus_misaligned=diff(4),
        )

    for p in PANELS:
        r = reg[p]
        truth = np.asarray(r["truth"])
        vocab = np.array(r["words"])
        d = pred[p]
        cols = np.stack([d[k] == truth for k in ("cohort", *PEOPLE, "misaligned")], axis=1).astype(
            float
        )
        x = np.column_stack([cols, cols[:, 1:4].mean(1)])
        summary["panels"][p] = calc(x, vocab[truth]) | dict(
            predictions_sha256=sha(a.out / p / "predictions.npz")
        )
        all_rows.append(x)
        cluster.extend(vocab[truth])
    require(len(set(cluster)) == 48, "Repeated objects across panels")
    summary["combined"] = calc(np.vstack(all_rows), np.array(cluster))
    passed = (
        summary["panels"]["prospective"]["cohort_minus_sub04"]["delta_hits"] >= 5
        and summary["combined"]["cohort_minus_sub04"]["delta_hits"] > 0
    )
    summary["passed_render_gate"] = bool(passed)
    if passed:
        export(a, reg, scale, bank, members, pred["prospective"]["cohort"], classset, summary)
    write(a.out / "result.json", summary)
    print(
        {p: v["hits"] for p, v in summary["panels"].items()},
        "total",
        summary["combined"]["hits"],
        "gate",
        passed,
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("register", "run"))
    for k, v in dict(
        out="outputs/cohort-r4",
        expanded="outputs/photo-structure/expanded",
        inputs="outputs/object-transfer",
        source="outputs/official-object-readout/sub-03",
        angle="outputs/photo-structure/residual_angle",
        within="outputs/photo-structure/within_photo",
        fresh="outputs/fresh-photo",
        prospective="outputs/prospective-photo",
        references="outputs/object-fidelity",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    a = p.parse_args()
    lock = plan(a)
    if a.stage == "register":
        write(a.out / "design.json", lock)
        print("Locked", sha(a.out / "design.json"))
    else:
        require(read(a.out / "design.json") == lock, "Different registered inputs")
        run(a, lock)


if __name__ == "__main__":
    main()
