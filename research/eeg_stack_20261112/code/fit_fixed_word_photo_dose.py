"""Freeze, fit, then evaluate nested same-photo EEG cohorts with the same predicted words."""

import argparse
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fit_personal_session_metric import load
from fit_repeat_photo_subspace import centered, ridge
from fit_support_structure import H14_SHA, population, unit
from fit_within_photo_embedding import images
from prepare_expanded_personal import extract
from register_fresh_photo_panel import freeze, read
from render_fixed_word_photo_dose import GROUPS, PANELS, PEOPLE, packet
from render_within_photo_ip import check, sha
from run_prospective_readout import frozen_npz

ROOT = Path("outputs/fixed-word-photo-dose")
INPUTS = Path("outputs/object-transfer")
SOURCE = Path("outputs/official-object-readout/sub-03")
PHOTO = Path("outputs/photo-structure")
REF = Path("outputs/object-fidelity")
PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
LOCK = PUBLIC / "fixed_word_photo_dose_design.json"
SEED = 261125


def design():
    return dict(
        study="Same-stimulus photo residual dose on two previously studied development panels. The nested people and all settings are fixed. No new population or prospective test is claimed.",
        panels=list(PANELS),
        nested_contributors={k: list(v) for k, v in GROUPS.items()},
        physical_repetitions_per_person=4,
        word_task="Use the previously frozen sub-04 512-object posterior R4 object choice, eight support rows, donor map, uniform H14 vector and individual angular magnitude. Other observers never vote on the word. The query EEG is available separately from every contributor for the very same photograph.",
        visual_fit="Each person's R4 (17 posterior electrodes, the original 1003 features) is fit on exactly the same original 128 generic objects, 10 photos each. Standardize within-object demeaned EEG, map to within-object demeaned unit H14 features with the fixed original ridge 2.0 and no visual whitening. No selection in these data.",
        pooling="For each predicted word, subtract the mean predicted EEG embedding of its eight support rows, project to the tangent of the frozen support prior, normalize each participant's tangent, normalize the equal sum. Every dose uses the old 512 sub-04 recipient angle and unit prior. The wrong arm retains that recipient angle, using the frozen wrong row's cohort tangent from the same word and fold. Uniform is byte-identical at every dose. The old single-person 512 head is reported as a separate reference.",
        primary=[
            "All 240 queries: top-1 raw cosine of each unit H14 condition against the ten original photos of its true object. The decoder only gets the predicted word. Eight class support images appear in the ten-way evaluation gallery whenever that word is correct.",
            "Two-way summed cosine assignment for a same-true-object, same-fold pair, conditional on both frozen words being correct. Ties split equally.",
        ],
        secondary="All ten-way hits restricted to correct predicted words; ten-way when subtracting the same frozen unit support prior from both decision and candidate (zero residual ties); strict two-way restricted to pairs sharing both recorded sessions for all nine; strict two-way where the frozen donors are the two actual partners; independently classified single members of strict pairs after subtracting the prior.",
        comparisons="Report 1,3,9 observation groups plus native512 reference; same-word/fold wrong and uniform controls. Two-sided paired percentile 95% CI on nine minus one, nine minus three, three minus one and nine minus native, uniform and wrong. Resample whole object clusters jointly across models and within each panel for combined statistics, 10,000 draws, seed 261125.",
        acquisition="Fixed participants, 4, 12 or 36 actual presentations per photo. Panel photos and labels were used historically; this is a predeclared new comparison on development stimuli. Different people view the same stimuli, so do not count people or repetitions as independent test cases.",
        setup="Each arm uses the same frozen four-step Turbo/IP protocol, 923761+original row, and radius from the original 32 generic image calibration. Only support embeddings are accessed at packet preparation. Score target photo embeddings only after every packet is frozen.",
        visual_sha256=H14_SHA,
        sources={
            "generic_receipts": {s: sha(INPUTS / "generic" / (s + ".json")) for s in PEOPLE},
            "target_half_receipts": {s: sha(INPUTS / "halves" / (s + ".json")) for s in PEOPLE},
            "target_full_receipts": {s: sha(INPUTS / (s + ".json")) for s in PEOPLE},
            "registration": {p: sha(REF / (p + "_ablation.json")) for p in PANELS},
            "words": {p: sha(PHOTO / "expanded" / p / "sub-04/evaluation.json") for p in PANELS},
            "native_packets": {
                p: sha(PHOTO / "residual_angle/conditions/sub-04" / p / "conditions_512.npz")
                for p in PANELS
            },
            "prior_radius": sha(PHOTO / "residual_angle/radius.json"),
            "angle_design": sha(PHOTO / "residual_angle/design.json"),
            "visual_selection": sha(PHOTO / "within_photo/selection.json"),
        },
    )


def study_eeg(subject, table):
    generic = INPUTS / "generic" / (subject + ".npz")
    target = INPUTS / "halves" / (subject + ".npz")
    full = INPUTS / (subject + ".npz")
    gr, tr, fr = [read(p.with_suffix(".json")) for p in (generic, target, full)]
    for p, r in ((generic, gr), (target, tr), (full, fr)):
        check(sha(p) == r["packet_sha256"] and r["subject"] == subject, "Changed physical packet")
    check(
        tr["parent_packet_sha256"] == fr["packet_sha256"]
        and gr["source_full_lfs_sha256"] == fr["source_full_lfs_sha256"]
        and gr["revision"] == tr["source_revision"] == fr["revision"]
        and gr["schedule_sha256"] == tr["schedule_sha256"],
        "Inconsistent public source, parent or acquisition schedule",
    )
    rr, _, _, gm = load(generic)
    qr, _, sessions, qm = load(target)
    classes = np.asarray(gr["concepts"])
    check(
        classes.shape == (128,)
        and np.array_equal(rr, (10 * classes[:, None] + np.arange(10)).ravel())
        and np.all(np.diff(qr) > 0)
        and len(qr) == 240
        and not np.intersect1d(classes, qr // 10).size,
        "Incorrect training or target population",
    )
    with (
        np.load(generic, allow_pickle=False) as gen,
        np.load(target, allow_pickle=False) as t,
        np.load(full, allow_pickle=False) as f,
    ):
        for z, rows in ((gen, rr), (f, qr)):
            check(
                np.array_equal(z["rows"], rows)
                and z["stimulus_stems"].tolist() == [table[r]["stimulus_stem"] for r in rows]
                and z["unique_ids"].tolist() == [table[r]["unique_id"] for r in rows]
                and int(z["repeats"]) == 4,
                "Physical rows or R4 budget changed",
            )
        check(
            all(
                int(z["repeats_per_half"]) == 2
                and sorted(z["stored_slots"].ravel()) == list(range(4))
                for z in (t, gen)
            ),
            "Two session means must cover four disjoint actual observations",
        )
        posterior = f["mean_eeg"][:, [f["channels"].tolist().index(c) for c in t["channels"]]]
        check(
            np.allclose(posterior, qm, atol=2e-6, rtol=1e-5),
            "The two session halves do not reconstruct the physical R4",
        )
    return (
        classes,
        rr,
        extract(gm).reshape(128, 10, -1).astype(float),
        qr,
        extract(qm).astype(float),
        sessions,
    )


def normalized_tangent(direction, axis):
    v = direction - (direction * axis).sum(1, keepdims=True) * axis
    check(np.all(np.linalg.norm(v, axis=1) > 1e-10), "Undefined EEG residual")
    return unit(v)


def prepare(root):
    plan = read(LOCK)
    table = population(SimpleNamespace(source=SOURCE))
    bank = images(SOURCE)
    projections = {}
    sessions = {}
    all_rows = classes = None
    for subject in PEOPLE:
        c, trainrows, x, qr, q, ss = study_eeg(subject, table)
        if all_rows is not None:
            check(
                np.array_equal(all_rows, qr) and np.array_equal(classes, c),
                "Different observations across people",
            )
        classes, all_rows = c, qr
        train = unit(bank[trainrows].astype(float)).reshape(128, 10, 1024)
        projections[subject] = ridge(x, centered(train), q)
        sessions[subject] = ss
        print("Fitted", subject, len(trainrows), "generic photos", flush=True)
    dest = root / "render_inputs"
    freeze(dest / "study_design.json", plan)
    hashes = {}
    audit = {}
    seen = []
    for panel in PANELS:
        reg = read(REF / (panel + "_ablation.json"))
        w = read(PHOTO / "expanded" / panel / "sub-04/evaluation.json")
        rows, folds, vocab, cal = (
            np.asarray(reg[k])
            for k in ("rows", "folds", "words", "calibration_rows_by_fold_and_word")
        )
        check(
            w["subject"] == "sub-04"
            and w["rows"] == reg["rows"]
            and w["folds"] == reg["folds"]
            and w["words"] == reg["words"],
            "Wrong fixed word source",
        )
        indices = np.asarray(w["results"]["expanded_r4"]["predicted"])
        words = vocab[indices]
        support = cal[folds, indices]
        oldfile = PHOTO / "residual_angle/conditions/sub-04" / panel / "conditions_512.npz"
        with np.load(oldfile, allow_pickle=False) as d:
            native = {
                k: d[k]
                for k in (
                    "rows",
                    "folds",
                    "uniform",
                    "own",
                    "predicted_words",
                    "donor_rows",
                    "support_rows",
                    "word_prediction_sha256",
                    "registration_sha256",
                )
            }
        check(
            np.array_equal(native["rows"], rows)
            and np.array_equal(native["folds"], folds)
            and np.array_equal(native["predicted_words"], words)
            and np.array_equal(native["support_rows"], support)
            and str(native["word_prediction_sha256"]) == plan["sources"]["words"][panel]
            and str(native["registration_sha256"]) == plan["sources"]["registration"][panel],
            "Old conditions contradict registration",
        )
        qix = np.searchsorted(all_rows, rows)
        six = np.searchsorted(all_rows, support)
        check(
            np.array_equal(all_rows[qix], rows) and np.array_equal(all_rows[six], support),
            "Missing EEG rows",
        )
        for f in range(5):
            check(
                set(cal[f].ravel()) == set(rows[folds != f])
                and np.unique(cal[f]).size == 96
                and not np.intersect1d(support[folds == f], rows[folds == f]).size,
                "Held-out photograph entered active fold supports",
            )
            prior = unit(unit(bank[support[folds == f]].astype(float)).mean(1)).astype(np.float32)
            check(
                np.array_equal(prior, native["uniform"][folds == f]),
                "Support-only image prior changed",
            )
        seen.extend(rows)
        axis = unit(native["uniform"].astype(float))
        old = unit(native["own"].astype(float))
        dot = (old * axis).sum(1, keepdims=True)
        angles = np.arctan2(np.linalg.norm(old - dot * axis, axis=1, keepdims=True), dot)
        tangents = {
            s: normalized_tangent(v[qix] - v[six].mean(1), axis) for s, v in projections.items()
        }
        with np.load(PHOTO / "within_photo/sub-04" / panel / "conditions_128.npz") as prior128:
            check(
                np.array_equal(prior128["training_classes"], classes)
                and np.array_equal(prior128["support_rows"], support),
                "Mismatch with original 128-class baseline",
            )
            original = normalized_tangent(prior128["own"].astype(float), axis)
            replay = float((original * tangents["sub-04"]).sum(1).min())
            check(replay > 0.999999, "Single-person 128-class direction fails historical replay")
        inv = {r: i for i, r in enumerate(rows)}
        donors = np.array([inv[r] for r in native["donor_rows"]])
        audit[panel] = dict(
            minimum_single128_tangent_cosine_with_historical=replay,
            native_packet_sha256=sha(oldfile),
        )
        for name, people in GROUPS.items():
            direction = unit(np.sum([tangents[s] for s in people], axis=0))
            own = (np.cos(angles) * axis + np.sin(angles) * direction).astype(np.float32)
            wrong = (np.cos(angles) * axis + np.sin(angles) * direction[donors]).astype(np.float32)
            path = dest / "conditions" / name / panel / "conditions_128.npz"
            frozen_npz(
                path,
                dict(
                    rows=rows,
                    folds=folds,
                    predicted_words=words,
                    support_rows=support,
                    donor_rows=native["donor_rows"],
                    own=own,
                    wrong=wrong,
                    uniform=native["uniform"],
                    recipient_angle_radians=angles[:, 0],
                    training_classes=classes,
                    contributing_subjects=np.asarray(people),
                    physical_trials_per_photo=4 * len(people),
                    study_design_sha256=sha(LOCK),
                    native_packet_sha256=sha(oldfile),
                    visual_sha256=H14_SHA,
                ),
            )
            hashes[f"{name}/{panel}"] = sha(path)
    check(np.array_equal(np.sort(seen), all_rows), "Expected exactly both panels' rows")
    frozen_npz(
        root / "sessions.npz",
        dict(
            rows=all_rows,
            subjects=np.asarray(PEOPLE),
            session_ids=np.stack([sessions[s] for s in PEOPLE]),
        ),
    )
    prior = PHOTO / "residual_angle/radius.json"
    dst = dest / "radius.json"
    if not dst.exists():
        shutil.copyfile(prior, dst)
    check(sha(dst) == plan["sources"]["prior_radius"], "Radius changed")
    freeze(
        dest / "design.json",
        dict(
            study_design_sha256=sha(LOCK),
            packets_sha256=hashes,
            visual_sha256=H14_SHA,
            radius_sha256=sha(dst),
            generic_radius_rows=read(dst)["generic_rows"],
            panels=list(PANELS),
            nested_contributors=plan["nested_contributors"],
            classifier_generic_objects=512,
            visual_generic_objects_per_person=128,
            session_ids_sha256=sha(root / "sessions.npz"),
            single128_replay=audit,
        ),
    )
    for name in GROUPS:
        for p in PANELS:
            packet(dest, name, p)
    print("Frozen all six packets", sha(dest / "design.json"), flush=True)


def hits(score, target):
    best = np.max(score, axis=1, keepdims=True)
    ties = abs(score - best) <= 1e-12
    return ties[np.arange(len(target)), target] / ties.sum(1)


def sign_hit(d):
    return (d > 1e-12).astype(float) + 0.5 * (abs(d) <= 1e-12)


def scored_panel(root, panel, bank, ss, er):
    reg = read(REF / (panel + "_ablation.json"))
    rows, folds, vocab, truth = [np.asarray(reg[k]) for k in ("rows", "folds", "words", "truth")]
    labels = vocab[truth]
    sources = {}
    for name in GROUPS:
        r, f, words, donors, support, vec, *_ = packet(root / "render_inputs", name, panel)
        check(np.array_equal(r, rows) and np.array_equal(f, folds), "Scoring registration differs")
        sources[name] = vec["own"]
        if name == "cohort-9":
            sources["nine_wrong"] = vec["wrong"]
            sources["uniform"] = vec["uniform"]
    op = PHOTO / "residual_angle/conditions/sub-04" / panel / "conditions_512.npz"
    with np.load(op, allow_pickle=False) as native:
        check(
            np.array_equal(words, native["predicted_words"])
            and np.array_equal(donors, native["donor_rows"])
            and np.array_equal(support, native["support_rows"])
            and np.array_equal(sources["uniform"], native["uniform"]),
            "Different word, supports or donors",
        )
        sources["native512"] = native["own"]
    refs = unit(bank[rows].astype(float))
    source_rows = np.array([np.flatnonzero(truth == k) for k in range(12)])
    gallery = source_rows[truth]
    local = (gallery == np.arange(120)[:, None]).argmax(1)
    correct = labels == words
    pairs = []
    for k in range(12):
        for f in range(5):
            part = np.flatnonzero((truth == k) & (folds == f))
            check(len(part) == 2, "Expected one doublet")
            if all(correct[part]):
                pairs.append(part)
    pairs = np.asarray(pairs)
    left, right = pairs.T
    sessions = ss[:, :, np.searchsorted(er, rows)]
    same_session = np.all(sessions[:, :, left] == sessions[:, :, right], axis=(0, 1))
    reciprocal = (donors[left] == rows[right]) & (donors[right] == rows[left])

    def entries(names, ids):
        return [
            dict(panel=panel, object=str(name), rows=np.atleast_1d(r).tolist(), scores={})
            for name, r in zip(names, ids, strict=True)
        ]

    # Each record is one stimulus, or one jointly assigned pair. Raw scores never enter preparation.
    data = {
        "ten_all": entries(labels, rows),
        "ten_correct": entries(labels[correct], rows[correct]),
        "centered_ten_all": entries(labels, rows),
        "centered_ten_correct": entries(labels[correct], rows[correct]),
        "strict_pair": entries(labels[left], rows[pairs]),
        "strict_session_matched": entries(labels[left[same_session]], rows[pairs[same_session]]),
        "strict_reciprocal_wrong": entries(labels[left[reciprocal]], rows[pairs[reciprocal]]),
        "centered_single_from_strict": entries(labels[pairs.ravel()], rows[pairs.ravel()]),
    }
    axis = unit(sources["uniform"].astype(float))
    for name, v in sources.items():
        v = unit(v.astype(float))
        sim = v @ refs.T
        raw = hits(np.take_along_axis(sim, gallery, axis=1), local)
        cent = unit(v - axis)
        cand = unit(refs[gallery] - axis[:, None])
        residual = np.einsum("ij,ikj->ik", cent, cand)
        ch = hits(residual, local)
        diag = sim[left, left] + sim[right, right] - sim[left, right] - sim[right, left]
        pairhit = sign_hit(diag)
        si = pairs.ravel()
        peer = pairs[:, ::-1].ravel()
        j = (gallery[si] == peer[:, None]).argmax(1)
        centered_single = sign_hit(residual[si, local[si]] - residual[si, j])
        for k, vv in dict(
            ten_all=raw,
            ten_correct=raw[correct],
            centered_ten_all=ch,
            centered_ten_correct=ch[correct],
            strict_pair=pairhit,
            strict_session_matched=pairhit[same_session],
            strict_reciprocal_wrong=pairhit[reciprocal],
            centered_single_from_strict=centered_single,
        ).items():
            for r, value in zip(data[k], vv, strict=True):
                r["scores"][name] = float(value)
    check(
        all(r["scores"]["uniform"] == 0.5 for r in data["strict_pair"])
        and all(r["scores"]["uniform"] == 0.1 for r in data["centered_ten_all"]),
        "Tied uniform control failed",
    )
    return data


def summarize(records, comparisons=None, continuous=False):
    if not records:
        return dict(n=0, objects=0, arms={}, contrasts={})
    models = list(records[0]["scores"])
    rows = np.array([[r["scores"][k] for k in models] for r in records])
    rng = np.random.default_rng(SEED)
    draws = np.zeros((10000, len(models)))
    ns = np.zeros(10000)
    nobj = 0
    for p in PANELS:
        groups = sorted({r["object"] for r in records if r["panel"] == p})
        if not groups:
            continue
        blocks = [rows[[r["panel"] == p and r["object"] == w for r in records]] for w in groups]
        nobj += len(groups)
        k = rng.integers(len(blocks), size=(10000, len(blocks)))
        draws += np.array([b.sum(0) for b in blocks])[k].sum(1)
        ns += np.array(list(map(len, blocks)))[k].sum(1)
    draws /= ns[:, None]
    counts = rows.sum(0)
    out = dict(n=len(rows), objects=nobj, arms={}, contrasts={})
    for j, k in enumerate(models):
        interval = np.quantile(draws[:, j], [0.025, 0.975]).tolist()
        if continuous:
            out["arms"][k] = dict(mean=float(counts[j] / len(rows)), ci95=interval)
        else:
            out["arms"][k] = dict(
                hits=float(counts[j]), rate=float(counts[j] / len(rows)), ci95=interval
            )
    if comparisons is None:
        comparisons = (
            ("cohort-9", "single-1"),
            ("cohort-9", "cohort-3"),
            ("cohort-3", "single-1"),
            ("cohort-9", "native512"),
            ("cohort-9", "nine_wrong"),
            ("cohort-9", "uniform"),
        )
    for a, b in comparisons:
        i, j = models.index(a), models.index(b)
        out["contrasts"][a + "_minus_" + b] = dict(
            net_hits=float(counts[i] - counts[j]),
            delta=float((counts[i] - counts[j]) / len(rows)),
            ci95=np.quantile(draws[:, i] - draws[:, j], [0.025, 0.975]).tolist(),
        )
        if continuous:
            del out["contrasts"][a + "_minus_" + b]["net_hits"]
    return out


def score(root):
    dest = root / "render_inputs"
    d = read(dest / "design.json")
    check(
        d["study_design_sha256"] == sha(LOCK)
        and d["session_ids_sha256"] == sha(root / "sessions.npz")
        and set(d["packets_sha256"]) == {k + "/" + p for k in GROUPS for p in PANELS},
        "Need six frozen packets",
    )
    for k in GROUPS:
        for p in PANELS:
            packet(dest, k, p)
    with np.load(root / "sessions.npz", allow_pickle=False) as z:
        check(
            z["subjects"].tolist() == list(PEOPLE),
            "Different participant order in recording schedule",
        )
        sessions, rows = z["session_ids"], z["rows"]
    bank = images(SOURCE)
    details = {p: scored_panel(root, p, bank, sessions, rows) for p in PANELS}
    freeze(root / "embedding_decisions.json", details)
    bypanel = {p: {m: summarize(r) for m, r in v.items()} for p, v in details.items()}
    combined = {
        m: summarize([r for p in PANELS for r in details[p][m]]) for m in details[PANELS[0]]
    }
    report = dict(
        design_sha256=sha(LOCK),
        render_inputs_sha256=sha(dest / "design.json"),
        packet_sha256=d["packets_sha256"],
        per_observation_report_sha256=sha(root / "embedding_decisions.json"),
        participants=list(PEOPLE),
        generic_classes_per_person=128,
        physical_repeats=[4, 12, 36],
        word_model="sub-04 posterior personal Fisher96 calibrated with 512 unrelated objects",
        space="Author ViT-H/14 target features, evaluated after frozen packets. Direct condition-space measurements; an independent image encoder must evaluate saved diffusion JPEGs separately.",
        uncertainty="Paired whole-object percentile bootstrap, stratified by registered/replicate panel, 10000 draws. Eligible objects are included for each endpoint. Rate differences use fractions.",
        per_panel=bypanel,
        combined=combined,
    )
    freeze(PUBLIC / "fixed_word_photo_dose.json", report)
    for m in (
        "ten_all",
        "ten_correct",
        "strict_pair",
        "strict_session_matched",
        "centered_ten_all",
        "centered_single_from_strict",
    ):
        s = combined[m]
        print(
            m,
            s["n"],
            {k: v["hits"] for k, v in s["arms"].items()},
            s["contrasts"].get("cohort-9_minus_single-1"),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("register", "prepare", "score"))
    p.add_argument("--root", type=Path, default=ROOT)
    a = p.parse_args()
    if a.stage == "register":
        freeze(LOCK, design())
        print(LOCK, sha(LOCK))
    else:
        check(read(LOCK) == design(), "Study sources differ from the registered comparison")
        if a.stage == "prepare":
            prepare(a.root)
        else:
            score(a.root)


if __name__ == "__main__":
    main()
