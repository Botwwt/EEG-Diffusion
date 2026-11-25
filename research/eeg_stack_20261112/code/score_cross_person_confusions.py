"""Score the nine frozen personal students using only the recipient's query and support EEG."""

from pathlib import Path

import numpy as np
import torch
from fit_cross_person_confusions import ARMS, teacher_bank, wrong_logits
from fit_episodic_object import SEED
from fit_fresh_mix import neural
from fit_fresh_photo_panel import fisher
from fit_hard_object_pairs import episodes
from fit_personal_session_metric import similarities
from fit_trial_object_encoder import TrialObjectEncoder
from prepare_cross_person_confusions import DESIGN, LOCK, PEOPLE, cli, frame, participant, setup
from prepare_scale_trial import PUBLIC
from prepare_trial_object_inputs import check
from register_fresh_photo_panel import freeze, read, sha
from run_prospective_readout import frozen_npz
from safetensors.torch import load_file


def verify_students(a, ids):
    folder = a.out
    fits = read(folder / "fits.json")
    teachers = read(folder / "teachers.json")
    check(
        all(fits[k] == v for k, v in teachers.items())
        and fits["design_sha256"] == LOCK
        and fits["student_driver_sha256"]
        == sha(Path(__file__).with_name("fit_cross_person_confusions.py"))
        and set(fits["students"]) == {f"{s}/{arm}" for s in PEOPLE for arm in ARMS},
        "Incomplete or changed study",
    )
    for key, h in fits["students"].items():
        p = folder / "students" / key / "receipt.json"
        r = read(p)
        check(
            sha(p) == h
            and sha(p.with_name("model.safetensors")) == r["model_sha256"]
            and np.array_equal(r["train_ids"], ids)
            and r["subject"] + "/" + r["arm"] == key
            and r["teachers_sha256"] == sha(folder / "teachers.json")
            and r["trace"] == fits["trace"][r["subject"]],
            "Invalid student, object fold, or control",
        )
    return fits


def patient_scores(a, sub, train_ids, held, session):
    raw, physical = participant(a, sub, train_ids)
    del raw
    query, physical_query = participant(a, sub, held)
    fish = fisher(physical, physical_query).reshape(-1, 96)
    del physical, physical_query
    views = {}
    for arm in ARMS:
        state = load_file(a.out / "students" / sub / arm / "model.safetensors", device=a.device)
        mu, sd = state.pop("input_mean"), state.pop("input_scale")
        model = TrialObjectEncoder().to(a.device)
        model.load_state_dict(state, strict=True)
        views[arm] = neural(model.eval(), query, mu, sd)
    del query
    previous = read(PUBLIC / "shared_trial_encoder.json")["decision_sha256"]
    rawrows = (held[:, None] * 10 + np.arange(10)).ravel()
    check(np.all(np.diff(rawrows) > 0), "Held row index unsorted")
    for k in (4, 12):
        packets = []
        for _, _, _, label, q, d, sup in episodes(
            rawrows.reshape(-1, 10), held, session, k, SEED + 1000 + k
        ):
            qi, di, si = (np.searchsorted(rawrows, v) for v in (q, d, sup))
            check(
                np.array_equal(rawrows[si], sup)
                and not np.isin(np.r_[q, d], sup).any()
                and np.array_equal(rawrows[qi], q)
                and np.array_equal(rawrows[di], d)
                and len(np.unique(session[si[:, 0] // 10])) == 1
                and np.all(q // 10 == sup[label, 0] // 10)
                and np.all(d // 10 != q // 10),
                "Invalid candidate support, donor or query",
            )
            ix = np.r_[qi, di]
            native = similarities(fish[ix], fish[si])
            v = dict(
                rows=q,
                donor_rows=d,
                truth=np.repeat(label, 2),
                candidate_ids=np.tile(sup[:, 0] // 10, (2, 1)),
                support_rows=np.broadcast_to(sup, (2, k, 8)),
                fisher=native[:2].argmax(1),
            )
            for arm, emb in views.items():
                deep = similarities(emb[ix], emb[si])
                votes = (native + deep).argmax(1)
                v.update(
                    {arm: votes[:2], arm + "_wrong": votes[2:], arm + "_neural": deep[:2].argmax(1)}
                )
            packets.append(v)
        arrays = {name: np.concatenate([v[name] for v in packets]) for name in packets[0]}
        p = a.data / "shared-trial-encoder/decisions" / sub / f"generic_{k}.npz"
        check(sha(p) == previous[f"{sub}/{k}"], "Historical individual comparator changed")
        with np.load(p, allow_pickle=False) as z:
            check(
                np.array_equal(arrays["rows"], z["rows"])
                and np.array_equal(arrays["truth"], z["truth"]),
                "Historical menus differ",
            )
            arrays["historical_416"] = z["individual"]
        if sub == "sub-04":
            op = a.data / "trial-object-encoder/generic" / f"generic_{k}.npz"
            with np.load(op, allow_pickle=False) as z:
                check(
                    np.array_equal(arrays["support_rows"], z["support_rows"])
                    and np.array_equal(arrays["candidate_ids"], z["candidate_ids"]),
                    "Sub-04 support rows changed",
                )
        check(np.array_equal(np.sort(arrays["rows"]), rawrows), "Missing or duplicated query")
        path = a.out / "decisions" / sub / f"generic_{k}.npz"
        frozen_npz(path, arrays)
        yield k, path


def metrics(paths, drawn):
    keys = (
        list(ARMS)
        + ["historical_416", "fisher"]
        + [a + s for a in ARMS for s in ("_neural", "_wrong")]
    )
    correct = {k: [] for k in keys}
    persons = {}
    for sub, p in paths.items():
        with np.load(p, allow_pickle=False) as z:
            order = np.argsort(z["rows"])
            ids = z["rows"][order].reshape(96, 10)
            check(
                np.all(ids // 10 == ids[:, :1] // 10) and np.all(ids % 10 == np.arange(10)),
                "Invalid photo/object clustering",
            )
            if not persons:
                first = ids
            check(np.array_equal(ids, first), "Subjects must share exact photos")
            counts = {}
            for k in keys:
                c = (z[k][order] == z["truth"][order]).reshape(96, 10)
                correct[k].append(c)
                counts[k] = int(c.sum())
            persons[sub] = counts
    correct = {k: np.stack(v) for k, v in correct.items()}

    def paired(lhs, rhs):
        x, y = correct[lhs], correct[rhs]
        delta = x.astype(float) - y
        ci = np.quantile(delta.mean((0, 2))[drawn].mean(1), [0.025, 0.975])
        return dict(
            hits_delta=int(x.sum()) - int(y.sum()),
            gained=int((x & ~y).sum()),
            lost=int((y & ~x).sum()),
            delta_percentage_points=round(100 * delta.mean(), 4),
            ci95_percentage_points=np.round(100 * ci, 4).tolist(),
        )

    return dict(
        n=2880,
        per_person_n=960,
        hits={k: int(v.sum()) for k, v in correct.items()},
        per_person=persons,
        matched_minus={
            k: paired("matched", k) for k in ("sham", "zero", "historical_416", "matched_wrong")
        },
        matched_neural_minus={k: paired("matched_neural", k + "_neural") for k in ("sham", "zero")},
    )


def teacher_agreement(a, ids, fold, cells):
    """Check if centered confusions distinguish two external training photos across viewers."""
    all_features, _ = teacher_bank(a, ids, fold)
    rng = np.random.default_rng(read(DESIGN)["seed"])
    correct, cosines = [], []
    for cell in cells:
        c = rng.choice(cell, 12, replace=False)
        p = np.stack([rng.permutation(10) for _ in c])
        c, p = (torch.as_tensor(v, device=all_features.device) for v in (c, p))
        z = all_features[:, :, c[:, None], p]
        # Compare one difference vector per pair. Counting its antisymmetric counterpart
        # a second time would duplicate the same forced-choice decision.
        centered = wrong_logits(z).mean(1)[:, :, :, 0, :]
        peer_mean = (centered.sum(0) - centered) / 2
        agreements = torch.nn.functional.cosine_similarity(centered, peer_mean, dim=-1)
        cosines.extend(agreements.cpu().numpy().ravel())
        correct.append((agreements > 0).cpu().numpy())
    rates = np.concatenate(correct, axis=1)
    nobjects = rates.shape[1]
    sampled = rng.integers(nobjects, size=(10000, nobjects))
    return dict(
        endpoint="Explanatory training-only check. For each photo pair, compare the recipient's hybrid wrong-object logit difference to the mean difference in the other viewers. A positive cosine identifies the correct ordering of the two photos.",
        jointly_matched_cells=len(cells),
        sampled_objects=nobjects,
        paired_orderings=rates.size,
        correct_orderings=int(rates.sum()),
        fraction_correct=float(rates.mean()),
        fraction_ci95_by_object=np.quantile(
            rates.mean((0, 2))[sampled].mean(1), [0.025, 0.975]
        ).tolist(),
        mean_difference_cosine=float(np.mean(cosines)),
        chance=0.5,
    )


def score(a):
    ids, held, schedules, fold, cells = frame(a)
    fits = verify_students(a, ids)
    agreement = teacher_agreement(a, ids, fold, cells)
    paths = {4: {}, 12: {}}
    for who, sub in enumerate(PEOPLE):
        _, session = np.unique(schedules[who, held], axis=0, return_inverse=True)
        for k, p in patient_scores(a, sub, ids, held, session):
            paths[k][sub] = p
    drawn = np.random.default_rng(read(DESIGN)["seed"]).integers(len(held), size=(10000, len(held)))
    twelve, four = (metrics(paths[k], drawn) for k in (12, 4))
    checks = dict()
    for arm in ("sham", "zero"):
        checks["twelve_lower_positive_vs_" + arm] = (
            twelve["matched_minus"][arm]["ci95_percentage_points"][0] > 0
        )
        checks["four_not_lower_vs_" + arm] = four["matched_minus"][arm]["hits_delta"] >= 0
    checks["twelve_not_lower_than_historical"] = (
        twelve["matched_minus"]["historical_416"]["hits_delta"] >= 0
    )
    check(twelve["hits"]["historical_416"] == 1257, "Historical twelve-way baseline changed")
    report = dict(
        design_sha256=LOCK,
        fit_receipts_sha256=sha(a.out / "fits.json"),
        source_sha256={
            name: sha(Path(__file__).with_name(name))
            for name in (
                "prepare_cross_person_confusions.py",
                "fit_cross_person_confusions.py",
                "score_cross_person_confusions.py",
            )
        },
        decision_sha256={
            f"{s}/{k}": sha(p) for k, paths_k in paths.items() for s, p in paths_k.items()
        },
        trained_objects=len(ids),
        jointly_session_matched_training_objects=sum(map(len, cells)),
        class_oof_teacher_fits=12,
        training_only_confusion_agreement=agreement,
        personal_student_fits=9,
        single_subject_test_presentations=4,
        calibration_support_photos_per_candidate=8,
        held_objects=len(held),
        held_photos=960,
        people=list(PEOPLE),
        sub04_zero_max_weight_error_vs_previous_928_model=fits["sub04_zero_reference_max_abs"],
        endpoint="Personal same-session object accuracy on the previously studied 96 held generic objects. Same 960 photos per person. Teachers use cross-fitted training EEG only.",
        intervals="Percentage-point contrasts with synchronous whole-object percentile resampling over 96 physical objects, all photos and all three fixed participants, 10000 draws; no population-of-participants inference.",
        twelve_way=twelve,
        four_way=four,
        decision_checks=checks,
        advance=all(checks.values()),
    )
    freeze(a.out / "report.json", report)
    if a.publish:
        freeze(PUBLIC / "cross_person_confusions.json", report)
    print(
        "12-way",
        twelve["hits"],
        twelve["matched_minus"],
        "4-way",
        four["hits"],
        "advance",
        report["advance"],
        flush=True,
    )


if __name__ == "__main__":
    setup()
    p = cli()
    p.add_argument("--publish", action="store_true")
    score(p.parse_args())
