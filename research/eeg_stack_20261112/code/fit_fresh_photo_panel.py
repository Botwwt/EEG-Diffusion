"""Replay historical predictions, then apply the registered 512-object personal decoder."""

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from audit_registered_sessions import audit_panel
from calibrate_photo_angles import rotate
from decode_eeg_layout import CHANNELS
from fit_personal_session_metric import load, object_directions, similarities
from fit_support_structure import H14_SHA, unit
from fit_within_photo_embedding import bases, generative, images, predict, training
from prepare_expanded_personal import extract
from prepare_registered_subjects import (
    OFFSET,
    REPO,
    REV,
    ROW_BYTES,
    SHAPE,
    SIZE,
    axes,
    digest,
    ranged,
)
from register_fresh_photo_panel import PUBLIC, SUBJECTS, freeze, read, sha
from render_within_photo_ip import packet


def require(v, message):
    if not v:
        raise ValueError(message)


def verify_lock(a):
    lock = read(PUBLIC)
    files = {
        "generic_residual_selection": a.within / "selection.json",
        "angle_scale": a.angle / "scale.json",
        "radius": a.angle / "radius.json",
        "old_render_design": a.angle / "design.json",
        "row_manifest": a.source / "atm_train_rows.csv",
        **{s + "_training": a.expanded / "features" / (s + ".json") for s in SUBJECTS},
    }
    require(
        sha(PUBLIC) == sha(a.out / "panel_lock.json")
        and lock["registration_sha256"] == sha(a.out / "references/fresh_ablation.json")
        and lock["exclusion_sets_sha256"] == sha(a.out / "registration/exclusion_sets.json")
        and all(sha(v) == lock["inputs_sha256"][k] for k, v in files.items()),
        "Public protocol, registration, training, scale or source input changed",
    )
    return lock


def physical(a, sub, rows, replay):
    """Read the two original stored trials for each half, preserving legacy float32 averaging."""
    concepts = np.unique(rows // 10)
    sorted_rows = (concepts[:, None] * 10 + np.arange(10)).ravel()
    require(np.array_equal(sorted_rows, np.sort(rows)), "Expected complete source blocks")
    source = read(a.inputs / (sub + ".json"))
    url = f"{REPO}/resolve/{REV}/Preprocessed_data_250Hz/{sub}/preprocessed_eeg_training.npy"
    require(
        source["subject"] == sub
        and source["source"] == url
        and source["revision"] == REV
        and source["source_size"] == SIZE
        and source["source_full_lfs_sha256"]
        == read(a.expanded / "features" / (sub + ".json"))["source_full_lfs_sha256"],
        "EEG source differs from this participant's 512-object training release",
    )
    existing = a.inputs / "ranges" / REV / sub
    cache = a.out / "ranges" / REV / sub
    cache.mkdir(parents=True, exist_ok=True)

    def raw(first, last):
        key = f"{first}-{last}"
        path = existing / (key + ".bin")
        if path.is_file():
            block = path.read_bytes()
            require(
                len(block) == last - first + 1
                and digest(block) == (existing / (key + ".sha256")).read_text().strip(),
                "Cached EEG range failed checksum",
            )
            return block
        require(not replay, "Replay requires original cached physical EEG ranges")
        return ranged(url, first, last, cache)

    channels, times = axes(raw(0, OFFSET - 1), raw(OFFSET + SHAPE[0] * ROW_BYTES, SIZE - 1))
    post = [channels.tolist().index(c) for c in CHANNELS]
    slots = np.random.RandomState(20200220).permutation(4)
    early, late = np.flatnonzero(slots < 2), np.flatnonzero(slots >= 2)
    require(early.tolist() == [1, 3] and late.tolist() == [0, 2], "Unexpected trial slot order")

    def fetch(c):
        lo = OFFSET + int(c) * 10 * ROW_BYTES
        hi = lo + 10 * ROW_BYTES - 1
        data = raw(lo, hi)
        x = np.take(np.frombuffer(data, "<f8").reshape(10, *SHAPE[1:]), post, axis=2).astype(
            np.float32
        )
        require(np.isfinite(x).all(), "Nonfinite posterior EEG")
        h = np.stack([x[:, early].mean(1), x[:, late].mean(1)])
        delta = np.abs(h.mean(0) - x.mean(1))
        require(
            np.all(delta <= 4 * np.finfo(np.float32).eps * np.abs(x).max(1)),
            "Half means disagree with physical four-trial mean",
        )
        return h, dict(
            concept_id=int(c),
            row_start=int(c) * 10,
            byte_start=lo,
            byte_end=hi,
            sha256=digest(data),
        )

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        blocks = list(pool.map(fetch, concepts))
    halves = np.concatenate([v[0] for v in blocks], axis=1)
    receipt = dict(
        subject=sub,
        source=url,
        source_revision=REV,
        source_size=SIZE,
        source_full_lfs_sha256=source["source_full_lfs_sha256"],
        four_physical_repeats=True,
        native_dtype="little-endian float64",
        computation_dtype="float32",
        stored_slots=[early.tolist(), late.tolist()],
        channels=CHANNELS,
        blocks=[b[1] for b in blocks],
        note="Four-trial query uses the mean of two two-trial means in the same order as the locked personal model. Physical ranges are verified by byte offsets, lengths and SHA256, the entire upstream LFS digest identifies the release.",
    )
    if replay:
        old = a.inputs / "halves" / (sub + ".npz")
        with np.load(old, allow_pickle=False) as z:
            take = np.searchsorted(z["rows"], sorted_rows)
            require(
                np.array_equal(z["rows"][take], sorted_rows)
                and np.array_equal(z["half_eeg"][:, take], halves)
                and np.array_equal(z["times"], times[50:])
                and z["channels"].tolist() == CHANNELS,
                "Historical source waveform replay failed",
            )
        original = {r["rows"][0]: r["sha256"] for r in source["row_ranges"]}
        require(
            all(original[b["row_start"]] == b["sha256"] for b in receipt["blocks"]),
            "Physical bytes differ from historical receipt",
        )
        receipt["historical_packet_sha256"] = sha(old)
        receipt["waveforms_bit_identical"] = True
        return sorted_rows, halves.mean(0), receipt
    dest = a.out / "eeg" / (sub + ".npz")
    dest.parent.mkdir(parents=True, exist_ok=True)
    sessions = np.array(read(a.inputs / "sessions.json")["subjects"][sub]["session_pairs"])[
        sorted_rows // 10
    ].T
    require(
        np.isin(sessions[0], [1, 2]).all() and np.isin(sessions[1], [3, 4]).all(),
        "Unexpected physical recording schedule",
    )
    np.savez(
        dest,
        rows=sorted_rows,
        half_eeg=halves,
        times=times[50:],
        channels=CHANNELS,
        stored_slots=[early, late],
        half_names=["early", "late"],
        session_ids=sessions,
        subject=sub,
        repeats_per_half=2,
    )
    freeze(
        dest.with_suffix(".json"),
        receipt
        | dict(
            packet_sha256=sha(dest),
            registration_sha256=sha(a.out / "references/fresh_ablation.json"),
        ),
    )
    compact = a.out / "target" / (sub + ".npz")
    compact.parent.mkdir(parents=True, exist_ok=True)
    mean = halves.mean(0)
    features = extract(mean)
    np.savez(
        compact,
        rows=sorted_rows,
        channels=CHANNELS,
        times=times[50:],
        mean_eeg=mean,
        features=features,
        feature_sha=digest(features.tobytes()),
        source_sha256=sha(dest),
        registration_sha256=sha(a.out / "references/fresh_ablation.json"),
        subject=sub,
    )
    return sorted_rows, mean, receipt


def fisher(train, queries):
    """The 96-direction expanded_r4 transform from prepare_expanded_personal.predict."""
    x = train.reshape(-1, train.shape[-1])
    mean, scale = x.mean(0), x.std(0).clip(1e-4)
    grouped = ((x - mean) / scale).reshape(train.shape)
    centers = grouped.mean(1)
    residual = (grouped - centers[:, None]).reshape(x.shape)
    basis = object_directions(
        centers.T @ centers / len(centers), residual.T @ residual / len(x), 96
    )
    return ((queries - mean) / scale) @ basis


def make_conditions(
    rows, folds, vocabulary, calibration, source_rows, features, residual, bank, scale
):
    """Infer using EEG and fold support membership. Query truth is absent from the interface."""
    chosen = np.full(len(rows), -1)
    own = np.empty((len(rows), 1024), np.float32)
    uniform = np.empty_like(own)
    support = np.empty((len(rows), 8), dtype=np.int64)
    for f, groups in enumerate(calibration):
        ix = np.flatnonzero(folds == f)
        require(
            groups.shape == (12, 8)
            and len(np.unique(groups)) == 96
            and set(groups.ravel()) == set(rows[folds != f]),
            "Held-out query present in support or support incomplete",
        )
        cal = np.searchsorted(source_rows, groups)
        query = np.searchsorted(source_rows, rows[ix])
        require(
            np.array_equal(source_rows[cal], groups)
            and np.array_equal(source_rows[query], rows[ix]),
            "Unknown EEG row",
        )
        ids = similarities(features[query], features[cal]).argmax(1)
        chosen[ix] = ids
        support[ix] = groups[ids]
        # Read image embeddings only for the predicted eight-shot support sets of the current fold.
        mean = unit(bank[groups[ids]].astype(np.float64)).mean(1)
        uniform[ix] = unit(mean)
        own[ix] = generative(mean, residual[query] - residual[cal[ids]].mean(1))
    words = vocabulary[chosen]
    donor = np.arange(len(rows))
    for f in range(5):
        for w in np.unique(words[folds == f]):
            group = np.flatnonzero((folds == f) & (words == w))
            donor[group] = np.roll(group, 1)
    rotated = rotate(uniform, own, scale)
    return (
        chosen,
        dict(
            rows=rows,
            folds=folds,
            predicted_words=words,
            support_rows=support,
            donor_rows=rows[donor],
            own=rotated,
            uniform=uniform,
            wrong=rotated[donor],
        ),
        own,
    )


def apply(a, sub, ref, bank, source_rows, mean):
    config = read(a.within / "selection.json")
    scale = read(a.angle / "scale.json")
    classes, x, y = training(a, sub, 512, bank)
    require(
        not np.intersect1d(classes, source_rows // 10).size and config["selected"] == [2.0, 0.0],
        "Generic fit overlaps target or config differs",
    )
    q = extract(mean).astype(np.float64)
    vec = fisher(x, q)
    basis = bases(x, y, q)
    residual = predict(basis, config["selected"]) @ basis[2].T
    args = [
        np.asarray(ref[k]) for k in ("rows", "folds", "words", "calibration_rows_by_fold_and_word")
    ]
    pred, data, base = make_conditions(*args, source_rows, vec, residual, bank, scale)
    return (
        pred,
        data
        | dict(
            selected=config["selected"],
            training_classes=classes,
            selection_sha256=sha(a.within / "selection.json"),
            visual_sha256=H14_SHA,
        ),
        base,
    )


def replay(a, bank):
    sub, panel = "sub-04", "registered"
    ref = read(a.references / (panel + "_ablation.json"))
    old_packet = a.angle / "conditions" / sub / panel / "conditions_512.npz"
    old_raw = a.within / sub / panel / "conditions_512.npz"
    # The historical GEMMs saw the joint 240-row matrix; duplicate that order for replay.
    old_rows, _, _, old_mean = load(a.inputs / "halves" / (sub + ".npz"))
    rows, mean, receipt = physical(a, sub, old_rows, True)
    require(
        np.array_equal(rows, old_rows) and np.array_equal(old_mean, mean), "Mean-four parity failed"
    )
    predicted, data, base = apply(a, sub, ref, bank, rows, mean)
    old_eeg = read(a.expanded / panel / sub / "evaluation.json")["results"]["expanded_r4"]
    require(predicted.tolist() == old_eeg["predicted"], "Historical 120-word replay failed")
    with np.load(old_raw, allow_pickle=False) as z:
        errors = {"base_own": float(np.max(np.abs(z["own"] - base)))}
    with np.load(old_packet, allow_pickle=False) as z:
        for k in (
            "rows",
            "folds",
            "support_rows",
            "donor_rows",
            "predicted_words",
            "training_classes",
        ):
            require(np.array_equal(data[k], z[k]), "Historical discrete replay failed: " + k)
        for k in ("own", "wrong", "uniform"):
            errors[k] = float(np.max(np.abs(data[k] - z[k])))
    require(max(errors.values()) <= 5e-6, "Replayed H/14 model exceeds float32 numerical tolerance")
    freeze(
        a.out / "replay.json",
        dict(
            subject=sub,
            panel=panel,
            raw_row_count=len(rows),
            all_240_waveforms_bit_identical=True,
            all_120_prediction_ids_identical=True,
            word_hits=old_eeg["hits"],
            max_abs_h14_errors=errors,
            raw_blocks_sha256=[r["sha256"] for r in receipt["blocks"]],
            row_registration_sha256=sha(a.references / (panel + "_ablation.json")),
            old_condition_sha256=sha(old_packet),
            historical_waveforms_sha256=receipt["historical_packet_sha256"],
            public_lock_sha256=sha(PUBLIC),
        ),
    )
    print("Physical and prediction replay passed", errors, flush=True)


def infer(a, bank, lock):
    registration = a.out / "references/fresh_ablation.json"
    ref = read(registration)
    pack_sha = {}
    results = {}
    for sub in SUBJECTS:
        eeg = a.out / "eeg" / (sub + ".npz")
        raw = read(eeg.with_suffix(".json"))
        require(
            raw["registration_sha256"] == sha(registration) and raw["packet_sha256"] == sha(eeg),
            "Unverified target EEG",
        )
        rows, _, _, mean = load(eeg)
        pred, data, base = apply(a, sub, ref, bank, rows, mean)
        word_file = a.out / "evaluations" / sub / "fresh/word_predictions.json"
        freeze(
            word_file,
            dict(
                subject=sub,
                rows=ref["rows"],
                folds=ref["folds"],
                predicted=pred.tolist(),
                words=ref["words"],
                training_receipt_sha256=sha(a.expanded / "features" / (sub + ".json")),
                target_sha256=sha(eeg),
            ),
        )
        dest = a.out / "conditions" / sub / "fresh/conditions_512.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            dest,
            **data,
            target_eeg_sha256=sha(eeg),
            registration_sha256=sha(registration),
            word_prediction_sha256=sha(word_file),
            angular_scale_sha256=sha(a.angle / "scale.json"),
            public_lock_sha256=sha(PUBLIC),
            unrotated_own_sha256=digest(base.tobytes()),
        )
        pack_sha[sub + "/fresh"] = sha(dest)
        results[sub] = (pred, word_file)
        print("Frozen conditions", sub, pack_sha[sub + "/fresh"], flush=True)
    original = read(a.angle / "design.json")
    design = {
        k: original[k]
        for k in (
            "protocol",
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
    freeze(
        a.out / "design.json",
        design
        | dict(
            cohort=list(SUBJECTS),
            panels=["fresh"],
            packets_sha256=pack_sha,
            registration_sha256=sha(registration),
            public_lock_sha256=sha(PUBLIC),
            original_design_sha256=sha(a.angle / "design.json"),
        ),
    )
    radius = a.out / "radius.json"
    if not radius.exists():
        shutil.copyfile(a.angle / "radius.json", radius)
    require(sha(radius) == lock["inputs_sha256"]["radius"], "Render radius changed")
    # Only now use the true query classes for diagnostic EEG accuracy. The generation packets already exist.
    for sub, (pred, word_file) in results.items():
        packet(a.out, sub, "fresh")
        hits = int((pred == np.asarray(ref["truth"])).sum())
        freeze(
            word_file.with_name("evaluation.json"),
            {k: ref[k] for k in ("rows", "folds", "words", "truth")}
            | dict(
                subject=sub,
                results={"expanded_r4": dict(n=len(pred), hits=hits, predicted=pred.tolist())},
                conditions_sha256=pack_sha[sub + "/fresh"],
                word_prediction_sha256=sha(word_file),
                registration_sha256=sha(registration),
            ),
        )
        print(sub, "12-way EEG", hits, "/120", flush=True)


def verify(a):
    """Validate all output links and derive descriptive classification statistics after freezing."""
    ref = read(a.out / "references/fresh_ablation.json")
    registration = sha(a.out / "references/fresh_ablation.json")
    schedule = read(a.inputs / "sessions.json")["subjects"]
    truth = np.array(ref["truth"])
    folds = np.array(ref["folds"])
    sampled = np.random.default_rng(260510).integers(0, 12, size=(10000, 12))
    summaries = {}
    for sub in SUBJECTS:
        rows, _, words, _, _, _, _, packet_sha = packet(a.out, sub, "fresh")
        cond = a.out / "conditions" / sub / "fresh/conditions_512.npz"
        eeg = a.out / "eeg" / (sub + ".npz")
        features = a.out / "target" / eeg.name
        evaluation = a.out / "evaluations" / sub / "fresh/evaluation.json"
        wordfile = evaluation.with_name("word_predictions.json")
        report = read(evaluation)
        require(
            report["subject"] == sub
            and all(report[k] == ref[k] for k in ("rows", "truth", "words", "folds"))
            and report["registration_sha256"] == registration
            and report["conditions_sha256"] == packet_sha
            and report["word_prediction_sha256"] == sha(wordfile),
            "Evaluation disagrees with frozen manifest",
        )
        with np.load(cond, allow_pickle=False) as c, np.load(features, allow_pickle=False) as f:
            reg_rows, _, _, mean = load(eeg)
            require(
                str(c["target_eeg_sha256"]) == str(f["source_sha256"]) == sha(eeg)
                and str(f["registration_sha256"]) == str(c["registration_sha256"]) == registration
                and str(c["word_prediction_sha256"]) == sha(wordfile)
                and str(f["subject"]) == sub
                and f["channels"].tolist() == CHANNELS
                and np.array_equal(f["rows"], reg_rows)
                and np.array_equal(f["mean_eeg"], mean)
                and np.array_equal(f["features"], extract(mean))
                and f["features"].shape == (120, 1003)
                and str(f["feature_sha"]) == digest(f["features"].tobytes()),
                "Derived EEG features, conditions or source disagree",
            )
        pred = np.array(report["results"]["expanded_r4"]["predicted"])
        hit = pred == truth
        counts = np.bincount(truth, weights=hit, minlength=12)
        require(
            read(wordfile)["predicted"] == pred.tolist()
            and np.array_equal(np.array(ref["words"])[pred], words)
            and np.array_equal(rows, ref["rows"])
            and hit.sum() == report["results"]["expanded_r4"]["hits"],
            "Published EEG counts differ from generated words",
        )
        strict = sum(
            bool(hit[(folds == i) & (truth == j)].all()) for i in range(5) for j in range(12)
        )
        session = audit_panel(schedule[sub], report)["methods"]["expanded_r4"]
        summaries[sub] = dict(
            n=120,
            hits=int(hit.sum()),
            rate=float(hit.mean()),
            rate_ci95=np.quantile(counts[sampled].mean(1) / 10, [0.025, 0.975]).tolist(),
            strict_pairs_both_words_correct=strict,
            possible_pairs=60,
            session_conditioned_expected_hits=session["conditional_expected_hits"],
            session_conditioned_excess_hits=float(hit.sum()) - session["conditional_expected_hits"],
            session_conditioned_label_permutation_p=session["permutation_one_sided_p"],
            source_eeg_sha256=sha(eeg),
            derived_feature_packet_sha256=sha(features),
            condition_sha256=packet_sha,
            predictions_sha256=sha(wordfile),
            evaluation_sha256=sha(evaluation),
        )
    replayed = read(a.out / "replay.json")
    result = dict(
        method="Frozen 512-concept personal Fisher decoder, 12 object choices, four physical trials per photograph, eight other labeled photographs per object in a fold.",
        sample="Each of three participants viewed the same 120 distinct new photos. Report each participant's 120 queries separately.",
        protocol_sha256=sha(PUBLIC),
        registration_sha256=registration,
        session_schedule_sha256=sha(a.inputs / "sessions.json"),
        uncertainty="Within each fixed participant, resample 12 whole objects with replacement 10,000 times (seed 260510). Rates and percentile 95% limits use fractions. The secondary session null permutes whole object labels within their published session pairs.",
        uniform_expected_hits=10,
        uniform_rate=1 / 12,
        participants=summaries,
        historical_replay={
            k: replayed[k]
            for k in (
                "subject",
                "panel",
                "all_240_waveforms_bit_identical",
                "all_120_prediction_ids_identical",
                "word_hits",
                "max_abs_h14_errors",
            )
        },
        replay_sha256=sha(a.out / "replay.json"),
    )
    freeze(a.out / "classification_summary.json", result)
    freeze(a.release_file, result)
    print(
        "Verified all three original EEG bundles, compact features, predictions and render packets",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("all", "replay", "fetch", "infer", "verify"))
    for key, value in dict(
        out="outputs/fresh-photo",
        within="outputs/photo-structure/within_photo",
        expanded="outputs/photo-structure/expanded",
        angle="outputs/photo-structure/residual_angle",
        inputs="outputs/object-transfer",
        references="outputs/object-fidelity",
        source="outputs/official-object-readout/sub-03",
    ).items():
        p.add_argument("--" + key, type=Path, default=Path(value))
    p.add_argument("--workers", type=int, choices=range(1, 9), default=5)
    p.add_argument("--release-file", type=Path, default=PUBLIC.with_name("fresh_panel_eeg.json"))
    a = p.parse_args()
    lock = verify_lock(a)
    bank = images(a.source) if a.stage in ("all", "replay", "infer") else None
    if a.stage in ("all", "replay"):
        replay(a, bank)
    if a.stage != "replay":
        passed = read(a.out / "replay.json")
        require(
            passed["all_120_prediction_ids_identical"]
            and passed["all_240_waveforms_bit_identical"]
            and passed["public_lock_sha256"] == sha(PUBLIC),
            "Source replay must pass first",
        )
    if a.stage in ("all", "fetch"):
        rows = np.array(read(a.out / "references/fresh_ablation.json")["rows"])
        for sub in SUBJECTS:
            physical(a, sub, rows, False)
            print(
                "Verified",
                sub,
                len(rows) * 4,
                "physical trials and exported 120 compact EEG feature rows",
                flush=True,
            )
    if a.stage in ("all", "infer"):
        infer(a, bank, lock)
    if a.stage in ("all", "verify"):
        verify(a)
