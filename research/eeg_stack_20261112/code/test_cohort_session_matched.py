"""Register and test a 12-way stress panel and three strict four-way same-session blocks."""

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from decode_eeg_layout import CHANNELS
from fit_fresh_photo_panel import fisher
from fit_personal_session_metric import similarities
from fit_shared_photo_eeg import load_photos, read, require, sha, write
from fuse_same_stimulus_r4 import PEOPLE
from prepare_expanded_personal import training_block
from prepare_registered_subjects import OFFSET, REV, ROW_BYTES, SHAPE, SIZE, axes, digest

SEED = 260533
NAMES = ("stress12", "new_three4")


def register(a):
    sd = a.double / "design.json"
    d = read(sd)
    schedule = read(a.inputs / "sessions.json")["subjects"]
    prior = set(d["old_objects"])
    extra = set(d["extra_objects"])
    unused = set(d["reserve_objects"]) - set(d["fifth_objects"])
    require(
        len(prior) == 512
        and len(extra) == 512
        and len(unused) == 76
        and not (unused & extra or unused & prior or extra & prior),
        "Double-calibration population differs",
    )
    groups = {}
    for k, cl in (("stress12", extra), ("new_three4", unused)):
        grouped = defaultdict(list)
        for c in sorted(cl):
            grouped[tuple(v for s in PEOPLE for v in schedule[s]["session_pairs"][c])].append(c)
        groups[k] = sorted(grouped.items(), key=lambda x: (-len(x[1]), x[0]))
    rng = np.random.default_rng(SEED)
    data = {}
    with (a.source / "atm_train_rows.csv").open(encoding="utf8", newline="") as f:
        table = list(csv.DictReader(f))
    for name, n, k, provenance in [
        (
            "stress12",
            1,
            12,
            "These objects belong to the separate method's 512 added training classes. The frozen original 512 participant encoders have never used them.",
        ),
        (
            "new_three4",
            3,
            4,
            "Strict leftover generic frame: 76 outside all 1024 training objects, all four older panels and all 12 fixed fifth-panel objects. No new EEG was read before this registration.",
        ),
    ]:
        pools = groups[name][:n]
        require(all(len(cs) >= k for _, cs in pools), "Matching frame too small")
        chosen = [sorted(rng.choice(cs, k, replace=False).tolist()) for _, cs in pools]
        ids = np.concatenate(chosen).astype(int)
        order = np.stack([10 * c + rng.permutation(10) for c in ids]).reshape(12, 5, 2)
        rows = order.transpose(1, 0, 2).ravel()
        words = [table[10 * c]["unique_id"] for c in ids]
        truth = np.tile(np.repeat(np.arange(12), 2), 5)
        require(
            all(
                table[int(r)]["unique_id"] == words[t] and int(table[int(r)]["train_row"]) == r
                for r, t in zip(rows, truth, strict=True)
            ),
            "Photo identity mismatch",
        )
        data[name] = dict(
            scope=provenance,
            n_words_per_primary_decision=k,
            selected_concepts=ids.tolist(),
            matched_cells=[
                dict(joint_session_ids=list(key), source_cell_size=len(pool), selected_concepts=v)
                for (key, pool), v in zip(pools, chosen, strict=True)
            ],
            registration=dict(
                words=words,
                rows=rows.tolist(),
                folds=np.repeat(np.arange(5), 24).tolist(),
                truth=truth.tolist(),
                calibration_rows_by_fold_and_word=[
                    order[:, np.arange(5) != fold].reshape(12, 8).tolist() for fold in range(5)
                ],
            ),
        )
    return dict(
        subject="cohort-3",
        contributing_subjects=list(PEOPLE),
        physical_trials_per_photo=12,
        subject_trials=4,
        selection_seed=SEED,
        rules="Take the largest available cell(s); ties sort by lexicographic six-integer session tuple in sub-04, sub-05, sub-01 order. Sample without replacement within each cell and draw the five two-photo folds before acquisition. Panel 1 uses one 12-way decision universe; panel 2 uses three disjoint four-way universes with no cross-block competitors. Also report the new strict panel's ordinary 12-class result separately.",
        analysis="Original 512-class 17-posterior-electrode Fisher only. Compute cosines to the eight calibration EEGs per eligible candidate; standardize within the 12 or 4 candidates for each participant and query, then sum equally across the three same-row participants. No model fit or tuning on any target. Report paired comparisons against fixed sub-04, average of the three single-person decisions and two misaligned participants. Resample 12 complete true objects, 10000 draws, seed 260532. These people and session tuples are fixed.",
        misalignment="Sub-04 remains aligned; within each allowed candidate block and held-out fold, sub-05 and sub-01 are cyclically shifted over sorted query row IDs by distinct nonzero offsets with seed 260534. The support EEG stays aligned.",
        schedule_sha256=sha(a.inputs / "sessions.json"),
        generic_1024_design_sha256=sha(sd),
        cohort_original_model_design_sha256=sha(a.cohort / "design.json"),
        manifest_sha256=sha(a.source / "atm_train_rows.csv"),
        fifth_classes_untouched=d["fifth_objects"],
        populations={
            k: dict(n=sum(len(v) for _, v in sets), max_joint_cell=len(sets[0][1]))
            for k, sets in groups.items()
        },
        panels=data,
    )


def get_strict(a, d):
    ids = d["panels"]["new_three4"]["selected_concepts"]
    for s in PEOPLE:
        meta = read(a.inputs / (s + ".json"))
        cache = a.inputs / "ranges" / REV / s
        header = cache / f"0-{OFFSET - 1}.bin"
        footer = cache / f"{OFFSET + SHAPE[0] * ROW_BYTES}-{SIZE - 1}.bin"
        require(
            sha(header) == meta["header_sha256"]
            and sha(footer) == meta["tail_sha256"]
            and meta["revision"] == REV,
            "Published physical axes changed",
        )
        channels, times = axes(header.read_bytes(), footer.read_bytes())
        require(
            np.allclose(times[50:], np.arange(250) / 250, atol=1e-7, rtol=0), "Time axis mismatch"
        )
        dest = a.out / "features" / s
        dest.mkdir(parents=True, exist_ok=True)
        post = [channels.tolist().index(ch) for ch in CHANNELS]
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(training_block, c, dest, cache, meta["source"], post) for c in ids]
            raw = {str(c): j.result()[1] for c, j in zip(ids, jobs, strict=True)}
        write(
            dest.with_suffix(".json"),
            dict(
                subject=s,
                source=meta["source"],
                revision=REV,
                official_full_source_sha256=meta["source_full_lfs_sha256"],
                raw_sha256=raw,
                feature_files_sha256={str(c): sha(dest / f"{c:04d}.npz") for c in ids},
                design_sha256=sha(a.out / "design.json"),
                bytes_per_class=10 * ROW_BYTES,
            ),
        )
        print("Verified strict participant range blocks", s, flush=True)


def compact(a, s, name, classes):
    root = (a.double if name == "stress12" else a.out) / "features"
    receipt = read(root / (s + ".json"))
    values = []
    for c in classes:
        file = root / s / f"{c:04d}.npz"
        with np.load(file, allow_pickle=False) as z:
            x = z["features"]
            require(
                x.shape == (10, 1003)
                and x.dtype == np.float32
                and int(z["start"]) == 10 * c
                and z["source"].item() == receipt["source"]
                and z["raw_sha"].item() == receipt["raw_sha256"][str(c)]
                and digest(x.tobytes()) == z["feature_sha"].item(),
                "Broken physical compact feature",
            )
            if name == "stress12":
                require(
                    receipt["feature_sha256"][str(c)] == z["feature_sha"].item(),
                    "Legacy compact receipt mismatch",
                )
            else:
                require(
                    sha(file) == receipt["feature_files_sha256"][str(c)],
                    "Strict compact receipt mismatch",
                )
            values.append(x)
    return np.concatenate(values).astype(float), sha(root / (s + ".json"))


def decision(features, source_rows, reg, blocks):
    rows, fold, cal = (
        np.array(reg["rows"]),
        np.array(reg["folds"]),
        np.array(reg["calibration_rows_by_fold_and_word"]),
    )
    scores = np.full((len(PEOPLE), len(rows), 12), -np.inf)
    shuffled = scores.copy()
    rng = np.random.default_rng(260534)
    for f in range(5):
        for block in blocks:
            # The preassigned candidate group is specified before collecting EEG. Only the
            # candidate group (four or twelve alternatives), never the within-group answer, routes the query.
            block_concepts = np.unique(cal[f, block] // 10)
            ix = np.flatnonzero((fold == f) & np.isin(rows // 10, block_concepts))
            ix = ix[np.argsort(rows[ix])]
            sup = np.searchsorted(source_rows, cal[f, block])
            query = np.searchsorted(source_rows, rows[ix])
            require(
                len(ix) == 2 * len(block)
                and not np.intersect1d(cal[f, block], rows[ix]).size
                and np.array_equal(source_rows[sup], cal[f, block]),
                "Block/fold support mismatch",
            )
            shifts = rng.choice(np.arange(1, len(ix)), 2, replace=False)
            for j, s in enumerate(PEOPLE):
                x = features[s]
                v = similarities(x[query], x[sup])
                scores[j][np.ix_(ix, block)] = v
                shuffled[j][np.ix_(ix, block)] = v if j == 0 else np.roll(v, shifts[j - 1], axis=0)
    require(np.all(np.isfinite(scores).sum(axis=2) == len(blocks[0])), "Incorrect candidate count")
    return dict(
        cohort=scores.sum(0).argmax(1),
        misaligned=shuffled.sum(0).argmax(1),
        **{s: v.argmax(1) for s, v in zip(PEOPLE, scores, strict=True)},
    )


def score(a, d, name):
    sample = d["panels"][name]
    concept = np.array(sample["selected_concepts"])
    reg = sample["registration"]
    flat_rows = (concept[:, None] * 10 + np.arange(10)).ravel()
    order = np.argsort(flat_rows)
    source_rows = flat_rows[order]
    features = {}
    receipts = {}
    old = np.array(read(a.double / "design.json")["old_objects"])
    require(not np.intersect1d(old, concept).size, "Test in original 512 fit")
    for s in PEOPLE:
        meta = read(a.expanded / "features" / (s + ".json"))
        require(np.array_equal(meta["training_concepts"], old), "Changed personal classifier fit")
        x = load_photos(a, s, old)
        xt, receipts[s] = compact(a, s, name, concept)
        features[s] = fisher(x, xt[order])
    configure = {"ordinary_12way": [np.arange(12)]}
    if name == "new_three4":
        configure["within_matched_four"] = [np.arange(j, j + 4) for j in range(0, 12, 4)]
    outcomes = {k: decision(features, source_rows, reg, groups) for k, groups in configure.items()}
    dest = a.out / name
    dest.mkdir(exist_ok=True, parents=True)
    raw = dest / "predictions.npz"
    np.savez(
        raw,
        rows=np.array(reg["rows"]),
        folds=np.array(reg["folds"]),
        **{k + "_" + s: v for k, pred in outcomes.items() for s, v in pred.items()},
    )
    truth = np.array(reg["truth"])
    data = {}
    for k, pred in outcomes.items():
        denom = len(configure[k][0])
        cond = {s: (v == truth).astype(float) for s, v in pred.items()}
        cond["mean_single"] = np.mean([cond[s] for s in PEOPLE], axis=0)
        total = {}
        for s, v in cond.items():
            by = np.array([v[truth == i].mean() for i in range(12)])
            idx = np.random.default_rng(260532).integers(12, size=(10000, 12))
            total[s] = dict(
                hits=float(v.sum()),
                accuracy=float(v.mean()),
                object_bootstrap_ci95=np.percentile(by[idx].mean(1), [2.5, 97.5]).tolist(),
            )
        contrasts = {}
        for s in ("sub-04", "mean_single", "misaligned"):
            v = cond["cohort"] - cond[s]
            by = np.array([v[truth == i].mean() for i in range(12)])
            contrasts[s] = dict(
                delta_hits=float(v.sum()),
                mean=float(v.mean()),
                object_bootstrap_ci95=np.percentile(by[idx].mean(1), [2.5, 97.5]).tolist(),
            )
        data[k] = dict(
            n=120,
            candidates=denom,
            uniform_chance=1 / denom,
            session_classifier_expected_hits=10 if name == "stress12" else 30,
            observed=total,
            cohort_minus=contrasts,
            correct_by_true_object=[int(cond["cohort"][truth == i].sum()) for i in range(12)],
        )
    write(
        dest / "result.json",
        dict(
            subject="cohort-3",
            contributing_subjects=list(PEOPLE),
            physical_trials_per_photo=12,
            individual_trials_per_photo=4,
            panel=name,
            scope=sample["scope"],
            primary="ordinary_12way" if name == "stress12" else "within_matched_four",
            n_unique_photos=120,
            session_cells=sample["matched_cells"],
            design_sha256=sha(a.out / "design.json"),
            target_feature_receipts=receipts,
            predictions_sha256=sha(raw),
            driver_sha256=sha(Path(__file__)),
            decisions=data,
        ),
    )
    print(
        name,
        {k: {s: q["hits"] for s, q in v["observed"].items()} for k, v in data.items()},
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("register", "fetch", "score"))
    for k, v in dict(
        out="outputs/session-matched-cohort",
        double="outputs/double-calibration",
        inputs="outputs/object-transfer",
        expanded="outputs/photo-structure/expanded",
        source="outputs/official-object-readout/sub-03",
        cohort="outputs/cohort-r4",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    p.add_argument("--panel", choices=NAMES)
    a = p.parse_args()
    d = register(a)
    if a.stage == "register":
        write(a.out / "design.json", d)
        print("Locked", sha(a.out / "design.json"))
    else:
        require(read(a.out / "design.json") == d, "Different registration")
        if a.stage == "fetch":
            get_strict(a, d)
        else:
            for k in [a.panel] if a.panel else NAMES:
                score(a, d, k)


if __name__ == "__main__":
    main()
