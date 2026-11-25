"""Aggregate original-RGB reconstruction scores on a single acquisition budget and panel.

Internal arms live in ROOT/images/TRAINING/ARM/BUDGET/sub-XX. An explicit selector
such as personal:native-paper can be compared with shared:d2. The released ATM
draw zero lives in ROOT/external/atm-subXX/{manifest.csv,scores.json,observations.npz}
and can be selected with --arms atm --training external --budget official-r80.
Use native-full:native-paper or native-full:native-code for a single-person fit
on all 1,654 training concepts. Its evaluations use official photographs. The
dev-r4 panel has 20 development concepts and its primary registered cohort is
listed in ROOT/data/prep/paper_development.json.

An optional --external-manifest uses folders relative to its JSON file.
{"arm":"atm","budget":"official-r80","training":"released ATM",
 "participants":{"sub-01":{"folder":"external/atm-sub01","manifest":"manifest.csv"}}}
Each entry may also pin manifest_sha256, scores_sha256 and observations_sha256.
The CSV identifies subject, eeg_row, unique_id, draw and reconstruction_sha256.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUBJECTS = tuple(f"sub-{i:02}" for i in range(1, 11))
TRACKS = ("personal", "shared", "native-full", "external")
BUDGETS = ("official-r4", "official-r4-balanced", "official-r80", "fresh-r4", "dev-r4")
PANELS = {"official": (200, 200), "fresh": (500, 50), "dev": (200, 20)}
FACTOR_CONDITION = {"native-layout": "native-paper", "d2-no-layout": "d2"}
INSTANCE_ARMS = ("instance-native", "instance-direct", "instance-residual")
NO_LAYOUT = ("native-code", "native-paper", "d2-no-layout", *INSTANCE_ARMS)
COMMON_METRICS = ("pixcorr", "ssim", "alex2_2way", "alex5_2way", "inception_2way", "clip_2way",
                  "effnet_distance", "swav_distance", "clip_image_top1", "generated_name1854",
                  "original_name1854")
FRESH_METRICS = ("clip_within10", "clip_within_swap")
RENDER_FIELDS = ("n", "prompt", "draws", "unet_calls_per_draw", "repeats", "row", "eeg_donor",
                 "seeds", "checkpoint_sha256", "split_sha256", "pixels_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def subject_id(value):
    s = str(value).removeprefix("sub-")
    require(s.isascii() and s.isdecimal() and 1 <= int(s) <= 10, f"Invalid participant {value!r}")
    return f"sub-{int(s):02}"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def row_order(rows, n):
    rows = np.asarray(rows)
    require(rows.shape == (n,) and rows.dtype.kind in "iu" and np.all(rows >= 0)
            and np.unique(rows).size == n, f"Expected {n} distinct nonnegative integer stimulus IDs")
    return np.argsort(rows)


def strings(array, n, field):
    require(array.shape == (n,) and array.dtype.kind in "SU", f"Invalid text column {field}")
    s = array.astype(str)
    require(np.all(s != ""), f"Empty value in {field}")
    return s


def registration(root, panel):
    """Load the independent row/label registrations once, before opening any scores."""
    prep = root / "data/prep"
    info = {"hashes": {}, "split": None, "development": None, "rows": None, "truth": None}
    for key, name in [("split", "paper_split.json"), ("development", "paper_development.json")]:
        p = prep / name
        if p.is_file() and (key == "split" or panel == "dev"):
            raw = p.read_bytes()
            info[key] = json.loads(raw)
            info["hashes"][name] = sha(raw)
    ids_name = "test_ids.json" if panel == "official" else "train_ids.json"
    ids_path = prep / ids_name
    cids = None
    if ids_path.is_file():
        raw = ids_path.read_bytes()
        ids = json.loads(raw)
        paths = [Path(s) for s in ids["paths"]]
        size = 200 if panel == "official" else 16540
        require(len(paths) == len(set(ids["ids"])) == size and [p.stem for p in paths] == ids["ids"],
                f"Invalid image identities in {ids_name}")
        folders = [p.parent.name for p in paths]
        require(all(re.fullmatch(r"\d+_.+", f) for f in folders), f"Invalid concept folder in {ids_name}")
        info["truth"] = np.array([f.split("_", 1)[1] for f in folders])
        mapping = {name: i for i, name in enumerate(dict.fromkeys(folders))}
        require(len(mapping) == len(set(info["truth"])), "Ambiguous concept names in the image registration")
        if panel != "official":
            cids = np.array([mapping[f] for f in folders])
        info["hashes"][ids_name] = sha(raw)
    cpath = prep / "train_concept.npy"
    if cpath.is_file():
        raw = cpath.read_bytes()
        index = np.load(io.BytesIO(raw), allow_pickle=False)
        require(index.shape == (16540,) and index.dtype.kind in "iu", "Invalid training concept index")
        require(cids is None or np.array_equal(index, cids), "Training concept index disagrees with train_ids.json")
        cids = index
        info["hashes"][cpath.name] = sha(raw)
    if cids is not None:
        ids, counts = np.unique(cids, return_counts=True)
        require(np.array_equal(ids, np.arange(1654)) and np.all(counts == 10), "Invalid THINGS training concept coverage")
    split, dev = info["split"], info["development"]
    if split is not None:
        require(np.array_equal(split["official_rows"], np.arange(200)), "Wrong official gallery in paper_split.json")
        fit = np.asarray(split["train_rows"])
        row_order(fit, len(fit))
        require(len(fit) > 0 and fit.max() < 16540, "Invalid train rows in paper_split.json")
        if cids is not None:
            require(len(np.unique(cids[fit])) == 1323, "Restricted training split must contain 1323 concepts")
        fresh_rows = np.asarray(split["fresh_rows"])
        row_order(fresh_rows, 500)
        fresh_concepts = set(split["fresh_concepts"])
        require(fresh_rows.max() < 16540 and len(fresh_concepts) == 50
                and not set(fresh_rows).intersection(fit), "Invalid fresh holdout or overlap with training rows")
        if cids is not None:
            require(set(cids[fresh_rows]) == fresh_concepts and not fresh_concepts.intersection(cids[fit]),
                    "Fresh concept identities disagree with held-out training concepts")
        if panel != "dev":
            info["rows"] = np.asarray(split[panel + "_rows"])
    if dev is not None:
        rows = np.asarray(dev["dev_rows"])
        row_order(rows, 200)
        concepts = set(dev["concepts"])
        require(rows.max() < 16540 and len(concepts) == 20 and dev["seed"] == 287311,
                "Invalid development gallery or registration seed")
        if split is not None:
            blocked = fresh_concepts | set(split["excluded_possible_historical_concepts"])
            require(not concepts.intersection(blocked) and not set(rows).intersection(fit)
                    and not set(rows).intersection(fresh_rows), "Development overlaps training, fresh or historical concepts")
            if cids is not None:
                require(set(cids[rows]) == concepts and not concepts.intersection(cids[fit]),
                        "Development concept identities disagree with held-out training concepts")
        info["rows"] = rows
    return info


def check_gallery(reg, rows, labels, panel, arm):
    if panel == "dev" or arm["track"] == "native-full":
        require(reg["split"] is not None and reg["rows"] is not None and reg["truth"] is not None,
                f"Missing independent split, gallery or original image IDs for {panel} {arm['track']}")
    if reg["rows"] is not None:
        require(np.array_equal(np.sort(reg["rows"]), rows), "Score rows disagree with the registered gallery")
    if reg["truth"] is not None:
        require(np.all(rows < len(reg["truth"])) and np.array_equal(reg["truth"][rows], labels),
                "Scored reference concept disagrees with the independent original image ID")


def panel_seeds(panel, n):
    return np.array([int.from_bytes(hashlib.sha256(f"paper-20261125/{panel}/{i}".encode()).digest()[:8],
                                    "big") % (2**63 - 1) for i in range(n)], dtype=np.int64)


def read_scores(folder, panel):
    n, concept_count = PANELS[panel]
    raw = (folder / "scores.json").read_bytes()
    receipt = json.loads(raw)
    require(receipt.get("raw_RGB") is True and receipt.get("panel") == panel and receipt.get("n") == n,
            f"Expected original RGB scores for {panel} with {n} stimuli")
    metrics = COMMON_METRICS
    if panel != "official":
        metrics += FRESH_METRICS
        sensitivity = (f"clip_global{n}_centered", "clip_within10_uncentered")
        if panel == "dev" or any(m in receipt.get("mean", {}) for m in sensitivity):
            metrics += sensitivity
    blob = (folder / "observations.npz").read_bytes()
    digest = sha(blob)
    require(receipt.get("observations_sha256") == digest, "Observations SHA-256 differs from scores.json")
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        required = {"row", "concept", "generated_top_word", "original_top_word", *metrics}
        require(required <= set(data.files), f"Missing observation columns {sorted(required - set(data.files))}")
        require(set(data.files) == required, f"Observation columns differ from score schema: {sorted(set(data.files) - required)}")
        require(set(receipt.get("mean", {})) == set(metrics), "Metric list differs from the scorer schema")
        order = row_order(data["row"], n)
        rows = data["row"][order]
        if panel == "official":
            require(np.array_equal(rows, np.arange(200)), "Official row IDs must be 0 through 199")
        labels = strings(data["concept"], n, "concept")[order]
        words = strings(data["original_top_word"], n, "original_top_word")[order]
        strings(data["generated_top_word"], n, "generated_top_word")
        counts = np.unique(labels, return_counts=True)[1]
        require(len(counts) == concept_count and np.all(counts == n // concept_count), "Unexpected gallery or photos per concept")
        cols = []
        for m in metrics:
            v = data[m]
            require(v.shape == (n,) and v.dtype.kind in "biuf" and np.isfinite(v).all(),
                    f"Invalid per-stimulus values for {m}")
            lo, hi = (-1, 1) if m in ("pixcorr", "ssim") else (0, 2 if m.endswith("_distance") else 1)
            require(np.all((lo - 1e-7 <= v) & (v <= hi + 1e-7)), f"Values outside the scale of {m}")
            require(np.isclose(v.mean(dtype=np.float64), receipt["mean"][m], rtol=1e-6, atol=1e-8),
                    f"Per-stimulus mean disagrees with scores.json for {m}")
            cols.append(v[order])
    sources = {"scores_sha256": sha(raw), "observations_sha256": digest}
    if "scorer_sha256" in receipt:
        require(valid_sha(receipt["scorer_sha256"]), "Invalid scorer hash")
        sources["scorer_sha256"] = receipt["scorer_sha256"]
    return metrics, rows, labels, words, np.column_stack(cols).astype(np.float64), receipt, sources


def verify_pixels(folder, filenames, hashes):
    for name, expected in zip(filenames, hashes, strict=True):
        with (folder / name).open("rb") as f:
            require(hashlib.file_digest(f, "sha256").hexdigest() == expected,
                    f"Generated image SHA-256 differs at {folder / name}")


def check_scored_images(receipt, hashes, source):
    if "ordered_image_hashes_sha256" in receipt:
        digest = sha("\n".join(hashes).encode())
        require(receipt["ordered_image_hashes_sha256"] == digest, "Scored image hashes differ from the source")
        source["ordered_image_hashes_sha256"] = digest


def internal_source(folder, arm, sub, budget, rows, labels, receipt, reg, verify):
    raw = (folder / "render.json").read_bytes()
    meta = json.loads(raw)
    digest = sha(raw)
    require(meta.get("arm") == arm["method"] and meta.get("budget") == budget
            and meta.get("training") == arm["track"] and subject_id(meta.get("subject")) == sub,
            "Render identity disagrees with participant, arm, training track or budget")
    if "render_sha256" in receipt:
        require(receipt["render_sha256"] == digest, "Scorer's render hash differs from render.json")
    panel = budget.split("-")[0]
    if panel == "dev" or arm["track"] == "native-full":
        require(all(k in meta for k in ("row", "seeds", "split_sha256")), "Missing registered render rows, seeds or split hash")
    n = len(rows)
    if budget == "official-r80":
        repeats = np.arange(80).reshape(20, 4).tolist()
    else:
        repeats = [[0, 20, 40, 60]] if budget == "official-r4-balanced" else [[0, 1, 2, 3]]
    for key, expected in dict(n=n, prompt="", draws=1, unet_calls_per_draw=4, repeats=repeats).items():
        require(key not in meta or meta[key] == expected, f"Unexpected render {key} for {budget}")
    src = {"render_sha256": digest, "render_bound_by_scorer": "render_sha256" in receipt,
           "missing_render_metadata": [k for k in RENDER_FIELDS if k not in meta],
           "pixels_locally_verified": 0}
    raw_rows = np.asarray(meta.get("row", rows))
    order = row_order(raw_rows, n)
    require(np.array_equal(raw_rows[order], rows), "Render and score row IDs disagree")
    for key in ("split_sha256", "checkpoint_sha256"):
        if key in meta:
            require(valid_sha(meta[key]), f"Invalid render {key}")
            src[key] = meta[key]
    if reg["split"] is not None and "split_sha256" in meta:
        require(meta["split_sha256"] == reg["hashes"]["paper_split.json"], "Wrong registered split hash")
    if meta.get("development_split_sha256") is not None:
        require(panel == "dev" and meta["development_split_sha256"] == reg["hashes"].get("paper_development.json"),
                "Wrong development split hash or evaluation panel")
        src["development_split_sha256"] = meta["development_split_sha256"]
    for key in ("semantic_condition", "layout_mode"):
        if key in meta:
            require(meta[key] == arm[key], f"Render {key} disagrees with component {arm['method']}")
            src[key] = meta[key]
    if "eeg_donor" in meta:
        donor = np.asarray(meta["eeg_donor"])
        require(donor.dtype.kind in "iu" and np.array_equal(np.sort(donor), np.arange(n)),
                "EEG donor indices are not a permutation")
        if arm["method"] == "wrong":
            require(np.all(donor != np.arange(n)), "Wrong-EEG control contains its own EEG")
            raw_labels = labels[np.argsort(order)]
            require(panel == "official" or np.array_equal(raw_labels, raw_labels[donor]),
                    "Within-concept wrong-EEG control crosses concepts")
        else:
            require(np.array_equal(donor, np.arange(n)), "Unexpected EEG donor for aligned arm")
    seeds = None
    if "seeds" in meta:
        seeds = np.asarray(meta["seeds"])
        require(seeds.shape == (n,) and seeds.dtype.kind in "iu" and len(np.unique(seeds)) == n,
                "Expected one distinct seed per stimulus")
        require("row" in meta, "Seed-to-stimulus mapping missing")
        seeds = seeds[order]
        registered = reg["rows"] if reg["rows"] is not None else rows
        require(np.array_equal(seeds, panel_seeds(panel, n)[np.argsort(registered)]),
                f"Render seeds disagree with the stimulus-blind {panel} seed schedule")
        src["panel_seed_schedule_verified"] = True
    if "pixels_sha256" in meta or verify or "ordered_image_hashes_sha256" in receipt:
        hashes = meta.get("pixels_sha256", [])
        require(isinstance(hashes, list) and len(hashes) == n and all(valid_sha(h) for h in hashes),
                "Incomplete generated pixel hash list")
        check_scored_images(receipt, hashes, src)
        if verify:
            verify_pixels(folder, [f"{i:05d}.png" for i in range(n)], hashes)
            src["pixels_locally_verified"] = n
    return src, seeds, None


def external_source(folder, manifest, sub, rows, labels, receipt, verify):
    raw = manifest.read_bytes()
    digest = sha(raw)
    if "source_manifest_sha256" in receipt:
        require(receipt["source_manifest_sha256"] == digest, "Scorer's external manifest hash differs")
    records = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    columns = {"subject", "eeg_row", "unique_id", "draw", "reconstruction_sha256"}
    require(len(records) == len(rows) and columns <= set(records[0]), "Incomplete external ATM manifest")
    order = row_order([int(r["eeg_row"]) for r in records], len(rows))
    records = [records[i] for i in order]
    reference = []
    for r, row, label in zip(records, rows, labels, strict=True):
        require(subject_id(r["subject"]) == sub and int(r["eeg_row"]) == row and r["unique_id"] == label
                and r["draw"] == "0" and valid_sha(r["reconstruction_sha256"]),
                f"External manifest disagrees with scored participant, stimulus or draw at {row}")
        if "reference_sha256" in r:
            require(valid_sha(r["reference_sha256"]), "Invalid reference hash in external manifest")
            reference.append(r["reference_sha256"])
    hashes = [r["reconstruction_sha256"] for r in records]
    info = {"manifest_sha256": digest, "manifest_bound_by_scorer": "source_manifest_sha256" in receipt,
            "released_draw_index": 0, "pixels_locally_verified": len(rows) if verify else 0}
    check_scored_images(receipt, hashes, info)
    if verify:
        names = []
        for r in records:
            named = Path(r.get("filename") or f"{int(r['eeg_row']):05d}.png")
            require(named.name == str(named), "Expected a local image filename in external manifest")
            numbered = f"{int(r['eeg_row']):05d}.png"
            names.append(numbered if (folder / numbered).is_file() else named)
        verify_pixels(folder, names, hashes)
    return info, None, reference or None


@dataclass
class Case:
    metrics: tuple
    rows: np.ndarray
    labels: np.ndarray
    words: np.ndarray
    values: np.ndarray
    source: dict
    seeds: np.ndarray | None
    reference_hashes: list | None


def arm_spec(selector, default_track, external, budget):
    parts = selector.split(":")
    track, method = (default_track, selector) if len(parts) == 1 else (parts[0], ":".join(parts[1:]))
    require(track in TRACKS and re.fullmatch(r"[A-Za-z0-9_-]+", method),
            f"Invalid arm selector {selector!r}")
    is_external = method in {"atm", "native-atm", external.get("arm", "atm")}
    if is_external:
        require(budget == "official-r80", "Released ATM draw zero requires the official-r80 budget")
        require(external.get("budget", budget) == budget, "External manifest budget differs")
        require(len(parts) == 1 or track == "external", f"Use external:{method} to select the release")
        training = external.get("training", "released ATM (training cohort unspecified)")
        count = external.get("training_participants_per_fit")
        concepts = external.get("training_concepts")
        require(isinstance(training, str) and bool(training.strip()), "Invalid external training description")
        require(count is None or (type(count) is int and count > 0), "Invalid external training participant count")
        require(concepts is None or (type(concepts) is int and concepts > 0), "Invalid external training concept count")
        return dict(method=method, track="external", training=training, training_participants_per_fit=count,
                    training_concepts=concepts, layout_training_participants_per_fit=None)
    require(track != "external", f"Unknown released arm {method!r}. Provide its external manifest.")
    if track == "native-full":
        require(method in ("native-code", "native-paper") and budget.startswith("official-"),
                "The full native fit accepts native-code and native-paper only on the official gallery")
    semantic = FACTOR_CONDITION.get(method, method)
    training = "personal" if semantic in ("native-code", "native-paper", *INSTANCE_ARMS) else track
    layout = "noise" if method in NO_LAYOUT else "ridge-shell"
    return dict(method=method, track=track, training=track if track == "native-full" else training,
                training_participants_per_fit=1 if training == "personal" else 10,
                training_concepts=1654 if track == "native-full" else 1323,
                semantic_condition=semantic, layout_mode=layout,
                layout_training_participants_per_fit=1 if layout == "ridge-shell" else None)


def bootstrap(values, resamples, seed):
    """Independent participant and stimulus draws. All participants and arms share each stimulus draw."""
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 3 and min(values.shape) > 0 and np.isfinite(values).all(),
            "Expected finite participant x stimulus-unit x metric scores")
    require(resamples >= 2 and seed >= 0, "Provide at least two resamples and a nonnegative seed")
    p, u, m = values.shape
    rng = np.random.default_rng(seed)
    grouped = values.transpose(1, 0, 2).reshape(u, p * m)
    fixed_people = np.empty((resamples, p, m))
    crossed = np.empty((resamples, m))
    for start in range(0, resamples, 256):
        stop = min(start + 256, resamples)
        n = stop - start
        items = rng.multinomial(u, np.full(u, 1 / u), size=n)
        people = rng.multinomial(p, np.full(p, 1 / p), size=n)
        block = (items.astype(np.float64) @ grouped).reshape(n, p, m) / u
        fixed_people[start:stop] = block
        crossed[start:stop] = np.einsum("bp,bpm->bm", people, block) / p
    q = [.025, .975]
    return np.quantile(crossed, q, axis=0), np.quantile(fixed_people, q, axis=0)


def bootstrap_ci(values, resamples=10000, seed=20261125):
    return bootstrap(values, resamples, seed)[0]


def units(values, labels, panel):
    if panel == "official":
        return values
    # Group by registered concept identity, even when its ten images are interleaved.
    return np.stack([values[:, labels == c].mean(axis=1) for c in np.unique(labels)], axis=1)


def credit_factor(metric, n):
    if metric.endswith("_2way"):
        return n - 1
    if metric == "clip_within_swap":
        return 9 / 2  # Each within-concept unordered pair occurs in two per-query rates.
    return 1 if metric in ("clip_image_top1", "clip_within10", "generated_name1854", "original_name1854",
                           f"clip_global{n}_centered", "clip_within10_uncentered") else None


def direction(metric):
    return "lower" if metric.endswith("_distance") else "higher"


def signs(delta):
    return {"better_a": int(np.sum(delta > 1e-12)), "equal": int(np.sum(np.abs(delta) <= 1e-12)),
            "better_b": int(np.sum(delta < -1e-12))}


def describe(a, b, people, labels, panel, metrics, resamples, seed):
    """Use identical cells for both arms and bootstrap their within-cell differences."""
    delta = a if b is None else a - b
    u = units(delta, labels, panel)
    ci, conditional = bootstrap(u, resamples, seed)
    per_person = {p: {} for p in people}
    result = {}
    for j, m in enumerate(metrics):
        x = delta[:, :, j]
        estimate_key = "mean" if b is None else "difference_a_minus_b"
        interval = ci[:, j].tolist() if len(people) >= 2 else None
        item = {estimate_key: float(x.mean()), "crossed_ci95": interval, "better": direction(m)}
        if m == "original_name1854":
            item["scope"] = "Identical reference scores repeated for each participant. The reference section counts unique photographs."
        factor = credit_factor(m, len(labels))
        if factor is not None:
            item["credit_denominator"] = int(x.size * factor)
        if b is None:
            if factor is not None:
                item["credit"] = float(a[:, :, j].sum() * factor)
        else:
            item.update(mean_a=float(a[:, :, j].mean()), mean_b=float(b[:, :, j].mean()))
            if factor is not None:
                item.update(credit_a=float(a[:, :, j].sum() * factor), credit_b=float(b[:, :, j].sum() * factor))
            sign = 1 if direction(m) == "higher" else -1
            item["participant_direction"] = signs(sign * x.mean(axis=1))
            item["stimulus_unit_direction"] = signs(sign * u[:, :, j].mean(axis=0))
            if interval is None:
                item["interval_direction"] = "insufficient_participants"
            elif min(sign * ci[:, j]) > 0:
                item["interval_direction"] = "favors_a"
            elif max(sign * ci[:, j]) < 0:
                item["interval_direction"] = "favors_b"
            else:
                item["interval_direction"] = "includes_zero"
        result[m] = item
        for i, p in enumerate(people):
            record = {estimate_key: float(x[i].mean()), "stimulus_ci95": conditional[:, i, j].tolist()}
            if factor is not None:
                record["credit_denominator"] = int(len(labels) * factor)
                if b is None:
                    record["credit"] = float(a[i, :, j].sum() * factor)
                else:
                    record.update(credit_a=float(a[i, :, j].sum() * factor), credit_b=float(b[i, :, j].sum() * factor))
            per_person[p][m] = record
    return result, per_person


def load_cases(root, spec, subjects, budget, external, manifest_dir, reg, verify):
    cases, missing = {}, {}
    released = spec["track"] == "external"
    panel = budget.split("-")[0]
    for sub in subjects:
        entry = {}
        if released and external:
            if sub not in external["participants"]:
                missing[sub] = ["participant absent from external manifest"]
                continue
            entry = external["participants"][sub]
            folder = manifest_dir / entry["folder"]
        elif released:
            folder = root / "external" / f"atm-sub{sub[-2:]}"
        else:
            folder = root / "images" / spec["track"] / spec["method"] / budget / sub
        proof = folder / entry.get("manifest", "manifest.csv") if released else folder / "render.json"
        files = [folder / "observations.npz", folder / "scores.json", proof]
        absent = [p.name for p in files if not p.is_file()]
        if absent:
            missing[sub] = absent
            continue
        try:
            metrics, rows, labels, words, vals, receipt, sources = read_scores(folder, panel)
            check_gallery(reg, rows, labels, panel, spec)
            if released:
                meta, seeds, refs = external_source(folder, proof, sub, rows, labels, receipt, verify)
            else:
                meta, seeds, refs = internal_source(folder, spec, sub, budget, rows, labels, receipt, reg, verify)
                if spec["semantic_condition"] in ("native-code", "native-paper", "instance-native") and spec["track"] != "shared":
                    full = spec["track"] == "native-full"
                    fold = -1 if full else 0
                    fit = root / "fits" / f"{spec['track']}-{sub[-2:]}/enc/native/{sub}/fold{fold}_seed2025/fit.json"
                    if fit.is_file():
                        raw = fit.read_bytes()
                        record = json.loads(raw)
                        args = record["args"]
                        require(subject_id(args["sub"]) == sub and args["fold"] == fold and args["seed"] == 2025,
                                "Wrong native training fit subject, fold or seed")
                        expected = 16540 if full else len(reg["split"]["train_rows"]) if reg["split"] is not None else None
                        require(expected is None or record["n_train"] == expected, "Wrong number of native training rows")
                        meta.update(native_fit_sha256=sha(raw), native_fit_rows=record["n_train"], native_fit_fold=fold)
            sources.update(meta)
            for k in ("scores_sha256", "observations_sha256", "manifest_sha256"):
                if k in entry:
                    require(sources.get(k) == entry[k], f"External manifest pins a different {k}")
            sources["folder"] = (folder.relative_to(root) if folder.is_relative_to(root) else folder).as_posix()
            cases[sub] = Case(metrics, rows, labels, words, vals, sources, seeds, refs)
        except (ValueError, OSError, KeyError, TypeError, csv.Error) as exc:
            raise ValueError(f"{folder}: {exc}") from exc
    if spec["training"] == "shared" and not released:
        checkpoints = {c.source["checkpoint_sha256"] for c in cases.values() if "checkpoint_sha256" in c.source}
        require(len(checkpoints) <= 1, f"Shared {spec['method']} uses different checkpoints across participants")
    return cases, {"scored": list(cases), "missing": list(missing), "missing_files": missing}


def check_component_sources(specs, runs):
    checkpoints = {}
    for arm, by_subject in runs.items():
        spec = specs[arm]
        semantic = spec.get("semantic_condition")
        group = {"native-paper": "native", "native-code": "native", "instance-native": "native",
                 "instance-direct": "instance calibration", "instance-residual": "instance calibration",
                 "wrong": "d2"}.get(semantic, semantic)
        for sub, case in by_subject.items():
            h = case.source.get("checkpoint_sha256")
            if h is not None:
                key = (spec["track"], sub, group)
                require(checkpoints.setdefault(key, h) == h,
                        f"Condition {group} uses different checkpoints across component arms for {sub} on {spec['track']}")


def compare_sources(cases):
    first = next(iter(cases), None)
    if first is None:
        return None
    reference = first.metrics.index("original_name1854")
    for c in cases:
        require(c.metrics == first.metrics, "Fresh scorer versions use different centering. Use a common scoring protocol.")
        require(np.array_equal(c.rows, first.rows) and np.array_equal(c.labels, first.labels),
                "Shared reference stimulus IDs or concept labels disagree")
        require(np.array_equal(c.values[:, reference], first.values[:, reference])
                and np.array_equal(c.words, first.words), "Original RGB reference word scores disagree by row")
    for key in ("scorer_sha256", "split_sha256"):
        hashes = {c.source[key] for c in cases if key in c.source}
        require(len(hashes) <= 1, f"Inconsistent {key} across score sources")
    seeds = [c.seeds for c in cases if c.seeds is not None]
    require(all(np.array_equal(s, seeds[0]) for s in seeds), "Render seed disagrees for a shared stimulus")
    refs = [c.reference_hashes for c in cases if c.reference_hashes is not None]
    require(all(r == refs[0] for r in refs), "External manifests disagree on original reference hashes")
    return first


def source_config(path):
    if path is None:
        return {}, None, None
    path = Path(path).resolve()
    raw = path.read_bytes()
    config = json.loads(raw)
    require(isinstance(config.get("participants"), dict), "Expected a participant-to-folder external manifest")
    entries = {}
    for provided, entry in config["participants"].items():
        sub = subject_id(provided)
        require(sub not in entries, f"Duplicate external participant {sub}")
        if isinstance(entry, str):
            entry = {"folder": entry}
        require(isinstance(entry.get("folder"), str) and bool(entry["folder"]), f"Missing external folder for {sub}")
        entries[sub] = entry
    config["participants"] = entries
    return config, path.parent, sha(raw)


def aggregate(root, training, budget, arms, subjects=None, pairs=(), resamples=10000,
              seed=20261125, external_manifest=None, verify_pixels=False):
    """Return a JSON-serializable report with missing cases and reject malformed completed cases."""
    root = Path(root).resolve()
    require(training in TRACKS and budget in BUDGETS, "Invalid track or budget")
    require(resamples >= 2 and seed >= 0, "Provide at least two resamples and a nonnegative seed")
    panel = budget.split("-")[0]
    reg = registration(root, panel)
    if subjects is None:
        subjects = (reg["development"] or {"subjects": [1, 3]})["subjects"] if panel == "dev" else SUBJECTS
    subjects = [subject_id(s) for s in subjects]
    require(subjects and len(set(subjects)) == len(subjects), "Specify distinct expected participants")
    require(arms and len(set(arms)) == len(arms), "Specify distinct requested arms")
    pairs = [tuple(p) for p in pairs]
    require(all(len(p) == 2 and p[0] != p[1] and set(p) <= set(arms) for p in pairs),
            "Each pair must name two different requested arms")
    external, manifest_dir, config_sha = source_config(external_manifest)
    specs = {a: arm_spec(a, training, external, budget) for a in arms}
    physical = {("external", "atm") if s["track"] == "external" else (s["track"], s["method"])
                for s in specs.values()}
    require(len(physical) == len(arms), "Two selectors identify the same source arm")
    n, concept_count = PANELS[panel]
    runs, coverage = {}, {}
    for arm, spec in specs.items():
        runs[arm], coverage[arm] = load_cases(root, spec, subjects, budget, external, manifest_dir, reg, verify_pixels)
    check_component_sources(specs, runs)
    first = compare_sources([case for by_subject in runs.values() for case in by_subject.values()])
    group = "shared photograph" if panel == "official" else "shared concept with all ten photographs"
    report = {"training": training, "budget": budget, "panel": panel, "requested_participants": subjects,
              "expected_photographs_per_participant": n, "stimulus_units_per_participant": concept_count,
              "repeats_per_query": 80 if budget == "official-r80" else 4,
              "complete": all(not v["missing"] for v in coverage.values()), "coverage": coverage,
              "complete_definition": "All requested participant-arm score files and corresponding source receipts are present and pass the available checks.",
              "resampling": {"seed": seed, "resamples": resamples, "stimulus_unit": group,
                "interval": "Pointwise 95% percentile intervals. Independently resample participants and stimulus units with replacement. Every arm and participant shares the same stimulus draw. Per-participant intervals resample only stimulus units.",
                "target": "Per-query scores conditional on the fixed gallery, distractors, gallery centering, fitted models and one draw. Within-concept swaps keep each concept's 45 unordered pairs together. A single participant has only a stimulus interval.",
                "multiplicity": "Endpoint and direction counts are descriptive. Each interval covers a single comparison."},
              "reference": None, "arms": {}, "comparisons": []}
    if config_sha:
        report["external_manifest_sha256"] = config_sha
    report["registration_sha256"] = reg["hashes"]
    if reg["split"] is not None:
        report["registered_split_sha256"] = reg["hashes"]["paper_split.json"]
    if reg["development"] is not None:
        report["development_registration"] = {k: v for k, v in reg["development"].items() if k != "dev_rows"}
    if first is not None:
        metrics = first.metrics
        idx = metrics.index("original_name1854")
        original = first.values[:, idx]
        ci = bootstrap_ci(units(original[None, :, None], first.labels, panel), resamples, seed)
        report["reference"] = {"scored_photographs": n, "original_name1854_credit": float(original.sum()),
                               "mean": float(original.mean()), "stimulus_ci95": ci[:, 0].tolist(),
                               "stimulus_ids": first.rows.tolist(), "concepts": first.labels.tolist()}
        if panel != "official":
            report[panel + "_clip_protocol"] = (f"Within-ten, swap and global-{n} use gallery-centered cosine. An additional within-ten measure uses uncentered cosine."
                if f"clip_global{n}_centered" in metrics else "Legacy uncentered cosine within-ten and swap.")
    for arm, by_subject in runs.items():
        people = list(by_subject)
        summary = {**specs[arm], "participants": people, "scored_images": len(people) * n,
                   "sources": {s: c.source for s, c in by_subject.items()}, "metrics": {}, "per_participant": {}}
        if specs[arm]["track"] == "native-full":
            summary.update(training_fold=-1, training_photographs=16540)
        elif specs[arm]["track"] != "external":
            summary.update(training_fold=0, training_photographs=len(reg["split"]["train_rows"]) if reg["split"] is not None else None)
        if people:
            a = np.stack([by_subject[s].values for s in people])
            summary["metrics"], summary["per_participant"] = describe(a, None, people, first.labels, panel, metrics, resamples, seed)
        report["arms"][arm] = summary
    for a, b in pairs:
        people = [s for s in subjects if s in runs[a] and s in runs[b]]
        external_pair = "external" in (specs[a]["track"], specs[b]["track"])
        train_a, train_b = (specs[k]["training_participants_per_fit"] for k in (a, b))
        concepts_a, concepts_b = (specs[k]["training_concepts"] for k in (a, b))
        same_scope = specs[a]["training"] == specs[b]["training"] and concepts_a == concepts_b
        if external_pair:
            known_difference = ((None not in (train_a, train_b) and train_a != train_b)
                                or (None not in (concepts_a, concepts_b) and concepts_a != concepts_b))
            same_scope = False if known_difference else None
        pair = {"arm_a": a, "arm_b": b, "participants": people, "missing": [s for s in subjects if s not in people],
                "scored_image_pairs": len(people) * n, "metrics": {}, "per_participant": {},
                "training_a": specs[a]["training"], "training_b": specs[b]["training"],
                "training_participants_per_fit_a": train_a, "training_participants_per_fit_b": train_b,
                "training_concepts_a": concepts_a, "training_concepts_b": concepts_b,
                "training_scope_matched": same_scope,
                "generation_settings_matched": None,
                "generation_comparison": ("Released native draw zero uses the external sampler, seed, model training and denoising budget."
                    if external_pair else "Compared generation settings are prompt, draws, UNet calls, per-stimulus seeds and repeat indices.")}
        if people:
            left, right = [np.stack([runs[arm][p].values for p in people]) for arm in (a, b)]
            pair["metrics"], pair["per_participant"] = describe(left, right, people, first.labels, panel, metrics, resamples, seed)
            if not external_pair:
                settings = {"prompt", "draws", "unet_calls_per_draw", "seeds", "repeats", "row"}
                if all(not settings.intersection(runs[arm][s].source["missing_render_metadata"])
                       for arm in (a, b) for s in people):
                    pair["generation_settings_matched"] = True
            outcomes = {m: v["interval_direction"] for m, v in pair["metrics"].items() if m != "original_name1854"}
            pair["interval_pattern"] = {k: [m for m, label in outcomes.items() if label == k]
                                        for k in ("favors_a", "includes_zero", "favors_b", "insufficient_participants")}
        report["comparisons"].append(pair)
    return report


def ci_text(interval):
    return "unavailable" if interval is None else f"[{interval[0]:.5f}, {interval[1]:.5f}]"


def print_metrics(summary, paired=False):
    estimate = "difference_a_minus_b" if paired else "mean"
    for name, metric in summary["metrics"].items():
        if name == "original_name1854":
            continue
        extra = ""
        if paired:
            d = metric["participant_direction"]
            extra = f"  A/equal/B participants {d['better_a']}/{d['equal']}/{d['better_b']}"
        if "credit_denominator" in metric:
            denom = metric["credit_denominator"]
            credit = (f"{metric['credit_a']:.9g} vs {metric['credit_b']:.9g}"
                      if paired else f"{metric['credit']:.9g}")
            extra += f"  credit {credit} / {denom}"
        print(f"  {name:<26} {metric[estimate]: .5f}  crossed 95% {ci_text(metric['crossed_ci95'])}"
              f"  {metric['better']} better{extra}")
    for sub, record in summary["per_participant"].items():
        print(f"  {sub}  {estimate} [stimulus 95%]")
        cells = [f"{m} {v[estimate]:.5f} {ci_text(v['stimulus_ci95'])}" for m, v in record.items()
                 if m != "original_name1854"]
        for i in range(0, len(cells), 2):
            print("    " + "  ".join(cells[i:i + 2]))


def print_report(report):
    print(f"{report['budget']} | default training {report['training']} | score-file coverage complete {report['complete']}")
    print(f"{report['resampling']['resamples']} crossed draws over participants and "
          f"{report['stimulus_units_per_participant']} units ({report['resampling']['stimulus_unit']}). "
          "Intervals condition on the original gallery. Rates and rate differences are fractions.")
    if report.get(report["panel"] + "_clip_protocol"):
        print(report[report["panel"] + "_clip_protocol"])
    if report["reference"] is not None:
        ref = report["reference"]
        print(f"Original RGB naming credit {ref['original_name1854_credit']:.9g}/{ref['scored_photographs']} unique "
              f"photographs, mean {ref['mean']:.5f} [stimulus 95%] {ci_text(ref['stimulus_ci95'])}")
    for arm, summary in report["arms"].items():
        cohort = report["coverage"][arm]
        print(f"\n{arm} | training {summary['training']} | EEG encoder training people per fit "
              f"{summary['training_participants_per_fit'] or 'unspecified'} | "
              f"training concepts {summary['training_concepts'] or 'unspecified'} | "
              f"{len(cohort['scored'])}/{len(report['requested_participants'])} participants, "
              f"{summary['scored_images']} scored images")
        if cohort["missing"]:
            print("  Missing " + ", ".join(f"{s} ({', '.join(cohort['missing_files'][s])})" for s in cohort["missing"]))
        if summary["layout_training_participants_per_fit"] == 1:
            print("  Each evaluated participant has a separately fitted EEG-to-layout ridge.")
        if cohort["scored"]:
            sources = list(summary["sources"].values())
            bound = sum(bool(s.get("render_bound_by_scorer") or s.get("manifest_bound_by_scorer")) for s in sources)
            ordered = sum("ordered_image_hashes_sha256" in s for s in sources)
            checked = sum(s["pixels_locally_verified"] for s in sources)
            print(f"  Scorer-bound source receipts {bound}/{len(sources)}, ordered image hashes {ordered}/{len(sources)}, "
                  f"generated files rehashed at aggregation {checked}/{summary['scored_images']}.")
        print_metrics(summary)
    for pair in report["comparisons"]:
        print(f"\nA={pair['arm_a']} ({pair['training_a']}) minus B={pair['arm_b']} ({pair['training_b']}) | "
              f"{len(pair['participants'])} paired participants, {pair['scored_image_pairs']} image pairs")
        if pair["missing"]:
            print("  Missing pairs " + ", ".join(pair["missing"]))
        print("  " + pair["generation_comparison"])
        print_metrics(pair, paired=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training", choices=TRACKS, required=True)
    parser.add_argument("--budget", choices=BUDGETS, required=True)
    parser.add_argument("--arms", nargs="+", required=True, help="Arm names or TRAINING:ARM selectors")
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"),
                        help="Estimate A minus B on their common participants. Repeat for additional contrasts.")
    parser.add_argument("--subjects", "--participants", nargs="+",
                        help="Expected participants. Defaults to the registered development cohort for dev-r4 and all ten otherwise.")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--external-manifest", type=Path, help="Optional JSON identifying released ATM score folders")
    parser.add_argument("--verify-pixels", action="store_true", help="Also hash every generated image on this machine")
    parser.add_argument("--output", "--out", type=Path, help="Save the report as JSON, including provenance and coverage")
    args = parser.parse_args()
    report = aggregate(args.root, args.training, args.budget, args.arms, args.subjects, args.pair,
                       args.resamples, args.seed, args.external_manifest, args.verify_pixels)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_name(f"{args.output.name}.{os.getpid()}.partial")
        tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(args.output)
    print_report(report)


if __name__ == "__main__":
    main()
