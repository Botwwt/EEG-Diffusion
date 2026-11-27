"""Test whether extra scalp measurements add held-out object evidence to shared D2."""

import argparse
import csv
import json
import os
from pathlib import Path
import tarfile

import numpy as np

from paper_aggregate import bootstrap_ci, units
from paper_raw_semantic_readout import leave_photo_out, normalize, read_json, top_credit
from paper_render_cache import PinnedFiles, atomic_file, require, save_json


POSTERIOR = slice(46, 63)
ADDITIONAL = slice(0, 46)
BINS = 40
BIN_SAMPLES = 5
RIDGE = .25
BLEND = .25
ARMS = ("posterior", "full", "additional", "ordinary", "repeat-residual", "reliable-residual")
METRICS = ("h14_name_top1", "siglip2_bridge_name_top1", "h14_name_top5", "siglip2_bridge_name_top5",
           "h14_original_top1", "h14_cross_photo_top1")


def ridge(covariance, cross):
    matrix = covariance.copy()
    matrix.flat[::len(matrix) + 1] += RIDGE * covariance.diagonal().mean()
    return np.linalg.solve(matrix, cross)


def within_concept(x, group):
    return (x - leave_photo_out(x, group).astype(x.dtype)) * .9


def positive_repeat_modes(p0, p1, a0, a1):
    """Discard directions that also occur after breaking the physical repeat pairing."""
    observed = (p0.T @ p0 + p1.T @ p1) * .5
    observed.flat[::len(observed) + 1] += RIDGE * observed.diagonal().mean()
    lower = np.linalg.cholesky(observed)
    q0 = np.linalg.solve(lower, p0.T).T
    q1 = np.linalg.solve(lower, p1.T).T
    paired = q0.T @ q1
    reliability, axes = np.linalg.eigh((paired + paired.T) * .5)
    null = []
    for offset in (len(q1) // 3, 2 * len(q1) // 3):
        shuffled = q0.T @ np.roll(q1, offset, axis=0)
        null.append(float(np.linalg.eigvalsh((shuffled + shuffled.T) * .5)[-1]))
    keep = reliability > max(1e-6, *null)
    selected = axes[:, keep]
    cross = (q0.T @ a1 + q1.T @ a0) * .5
    mapping = np.linalg.solve(lower.T, selected @ ((selected.T @ cross) / reliability[keep, None]))
    return mapping, dict(retained=int(keep.sum()), total=len(keep), unpaired_maximum=null,
                         paired_maximum=float(reliability[-1]))


def nuisance_maps(trials, group):
    """Estimate the posterior-to-additional relation using only within-object deviation."""
    p, a = (trials[:, :, channels].reshape(len(trials), 2, -1) for channels in (POSTERIOR, ADDITIONAL))
    p0, p1 = (within_concept(p[:, half], group) for half in (0, 1))
    a0, a1 = (within_concept(a[:, half], group) for half in (0, 1))
    p_mean, a_mean = (p0 + p1) * .5, (a0 + a1) * .5
    ordinary = ridge(p_mean.T @ p_mean, p_mean.T @ a_mean)

    # Use different physical presentations on the two sides of the moments.
    # Project the finite-sample symmetric covariance onto the PSD cone before solving.
    cross = p0.T @ p1
    eigenvalues, axes = np.linalg.eigh((cross + cross.T) * .5)
    repeat_covariance = (axes * eigenvalues.clip(0)) @ axes.T
    repeated = ridge(repeat_covariance, (p0.T @ a1 + p1.T @ a0) * .5)
    reliable, repeat_proof = positive_repeat_modes(p0, p1, a0, a1)
    return (ordinary, repeated, reliable), repeat_proof


def views(eeg, center, scale, maps):
    centered = (eeg - center) / scale
    p = centered[:, POSTERIOR].reshape(len(eeg), -1)
    others = centered[:, ADDITIONAL].reshape(len(eeg), -1)
    return {"posterior": p, "full": centered.reshape(len(eeg), -1), "additional": others,
            "ordinary": others - p @ maps[0], "repeat-residual": others - p @ maps[1],
            "reliable-residual": others - p @ maps[2]}


def fit(trials, targets, group):
    averages = trials.mean(axis=1)
    center = averages.mean(axis=0)
    scale = (averages - center).std(axis=(0, 2), keepdims=True)
    require((scale > 0).all(), "A full-scalp training channel has no variance")
    maps, repeat_proof = nuisance_maps(trials / scale[:, None], group)
    heads = {name: ridge(x.T @ x, x.T @ targets)
             for name, x in views(averages, center, scale, maps).items()}
    return center, scale, maps, heads, repeat_proof


def infer(average, model):
    center, scale, maps, heads, _ = model
    return {name: normalize(value @ heads[name]) for name, value in views(average, center, scale, maps).items()}


def read_recording(inputs, assets, subject, phase, ids, rows, pins):
    from common import CH17, safe_load_pt

    stem = f"sub-{subject:02d}"
    meta = inputs / "prep17" / f"{stem}_{phase}.json"
    receipt = read_json(meta)
    original = inputs / receipt["source_path"]
    published = assets[receipt["source_path"]]
    require(original.name == f"{phase}.pt" and original.parent.name == stem
            and original.stat().st_size == published["bytes"] == receipt["source_bytes"]
            and pins.pin(original) == published["sha256"] == receipt["source_sha256"], "Unregistered full-scalp EEG")
    data = safe_load_pt(original)
    shape = (len(ids), 4 if phase == "train" else 80, 63, 250)
    raw, image = data["eeg"], data["img"]
    require(raw.shape == shape and raw.dtype == np.float16 and image.shape == shape[:2]
            and np.all(image == image[:, :1]) and list(data["ch_names"])[POSTERIOR] == CH17
            and len(set(data["ch_names"])) == 63 and receipt["source_channel_indices"] == list(range(46, 63)),
            "Full-scalp source does not have the verified montage, stimulus or repeat axes")
    path_to_index = {path: i for i, path in enumerate(image[:, 0])}
    require(len(path_to_index) == len(ids) and set(path_to_index) == set(ids), "The source image roster changed")
    source_rows = np.asarray([path_to_index[ids[i]] for i in rows])
    result = np.empty((len(rows), 2, 63, BINS), dtype=np.float32)
    for start in range(0, len(rows), 80):
        excerpt = raw[source_rows[start:start + 80], :, :, :BINS * BIN_SAMPLES].astype(np.float32)
        binned = excerpt.reshape(len(excerpt), shape[1] // 2, 2, 63, BINS, BIN_SAMPLES)
        result[start:start + len(excerpt)] = binned.mean(axis=(1, 5))
    require(np.isfinite(result).all(), "Non-finite 63-channel observations")
    return result, dict(source=published["sha256"], array_sha256=receipt["output_sha256"], receipt_sha256=pins.pin(meta))


def load_native(benchmark, subject, held, registration, physical, pins):
    from paper_r80_value_guidance import guidance_target

    study = benchmark / "semantic-energy-check"
    native = study / "horizon-readout-20261127/native-h14" / f"sub-{subject:02d}.h14.npz"
    meta = native.with_name(f"sub-{subject:02d}.json")
    receipt = read_json(meta)
    proof = receipt["proof"]
    require(receipt["h14_sha256"] == pins.pin(native) and proof["subject"] == subject
            and proof["repeats"] == list(range(4)) and proof["views"][0] == "native"
            and all(proof["registration"][key] == registration[key] for key in ("split", "development", "labels"))
            and all(proof["eeg"][key] == physical["train"][key] for key in ("array_sha256", "receipt_sha256")),
            "The native R4 condition reads different physical EEG or rows")
    with np.load(native, allow_pickle=False) as data:
        require(np.array_equal(data["row"], held) and data["embedding"].shape == (len(held), 4, 1024),
                "Unexpected native R4 conditions")
        r4 = normalize(data["embedding"][:, 0].copy())
    archive = study / "background-invariant-20261127/guide" / f"sub-{subject:02d}.compact.tar"
    with tarfile.open(archive) as pack:
        official = json.load(pack.extractfile("h14-guidance.json"))["record"]["physical_eeg"]
    require(all(official[key] == physical["test"][key] for key in ("array_sha256", "receipt_sha256")),
            "The native R80 condition reads different physical EEG")
    binding = dict(row=list(range(200)), repeats=[list(range(i, i + 4)) for i in range(0, 80, 4)],
                   physical_eeg=official, split_sha256=registration["split"])
    r80, sealed = guidance_target(archive, subject, binding, pins)
    return r4, r80.numpy(), dict(r4=pins.pin(meta), r80=sealed)


def read_registry(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require([int(r["row0"]) for r in rows] == list(range(len(rows))), "Out-of-order photograph registry")
    return rows


def held_cross_photo(similarity, concepts):
    names, targets, counts = np.unique(concepts, return_inverse=True, return_counts=True)
    require((counts == 10).all(), "Expected all ten distinct photos for each held-out concept")
    order = np.argsort(targets, kind="stable").reshape(len(names), 10)
    position = np.empty(len(concepts), dtype=int)
    position[order] = np.arange(10)
    matrix = similarity[..., order]
    # Each candidate excludes the same position. The correct class excludes the query photograph.
    other = matrix.sum(-1) - matrix[..., np.arange(len(concepts))[:, None], np.arange(len(names)), position[:, None]]
    return top_credit(other, targets, 1)


def run(benchmark, output):
    import torch
    from paper_categorical_readout import evaluation_sources, scores

    torch.set_num_threads(4)
    inputs, prep = benchmark / "inputs", benchmark / "verified/data/prep"
    output.mkdir(parents=True, exist_ok=True)
    pins = PinnedFiles()
    paths = dict(split=prep / "paper_split.json", development=prep / "paper_development.json",
                 labels=prep / "train_concept.npy", source=inputs / "public_manifest.json", code=Path(__file__))
    registration = {key: pins.pin(path) for key, path in paths.items()}
    train_split = read_json(paths["split"])
    train, fresh = (np.asarray(train_split[key]) for key in ("train_rows", "fresh_rows"))
    dev = np.asarray(read_json(paths["development"])["dev_rows"])
    held = np.concatenate((dev, fresh))
    labels = np.load(paths["labels"], allow_pickle=False)
    group_ids, groups, counts = np.unique(labels[train], return_inverse=True, return_counts=True)
    require(train.shape == (13230,) and dev.shape == (200,) and fresh.shape == (500,)
            and len(group_ids) == 1323 and (counts == 10).all() and len(np.unique(held)) == 700
            and set(labels[held]).isdisjoint(group_ids) and set(labels[dev]).isdisjoint(labels[fresh])
            and train_split["trial_indices_R4"] == list(range(4))
            and train_split["trial_indices_R80"] == list(range(80)), "Different training-only and evaluation splits")

    assets = {item["path"]: item for item in read_json(paths["source"])["assets"] if item["group"] == "eeg"}
    ids, names = {}, {}
    for phase in ("train", "test"):
        path = inputs / "prep17" / f"{phase}_ids.json"
        registry = inputs / "image_set" / f"{phase}_photos.csv"
        registration[f"{phase}_ids"], registration[f"{phase}_registry"] = pins.pin(path), pins.pin(registry)
        published = read_json(path)
        photos = read_registry(registry)
        require([item["image_id"] for item in photos] == published["ids"], "EEG and visual registration disagree")
        ids[phase], names[phase] = published["paths"], np.array([item["unique_id"] for item in photos])
    require(len(set(names["test"])) == 200 and set(names["train"][train]).isdisjoint(names["test"]), "Official concept overlap")
    require(np.all(np.roll(names["train"][held], 10) != names["train"][held]), "Wrong EEG control must swap the object")

    archive = benchmark / "semantic-energy-check/alignvis-20261127/alignvis-image-teachers.npz"
    proof = read_json(archive.with_suffix(".json"))
    require(pins.pin(archive) == proof["array_sha256"] and proof["photo_registries"] ==
            {key: registration[f"{key}_registry"] for key in ("train", "test")}, "Changed H/14 reference images")
    with np.load(archive, allow_pickle=False) as source:
        visual = {key: normalize(source[f"{key}/h14"].astype(np.float32)) for key in ("train", "test")}
    evaluation = evaluation_sources(benchmark, train, torch.from_numpy(visual["train"][train]),
                                    np.concatenate((names["train"][held], names["test"])), pins, "cpu")
    semantic = normalize(leave_photo_out(visual["train"][train], groups))
    target = (semantic - semantic.mean(axis=0)).astype(np.float32)
    r4, r80, provenance, dispatch = [], [], {}, {}
    for subject in range(1, 11):
        train_trial, fit_proof = read_recording(inputs, assets, subject, "train", ids["train"], np.r_[train, held], pins)
        model = fit(train_trial[:len(train)], target, groups)
        held_trials = train_trial[len(train):].mean(axis=1)
        official_trial, test_proof = read_recording(inputs, assets, subject, "test", ids["test"], np.arange(200), pins)
        test_trial = official_trial.mean(axis=1)
        physical = dict(train=fit_proof, test=test_proof)
        native, official, native_proof = load_native(benchmark, subject, held, registration, physical, pins)
        provenance[str(subject)] = dict(physical=physical, native=native_proof)
        for current, value, block in ((native, held_trials, r4), (official, test_trial, r80)):
            append_conditions(current, infer(value, model), block)
        dispatch[str(subject)] = public_model(output, subject, model, provenance[str(subject)], registration, pins)
        print(f"Participant {subject:02d} readouts verified, repeat rank {model[-1]['retained']}", flush=True)

    summary, details = {}, {}
    for panel, panel_rows, sources, candidates, concept in (
        ("dev", slice(0, 200), r4, visual["train"][dev], names["train"][dev]),
        ("fresh", slice(200, 700), r4, visual["train"][fresh], names["train"][fresh]),
        ("official", slice(None), r80, visual["test"], names["test"]),
    ):
        results = {}
        for arm in sources[0]:
            vectors = np.stack([per_person[arm][panel_rows] for per_person in sources])
            values = scores(torch.from_numpy(vectors).float(), torch.from_numpy(candidates), concept, evaluation)
            if panel != "official":
                values = np.concatenate((values, held_cross_photo(vectors @ candidates.T, concept)[..., None]), axis=-1)
            results[arm] = values
            details[f"{panel}/{arm}"] = values
        metrics = METRICS[:results["native"].shape[-1]]
        summary[panel] = dict(n=10 * len(concept), hits={name: dict(zip(metrics, v.sum((0, 1)).tolist(), strict=True))
                                                        for name, v in results.items()}, paired={})
        for name, reference in ((arm, "native") for arm in (*ARMS, "reliable-residual-wrong")):
            save_delta(summary[panel], results, name, reference, concept, panel, metrics)
        for reference in ("ordinary", "posterior", "full", "repeat-residual", "reliable-residual-wrong"):
            save_delta(summary[panel], results, "reliable-residual", reference, concept, panel, metrics)
        print(panel, summary[panel]["hits"], flush=True)

    conditions = output / "scalp-conditions.npz"
    atomic_file(conditions, lambda f: np.savez_compressed(f, held_row=held, official_row=np.arange(200),
                **{f"{name}/{arm}": np.stack([p[arm] for p in source]) for name, source in (("r4", r4), ("r80", r80))
                   for arm in source[0]}))
    points = output / "scalp-scores.npz"
    atomic_file(points, lambda f: np.savez_compressed(f, **details))
    pins.verify()
    save_json(output / "scalp-results.json", dict(registration=registration, participants=provenance, head_files=dispatch,
              h14_image_proof=pins.pin(archive.with_suffix(".json")), evaluators=evaluation[-1], n_train=13230,
              ridge=RIDGE, mix=BLEND, bins=BINS, sample_per_bin=BIN_SAMPLES, scores_sha256=pins.pin(points),
              condition_sha256=pins.pin(conditions), panels=summary))


def append_conditions(native, estimates, output):
    values = {"native": native}
    values.update({name: normalize((1 - BLEND) * native + BLEND * x) for name, x in estimates.items()})
    values["reliable-residual-wrong"] = normalize(
        (1 - BLEND) * native + BLEND * np.roll(estimates["reliable-residual"], 10, axis=0))
    output.append(values)


def public_model(output, subject, model, provenance, registration, pins):
    center, scale, maps, heads, repeat_proof = model
    path = output / f"sub-{subject:02d}.npz"
    atomic_file(path, lambda f: np.savez_compressed(f, center=center, scale=scale, ordinary=maps[0], repeated=maps[1], reliable=maps[2],
                **{f"head/{name}": head for name, head in heads.items()}))
    save_json(path.with_suffix(".json"), dict(subject=subject, model_sha256=pins.pin(path), inputs=provenance,
                                            registration=registration, fit_concepts=1323, fit_photos=13230,
                                            repeat_modes=repeat_proof))
    return pins.pin(path.with_suffix(".json"))


def save_delta(record, results, name, reference, concepts, panel, metrics):
    delta = results[name] - results[reference]
    lo, hi = bootstrap_ci(units(delta, concepts, panel))
    record["paired"][f"{name} minus {reference}"] = dict(zip(
        metrics, np.stack((delta.mean((0, 1)), lo, hi), axis=-1).tolist(), strict=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("RAD_ROOT", str(args.benchmark / "inputs"))
    run(args.benchmark, args.output)
