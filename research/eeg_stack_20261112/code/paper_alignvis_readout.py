"""Test same-backbone perceptual alignment as EEG supervision before rendering."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_aggregate import bootstrap_ci, units
from paper_alignvis_teacher import REVISION, TOWERS
from paper_fusion_siglip_probe import (
    measure, official_conditions, predict, rank_credit, ridge_system, solve_ridge, train_coordinates,
)
from paper_render_cache import PinnedFiles, atomic_file, require, save_json
from paper_tangent_calibration import read_inputs


FAMILIES = ("synclr", "clip", "dinov2")
TEACHERS = tuple(key for key in TOWERS if key != "h14")
EEG_RIDGE = .1
PHOTO_RIDGE = .01
RESIDUAL = .5
METRICS = ("teacher_photo_top1", "common_photo_top1", "h14_name_top1",
           "h14_name_top5", "siglip2_name_top1", "siglip2_name_top5")
ERROR_TARGETS = ("photo-error", "concept-error", "instance-error")


def pack_eeg(record, rows):
    result = np.empty((len(rows), 17 * 40), dtype=np.float32)
    for start in range(0, len(rows), 64):
        section = rows[start:start + 64]
        value = record[section, :, :, :200].mean(axis=1, dtype=np.float32)
        result[start:start + len(section)] = value.reshape(len(section), 17, 40, 5).mean(-1).reshape(len(section), -1)
    require(np.isfinite(result).all(), "Malformed EEG")
    return result


def raw_coordinates(benchmark, train, dev, fresh, pins):
    prep = benchmark / "inputs/prep17"
    training, held = [], {panel: [] for panel in ("dev", "fresh", "official")}
    physical = {}
    for p in range(1, 11):
        sources = {}
        arrays = {}
        for phase, shape in (("train", (16540, 4, 17, 250)), ("test", (200, 80, 17, 250))):
            file = prep / f"sub-{p:02d}_{phase}.npy"
            meta = file.with_suffix(".json")
            digest = pins.pin(file)
            receipt = json.loads(meta.read_bytes())
            require(receipt["output_sha256"] == digest and receipt["subject"] == f"sub-{p:02d}"
                    and receipt["split"] == phase, "Changed physical EEG array")
            data = np.load(file, mmap_mode="r", allow_pickle=False)
            require(data.shape == shape and data.dtype == np.float16, "Unexpected physical EEG shape")
            arrays[phase] = data
            sources[phase] = {"array_sha256": digest, "receipt_sha256": pins.pin(meta)}
        physical[str(p)] = sources
        x = pack_eeg(arrays["train"], train)
        scale = np.maximum(x.std(0), 1e-5)
        training.append(torch.from_numpy(x / scale))
        for panel, record, rows in (
            ("dev", arrays["train"], dev), ("fresh", arrays["train"], fresh),
            ("official", arrays["test"], np.arange(200)),
        ):
            held[panel].append(torch.from_numpy(pack_eeg(record, rows) / scale))
    return torch.stack(training), {name: torch.stack(value) for name, value in held.items()}, physical


def unit(value, device):
    return F.normalize(torch.as_tensor(np.asarray(value).copy(), device=device).float(), dim=-1)


def compare(tests, pairs, labels, panel):
    by_arm = {arm: values.sum((0, 1)).tolist() for arm, values in tests.items()}
    diff = {}
    for candidate, baseline in pairs:
        values = tests[candidate] - tests[baseline]
        lo, hi = bootstrap_ci(units(values, labels, panel), resamples=4000)
        diff[f"{candidate} minus {baseline}"] = np.stack((values.mean((0, 1)), lo, hi), axis=1).tolist()
    return {"credits": by_arm, "paired": diff}


def aligned_condition(native, aligned, plain):
    return correct_direction(native, aligned - plain)


def correct_direction(native, direction):
    tangent = direction - (direction * native).sum(-1, keepdim=True) * native
    return F.normalize(native + RESIDUAL * tangent, dim=-1)


def crossfit_semantic_error(plain, aligned, target, concepts):
    """Predict the photo-space error elsewhere, then isolate the shared concept translation."""
    labels = torch.as_tensor(concepts, device=target.device)
    counts = torch.bincount(labels)
    require(bool((counts == 10).all()), "Expected ten registered images per training concept")
    difference = torch.empty_like(target)
    for fold in range(5):
        valid = labels % 5 == fold
        fit = []
        for feature in (plain, aligned):
            bridge = solve_ridge(ridge_system(feature[None, ~valid], target[~valid]), PHOTO_RIDGE)
            fit.append(predict(feature[None, valid], bridge)[0])
        difference[valid] = fit[0] - fit[1]
    totals = torch.zeros(len(counts), target.shape[1], device=target.device, dtype=target.dtype)
    totals.index_add_(0, labels, difference)
    shared = (totals[labels] - difference) / (counts[labels, None] - 1)
    specific = difference - shared
    return tuple(x - x.mean(0) for x in (difference, shared, specific))


def run(benchmark, archive, output, mode, device, error_calibration=False):
    pins = PinnedFiles()
    source, train, dev, original_fused, groups, _, _, _, proof, audit = read_inputs(benchmark, pins)
    split = json.loads((benchmark / "verified/data/prep/paper_split.json").read_bytes())
    fresh = np.asarray(split["fresh_rows"], dtype=np.int64)
    require(len(fresh) == 500 and set(fresh).isdisjoint(np.concatenate((train, dev))),
            "Fresh photographs overlap another panel")
    full_bank = archive.with_suffix(".json")
    author = json.loads(full_bank.read_bytes())
    require(author["revision"] == REVISION and pins.pin(archive) == author["array_sha256"]
            and author["photo_registries"]["train"] == proof["registry"], "Invalid AlignVis image embedding release")
    verified_assets = benchmark / "inputs/image_set"
    photo_tests = verified_assets / "test_photos.csv"
    require(pins.pin(photo_tests) == author["photo_registries"]["test"], "Independent target photographs differ")
    author_sha = pins.pin(full_bank)
    with np.load(archive, allow_pickle=False) as value:
        names = {part: value[f"{part}_word"] for part in ("train", "test")}
        teacher = {(part, key): unit(value[f"{part}/{key}"], device)
                   for part in ("train", "test") for key in TOWERS}
    require(set(names["train"]).isdisjoint(names["test"])
            and set(names["train"][train]).isdisjoint(names["train"][np.concatenate((dev, fresh))])
            and set(names["train"][dev]).isdisjoint(names["train"][fresh]), "Held-out concepts overlap")

    locked = benchmark / "token-rank-check/fused-r80/official-r80-fused.npz"
    parent = json.loads(locked.with_suffix(".json").read_bytes())
    fused_sha = pins.pin(locked)
    require(fused_sha == parent["output_sha256"] and parent["split"] == proof["split"], "Official D2 receipt differs")
    official_x, official_photo, hits = official_conditions(locked, device)
    require(hits.sum() == 937, "Expected the registered official D2 feature panel")
    siglip_path = benchmark / "token-rank-check/siglip-independent/siglip-fused-evaluation-bridge.pt"
    siglip_meta = json.loads(siglip_path.with_name("siglip-fused-probe.json").read_bytes())
    siglip_sha = pins.pin(siglip_path)
    require(siglip_sha == siglip_meta["evaluation_bridge_sha256"] and
            fused_sha == siglip_meta["input"]["official_archive"], "Independent frozen measurement changed")
    siglip = torch.load(siglip_path, map_location=device, weights_only=True)
    words_path = verified_assets / "concepts1854.csv"
    words_sha = pins.pin(words_path)
    require(words_sha == siglip_meta["vision_and_text"]["test"]["concepts"], "Different independent vocabulary")
    with words_path.open(newline="", encoding="utf-8-sig") as handle:
        independent_ids = {r["unique_id"]: i for i, r in enumerate(csv.DictReader(handle))}
    text_path = source / "h14-text-bank.npz"
    with np.load(text_path, allow_pickle=False) as data:
        h14_words = {name: i for i, name in enumerate(data["concept_id"])}
        h14_bank = unit(data["text"], device)
    require(len(h14_words) == len(independent_ids) == 1854, "Malformed 1,854-word evaluation")
    h14_sha = pins.pin(text_path)

    fused_photo = unit(original_fused, device)
    h14_photo = teacher["train", "h14"]
    target = torch.cat([fused_photo[train], h14_photo[train]], dim=-1)
    visual_readouts = {
        key: solve_ridge(ridge_system(teacher["train", key][train][None], target), PHOTO_RIDGE)
        for key in TEACHERS
    }
    base_h14 = solve_ridge(ridge_system(fused_photo[train][None], h14_photo[train]), PHOTO_RIDGE)
    if mode == "frozen-d2":
        x, d, _, eeg_proof = train_coordinates(source, train, dev, proof, audit, pins, device)
        queries = {"dev": d, "official": official_x}
    else:
        x, queries, eeg_proof = raw_coordinates(benchmark, train, dev, fresh, pins)
        x = x.to(device)
        queries = {panel: q.to(device) for panel, q in queries.items()}
    measures = [teacher["train", key][train] for key in TEACHERS] + [fused_photo[train]]
    if error_calibration:
        measures.extend(crossfit_semantic_error(teacher["train", "synclr-base"][train],
                                               teacher["train", "synclr-aligned"][train], fused_photo[train], groups))
    fit = solve_ridge(ridge_system(x, torch.cat(measures, dim=-1)), EEG_RIDGE)
    slices = [v.shape[1] for v in measures]
    del x, target, measures
    output_arrays, summary = {}, {}
    for panel, q in queries.items():
        phase = "test" if panel == "official" else "train"
        rows = {"dev": dev, "fresh": fresh, "official": np.arange(200)}[panel]
        ids = names[phase][rows]
        h14_match = np.array([h14_words[y] for y in ids])
        siglip_match = np.array([independent_ids[y] for y in ids])
        if panel == "official":
            require(np.array_equal(siglip_match, siglip["official_target"].cpu()), "Independent stimulus order changed")
        reference = official_photo if panel == "official" else fused_photo[rows]
        weight, mean_x, mean_y = fit
        slots = ((q - mean_x) @ weight + mean_y).split(slices, dim=-1)
        native = q if mode == "frozen-d2" else F.normalize(slots[len(TEACHERS)], dim=-1)
        arms = {"reference": (native, predict(native, base_h14), native, reference)}
        for key, embedding in zip(TEACHERS, slots[:len(TEACHERS)], strict=True):
            embedding = F.normalize(embedding, dim=-1)
            ref = teacher[phase, key][rows]
            fused, semantic = (F.normalize(x, dim=-1) for x in predict(embedding, visual_readouts[key]).split(1024, dim=-1))
            arms[key] = (fused, semantic, embedding, ref)
            truth_fused, truth_sem = (F.normalize(x, dim=-1) for x in predict(ref[None], visual_readouts[key]).split(1024, dim=-1))
            arms[f"true-{key}"] = (truth_fused.expand_as(fused), truth_sem.expand_as(semantic),
                                   ref.expand_as(embedding), ref)
        comparisons = []
        donor = np.roll(np.arange(len(rows)), 1 if panel == "official" else 10)
        require(np.all(ids != ids[donor]), "Wrong-EEG contrasts must change the stimulus concept")
        for family in FAMILIES:
            a, b = (f"{family}-{suffix}" for suffix in ("aligned", "base"))
            comparisons.extend(((a, b), (a, "reference")))
            residual = aligned_condition(native, arms[a][0], arms[b][0])
            swapped = aligned_condition(native, arms[a][0][:, donor], arms[b][0][:, donor])
            for key, value in ((f"{family}-residual", residual), (f"{family}-residual-wrong", swapped)):
                arms[key] = (value, predict(value, base_h14), value, reference)
            comparisons.extend(((f"{family}-residual", "reference"),
                                (f"{family}-residual", f"{family}-residual-wrong")))
        if error_calibration:
            for name, direction in zip(ERROR_TARGETS, slots[len(TEACHERS) + 1:], strict=True):
                for key, change in ((name, direction), (f"{name}-wrong", direction[:, donor])):
                    result = correct_direction(native, change)
                    arms[key] = (result, predict(result, base_h14), result, reference)
                comparisons.extend(((name, "reference"), (name, f"{name}-wrong")))
                output_arrays[f"{panel}/{name}/fused"] = arms[name][0].cpu().numpy()
        per_arm = {}
        diagonal = np.arange(len(rows))
        for key, (fused, semantic, estimate, photos) in arms.items():
            h14 = measure(semantic, h14_bank, np.arange(len(h14_words)), h14_match)
            second = measure(predict(fused, siglip["readout"]), siglip["text_bank"],
                             siglip["names"].cpu().numpy(), siglip_match)
            teacher_rank = rank_credit((estimate @ photos.T).cpu().numpy(), diagonal, 1)
            fused_rank = rank_credit((fused @ reference.T).cpu().numpy(), diagonal, 1)
            per_arm[key] = np.concatenate((teacher_rank[..., None], fused_rank[..., None], h14, second), axis=-1)
            output_arrays[f"{panel}/{key}"] = per_arm[key]
        n = len(rows) * len(q)
        summary[panel] = dict(n=n, **compare(per_arm, comparisons, ids, panel), row=rows.tolist())
        for family in FAMILIES:
            for stage in ("aligned", "base", "residual"):
                label = f"{family}-{stage}"
                output_arrays[f"{panel}/{label}/fused"] = arms[label][0].cpu().numpy()
        print(mode, panel, {key: val.sum((0, 1)).tolist() for key, val in per_arm.items()
                            if not key.startswith("true-")}, flush=True)
    output.mkdir(parents=True, exist_ok=True)
    name = f"{mode}-error-calibration" if error_calibration else f"{mode}-readout"
    score = output / f"{name}.npz"
    atomic_file(score, lambda stream: np.savez_compressed(stream, **output_arrays))
    pins.verify()
    save_json(output / f"{name}.json", dict(
        mode=mode, proof=proof, release=author_sha, array_sha256=pins.pin(score), vision_archive=pins.pin(archive),
        official_fused=fused_sha, official_receipt=pins.pin(locked.with_suffix(".json")), eeg=eeg_proof,
        code=pins.pin(Path(__file__)), h14_words=h14_sha, vocabulary=words_sha, siglip_bridge=siglip_sha,
        electrodes=17, train_queries_per_person=len(train), train_repeat_count=4, official_repeat_count=80,
        metrics=METRICS, eeg_ridge=EEG_RIDGE, photo_ridge=PHOTO_RIDGE, residual=RESIDUAL,
        error_calibration={"teacher": "synclr", "folds": 5, "targets": ERROR_TARGETS} if error_calibration else None,
        results=summary,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("benchmark", type=Path)
    cli.add_argument("archive", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--mode", choices=("frozen-d2", "raw-eeg"), required=True)
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--error-calibration", action="store_true", help="Predict the shared and photo-specific SynCLR bridge errors")
    args = cli.parse_args()
    torch.set_num_threads(6)
    with torch.inference_mode():
        run(args.benchmark, args.archive, args.output, args.mode, args.device, args.error_calibration)
