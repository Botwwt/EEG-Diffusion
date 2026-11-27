"""Train a continuous semantic EEG score with matched appearance controls."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_aggregate import bootstrap_ci, require, units
from paper_raw_semantic_readout import leave_photo_out, normalize, pool_trials, read_json, sha, top_credit


ARMS = ("exact-photo", "other-nine", "semantic-text", "semantic-random", "semantic-layout")
METRICS = ("name_top1", "name_top5", "image_top1", "within_concept_top1", "matched_pair")
IMAGE_BRANCH = "CLIP-ViT-H-14-laion2B-s32B-b79K"
SEED = 20261127
EPOCHS = 20


def data_registry(benchmark):
    prep = benchmark / "verified/data/prep"
    raw = benchmark / "inputs/prep17"
    native = benchmark / "native_baseline_candidate/enigma/targets"
    sources = dict(split=prep / "paper_split.json", dev=prep / "paper_development.json",
                   labels=prep / "train_concept.npy", physical_ids=raw / "train_ids.json",
                   photo_registry=benchmark / "inputs/image_set/train_photos.csv",
                   targets=native / "train_h14.npz", target_receipt=native / "receipt.json",
                   visual=benchmark / "sketch_layout_20261126/emb_train.pt")
    proof = {k: sha(v) for k, v in sources.items()}
    train = np.asarray(read_json(sources["split"])["train_rows"])
    dev = np.asarray(read_json(sources["dev"])["dev_rows"])
    labels = np.load(sources["labels"], allow_pickle=False)
    require(train.shape == (13230,) and dev.shape == (200,)
            and len(set(train)) == len(train) and len(set(dev)) == len(dev)
            and set(labels[train]).isdisjoint(labels[dev]), "Unexpected concept holdout")
    receipt = read_json(sources["target_receipt"])
    require(proof["targets"] == receipt["pack_sha256"]
            and proof["photo_registry"] == receipt["train_photo_index_sha256"], "Changed native H/14 targets")
    with sources["photo_registry"].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require([int(r["row0"]) for r in rows] == list(range(16540)), "Unexpected photo registration")
    ids = [r["image_id"] for r in rows]
    require(ids == read_json(sources["physical_ids"])["ids"], "EEG and photo registrations differ")
    with np.load(sources["targets"], allow_pickle=False) as values:
        images, text, category = (values[k] for k in ("images", "texts", "category"))
        require(np.array_equal(values["class_index"], labels) and np.array_equal(values["photo_ids"], ids)
                and np.array_equal(category[labels], [r["unique_id"] for r in rows]),
                "H/14 and concepts do not match the EEG registration")
    emb = torch.load(sources["visual"], map_location="cpu", weights_only=True)
    require(np.array_equal(emb[IMAGE_BRANCH].numpy(), images), "Layout release uses another photo ordering")
    layout = F.avg_pool2d(emb["vae"].float().reshape(16540, 4, 16, 16), 2).flatten(1).numpy()
    center, scale = layout[train].mean(0), layout[train].std(0)
    require(np.isfinite(scale).all() and np.all(scale > 0), "Invalid appearance descriptors")
    layout = (layout - center) / scale
    return raw, train, dev, labels, category, normalize(images), normalize(text), layout, proof


def nearest_other(layout, labels):
    """Use pooled visual latents to match layouts, excluding the query's concept."""
    donors = np.empty(len(layout), dtype=np.int64)
    norm = (layout ** 2).sum(-1)
    for first in range(0, len(layout), 256):
        block = slice(first, first + 256)
        distance = norm[block, None] + norm[None, :] - 2 * layout[block] @ layout.T
        distance[labels[block, None] == labels[None, :]] = np.inf
        donors[block] = distance.argmin(-1)
    require(np.all(labels[donors] != labels), "Matched donors must come from another concept")
    return donors


def physical_features(raw_dir, train, dev, ids_sha):
    subjects, held, pins = [], [], {}
    for subject in range(1, 11):
        path = raw_dir / f"sub-{subject:02}_train.npy"
        receipt = read_json(path.with_suffix(".json"))
        pins[str(subject)] = checksum = sha(path)
        require(checksum == receipt["output_sha256"] and ids_sha == receipt["ids_sha256"]
                and receipt["checked_physical_photo_ids"] == 66160 and receipt["sample_rate_hz"] == 250,
                "Physical EEG inputs have changed")
        sensor = np.load(path, mmap_mode="r", allow_pickle=False)
        require(sensor.shape == (16540, 4, 17, 250) and sensor.dtype == np.float16, "Invalid EEG")
        train_sensor = pool_trials(sensor, train).reshape(len(train), -1)
        center, scale = train_sensor.mean(0), train_sensor.std(0).clip(1e-5)
        subjects.append((train_sensor - center) / scale)
        held.append((pool_trials(sensor, dev).reshape(len(dev), -1) - center) / scale)
        print(f"Semantic contrast loaded subject {subject:02}", flush=True)
    return torch.from_numpy(np.stack(subjects)), torch.from_numpy(np.stack(held)), pins


def matched_montages(raw_dir, full_dir, train, dev, registry_sha, *, bin_samples=5, dtype=np.float32, with_normalization=False):
    """Use the same 63-input model and mask only the non-posterior electrodes."""
    require(0 < bin_samples <= 200 and 200 % bin_samples == 0, "The time bins must span the registered 800 ms")
    coverage_path = full_dir / "coverage.json"
    coverage = read_json(coverage_path)
    bins = 200 // bin_samples
    train_set = np.empty((10, len(train), 63, bins), dtype=dtype)
    dev_set = np.empty((10, len(dev), 63, bins), dtype=dtype)
    normalization = np.empty((10, 2, 63, bins), dtype=np.float32)
    pins = {"coverage_sha256": sha(coverage_path)}
    posterior = None
    for subject in range(1, 11):
        path = full_dir / f"sub-{subject:02}_train.npy"
        info = path.with_suffix(".json")
        receipt = read_json(info)
        source = read_json(raw_dir / info.name)
        idx = np.asarray(source["source_channel_indices"])
        require(len(idx) == 17 and [receipt["montage"][i] for i in idx] == source["channel_names"],
                "Brain-HIVE posterior electrodes differ from the native montage")
        if posterior is None:
            posterior = idx
        require(np.array_equal(idx, posterior) and source["source_sha256"] == receipt["source_sha256"]
                and sha(info) == coverage["receipt_files"][info.name] and receipt["photo_index_sha256"] == registry_sha
                and receipt["repeats_per_photo"] == 4 and receipt["physical_trial_count"] == 66160,
                "Full and limited montage must use the same registered physical trials")
        pins[str(subject)] = checksum = sha(path)
        require(checksum == receipt["output_sha256"], "Averaged full EEG bytes changed")
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        require(raw.shape == (16540, 63, 250) and raw.dtype == np.float32, "Invalid full EEG")

        def binned(rows):
            result = np.empty((len(rows), 63, bins), dtype=np.float32)
            for first in range(0, len(rows), 256):
                block = slice(first, first + 256)
                x = raw[rows[block], :, :200]
                result[block] = x.reshape(len(x), 63, bins, bin_samples).mean(-1)
            return result

        fit, held = binned(train), binned(dev)
        mean, scale = fit.mean(0), fit.std(0).clip(1e-5)
        normalization[subject - 1] = [mean, scale]
        train_set[subject - 1] = (fit - mean) / scale
        dev_set[subject - 1] = (held - mean) / scale
        print("Verified 63-channel ERP", subject, flush=True)
    result = train_set, dev_set, posterior, pins
    return (*result, normalization) if with_normalization else result


def contrast(query, reference):
    similarity = 15 * (query @ reference.T)
    index = torch.arange(len(query), device=query.device)
    return .5 * (F.cross_entropy(similarity, index) + F.cross_entropy(similarity.T, index))


def fit_arm(arm, x, dev, images, other, texts, label, hard, random, groups):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = nn.Sequential(nn.Linear(x.shape[-1], 512), nn.LayerNorm(512), nn.GELU(),
                          nn.Linear(512, images.shape[-1])).to(x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=.02)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=8e-5)
    source_offset = rng.integers(len(x), size=x.shape[1])
    visual = images if arm == "exact-photo" else other
    for epoch in range(EPOCHS):
        order = rng.random(groups.shape).argsort(axis=-1)
        indexed = np.take_along_axis(groups, order, axis=-1)
        losses = []
        for slot in range(indexed.shape[1]):
            for batch in np.array_split(rng.permutation(len(indexed)), 3):
                i = indexed[batch, slot]
                s = torch.as_tensor((source_offset[i] + epoch) % len(x), device=x.device)
                i = torch.as_tensor(i, device=x.device)
                q = F.normalize(model(x[s, i]), dim=-1)
                loss = contrast(q, visual[i])
                if arm.startswith("semantic-"):
                    matched = texts[label[i]]
                    loss = loss + contrast(q, matched)
                    if arm != "semantic-text":
                        donor = hard[i] if arm == "semantic-layout" else random[i]
                        logit = 15 * ((texts[label[donor]] - matched) * q).sum(-1)
                        loss = loss + .5 * F.softplus(logit).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
                opt.step()
                losses.append(float(loss.detach()))
        schedule.step()
        print(arm, epoch + 1, round(float(np.mean(losses)), 5), flush=True)
    with torch.no_grad():
        prediction = F.normalize(model(dev), dim=-1).cpu().numpy()
    return prediction, model.state_dict()


def fit(benchmark, output, device, arm_names, montage):
    raw, train, dev, labels, category, image, text, layout, proof = data_registry(benchmark)
    if montage == "posterior17":
        x, v, raw_pins = physical_features(raw, train, dev, proof["physical_ids"])
    else:
        eeg = benchmark / "native_baseline_candidate/enigma/prep"
        train_set, dev_set, posterior, raw_pins = matched_montages(raw, eeg, train, dev, proof["photo_registry"])
        if montage == "masked63":
            omit = np.setdiff1d(np.arange(63), posterior)
            train_set[:, :, omit] = 0
            dev_set[:, :, omit] = 0
        x = torch.from_numpy(train_set.reshape(*train_set.shape[:2], -1))
        v = torch.from_numpy(dev_set.reshape(*dev_set.shape[:2], -1))
    x, v = x.to(device), v.to(device)
    group = np.unique(labels[train], return_inverse=True)[1]
    groups = group.argsort(stable=True).reshape(1323, 10)
    require(np.all(np.bincount(group) == 10), "Expected ten photos per fit concept")
    hard = nearest_other(layout[train], group)
    rng = np.random.default_rng(SEED)
    random_group = (group + rng.integers(1, len(groups), size=len(group))) % len(groups)
    random = groups[random_group, rng.integers(10, size=len(group))]
    require(np.all(group[random] != group), "Random donors must come from another concept")
    fit_image = image[train]
    cross_photo = normalize(leave_photo_out(fit_image, group)).astype(np.float32)
    d2_receipt = read_json(benchmark / "semantic-energy-check/matching-targets-verified.json")
    require(d2_receipt["input"]["split"] == proof["split"]
            and d2_receipt["input"]["development"] == proof["dev"], "Baselines used different queries")
    output.mkdir(parents=True, exist_ok=True)
    geometry = {"hard": float(((layout[train[hard]] - layout[train]) ** 2).mean()),
                "random": float(((layout[train[random]] - layout[train]) ** 2).mean())}
    tensors = [torch.from_numpy(a).to(device)
               for a in (fit_image, cross_photo, text, labels[train], hard, random)]
    for arm in arm_names:
        prediction, state = fit_arm(arm, x, v, *tensors, groups)
        features = output / f"{arm}.npz"
        weights = output / f"{arm}.pt"
        np.savez_compressed(features, embedding=prediction, row=dev, concept=category[labels[dev]])
        torch.save({k: v.cpu() for k, v in state.items()}, weights)
        report = dict(arm=arm, proof=proof, raw_sha256=raw_pins, code_sha256=sha(Path(__file__)),
                      predictions_sha256=sha(features), weights_sha256=sha(weights), layout_gap=geometry,
                      seed=SEED, epochs=EPOCHS, fit_concepts=len(groups), rows=len(train),
                      torch_version=torch.__version__, subjects=10, query_repeats=[0, 1, 2, 3],
                      montage=montage, channels_in_model=x.shape[-1] // 40)
        features.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def score(benchmark, output, arm_names):
    _, _, dev, labels, category, image, _, layout, proof = data_registry(benchmark)
    evaluation = benchmark / "semantic-energy-check/h14-text-bank.npz"
    meta = read_json(evaluation.with_suffix(".json"))
    require(sha(evaluation) == meta["bank_sha256"], "Frozen H/14 evaluator has changed")
    with np.load(evaluation, allow_pickle=False) as z:
        bank, ids = normalize(z["text"]), z["concept_id"]
    require(bank.shape == (1854, 1024) and len(set(ids)) == 1854, "Invalid H/14 vocabulary")
    concept = category[labels[dev]]
    name_index = {c: i for i, c in enumerate(ids)}
    names = np.array([name_index[c] for c in concept])
    held = image[dev]
    same = concept[:, None] == concept[None, :]
    donor = nearest_other(layout[dev], labels[dev])
    row = np.arange(len(dev))
    scores, sources = {}, {}
    for arm in arm_names:
        path = output / f"{arm}.npz"
        record = read_json(path.with_suffix(".json"))
        require(record["arm"] == arm and record["proof"] == proof and sha(path) == record["predictions_sha256"],
                "Unsealed held-concept prediction")
        sources[arm] = record["predictions_sha256"]
        with np.load(path, allow_pickle=False) as z:
            require(np.array_equal(z["row"], dev) and np.array_equal(z["concept"], concept), "Query order changed")
            estimate = normalize(z["embedding"])
        p = estimate @ held.T
        s = estimate @ bank.T
        margin = s[:, row, names] + s[:, donor, names[donor]] - s[:, row, names[donor]] - s[:, donor, names]
        scores[arm] = np.stack((top_credit(s, names, 1), top_credit(s, names, 5),
                                top_credit(p, row, 1), top_credit(np.where(same, p, -np.inf), row, 1),
                                (margin > 0) + .5 * (margin == 0)), axis=-1)
    pairs = (("other-nine", "exact-photo"), ("semantic-text", "other-nine"),
             ("semantic-layout", "semantic-random"), ("semantic-layout", "semantic-text"))
    paired = {}
    for a, b in pairs:
        if a in scores and b in scores:
            delta = scores[a] - scores[b]
            lo, hi = bootstrap_ci(units(delta, concept, "dev"))
            paired[f"{a} minus {b}"] = {key: [float(delta[:, :, i].mean()), float(lo[i]), float(hi[i])]
                                        for i, key in enumerate(METRICS)}
    result = dict(queries=2000, held_concepts=len(set(concept)), proof=proof, sources=sources,
                  text_bank_sha256=meta["bank_sha256"], truth_name_top1=float(top_credit(held @ bank.T, names, 1).sum()),
                  hits={arm: dict(zip(METRICS, s.sum((0, 1)).tolist(), strict=True)) for arm, s in scores.items()},
                  paired=paired)
    (output / "semantic-contrast-verified.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output / "semantic-contrast-scores.npz", row=dev, concept=concept, donor=donor, **scores)
    print(json.dumps(result["hits"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "score"))
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--montage", default="posterior17", choices=("posterior17", "masked63", "whole63"))
    args = parser.parse_args()
    torch.set_num_threads(6)
    if args.action == "fit":
        fit(args.benchmark, args.output, args.device, args.arms, args.montage)
    else:
        score(args.benchmark, args.output, args.arms)
