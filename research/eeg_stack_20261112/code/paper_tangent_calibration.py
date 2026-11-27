"""Calibrate the frozen D2 image direction using paired training photographs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_aggregate import bootstrap_ci, units
from paper_raw_semantic_readout import leave_photo_out, normalize, top_credit
from paper_render_cache import PinnedFiles, atomic_file, require, save_json


RIDGE = 2.0
ANGLE_MIX = .5
TARGETS = ("photo", "concept")


def tangent(values, axis):
    return F.normalize(values - (values @ axis)[..., None] * axis, dim=-1)


def aligned_tangent(query, axis, center, edit=None, weight=ANGLE_MIX):
    initial = tangent(query, axis)
    translated = initial - center
    centered = torch.where(translated.norm(dim=-1, keepdim=True) > 1e-6, tangent(translated, axis), initial)
    if edit is None:
        return centered
    rotated = torch.lerp(centered, centered @ edit, weight)
    return torch.where(rotated.norm(dim=-1, keepdim=True) > 1e-6, tangent(rotated, axis), centered)


def fit_orthogonal(source, target, axis, ridge=RIDGE):
    width = source.shape[-1]
    axis64 = F.normalize(axis.double(), dim=0)
    reflect = axis64.clone()
    reflect[0] += torch.copysign(torch.ones_like(reflect[0]), reflect[0])
    reflect = F.normalize(reflect, dim=0)
    basis = torch.eye(width, device=source.device, dtype=torch.float64)[:, 1:] - 2. * reflect[:, None] * reflect[1:]
    moment = (source.T @ target / len(source)).double()
    moment = basis.T @ moment @ basis
    moment.diagonal().add_(ridge / (width - 1))
    left, _, right = torch.linalg.svd(moment)
    estimate = (basis @ (left @ right) @ basis.T + axis64[:, None] * axis64).to(source)
    require(torch.allclose(estimate @ axis, axis, rtol=0., atol=2e-5)
            and torch.allclose(estimate.T @ estimate, torch.eye(width, device=estimate.device, dtype=estimate.dtype),
                               rtol=0., atol=2e-5), "Calibration changed the cone axis or lost orthogonality")
    return estimate


def read_inputs(benchmark, pins):
    prep = benchmark / "verified/data/prep"
    source = benchmark / "semantic-energy-check"
    reference = benchmark.parent / "crossfit-eeg-generator/F_train.npy"
    bank = benchmark / "token-rank-check/naming-frozen-bank.npz"
    registry = benchmark / "inputs/image_set/train_photos.csv"
    prevalidated = source / "matching-targets-verified.json"
    paths = dict(split=prep / "paper_split.json", development=prep / "paper_development.json",
                 labels=prep / "train_concept.npy", truth=reference, bank=bank, registry=registry,
                 validation=prevalidated)
    proof = {key: pins.pin(path) for key, path in paths.items()}
    audit = json.loads(prevalidated.read_bytes())
    require(all(proof[k] == audit["input"][k] for k in ("split", "development", "truth", "bank")),
            "Calibration reads different images or EEG rows from the verified export")
    training = np.asarray(json.loads(paths["split"].read_bytes())["train_rows"])
    dev = np.asarray(json.loads(paths["development"].read_bytes())["dev_rows"])
    labels = np.load(paths["labels"], allow_pickle=False)
    groups, counts = np.unique(labels[training], return_inverse=True, return_counts=True)[1:]
    require(training.shape == (13230,) and dev.shape == (200,) and labels.shape == (16540,)
            and len(counts) == 1323 and np.all(counts == 10)
            and set(labels[training]).isdisjoint(labels[dev]), "The concept-disjoint experiment split changed")
    with registry.open(encoding="utf-8-sig", newline="") as stream:
        photographs = list(csv.DictReader(stream))
    require([int(item["row0"]) for item in photographs] == list(range(16540)), "Training photo order changed")
    names = np.array([item["unique_id"] for item in photographs])
    with np.load(bank, allow_pickle=False) as contents:
        text = contents["h14_only_vae_mean"]
        words = contents["concept_id"]
        require(text.shape == (1854, 1024) and len(set(words)) == 1854, "Expected the full frozen vocabulary")
        word_rows = {word: i for i, word in enumerate(words)}
    targets = np.load(reference, mmap_mode="r", allow_pickle=False)
    require(targets.shape == (16540, 1024), "Expected the registered visual fusion space")
    return source, training, dev, targets, groups, names[dev], np.array([word_rows[x] for x in names[dev]]), text, proof, audit


def fit(benchmark, output, device):
    pins = PinnedFiles()
    source, train, held, targets, groups, concepts, name_rows, text, proof, audit = read_inputs(benchmark, pins)
    visual = torch.from_numpy(np.array(targets[train], copy=True)).float().to(device)
    axis = F.normalize(F.normalize(visual).mean(0), dim=0)
    reference = tangent(visual, axis)
    peers = torch.from_numpy(leave_photo_out(reference.cpu().numpy(), groups)).to(reference)
    teacher = {
        "photo": tangent(reference - reference.mean(0), axis),
        "concept": tangent(peers - reference.mean(0), axis),
    }
    cohort_centered = torch.zeros_like(visual)
    centers, raw, manifests = [], [], {}
    model_source = None
    for subject in range(1, 11):
        path = source / f"d2-train-dev-sub-{subject:02d}.npz"
        meta = path.with_suffix(".json")
        receipt = json.loads(meta.read_bytes())
        checksum, signed = pins.pin(path), pins.pin(meta)
        require(receipt["subject"] == subject and receipt["weight_updates"] == 0 and receipt["n_train"] == 13230
                and receipt["repeats"] == [[0, 1, 2, 3]] and receipt["data_sha256"] == checksum == audit["conditions"][str(subject)]
                and all(receipt[k] == proof[k] for k in ("split", "development", "labels")), "Unexpected frozen D2 input")
        if model_source is None:
            model_source = receipt["condition"]
        require(model_source == receipt["condition"], "D2 calibration differs across participant checkpoints")
        with np.load(path, allow_pickle=False) as value:
            require(np.array_equal(value["train_rows"], train) and np.array_equal(value["dev_rows"], held),
                    "D2 vectors use a different photo registration")
            d = torch.from_numpy(value["train_fused"]).float().to(device)
            raw.append(torch.from_numpy(value["dev_fused"]).float().to(device))
        original = tangent(d, axis)
        center = original.mean(0)
        centers.append(center)
        cohort_centered += aligned_tangent(d, axis, center) / 10.
        manifests[str(subject)] = dict(array=checksum, receipt=signed, eeg=receipt["eeg"])
    rotations = {name: fit_orthogonal(cohort_centered, target, axis) for name, target in teacher.items()}
    q = F.normalize(torch.stack(raw), dim=-1)
    alpha = (q @ axis).mean()
    require(float((q @ axis - alpha).abs().max()) < 2e-4 and 0.<float(alpha)<1.,
            "Archived D2 vectors are not on the registered cone")
    beta = (1. - alpha.square()).sqrt()
    estimate = {"d2": q}
    edits = {"center": None, **rotations}
    edits.update({f"{target}-inverse": warp.T for target, warp in rotations.items()})
    for tag, warp in edits.items():
        d = torch.stack([aligned_tangent(x, axis, mean, warp) for x, mean in zip(q, centers, strict=True)])
        estimate[tag] = F.normalize(alpha * axis + beta * d, dim=-1)

    output.mkdir(parents=True, exist_ok=True)
    state = output / "tangent-calibration.pt"
    coefficients = dict(axis=axis.cpu(), centers=torch.stack(centers).cpu(),
                        rotations={k: v.cpu() for k, v in rotations.items()}, alpha=float(alpha))
    atomic_file(state, lambda stream: torch.save(coefficients, stream))
    pictures, text = normalize(targets[held]), normalize(text)
    same_object = np.equal.outer(concepts, concepts)
    index = np.arange(len(held))
    observations = {}
    angles = {}
    for variant, x in estimate.items():
        q_np = x.cpu().numpy()
        scores = q_np @ pictures.T
        names = q_np @ text.T
        observations[variant] = np.stack((top_credit(names, name_rows, 1), top_credit(names, name_rows, 5),
                                         top_credit(scores, index, 1),
                                         top_credit(np.where(same_object, scores, -np.inf), index, 1)), axis=-1)
        angles[variant] = float((x * q).sum(-1).mean())
    comparisons = [(k, "d2") for k in estimate if k != "d2"] + [(label, label + "-inverse") for label in TARGETS]
    metrics = ("name_top1", "name_top5", "photo_top1", "within10")
    paired = {}
    for a, b in comparisons:
        difference = observations[a] - observations[b]
        lo, hi = bootstrap_ci(units(difference, concepts, "dev"))
        paired[f"{a} minus {b}"] = dict(zip(metrics, np.stack((difference.mean((0, 1)), lo, hi), axis=1).tolist(), strict=True))
    result = dict(n=2000, weights=pins.pin(state), model=model_source, training=proof, references=manifests,
                  ridge=RIDGE, angle_mix=ANGLE_MIX, code_sha256=pins.pin(Path(__file__)), cosines=angles,
                  hits={k: dict(zip(metrics, v.sum((0, 1)).tolist(), strict=True)) for k, v in observations.items()}, paired=paired)
    atomic_file(output / "tangent-development-observations.npz",
                lambda stream: np.savez_compressed(stream, row=held, concept_ids=concepts, **observations))
    pins.verify()
    save_json(output / "tangent-calibration.json", result)
    print(json.dumps(result["hits"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    torch.set_num_threads(5)
    fit(args.benchmark, args.output, torch.device(args.device))
