"""Check whether real R4 EEG can estimate the visual directions used in the true-photo diagnostic."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_forward_guidance import forward_weights
from paper_fused_identity_diagnostic import RANK, identity_axes
from paper_fused_posterior import summarize_trials
from paper_fused_prior import fit_coordinates, from_coordinates, to_coordinates
from paper_object_margin_encoder import visual_transfer
from paper_oof_conditions import SOURCES
from paper_render_cache import PinnedFiles, require, save_json


BLENDS = (.25, .5, 1.)
RIDGE = (1., .1)
WEIGHTS = (.003, .01, .03, .1, .3, 1., 3., 10.)


def direct_readouts(erp, gaps, target, check):
    scale = erp.std(0).clamp_min(1e-5)
    center = erp.mean(0)
    noise = gaps / scale
    cov = noise.T @ noise / len(noise)
    cov.diagonal().add_(.3 * cov.diagonal().mean())
    factor = torch.linalg.cholesky(cov)

    def white(value):
        return torch.linalg.solve_triangular(factor, ((value - center) / scale).T, upper=False).T.contiguous()

    readout, apply = white(erp), white(check)
    target_center = target.mean(0)
    gram = readout.T @ readout / len(readout)
    cross = readout.T @ (target - target_center) / len(readout)
    for penalty in RIDGE:
        regular = gram.clone()
        regular.diagonal().add_(penalty * gram.diagonal().mean())
        yield penalty, apply @ torch.linalg.solve(regular, cross) + target_center


def restricted_likelihood(train, gaps, observed, original, target, basis):
    weights = forward_weights(target, train, gaps)
    observation = (observed - weights["eeg_center"]) / weights["eeg_scale"]
    whitened = torch.linalg.solve_triangular(weights["noise_factor"], observation.T, upper=False).T
    coords = to_coordinates(original, basis)
    prior = (coords - weights["visual_center"]) / weights["visual_scale"]
    residual = whitened - prior @ weights["weighted"]
    projection = weights["weighted"][-RANK:] / weights["visual_scale"][-RANK:, None]
    gram, rhs = projection @ projection.T, residual @ projection.T
    for weight in WEIGHTS:
        matrix = gram * weight
        matrix.diagonal().add_(1.)
        updated = coords.clone()
        updated[:, -RANK:] += weight * torch.linalg.solve(matrix, rhs.T).T
        yield weight, from_coordinates(updated, basis)


@torch.inference_mode()
def run(prep, eeg_root, photo_path, source, output):
    files = PinnedFiles()
    photo_sha = files.pin(photo_path)
    split = prep / "paper_split.json"
    split_sha = files.pin(split)
    dino_path = source / "object-identity/training-photo-dino.pt"
    dino_sha = files.pin(dino_path)
    bank = torch.load(dino_path, map_location="cpu", weights_only=True)
    rows = bank["rows"].numpy()
    require(len(rows) == 13230 and bank["proof"]["split"] == split_sha
            and rows.tolist() == json.loads(split.read_bytes())["train_rows"], "Incorrect registered targets")
    fused = torch.from_numpy(np.load(photo_path, mmap_mode="r", allow_pickle=False)[rows].copy()).cuda()
    names, dino, sums = bank["labels"].cuda(), bank["image_features"].cuda(), bank["sums"].cuda()
    waveforms, native, sources, folds = {}, {}, {}, None
    for person in SOURCES:
        array = eeg_root / f"sub-{person:02d}_train.npy"
        eeg_sha = files.pin(array)
        condition = source / f"conditions-sub-{person:02d}.pt"
        checkpoint_sha = files.pin(condition)
        cache = torch.load(condition, map_location="cpu", weights_only=True)
        note = cache["sources"]
        require((note["subject"], note["images"], note["eeg"]["array_sha256"], note["split"]) ==
                (person, photo_sha, eeg_sha, split_sha) and np.array_equal(cache["rows"].numpy(), rows),
                "EEG and visual rows differ")
        current_folds = cache["folds"].numpy()
        if folds is None:
            folds = current_folds
        require(np.array_equal(folds, current_folds) and set(folds) == {1, 2, 3, 4}, "Incorrect concept-level folds")
        waveforms[person] = summarize_trials(np.load(array, mmap_mode="r", allow_pickle=False), rows)
        native[person] = cache["out_of_fold"].cuda()
        sources[person] = dict(physical_sha256=eeg_sha, oof_sha256=checkpoint_sha)

    keys = ("native", *[f"{mode}/{penalty}/{blend}" for mode in ("object", "pca") for penalty in RIDGE for blend in BLENDS],
            *[f"pca-likelihood/{weight}" for weight in WEIGHTS])
    totals = {p: dict.fromkeys(keys, 0) for p in SOURCES}
    target_cosines = {p: {key: [] for key in ("object_native", "object_ridge", "pca_native", "pca_ridge")} for p in SOURCES}
    for fold in range(1, 5):
        tr, va = folds != fold, folds == fold
        require(not np.intersect1d(names[tr].cpu().numpy(), names[va].cpu().numpy()).size, "A class entered the held fold")
        object_axes, pca = identity_axes(fused[tr], names[tr], RANK)
        basis = fit_coordinates(fused[tr])
        coordinates = to_coordinates(fused[tr], basis)
        center, vision_center, lift = visual_transfer(fused[tr], dino[tr])
        cats, inverse = torch.unique(names[va], return_inverse=True)
        gallery = F.normalize(F.normalize(sums[cats]) - vision_center)
        own = F.normalize(F.normalize(sums[names[va]] - dino[va]) - vision_center)

        def measure(features):
            v = F.normalize((features - center) @ lift)
            other = (v @ gallery.T).scatter_(1, inverse[:, None], float("-inf"))
            return int(((v * own).sum(1) > other.max(1).values).sum())

        for person in SOURCES:
            original = native[person][va]
            totals[person]["native"] += measure(original)
            erp, gap = waveforms[person]
            for name, axes in (("object", object_axes), ("pca", pca)):
                projected, baseline = original @ axes, center @ axes
                truth = fused[va] @ axes - baseline
                target_cosines[person][name + "_native"].append(float(F.cosine_similarity(projected - baseline, truth).mean()))
                for penalty, pred in direct_readouts(erp[tr], gap[tr], fused[tr] @ axes, erp[va]):
                    if penalty == .1:
                        target_cosines[person][name + "_ridge"].append(float(F.cosine_similarity(pred - baseline, truth).mean()))
                    delta = (pred - projected) @ axes.T
                    for blend in BLENDS:
                        totals[person][f"{name}/{penalty}/{blend}"] += measure(original + blend * delta)
            for weight, features in restricted_likelihood(erp[tr], gap[tr], erp[va], original, coordinates, basis):
                totals[person][f"pca-likelihood/{weight}"] += measure(features)
        print("finished fold", fold, flush=True)
    files.verify()
    scores = {key: sum(p[key] for p in totals.values()) for key in keys}
    save_json(output, dict(n=len(rows) * len(SOURCES), rank=RANK, scores=scores, per_subject=totals,
                           target_cosines=target_cosines, split_sha256=split_sha, photo_sha256=photo_sha,
                           dino_sha256=dino_sha, sources=sources, code_sha256=files.pin(Path(__file__))))
    print(scores, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--eeg-root", type=Path, required=True)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.prep, args.eeg_root, args.photos, args.source_root, args.output)
