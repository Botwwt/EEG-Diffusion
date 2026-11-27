"""Test which directions in the whole-photo condition carry object identity into the generator."""
import argparse
import inspect
from pathlib import Path

import numpy as np
import torch

from paper_render_cache import PinnedFiles, require, save_json


RANK = 128
ARMS = ("true-object-axes", "true-pca-axes", "wrong-object-axes")


def identity_axes(photos, labels, rank):
    """Separate reproducible object differences from variation across the object's photographs."""
    _, count = torch.unique(labels, sorted=True, return_counts=True)
    per_class = int(count[0])
    require(bool((count == per_class).all()) and per_class >= 4 and per_class % 2 == 0
            and 0 < rank < photos.shape[1], "Expected equally sized pairs of photo groups")
    ordered = photos[torch.argsort(labels, stable=True)].double().reshape(len(count), per_class, -1)
    first = ordered[:, ::2].mean(1)
    second = ordered[:, 1::2].mean(1)
    first -= first.mean(0)
    second -= second.mean(0)
    across = (first.T @ second + second.T @ first) / (2 * len(count))
    residual = (ordered - ordered.mean(1, keepdim=True)).flatten(0, 1)
    within = residual.T @ residual / len(residual)
    within.diagonal().add_(.3 * within.diagonal().mean().clamp_min(1e-8))
    noise = torch.linalg.cholesky(within)
    left = torch.linalg.solve_triangular(noise, across, upper=False)
    white = torch.linalg.solve_triangular(noise, left.T, upper=False).T
    _, signal = torch.linalg.eigh((white + white.T) * .5)
    directional = torch.linalg.solve_triangular(noise.T, signal[:, -rank:], upper=True)
    stable = torch.linalg.qr(directional).Q
    all_centered = photos.double() - photos.double().mean(0)
    _, pca = torch.linalg.eigh(all_centered.T @ all_centered / len(photos))
    return stable.float(), pca[:, -rank:].float()


def matched_updates(directions):
    sizes = [value.norm(dim=1, keepdim=True) for value in directions]
    shared = torch.stack(sizes).amin(0)
    return [value * (shared / norm.clamp_min(1e-8)) for value, norm in zip(directions, sizes, strict=True)]


@torch.inference_mode()
def run(subject):
    import hashlib
    from common import PREP
    from paper_identity_render import load_generator
    from paper_oof_conditions import registered_rows, study_root
    from paper_render import condition, draw_images, generator_inputs, groups, physical_eeg, query_seed
    from paper_token_evidence import project_tokens, render_arms, score_outputs
    from paper_visual_ceiling import official_features

    root, files = study_root(), PinnedFiles()
    registered, labels, _ = registered_rows(PREP)
    train_path = PREP / "F_train.npy"
    training_sha = files.pin(train_path)
    class_sha = files.pin(PREP / "train_concept.npy")
    train = torch.from_numpy(np.load(train_path, mmap_mode="r", allow_pickle=False)[registered].copy()).cuda()
    axes, variance = identity_axes(train, torch.from_numpy(labels).cuda(), RANK)
    axes_sha = [hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest() for value in (axes, variance)]
    truth, rows, references = official_features(PREP, files)
    eeg, acquisition = physical_eeg(subject, "test", files)
    repeats = groups("official-r4")
    original, checkpoint = condition("native-paper", subject, eeg[rows], repeats, features_only=True, files=files)
    original = original.cuda().float()
    truth = truth.cuda()
    other = np.roll(np.arange(len(rows)), len(rows) // 2)
    residual = truth - original
    class_shift = (residual @ axes) @ axes.T
    variance_shift = (residual @ variance) @ variance.T
    wrong_shift = ((truth[other] - original) @ axes) @ axes.T
    updates = matched_updates([class_shift, variance_shift, wrong_shift])
    require(all(torch.isfinite(u).all() for u in updates), "Nonfinite visual diagnostic update")

    pipeline = load_generator(torch.float16, False)
    _, adapter, projection, _, vae = pipeline
    shared = dict(subject=subject, budget="official-r4", repeats=repeats, diagnostic_only=True,
                  diagnostic="correct or incorrect test photograph feature; matched intervention norm",
                  physical_eeg=acquisition, native_checkpoint_sha256=checkpoint,
                  generator_files_sha256=generator_inputs(files), reference_visual_sha256=references,
                  training_photo_sha256=training_sha, training_concept_sha256=class_sha,
                  basis_sha256=axes_sha, axes_per_arm=RANK, training_rows=len(registered),
                  code_sha256=files.pin(Path(__file__)), renderer_sha256=files.pin(Path(inspect.getfile(draw_images))))
    variants = []
    for arm, update in zip(ARMS, updates, strict=True):
        features = original + update
        tokens = project_tokens(features, adapter, projection).cpu().half()
        variants.append((arm, tokens, dict(donor=other.tolist() if arm.startswith("wrong-") else rows)))
    seeds = [query_seed("official", i) for i in rows]
    folders = render_arms(root / "identity-diagnostic" / f"sub-{subject:02d}", variants, rows, seeds, shared, files, pipeline)
    del pipeline, adapter, projection, vae
    torch.cuda.empty_cache()
    score_outputs(folders, "official")
    radius = residual.norm(dim=1).clamp_min(1e-8)
    original_radius = (original - train.mean(0)).norm(dim=1).clamp_min(1e-8)
    files.verify()
    save_json(root / f"identity-diagnostic-sub-{subject:02d}.json",
              dict(n=len(rows), diagnostic_only=True, training_rows=len(registered), rank=RANK, basis_sha256=axes_sha,
                   mean_fraction_full_condition_delta=float((updates[0].norm(dim=1) / radius).mean()),
                   mean_fraction_native_centered_norm=float((updates[0].norm(dim=1) / original_radius).mean()),
                   raw_delta_rms=[float(value.square().mean().sqrt()) for value in (class_shift, variance_shift, wrong_shift)],
                   matched_delta_rms=[float(value.square().mean().sqrt()) for value in updates]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=int, choices=range(5, 11))
    run(parser.parse_args().subject)
