"""Measure generated-image fidelity after reducing reference relative-token rank."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from paper_render_cache import PinnedFiles, render_transaction, require, save_json
from paper_token_evidence import development, project_tokens, score_outputs, token_parts


RANKS = (32, 128, 512)
SEED = 20261126


@torch.inference_mode()
def run(output, subject):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_photo_top1 import render_inputs
    from paper_render import draw_images, generator_inputs, layout, query_seed, runtime, tensor_fingerprint
    from paper_uniform_baselines import reference_layout

    files = PinnedFiles()
    _, registered = development(PREP, files)
    raw, rows, training, budget, groups, _, split_sha, dev_sha, eeg = render_inputs(subject, "dev", files)
    require(rows == registered, "Changed development order")
    label_path = PREP / "train_concept.npy"
    labels_sha = files.pin(label_path)
    labels = np.load(label_path, allow_pickle=False)
    require(set(labels[rows]).isdisjoint(labels[training]), "Development concepts entered PCA fitting")
    feature_path = PREP / "F_train.npy"
    feature_sha = files.pin(feature_path)
    fused = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    require(fused.shape == (16540, 1024), "Unexpected photo feature matrix")

    source = generator_inputs(files)
    pipe, adapter, projector, prompts, vae = load_generator(torch.float16, False)
    true = project_tokens(torch.from_numpy(fused[rows].copy()), adapter, projector)
    common, relative = token_parts(true)
    _, training_relative = token_parts(project_tokens(torch.from_numpy(fused[training].copy()), adapter, projector))
    flattened = training_relative.flatten(1)
    center = flattened.mean(dim=0)
    centered = flattened - center
    torch.manual_seed(SEED)
    _, _, axes = torch.pca_lowrank(centered, q=576, center=False, niter=4)
    projected_training = centered @ axes
    total = float(centered.square().sum())
    fractions = {str(k): float(projected_training[:, :k].square().sum()) / total for k in RANKS}
    held = relative.flatten(1) - center
    variants = {"native-truth": true.half().cpu()}
    held_retention = {}
    for rank in RANKS:
        coords = held @ axes[:, :rank]
        reconstructed = (center + coords @ axes[:, :rank].mT).reshape_as(relative)
        reconstructed -= reconstructed.mean(dim=1, keepdim=True)
        combined = common + reconstructed
        require(torch.allclose(combined.mean(dim=1, keepdim=True), common, atol=2e-5, rtol=0),
                "Rank truncation changed the reference common token")
        variants[f"rank-{rank}"] = combined.half().cpu()
        held_retention[str(rank)] = float(coords.square().sum() / held.square().sum())
    del training_relative, flattened, centered, projected_training, adapter, projector
    torch.cuda.empty_cache()

    ridge, layout_source = reference_layout(subject, split_sha, files)
    starts = layout(ridge, raw, groups)
    seeds = [query_seed("dev", i) for i in range(len(rows))]
    proof = dict(render_cache_version=2, n=len(rows), row=rows, subject=subject, training="reference-pca", budget=budget,
                 repeats=groups, prompt="", draws=1, unet_calls_per_draw=4, seeds=seeds, diagnostic_only=True,
                 split_sha256=split_sha, development_split_sha256=dev_sha, physical_eeg=eeg,
                 eeg_donor=list(range(len(rows))), layout_strength=.375, layout_source=layout_source,
                 layout_sha256=tensor_fingerprint(starts), common_token_sha256=tensor_fingerprint(common.cpu()),
                 generator_files_sha256=source, photo_feature_sha256=feature_sha, concept_sha256=labels_sha,
                 code_sha256=files.pin(Path(__file__)), runtime=runtime(), pca_seed=SEED,
                 n_training_photos=len(training), training_variance_fraction=fractions, held_variance_fraction=held_retention)
    destinations = []
    for arm, tokens in variants.items():
        folder = output / "images" / "reference-pca" / arm / budget / f"sub-{subject:02d}"
        manifest = dict(proof, arm=arm, token_sha256=tensor_fingerprint(tokens))

        def generate(stage, batch_starts, batch_complete):
            draw_images(stage, pipe, vae, prompts, tokens, seeds, starts,
                        batch_starts=batch_starts, batch_complete=batch_complete)

        render_transaction(folder, manifest, generate, files.verify)
        destinations.append(folder)
        print("token rank render", arm, len(rows), flush=True)
    del pipe, vae
    torch.cuda.empty_cache()
    score_outputs(destinations, "dev")
    save_json(output / "token-rank.json", dict(design=proof, arms={a: json.loads((p / "scores.json").read_bytes())["mean"]
                                               for a, p in zip(variants, destinations, strict=True)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject", type=int, default=1)
    args = parser.parse_args()
    run(args.output, args.subject)
