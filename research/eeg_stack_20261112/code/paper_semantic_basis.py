"""Construct SDXL text coordinates prioritized by the paired training images."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch


STREAM_WEIGHTS = (.20, .20, .60)
VISUAL_RIDGE = .40
RANK = 160


def predictable_basis(gram, visual, rank):
    """Orthonormal text axes with the largest regularized predictable energy."""
    features = visual - visual.mean(0)
    features = features / features.square().mean(0).sqrt().clamp_min(1e-8)
    covariance = features.T @ features / len(features)
    covariance.diagonal().add_(VISUAL_RIDGE)
    leverage = features @ torch.linalg.solve(covariance, features.T / len(features))
    predicted_gram = leverage @ gram @ leverage.T
    values, vectors = torch.linalg.eigh(predicted_gram.cpu().double())
    selected = values[-rank:].flip(0)
    if len(selected) != rank or selected[-1] <= 1e-10:
        raise ValueError("The training photographs do not support the requested text rank")
    mapped = leverage.T @ vectors[:, -rank:].flip(1).to(leverage)
    return mapped / selected.sqrt().to(leverage)


def weighted_gram(blocks, device):
    gram = torch.zeros(len(blocks[0]), len(blocks[0]), device=device)
    centers = []
    for values, weight in zip(blocks, STREAM_WEIGHTS, strict=True):
        centered_gram = torch.zeros_like(gram)
        center = []
        for chunk in values.flatten(1).split(4096, dim=1):
            data = chunk.to(device)
            mean = data.mean(0)
            residual = data - mean
            centered_gram.addmm_(residual, residual.T)
            center.append(mean.cpu())
        energy = centered_gram.diagonal().mean().item()
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError("Degenerate training text state")
        gram.add_(centered_gram, alpha=weight / energy)
        centers.append(torch.cat(center).reshape(values.shape[1:]))
    return gram, centers


@torch.inference_mode()
def build(output):
    from common import PREP, TURBO
    from paper_render_cache import PinnedFiles, atomic_file, save_json
    from paper_soft_text import deploy_states, encode_names, registry

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    device = torch.device("cuda")
    rows, labels, names, source = registry()
    path = PREP / "F_train.npy"
    pins = PinnedFiles()
    proof = dict(inputs=source, photo_features=pins.pin(path), code=pins.pin(Path(__file__)), rank=RANK,
                 weights=STREAM_WEIGHTS, visual_ridge=VISUAL_RIDGE)
    photo = torch.from_numpy(np.load(path, mmap_mode="r", allow_pickle=False)[rows].copy()).float().cuda()
    prototypes = photo.new_zeros(len(names), photo.shape[1])
    label = labels.cuda()
    prototypes.index_add_(0, label, photo)
    prototypes /= torch.bincount(label)[:, None]
    del photo
    blocks, empty, text_sources = encode_names(names, TURBO, device)
    gram, centers = weighted_gram(blocks, device)
    dual = predictable_basis(gram, prototypes, RANK)
    coordinate = (gram @ dual).cpu()
    basis = []
    for block, mean in zip(blocks, centers, strict=True):
        projected = []
        for part, center in zip(block.flatten(1).split(4096, 1), mean.flatten().split(4096), strict=True):
            projected.append((dual.T @ (part.cuda() - center.cuda())).cpu())
        basis.append(torch.cat(projected, dim=1).reshape(RANK, *block.shape[1:]))
    states = deploy_states(dict(mean=centers, basis=basis), empty)
    retained = float(coordinate.square().sum() / gram.diagonal().sum().cpu())
    if not 0 < retained <= 1.001:
        raise ValueError("The semantic basis does not define an orthogonal projection")
    proof.update(text_sources_sha256=text_sources, retained_variance=retained)
    pins.verify()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_file(output, lambda f: torch.save(dict(meta=proof, final_codes=coordinate, states=states), f))
    save_json(output.with_suffix(".json"), dict(proof, checkpoint_sha256=pins.pin(output)))
    print("predictable SDXL text basis", RANK, retained, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
