"""Fit an image-only feature flow and test EEG source-space optimization."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_aggregate import bootstrap_ci, require, units
from paper_raw_semantic_readout import METRICS, measure, normalize, read_json, sha


SEED = 20261127
EPOCHS = 30
STEPS = 20
ODE_STEPS = 6
SOURCE_COUNT = 4


class Velocity(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension + 3, 768), nn.SiLU(),
                                 nn.Linear(768, 768), nn.SiLU(), nn.Linear(768, dimension))

    def forward(self, x, t):
        time = torch.cat((t, torch.sin(math.pi * t), torch.cos(math.pi * t)), dim=-1)
        return self.net(torch.cat((x, time), dim=-1))


def source_to_image(net, source, center, unwhiten):
    if net is None:
        return source @ unwhiten + center
    x = source
    step = 1 / ODE_STEPS
    for k in range(ODE_STEPS):
        v = net(x, source.new_full((len(source), 1), k * step))
        x = x + step * v
    return x @ unwhiten + center


def fit_flow(image, device):
    """Learn a velocity field only from training images, without class labels or EEG."""
    torch.manual_seed(SEED)
    x = torch.from_numpy(image).to(device)
    center = x.mean(0)
    centered = x - center
    vals, basis = torch.linalg.eigh(centered.T @ centered / len(x))
    scale = vals.clamp_min(.01 * vals.mean()).sqrt()
    whiten = basis / scale
    unwhiten = (basis * scale).T.contiguous()
    x = centered @ whiten
    net = Velocity(x.shape[-1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=6e-4, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=8e-5)
    for epoch in range(EPOCHS):
        errors = []
        for rows in torch.randperm(len(x), device=device).split(256):
            target = x[rows]
            source = torch.randn_like(target)
            t = torch.rand((len(rows), 1), device=device)
            ve = net(torch.lerp(source, target, t), t)
            loss = F.mse_loss(ve, target - source)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2., error_if_nonfinite=True)
            opt.step()
            errors.append(float(loss.detach()))
        schedule.step()
        print("Image-only flow epoch", epoch + 1, float(np.mean(errors)), flush=True)
    return net.eval().requires_grad_(False), center, unwhiten


def optimize_sources(net, query, sources, center, unwhiten):
    """Use the same source radii and optimizer with and without the learned velocity."""
    count, dim = sources.shape
    radii = sources.norm(dim=-1, keepdim=True)
    query = F.normalize(query, dim=-1)
    initial = sources.expand(len(query), count, dim).clone()
    state = initial.requires_grad_()
    opt = torch.optim.Adam([state], lr=.12)
    for _ in range(STEPS):
        source = F.normalize(state, dim=-1) * radii
        v = source_to_image(net, source.flatten(0, 1), center, unwhiten).reshape_as(source)
        score = (F.normalize(v, dim=-1) * query[:, None]).sum(-1)
        opt.zero_grad(set_to_none=True)
        (-score.mean()).backward()
        opt.step()
    with torch.no_grad():
        source = F.normalize(state, dim=-1) * radii
        final = source_to_image(net, source.flatten(0, 1), center, unwhiten).reshape_as(source)
        score = (F.normalize(final, dim=-1) * query[:, None]).sum(-1)
        return final[torch.arange(len(query), device=query.device), score.argmax(1)].cpu().numpy()


def fit(benchmark, output, device):
    checks = benchmark / "semantic-energy-check"
    reference = read_json(checks / "matching-targets-verified.json")
    prep = benchmark / "verified/data/prep"
    split_path, dev_path = (prep / f"paper_{name}.json" for name in ("split", "development"))
    fused_path = benchmark.parent / "crossfit-eeg-generator/F_train.npy"
    label_path = prep / "train_concept.npy"
    bank_path = benchmark / "token-rank-check/naming-frozen-bank.npz"
    proof = {k: sha(p) for k, p in dict(split=split_path, development=dev_path,
                                        truth=fused_path, labels=label_path, bank=bank_path).items()}
    require(proof == reference["input"], "Inputs differ from the existing held-concept benchmark")
    train = np.asarray(read_json(split_path)["train_rows"])
    dev = np.asarray(read_json(dev_path)["dev_rows"])
    labels = np.load(label_path, allow_pickle=False)
    require(train.shape == (13230,) and dev.shape == (200,)
            and set(labels[train]).isdisjoint(labels[dev]), "Expected held-out visual concepts")
    images = np.load(fused_path, mmap_mode="r", allow_pickle=False)
    require(images.shape == (16540, 1024) and images.dtype == np.float32, "Invalid fused images")
    net, center, scale = fit_flow(images[train], device)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "flow.pt"
    torch.save(dict(state=net.cpu().state_dict(), center=center.cpu(), unwhiten=scale.cpu(),
                    proof=proof, seed=SEED, epochs=EPOCHS), checkpoint)
    net.to(device)
    sources = torch.randn((SOURCE_COUNT, images.shape[-1]), generator=torch.Generator().manual_seed(SEED + 1)).to(device)
    queries = []
    for subject in range(1, 11):
        path = checks / f"d2-train-dev-sub-{subject:02}.npz"
        require(sha(path) == reference["conditions"][str(subject)], "Changed D2 condition")
        with np.load(path, allow_pickle=False) as z:
            require(np.array_equal(z["dev_rows"], dev), "Unpaired query order")
            queries.append(z["dev_fused"])
    queries = np.stack(queries)
    require(queries.shape == (10, 200, 1024), "Invalid D2 conditions")
    flattened = torch.from_numpy(queries.reshape(-1, 1024)).to(device)
    candidates = {"direct": queries}
    for name, mapping in (("gaussian", None), ("flow", net)):
        blocks = []
        for i, q in enumerate(flattened.split(40), 1):
            blocks.append(optimize_sources(mapping, q, sources, center, scale))
            if i % 10 == 0:
                print(name, i, "batches", flush=True)
        candidates[name] = np.concatenate(blocks).reshape(queries.shape)
    # The same initial sources are used for every query, so a different-concept donor
    # changes only the brain condition. No scoring targets are used to optimize a source.
    donors = (np.arange(len(dev)) + len(dev) // 2) % len(dev)
    require(np.all(labels[dev] != labels[dev][donors]), "Wrong EEG donors must depict different concepts")
    candidates["flow-wrong"] = candidates["flow"][:, donors]
    prior_pack = output / "prior-conditions.npz"
    np.savez_compressed(prior_pack, rows=dev, donors=donors, **candidates)
    with (benchmark / "inputs/image_set/train_photos.csv").open(encoding="utf-8-sig", newline="") as stream:
        registry = list(csv.DictReader(stream))
    concepts = np.array([registry[row]["unique_id"] for row in dev])
    with np.load(bank_path, allow_pickle=False) as z:
        bank, ids = normalize(z["h14_only_vae_mean"]), z["concept_id"]
    index = {name: i for i, name in enumerate(ids)}
    names = np.array([index[c] for c in concepts])
    truth = normalize(images[dev])
    scores = {name: measure(feature, bank, truth, names) for name, feature in candidates.items()}
    paired = {}
    for a, b in (("flow", "direct"), ("flow", "gaussian"), ("flow", "flow-wrong")):
        d = scores[a] - scores[b]
        lo, hi = bootstrap_ci(units(d, concepts, "dev"))
        paired[f"{a} minus {b}"] = {m: [float(d[:, :, j].mean()), float(lo[j]), float(hi[j])]
                                    for j, m in enumerate(METRICS)}
    result = dict(protocol="Held-concept R4 image-only flow diagnostic in all 1024 fused dimensions",
                  queries=int(np.prod(queries.shape[:-1])), proof=proof, seed=SEED, epochs=EPOCHS,
                  source_restarts=SOURCE_COUNT, source_steps=STEPS, ode_steps=ODE_STEPS,
                  code_sha256=sha(Path(__file__)), checkpoint_sha256=sha(checkpoint),
                  source_conditions_sha256=sha(prior_pack), paired=paired,
                  hits={a: dict(zip(METRICS, x.sum((0, 1)).tolist(), strict=True)) for a, x in scores.items()},
                  mean_cosine_to_eeg={a: float((normalize(x) * normalize(queries)).sum(-1).mean())
                                      for a, x in candidates.items()})
    (output / "prior-readout-verified.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output / "prior-readout-scores.npz", row=dev, concept=concepts, **scores)
    print(json.dumps(result["hits"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(6)
    fit(args.benchmark, args.output, args.device)
