"""Fit a denoiser of genuine whole-photo Brain-HiVE conditions."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from paper_render_cache import PinnedFiles, atomic_file, require, save_json


SEED = 20261126
UPDATES = 2000
BATCH = 192
HIDDEN = 640
SIGMA_LIMITS = (.06, 2.0)


def fit_coordinates(photos):
    values = photos.double()
    center = values.mean(0)
    centered = values - center
    eigenvalues, axes = torch.linalg.eigh(centered.T @ centered / (len(values) - 1))
    index = torch.arange(axes.shape[1], device=values.device)
    sign = axes[axes.abs().argmax(0), index].sign()
    scale = eigenvalues.clamp_min(1e-5).sqrt()
    return dict(center=center.float(), axes=(axes * sign).float(), scale=scale.float())


def to_coordinates(features, basis):
    return ((features - basis["center"]) @ basis["axes"]) / basis["scale"]


def from_coordinates(coords, basis):
    return (coords * basis["scale"]) @ basis["axes"].T + basis["center"]


class PhotoDenoiser(nn.Module):
    """Estimate a Gaussian-corrupted photograph condition using training photographs only."""

    def __init__(self, channels, width=HIDDEN):
        super().__init__()
        self.input = nn.Linear(channels + 1, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
            for _ in range(3)
        )
        self.output = nn.Linear(width, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy, sigma):
        sigma = sigma.reshape(-1, 1)
        span = (1 + sigma.square()).sqrt()
        hidden = self.input(torch.cat((noisy / span, sigma.log().expand(len(noisy), 1) / 2), dim=-1))
        for block in self.blocks:
            hidden = hidden + .5 * block(hidden)
        return noisy / span.square() + (sigma / span) * self.output(hidden)


def prior_file(root, held):
    return root / ("photo-all.pt" if held == 0 else f"photo-excluding-fold-{held}.pt")


def load_prior(root, held, device, files, split_sha, photo_sha):
    path = prior_file(root, held)
    digest = files.pin(path)
    cache = torch.load(path, map_location=device, weights_only=True)
    source = cache["source"]
    require((source["excluded_fold"], source["split_sha256"], source["photo_features_sha256"],
             source["code_sha256"]) == (held, split_sha, photo_sha, files.pin(Path(__file__))),
            "The visual prior uses a different fitting split or implementation")
    net = PhotoDenoiser(channels=len(cache["basis"]["center"])).to(device)
    net.load_state_dict(cache["weights"])
    return cache["basis"], net.eval().requires_grad_(False), digest


def fit(root, held):
    from common import PREP
    from paper_oof_conditions import registered_rows

    files = PinnedFiles()
    rows, _, folds = registered_rows(PREP)
    if held:
        rows = rows[folds != held]
    source_data = PREP / "F_train.npy"
    source = dict(excluded_fold=held, split_sha256=files.pin(PREP / "paper_split.json"),
                  photo_features_sha256=files.pin(source_data), code_sha256=files.pin(Path(__file__)),
                  row_count=len(rows), seed=SEED + held, updates=UPDATES, batch_size=BATCH,
                  sigma_limits=list(SIGMA_LIMITS))
    out = prior_file(root, held)
    if out.is_file():
        existing = torch.load(out, map_location="cpu", weights_only=True)
        require(existing["source"] == source and np.array_equal(existing["rows"].numpy(), rows),
                "Existing photo prior fitted from different inputs")
        return
    data = torch.from_numpy(np.load(source_data, mmap_mode="r", allow_pickle=False)[rows].copy()).cuda().float()
    require(data.shape == (len(rows), 1024) and torch.isfinite(data).all(), "Invalid fused visual training conditions")
    torch.manual_seed(SEED + held)
    generator = torch.Generator(device="cuda").manual_seed(SEED + held + 971)
    basis = fit_coordinates(data)
    targets = to_coordinates(data, basis)
    net = PhotoDenoiser(data.shape[1]).cuda().train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=.01)
    log_span = torch.tensor(SIGMA_LIMITS, device="cuda").log()
    recent = []
    for step in range(UPDATES):
        index = torch.randint(len(data), (BATCH,), device="cuda", generator=generator)
        clean = targets[index]
        log_sigma = log_span[0] + (log_span[1] - log_span[0]) * torch.rand(BATCH, device="cuda", generator=generator)
        sigma = log_sigma.exp()
        noise = torch.randn(clean.shape, device="cuda", generator=generator)
        noisy = clean + sigma[:, None] * noise
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            recovered = net(noisy, sigma)
            loss = ((recovered - clean).square().mean(1) * (1 + sigma.square()) / sigma.square()).mean()
        loss.backward()
        optimizer.step()
        recent.append(loss.detach())
        if (step + 1) % 250 == 0:
            mean = float(torch.stack(recent[-250:]).mean())
            require(np.isfinite(mean), "Photo denoiser diverged")
            print("photo prior", held, step + 1, "weighted denoising loss", mean, flush=True)
    stats = dict(first_250=float(torch.stack(recent[:250]).mean()), last_250=float(torch.stack(recent[-250:]).mean()))
    weights = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    files.verify()
    atomic_file(out, lambda stream: torch.save(dict(source=source, weights=weights, rows=torch.from_numpy(rows.copy()),
                                                    basis={k: v.cpu() for k, v in basis.items()}, stats=stats), stream))
    save_json(out.with_suffix(".json"), dict(source=source, stats=stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=os.environ.get("EEG_CROSSFIT_ROOT"))
    parser.add_argument("--folds", type=int, choices=range(5), nargs="+", default=list(range(5)))
    args = parser.parse_args()
    require(args.root is not None, "Supply a destination for the visual-only prior")
    for held in args.folds:
        fit(args.root, held)
