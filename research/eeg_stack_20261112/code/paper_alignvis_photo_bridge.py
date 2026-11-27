"""Check if a nonlinear visual bridge recovers the semantic information in AlignVis features."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_aggregate import bootstrap_ci, units
from paper_alignvis_readout import PHOTO_RIDGE, unit
from paper_alignvis_teacher import REVISION
from paper_fusion_siglip_probe import measure, official_conditions, predict, ridge_system, solve_ridge
from paper_render_cache import PinnedFiles, require, save_json
from paper_tangent_calibration import read_inputs


TEACHERS = ("synclr-base", "synclr-aligned", "clip-base", "clip-aligned", "dinov2-base", "dinov2-aligned")
EPOCHS = 24
WIDTH = 768
BATCH = 256
SEED = 20261127


class PhotoBridge(nn.Module):
    def __init__(self, fit):
        super().__init__()
        weight, mean_x, mean_y = fit
        self.register_buffer("weight", weight)
        self.register_buffer("mean_x", mean_x.reshape(1, -1))
        self.register_buffer("mean_y", mean_y.reshape(1, -1))
        self.head = nn.Sequential(nn.Linear(weight.shape[0], WIDTH), nn.GELU(), nn.Linear(WIDTH, weight.shape[1]))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, value):
        centered = value - self.mean_x
        return centered @ self.weight + self.mean_y + self.head(centered)


def train_bridge(x, y, order, device):
    fit = solve_ridge(ridge_system(x[None], y), PHOTO_RIDGE)
    torch.manual_seed(SEED)
    bridge = PhotoBridge(fit).to(device)
    with torch.no_grad():
        torch.testing.assert_close(F.normalize(bridge(x[:16]), dim=-1), predict(x[None, :16], fit)[0])
    optimizer = torch.optim.AdamW(bridge.head.parameters(), lr=8e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(order))
    scores = []
    for epoch in order:
        total = 0.
        for idx in epoch.split(BATCH):
            output = bridge(x[idx]).reshape(len(idx), 2, 1024)
            actual = y[idx].reshape(len(idx), 2, 1024)
            loss = 1. - F.cosine_similarity(output, actual, dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(idx)
        scheduler.step()
        scores.append(float(total / len(x)))
    return bridge.eval(), fit, scores


def run(benchmark, archive, output, device):
    pins = PinnedFiles()
    source, train, dev, fused, _, _, _, _, proof, _ = read_inputs(benchmark, pins)
    prep = benchmark / "verified/data/prep"
    fresh = np.asarray(json.loads((prep / "paper_split.json").read_bytes())["fresh_rows"])
    require(len(fresh) == 500 and set(fresh).isdisjoint(np.concatenate((train, dev))), "Held-out scope changed")
    author = json.loads(archive.with_suffix(".json").read_bytes())
    archive_sha = pins.pin(archive)
    require(author["revision"] == REVISION and author["array_sha256"] == archive_sha
            and author["photo_registries"]["train"] == proof["registry"], "Unexpected frozen image release")
    siglip_path = benchmark / "token-rank-check/siglip-independent/siglip-fused-evaluation-bridge.pt"
    siglip_sha = pins.pin(siglip_path)
    reviewer = json.loads(siglip_path.with_name("siglip-fused-probe.json").read_bytes())
    assets = benchmark / "inputs/image_set"
    word_ids = assets / "concepts1854.csv"
    require(siglip_sha == reviewer["evaluation_bridge_sha256"]
            and pins.pin(word_ids) == reviewer["vision_and_text"]["test"]["concepts"]
            and pins.pin(assets / "test_photos.csv") == author["photo_registries"]["test"], "Scoring inputs changed")
    with word_ids.open(newline="", encoding="utf-8-sig") as words:
        siglip_lookup = {row["unique_id"]: i for i, row in enumerate(csv.DictReader(words))}
    siglip = torch.load(siglip_path, weights_only=True, map_location=device)
    text_path = source / "h14-text-bank.npz"
    with np.load(text_path, allow_pickle=False) as text:
        clip_lookup = {name: i for i, name in enumerate(text["concept_id"])}
        bank = unit(text["text"], device)
    require(len(clip_lookup) == len(siglip_lookup) == 1854, "Require the full open vocabulary")
    official = benchmark / "token-rank-check/fused-r80/official-r80-fused.npz"
    require(pins.pin(official) == reviewer["input"]["official_archive"], "Different official photo features")
    _, official_fused, _ = official_conditions(official, device)

    render_rows = {"dev": ("train", dev), "fresh": ("train", fresh), "official": ("test", np.arange(200))}
    rng = torch.Generator().manual_seed(SEED)
    order = [torch.randperm(len(train), generator=rng).to(device) for _ in range(EPOCHS)]
    outputs = {panel: {} for panel in render_rows}
    all_names, histories, maximum = {}, {}, {}
    with np.load(archive, allow_pickle=False) as data:
        train_fused = unit(fused, device)
        h14 = {phase: unit(data[f"{phase}/h14"], device) for phase in ("train", "test")}
        train_target = torch.cat((train_fused[train], h14["train"][train]), dim=-1)
        for panel, (phase, rows) in render_rows.items():
            names = data[f"{phase}_word"][rows]
            require(set(names).isdisjoint(data["train_word"][train]), "A scoring concept was used in bridge training")
            all_names[panel] = names
        for name in TEACHERS:
            inputs = {phase: unit(data[f"{phase}/{name}"], device) for phase in ("train", "test")}
            nonlinear, linear, history = train_bridge(inputs["train"][train], train_target, order, device)
            histories[name] = history
            with torch.inference_mode():
                for panel, (phase, idx) in render_rows.items():
                    selected = inputs[phase][idx]
                    labels = all_names[panel]
                    first = np.asarray([clip_lookup[x] for x in labels])
                    second = np.asarray([siglip_lookup[x] for x in labels])
                    ref = official_fused if panel == "official" else train_fused[idx]
                    if panel not in maximum:
                        maximum[panel] = np.concatenate((
                            measure(h14[phase][idx], bank, np.arange(len(clip_lookup)), first),
                            measure(predict(ref[None], siglip["readout"])[0], siglip["text_bank"],
                                    siglip["names"].cpu().numpy(), second),
                        ), axis=-1)
                    for condition, features in (("linear", predict(selected[None], linear)[0]), ("nonlinear", nonlinear(selected))):
                        a, b = (F.normalize(part, dim=-1) for part in features.split(1024, dim=-1))
                        v1 = measure(b, bank, np.arange(len(clip_lookup)), first)
                        v2 = measure(predict(a[None], siglip["readout"])[0], siglip["text_bank"],
                                     siglip["names"].cpu().numpy(), second)
                        outputs[panel][f"{name}/{condition}"] = np.concatenate((v1, v2), axis=-1)
            print(name, "final train loss", history[-1], flush=True)
    rows = {}
    for panel, data in outputs.items():
        comparisons = {}
        for family in ("synclr", "clip", "dinov2"):
            for condition in ("linear", "nonlinear"):
                a, b = (f"{family}-{role}/{condition}" for role in ("aligned", "base"))
                diff = data[a] - data[b]
                lo, hi = bootstrap_ci(units(diff[None], all_names[panel], panel), resamples=6000)
                comparisons[f"{a} minus {b}"] = np.stack((diff.mean(0), lo, hi), axis=1).tolist()
        rows[panel] = dict(n=len(all_names[panel]), paired=comparisons, upper_target=maximum[panel].sum(0).tolist(),
                           credits={k: v.sum(0).tolist() for k, v in data.items()})
    pins.verify()
    save_json(output, dict(metrics=("h14_name_top1", "h14_name_top5", "siglip2_name_top1", "siglip2_name_top5"),
                           seed=SEED, epochs=EPOCHS, width=WIDTH, batch_size=BATCH, proof=proof,
                           train_photos=len(train), archive=archive_sha, siglip=siglip_sha, text=pins.pin(text_path),
                           code=pins.pin(Path(__file__)), training=histories, results=rows))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("benchmark", type=Path)
    cli.add_argument("archive", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--device", default="cuda")
    args = cli.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run(args.benchmark, args.archive, args.output, args.device)
