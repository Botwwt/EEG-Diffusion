"""Build train-only paired foreground interventions and isolate their object direction."""

import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from paper_horizon_readout import registration
from paper_middle_teacher import CONFIG_SHA, MODEL_SHAS, image_processor
from paper_outline_routing import MASK_RECEIPTS
from paper_render_cache import PinnedFiles, atomic_file, require, save_json


SIDE = 224
FOREGROUND_SIDE = 160
BACKGROUND_GRID = 14
RIDGE = .05
PAIR_SEED = 20261127
ARMS = ("natural-pair", "shared-background", "background-residual", "unrelated-residual")


def blend_contexts(objects, alpha, backgrounds):
    return objects[:, :, None] * alpha[:, :, None] + backgrounds[:, None] * (1 - alpha[:, :, None])


def matched_pairs(layout, groups, device):
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(PAIR_SEED)
    ordered = np.argsort(groups, kind="stable").reshape(-1, 10)
    shuffled = np.take_along_axis(ordered, rng.random(ordered.shape).argsort(axis=1), axis=1)
    left, right = shuffled[:, :5].ravel(), shuffled[:, 5:].ravel()
    x = torch.from_numpy(layout).to(device)
    cost = torch.cdist(x[left], x[right]).square().div_(x.shape[1]).cpu().numpy()
    cost[groups[left, None] == groups[None, right]] = np.inf
    i, j = linear_sum_assignment(cost)
    matches = np.stack((left[i], right[j]), axis=-1)
    require(len(matches) * 2 == len(groups) and len(np.unique(matches)) == len(groups)
            and np.all(groups[matches[:, 0]] != groups[matches[:, 1]]), "Must pair each photo with one other object")
    return matches, float(cost[i, j].mean())


def choose_backgrounds(groups, pairs):
    require(len(np.unique(groups)) >= 4, "Two distinct source backgrounds need four or more concepts")
    rng = np.random.default_rng(PAIR_SEED + 1)
    backgrounds = np.empty_like(pairs)
    for i, (a, b) in enumerate(pairs):
        forbidden = {groups[a], groups[b]}
        for context in range(2):
            candidate = int(rng.integers(len(groups)))
            while groups[candidate] in forbidden:
                candidate = int(rng.integers(len(groups)))
            backgrounds[i, context] = candidate
            forbidden.add(groups[candidate])
    return backgrounds


def object_directions(views, original, ridge=RIDGE, nuisance_indices=None):
    distinction = views.mean(dim=2)
    delta = distinction[:, 0] - distinction[:, 1]
    different_backgrounds = F.normalize(views[:, :, 0] - views[:, :, 1], dim=-1)
    if nuisance_indices is not None:
        different_backgrounds = different_backgrounds[nuisance_indices]
    axes = different_backgrounds.transpose(1, 2)
    gram = different_backgrounds @ axes + ridge * torch.eye(2, device=views.device)
    weights = torch.linalg.solve(gram, different_backgrounds @ delta.unsqueeze(-1))
    residual = delta - (axes @ weights).squeeze(-1)
    return tuple(F.normalize(target, dim=-1) for target in (original[:, 0] - original[:, 1], delta, residual))


def foreground_view(image, soft):
    active = soft >= max(1, int(soft.max()) // 2)
    ys, xs = np.where(active)
    require(len(xs) > 0, "A training mask has no foreground")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    width, height = x1 - x0, y1 - y0
    factor = FOREGROUND_SIDE / max(width, height)
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    left, top = (SIDE - size[0]) // 2, (SIDE - size[1]) // 2
    view = Image.new("RGB", (SIDE, SIDE))
    alpha = Image.new("L", (SIDE, SIDE))
    view.paste(image.crop((x0, y0, x1, y1)).resize(size, Image.Resampling.BICUBIC), (left, top))
    alpha.paste(Image.fromarray(soft[y0:y1, x0:x1]).resize(size, Image.Resampling.BILINEAR), (left, top))
    return np.asarray(view), np.asarray(alpha)


def background_view(image, soft):
    pixels = np.asarray(image, dtype=np.float32)
    spare = 1. - soft.astype(np.float32) / 255.
    fallback = (pixels * spare[..., None]).sum(axis=(0, 1)) / max(spare.sum(), 1e-6)
    grid = BACKGROUND_GRID
    tile = SIDE // grid
    weights = spare.reshape(grid, tile, grid, tile).mean(axis=(1, 3))
    sums = (pixels * spare[..., None]).reshape(grid, tile, grid, tile, 3).mean(axis=(1, 3))
    area = np.where(weights[..., None] >= .25, sums / weights[..., None].clip(1e-6), fallback)
    return area.round().clip(0, 255).astype(np.uint8)


@torch.inference_mode()
def prepare(benchmark, model_dir, masks, panel, output, batch_size):
    from transformers import CLIPVisionModelWithProjection
    from paper_semantic_contrast import data_registry

    pins = PinnedFiles()
    prep = benchmark / "verified/data/prep"
    (train, development, _), labels, registry = registration(prep, pins)
    _, training, dev, _, _, _, _, layout, sources = data_registry(benchmark)
    require(np.array_equal(training, train) and np.array_equal(dev, development), "Visual layout has different registered rows")
    rows = train if panel == "train" else development
    pair, distance = matched_pairs(layout[rows], labels[rows], "cuda")
    background = choose_backgrounds(labels[rows], pair)
    directory = benchmark / "inputs/image_set"
    paths = dict(archive=directory / "training_images.zip", registry=directory / "train_photos.csv",
                 mask=masks, weights=model_dir / "model.safetensors", config=model_dir / "config.json", code=Path(__file__),
                 clip_preprocess=Path(__file__).with_name("paper_middle_teacher.py"))
    proof = {k: pins.pin(path) for k, path in paths.items()}
    require(proof["mask"] == MASK_RECEIPTS[panel] and proof["weights"] in MODEL_SHAS
            and proof["config"] == CONFIG_SHA and proof["registry"] == sources["photo_registry"],
            "Training masks, visual identity list, or frozen model changed")
    with paths["registry"].open(encoding="utf-8-sig", newline="") as stream:
        photos = list(csv.DictReader(stream))
    with np.load(masks, allow_pickle=False) as values:
        soft = values["masks"]
        require(np.array_equal(values["rows"], rows) and values["valid"].all()
                and soft.shape == (len(rows), 160, 160) and soft.dtype == np.uint8, "Mask bank has another row order")

    foreground = np.empty((len(rows), SIDE, SIDE, 3), dtype=np.uint8)
    alphas = np.empty((len(rows), SIDE, SIDE), dtype=np.uint8)
    scene = np.empty((len(rows), BACKGROUND_GRID, BACKGROUND_GRID, 3), dtype=np.uint8)
    with zipfile.ZipFile(paths["archive"]) as images:
        require(len(images.namelist()) == len(set(images.namelist())), "Ambiguous image archive")
        for i, row in enumerate(rows):
            entry = photos[row]
            raw = images.read(entry["archive_member"])
            require(sha256(raw).hexdigest() == entry["image_sha256"], "Image content differs from its registered SHA")
            with Image.open(io.BytesIO(raw)) as picture:
                require(picture.size == (500, 500), "Training mask requires square source pixels")
                image = picture.convert("RGB").resize((SIDE, SIDE), Image.Resampling.BICUBIC)
            alpha = np.asarray(Image.fromarray(soft[i]).resize((SIDE, SIDE), Image.Resampling.BILINEAR))
            foreground[i], alphas[i] = foreground_view(image, alpha)
            scene[i] = background_view(image, alpha)
            if (i + 1) % 2000 == 0:
                print(panel, "registered source photos", i + 1, "/", len(rows), flush=True)
    del soft

    tower = CLIPVisionModelWithProjection.from_pretrained(model_dir, local_files_only=True, use_safetensors=True,
                                                          torch_dtype=torch.float16).cuda().eval()
    processor = image_processor()
    mu = torch.tensor(processor.image_mean, device="cuda")[None, :, None, None]
    sigma = torch.tensor(processor.image_std, device="cuda")[None, :, None, None]
    context_views = np.empty((len(pair), 2, 2, 1024), dtype=np.float32)
    ordered = np.arange(len(pair))
    for first in range(0, len(ordered), batch_size):
        block = slice(first, first + batch_size)
        a = torch.from_numpy(foreground[pair[block]]).cuda().float().permute(0, 1, 4, 2, 3) / 255
        m = torch.from_numpy(alphas[pair[block]]).cuda().float().unsqueeze(2) / 255
        b = torch.from_numpy(scene[background[block]]).cuda().float().permute(0, 1, 4, 2, 3) / 255
        b = F.interpolate(b.flatten(0, 1), size=(SIDE, SIDE), mode="bilinear", align_corners=False).reshape_as(a)
        x = blend_contexts(a, m, b).flatten(0, 2)
        result = [tower(pixel_values=((part - mu) / sigma).half()).image_embeds.float() for part in x.split(32)]
        context_views[block] = F.normalize(torch.cat(result)).reshape(len(a), 2, 2, -1).cpu().numpy()
        if first % (batch_size * 80) == 0 or first + len(a) == len(pair):
            print(panel, "composites", 4 * (first + len(a)), "/", 4 * len(pair), flush=True)
    require(np.isfinite(context_views).all(), "Nonfinite visual features")
    pins.verify()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_file(output, lambda stream: np.savez_compressed(stream, rows=rows, pair=pair,
                                                           background=background, views=context_views))
    save_json(output.with_suffix(".json"), dict(registration=registry, sources=proof, layout=sources["visual"], panel=panel,
                                               scope="training-only" if panel == "train" else "visual-diagnostic-only",
                                               pairs=len(pair), mean_layout_distance=distance, seed=PAIR_SEED,
                                               output_sha256=pins.pin(output)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("benchmark", type=Path)
    cli.add_argument("model", type=Path)
    cli.add_argument("masks", type=Path)
    cli.add_argument("panel", choices=("dev", "train"))
    cli.add_argument("output", type=Path)
    cli.add_argument("--batch-size", type=int, default=16)
    args = cli.parse_args()
    prepare(args.benchmark, args.model, args.masks, args.panel, args.output, args.batch_size)
