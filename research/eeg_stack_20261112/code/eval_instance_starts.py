"""Compare four sub-03 EEG starting latents with a frozen native Brain-HIVE/Turbo generator."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

ARMS = ("raw_baseline", "residual_head", "within_raw", "within_residual")
LEGACY = Path("/dfs/share/maleiLab/wwt/eeg")
PARENT_SHA = "8e8b1cea28a5cae88f3acc3f136c8a638109949499fdc852ea95a306e0ff9f2b"
EEG_SHA = "85a25128f6b94e3739478c60e089d33f437a8dcbeb31cf136ce9772d2cc1615d"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def relocated(path, root):
    path = Path(path)
    try:
        return root / path.relative_to(LEGACY)
    except ValueError:
        return path if path.is_absolute() else root / path


def load_visual_scorer(root):
    import torch

    archive = root / "assets/evaluation_v272"
    receipt = read_json(archive / "completion.json")
    row, = [item for item in receipt["assets"] if item["name"] == "CLIP_L14"]
    weights = relocated(row["path"], root)
    if sha(weights) != row["sha256"]:
        raise ValueError("CLIP-L/14 checkpoint digest differs from its archive receipt")
    source = root / "code/metric_sources_v272/openai_clip_model.py"
    spec = importlib.util.spec_from_file_location("instance_pilot_clip", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scripted = torch.jit.load(str(weights), map_location="cpu")
    model = module.build_model(scripted.state_dict()).eval().requires_grad_(False).cuda()
    return model, row["sha256"]


def clip_features(model, x):
    import torch
    from torchvision.transforms import Normalize, Resize, InterpolationMode

    x = Resize(224, interpolation=InterpolationMode.BILINEAR, antialias=True)(x.float())
    x = Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])(x)
    features = model.encode_image(x.cuda()).float()
    if not torch.isfinite(features).all():
        raise ValueError("Nonfinite CLIP embeddings")
    return features.cpu().numpy()


def make_conditions(root, rows, native_classes):
    import torch

    parent = root / "runs/HIVE_native_sub03_5096542/full/brain_final.pt"
    recovery = root / "data/recovery_20261108_v236"
    eeg_path = recovery / "sub-03_train_mean4.pt"
    if sha(parent) != PARENT_SHA or sha(eeg_path) != EEG_SHA:
        raise ValueError("Unexpected sub-03 encoder or mean-four EEG archive digest")
    state = torch.load(parent, map_location="cpu", weights_only=True)
    if state["role"] != "full" or state["subject"] != "sub-03":
        raise ValueError("Unexpected encoder role or participant")
    layout = read_json(recovery / "sub-03_layout.json")
    policy = read_json(root / "code/HIVE_native_sub03_protocol_v239.json")
    channels = [layout["channels"].index(name) for name in policy["electrodes"]]
    if len(set(channels)) != 17:
        raise ValueError("Expected 17 distinct posterior electrodes")
    eeg = torch.load(eeg_path, map_location="cpu", weights_only=True, mmap=True)
    x = eeg[torch.tensor(rows, dtype=torch.long)][:, channels, :].float().contiguous()
    if tuple(x.shape) != (len(rows), 17, 250) or not torch.isfinite(x).all():
        raise ValueError("Invalid EEG tensor")
    model = native_classes.classes()["BrainMLP"](4250, 1024, num_layers=1, dropout=0.1)
    model.load_state_dict(state["model"], strict=True)
    model = model.cuda().eval().requires_grad_(False)
    return model(x.cuda()).float()


def rollout(pipe, prompts, tokens, start, noise):
    """Archived s375 Euler-ancestral stepping, 75% first-step hold and coarse tether."""
    import torch.nn.functional as F

    eps, steps = noise[:, 0], noise[:, 1:]
    sigmas = [float(s) for s in pipe.scheduler.sigmas]
    z = start + sigmas[0] * eps
    prompt, pooled, timeids = prompts
    n = len(z)
    for i, timestep in enumerate(pipe.scheduler.timesteps):
        sigma, nxt = sigmas[i:i + 2]
        inp = (z / math.sqrt(1.0 + sigma**2)).to(pipe.unet.dtype)
        pred = pipe.unet(
            inp, timestep.expand(n), encoder_hidden_states=prompt.expand(n, -1, -1),
            added_cond_kwargs={"text_embeds": pooled.expand(n, -1), "time_ids": timeids.expand(n, -1)},
            cross_attention_kwargs={"ip_hidden_states": tokens}, return_dict=False,
        )[0].float()
        up = min(nxt, math.sqrt(nxt**2 * (sigma**2 - nxt**2) / sigma**2))
        down = math.sqrt(max(0.0, nxt**2 - up**2))
        z = z + (down - sigma) * pred + up * steps[:, i]
        anchor = start + nxt * eps
        if i == 0:
            z = 0.75 * anchor + 0.25 * z
        if i in (2, 3):
            low = F.interpolate(F.avg_pool2d(anchor - z, 8), size=(64, 64), mode="bilinear",
                                align_corners=False)
            z = z + (0.11 if i == 2 else 0.02) * low
    return z


def summarize(gallery, generated, groups, rows):
    import numpy as np

    def normalize(z):
        z = np.asarray(z, dtype=np.float64)
        z = z - z.mean(axis=0)
        norm = np.linalg.norm(z, axis=1, keepdims=True)
        if not np.all(np.isfinite(norm) & (norm > 0)):
            raise ValueError("Invalid centered feature norm")
        return z / norm

    scores = normalize(generated) @ normalize(gallery).T
    concept_rows = []
    photo_rows = []
    for concept, indices in groups.items():
        sub = scores[np.ix_(indices, indices)]
        correct = sub.diagonal()
        winners = sub.argmax(axis=1)
        ranks = np.argsort(np.argsort(-sub, axis=1, kind="stable"), axis=1, kind="stable").diagonal() + 1
        competitors = sub.copy()
        np.fill_diagonal(competitors, -np.inf)
        margin = correct - competitors.max(axis=1)
        a, b = np.triu_indices(len(indices), 1)
        swap = correct[a] + correct[b] - sub[a,b] - sub[b,a]
        hits = winners == np.arange(len(indices))
        concept_rows.append(dict(concept=concept, top1=float(hits.mean()),
                                 swap=float((swap > 0).mean()), margin=float(margin.mean()),
                                 swap_margin=float(swap.mean())))
        for i, index in enumerate(indices):
            photo_rows.append(dict(row=int(rows[index]), concept=concept, rank=int(ranks[i]),
                                   hit=bool(hits[i]), margin=float(margin[i])))
    mean = {key: float(np.mean([item[key] for item in concept_rows]))
            for key in ("top1", "swap", "margin", "swap_margin")}
    return dict(mean=mean, concepts=concept_rows, photos=photo_rows), scores


def write_preview(out, indices, rows, identities):
    from PIL import Image, ImageDraw

    arms = ("original",) + ARMS
    tile, gap, left, top = 152, 5, 132, 30
    canvas = Image.new("RGB", (left + len(arms)*(tile+gap), top+len(indices)*(tile+gap)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, name in enumerate(arms):
        x = left+col*(tile+gap)
        draw.text((x, 8), name, fill="black")
        for row_index, i in enumerate(indices):
            y = top+row_index*(tile+gap)
            if col == 0:
                caption = "row %s\nconcept %s" % (rows[i], identities[int(rows[i])]["concept_id"])
                draw.multiline_text((4,y+tile//2), caption, fill="black")
            with Image.open(out / "preview_cache" / f"{name}_{i}.png") as im:
                canvas.paste(im, (x,y))
    temp = out / "preview.tmp"
    canvas.save(temp, format="JPEG", quality=92)
    os.replace(temp, out / "preview.jpg")


def run(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms as T

    root = args.archive.resolve()
    out = args.out
    out.mkdir(exist_ok=True, parents=True)
    if not 1 <= args.batch_size <= 10:
        raise ValueError("Batch size must be 1–10")
    with np.load(args.starts, allow_pickle=False) as data:
        rows = data["rows"]
        if rows.ndim != 1 or rows.dtype.kind not in "iu" or len(np.unique(rows)) != len(rows):
            raise ValueError("Expected unique integer array rows")
        starts = {arm: data[arm].copy() for arm in ARMS}
    if any(x.shape != (len(rows), 4, 4, 4) or not np.isfinite(x).all() for x in starts.values()):
        raise ValueError("Expected finite N x 4 x 4 x 4 scaled latent grids")
    split = read_json(root / "data/recovery_20261108_v236/current_split_manifest.json")["image_rows"]
    if set(rows) & set(split["T"]) or not set(rows) <= set(split["D"] + split["V"]):
        raise ValueError("Pilot rows must be outside encoder training T")
    manifest = root / "data/full_T_recovery_20261108_v238/image_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        identities = {int(d["array_row"]): d for d in csv.DictReader(stream) if d["dataset_split"] == "train"}
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(identities[int(row)]["concept_id"], []).append(i)
    if len(groups) != 20 or any(len(indices) != 10 for indices in groups.values()):
        raise ValueError("Expected 20 complete concepts with ten distinct photos each")
    if set(groups) & {identities[i]["concept_id"] for i in split["T"]}:
        raise ValueError("Pilot concepts overlap encoder training T")
    preview_indices = [i for group in list(groups.values())[:2] for i in group[:2]]
    preview_dir = out / "preview_cache"
    preview_dir.mkdir(exist_ok=True)
    def thumbnail(x, name, i):
        picture = T.ToPILImage()(x.detach().float().cpu().clamp(0, 1)).resize((152,152), Image.BICUBIC)
        destination = preview_dir / f"{name}_{i}.png"
        temp = destination.with_suffix(".tmp")
        picture.save(temp, format="PNG")
        os.replace(temp, destination)
    cfg = dict(starts_sha256=sha(args.starts), script_sha256=sha(Path(__file__)),
               archive=str(root), checkpoint_sha256=PARENT_SHA, rows=rows.tolist(),
               batch_size=args.batch_size, seed=args.seed, arms=list(ARMS), s375=True)
    config_path = out / "settings.json"
    if config_path.exists() and read_json(config_path) != cfg:
        raise ValueError("Existing output settings differ; choose a separate output directory")
    write_json(config_path, cfg)
    start_time = time.monotonic()
    def progress(stage, **details):
        report = dict(stage=stage, seconds=round(time.monotonic()-start_time, 2), **details)
        write_json(out / "progress.json", report)
        print(json.dumps(report), flush=True)

    sys.path.insert(0, str(root / "code"))
    import PKU_HIVE_native_classes_20261109_v239 as classes
    import PKU_native_generator_20261109_v253 as native

    classes.ROOT = native.ROOT = root
    # The loader's file proof stores absolute paths from the old cluster mount.
    native.Path = lambda path: relocated(path, root)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    progress("loading_legacy_models")
    with torch.inference_mode():
        conditions = make_conditions(root, rows, classes)
        pipe, adapter, projector, prompts, host = native.load_native()
        tokens = projector(adapter(conditions.half()))
        if tuple(tokens.shape) != (len(rows), 4, 2048) or not torch.isfinite(tokens).all():
            raise ValueError("Unexpected native condition tokens")
        pipe.scheduler.set_timesteps(4, device="cuda")
        if pipe.scheduler.timesteps.tolist() != [999,749,499,249]:
            raise ValueError("Unexpected SDXL-Turbo timesteps")
        del adapter, projector, conditions
        pipe.text_encoder = pipe.text_encoder_2 = None
        pipe.vae.float().enable_slicing()
        torch.cuda.empty_cache()
        model, clip_sha = load_visual_scorer(root)
        progress("models_ready", gpu=torch.cuda.get_device_name(), ip_layers=host["IP_layers"])
        noise = torch.empty((len(rows),5,4,64,64))
        for concept, indices in groups.items():
            seed = int(hashlib.sha256(f"{args.seed}|{concept}".encode()).hexdigest()[:16], 16) % (2**63-1)
            gen = torch.Generator(device="cpu").manual_seed(seed)
            noise[indices] = torch.randn(5,4,64,64,generator=gen)
        gt = []
        pre = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BILINEAR),
                         T.CenterCrop(512), T.ToTensor()])
        with zipfile.ZipFile(root / "assets/things_images_v254/training_images.zip") as zipped:
            for lo in range(0, len(rows), args.batch_size):
                rgb = []
                for i, row in enumerate(rows[lo:lo+args.batch_size], start=lo):
                    identity = identities[int(row)]
                    raw = zipped.read(identity["image_member"])
                    if hashlib.sha256(raw).hexdigest() != identity["image_sha256"]:
                        raise ValueError(f"Incorrect reference photo {row}")
                    with Image.open(io.BytesIO(raw)) as im:
                        rgb.append(pre(im.convert("RGB")))
                    if i in preview_indices:
                        thumbnail(rgb[-1], "original", i)
                gt.append(clip_features(model, torch.stack(rgb)))
        gt = np.concatenate(gt)
        merged = {}
        for arm in ARMS:
            embeddings = []
            for lo in range(0, len(rows), args.batch_size):
                hi = min(lo+args.batch_size, len(rows))
                cache = out / f"{arm}_{lo:04d}_{hi:04d}.npy"
                examples = [i for i in preview_indices if lo <= i < hi]
                if not cache.exists() or any(not (preview_dir / f"{arm}_{i}.png").exists() for i in examples):
                    grid = torch.as_tensor(starts[arm][lo:hi], dtype=torch.float32, device="cuda")
                    z0 = F.interpolate(grid, size=(64,64), mode="bilinear", align_corners=False)
                    z = rollout(pipe, prompts, tokens[lo:hi], z0, noise[lo:hi].cuda())
                    rgb = (pipe.vae.decode(z/pipe.vae.config.scaling_factor).sample * .5 + .5).clamp(0,1)
                    for i in examples:
                        thumbnail(rgb[i-lo], arm, i)
                    # Score decoded floating-point RGB as in the R4 evaluation.
                    features = clip_features(model, rgb)
                    temp = cache.with_suffix(".tmp")
                    with temp.open("wb") as stream:
                        np.save(stream, features, allow_pickle=False)
                    os.replace(temp, cache)
                block = np.load(cache, allow_pickle=False)
                if block.shape != (hi-lo, gt.shape[1]) or not np.isfinite(block).all():
                    raise ValueError(f"Invalid cached CLIP features in {cache}")
                embeddings.append(block)
                progress("scoring_generated_images", arm=arm, images=hi, total=len(rows))
            result, scores = summarize(gt, np.concatenate(embeddings), groups, rows)
            np.save(out / f"{arm}_scores.npy", scores, allow_pickle=False)
            write_json(out / f"{arm}_ranks.json", result)
            merged[arm] = result
    write_preview(out, preview_indices, rows, identities)
    paired = {}
    rng = np.random.default_rng(20261124)
    bootstrap = rng.integers(len(groups), size=(10000,len(groups)))
    for a, b in (("residual_head","raw_baseline"), ("within_residual","within_raw"),
                 ("within_residual", "raw_baseline")):
        measures = {}
        for k in ("top1", "swap", "margin", "swap_margin"):
            differences = np.array([u[k]-v[k] for u,v in zip(merged[a]["concepts"],merged[b]["concepts"])])
            measures[k] = dict(delta=float(differences.mean()), ci95=np.quantile(
                differences[bootstrap].mean(axis=1), [.025,.975]).tolist())
        paired[a+" minus "+b] = measures
    write_json(out / "result.json", dict(arms={s:r["mean"] for s,r in merged.items()},
               paired=paired, unit="20 paired concepts, fixed sub-03, percentile bootstrap",
               subject="sub-03", photos=len(rows), clip_sha256=clip_sha, generated_per_arm=len(rows),
               nfe=4, nsel=1, original_rgb_targets=True))
    progress("complete", arms=len(ARMS), photos_per_arm=len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261124)
    run(parser.parse_args())
