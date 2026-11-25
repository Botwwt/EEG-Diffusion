"""Measure the frozen generator with real visual features or a constant training mean on development."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL

from common import PREP, ROOT, TURBO, write_json
from paper_render import draw_images, query_seed, sha
from paper_score import score
from paper_training import generator


@torch.inference_mode()
def run():
    rows = json.loads((PREP / "paper_development.json").read_bytes())["dev_rows"]
    train = json.loads((PREP / "paper_split.json").read_bytes())["train_rows"]
    if len(rows) != 200 or not set(rows).isdisjoint(train):
        raise ValueError("Incompatible registered development photos")
    file = PREP / "F_train.npy"
    source = np.load(file, mmap_mode="r")
    ref = torch.from_numpy(source[rows].copy()).float()
    mean = torch.from_numpy(source[train].copy()).float().mean(0, keepdim=True).expand_as(ref)
    seeds = [query_seed("dev", i) for i in range(len(rows))]
    pipe, adapter, proj, prompt, pooled, dims = generator(torch.float16)
    pipe.vae = None
    vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).cuda().eval()
    vae.enable_slicing()
    for arm, features in (("training-mean", mean), ("original-photo", ref)):
        out = ROOT / "calibration" / "dev" / arm
        out.mkdir(parents=True, exist_ok=True)
        if not (out / "render.json").is_file():
            tokens = proj(adapter(features.cuda().half())).cpu()
            draw_images(out, pipe, vae, (prompt, pooled, dims), tokens, seeds)
            write_json(out / "render.json", dict(input=arm, eeg=False, panel="dev", row=rows, seeds=seeds, prompt="",
                       draws=1, unet_calls_per_draw=4, source_features_sha256=sha(file),
                       split_sha256=sha(PREP / "paper_development.json"),
                       pixels_sha256=[sha(out / f"{i:05d}.png") for i in range(len(rows))]))
    del pipe, vae, proj, adapter
    torch.cuda.empty_cache()
    for arm in ("training-mean", "original-photo"):
        out = ROOT / "calibration" / "dev" / arm
        if not (out / "scores.json").is_file():
            score(out, "dev", Path(os.environ["ARCHIVE"]), out)


if __name__ == "__main__":
    run()
