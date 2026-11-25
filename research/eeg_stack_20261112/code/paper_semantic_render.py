"""Score a development-only continuous text-conditioning experiment with an EEG-derived image condition."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL

from common import PREP, ROOT, TURBO, write_json
from paper_render import condition, draw_images, groups, query_seed, sha
from paper_score import score
from paper_soft_text import apply_soft, wrong_donors
from paper_training import generator


@torch.inference_mode()
def run(subject):
    lock = PREP / "paper_development.json"
    plan = json.loads(lock.read_bytes())
    if subject not in plan["subjects"]:
        raise ValueError("Use the registered development subjects")
    rows = plan["dev_rows"]
    x = np.load(PREP / f"sub-{subject:02d}_train.npy", mmap_mode="r")[rows]
    pipe, adapter, proj, _, _, dims = generator(torch.float16)
    pipe.vae = None
    vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).cuda().eval()
    vae.enable_slicing()
    repeat = groups("dev-r4")
    tokens, native_hash = condition("native-paper", subject, x, repeat, adapter, proj)
    seeds = [query_seed("dev", i) for i in range(len(x))]
    arms = ("soft-mean", "soft-eeg", "soft-wrong")
    own = None
    for arm in arms:
        out = ROOT / f"exploratory/{arm}/dev-r4/sub-{subject:02d}"
        out.mkdir(parents=True, exist_ok=True)
        if (out / "render.json").is_file():
            continue
        donors = wrong_donors(len(x)).tolist() if arm == "soft-wrong" else list(range(len(x)))
        if arm == "soft-mean":
            text, pooled, ckpt = apply_soft(arm, subject, x, repeat)
        else:
            if own is None:
                own = apply_soft("soft-eeg", subject, x, repeat)
            text, pooled, ckpt = own
            if arm == "soft-wrong":
                text, pooled = text[donors], pooled[donors]
        if text.shape != (len(x), 77, 2048) or pooled.shape != (len(x), 1280):
            raise ValueError("Incorrect continuous text condition dimensions")
        draw_images(out, pipe, vae, (text, pooled, dims), tokens, seeds)
        write_json(out / "render.json", dict(arm=arm, subject=subject, row=rows, seeds=seeds, repeats=repeat,
                   draws=1, unet_calls_per_draw=4, text_supervision="Training object names only",
                   inference="One participant's EEG, native EEG image tokens and continuous SDXL text states",
                   text_eeg_donor=donors if arm != "soft-mean" else [], ip_eeg_donor=list(range(len(x))),
                   native_checkpoint_sha256=native_hash, soft_checkpoint_sha256=ckpt, dev_sha256=sha(lock),
                   pixels_sha256=[sha(out / f"{i:05d}.png") for i in range(len(x))]))
    del pipe, vae, proj, adapter
    torch.cuda.empty_cache()
    for arm in arms:
        out = ROOT / f"exploratory/{arm}/dev-r4/sub-{subject:02d}"
        if not (out / "scores.json").is_file():
            score(out, "dev", Path(os.environ["ARCHIVE"]), out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", type=int, choices=(1, 3), required=True)
    run(p.parse_args().subject)
