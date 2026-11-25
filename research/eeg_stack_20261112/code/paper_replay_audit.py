"""Replay an archived official R4 condition and compare all 200 original PNGs byte for byte."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import PREP, ROOT, RUNS, TURBO, write_json
from paper_render import condition, draw_images, groups, layout, query_seed, sha
from paper_training import D2, check_eeg, generator


@torch.inference_mode()
def replay(subject, arm):
    from diffusers import AutoencoderKL
    track = "shared" if RUNS.name == "shared" else "personal"
    if track == "personal" and RUNS.name != f"personal-{subject:02d}":
        raise ValueError("Participant does not match the fitting directory")
    if track == "shared" and arm == "native-paper":
        raise ValueError("Native EEG encoders are personal fits")
    folder = ROOT / f"images/{track}/{arm}/official-r4/sub-{subject:02d}"
    meta = json.loads((folder/"render.json").read_bytes())
    if meta["row"] != list(range(200)) or meta["arm"] != arm or meta["training"] != track or meta["subject"] != subject:
        raise ValueError("Unexpected original render manifest")
    check_eeg((subject,), "test")
    x = np.load(PREP / f"sub-{subject:02d}_test.npy", mmap_mode="r")
    group = groups("official-r4")
    seeds = [query_seed("official", i) for i in range(len(x))]
    if seeds != meta["seeds"] or group != meta["repeats"] or meta["eeg_donor"] != list(range(200)):
        raise ValueError("Original run uses different input or draw schedule")
    pipe, adapter, proj, text, pooled, dims = generator(torch.float16)
    pipe.vae = None
    vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).cuda().eval()
    vae.enable_slicing()
    ip, ckpt = condition(arm, subject, x, group, adapter, proj)
    if ckpt != meta["checkpoint_sha256"]:
        raise ValueError("Effective top-level checkpoint changed")
    z, ridge_sha, backbone_sha = None, None, None
    if arm == "d2":
        ridge = ROOT / f"layouts/sub-{subject:02d}/ridge.pt"
        z = layout(torch.load(ridge, map_location="cpu", weights_only=True), x, group)
        ridge_sha = sha(ridge)
        base = Path(torch.load(D2, map_location="cpu", weights_only=True)["backbone_ckpt"])
        if base.resolve().parent != D2.resolve().parents[1]:
            raise ValueError("D2 effective backbone is outside this fit")
        backbone_sha = sha(base)
    source = [sha(folder / f"{i:05d}.png") for i in range(len(x))]
    if source != meta["pixels_sha256"]:
        raise ValueError("Source pixels disagree with original receipt")
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        out = Path(tmp)
        draw_images(out, pipe, vae, (text, pooled, dims), ip, seeds, z)
        repeated = [sha(out / f"{i:05d}.png") for i in range(len(x))]
        exact = [x == y for x,y in zip(source, repeated, strict=True)]
        abs_error, largest = 0., 0
        for i, okay in enumerate(exact):
            if not okay:
                name = f"{i:05d}.png"
                with Image.open(folder/name) as original, Image.open(out/name) as duplicate:
                    delta = np.abs(np.asarray(original, dtype=np.int16) - np.asarray(duplicate, dtype=np.int16))
                    abs_error += float(delta.mean())
                    largest = max(largest, int(delta.max()))
    record = dict(subject=subject, arm=arm, training=track, budget="official-r4", photos=len(x),
                  original_render_sha256=sha(folder/"render.json"), checkpoint_sha256=ckpt, effective_backbone_sha256=backbone_sha,
                  layout_file_sha256=ridge_sha, eeg_sha256=sha(PREP / f"sub-{subject:02d}_test.npy"),
                  seed_schedule_sha256=hashlib.sha256(json.dumps(seeds).encode()).hexdigest(),
                  exact_pngs=sum(exact), replay_pixel_sha256=repeated, mean_abs_error_0_255=abs_error/len(x),
                  max_abs_error_0_255=largest, torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                  replay_code_sha256=sha(Path(__file__)))
    dest = ROOT / f"audit/replay/{track}/{arm}/sub-{subject:02d}.json"
    write_json(dest, record)
    print(dest.name, arm, sum(exact), "/", len(x), record["mean_abs_error_0_255"], largest, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, type=int, choices=range(1,11))
    parser.add_argument("--arms", nargs="+", choices=("native-paper", "d2"), required=True)
    args = parser.parse_args()
    for arm in args.arms:
        replay(args.subject, arm)
