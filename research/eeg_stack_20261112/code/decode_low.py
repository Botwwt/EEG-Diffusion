"""Decode z-scored latent predictions into start images: python decode_low.py <tag> [spread=0.8]
runs/low/sub-XX/<tag>_pred_z.npy -> runs/low/sub-XX/<tag>_n<spread>.npy, deviation from the mean latent rescaled so its
RMS over the 200 test images is <spread> x the RMS deviation of real training latents (label-free, as low.py)."""
import sys

import numpy as np
import torch

from common import PREP, RUNS, TURBO

tag = sys.argv[1]
spread = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
from diffusers import AutoencoderKL  # noqa: E402

dev = "cuda"
Y = torch.load(PREP / "emb_train.pt", weights_only=True)["vae"].float()
ym, ys = Y.mean(0), Y.std(0)
rms_real = float(((Y - ym) ** 2).sum(1).mean().sqrt())
vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval()
vae.enable_slicing()
for s in range(1, 11):
    pz = torch.from_numpy(np.load(RUNS / "low" / f"sub-{s:02d}" / f"{tag}_pred_z.npy")).float()
    d = pz * ys
    d = d * (spread * rms_real / float(d.pow(2).sum(1).mean().sqrt()))
    lat = ((ym + d).view(-1, 4, 16, 16) / vae.config.scaling_factor).to(dev)
    with torch.no_grad():
        im = torch.cat([(vae.decode(lat[k:k + 20]).sample / 2 + 0.5).clamp(0, 1).cpu() for k in range(0, len(lat), 20)])
    np.save(RUNS / "low" / f"sub-{s:02d}" / f"{tag}_n{spread}.npy", im.numpy().astype(np.float16))
print("DECODED", tag, flush=True)
