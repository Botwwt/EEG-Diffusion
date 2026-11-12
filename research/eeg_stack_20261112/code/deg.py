"""Degraded-image targets (image-side prior of NeuroBridge, arXiv 2511.06836 / UBP, arXiv 2503.04207), for the fused target.

python deg.py  ->  data/prep/emb_{train,test}_{deg4,blur}.pt (the three encoder outputs) and F_{train,test}_{deg4,blur}.npy

Each image is resized to 224 px and degraded four ways with the NeuroBridge settings (Gaussian blur kernel 15, Gaussian
noise std 25/255, low resolution x0.1 bilinear, mosaic x1/5 nearest); every encoder output is averaged over the four
(deg4) or taken from the blurred image alone (blur). The clean image is not part of the average, as in NeuroBridge.
Unreadable training images keep their published clean embeddings.
"""
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from common import EMB_KEYS, IMG, PREP
from fenc import FEmbedder

dev = "cuda"
fe = FEmbedder(dev)
tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(224), T.ToTensor()])
blur = T.GaussianBlur(15, sigma=0.3 * ((15 - 1) * 0.5 - 1) + 0.8)   # OpenCV's automatic sigma for kernel 15
up = torch.nn.functional.interpolate


def degrade(x, gen):
    return {
        "blur": blur(x),
        "noise": (x + torch.randn(x.shape, generator=gen, device=x.device) * 25 / 255).clamp(0, 1),
        "lowres": up(up(x, scale_factor=0.1, mode="bilinear", antialias=True), size=224, mode="bilinear"),
        "mosaic": up(up(x, scale_factor=0.2, mode="nearest"), size=224, mode="nearest"),
    }


for split in ("train", "test"):
    ids = json.loads((PREP / f"{split}_ids.json").read_text())["ids"]
    files = {f.stem: f for f in (IMG / ("training_images" if split == "train" else "test_images")).glob("*/*.jpg")}
    ref = torch.load(PREP / f"emb_{split}.pt", weights_only=True)
    res = {v: {k: ref[k].clone().float() for k in EMB_KEYS} for v in ("deg4", "blur")}
    gen = torch.Generator(device=dev).manual_seed(0)
    for s in range(0, len(ids), 100):
        idx, ims = [], []
        for j in range(s, min(len(ids), s + 100)):
            try:
                ims.append(tf(Image.open(files[ids[j]]).convert("RGB")))
                idx.append(j)
            except OSError:
                pass
        parts = {n: fe.parts(v) for n, v in degrade(torch.stack(ims).to(dev), gen).items()}
        for k in EMB_KEYS:
            res["deg4"][k][idx] = torch.stack([parts[n][k] for n in parts]).mean(0).cpu()
            res["blur"][k][idx] = parts["blur"][k].cpu()
        if s % 2000 == 0:
            print(split, s, flush=True)
    for v, e in res.items():
        torch.save(e, PREP / f"emb_{split}_{v}.pt")
        with torch.no_grad():
            F = torch.cat([fe.fuse(**{k: e[k][i:i + 2000].to(dev) for k in EMB_KEYS}).float().cpu() for i in range(0, len(ids), 2000)])
        np.save(PREP / f"F_{split}_{v}.npy", F.numpy())
        Fc = torch.from_numpy(np.load(PREP / f"F_{split}.npy"))
        nz = torch.nn.functional.normalize
        print(f"{split} {v}: cos(F_degraded, F_clean) {float((nz(F, dim=1) * nz(Fc, dim=1)).sum(1).mean()):.3f}, norm {float(F.norm(dim=1).mean()):.2f}", flush=True)
print("DEG_DONE")
