"""Native 512px (4x64x64) multi-scale oriented contour & contrast-energy target builder (spat_ms512.py).

Strictly uses ONLY training-set lat512_train.npy[lat512_train_ok.npy]:
  - Zero test images or test embeddings are opened or touched.
  - Zero evaluation-network weights are used (pure multi-scale finite differences on SDXL-VAE latents).
  - Extracts 8x8 & 16x16 coarse layout + 16x16 & 32x32 4-orientation (0, 45, 90, 135 deg) gradients +
    phase-invariant local edge energy, standardized per block and projected onto top-1024 orthonormal PCA axes.
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import PREP

dev = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_WEIGHTS = [0.90, 1.00, 0.90, 0.90, 0.70, 0.70, 0.65, 0.70, 0.70, 0.55, 0.55, 0.60]


def ms512_raw_blocks(lat64_centered):
    """lat64_centered: [N, 4, 64, 64] centered by training mean my."""
    v8 = F.avg_pool2d(lat64_centered, 8)
    v16 = F.avg_pool2d(lat64_centered, 4)
    v32 = F.avg_pool2d(lat64_centered, 2)

    def ori_and_energy(v):
        dx = v[:, :, :, 1:] - v[:, :, :, :-1]
        dy = v[:, :, 1:, :] - v[:, :, :-1, :]
        d1 = v[:, :, 1:, 1:] - v[:, :, :-1, :-1]
        d2 = v[:, :, 1:, :-1] - v[:, :, :-1, 1:]
        en = F.avg_pool2d(dx.abs()[:, :, :-1, :] + dy.abs()[:, :, :, :-1], 2)
        return [dx.flatten(1), dy.flatten(1), d1.flatten(1), d2.flatten(1), en.flatten(1)]

    return [v8.flatten(1), v16.flatten(1)] + ori_and_energy(v16) + ori_and_energy(v32)


def project_lat64_to_spat_ms512(lat64, pack):
    dev_ = lat64.device
    lat_c = lat64.float() - pack["lat_mean"].to(dev_)
    blks = ms512_raw_blocks(lat_c)
    parts = [w * (b - m.to(dev_)) / r for b, m, r, w in zip(blks, pack["mu"], pack["rms"], pack["weights"])]
    return torch.cat(parts, dim=1) @ pack["P_spat"].to(dev_)


if __name__ == "__main__":
    out_pt = PREP / "spat_ms512.pt"
    ok_np = np.load(PREP / "lat512_train_ok.npy")
    lat_np = np.load(PREP / "lat512_train.npy")
    N = len(lat_np)
    ok = torch.from_numpy(ok_np).to(dev)
    Y_all = torch.from_numpy(lat_np).float().to(dev)
    del lat_np

    lat_mean = Y_all[ok].mean(0, keepdim=True)  # [1, 4, 64, 64]
    Y_ok_c = Y_all[ok] - lat_mean

    blks_ok = ms512_raw_blocks(Y_ok_c)
    mu_list = [b.mean(0, keepdim=True) for b in blks_ok]
    rms_list = [max(float((b - m).pow(2).sum(1).mean().sqrt()), 1e-6) for b, m in zip(blks_ok, mu_list)]
    D_ok = torch.cat([w * (b - m) / r for b, m, r, w in zip(blks_ok, mu_list, rms_list, BLOCK_WEIGHTS)], dim=1)

    torch.manual_seed(20261116)
    _, S_vals, V_p = torch.pca_lowrank(D_ok, q=1056, center=False, niter=5)
    P_spat, _ = torch.linalg.qr(V_p[:, :1024].double(), mode="reduced")
    P_spat = P_spat.float().contiguous()
    expl = float(S_vals[:1024].pow(2).sum() / D_ok.pow(2).sum())
    del D_ok, blks_ok

    S_train = torch.zeros((N, 1024), dtype=torch.float32)
    pack = dict(
        lat_mean=lat_mean.cpu(),
        mu=[m.cpu() for m in mu_list],
        rms=rms_list,
        weights=BLOCK_WEIGHTS,
        P_spat=P_spat.cpu(),
        ok=torch.from_numpy(ok_np),
    )
    with torch.no_grad():
        for i in range(0, N, 1024):
            S_train[i:i + 1024] = project_lat64_to_spat_ms512(Y_all[i:i + 1024], pack).cpu()
    # Zero out non-ok training entries cleanly
    S_train[~pack["ok"]] = 0.0
    pack["S_train"] = S_train
    torch.save(pack, out_pt)
    print(f"Saved {out_pt} | N={N} (ok={int(ok_np.sum())}) -> 1024 | explained_var={expl:.4f}", flush=True)
