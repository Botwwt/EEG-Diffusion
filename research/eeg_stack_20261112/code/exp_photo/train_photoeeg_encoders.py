"""Train PhotoEEG-Plus Strictly Single-Query Cross-Brain Encoders (ZERO Test-Batch TTA / ZERO Test Leakage!):
  1. --task plus16 : CrossBrainPlus16Net (EEG -> 16x2048 IP-Adapter-Plus spatial visual tokens + H14-L12 4x4 + H14-Final)
  2. --task layout : SharpLayoutNet (EEG -> Sharp 4x64x64 SDXL-VAE latent + 3x16x16 vivid RGB map)
  3. --task sqsem  : SQ-ObjSemEnc (Strictly single-query EEG -> LAION H14 text + SDXL bigG text)
All normalization & radial spread calibration statistics are fitted STRICTLY on training concepts only!
"""
import os
import re
import sys
import json
import time
import math
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RAD = Path(os.environ.get("RAD_ROOT", "/path/to/rad"))
PREP = RAD / "data/prep"
TGT_DIR = RAD / "runs/photo_eeg/targets"
OUT_DIR = RAD / "runs/photo_eeg/models"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def nz(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x.float(), dim=dim, eps=1e-6)


def prec_erp_analytic(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Zero-parameter M1a inverse-RMS precision ERP across repeats dim=1: x [B, R, C, T] -> [B, C, T]."""
    rms2 = x.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-6)  # [B, R, C, 1]
    w = rms2.pow(-p)
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (x.float() * w).sum(dim=1)


def load_all10_train_eeg():
    """Load all 10 subjects' training EEG [10, 16540, 4, 17, 250] as float16/memmap-backed tensors for fast indexing."""
    # Precompute M1a precision ERP [10, 16540, 17, 250] float16 (~1.4 GB total in RAM!) + raw memmaps for trial-set
    cache_erp = OUT_DIR / "all10_train_prec_erp_f16.npy"
    raw_maps = [np.load(PREP / f"sub-{s:02d}_train.npy", mmap_mode="r") for s in range(1, 11)]
    if cache_erp.exists():
        erp = np.load(cache_erp, mmap_mode="r")
        return erp, raw_maps
    print("Precomputing M1a precision ERP for all 10 subjects (train 16,540)...", flush=True)
    erp = np.empty((10, 16540, 17, 250), dtype=np.float16)
    for si in range(10):
        arr = raw_maps[si]
        for j in range(0, 16540, 512):
            b = torch.from_numpy(np.array(arr[j:j + 512], copy=True)).cuda().float()
            erp[si, j:j + 512] = prec_erp_analytic(b, p=0.5).half().cpu().numpy()
        print(f"  sub-{si+1:02d} prec_erp done", flush=True)
    np.save(cache_erp, erp)
    return np.load(cache_erp, mmap_mode="r"), raw_maps


def compute_train_eeg_stats(erp_all10):
    """Strictly train-only per-subject channel mean/std [10, 1, 17, 1]."""
    p = OUT_DIR / "train_eeg_stats.pt"
    if p.exists():
        return torch.load(p, map_location="cpu", weights_only=True)
    means, stds = [], []
    for si in range(10):
        x = torch.from_numpy(np.array(erp_all10[si], dtype=np.float32))  # [16540, 17, 250]
        means.append(x.mean(dim=(0, 2), keepdim=True))
        stds.append(x.std(dim=(0, 2), keepdim=True).clamp_min(1e-5))
    st = {"mean": torch.stack(means, 0), "std": torch.stack(stds, 0)}  # [10, 1, 17, 1]
    torch.save(st, p)
    return st


def get_train_concept_ids():
    train_raw = json.load(open(PREP / "train_ids.json"))
    names = []
    c2i = {}
    cids = []
    for p in train_raw["paths"]:
        folder = p.split("/")[1]
        name = re.sub(r"\d+$", "", folder.split("_", 1)[1] if "_" in folder else folder).replace("_", " ").strip()
        if name not in c2i:
            c2i[name] = len(names)
            names.append(name)
        cids.append(c2i[name])
    return np.array(cids, dtype=np.int64), names


# ==============================================================================
# 1. CrossBrainPlus16Net: Multi-Horizon EEG -> 16 Spatial IP-Adapter-Plus Tokens
# ==============================================================================
class ResConvBlock(nn.Module):
    def __init__(self, c: int, k: int = 5, dil: int = 1, drop: float = 0.15):
        super().__init__()
        pad = (k - 1) * dil // 2
        self.net = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv1d(c, c, k, padding=pad, dilation=dil),
            nn.Dropout(drop),
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv1d(c, c, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class CrossBrainPlus16Net(nn.Module):
    """Multi-horizon cross-subject EEG encoder with 16 learned spatial-visual cross-attention queries."""
    def __init__(self, n_subs: int = 10, d_model: int = 384, n_queries: int = 16, drop: float = 0.20):
        super().__init__()
        self.sub_proj = nn.ModuleList([nn.Conv1d(17, 128, 1) for _ in range(n_subs)])
        self.sub_emb = nn.Embedding(n_subs, d_model)
        self.stem = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3),  # 250 -> 125
            ResConvBlock(256, k=5, dil=1, drop=drop),
            nn.Conv1d(256, d_model, kernel_size=5, stride=2, padding=2),  # 125 -> 63
            ResConvBlock(d_model, k=5, dil=1, drop=drop),
            ResConvBlock(d_model, k=5, dil=2, drop=drop),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),  # 63 -> 32 temporal tokens
            ResConvBlock(d_model, k=3, dil=1, drop=drop),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, 32, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 3,
            dropout=drop, activation="gelu", batch_first=True, norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(enc_layer, num_layers=3)

        # 16 learned spatial visual query tokens attending to the 32 temporal-spatial EEG tokens
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 3,
            dropout=drop, activation="gelu", batch_first=True, norm_first=True,
        )
        self.q_dec = nn.TransformerDecoder(dec_layer, num_layers=2)
        self.q_norm = nn.LayerNorm(d_model)

        # Output heads
        self.head_plus16 = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.GELU(),
            nn.Dropout(drop * 0.5),
            nn.Linear(1024, 2048),
        )
        self.head_l12 = nn.Linear(d_model, 1280)  # 16 (4x4) spatial Layer-12 mid-level visual grid!
        self.head_h14 = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, 1024),
        )
        self.head_f = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, 1024),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(14.0)))
        self.in_drop = nn.Dropout(0.10)

    def forward(self, x: torch.Tensor, sub_ids: torch.Tensor):
        """x: [B, 17, 250] standardized EEG, sub_ids: [B] long in 0..9."""
        B = x.shape[0]
        x = self.in_drop(x)
        h_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x.dtype
        h = torch.empty((B, 128, 250), device=x.device, dtype=h_dtype)
        for s in sub_ids.unique():
            m = (sub_ids == s)
            h[m] = self.sub_proj[int(s.item())](x[m]).to(h_dtype)
        seq = self.stem(h).transpose(1, 2)  # [B, 32, d_model]
        seq = seq + self.pos_emb + self.sub_emb(sub_ids).unsqueeze(1)
        mem = self.trunk(seq)  # [B, 32, d_model]

        q = self.queries.expand(B, -1, -1)
        q_out = self.q_norm(self.q_dec(q, mem))  # [B, 16, d_model]

        plus16 = self.head_plus16(q_out)  # [B, 16, 2048]
        l12_16 = self.head_l12(q_out)     # [B, 16, 1280]

        g_feat = torch.cat([mem.mean(dim=1), q_out.mean(dim=1)], dim=-1)  # [B, 2*d_model]
        h14 = self.head_h14(g_feat)
        f_pred = self.head_f(g_feat)
        return plus16, l12_16, h14, f_pred


def sym_infonce(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    logits = (nz(a) @ nz(b).T) * scale.exp().clamp(max=40.0)
    lbl = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.T, lbl))


def train_plus16(seed: int, dev: str, epochs: int = 38):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    erp_all, raw_maps = load_all10_train_eeg()
    stats = compute_train_eeg_stats(erp_all)
    cids, _ = get_train_concept_ids()

    # Load targets (strictly train 16,540!)
    plus16_tr = np.load(TGT_DIR / "plus16_train.npy", mmap_mode="r")  # [16540, 16, 2048]
    l12_tr = np.load(TGT_DIR / "h14_l12_train.npy", mmap_mode="r")    # [16540, 4, 4, 1280]
    h14_tr = np.load(TGT_DIR / "h14_fin_train.npy", mmap_mode="r")    # [16540, 1024]
    F_tr = np.load(PREP / "F_train.npy", mmap_mode="r")               # [16540, 1024]

    # Hold out 10% of training concepts (disjoint concepts!) ONLY for train-only radial spread calibration & val tracking
    val_mask = (cids % 10 == (seed % 10))
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]

    # Precompute strictly train-only mean of plus16 [1, 16, 2048]
    mu_p16 = torch.from_numpy(np.array(plus16_tr[tr_idx], dtype=np.float32).mean(axis=0, keepdims=True)).to(dev)
    mu_h14 = torch.from_numpy(np.array(h14_tr[tr_idx], dtype=np.float32).mean(axis=0, keepdims=True)).to(dev)
    mu_f = torch.from_numpy(np.array(F_tr[tr_idx], dtype=np.float32).mean(axis=0, keepdims=True)).to(dev)

    # Load all10 erp into RAM for ultra-fast GPU training (~1.4GB)
    erp_ram = torch.from_numpy(np.array(erp_all, dtype=np.float16))  # [10, 16540, 17, 250]
    eeg_mean = stats["mean"].to(dev)  # [10, 1, 17, 1]
    eeg_std = stats["std"].to(dev)

    p16_ram = torch.from_numpy(np.array(plus16_tr, dtype=np.float16))
    l12_ram = torch.from_numpy(np.array(l12_tr, dtype=np.float16).reshape(16540, 16, 1280))
    h14_ram = torch.from_numpy(np.array(h14_tr, dtype=np.float16))
    f_ram = torch.from_numpy(np.array(F_tr, dtype=np.float16))

    model = CrossBrainPlus16Net(n_subs=10, d_model=384, n_queries=16, drop=0.22).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=2e-2)
    B = 192
    steps_per_ep = (len(tr_idx) * 2) // B
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_ep, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    best_va = -1.0
    best_state = None
    print(f"[Plus16 seed={seed}] tr={len(tr_idx)} va={len(va_idx)} epochs={epochs} steps/ep={steps_per_ep}", flush=True)

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for st in range(steps_per_ep):
            # Sample B distinct training images, and 2 distinct random subjects per image (cross-subject same-photo alignment!)
            img_b = np.random.choice(tr_idx, size=B, replace=False)
            s1 = np.random.randint(0, 10, size=B)
            s2 = (s1 + np.random.randint(1, 10, size=B)) % 10

            x1 = erp_ram[s1, img_b].to(dev, dtype=torch.float32)
            x2 = erp_ram[s2, img_b].to(dev, dtype=torch.float32)
            s1_t = torch.from_numpy(s1).to(dev)
            s2_t = torch.from_numpy(s2).to(dev)

            x1 = (x1 - eeg_mean[s1_t].squeeze(1)) / eeg_std[s1_t].squeeze(1)
            x2 = (x2 - eeg_mean[s2_t].squeeze(1)) / eeg_std[s2_t].squeeze(1)

            # Mild realistic EEG augmentation (tiny temporal shift +-2 samples + tiny Gaussian noise)
            if np.random.rand() < 0.7:
                sh1 = int(np.random.randint(-2, 3))
                sh2 = int(np.random.randint(-2, 3))
                if sh1 != 0:
                    x1 = torch.roll(x1, sh1, dims=-1)
                if sh2 != 0:
                    x2 = torch.roll(x2, sh2, dims=-1)
                x1 = x1 + 0.08 * torch.randn_like(x1)
                x2 = x2 + 0.08 * torch.randn_like(x2)

            y_p16 = p16_ram[img_b].to(dev, dtype=torch.float32)
            y_l12 = l12_ram[img_b].to(dev, dtype=torch.float32)
            y_h14 = h14_ram[img_b].to(dev, dtype=torch.float32)
            y_f = f_ram[img_b].to(dev, dtype=torch.float32)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                p1, l1, h1, f1 = model(x1, s1_t)
                p2, l2, h2, f2 = model(x2, s2_t)

                # Residual relative to train mean for crisp photo-specific InfoNCE!
                dp1 = (p1.float() - mu_p16).flatten(1)
                dp2 = (p2.float() - mu_p16).flatten(1)
                dgt = (y_p16 - mu_p16).flatten(1)

                # Token-level cosine + Smooth-L1 + Flattened Photo InfoNCE + Cross-Brain Consistency
                l_tok_cos = 0.5 * ((1.0 - (nz(p1) * nz(y_p16)).sum(-1).mean()) +
                                   (1.0 - (nz(p2) * nz(y_p16)).sum(-1).mean()))
                l_tok_l1 = 0.5 * (F.smooth_l1_loss(p1.float(), y_p16, beta=0.5) +
                                  F.smooth_l1_loss(p2.float(), y_p16, beta=0.5))
                l_nce_p16 = 0.5 * (sym_infonce(dp1, dgt, model.logit_scale) +
                                   sym_infonce(dp2, dgt, model.logit_scale))
                l_xsub = 0.25 * sym_infonce(dp1, dp2, model.logit_scale)

                # Mid-level Layer-12 4x4 spatial grid loss (2D geometry + parts!)
                l_l12 = 0.5 * ((1.0 - (nz(l1) * nz(y_l12)).sum(-1).mean()) +
                               (1.0 - (nz(l2) * nz(y_l12)).sum(-1).mean())) + \
                        0.25 * (sym_infonce(l1.flatten(1), y_l12.flatten(1), model.logit_scale) +
                                sym_infonce(l2.flatten(1), y_l12.flatten(1), model.logit_scale))

                # Global H14 + Fusion-F alignment
                l_h14 = 0.35 * (sym_infonce(h1.float() - mu_h14, y_h14 - mu_h14, model.logit_scale) +
                                sym_infonce(h2.float() - mu_h14, y_h14 - mu_h14, model.logit_scale))
                l_f = 0.35 * (sym_infonce(f1.float() - mu_f, y_f - mu_f, model.logit_scale) +
                              sym_infonce(f2.float() - mu_f, y_f - mu_f, model.logit_scale))

                loss = 1.5 * l_tok_cos + 0.8 * l_tok_l1 + 1.0 * l_nce_p16 + l_xsub + 0.6 * l_l12 + l_h14 + l_f

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += float(loss.item())

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Evaluate on unseen training-fold concepts (sub-08 + sub-10)
                va_sample = va_idx[:400]
                gt_p = (p16_ram[va_sample].to(dev, dtype=torch.float32) - mu_p16).flatten(1)
                gt_h = (h14_ram[va_sample].to(dev, dtype=torch.float32) - mu_h14)
                accs_p, accs_h, ratios = [], [], []
                for s_eval in (0, 7, 9):
                    xv = erp_ram[s_eval, va_sample].to(dev, dtype=torch.float32)
                    xv = (xv - eeg_mean[s_eval]) / eeg_std[s_eval]
                    sid = torch.full((len(va_sample),), s_eval, device=dev, dtype=torch.long)
                    pv, _, hv, _ = model(xv, sid)
                    dpv = (pv.float() - mu_p16).flatten(1)
                    sp = nz(dpv) @ nz(gt_p).T
                    sh = nz(hv.float() - mu_h14) @ nz(gt_h).T
                    idx_v = torch.arange(len(va_sample), device=dev)
                    accs_p.append((sp.argmax(-1) == idx_v).float().mean().item() * 100.0)
                    accs_h.append((sh.argmax(-1) == idx_v).float().mean().item() * 100.0)
                    # Per-token spread ratio on OOF val concepts
                    r = ((p16_ram[va_sample].to(dev, dtype=torch.float32) - mu_p16).norm(dim=-1).mean(0) /
                         (pv.float() - mu_p16).norm(dim=-1).mean(0).clamp_min(1e-4))
                    ratios.append(r.cpu())
                sc = float(np.mean(accs_p))
                if sc > best_va:
                    best_va = sc
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    best_spread = torch.stack(ratios).mean(0).clamp(1.0, 2.2)
                print(f"  [Plus16 s={seed} ep={ep:02d}] loss={tot_loss/steps_per_ep:.3f} | "
                      f"va400 Plus16-T1={sc:.2f}% H14-T1={np.mean(accs_h):.2f}% | "
                      f"spread={best_spread.mean().item():.2f} ({time.time()-t0:.0f}s)", flush=True)

    # Save checkpoint + strictly single-query test predictions (R80 and R4, ZERO test-batch stats!)
    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    model.eval()
    ckpt = {
        "seed": seed,
        "state_dict": best_state,
        "mu_p16": mu_p16.cpu(),
        "mu_h14": mu_h14.cpu(),
        "mu_f": mu_f.cpu(),
        "spread_p16": best_spread.cpu(),
        "best_va_t1": best_va,
    }
    torch.save(ckpt, OUT_DIR / f"plus16_seed{seed}.pt")
    print(f"[SAVED] plus16_seed{seed}.pt (best_va400={best_va:.2f}%)", flush=True)


# ==============================================================================
# 2. SharpLayoutNet: Cross-Brain Nonlinear Sharp Spatial VAE Latent Decoder
# ==============================================================================
class SharpLayoutNet(nn.Module):
    """Cross-subject nonlinear spatial EEG -> 4x64x64 SDXL-VAE latent + 3x16x16 vivid RGB decoder."""
    def __init__(self, n_subs: int = 10, drop: float = 0.20):
        super().__init__()
        self.sub_proj = nn.ModuleList([nn.Conv1d(17, 96, 1) for _ in range(n_subs)])
        self.sub_emb = nn.Embedding(n_subs, 512)
        self.enc = nn.Sequential(
            nn.Conv1d(96, 192, kernel_size=7, stride=2, padding=3),  # 125
            ResConvBlock(192, k=5, drop=drop),
            nn.Conv1d(192, 320, kernel_size=5, stride=2, padding=2),  # 63
            ResConvBlock(320, k=5, drop=drop),
            nn.Conv1d(320, 512, kernel_size=5, stride=2, padding=2),  # 32
            ResConvBlock(512, k=3, drop=drop),
        )
        # Multi-window pooling (early retinotopic 0:10 + mid 6:18 + late 14:32 + global)
        self.fc = nn.Sequential(
            nn.LayerNorm(512 * 4),
            nn.Linear(512 * 4, 1024),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(1024, 256 * 8 * 8),
        )
        # Spatial conv upsampling decoder: 8x8 -> 16x16 -> 32x32 -> 64x64 (sharp boundaries!)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
        )
        self.rgb16_head = nn.Conv2d(128, 3, 1)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 4, 3, padding=1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(12.0)))

    def forward(self, x: torch.Tensor, sub_ids: torch.Tensor):
        B = x.shape[0]
        h_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x.dtype
        h = torch.empty((B, 96, 250), device=x.device, dtype=h_dtype)
        for s in sub_ids.unique():
            m = (sub_ids == s)
            h[m] = self.sub_proj[int(s.item())](x[m]).to(h_dtype)
        t = self.enc(h)  # [B, 512, 32]
        w1 = t[:, :, 2:10].mean(-1)
        w2 = t[:, :, 6:16].mean(-1)
        w3 = t[:, :, 12:26].mean(-1)
        w4 = t.mean(-1) + self.sub_emb(sub_ids)
        z8 = self.fc(torch.cat([w1, w2, w3, w4], dim=-1)).view(B, 256, 8, 8)
        z16 = self.up1(z8)
        rgb16 = torch.sigmoid(self.rgb16_head(z16))  # [B, 3, 16, 16] in [0, 1]
        z32 = self.up2(z16)
        lat64 = self.up3(z32)  # [B, 4, 64, 64]
        return lat64, rgb16


def spatial_grad_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)


def train_layout(seed: int, dev: str, epochs: int = 30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    erp_all, _ = load_all10_train_eeg()
    stats = compute_train_eeg_stats(erp_all)
    cids, _ = get_train_concept_ids()

    lat_tr = np.load(PREP / "lat512_train.npy", mmap_mode="r")  # [16540, 4, 64, 64]
    rgb_tr = np.load(TGT_DIR / "rgb16_train.npy", mmap_mode="r")  # [16540, 3, 16, 16]

    val_mask = (cids % 10 == (seed % 10))
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]

    lat_ram = torch.from_numpy(np.array(lat_tr, dtype=np.float16))
    rgb_ram = torch.from_numpy(np.array(rgb_tr, dtype=np.float16))
    erp_ram = torch.from_numpy(np.array(erp_all, dtype=np.float16))
    eeg_mean = stats["mean"].to(dev)
    eeg_std = stats["std"].to(dev)

    mu_lat = lat_ram[tr_idx].float().mean(0, keepdim=True).to(dev)  # [1, 4, 64, 64]
    mu_rgb = rgb_ram[tr_idx].float().mean(0, keepdim=True).to(dev)

    model = SharpLayoutNet(n_subs=10, drop=0.20).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-2)
    B = 192
    steps_per_ep = (len(tr_idx) * 2) // B
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_ep, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    best_va = -1e9
    best_state = None
    best_gamma = 1.0
    print(f"[SharpLayout seed={seed}] tr={len(tr_idx)} va={len(va_idx)} epochs={epochs}", flush=True)

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for _ in range(steps_per_ep):
            img_b = np.random.choice(tr_idx, size=B, replace=False)
            s1 = np.random.randint(0, 10, size=B)
            x1 = erp_ram[s1, img_b].to(dev, dtype=torch.float32)
            s1_t = torch.from_numpy(s1).to(dev)
            x1 = (x1 - eeg_mean[s1_t].squeeze(1)) / eeg_std[s1_t].squeeze(1)
            x1 = x1 + 0.06 * torch.randn_like(x1)

            y_lat = lat_ram[img_b].to(dev, dtype=torch.float32)
            y_rgb = rgb_ram[img_b].to(dev, dtype=torch.float32)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                p_lat, p_rgb = model(x1, s1_t)
                p_lat_f = p_lat.float() + mu_lat  # residual prediction around train mean!
                p_rgb_f = p_rgb.float()

                # Multi-scale latent fidelity + spatial contrastive (prevents grey muddy collapse!) + sharp edge loss + vivid color!
                l_lat = F.smooth_l1_loss(p_lat_f, y_lat, beta=0.5)
                l_low16 = F.smooth_l1_loss(F.avg_pool2d(p_lat_f, 4), F.avg_pool2d(y_lat, 4), beta=0.3)
                l_edge = spatial_grad_loss(F.avg_pool2d(p_lat_f, 2), F.avg_pool2d(y_lat, 2))
                l_nce16 = sym_infonce(
                    F.avg_pool2d(p_lat_f - mu_lat, 4).flatten(1),
                    F.avg_pool2d(y_lat - mu_lat, 4).flatten(1),
                    model.logit_scale,
                )
                l_rgb = F.l1_loss(p_rgb_f, y_rgb) + 0.5 * spatial_grad_loss(p_rgb_f, y_rgb) + \
                        0.3 * sym_infonce((p_rgb_f - mu_rgb).flatten(1), (y_rgb - mu_rgb).flatten(1), model.logit_scale)

                loss = 1.0 * l_lat + 1.5 * l_low16 + 0.8 * l_edge + 0.25 * l_nce16 + 1.5 * l_rgb

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += float(loss.item())

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.inference_mode():
                va_sample = va_idx[:400]
                y_v = lat_ram[va_sample].to(dev, dtype=torch.float32)
                r_v = rgb_ram[va_sample].to(dev, dtype=torch.float32)
                corrs, rcorrs, gammas = [], [], []
                for s_eval in (0, 7, 9):
                    xv = (erp_ram[s_eval, va_sample].to(dev, dtype=torch.float32) - eeg_mean[s_eval]) / eeg_std[s_eval]
                    sid = torch.full((len(va_sample),), s_eval, device=dev, dtype=torch.long)
                    pl, pr = model(xv, sid)
                    pl_low = F.avg_pool2d(pl.float(), 4).flatten(1)
                    yt_low = F.avg_pool2d(y_v - mu_lat, 4).flatten(1)
                    corrs.append((nz(pl_low) * nz(yt_low)).sum(-1).mean().item())
                    rcorrs.append((nz((pr.float() - mu_rgb).flatten(1)) * nz((r_v - mu_rgb).flatten(1))).sum(-1).mean().item())
                    gammas.append(float(0.85 * yt_low.square().sum(1).mean().sqrt() / pl_low.square().sum(1).mean().sqrt().clamp_min(1e-5)))
                sc = float(np.mean(corrs) + np.mean(rcorrs))
                if sc > best_va:
                    best_va = sc
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    best_gamma = float(np.clip(np.mean(gammas), 0.8, 2.0))
                print(f"  [SharpLayout s={seed} ep={ep:02d}] loss={tot_loss/steps_per_ep:.3f} | "
                      f"lat16_cos={np.mean(corrs):.3f} rgb16_cos={np.mean(rcorrs):.3f} gamma={best_gamma:.2f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({
        "seed": seed,
        "state_dict": best_state,
        "mu_lat": mu_lat.cpu(),
        "mu_rgb": mu_rgb.cpu(),
        "gamma": best_gamma,
        "best_va": best_va,
    }, OUT_DIR / f"layout_seed{seed}.pt")
    print(f"[SAVED] layout_seed{seed}.pt (gamma={best_gamma:.2f})", flush=True)


# ==============================================================================
# 3. SQ-ObjSemEnc: Strictly Single-Query Semantic Text Encoder (ZERO Test-Batch TTA!)
# ==============================================================================
class SQSemNet(nn.Module):
    def __init__(self, n_subs: int = 10, drop: float = 0.25):
        super().__init__()
        self.sub_proj = nn.ModuleList([nn.Conv1d(17, 128, 1) for _ in range(n_subs)])
        self.sub_emb = nn.Embedding(n_subs, 384)
        self.full_net = nn.Sequential(
            nn.Conv1d(128, 256, 7, stride=2, padding=3),
            ResConvBlock(256, k=5, drop=drop),
            nn.Conv1d(256, 384, 5, stride=2, padding=2),
            ResConvBlock(384, k=5, dil=2, drop=drop),
            nn.AdaptiveAvgPool1d(8),
        )
        # Late semantic ERP stream (160-880 ms -> samples 40:220)
        self.late_net = nn.Sequential(
            nn.Conv1d(128, 256, 7, stride=2, padding=3),
            ResConvBlock(256, k=5, drop=drop),
            nn.Conv1d(256, 384, 5, stride=2, padding=2),
            ResConvBlock(384, k=5, drop=drop),
            nn.AdaptiveAvgPool1d(6),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(384 * 14 + 384),
            nn.Linear(384 * 14 + 384, 1024),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(1024),
        )
        self.h14_head = nn.Linear(1024, 1024)
        self.bigg_head = nn.Linear(1024, 1280)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(16.0)))

    def forward(self, x: torch.Tensor, sub_ids: torch.Tensor):
        B = x.shape[0]
        h_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x.dtype
        h = torch.empty((B, 128, 250), device=x.device, dtype=h_dtype)
        for s in sub_ids.unique():
            m = (sub_ids == s)
            h[m] = self.sub_proj[int(s.item())](x[m]).to(h_dtype)
        f_full = self.full_net(h).flatten(1)
        f_late = self.late_net(h[:, :, 40:220]).flatten(1)
        z = self.fuse(torch.cat([f_full, f_late, self.sub_emb(sub_ids)], dim=-1))
        return self.h14_head(z), self.bigg_head(z)


def train_sqsem(seed: int, dev: str, epochs: int = 30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    tb = torch.load(RAD / "runs/obj_bench/text_banks/text_banks_laion.pt", map_location="cpu", weights_only=False)
    erp_all, _ = load_all10_train_eeg()
    stats = compute_train_eeg_stats(erp_all)

    tr_cid = tb["tr_img2cid"].long().numpy()  # [16540]
    h14_txt = tb["h14_train"].float()         # [n_c, 1024]
    bigg_txt = tb["bigg_train"].float()       # [n_c, 1280]
    mu_h = h14_txt[tr_cid].mean(0, keepdim=True).to(dev)
    mu_b = bigg_txt[tr_cid].mean(0, keepdim=True).to(dev)
    h14_c = nz(h14_txt.to(dev) - mu_h, dim=-1)
    bigg_c = nz(bigg_txt.to(dev) - mu_b, dim=-1)

    val_mask = (tr_cid % 10 == (seed % 10))
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]
    va_cids = np.unique(tr_cid[va_idx])

    erp_ram = torch.from_numpy(np.array(erp_all, dtype=np.float16))
    eeg_mean = stats["mean"].to(dev)
    eeg_std = stats["std"].to(dev)

    model = SQSemNet(n_subs=10, drop=0.25).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=2e-2)
    B = 256
    steps_per_ep = (len(tr_idx) * 2) // B
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_ep, eta_min=1e-5)

    best_va = -1.0
    best_state = None
    print(f"[SQSem seed={seed}] tr={len(tr_idx)} va_concepts={len(va_cids)} epochs={epochs}", flush=True)

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for _ in range(steps_per_ep):
            img_b = np.random.choice(tr_idx, size=B, replace=False)
            s1 = np.random.randint(0, 10, size=B)
            x1 = erp_ram[s1, img_b].to(dev, dtype=torch.float32)
            s1_t = torch.from_numpy(s1).to(dev)
            x1 = (x1 - eeg_mean[s1_t].squeeze(1)) / eeg_std[s1_t].squeeze(1)
            x1 = x1 + 0.08 * torch.randn_like(x1)

            cid_b = torch.from_numpy(tr_cid[img_b]).to(dev)
            yh = h14_c[cid_b]
            yb = bigg_c[cid_b]

            opt.zero_grad(set_to_none=True)
            ph, pb = model(x1, s1_t)
            # Multi-positive concept-level SupCon against ALL training concept text bank!
            sc = model.logit_scale.exp().clamp(max=40.0)
            l_h = F.cross_entropy((nz(ph) @ h14_c.T) * sc, cid_b)
            l_b = F.cross_entropy((nz(pb) @ bigg_c.T) * sc, cid_b)
            loss = l_h + l_b
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot_loss += float(loss.item())

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.inference_mode():
                # Strictly single-query evaluation on unseen validation concepts!
                va_h_bank = h14_c[va_cids]
                va_b_bank = bigg_c[va_cids]
                cid_to_local = {int(c): i for i, c in enumerate(va_cids)}
                va_sub_idx = va_idx[::5]
                va_lbl = torch.tensor([cid_to_local[int(tr_cid[i])] for i in va_sub_idx], device=dev)
                accs = []
                for s_eval in (0, 7, 9):
                    xv = (erp_ram[s_eval, va_sub_idx].to(dev, dtype=torch.float32) - eeg_mean[s_eval]) / eeg_std[s_eval]
                    sid = torch.full((len(va_sub_idx),), s_eval, device=dev, dtype=torch.long)
                    ph, pb = model(xv, sid)
                    sim = 0.5 * (nz(ph) @ va_h_bank.T) + 0.5 * (nz(pb) @ va_b_bank.T)
                    accs.append((sim.argmax(-1) == va_lbl).float().mean().item() * 100.0)
                sc_v = float(np.mean(accs))
                if sc_v > best_va:
                    best_va = sc_v
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                print(f"  [SQSem s={seed} ep={ep:02d}] loss={tot_loss/steps_per_ep:.3f} | "
                      f"va_unseen({len(va_cids)}w)={sc_v:.2f}% ({time.time()-t0:.0f}s)", flush=True)

    torch.save({
        "seed": seed,
        "state_dict": best_state,
        "mu_h": mu_h.cpu(),
        "mu_b": mu_b.cpu(),
        "best_va": best_va,
    }, OUT_DIR / f"sqsem_seed{seed}.pt")
    print(f"[SAVED] sqsem_seed{seed}.pt (best_va={best_va:.2f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True, choices=["plus16", "layout", "sqsem"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--dev", type=str, default="cuda:0")
    args = ap.parse_args()
    if args.task == "plus16":
        train_plus16(args.seed, args.dev, args.epochs)
    elif args.task == "layout":
        train_layout(args.seed, args.dev, args.epochs)
    elif args.task == "sqsem":
        train_sqsem(args.seed, args.dev, args.epochs)


if __name__ == "__main__":
    main()
