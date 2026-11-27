"""StrongDualH14PlusNet + Fixed-CFG SDXL-Base Single & Dual IP-Adapter Visual Fidelity Diagnostics.
Fixes the 10.6x CFG uncond=mu_p bug (uses true uncond=ZERO!) and upgrades EEG->H14/Plus16
with a phase-locked Multi-Horizon MLP + ResConv1D hybrid backbone (100% strictly single-query!).
"""
import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler

RAD = Path(os.environ.get("RAD_ROOT", "/path/to/rad"))
sys.path.insert(0, str(RAD / "code/exp_photo"))

from train_photoeeg_encoders import (
    ResConvBlock, SharpLayoutNet, SQSemNet,
    load_all10_train_eeg, compute_train_eeg_stats, get_train_concept_ids,
    prec_erp_analytic, nz, sym_infonce,
)
from run_photoeeg_pipeline import (
    IdentityImageProj, SpatialPhotoVerifier, query_seed, save_gallery, build_prompt_list,
)
from eval_photo_suite import PhotoFidelityEvaluator

PREP = RAD / "data/prep"
TURBO = RAD / "models/sdxl-turbo"
SDXL_BASE = RAD / "models/sdxl-base-unet"
IPA_DIR = RAD / "models/IP-Adapter"
TGT_DIR = RAD / "runs/photo_eeg/targets"
MOD_DIR = RAD / "runs/photo_eeg/models"
OUT_ROOT = RAD / "runs/photo_eeg/diag_v2"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. StrongDualH14PlusNet: Phase-Locked Multi-Horizon MLP + Conv1D Cross-Brain Backbone
# ==============================================================================
class StrongDualH14PlusNet(nn.Module):
    """Preserves exact phase-locked ERP timing (Multi-Horizon MLP) + temporal-conv dynamics.
    Jointly predicts high-SNR CLIP-H/14 (1024-D for ip-adapter_sdxl_vit-h),
    F_1024, H14-L12 4x4 spatial grid, and calibrated Plus16 (16x2048).
    """
    WINDOWS = ((10, 75), (25, 115), (45, 175), (0, 250))

    def __init__(self, n_subs: int = 10, d_trunk: int = 1024, drop: float = 0.25):
        super().__init__()
        self.in_drop = nn.Dropout(0.15)
        # Per-subject spatial channel mixing (preserves exact timepoints!)
        self.sub_mix = nn.ModuleList([nn.Conv1d(17, 24, 1) for _ in range(n_subs)])
        self.sub_emb = nn.Embedding(n_subs, d_trunk)

        # Branch A: Phase-locked Multi-Horizon MLP (downsampled 2x to preserve fine latency with compact width)
        self.win_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(24 * ((stop - start + 1) // 2), 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(drop),
            )
            for start, stop in self.WINDOWS
        ])

        # Branch B: Multi-scale dilated temporal ResConv
        self.conv_branch = nn.Sequential(
            nn.Conv1d(24, 128, kernel_size=7, stride=2, padding=3),
            ResConvBlock(128, k=5, dil=1, drop=drop * 0.7),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            ResConvBlock(256, k=5, dil=2, drop=drop * 0.7),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(1),
            nn.Linear(256 * 8, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(drop),
        )

        # Shared cross-brain residual trunk
        in_dim = 512 * (len(self.WINDOWS) + 1)
        self.fuse_in = nn.Sequential(
            nn.Linear(in_dim, d_trunk),
            nn.LayerNorm(d_trunk),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_trunk),
                nn.Linear(d_trunk, d_trunk * 2),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(d_trunk * 2, d_trunk),
            )
            for _ in range(3)
        ])
        self.trunk_norm = nn.LayerNorm(d_trunk)

        # Dedicated high-SNR visual heads
        self.head_h14 = nn.Sequential(nn.Linear(d_trunk, 1024), nn.GELU(), nn.Linear(1024, 1024))
        self.head_f = nn.Sequential(nn.Linear(d_trunk, 1024), nn.GELU(), nn.Linear(1024, 1024))
        # Factorized Plus16 head (bottleneck 16x256 -> 16x2048 prevents high-dim noise!)
        self.head_p16_low = nn.Sequential(nn.Linear(d_trunk, 16 * 192), nn.GELU())
        self.head_p16_out = nn.Linear(192, 2048)
        self.head_l12_low = nn.Sequential(nn.Linear(d_trunk, 16 * 128), nn.GELU())
        self.head_l12_out = nn.Linear(128, 1280)

        self.scale_h14 = nn.Parameter(torch.tensor(math.log(16.0)))
        self.scale_p16 = nn.Parameter(torch.tensor(math.log(14.0)))
        self.scale_l12 = nn.Parameter(torch.tensor(math.log(14.0)))

    def forward(self, x: torch.Tensor, sub_ids: torch.Tensor):
        B = x.shape[0]
        x = self.in_drop(x)
        h_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x.dtype
        xm = torch.empty((B, 24, 250), device=x.device, dtype=h_dtype)
        for s in sub_ids.unique():
            m = (sub_ids == s)
            xm[m] = self.sub_mix[int(s.item())](x[m]).to(h_dtype)

        feats = []
        for (start, stop), proj in zip(self.WINDOWS, self.win_projs):
            seg = xm[:, :, start:stop]
            seg2 = F.avg_pool1d(seg, kernel_size=2, stride=2, ceil_mode=True)
            feats.append(proj(seg2.flatten(1)))
        feats.append(self.conv_branch(xm))

        z = self.fuse_in(torch.cat(feats, dim=-1)) + self.sub_emb(sub_ids)
        for blk in self.res_blocks:
            z = z + blk(z)
        z = self.trunk_norm(z)

        h14 = self.head_h14(z)
        f_pred = self.head_f(z)
        p16 = self.head_p16_out(self.head_p16_low(z).view(B, 16, 192))
        l12 = self.head_l12_out(self.head_l12_low(z).view(B, 16, 128))
        return h14, f_pred, p16, l12


def train_strong_h14(seed: int, dev: str = "cuda:0", epochs: int = 35):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    erp_all, _ = load_all10_train_eeg()
    stats = compute_train_eeg_stats(erp_all)
    cids, _ = get_train_concept_ids()

    h14_tr = np.load(TGT_DIR / "h14_fin_train.npy", mmap_mode="r")
    plus16_tr = np.load(TGT_DIR / "plus16_train.npy", mmap_mode="r")
    l12_tr = np.load(TGT_DIR / "h14_l12_train.npy", mmap_mode="r")
    F_tr = np.load(PREP / "F_train.npy", mmap_mode="r")

    # Note: H14 in CLIP-ViT-H-14 unnormalized norm (exact input norm expected by ip-adapter_sdxl_vit-h!)
    # Check raw emb_train.pt for true unnormalized H14 scale (typically ~18.5 for CLIP-ViT-H-14)
    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=False)
    h14_raw_key = [k for k in emb_tr.keys() if "H" in k or "h14" in k.lower()][0]
    h14_raw_norm = float(emb_tr[h14_raw_key].float().norm(dim=-1).mean().item())
    del emb_tr

    val_mask = (cids % 10 == (seed % 10))
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]

    erp_ram = torch.from_numpy(np.array(erp_all, dtype=np.float16))
    h14_ram = torch.from_numpy(np.array(h14_tr, dtype=np.float16))
    p16_ram = torch.from_numpy(np.array(plus16_tr, dtype=np.float16))
    l12_ram = torch.from_numpy(np.array(l12_tr, dtype=np.float16).reshape(16540, 16, 1280))
    f_ram = torch.from_numpy(np.array(F_tr, dtype=np.float16))

    eeg_mean = stats["mean"].to(dev)
    eeg_std = stats["std"].to(dev)
    mu_h14 = h14_ram[tr_idx].float().mean(0, keepdim=True).to(dev)
    mu_p16 = p16_ram[tr_idx].float().mean(0, keepdim=True).to(dev)
    mu_f = f_ram[tr_idx].float().mean(0, keepdim=True).to(dev)

    model = StrongDualH14PlusNet(n_subs=10, d_trunk=1024, drop=0.25).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=3e-2)
    B = 256
    steps_per_ep = (len(tr_idx) * 2) // B
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_ep, eta_min=1e-5)

    best_va = -1.0
    best_state = None
    best_h_spread = 1.0
    print(f"[StrongH14 s={seed}] tr={len(tr_idx)} h14_raw_norm={h14_raw_norm:.2f} epochs={epochs}", flush=True)

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for _ in range(steps_per_ep):
            img_b = np.random.choice(tr_idx, size=B, replace=False)
            s1 = np.random.randint(0, 10, size=B)
            s2 = (s1 + np.random.randint(1, 10, size=B)) % 10

            s1_t = torch.from_numpy(s1).to(dev)
            s2_t = torch.from_numpy(s2).to(dev)
            x1 = (erp_ram[s1, img_b].to(dev, dtype=torch.float32) - eeg_mean[s1_t].squeeze(1)) / eeg_std[s1_t].squeeze(1)
            x2 = (erp_ram[s2, img_b].to(dev, dtype=torch.float32) - eeg_mean[s2_t].squeeze(1)) / eeg_std[s2_t].squeeze(1)

            if np.random.rand() < 0.7:
                x1 = x1 + 0.10 * torch.randn_like(x1)
                x2 = x2 + 0.10 * torch.randn_like(x2)
                # Random channel/time mild mask
                if np.random.rand() < 0.5:
                    sh = int(np.random.randint(-2, 3))
                    if sh != 0:
                        x1 = torch.roll(x1, sh, dims=-1)

            yh = h14_ram[img_b].to(dev, dtype=torch.float32)
            yf = f_ram[img_b].to(dev, dtype=torch.float32)
            yp = p16_ram[img_b].to(dev, dtype=torch.float32)
            yl = l12_ram[img_b].to(dev, dtype=torch.float32)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h1, f1, p1, l1 = model(x1, s1_t)
                h2, f2, p2, l2 = model(x2, s2_t)

                # Primary: High-SNR H14 (both raw cosine + centered InfoNCE + cross-subject alignment!)
                l_h_nce = 0.5 * (sym_infonce(h1.float() - mu_h14, yh - mu_h14, model.scale_h14) +
                                 sym_infonce(h2.float() - mu_h14, yh - mu_h14, model.scale_h14))
                l_h_cos = 0.5 * ((1.0 - (nz(h1) * nz(yh)).sum(-1).mean()) + (1.0 - (nz(h2) * nz(yh)).sum(-1).mean()))
                l_h_xsub = 0.35 * sym_infonce(h1.float() - mu_h14, h2.float() - mu_h14, model.scale_h14)

                # Fusion F alignment
                l_f = 0.5 * (sym_infonce(f1.float() - mu_f, yf - mu_f, model.scale_h14) +
                             sym_infonce(f2.float() - mu_f, yf - mu_f, model.scale_h14))

                # Plus16 + Layer-12 spatial geometry
                dp1 = (p1.float() - mu_p16).flatten(1)
                dp2 = (p2.float() - mu_p16).flatten(1)
                dgt = (yp - mu_p16).flatten(1)
                l_p16 = 0.5 * (sym_infonce(dp1, dgt, model.scale_p16) + sym_infonce(dp2, dgt, model.scale_p16)) + \
                        0.4 * (F.smooth_l1_loss(p1.float(), yp, beta=0.5) + F.smooth_l1_loss(p2.float(), yp, beta=0.5))
                l_l12 = 0.4 * (sym_infonce(l1.flatten(1), yl.flatten(1), model.scale_l12) +
                               sym_infonce(l2.flatten(1), yl.flatten(1), model.scale_l12))

                loss = 2.0 * l_h_nce + 2.0 * l_h_cos + l_h_xsub + 0.8 * l_f + 0.8 * l_p16 + 0.5 * l_l12

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot_loss += float(loss.item())

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.inference_mode():
                va_s = va_idx[:400]
                gt_h = h14_ram[va_s].to(dev, dtype=torch.float32)
                gt_p = (p16_ram[va_s].to(dev, dtype=torch.float32) - mu_p16).flatten(1)
                ah, ap, h_rats = [], [], []
                for se in (0, 7, 9):
                    xv = (erp_ram[se, va_s].to(dev, dtype=torch.float32) - eeg_mean[se]) / eeg_std[se]
                    sid = torch.full((len(va_s),), se, device=dev, dtype=torch.long)
                    hv, _, pv, _ = model(xv, sid)
                    sh = nz(hv.float() - mu_h14) @ nz(gt_h - mu_h14).T
                    sp = nz((pv.float() - mu_p16).flatten(1)) @ nz(gt_p).T
                    idx_v = torch.arange(len(va_s), device=dev)
                    ah.append((sh.argmax(-1) == idx_v).float().mean().item() * 100.0)
                    ap.append((sp.argmax(-1) == idx_v).float().mean().item() * 100.0)
                    # Calibrated angle/spread of H14 relative to mu_h14 (strictly OOF train concepts!)
                    r = (gt_h - mu_h14).norm(dim=-1).mean() / (nz(hv.float()) - mu_h14).norm(dim=-1).mean().clamp_min(1e-4)
                    h_rats.append(float(r.item()))
                sc = float(np.mean(ah) + 0.5 * np.mean(ap))
                if sc > best_va:
                    best_va = sc
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    best_h_spread = float(np.clip(np.mean(h_rats), 1.0, 2.2))
                print(f"  [StrongH14 s={seed} ep={ep:02d}] loss={tot_loss/steps_per_ep:.3f} | "
                      f"va400 H14-T1={np.mean(ah):.2f}% Plus16-T1={np.mean(ap):.2f}% h_spread={best_h_spread:.2f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({
        "seed": seed,
        "state_dict": best_state,
        "mu_h14": mu_h14.cpu(),
        "mu_p16": mu_p16.cpu(),
        "h14_raw_norm": h14_raw_norm,
        "h_spread": best_h_spread,
        "best_va": best_va,
    }, MOD_DIR / f"strong_h14_seed{seed}.pt")
    print(f"[SAVED] strong_h14_seed{seed}.pt", flush=True)


# ==============================================================================
# 2. Strictly Single-Query Prediction Loader (StrongDualH14PlusNet + Plus16 + Layout + SQSem)
# ==============================================================================
@torch.inference_mode()
def predict_v2_subject(sub: int, budget: str = "r80", dev: str = "cuda:0"):
    cache_p = MOD_DIR / f"sq_preds_v2_{budget}_sub-{sub:02d}.pt"
    if cache_p.exists():
        return torch.load(cache_p, map_location="cpu", weights_only=False)

    stats = torch.load(MOD_DIR / "train_eeg_stats.pt", map_location="cpu", weights_only=True)
    eeg_mean = stats["mean"][sub - 1].to(dev)
    eeg_std = stats["std"][sub - 1].to(dev)
    x_test = np.load(PREP / f"sub-{sub:02d}_test.npy", mmap_mode="r")
    groups = np.arange(80).reshape(20, 4).tolist() if budget == "r80" else [[0, 1, 2, 3]]

    # Load StrongDualH14PlusNet ensemble
    sh_ckpts = sorted(MOD_DIR.glob("strong_h14_seed*.pt"))
    sh_models = []
    for cp in sh_ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        m = StrongDualH14PlusNet(n_subs=10, d_trunk=1024, drop=0.0).to(dev).eval()
        m.load_state_dict(ck["state_dict"])
        sh_models.append((m, ck["mu_h14"].to(dev), ck["mu_p16"].to(dev), ck["h14_raw_norm"], ck["h_spread"]))

    # Load existing CrossBrainPlus16Net + SharpLayoutNet + SQSemNet
    from train_photoeeg_encoders import CrossBrainPlus16Net
    p16_models = []
    for cp in sorted(MOD_DIR.glob("plus16_seed*.pt")):
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        m = CrossBrainPlus16Net(n_subs=10, d_model=384, n_queries=16, drop=0.0).to(dev).eval()
        m.load_state_dict(ck["state_dict"])
        p16_models.append((m, ck["mu_p16"].to(dev)))

    lay_models = []
    for cp in sorted(MOD_DIR.glob("layout_seed*.pt")):
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        m = SharpLayoutNet(n_subs=10, drop=0.0).to(dev).eval()
        m.load_state_dict(ck["state_dict"])
        lay_models.append((m, ck["mu_lat"].to(dev), ck["gamma"]))

    sem_models = []
    for cp in sorted(MOD_DIR.glob("sqsem_seed*.pt")):
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        m = SQSemNet(n_subs=10, drop=0.0).to(dev).eval()
        m.load_state_dict(ck["state_dict"])
        sem_models.append(m)

    N = len(x_test)
    out_h14 = torch.zeros((N, 1024), device=dev)
    out_p16 = torch.zeros((N, 16, 2048), device=dev)
    out_l12 = torch.zeros((N, 16, 1280), device=dev)
    out_lat = torch.zeros((N, 4, 64, 64), device=dev)
    out_rgb = torch.zeros((N, 3, 16, 16), device=dev)
    out_sem_h = torch.zeros((N, 1024), device=dev)
    out_sem_b = torch.zeros((N, 1280), device=dev)

    mu_h14 = sh_models[0][1]
    mu_p16 = sh_models[0][2]
    h14_raw_norm = sh_models[0][3]

    for start in range(0, N, 40):
        sl = slice(start, min(N, start + 40))
        xb_all = torch.from_numpy(np.array(x_test[sl], copy=True)).to(dev, dtype=torch.float32)
        B = xb_all.shape[0]
        sid = torch.full((B,), sub - 1, device=dev, dtype=torch.long)
        sid_t = torch.full((B * 4,), sub - 1, device=dev, dtype=torch.long)

        gh, gp, gl12, glat, grgb, gsh, gsb = 0, 0, 0, 0, 0, 0, 0
        for grp in groups:
            bb = xb_all[:, grp]
            x_erp = (prec_erp_analytic(bb, p=0.5) - eeg_mean) / eeg_std
            x_tri = (bb.reshape(B * 4, 17, 250) - eeg_mean) / eeg_std

            for m, _, _, _, _ in sh_models:
                he, _, pe, le = m(x_erp, sid)
                ht, _, pt, lt = m(x_tri, sid_t)
                h = 0.72 * nz(he) + 0.28 * nz(ht.view(B, 4, 1024).mean(1))
                p = 0.72 * pe + 0.28 * pt.view(B, 4, 16, 2048).mean(1)
                l = 0.72 * nz(le) + 0.28 * nz(lt.view(B, 4, 16, 1280).mean(1))
                gh = gh + 0.30 * h / len(sh_models)
                gp = gp + 0.25 * p / len(sh_models)
                gl12 = gl12 + 0.25 * l / len(sh_models)

            for m, _ in p16_models:
                pe, le, he, _ = m(x_erp, sid)
                pt, lt, ht, _ = m(x_tri, sid_t)
                h = 0.72 * nz(he) + 0.28 * nz(ht.view(B, 4, 1024).mean(1))
                p = 0.72 * pe + 0.28 * pt.view(B, 4, 16, 2048).mean(1)
                l = 0.72 * nz(le) + 0.28 * nz(lt.view(B, 4, 16, 1280).mean(1))
                gh = gh + 0.70 * h / len(p16_models)
                gp = gp + 0.75 * p / len(p16_models)
                gl12 = gl12 + 0.75 * l / len(p16_models)

            for m, mu_l, gam in lay_models:
                pl, pr = m(x_erp, sid)
                glat = glat + (mu_l + gam * pl) / len(lay_models)
                grgb = grgb + pr / len(lay_models)

            for m in sem_models:
                she, sbe = m(x_erp, sid)
                sht, sbt = m(x_tri, sid_t)
                gsh = gsh + (0.72 * nz(she) + 0.28 * nz(sht.view(B, 4, 1024).mean(1))) / len(sem_models)
                gsb = gsb + (0.72 * nz(sbe) + 0.28 * nz(sbt.view(B, 4, 1280).mean(1))) / len(sem_models)

        ng = float(len(groups))
        out_h14[sl] = nz(gh / ng)
        out_p16[sl] = gp / ng
        out_l12[sl] = nz(gl12 / ng)
        out_lat[sl] = glat / ng
        out_rgb[sl] = (grgb / ng).clamp(0, 1)
        out_sem_h[sl] = nz(gsh / ng)
        out_sem_b[sl] = nz(gsb / ng)

    # Strictly single-query CV-OpenVocab (8,141) words (train-only centering!)
    tb = torch.load(RAD / "runs/obj_bench/text_banks/text_banks_laion.pt", map_location="cpu", weights_only=False)
    tr_cid = tb["tr_img2cid"].long()
    mu_th = tb["h14_train"][tr_cid].float().mean(0, keepdim=True)
    mu_tb = tb["bigg_train"][tr_cid].float().mean(0, keepdim=True)
    H_cv = nz(tb["h14_cv_open"].float() - mu_th, dim=-1).to(dev)
    B_cv = nz(tb["bigg_cv_open"].float() - mu_tb, dim=-1).to(dev)
    words = tb["cv_open"]

    # Fuse visual H14 (mapped via train-only ridge bridge) + SQSem for richer open-world words!
    H_tr_img = torch.from_numpy(np.load(TGT_DIR / "h14_fin_train.npy")).float()
    Yh = tb["h14_train"][tr_cid].float() - mu_th
    Hc = H_tr_img - H_tr_img.mean(0, keepdim=True)
    W_it = torch.linalg.solve(Hc.T @ Hc + 80.0 * torch.eye(1024), Hc.T @ Yh).to(dev)
    h14_to_txt = nz((out_h14 - mu_h14) @ W_it, dim=-1)

    sim_cv = 0.40 * (out_sem_h @ H_cv.T) + 0.35 * (out_sem_b @ B_cv.T) + 0.25 * (h14_to_txt @ H_cv.T)
    top_vals, top_idx = sim_cv.topk(5, dim=-1)
    top_words = [[words[int(i)] for i in row] for row in top_idx.cpu().tolist()]

    # Diagnostic retrieval printout (never fed back into model!)
    gt_h14 = torch.from_numpy(np.load(TGT_DIR / "h14_fin_test.npy")).float().to(dev)
    gt_p16 = torch.from_numpy(np.load(TGT_DIR / "plus16_test.npy")).float().to(dev)
    idx200 = torch.arange(200, device=dev)
    sh = nz(out_h14 - mu_h14) @ nz(gt_h14 - mu_h14).T
    sp = nz((out_p16 - mu_p16).flatten(1)) @ nz((gt_p16 - mu_p16).flatten(1)).T
    h_t1 = (sh.argmax(-1) == idx200).float().mean().item() * 100.0
    h_t5 = (sh.topk(5, -1).indices == idx200[:, None]).any(-1).float().mean().item() * 100.0
    p_t1 = (sp.argmax(-1) == idx200).float().mean().item() * 100.0
    p_t5 = (sp.topk(5, -1).indices == idx200[:, None]).any(-1).float().mean().item() * 100.0
    print(f"  [V2 SQ-Preds {budget} sub-{sub:02d}] H14-200w={h_t1:.1f}%/{h_t5:.1f}% | Plus16-200w={p_t1:.1f}%/{p_t5:.1f}% | "
          f"words0-5={[w[0] for w in top_words[:6]]}", flush=True)

    pack = {
        "h14": out_h14.half().cpu(),
        "mu_h14": mu_h14.half().cpu(),
        "h14_raw_norm": float(h14_raw_norm),
        "p16": out_p16.half().cpu(),
        "mu_p16": mu_p16.half().cpu(),
        "l12": out_l12.half().cpu(),
        "lat": out_lat.half().cpu(),
        "rgb16": out_rgb.half().cpu(),
        "sem_h": out_sem_h.half().cpu(),
        "sem_b": out_sem_b.half().cpu(),
        "top_words": top_words,
        "diag": {"h14_t1": h_t1, "h14_t5": h_t5, "p16_t1": p_t1, "p16_t5": p_t5},
    }
    torch.save(pack, cache_p)
    return pack


# ==============================================================================
# 3. Dual-IP-Adapter (Global H14 + Spatial Plus16) SDXL-Base-1.0 Photorealistic Generator
# ==============================================================================
class DualIPSDXLBaseGenerator:
    """Loads BOTH ip-adapter_sdxl_vit-h (1024-D global H14 -> 4 tokens, 100% on-manifold!)
    AND ip-adapter-plus_sdxl_vit-h (16x2048 spatial tokens!) into SDXL-Base-1.0 simultaneously!
    Uses TRUE uncond=ZERO (fixing the 10.6x CFG over-amplification bug!).
    """
    def __init__(self, dev: str = "cuda:0"):
        self.dev = dev
        pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=torch.float16, variant="fp16")
        pipe.unet = UNet2DConditionModel.from_pretrained(str(SDXL_BASE / "unet"), torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(str(SDXL_BASE / "scheduler"), use_karras_sigmas=True)
        # Load BOTH adapters in exact order [0: standard H14 (1024->4x2048), 1: plus (16x2048)]!
        pipe.load_ip_adapter(
            [str(IPA_DIR), str(IPA_DIR)],
            subfolder=["sdxl_models", "sdxl_models"],
            weight_name=["ip-adapter_sdxl_vit-h.safetensors", "ip-adapter-plus_sdxl_vit-h.safetensors"],
            image_encoder_folder=None,
        )
        # Keep adapter 0's official ImageProjection (1024 -> 4x2048)! Replace ONLY adapter 1's Resampler with Identity!
        pipe.unet.encoder_hid_proj.image_projection_layers[1] = IdentityImageProj()
        pipe.to(dev)
        pipe.vae.to(dtype=torch.float32)
        self.pipe = pipe
        self.unet = pipe.unet.eval()
        self.vae = pipe.vae.eval()
        self.scheduler = pipe.scheduler
        self._prompt_cache = {}

    @torch.inference_mode()
    def encode_prompt_cached(self, text: str):
        if text not in self._prompt_cache:
            pe, _, ppe, _ = self.pipe.encode_prompt(
                prompt=text, device=self.dev, num_images_per_prompt=1, do_classifier_free_guidance=False
            )
            self._prompt_cache[text] = (pe.half(), ppe.half())
        return self._prompt_cache[text]

    @torch.inference_mode()
    def generate(
        self,
        h14_emb: torch.Tensor,          # [N, 1024] scaled to true H14 norm (~18.5)
        plus16_tok: torch.Tensor,       # [N, 16, 2048]
        z0_lat: torch.Tensor = None,
        prompts: list = None,
        neg_prompt: str = "clay, plastic, 3d render, blurry, low contrast, abstract, distorted, ugly, artifact",
        scale_std: float = 0.75,
        scale_plus: float = 0.25,
        cfg_scale: float = 4.5,
        num_steps: int = 20,
        strength: float = 0.80,
        sub_id: int = 1,
        draw_k: int = 0,
        batch_size: int = 5,
    ) -> torch.Tensor:
        self.pipe.set_ip_adapter_scale([scale_std, scale_plus])
        N = h14_emb.shape[0]
        if prompts is None:
            prompts = ["a realistic natural photograph, crisp detail, natural lighting"] * N

        neg_pe_1, neg_ppe_1 = self.encode_prompt_cached(neg_prompt)
        time_ids_1 = torch.tensor([[512.0, 512.0, 0.0, 0.0, 512.0, 512.0]], device=self.dev, dtype=torch.float16)

        # TRUE official IP-Adapter unconditional embeddings = ZEROS! (NO 10.6x over-amplification!)
        u_h14_1 = torch.zeros((1, 1, 1024), device=self.dev, dtype=torch.float16)
        u_p16_1 = torch.zeros((1, 16, 2048), device=self.dev, dtype=torch.float16)

        out_u8 = torch.empty((N, 3, 512, 512), dtype=torch.uint8)
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            B = end - start
            noises = []
            for qi in range(start, end):
                g = torch.Generator(device="cpu").manual_seed(query_seed(sub_id, qi, draw_k))
                noises.append(torch.randn((4, 64, 64), generator=g))
            eps = torch.stack(noises, 0).to(self.dev, dtype=torch.float16)

            pes, ppes = zip(*[self.encode_prompt_cached(prompts[i]) for i in range(start, end)])
            cond_pe = torch.cat(pes, 0)
            cond_ppe = torch.cat(ppes, 0)
            neg_pe = neg_pe_1.expand(B, -1, -1)
            neg_ppe = neg_ppe_1.expand(B, -1)
            time_ids = time_ids_1.expand(B, -1)

            c_h14 = h14_emb[start:end].to(self.dev, dtype=torch.float16).unsqueeze(1)  # [B, 1, 1024]
            c_p16 = plus16_tok[start:end].to(self.dev, dtype=torch.float16)           # [B, 16, 2048]
            u_h14 = u_h14_1.expand(B, -1, -1)
            u_p16 = u_p16_1.expand(B, -1, -1)

            self.scheduler.set_timesteps(num_steps, device=self.dev)
            timesteps = self.scheduler.timesteps
            if z0_lat is not None and strength < 0.999:
                t_start = max(0, min(num_steps - 1, int(round(num_steps * (1.0 - strength)))))
                timesteps = timesteps[t_start:]
                z0_b = z0_lat[start:end].to(self.dev, dtype=torch.float16)
                latents = self.scheduler.add_noise(z0_b, eps, timesteps[:1].expand(B))
            else:
                latents = eps * self.scheduler.init_noise_sigma

            for t in timesteps:
                lat_in = torch.cat([latents, latents], dim=0)
                lat_in = self.scheduler.scale_model_input(lat_in, t)
                pe_in = torch.cat([neg_pe, cond_pe], dim=0)
                ppe_in = torch.cat([neg_ppe, cond_ppe], dim=0)
                tid_in = torch.cat([time_ids, time_ids], dim=0)
                ip_h_in = torch.cat([u_h14, c_h14], dim=0)
                ip_p_in = torch.cat([u_p16, c_p16], dim=0)
                noise_pred = self.unet(
                    lat_in, t, encoder_hidden_states=pe_in,
                    added_cond_kwargs={"text_embeds": ppe_in, "time_ids": tid_in, "image_embeds": [ip_h_in, ip_p_in]},
                ).sample
                n_uncond, n_cond = noise_pred.chunk(2)
                noise_pred = n_uncond + cfg_scale * (n_cond - n_uncond)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            img = self.vae.decode(latents.float() / self.vae.config.scaling_factor).sample
            out_u8[start:end] = (img * 0.5 + 0.5).clamp(0, 1).mul(255).round().byte().cpu()
        return out_u8


def calibrate_h14(h14_u: torch.Tensor, mu_h14: torch.Tensor, raw_norm: float, angle_gain: float = 1.60) -> torch.Tensor:
    """Strictly train-only cone angle calibration (matches natural H14 spread on unit sphere * raw_norm!)."""
    u = nz(mu_h14.float(), dim=-1)
    h = nz(h14_u.float(), dim=-1)
    # Decompose along train mean direction u and orthogonal component
    proj = (h * u).sum(-1, keepdim=True) * u
    orth = h - proj
    cal = nz(proj + angle_gain * orth, dim=-1) * raw_norm
    return cal


def run_diag(subs: list, budget: str = "r80", dev: str = "cuda:0"):
    gen = DualIPSDXLBaseGenerator(dev=dev)
    ver = SpatialPhotoVerifier(dev=dev)
    ev = PhotoFidelityEvaluator(dev)

    # First quick 20-image sanity check of Dual-IP Oracle on sub-10!
    gt_h14 = torch.from_numpy(np.load(TGT_DIR / "h14_fin_test.npy")).float() * 18.5
    gt_p16 = torch.from_numpy(np.load(TGT_DIR / "plus16_test.npy")).float()
    if not (OUT_ROOT / "oracle_std_h14_check.json").exists():
        imgs_or = gen.generate(gt_h14, gt_p16, scale_std=0.85, scale_plus=0.0, cfg_scale=4.5, num_steps=16, sub_id=0, draw_k=0)
        m_or = ev.evaluate(imgs_or, compute_lowlevel=True)
        print(f"[ORACLE Std-H14 SDXL-Base] DINO_T1={m_or['dino_t1']:.1f}% CLIP_Img={m_or['clip_img200_t1']:.1f}% "
              f"ZS200={m_or['zs_obj200_t1']:.1f}% Crisp={m_or['obj_crisp']:.1f}% 2wC={m_or['clip_2way']:.1f}%", flush=True)
        json.dump(m_or, open(OUT_ROOT / "oracle_std_h14_check.json", "w"), indent=2)

    arms = [
        "D0_Plus16_FixZeroCFG",
        "D1_StdH14_NoLat_g16",
        "D2_StdH14_SDE78_g16",
        "D3_StdH14_SDE78_g20_SQP2",
        "D4_DualIP_70_30_SDE78",
        "D5_DualIP_65_35_SDE78_SQP2",
        "D6_DualIP_PhotoVerify_Sel4",
    ]
    for s in subs:
        pk = predict_v2_subject(s, budget=budget, dev=dev)
        h14_g16 = calibrate_h14(pk["h14"], pk["mu_h14"], pk["h14_raw_norm"], angle_gain=1.60)
        h14_g20 = calibrate_h14(pk["h14"], pk["mu_h14"], pk["h14_raw_norm"], angle_gain=2.00)
        # Mildly calibrated Plus16 (1.25x, NOT 10.6x!)
        p16_125 = pk["mu_p16"].float() + 1.25 * (pk["p16"].float() - pk["mu_p16"].float())
        p16_100 = pk["p16"].float()

        p_gen = build_prompt_list(pk["top_words"], "photo_generic")
        p_sq2 = build_prompt_list(pk["top_words"], "sq_top2")
        p_sq1 = build_prompt_list(pk["top_words"], "sq_top1")

        gal_cols = []
        for arm in arms:
            arm_d = OUT_ROOT / f"{budget}_{arm}"
            arm_d.mkdir(parents=True, exist_ok=True)
            cp = arm_d / f"sub-{s:02d}.pt"
            if cp.exists():
                d = torch.load(cp, map_location="cpu", weights_only=False)
                imgs, m = d["imgs_u8"], d["metrics"]
            else:
                t0 = time.time()
                hist = None
                if arm == "D0_Plus16_FixZeroCFG":
                    imgs = gen.generate(h14_g16, p16_125, z0_lat=pk["lat"], prompts=p_gen,
                                        scale_std=0.0, scale_plus=0.75, cfg_scale=4.5, num_steps=20, strength=0.78, sub_id=s, draw_k=0)
                elif arm == "D1_StdH14_NoLat_g16":
                    imgs = gen.generate(h14_g16, p16_100, z0_lat=None, prompts=p_gen,
                                        scale_std=0.85, scale_plus=0.0, cfg_scale=4.5, num_steps=18, sub_id=s, draw_k=0)
                elif arm == "D2_StdH14_SDE78_g16":
                    imgs = gen.generate(h14_g16, p16_100, z0_lat=pk["lat"], prompts=p_gen,
                                        scale_std=0.85, scale_plus=0.0, cfg_scale=4.5, num_steps=20, strength=0.78, sub_id=s, draw_k=0)
                elif arm == "D3_StdH14_SDE78_g20_SQP2":
                    imgs = gen.generate(h14_g20, p16_100, z0_lat=pk["lat"], prompts=p_sq2,
                                        scale_std=0.80, scale_plus=0.0, cfg_scale=4.8, num_steps=20, strength=0.78, sub_id=s, draw_k=0)
                elif arm == "D4_DualIP_70_30_SDE78":
                    imgs = gen.generate(h14_g20, p16_125, z0_lat=pk["lat"], prompts=p_gen,
                                        scale_std=0.70, scale_plus=0.30, cfg_scale=4.5, num_steps=20, strength=0.78, sub_id=s, draw_k=0)
                elif arm == "D5_DualIP_65_35_SDE78_SQP2":
                    imgs = gen.generate(h14_g20, p16_125, z0_lat=pk["lat"], prompts=p_sq2,
                                        scale_std=0.65, scale_plus=0.35, cfg_scale=4.8, num_steps=20, strength=0.78, sub_id=s, draw_k=0)
                elif arm == "D6_DualIP_PhotoVerify_Sel4":
                    cands = [
                        gen.generate(h14_g16, p16_100, z0_lat=pk["lat"], prompts=p_gen,
                                     scale_std=0.85, scale_plus=0.0, cfg_scale=4.5, num_steps=20, strength=0.78, sub_id=s, draw_k=0),
                        gen.generate(h14_g20, p16_125, z0_lat=pk["lat"], prompts=p_gen,
                                     scale_std=0.70, scale_plus=0.30, cfg_scale=4.5, num_steps=20, strength=0.75, sub_id=s, draw_k=1),
                        gen.generate(h14_g20, p16_125, z0_lat=pk["lat"], prompts=p_sq2,
                                     scale_std=0.70, scale_plus=0.25, cfg_scale=4.8, num_steps=20, strength=0.78, sub_id=s, draw_k=2),
                        gen.generate(h14_g20, p16_125, z0_lat=pk["lat"], prompts=p_sq1,
                                     scale_std=0.65, scale_plus=0.25, cfg_scale=5.0, num_steps=20, strength=0.80, sub_id=s, draw_k=3),
                    ]
                    imgs, hist = ver.select_best(cands, pk["l12"], pk["h14"], pk["sem_h"], pk["rgb16"],
                                                 w_l12=0.35, w_h14=0.40, w_sem=0.15, w_rgb=0.10)
                m = ev.evaluate(imgs, compute_lowlevel=True)
                if hist is not None:
                    m["cand_hist"] = hist
                torch.save({"imgs_u8": imgs, "metrics": m}, cp)
                print(
                    f"  [{budget}:{arm:26s}] sub-{s:02d} ({time.time()-t0:5.1f}s) | "
                    f"DINO_T1={m['dino_t1']:5.2f}% cos={m['dino_cos']:.3f} p8={m['dino_patch_cos']:.3f} | "
                    f"CLIP_Img={m['clip_img200_t1']:5.2f}% ZS200={m['zs_obj200_t1']:5.2f}% ZS1823={m['zs_obj1823_t1']:4.2f}% | "
                    f"Crisp={m['obj_crisp']:4.1f}% Alex2/5={m['alex2']:4.1f}/{m['alex5']:4.1f} Incep={m['incep_2way']:4.1f} "
                    f"2wC={m['clip_2way']:4.1f} Pix={m['pixcorr']:.3f} SSIM={m['ssim']:.3f}" +
                    (f" hist={hist}" if hist else ""),
                    flush=True,
                )
            gal_cols.append((arm, imgs))
        save_gallery(ev.gt_512, gal_cols, ev.test_names, OUT_ROOT / f"gallery_v2_{budget}_sub-{s:02d}.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["train", "diag"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--subs", type=str, default="10")
    ap.add_argument("--budget", type=str, default="r80")
    ap.add_argument("--dev", type=str, default="cuda:0")
    args = ap.parse_args()
    if args.mode == "train":
        train_strong_h14(args.seed, args.dev, args.epochs)
    else:
        subs = [int(x) for x in args.subs.split(",") if x.strip()]
        run_diag(subs, budget=args.budget, dev=args.dev)


if __name__ == "__main__":
    main()
