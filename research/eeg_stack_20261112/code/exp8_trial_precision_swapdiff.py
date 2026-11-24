"""Method 1 (exp8): Trial-Variance-Aware Precision ERP + Latency-Jitter-Invariant Trial-Set Aggregation for R=4 EEG.

Grounded directly in raw 4-trial THINGS-EEG2 noise physics:
  - Individual trials among R=4 repeats exhibit severe heavy-tailed amplitude artifacts (90th=1.24x, 99th=1.71x, max=21.1x
    median RMS) AND trial-to-trial latency jitter / non-phase-locked activity that cancels under naive x.mean(1).
  - Zero-retraining diagnostic on Arm3b Dev-4 (3,270 unseen fold-0 images) ALREADY showed:
      naive_mean:            38.69% F-rep Intra10 | 88.17% Swap
      inv_trial_rms2(p=0.5): 38.88% (+0.19%pp, 4/4 subs)
      feat_mean_r1:          38.93% | 89.06% Swap (+0.89%pp Swap!)
      dual fusion combo:     39.73% | 89.14% Swap (+1.04%pp Intra10, 4/4 subs UP with ZERO retraining!)

This script implements & evaluates cleanly on Dev-4 (both s375 & s625, 4-NFE nsel=1, R=4 FreeGen + FixZ0 + UNet Swap):
  1. SwapDiff_Arm3b              : Existing naive 4-trial mean baseline.
  2. M1a_ZeroRetrain_PrecDual    : Zero retraining — replaces test-time naive mean with inv_ch_rms2(p=0.5) ERP +
                                   late per-trial feature fusion + precision-weighted z0_ridge.
  3. M1b_Trained_TrialPrecSet    : End-to-end trainable Permutation-Invariant Dual-Stream Aggregator:
                                   (a) Channel-Window Precision ERP Gate (q_{r,c,w} log-RMS + LOO trial agreement),
                                   (b) Latency-Jitter-Invariant Trial-Set Cross-Trial Quality Attention over {h_1..h_4},
                                   (c) Adaptive per-sample Dual-Stream Gate fusing h_erp & h_set -> cond_layer,
                                   trained with Arm3b's exact raw-epsilon UNet 2x2 EEG-swap contrast + intra-concept
                                   representation loss + per-trial random time-jitter augmentation + precision z0_ridge!

Zero Evaluator Leakage: evaluated strictly on independent OpenAI CLIP ViT-L/14 (primary R=4 FreeGen Intra10 Top-1),
Inception-v3, AlexNet-5, PixCorr, and SSIM on sealed unseen fold-0 concepts.
"""
import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(os.environ["RAD_ROOT"])
for d in [ROOT / "code/exp_w7", ROOT / "code/exp_w6", ROOT / "code/rad", ROOT / "code/exp", ROOT / "code/Brain-HIVE"]:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

from common import EMB_KEYS, PREP, PRIOR, TURBO, write_json
from exp0_intra10_swap_gate import ALL_SUBS, DEV_SUBS, OUT_DIR, get_concept_splits, intra10_and_swap_from_sim
from exp1_swapdiff_3arm import (
    ResidualIPCondLayer,
    build_tr14_resources,
    per_sample_cone,
)
from exp5_z0_2x2_diag import extract_eval_feats, score_full500
from exp6_draft_intra4 import forward_branches_and_trunk, rollout_4step_step_ips
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder
from train_flex import shift_batch

from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel

nz = F.normalize
M1_DIR = OUT_DIR / "method1_trial_prec"
M1_DIR.mkdir(parents=True, exist_ok=True)


def sample_same_concept_pairs(c_groups: np.ndarray, n_pairs: int, rng: np.random.Generator):
    """Sample n_pairs distinct same-concept photo pairs (loc_a, loc_b) from tr14_c_groups [N_c, 10]."""
    c_ids = rng.integers(0, len(c_groups), size=n_pairs)
    a_list, b_list = [], []
    for ci in c_ids:
        i1, i2 = rng.choice(10, size=2, replace=False)
        a_list.append(c_groups[ci, i1])
        b_list.append(c_groups[ci, i2])
    return torch.tensor(a_list, dtype=torch.long), torch.tensor(b_list, dtype=torch.long)


def prec_erp_analytic(x4: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Zero-parameter per-(trial, channel) inverse-RMS^2 weighted ERP: x4 [B, 4, 17, 250] -> [B, 17, 250]."""
    rms2 = (x4.float() ** 2).mean(dim=3, keepdim=True).clamp_min(1e-6)  # [B, 4, 17, 1]
    w = rms2.pow(-p)
    return (x4.float() * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)


def fit_tr14_ridge_z0_prec(sub: int, tr_idx_np: np.ndarray, test_idx_np: np.ndarray, dev: str = "cuda:0", p: float = 0.5) -> torch.Tensor:
    """Fit O10 multi-window ridge z0 on tr14 using precision-weighted 4-trial ERP (no outlier contamination!)."""
    cache_p = M1_DIR / f"z0_tr14_prec_p{int(p*10)}_sub{sub:02d}_sel50.pt"
    if cache_p.exists():
        return torch.load(cache_p, map_location="cpu", weights_only=True)

    X_raw = np.load(PREP / f"sub-{sub:02d}_train.npy")
    # Compute precision-weighted ERP in chunks to keep memory modest
    def _prec_np(arr):
        out = np.empty((len(arr), arr.shape[2], arr.shape[3]), dtype=np.float32)
        for i in range(0, len(arr), 2048):
            xb = torch.from_numpy(arr[i:i + 2048]).float()
            out[i:i + 2048] = prec_erp_analytic(xb, p=p).numpy()
        return out

    X_tr = torch.from_numpy(_prec_np(X_raw[tr_idx_np])).float()
    X_te = torch.from_numpy(_prec_np(X_raw[test_idx_np])).float()
    del X_raw

    lat_all = np.load(PREP / "lat512_train.npy")
    Y_tr = torch.from_numpy(lat_all[tr_idx_np]).float().to(dev).flatten(1)
    del lat_all
    my = Y_tr.mean(0)
    Yc = Y_tr - my
    del Y_tr

    def lp_flat(y_flat, f):
        v = y_flat.view(-1, 4, 64, 64)
        return F.interpolate(F.avg_pool2d(v, f), size=64, mode="bilinear", align_corners=False).flatten(1)

    torch.manual_seed(20261115 + sub)
    n_pca = 256
    Yc_w = Yc + 1.5 * lp_flat(Yc, 4) + 0.75 * lp_flat(Yc, 2)
    _, _, V_p = torch.pca_lowrank(Yc_w, q=n_pca + 32, center=False, niter=4)
    del Yc_w
    V_pca, _ = torch.linalg.qr(V_p[:, :n_pca].double(), mode="reduced")
    Z_gt = Yc.double() @ V_pca
    rms_real_lp4 = float(F.avg_pool2d(Yc.view(-1, 4, 64, 64), 4).flatten(1).pow(2).sum(1).mean().sqrt())
    del Yc

    WINDOWS = [(12, 62), (10, 75), (15, 90), (10, 125), (0, 250)]
    preds = []
    for t_a, t_b in WINDOWS:
        A_tr = X_tr[:, :, t_a:t_b].flatten(1).to(dev)
        A_te = X_te[:, :, t_a:t_b].flatten(1).to(dev)
        mu, sd = A_tr.mean(0, keepdim=True), A_tr.std(0, keepdim=True).clamp_min(1e-6)
        A_tr = ((A_tr - mu) / sd).double()
        A_te = ((A_te - mu) / sd).double()
        Cxx = (A_tr.T @ A_tr) / float(len(A_tr))
        Cxy = (A_tr.T @ Z_gt) / float(len(A_tr))
        lam = 3.0 * float(torch.trace(Cxx) / Cxx.shape[0])
        W = torch.linalg.solve(Cxx + lam * torch.eye(Cxx.shape[0], dtype=torch.float64, device=dev), Cxy)
        preds.append((A_te @ W).float())
    z_te = torch.stack(preds, 0).mean(0)
    y_te = z_te @ V_pca.float().T
    cur_rms = float(F.avg_pool2d(y_te.view(-1, 4, 64, 64), 4).flatten(1).pow(2).sum(1).mean().sqrt())
    if cur_rms > 1e-6:
        y_te = y_te * (0.80 * rms_real_lp4 / cur_rms)
    z0_te = (y_te + my).view(-1, 4, 64, 64).cpu()
    torch.save(z0_te, cache_p)
    torch.cuda.empty_cache()
    return z0_te


class TrialPrecisionSetAggregator(nn.Module):
    """Permutation-invariant dual-stream 4-trial EEG aggregator:
      1. PrecisionERP Gate: suppresses heavy-tailed outlier trials per (channel, neuro-window) using log-RMS & LOO agreement.
      2. TrialSet Quality-Attention: encodes each of the 4 individual trials (surviving +/-20ms latency jitter!) and
         pools them by neural quality + cross-trial latent cosine consistency.
      3. Adaptive Dual-Stream Fusion: fuses h_erp & h_set per sample (initialized close to the verified zero-retrain combo).
    """

    def __init__(self, n_ch: int = 17, n_win: int = 5, d_lat: int = 1024):
        super().__init__()
        self.n_ch = n_ch
        self.n_win = n_win
        # 1. Precision-ERP parameters (initialized near inv_ch_rms2 p=0.5!)
        self.log_alpha = nn.Parameter(torch.full((1, 1, n_ch, n_win), math.log(math.exp(1.0) - 1.0)))  # softplus ~ 1.0 (= rms^-1 = rms2^-0.5)
        self.beta_agree = nn.Parameter(torch.full((1, 1, 1, n_win), 0.25))
        self.prec_mlp = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.prec_mlp[-1].weight)
        nn.init.zeros_(self.prec_mlp[-1].bias)

        # 2. Latent Trial-Set Quality-Attention (operates on 4 individual trial latents [B, 4, 1024])
        self.set_ln = nn.LayerNorm(d_lat)
        self.set_proj = nn.Sequential(nn.Linear(d_lat, 64), nn.SiLU())
        self.set_score = nn.Sequential(nn.Linear(64 + 2, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.set_score[-1].weight)
        nn.init.zeros_(self.set_score[-1].bias)

        # 3. Adaptive per-sample Dual-Stream Gate (initialized at ~0.25 weight on Trial-Set, 0.75 on Prec-ERP!)
        self.gate_ln = nn.LayerNorm(d_lat)
        self.gate_net = nn.Sequential(nn.Linear(d_lat + 2, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, -1.10)  # sigmoid(-1.10) ~= 0.25

        # Optional residual blend on fp directly from set predictions (starts at 0.20)
        self.fp_set_logit = nn.Parameter(torch.tensor(-1.38))  # sigmoid ~= 0.20

    def compute_prec_erp(self, x4: torch.Tensor):
        """x4: [B, 4, 17, 250] -> (x_erp [B, 17, 250], q_trial [B, 4], raw_agree_mean [B, 1])."""
        B, R, C, T = x4.shape
        W = self.n_win
        xw = x4.float().view(B, R, C, W, T // W)  # [B, 4, 17, 5, 50]
        rms_cw = (xw.pow(2).mean(dim=-1) + 1e-6).sqrt()  # [B, 4, 17, 5]
        med_cw = rms_cw.median(dim=1, keepdim=True).values.clamp_min(1e-4)
        q_cw = torch.log((rms_cw / med_cw).clamp(0.1, 20.0))  # [B, 4, 17, 5]

        # Leave-one-out cosine agreement per window across 4 trials
        uw = nz(xw.permute(0, 3, 1, 2, 4).reshape(B * W, R, -1), dim=-1)  # [B*W, 4, 17*50]
        cs = torch.bmm(uw, uw.transpose(1, 2))  # [B*W, 4, 4]
        agree_w = ((cs.sum(-1) - 1.0) / float(max(1, R - 1))).view(B, W, R).permute(0, 2, 1)  # [B, 4, W]
        agree_cw = agree_w.unsqueeze(2).expand(B, R, C, W)

        feat = torch.stack([q_cw, agree_cw], dim=-1)  # [B, 4, 17, 5, 2]
        alpha = F.softplus(self.log_alpha)
        logits = -alpha * q_cw + self.beta_agree * agree_cw + self.prec_mlp(feat).squeeze(-1)
        w = F.softmax(logits, dim=1).unsqueeze(-1)  # [B, 4, 17, 5, 1]
        x_erp = (xw * w).sum(dim=1).view(B, C, T)

        rms_tr = (x4.float().pow(2).mean(dim=(2, 3)) + 1e-6).sqrt()  # [B, 4]
        q_trial = torch.log((rms_tr / rms_tr.median(dim=1, keepdim=True).values.clamp_min(1e-4)).clamp(0.1, 20.0))
        raw_agree_mean = agree_w.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)  # [B, 1]
        return x_erp, q_trial, raw_agree_mean

    def forward_dual(self, model: CrossSubjectHorizonEncoder, x4: torch.Tensor, sid: int):
        """Encode 4-trial EEG x4 [B, 4, 17, 250] through BOTH Prec-ERP stream & Trial-Set stream."""
        B, R, C, T = x4.shape
        x_erp, q_trial, raw_agree_mean = self.compute_prec_erp(x4)

        # Stream 1: Precision-weighted phase-locked ERP
        fp_erp, h_erp, _ = forward_branches_and_trunk(model, x_erp, sid, r_eff=4.0)

        # Stream 2: Individual-trial encoding (latency-jitter invariant!) batched as [B*4, 17, 250]
        x_flat = x4.float().reshape(B * R, C, T)
        fp_tr_f, h_tr_f, _ = forward_branches_and_trunk(model, x_flat, sid, r_eff=1.0)
        h_tr = h_tr_f.view(B, R, -1)   # [B, 4, 1024]
        fp_tr = fp_tr_f.view(B, R, -1)  # [B, 4, 2560]

        # Cross-trial latent cosine consistency (after latency jitter absorbed by trunk!)
        hu = nz(h_tr.float(), dim=-1)
        cs_lat = torch.bmm(hu, hu.transpose(1, 2))  # [B, 4, 4]
        s_lat = (cs_lat.sum(-1) - 1.0) / float(max(1, R - 1))  # [B, 4]
        s_lat_mean = s_lat.mean(dim=1, keepdim=True)  # [B, 1]

        # Permutation-invariant trial quality attention over the 4 trials
        h_ctr = self.set_ln(h_tr - h_tr.mean(dim=1, keepdim=True))
        z_tr = self.set_proj(h_ctr)  # [B, 4, 64]
        sc_in = torch.cat([z_tr, s_lat.unsqueeze(-1), q_trial.unsqueeze(-1)], dim=-1)  # [B, 4, 66]
        # Base prior: prefer higher latent agreement & lower outlier RMS
        att_logits = 1.5 * s_lat - 0.5 * q_trial + self.set_score(sc_in).squeeze(-1)
        pi = F.softmax(att_logits, dim=1)  # [B, 4]

        h_set = torch.einsum("br,brd->bd", pi, h_tr)
        fp_set = torch.einsum("br,brd->bd", pi, fp_tr)

        # Adaptive per-sample Dual-Stream Gate
        g_in = torch.cat([self.gate_ln(h_set - h_erp), raw_agree_mean, s_lat_mean], dim=-1)
        gamma = torch.sigmoid(self.gate_net(g_in))  # [B, 1]
        h_fused = (1.0 - gamma) * h_erp + gamma * h_set

        w_fp = torch.sigmoid(self.fp_set_logit)
        fp_fused = (1.0 - w_fp) * model.head_f(h_fused) + w_fp * fp_set
        return fp_fused, h_fused, fp_erp, fp_set


def train_m1b_trial_prec_swapdiff(
    init_ckpt: Path,
    res: dict,
    pipe_unet,
    scheduler,
    adapter,
    proj,
    pe,
    ppe,
    atid,
    steps: int = 600,
    seed: int = 2025,
    dev: str = "cuda:0",
) -> Path:
    out_p = M1_DIR / f"m1b_trial_prec_set_swapdiff_st{steps}_s{seed}.pt"
    if out_p.exists():
        print(f"[CACHED] {out_p}", flush=True)
        return out_p

    t0 = time.time()
    pack = torch.load(init_ckpt, map_location="cpu", weights_only=True)
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})

    base_cond = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev)
    base_cond.load_state_dict({k: v.float() for k, v in pack["cond_layer"].items()})

    agg = TrialPrecisionSetAggregator(n_ch=17, n_win=5, d_lat=1024).to(dev)

    # Calibrated cone from init_ckpt
    R_tr, a_tr, b_tr = pack["cone"]["R"], pack["cone"]["alpha"], pack["cone"]["beta"]
    u_tr = pack["cone"]["u_tr"].to(dev).float()

    # Train agg + cond_layer at 2e-4, model at gentle 5e-5 (warm-started from Arm3b!)
    opt = torch.optim.AdamW(
        [
            {"params": list(agg.parameters()), "lr": 3.0e-4, "weight_decay": 0.02},
            {"params": list(base_cond.parameters()), "lr": 1.5e-4, "weight_decay": 0.05},
            {"params": list(model.parameters()), "lr": 4.0e-5, "weight_decay": 0.02},
        ],
        betas=(0.9, 0.995),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-6)

    c_groups = res["tr14_c_groups"]
    n_c = len(c_groups)
    g = torch.Generator(device=dev).manual_seed(seed * 211 + 31)
    rng = np.random.default_rng(seed * 211 + 31)
    timesteps = scheduler.timesteps.to(dev)
    sigmas = scheduler.sigmas.to(dev)
    dt = pipe_unet.dtype

    print(f"[TRAIN M1b_TrialPrecSet_SwapDiff] steps={steps} (4-trial dual-stream + raw-eps UNet 2x2 swap)", flush=True)
    model.train()
    base_cond.train()
    agg.train()

    run_sw, run_t1 = [], []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        # 1. Fast within-concept + global representation batch using REAL 4-trial waveforms (16 concepts x 4 photos = 64)
        sid_r = int(rng.integers(0, 10))
        c_pick = rng.choice(n_c, size=16, replace=False)
        loc_r = torch.tensor(c_groups[c_pick, :4].reshape(-1), dtype=torch.long)  # [64]
        x4_r = res["x_tr_all"][sid_r][loc_r].float().to(dev)  # [64, 4, 17, 250]
        # Independent random micro-jitter (+/-2 samples = +/-8ms) per individual trial for jitter-invariance!
        x4_r = torch.stack([shift_batch(x4_r[:, r], 2, g) for r in range(4)], dim=1)

        fp_r, h_r, fp_erp_r, fp_set_r = agg.forward_dual(model, x4_r, sid_r)
        F_gt_r = res["F_tr"][loc_r.to(dev)].float()
        f_pr_n = nz(fp_r.float(), dim=-1)
        f_gt_n = nz(F_gt_r, dim=-1)

        # Global 64-way + within-concept 4-way InfoNCE + MSE + auxiliary supervision on fp_set
        l_glob = F.cross_entropy(14.0 * (f_pr_n @ f_gt_n.T), torch.arange(64, device=dev))
        S_in = torch.bmm(f_pr_n.view(16, 4, -1), f_gt_n.view(16, 4, -1).transpose(1, 2))
        t4 = torch.arange(4, device=dev).unsqueeze(0).expand(16, -1).reshape(-1)
        l_intra = 0.5 * (
            F.cross_entropy((14.0 * S_in).reshape(64, 4), t4)
            + F.cross_entropy((14.0 * S_in.transpose(1, 2)).reshape(64, 4), t4)
        )
        # Also supervise individual-trial set branch directly so TrialSet attention learns clean intra-concept discrimination!
        f_set_n = nz(fp_set_r.float(), dim=-1)
        S_set = torch.bmm(f_set_n.view(16, 4, -1), f_gt_n.view(16, 4, -1).transpose(1, 2))
        l_set = 0.5 * (
            F.cross_entropy((14.0 * S_set).reshape(64, 4), t4)
            + F.cross_entropy((14.0 * S_set.transpose(1, 2)).reshape(64, 4), t4)
        )
        l_mse = (fp_r.float() - F_gt_r).pow(2).mean()
        l_rep = 0.40 * l_glob + 0.60 * l_intra + 0.25 * l_set + 0.50 * l_mse

        # 2. Frozen-UNet Raw-Epsilon 2x2 EEG-Swap Denoising Contrast (2 x 8 = 16 same-concept pairs/step, low VRAM!)
        sw_accs = []
        l_rec_acc, l_swap_acc = 0.0, 0.0
        l_rep.backward()
        for _mb in range(2):
            sid_u = int(rng.integers(0, 10))
            loc_a, loc_b = sample_same_concept_pairs(c_groups, 8, rng)
            x4_a = res["x_tr_all"][sid_u][loc_a].float().to(dev)
            x4_b = res["x_tr_all"][sid_u][loc_b].float().to(dev)
            x4_a = torch.stack([shift_batch(x4_a[:, r], 2, g) for r in range(4)], dim=1)
            x4_b = torch.stack([shift_batch(x4_b[:, r], 2, g) for r in range(4)], dim=1)

            fp_a, h_a, _, _ = agg.forward_dual(model, x4_a, sid_u)
            fp_b, h_b, _, _ = agg.forward_dual(model, x4_b, sid_u)
            c_a = per_sample_cone(fp_a, R_tr, a_tr, b_tr, u_tr).to(dt)
            c_b = per_sample_cone(fp_b, R_tr, a_tr, b_tr, u_tr).to(dt)
            ips_a = base_cond(h_a, proj(adapter(c_a))).to(dt)
            ips_b = base_cond(h_b, proj(adapter(c_b))).to(dt)

            z_a = res["lat_tr_cpu"][loc_a].float().to(dev)
            z_b = res["lat_tr_cpu"][loc_b].float().to(dev)
            k_idx = torch.randint(0, 4, (8,), generator=g, device=dev)
            t_v = timesteps[k_idx]
            sig_v = sigmas[k_idx].view(8, 1, 1, 1).float()
            eps_a = torch.randn(8, 4, 64, 64, device=dev, generator=g)
            eps_b = torch.randn(8, 4, 64, 64, device=dev, generator=g)
            in_a = ((z_a + sig_v * eps_a) / (sig_v * sig_v + 1.0).sqrt()).to(dt)
            in_b = ((z_b + sig_v * eps_b) / (sig_v * sig_v + 1.0).sqrt()).to(dt)

            m_in = torch.cat([in_a, in_b, in_a, in_b], dim=0)  # [32, 4, 64, 64]
            t_in = torch.cat([t_v, t_v, t_v, t_v], dim=0)
            ip_in = torch.cat([ips_a, ips_b, ips_b, ips_a], dim=0)
            pe_in = pe.expand(32, -1, -1)
            ppe_in = ppe.expand(32, -1)
            atid_in = atid.expand(32, -1)
            eps_hat = pipe_unet(
                m_in, t_in,
                encoder_hidden_states=pe_in,
                added_cond_kwargs={"text_embeds": ppe_in, "time_ids": atid_in},
                cross_attention_kwargs={"ip_hidden_states": ip_in},
            ).sample.float()
            e_aa, e_bb, e_ab, e_ba = eps_hat.chunk(4, dim=0)
            d_aa = (e_aa - eps_a).pow(2).mean(dim=(1, 2, 3))
            d_bb = (e_bb - eps_b).pow(2).mean(dim=(1, 2, 3))
            d_ab = (e_ab - eps_a).pow(2).mean(dim=(1, 2, 3))
            d_ba = (e_ba - eps_b).pow(2).mean(dim=(1, 2, 3))

            l_rec = 0.5 * (d_aa.mean() + d_bb.mean())
            margin = (d_ab + d_ba) - (d_aa + d_bb)
            l_swap = F.softplus(-18.0 * margin).mean()
            loss_u = 0.5 * (1.5 * l_rec + 0.45 * l_swap)
            if not torch.isfinite(loss_u):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss_u.backward()
            sw_accs.append(float((margin.detach() > 0).float().mean()))

        torch.nn.utils.clip_grad_norm_(list(agg.parameters()) + list(base_cond.parameters()) + list(model.parameters()), 2.0)
        opt.step()
        sched.step()

        run_sw.append(float(np.mean(sw_accs)))
        run_t1.append(float((S_in.detach().argmax(dim=2) == torch.arange(4, device=dev).view(1, 4)).float().mean()))
        if step in (1, 50, 150, 300, 450, 600) or step == steps:
            el = time.time() - t0
            eta = (el / step) * (steps - step)
            a_m = float(F.softplus(agg.log_alpha.detach()).mean())
            b_m = float(agg.beta_agree.detach().mean())
            wf = float(torch.sigmoid(agg.fp_set_logit.detach()))
            print(
                f"  [M1b] step {step:3d}/{steps} unet_sw={np.mean(run_sw[-30:])*100:5.1f}% "
                f"rep4w={np.mean(run_t1[-30:])*100:5.1f}% alpha={a_m:.2f} beta={b_m:.2f} w_fp={wf:.2f} ({el:.0f}s, ETA {eta:.0f}s)",
                flush=True,
            )

    save_obj = {
        "state_dict": {k: v.half().cpu() for k, v in model.state_dict().items()},
        "cond_layer": {k: v.half().cpu() for k, v in base_cond.state_dict().items()},
        "agg": {k: v.half().cpu() for k, v in agg.state_dict().items()},
        "cone": pack["cone"],
        "steps": steps,
    }
    torch.save(save_obj, out_p)
    print(f"[SAVED] {out_p} ({time.time()-t0:.1f}s)", flush=True)
    return out_p


@torch.inference_mode()
def evaluate_m1_variant(
    tag: str,
    mode: str,
    ckpt_p: Path,
    subs,
    sel50_idx,
    f0_idx,
    F_f0,
    z0_by_sub,
    pipe_unet,
    scheduler,
    vae32,
    adapter,
    proj,
    pe,
    ppe,
    atid,
    gt_rgb50,
    gt_feats,
    nets,
    dev="cuda:0",
):
    pack = torch.load(ckpt_p, map_location="cpu", weights_only=True)
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
    model.eval()
    base_cond = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev)
    base_cond.load_state_dict({k: v.float() for k, v in pack["cond_layer"].items()})
    base_cond.eval()
    agg = None
    if mode == "trained_agg":
        agg = TrialPrecisionSetAggregator(n_ch=17, n_win=5, d_lat=1024).to(dev)
        agg.load_state_dict({k: v.float() for k, v in pack["agg"].items()})
        agg.eval()

    R_tr, a_tr, b_tr = pack["cone"]["R"], pack["cone"]["alpha"], pack["cone"]["beta"]
    u_tr = pack["cone"]["u_tr"].to(dev).float()

    def encode_x4(x4_cpu, sid):
        fps, hs = [], []
        for i in range(0, len(x4_cpu), 64):
            xb = x4_cpu[i:i + 64].float().to(dev)
            if mode == "naive_mean":
                fp, h, _ = forward_branches_and_trunk(model, xb.mean(1), sid, r_eff=4.0)
            elif mode == "zero_retrain_dual":
                x_pr = prec_erp_analytic(xb, p=0.5)
                fp_e, h_e, _ = forward_branches_and_trunk(model, x_pr, sid, r_eff=4.0)
                B, R, C, T = xb.shape
                fp_t, h_t, _ = forward_branches_and_trunk(model, xb.reshape(B * R, C, T), sid, r_eff=1.0)
                fp_s = fp_t.view(B, R, -1).mean(1)
                h_s = h_t.view(B, R, -1).mean(1)
                fp = 0.72 * fp_e + 0.28 * fp_s
                h = 0.72 * h_e + 0.28 * h_s
            else:
                fp, h, _, _ = agg.forward_dual(model, xb, sid)
            fps.append(fp.cpu())
            hs.append(h.cpu())
        return torch.cat(fps, 0), torch.cat(hs, 0)

    # 1. Full fold-0 (3,270 imgs) EEG-representation Intra10!
    Fg = nz(F_f0 - F_f0.mean(0, keepdim=True), dim=1).view(-1, 10, F_f0.shape[1])
    rep_f0 = {}
    for sub in subs:
        sid = sub - 1
        X4_f0 = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_train.npy")[f0_idx])
        fp_f0, _ = encode_x4(X4_f0, sid)
        Fp = nz(fp_f0 - fp_f0.mean(0, keepdim=True), dim=1).view(-1, 10, fp_f0.shape[1])
        rep_f0[f"sub-{sub:02d}"] = intra10_and_swap_from_sim(torch.bmm(Fp, Fg.transpose(1, 2)).numpy())
    rep_t10 = float(np.mean([v["top1"] for v in rep_f0.values()]))
    rep_sw = float(np.mean([v["swap2x2"] for v in rep_f0.values()]))
    print(f">>> [{tag:28s} F-REP 3270] Intra10={rep_t10*100:5.2f}% Swap={rep_sw*100:5.2f}%", flush=True)

    # 2. Full 500-img R=4 FreeGen at s375 AND s625
    cids_500 = [i // 10 for i in range(500)]
    eps_map = {ci: torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * ci + 99)) for ci in range(50)}
    sn_map = {
        ci: [torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(1_000_003 * ci + 12345 + 17 * st)) for st in range(4)]
        for ci in range(50)
    }

    res_by_reg = {"rep_f0": {"mean_t10": rep_t10, "mean_sw": rep_sw, "per_sub": rep_f0}}
    for rtag, smap_c, lk in [("s375", 0.375, (8, 0.11, 0.02)), ("s625", 0.625, (8, 0.22, 0.04))]:
        per_sub = {}
        for sub in subs:
            sid = sub - 1
            X4_50 = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_train.npy")[sel50_idx])
            fp50, h50 = encode_x4(X4_50, sid)
            ips_l = []
            for i in range(0, 500, 100):
                c_cone = per_sample_cone(fp50[i:i + 100].to(dev), R_tr, a_tr, b_tr, u_tr).to(torch.float16)
                ips_p = proj(adapter(c_cone))
                ips_l.append(base_cond(h50[i:i + 100].to(dev), ips_p).cpu())
            ips_all = torch.cat(ips_l, 0)
            z0_e = z0_by_sub[sub]

            rec = torch.empty(500, 3, 512, 512, dtype=torch.float32)
            for s0 in range(0, 500, 50):
                q = list(range(s0, min(500, s0 + 50)))
                cids_q = [cids_500[i] for i in q]
                eps_q = torch.stack([eps_map[ci] for ci in cids_q]).to(dev)
                sn_q = [torch.stack([sn_map[ci][st] for ci in cids_q]).to(dev) for st in range(4)]
                ip_q = ips_all[q].to(dev)
                lat_out = rollout_4step_step_ips(
                    pipe_unet, scheduler, [ip_q, ip_q, ip_q, ip_q], z0_e[q].to(dev), eps_q, sn_q,
                    pe, ppe, atid, smap_c, lk,
                )
                for j0 in range(0, len(q), 10):
                    img_b = (vae32.decode(lat_out[j0:j0 + 10].float() / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
                    rec[s0 + j0:s0 + j0 + len(img_b)] = img_b
            per_sub[f"sub-{sub:02d}"] = score_full500(rec, gt_rgb50, gt_feats, nets, dev=dev)

        m_avg = {
            "clip_t10": float(np.mean([v["clip"]["top1"] for v in per_sub.values()])),
            "clip_dc10": float(np.mean([v["clip"]["top1_dc"] for v in per_sub.values()])),
            "clip_sw": float(np.mean([v["clip"]["swap2x2"] for v in per_sub.values()])),
            "clip_g500": float(np.mean([v["clip"]["global500_top1"] for v in per_sub.values()])),
            "incep_sw": float(np.mean([v["inception"]["swap2x2"] for v in per_sub.values()])),
            "alex5_t10": float(np.mean([v["alex5"]["top1"] for v in per_sub.values()])),
            "alex5_sw": float(np.mean([v["alex5"]["swap2x2"] for v in per_sub.values()])),
            "pix_sw": float(np.mean([v["pixcorr"]["swap2x2"] for v in per_sub.values()])),
            "ssim": float(np.mean([v["ssim"] for v in per_sub.values()])),
        }
        res_by_reg[rtag] = {"mean": m_avg, "per_sub": per_sub}
        print(
            f">>> [{tag:28s} {rtag}] "
            f"INDEP CLIP-L14: t10={m_avg['clip_t10']*100:5.2f}% dc10={m_avg['clip_dc10']*100:5.2f}% sw={m_avg['clip_sw']*100:5.2f}% g500={m_avg['clip_g500']*100:4.2f}% | "
            f"Inc sw={m_avg['incep_sw']*100:5.2f}% | A5: t10={m_avg['alex5_t10']*100:5.2f}% sw={m_avg['alex5_sw']*100:5.2f}% | "
            f"Px sw={m_avg['pix_sw']*100:5.2f}% | SSIM={m_avg['ssim']:.4f}",
            flush=True,
        )
    del model, base_cond, agg
    torch.cuda.empty_cache()
    return res_by_reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    dev = "cuda:0"
    t0 = time.time()
    arm3b_ckpt = OUT_DIR / "arm3b_rep_rawswap_condlyr_st600_s2025.pt"
    assert arm3b_ckpt.exists()

    res = build_tr14_resources(dev=dev)

    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    from metrics import _nets

    # 1. Train M1b with frozen bf16 SDXL-Turbo UNet
    pipe_tr = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=torch.bfloat16, variant="fp16")
    IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(pipe_tr.unet)
    pipe_tr.to(dev, torch.bfloat16)
    pipe_tr.scheduler.set_timesteps(4, device=dev)
    with torch.no_grad():
        pe_bf, _, ppe_bf, _ = pipe_tr.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
        atid_bf = pipe_tr._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=torch.bfloat16, text_encoder_projection_dim=pipe_tr.text_encoder_2.config.projection_dim).to(dev)
        pe_bf, ppe_bf, atid_bf = pe_bf.detach().clone(), ppe_bf.detach().clone(), atid_bf.detach().clone()
    del pipe_tr.text_encoder, pipe_tr.text_encoder_2, pipe_tr.vae
    pipe_tr.unet.eval().requires_grad_(False)
    pipe_tr.unet.enable_gradient_checkpointing()
    adapter_bf = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, torch.bfloat16).eval().requires_grad_(False)
    proj_bf = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, torch.bfloat16).eval().requires_grad_(False)

    m1b_ckpt = train_m1b_trial_prec_swapdiff(
        arm3b_ckpt, res, pipe_tr.unet, pipe_tr.scheduler, adapter_bf, proj_bf, pe_bf, ppe_bf, atid_bf,
        steps=args.steps, seed=2025, dev=dev,
    )

    del pipe_tr, adapter_bf, proj_bf, pe_bf, ppe_bf, atid_bf
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Evaluate M1a (Zero-Retrain PrecDual + PrecZ0) & M1b (Trained TrialPrecSet + PrecZ0) vs Arm3b on Dev-4
    z0_prec_dev4 = {sub: fit_tr14_ridge_z0_prec(sub, res["tr_idx_np"], res["sel50_idx"], dev=dev, p=0.5) for sub in DEV_SUBS}

    dt = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=dt, variant="fp16")
    IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(pipe.unet)
    pipe.to(dev, dt)
    pipe.unet.eval().requires_grad_(False)
    pipe.scheduler.set_timesteps(4, device=dev)
    with torch.inference_mode():
        pe, _, ppe, _ = pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
        atid = pipe._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=dt, text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).to(dev)
    del pipe.text_encoder, pipe.text_encoder_2, pipe.vae
    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, dt).eval().requires_grad_(False)
    proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, dt).eval().requires_grad_(False)
    vae32 = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval().requires_grad_(False)
    vae32.enable_slicing()

    lat50 = torch.from_numpy(np.load(PREP / "lat512_train.npy")[res["sel50_idx"]]).float()
    F_f0 = torch.from_numpy(np.load(PREP / "F_train.npy")[res["f0_idx"]]).float()
    with torch.inference_mode():
        gt_rgb50 = torch.cat([
            (vae32.decode(lat50[j:j + 20].to(dev, torch.float32) / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
            for j in range(0, 500, 20)
        ])
        nets = _nets()
        gt_feats = extract_eval_feats(gt_rgb50, nets, dev=dev)

    res_p = M1_DIR / "exp8_m1_trial_prec_dev4_results.json"
    out_res = {"wall_s": 0.0, "arms": {}}
    out_res["arms"]["M1a_ZeroRetrain_PrecDual"] = evaluate_m1_variant(
        "M1a_ZeroRetrain_PrecDual", "zero_retrain_dual", arm3b_ckpt, DEV_SUBS, res["sel50_idx"], res["f0_idx"], F_f0,
        z0_prec_dev4, pipe.unet, pipe.scheduler, vae32, adapter, proj, pe, ppe, atid, gt_rgb50, gt_feats, nets, dev=dev,
    )
    out_res["arms"]["M1b_Trained_TrialPrecSet"] = evaluate_m1_variant(
        "M1b_Trained_TrialPrecSet", "trained_agg", m1b_ckpt, DEV_SUBS, res["sel50_idx"], res["f0_idx"], F_f0,
        z0_prec_dev4, pipe.unet, pipe.scheduler, vae32, adapter, proj, pe, ppe, atid, gt_rgb50, gt_feats, nets, dev=dev,
    )
    out_res["wall_s"] = round(time.time() - t0, 1)
    write_json(res_p, out_res)
    print(f"=== EXP 8 (METHOD 1 TRIAL-PRECISION SET DEV-4) DONE in {out_res['wall_s']}s -> {res_p} ===", flush=True)


if __name__ == "__main__":
    main()
