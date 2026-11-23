"""Step 2b (exp6b): Clean Ablation Ladder for Full-Chain 4-Step DRaFT-Intra4.

Fixes the step-1 zero-init torch.norm(0) NaN bug from exp6 via safe_norm = sqrt(sum(x^2) + 1e-12),
removes the blur-inducing uncontrasted l_sim term (pure symmetric 4-way within-concept competition l_ce ONLY),
and tests a clean 3-arm causal ablation ladder on top of Arm3b / D1_SimOnly / D2_Compete4:

  - d3_steproute_z0_purece : Frozen EEG encoder + Neuro-Temporal Step Routing (4 UNet steps x 3 EEG horizons)
                             + PCA-256 start-latent residual Delta_z0(E), pure 4-way competition (w_sim=0).
  - d4_unfreeze_enc_condz0 : Exact same as d3 + unfreezing cross-subject shared EEG trunk (block + head_f +
                             horizon_logits, lr=2.5e-5) regularized by fast tr14 representation anchor.
  - d5_unfreeze_dualca     : Exact same as d4 + Zero-Init SDXL Main Cross-Attention Soft Prompt (Delta_pe on
                             4 tokens of encoder_hidden_states [77, 2048]) & AdaGN residual (Delta_ppe on
                             text_embeds [1280]), unlocking SDXL's idle main text channels at ZERO extra NFE!

Zero Evaluator Leakage:
  Training scorer uses ONLY Brain-HIVE LAION OpenCLIP-H14/B32 + multi-scale (16x16 + 32x32) VAE latent.
  Evaluation uses strictly independent OpenAI CLIP ViT-L/14 (primary: R=4 FreeGen Intra10 Top-1 & g500),
  Inception-v3, AlexNet-5, PixCorr, and SSIM (plus transparent separate reporting of train-H14).
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
from exp0_intra10_swap_gate import ALL_SUBS, DEV_SUBS, OUT_DIR, fit_tr14_ridge_z0, intra10_and_swap_from_sim
from exp1_swapdiff_3arm import ResidualIPCondLayer, build_tr14_resources, per_sample_cone
from exp5_z0_2x2_diag import extract_eval_feats, score_full500
from exp6_draft_intra4 import DRAFT_DIR, forward_branches_and_trunk, precompute_or_load_tr14_z0_oof
from o11_tri_stream_bigg import _lp_op
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder
from train_flex import shift_batch

from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel

nz = F.normalize


def safe_norm(t: torch.Tensor, dim, keepdim: bool = True, eps: float = 1e-12) -> torch.Tensor:
    """Autograd-safe L2 norm (derivative is 0 at t=0, never NaN)."""
    return (t.pow(2).sum(dim=dim, keepdim=keepdim) + eps).sqrt()


def safe_rms(t: torch.Tensor, dim, keepdim: bool = True, eps: float = 1e-12) -> torch.Tensor:
    """Autograd-safe RMS (derivative is 0 at t=0, never NaN)."""
    return (t.pow(2).mean(dim=dim, keepdim=keepdim) + eps).sqrt()


class StepRouteZ0DualCA(nn.Module):
    """NaN-safe Neuro-Temporal Step Routing (4x3) + PCA-256 Delta_z0 + optional Zero-Init SDXL DualCA (Delta_pe/ppe)."""

    def __init__(
        self,
        base_cond: ResidualIPCondLayer,
        v_pca_256: torch.Tensor,
        z0_rms_ref: float,
        pe_rms_ref: float,
        ppe_norm_ref: float,
        use_dual_ca: bool = False,
        rank: int = 96,
        max_rel_ips: float = 0.25,
        max_rel_z0: float = 0.35,
        max_rel_pe: float = 0.25,
        max_rel_ppe: float = 0.30,
    ):
        super().__init__()
        self.base_cond = base_cond
        self.use_dual_ca = bool(use_dual_ca)
        self.max_rel_ips = max_rel_ips
        self.max_rel_z0 = max_rel_z0
        self.max_rel_pe = max_rel_pe
        self.max_rel_ppe = max_rel_ppe
        self.z0_rms_ref = float(z0_rms_ref)
        self.pe_rms_ref = float(pe_rms_ref)
        self.ppe_norm_ref = float(ppe_norm_ref)
        self.register_buffer("V_pca", v_pca_256.float())  # [16384, 256]

        # 1. Neuro-Temporal Step Routing: 4 UNet steps x 3 EEG horizons (0-460 / 0-640 / 0-900ms)
        self.route_logits = nn.Parameter(torch.zeros(4, 3))
        self.step_ln = nn.LayerNorm(1024)
        self.step_down = nn.ModuleList([nn.Linear(1024, rank) for _ in range(4)])
        self.step_up = nn.ModuleList([nn.Linear(rank, 4 * 2048, bias=False) for _ in range(4)])
        for up in self.step_up:
            nn.init.zeros_(up.weight)

        # 2. Zero-Init PCA-256 Start-Latent Residual Delta_z0(E)
        self.z0_ln = nn.LayerNorm(1024)
        self.z0_down = nn.Linear(1024, 128)
        self.z0_up = nn.Linear(128, v_pca_256.shape[1], bias=False)
        nn.init.zeros_(self.z0_up.weight)

        # 3. Optional Zero-Init SDXL Main Cross-Attention Soft Prompt (tokens 1..4) & AdaGN Residual (text_embeds)
        if self.use_dual_ca:
            self.ca_ln = nn.LayerNorm(1024)
            self.pe_down = nn.Linear(1024, 128)
            self.pe_up = nn.Linear(128, 4 * 2048, bias=False)
            nn.init.zeros_(self.pe_up.weight)
            self.ppe_down = nn.Linear(1024, 128)
            self.ppe_up = nn.Linear(128, 1280, bias=False)
            nn.init.zeros_(self.ppe_up.weight)

    def forward_ips_steps(self, c_ens: torch.Tensor, cs_branches: tuple, ips_proj: torch.Tensor) -> list:
        ips0 = self.base_cond(c_ens, ips_proj)
        br_stack = torch.stack([b.float() for b in cs_branches], dim=1)  # [B, 3, 1024]
        w_route = F.softmax(self.route_logits, dim=-1)  # [4, 3]
        base_n = safe_norm(ips0.float(), dim=(1, 2), keepdim=True, eps=1e-8)

        out_steps = []
        for t_i in range(4):
            h_t = torch.einsum("h,bhd->bd", w_route[t_i], br_stack)
            delta = self.step_up[t_i](F.silu(self.step_down[t_i](self.step_ln(h_t)))).view(-1, 4, 2048)
            del_n = safe_norm(delta, dim=(1, 2), keepdim=True, eps=1e-12)
            sc = torch.minimum(torch.ones_like(del_n), (self.max_rel_ips * base_n) / del_n)
            out_steps.append(ips0 + (delta * sc).to(ips0.dtype))
        return out_steps

    def forward_z0(self, c_ens: torch.Tensor, cs_branches: tuple, z0_ridge: torch.Tensor) -> torch.Tensor:
        h_z = 0.5 * c_ens.float() + 0.5 * cs_branches[0].float()
        dz_coeff = self.z0_up(F.silu(self.z0_down(self.z0_ln(h_z))))  # [B, 256]
        dz = (dz_coeff @ self.V_pca.T).view(-1, 4, 64, 64)
        dz_rms = safe_rms(dz, dim=(1, 2, 3), keepdim=True, eps=1e-12)
        sc = torch.minimum(torch.ones_like(dz_rms), (self.max_rel_z0 * self.z0_rms_ref) / dz_rms)
        return z0_ridge.float() + dz * sc

    def forward_dual_ca(self, c_ens: torch.Tensor, pe_base: torch.Tensor, ppe_base: torch.Tensor):
        B = c_ens.shape[0]
        dt = pe_base.dtype
        pe_b = pe_base.expand(B, -1, -1)
        ppe_b = ppe_base.expand(B, -1)
        if not self.use_dual_ca:
            return pe_b, ppe_b

        h = self.ca_ln(c_ens.float())
        d_pe = self.pe_up(F.silu(self.pe_down(h))).view(B, 4, 2048)
        d_pe_rms = safe_rms(d_pe, dim=(1, 2), keepdim=True, eps=1e-12)
        sc_pe = torch.minimum(torch.ones_like(d_pe_rms), (self.max_rel_pe * self.pe_rms_ref) / d_pe_rms)
        d_pe_b = (d_pe * sc_pe).to(dt)
        # Inject residual into tokens 1..4 without touching BOS (0) or padding (5..76)
        pe_out = torch.cat([pe_b[:, :1, :], pe_b[:, 1:5, :] + d_pe_b, pe_b[:, 5:, :]], dim=1)

        d_ppe = self.ppe_up(F.silu(self.ppe_down(h)))
        d_ppe_n = safe_norm(d_ppe, dim=-1, keepdim=True, eps=1e-12)
        sc_ppe = torch.minimum(torch.ones_like(d_ppe_n), (self.max_rel_ppe * self.ppe_norm_ref) / d_ppe_n)
        ppe_out = ppe_b + (d_ppe * sc_ppe).to(dt)
        return pe_out, ppe_out


def rollout_4step_dualca(
    unet,
    scheduler,
    ips_steps: list,
    z0: torch.Tensor,
    eps_init: torch.Tensor,
    step_noises: list,
    pe_b: torch.Tensor,
    ppe_b: torch.Tensor,
    atid: torch.Tensor,
    smap_c: float,
    low_keep,
) -> torch.Tensor:
    """Exact 4-step differentiable Euler-Ancestral rollout supporting per-step ips_steps + batched pe_b/ppe_b."""
    B = z0.shape[0]
    dt = unet.dtype
    steps = 4
    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas
    hold_val = 2.0 * smap_c if smap_c > 0.0 else 0.0
    sig0 = float(sigmas[0])
    lt = (z0.float() + sig0 * eps_init.float()) if smap_c > 0.0 else (sig0 * eps_init.float())

    pe_b = pe_b.to(dt)
    ppe_b = ppe_b.to(dt)
    atid_b = atid.expand(B, -1).to(dt)

    for i_step in range(steps):
        sigma = float(sigmas[i_step])
        sigma_next = float(sigmas[i_step + 1])
        t_cur = timesteps[i_step].expand(B)
        ips_b = ips_steps[i_step].to(dt)

        model_in = (lt / math.sqrt(sigma * sigma + 1.0)).to(dt)
        eps_pred = unet(
            model_in,
            t_cur,
            encoder_hidden_states=pe_b,
            added_cond_kwargs={"text_embeds": ppe_b, "time_ids": atid_b},
            cross_attention_kwargs={"ip_hidden_states": ips_b},
        ).sample.float()

        sigma_up = min(sigma_next, math.sqrt((sigma_next ** 2) * (sigma ** 2 - sigma_next ** 2) / (sigma ** 2))) if sigma_next > 0 else 0.0
        sigma_down = math.sqrt(max(0.0, sigma_next ** 2 - sigma_up ** 2))
        lt = lt + eps_pred * (sigma_down - sigma)
        if sigma_up > 0.0:
            lt = lt + sigma_up * step_noises[i_step].float()

        if smap_c > 0.0:
            a = max(0.0, min(1.0, hold_val - i_step))
            if a > 0.0:
                lt = a * (z0.float() + sigma_next * eps_init.float()) + (1.0 - a) * lt
            if low_keep is not None and i_step in (steps - 2, steps - 1):
                f_lp, w2, w3 = low_keep
                g_t = w2 if i_step == steps - 2 else w3
                if g_t > 0.0:
                    lt = lt + g_t * _lp_op((z0.float() + sigma_next * eps_init.float()) - lt, f_lp)
    return lt


def train_v2_arm(
    mode: str,
    init_ckpt: Path,
    res: dict,
    z0_oof_pack: dict,
    pipe_unet,
    scheduler,
    vae_bf,
    adapter_bf,
    proj_bf,
    h14_vis,
    b32_vis,
    b32_fix,
    pe_bf,
    ppe_bf,
    atid_bf,
    steps: int = 150,
    seed: int = 2025,
    dev: str = "cuda:0",
    dry_run: bool = False,
) -> Path:
    assert mode in ("d3_steproute_z0_purece", "d4_unfreeze_enc_condz0", "d5_unfreeze_dualca")
    out_p = DRAFT_DIR / f"{mode}_st{steps}_s{seed}.pt"
    if out_p.exists() and not dry_run:
        print(f"[CACHED] {out_p}", flush=True)
        return out_p

    unfreeze_enc = mode in ("d4_unfreeze_enc_condz0", "d5_unfreeze_dualca")
    use_dual_ca = mode == "d5_unfreeze_dualca"

    t0 = time.time()
    pack = torch.load(init_ckpt, map_location="cpu", weights_only=True)
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
    model.eval()
    model.requires_grad_(False)

    enc_params = []
    if unfreeze_enc:
        # Unfreeze ONLY cross-subject shared trunk + head_f + horizon_logits (keep subject stems frozen to prevent sub overfitting!)
        for mod in (model.block, model.head_f):
            for p in mod.parameters():
                p.requires_grad_(True)
                enc_params.append(p)
        model.horizon_logits.requires_grad_(True)
        enc_params.append(model.horizon_logits)

        # Frozen reference encoder for representation stability anchor
        model_ref = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
        model_ref.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
        model_ref.eval().requires_grad_(False)
    else:
        model_ref = None

    base_cond = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev)
    base_cond.load_state_dict({k: v.float() for k, v in pack["cond_layer"].items()})

    cond_ref = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev)
    cond_ref.load_state_dict(base_cond.state_dict())
    cond_ref.eval().requires_grad_(False)

    V_pca = z0_oof_pack["V_pca"].to(dev)
    my_z0 = z0_oof_pack["my"].to(dev)
    Z_oof = z0_oof_pack["Z_oof"]
    pe_rms_ref = float(pe_bf.float().pow(2).mean().sqrt().clamp_min(1e-3))
    ppe_norm_ref = float(ppe_bf.float().norm().clamp_min(1e-3))

    head_mod = StepRouteZ0DualCA(
        base_cond,
        V_pca,
        z0_oof_pack["z0_rms_ref"],
        pe_rms_ref,
        ppe_norm_ref,
        use_dual_ca=use_dual_ca,
    ).to(dev)

    head_params = list(head_mod.parameters())
    param_groups = [{"params": head_params, "lr": 1.5e-4, "weight_decay": 0.05}]
    if unfreeze_enc:
        param_groups.append({"params": enc_params, "lr": 2.5e-5, "weight_decay": 0.02})
    all_train_params = head_params + enc_params

    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.995))

    def lr_lambda(st_i):
        warm = min(10, max(1, steps // 10))
        if st_i < warm:
            return float(st_i + 1) / float(warm)
        prog = float(st_i - warm) / float(max(1, steps - warm))
        return 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    R_tr, a_tr, b_tr = pack["cone"]["R"], pack["cone"]["alpha"], pack["cone"]["beta"]
    u_tr = pack["cone"]["u_tr"].to(dev).float()

    smap_c = 0.375
    low_keep = (8, 0.11, 0.02)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)

    c_groups = res["tr14_c_groups"]
    n_c = len(c_groups)
    g = torch.Generator(device=dev).manual_seed(seed * 131 + 17)
    rng = np.random.default_rng(seed * 131 + 17)

    n_micro = 4
    tau = 15.0

    print(
        f"[TRAIN {mode}] steps={steps} unfreeze_enc={unfreeze_enc} dual_ca={use_dual_ca} "
        f"pure_ce=True (w_sim=0) s375 | pe_rms={pe_rms_ref:.3f} ppe_norm={ppe_norm_ref:.3f}",
        flush=True,
    )
    head_mod.train()

    run_top1, run_sim, run_ce = [], [], []
    n_run = 2 if dry_run else steps
    for step in range(1, n_run + 1):
        opt.zero_grad(set_to_none=True)
        st_top1, st_sim, st_ce = 0.0, 0.0, 0.0

        for mb in range(n_micro):
            sid = int(rng.integers(0, 10))
            ci = int(rng.integers(0, n_c))
            pick4 = rng.choice(10, size=4, replace=False)
            locs = torch.tensor(c_groups[ci, pick4], dtype=torch.long)
            locs_dev = locs.to(dev)

            x4 = res["x_tr_all"][sid][locs].float().mean(1).to(dev)
            x4 = shift_batch(x4, 2, g)

            with torch.no_grad():
                z0_oof = (Z_oof[sid, locs].float().to(dev) @ V_pca.T + my_z0).view(4, 4, 64, 64)
                z0_gt = res["lat_tr_cpu"][locs].float().to(dev)
                h14_gt = nz(res["H14_tr"][locs_dev].float(), dim=-1)
                b32_gt = nz(res["B32_tr"][locs_dev].float(), dim=-1)
                z0_gt_lp16 = nz(F.avg_pool2d(z0_gt, 4).flatten(1), dim=-1)
                z0_gt_lp32 = nz(F.avg_pool2d(z0_gt, 2).flatten(1), dim=-1)

            if unfreeze_enc:
                # Keep dropout off (model.eval()) for clean low-variance 4-step gradients while updating shared trunk!
                fp, c_ens, cs_br = forward_branches_and_trunk(model, x4, sid, r_eff=4.0)
                c_cone = per_sample_cone(fp, R_tr, a_tr, b_tr, u_tr).to(torch.bfloat16)
                ips_proj = proj_bf(adapter_bf(c_cone))
                with torch.no_grad():
                    fp_r, c_r, _ = forward_branches_and_trunk(model_ref, x4, sid, r_eff=4.0)
                    ips_proj_r = proj_bf(adapter_bf(per_sample_cone(fp_r, R_tr, a_tr, b_tr, u_tr).to(torch.bfloat16)))
                    ips_ref = cond_ref(c_r, ips_proj_r).detach().float()
            else:
                with torch.no_grad():
                    fp, c_ens, cs_br = forward_branches_and_trunk(model, x4, sid, r_eff=4.0)
                    c_cone = per_sample_cone(fp, R_tr, a_tr, b_tr, u_tr).to(torch.bfloat16)
                    ips_proj = proj_bf(adapter_bf(c_cone))
                    ips_ref = cond_ref(c_ens, ips_proj).detach().float()

            eps_sh = torch.randn(1, 4, 64, 64, device=dev, generator=g).expand(4, -1, -1, -1)
            sn_sh = [torch.randn(1, 4, 64, 64, device=dev, generator=g).expand(4, -1, -1, -1) for _ in range(4)]

            ips_steps = head_mod.forward_ips_steps(c_ens, cs_br, ips_proj)
            z0_use = head_mod.forward_z0(c_ens, cs_br, z0_oof)
            pe_use, ppe_use = head_mod.forward_dual_ca(c_ens, pe_bf, ppe_bf)

            l_reg = (
                0.05 * sum((ip.float() - ips_ref).pow(2).mean() for ip in ips_steps) / 4.0
                + 0.05 * (z0_use - z0_oof).pow(2).mean()
            )
            if use_dual_ca:
                l_reg = l_reg + 0.02 * (pe_use[:, 1:5, :].float() - pe_bf[:, 1:5, :].float()).pow(2).mean()
                l_reg = l_reg + 0.02 * (ppe_use.float() - ppe_bf.float()).pow(2).mean()

            lat_out = rollout_4step_dualca(
                pipe_unet, scheduler, ips_steps, z0_use, eps_sh, sn_sh,
                pe_use, ppe_use, atid_bf, smap_c, low_keep,
            )

            imgs = []
            for j0 in range(0, 4, 2):
                im = (vae_bf.decode((lat_out[j0:j0 + 2] / vae_bf.config.scaling_factor).to(torch.bfloat16)).sample.float() / 2.0 + 0.5).clamp(0.0, 1.0)
                imgs.append(im)
            img_gen = torch.cat(imgs, dim=0)

            x224 = ((F.interpolate(img_gen, size=224, mode="bilinear", align_corners=False) - clip_mean) / clip_std).to(torch.bfloat16)
            e_h14 = nz(h14_vis(pixel_values=x224).image_embeds.float(), dim=-1)
            e_b32 = nz(b32_vis(pixel_values=x224).image_embeds.float() @ b32_fix, dim=-1)
            z_out_lp16 = nz(F.avg_pool2d(lat_out.float(), 4).flatten(1), dim=-1)
            z_out_lp32 = nz(F.avg_pool2d(lat_out.float(), 2).flatten(1), dim=-1)

            S_4 = (
                0.55 * (e_h14 @ h14_gt.T)
                + 0.25 * (e_b32 @ b32_gt.T)
                + 0.10 * (z_out_lp16 @ z0_gt_lp16.T)
                + 0.10 * (z_out_lp32 @ z0_gt_lp32.T)
            )
            tgt = torch.arange(4, device=dev)
            l_ce = 0.5 * (F.cross_entropy(tau * S_4, tgt) + F.cross_entropy(tau * S_4.T, tgt))

            # Pure 4-way within-concept competition (ZERO uncontrasted l_sim blur pull!)
            loss_mb = (1.0 * l_ce + l_reg) / float(n_micro)
            if not torch.isfinite(loss_mb):
                raise RuntimeError(f"[{mode}] Non-finite loss at step {step} mb {mb}: {loss_mb.item()}")
            loss_mb.backward()

            st_top1 += float((S_4.detach().argmax(dim=1) == tgt).float().mean()) / n_micro
            st_sim += float(S_4.detach().diag().mean()) / n_micro
            st_ce += float(l_ce.detach()) / n_micro

        # Fast tr14 representation stability anchor when unfreezing the shared EEG trunk (prevents forgetting!)
        if unfreeze_enc:
            sid_a = int(rng.integers(0, 10))
            c_pick = rng.choice(n_c, size=8, replace=False)
            # 8 concepts x 4 photos = 32 EEG samples: keep global + intra-concept F alignment anchored!
            loc_a = torch.tensor(c_groups[c_pick, :4].reshape(-1), dtype=torch.long)
            xa = res["x_tr_all"][sid_a][loc_a].float().mean(1).to(dev)
            fp_a, c_a, _ = forward_branches_and_trunk(model, xa, sid_a, r_eff=4.0)
            with torch.no_grad():
                fp_ar, c_ar, _ = forward_branches_and_trunk(model_ref, xa, sid_a, r_eff=4.0)
                f_gt_a = nz(res["F_tr"][loc_a.to(dev)].float(), dim=-1)
            f_pr_a = nz(fp_a.float(), dim=-1)
            # Global 32-way InfoNCE + trust-region to Arm3b representation
            l_anc = 0.25 * F.cross_entropy(14.0 * (f_pr_a @ f_gt_a.T), torch.arange(32, device=dev))
            l_anc = l_anc + 0.50 * (fp_a.float() - fp_ar.float()).pow(2).mean() + 0.50 * (c_a.float() - c_ar.float()).pow(2).mean()
            l_anc.backward()

        # Verify finite grads on EVERY parameter before stepping
        bad = [n for n, p in head_mod.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            raise RuntimeError(f"[{mode}] Non-finite grads at step {step}: {bad}")

        if step == 1:
            g_head = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in head_params if p.grad is not None))
            g_z0 = float(head_mod.z0_up.weight.grad.float().norm()) if head_mod.z0_up.weight.grad is not None else 0.0
            g_st0 = float(head_mod.step_up[0].weight.grad.float().norm()) if head_mod.step_up[0].weight.grad is not None else 0.0
            g_enc = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in enc_params if p.grad is not None)) if enc_params else 0.0
            g_pe = float(head_mod.pe_up.weight.grad.float().norm()) if use_dual_ca and head_mod.pe_up.weight.grad is not None else 0.0
            g_ppe = float(head_mod.ppe_up.weight.grad.float().norm()) if use_dual_ca and head_mod.ppe_up.weight.grad is not None else 0.0
            print(
                f"  [{mode} STEP-1 DIAG] g_head={g_head:.4f} g_st0={g_st0:.4f} g_z0={g_z0:.4f} "
                f"g_enc={g_enc:.4f} g_pe={g_pe:.4f} g_ppe={g_ppe:.4f} mem_MB={torch.cuda.max_memory_allocated()//(1024*1024)}",
                flush=True,
            )

        torch.nn.utils.clip_grad_norm_(all_train_params, 2.0)
        opt.step()
        sched.step()

        run_top1.append(st_top1)
        run_sim.append(st_sim)
        run_ce.append(st_ce)
        if step in (1, 10, 30, 60, 90, 120, 150) or step == n_run:
            el = time.time() - t0
            eta = (el / step) * (n_run - step)
            wr = F.softmax(head_mod.route_logits.detach(), dim=-1).cpu().numpy()
            zn = float(head_mod.z0_up.weight.detach().float().norm())
            print(
                f"  [{mode}] step {step:3d}/{n_run} 4w_t1={np.mean(run_top1[-20:])*100:.1f}% "
                f"sim={np.mean(run_sim[-20:]):.4f} ce={np.mean(run_ce[-20:]):.3f} | "
                f"route(t0/t3)={wr[0].round(2).tolist()}/{wr[3].round(2).tolist()} z0n={zn:.3f} ({el:.0f}s, ETA {eta:.0f}s)",
                flush=True,
            )

    if dry_run:
        print(f"[DRY-RUN OK] {mode}", flush=True)
        return out_p

    save_obj = {
        "backbone_ckpt": str(init_ckpt),
        # Save only unfrozen shared trunk params if unfreeze_enc else None (tiny checkpoint!)
        "enc_shared": {
            "block": {k: v.half().cpu() for k, v in model.block.state_dict().items()},
            "head_f": {k: v.half().cpu() for k, v in model.head_f.state_dict().items()},
            "horizon_logits": model.horizon_logits.detach().half().cpu(),
        } if unfreeze_enc else None,
        "head_mod": {k: v.half().cpu() for k, v in head_mod.state_dict().items()},
        "use_dual_ca": use_dual_ca,
        "pe_rms_ref": pe_rms_ref,
        "ppe_norm_ref": ppe_norm_ref,
        "cone": pack["cone"],
        "mode": mode,
        "steps": steps,
    }
    torch.save(save_obj, out_p)
    print(f"[SAVED] {out_p} ({time.time()-t0:.1f}s)", flush=True)
    return out_p


@torch.inference_mode()
def evaluate_v2_arm(
    arm_name: str,
    ckpt_p: Path,
    subs,
    sel50_idx,
    z0_by_sub,
    z0_oof_pack,
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
    h14_vis,
    H14_gt50,
    dev="cuda:0",
):
    pack = torch.load(ckpt_p, map_location="cpu", weights_only=True)
    sd_backbone = torch.load(pack["backbone_ckpt"], map_location="cpu", weights_only=True)["state_dict"]
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in sd_backbone.items()})
    if pack.get("enc_shared") is not None:
        es = pack["enc_shared"]
        model.block.load_state_dict({k: v.float() for k, v in es["block"].items()})
        model.head_f.load_state_dict({k: v.float() for k, v in es["head_f"].items()})
        model.horizon_logits.copy_(es["horizon_logits"].float().to(dev))
    model.eval()

    base_cond = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev)
    head_mod = StepRouteZ0DualCA(
        base_cond,
        z0_oof_pack["V_pca"].to(dev),
        z0_oof_pack["z0_rms_ref"],
        pack["pe_rms_ref"],
        pack["ppe_norm_ref"],
        use_dual_ca=pack["use_dual_ca"],
    ).to(dev)
    head_mod.load_state_dict({k: v.float() for k, v in pack["head_mod"].items()})
    head_mod.eval()

    R_tr, a_tr, b_tr = pack["cone"]["R"], pack["cone"]["alpha"], pack["cone"]["beta"]
    u_tr = pack["cone"]["u_tr"].to(dev).float()

    cids_500 = [i // 10 for i in range(500)]
    eps_map = {ci: torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * ci + 99)) for ci in range(50)}
    sn_map = {
        ci: [torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(1_000_003 * ci + 12345 + 17 * st)) for st in range(4)]
        for ci in range(50)
    }
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)

    res_by_reg = {}
    for rtag, smap_c, lk in [("s375", 0.375, (8, 0.11, 0.02)), ("s625", 0.625, (8, 0.22, 0.04))]:
        per_sub = {}
        for sub in subs:
            sid = sub - 1
            X_raw = np.load(PREP / f"sub-{sub:02d}_train.npy")
            xm4 = torch.from_numpy(X_raw[sel50_idx].mean(1)).float()
            del X_raw
            z0_e = z0_by_sub[sub]

            ips_steps_all = [[], [], [], []]
            z0_use_l, pe_use_l, ppe_use_l = [], [], []
            for i in range(0, 500, 100):
                fp, c_ens, cs_br = forward_branches_and_trunk(model, xm4[i:i + 100].to(dev), sid, r_eff=4.0)
                c_cone = per_sample_cone(fp, R_tr, a_tr, b_tr, u_tr).to(torch.float16)
                ips_p = proj(adapter(c_cone))
                st_l = head_mod.forward_ips_steps(c_ens, cs_br, ips_p)
                for k in range(4):
                    ips_steps_all[k].append(st_l[k].cpu())
                z0_use_l.append(head_mod.forward_z0(c_ens, cs_br, z0_e[i:i + 100].to(dev)).cpu())
                pe_u, ppe_u = head_mod.forward_dual_ca(c_ens, pe, ppe)
                pe_use_l.append(pe_u.cpu())
                ppe_use_l.append(ppe_u.cpu())

            ips_steps_cat = [torch.cat(lst, 0) for lst in ips_steps_all]
            z0_final = torch.cat(z0_use_l, 0)
            pe_final = torch.cat(pe_use_l, 0)
            ppe_final = torch.cat(ppe_use_l, 0)

            rec = torch.empty(500, 3, 512, 512, dtype=torch.float32)
            for s0 in range(0, 500, 50):
                q = list(range(s0, min(500, s0 + 50)))
                cids_q = [cids_500[i] for i in q]
                eps_q = torch.stack([eps_map[ci] for ci in cids_q]).to(dev)
                sn_q = [torch.stack([sn_map[ci][st] for ci in cids_q]).to(dev) for st in range(4)]
                ips_q = [ips_steps_cat[st][q].to(dev) for st in range(4)]
                lat_out = rollout_4step_dualca(
                    pipe_unet, scheduler, ips_q, z0_final[q].to(dev), eps_q, sn_q,
                    pe_final[q].to(dev), ppe_final[q].to(dev), atid, smap_c, lk,
                )
                for j0 in range(0, len(q), 10):
                    img_b = (vae32.decode(lat_out[j0:j0 + 10].float() / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
                    rec[s0 + j0:s0 + j0 + len(img_b)] = img_b

            m = score_full500(rec, gt_rgb50, gt_feats, nets, dev=dev)
            h_l = []
            for i in range(0, 500, 50):
                x224 = ((F.interpolate(rec[i:i + 50].to(dev), size=224, mode="bilinear", align_corners=False) - clip_mean) / clip_std).to(torch.float16)
                h_l.append(h14_vis(pixel_values=x224).image_embeds.float().cpu())
            H_rec = torch.cat(h_l, 0)
            Hr = nz(H_rec - H_rec.mean(0, keepdim=True), dim=1).view(50, 10, -1)
            Hg = nz(H14_gt50 - H14_gt50.mean(0, keepdim=True), dim=1).view(50, 10, -1)
            m["h14_train_scorer"] = intra10_and_swap_from_sim(torch.bmm(Hr, Hg.transpose(1, 2)).numpy())
            per_sub[f"sub-{sub:02d}"] = m

        m_avg = {
            "clip_t10": float(np.mean([v["clip"]["top1"] for v in per_sub.values()])),
            "clip_dc10": float(np.mean([v["clip"]["top1_dc"] for v in per_sub.values()])),
            "clip_sw": float(np.mean([v["clip"]["swap2x2"] for v in per_sub.values()])),
            "clip_g500": float(np.mean([v["clip"]["global500_top1"] for v in per_sub.values()])),
            "h14_t10": float(np.mean([v["h14_train_scorer"]["top1"] for v in per_sub.values()])),
            "h14_sw": float(np.mean([v["h14_train_scorer"]["swap2x2"] for v in per_sub.values()])),
            "incep_sw": float(np.mean([v["inception"]["swap2x2"] for v in per_sub.values()])),
            "alex5_t10": float(np.mean([v["alex5"]["top1"] for v in per_sub.values()])),
            "alex5_sw": float(np.mean([v["alex5"]["swap2x2"] for v in per_sub.values()])),
            "pix_sw": float(np.mean([v["pixcorr"]["swap2x2"] for v in per_sub.values()])),
            "ssim": float(np.mean([v["ssim"] for v in per_sub.values()])),
        }
        res_by_reg[rtag] = {"mean": m_avg, "per_sub": per_sub}
        print(
            f">>> [{arm_name:24s} {rtag}] "
            f"INDEP CLIP-L14: t10={m_avg['clip_t10']*100:5.2f}% dc10={m_avg['clip_dc10']*100:5.2f}% sw={m_avg['clip_sw']*100:5.2f}% g500={m_avg['clip_g500']*100:4.2f}% | "
            f"Inc sw={m_avg['incep_sw']*100:5.2f}% | A5: t10={m_avg['alex5_t10']*100:5.2f}% sw={m_avg['alex5_sw']*100:5.2f}% | "
            f"Px sw={m_avg['pix_sw']*100:5.2f}% | SSIM={m_avg['ssim']:.4f} || (train-H14: t10={m_avg['h14_t10']*100:5.2f}% sw={m_avg['h14_sw']*100:5.2f}%)",
            flush=True,
        )
    del model, base_cond, head_mod
    torch.cuda.empty_cache()
    return res_by_reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--modes", default="d3_steproute_z0_purece,d4_unfreeze_enc_condz0,d5_unfreeze_dualca")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    dev = "cuda:0"
    t_all = time.time()
    arm3b_ckpt = OUT_DIR / "arm3b_rep_rawswap_condlyr_st600_s2025.pt"
    assert arm3b_ckpt.exists()

    res = build_tr14_resources(dev=dev)
    z0_oof_pack = precompute_or_load_tr14_z0_oof(res["tr_idx_np"], res["concept_np"], dev=dev)

    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    from transformers import CLIPVisionModelWithProjection
    from metrics import _nets

    pipe_tr = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=torch.bfloat16, variant="fp16")
    IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(pipe_tr.unet)
    pipe_tr.to(dev, torch.bfloat16)
    pipe_tr.scheduler.set_timesteps(4, device=dev)
    with torch.no_grad():
        pe_bf, _, ppe_bf, _ = pipe_tr.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
        atid_bf = pipe_tr._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=torch.bfloat16, text_encoder_projection_dim=pipe_tr.text_encoder_2.config.projection_dim).to(dev)
        pe_bf, ppe_bf, atid_bf = pe_bf.detach().clone(), ppe_bf.detach().clone(), atid_bf.detach().clone()
    del pipe_tr.text_encoder, pipe_tr.text_encoder_2
    pipe_tr.text_encoder = None
    pipe_tr.text_encoder_2 = None
    pipe_tr.unet.eval().requires_grad_(False)
    pipe_tr.unet.enable_gradient_checkpointing()
    vae_bf = pipe_tr.vae.to(dev, torch.bfloat16).eval().requires_grad_(False)
    vae_bf.enable_gradient_checkpointing()

    adapter_bf = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, torch.bfloat16).eval().requires_grad_(False)
    proj_bf = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, torch.bfloat16).eval().requires_grad_(False)

    h14_vis = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / "models" / EMB_KEYS[0]), torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    h14_vis.gradient_checkpointing_enable()
    b32_vis = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / "models" / EMB_KEYS[1]), torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    b32_vis.gradient_checkpointing_enable()
    b32_fix = torch.load(PREP / "b32_fix.pt", weights_only=True).float().to(dev)

    ckpts = {}
    for m in modes:
        ckpts[m] = train_v2_arm(
            m, arm3b_ckpt, res, z0_oof_pack, pipe_tr.unet, pipe_tr.scheduler, vae_bf,
            adapter_bf, proj_bf, h14_vis, b32_vis, b32_fix, pe_bf, ppe_bf, atid_bf,
            steps=args.steps, seed=2025, dev=dev, dry_run=args.dry_run,
        )

    if args.dry_run:
        return

    del pipe_tr, vae_bf, adapter_bf, proj_bf, h14_vis, b32_vis, pe_bf, ppe_bf, atid_bf
    gc.collect()
    torch.cuda.empty_cache()

    z0_by_sub = {sub: fit_tr14_ridge_z0(sub, res["tr_idx_np"], res["sel50_idx"], dev=dev) for sub in DEV_SUBS}

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
    h14_eval = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / "models" / EMB_KEYS[0]), torch_dtype=dt).to(dev).eval().requires_grad_(False)

    lat50 = torch.from_numpy(np.load(PREP / "lat512_train.npy")[res["sel50_idx"]]).float()
    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    H14_gt50 = emb_tr[EMB_KEYS[0]][res["sel50_idx"]].float()
    with torch.inference_mode():
        gt_rgb50 = torch.cat([
            (vae32.decode(lat50[j:j + 20].to(dev, torch.float32) / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
            for j in range(0, 500, 20)
        ])
        nets = _nets()
        gt_feats = extract_eval_feats(gt_rgb50, nets, dev=dev)

    res_p = DRAFT_DIR / "exp6b_draft_v2_dev4_results.json"
    out_res = json.loads(res_p.read_text()) if res_p.exists() else {"arms": {}}
    for arm_name, cp in ckpts.items():
        if arm_name in out_res["arms"]:
            print(f"[CACHED EVAL] {arm_name}", flush=True)
            continue
        out_res["arms"][arm_name] = evaluate_v2_arm(
            arm_name, cp, DEV_SUBS, res["sel50_idx"], z0_by_sub, z0_oof_pack,
            pipe.unet, pipe.scheduler, vae32, adapter, proj, pe, ppe, atid,
            gt_rgb50, gt_feats, nets, h14_eval, H14_gt50, dev=dev,
        )
        out_res["wall_s"] = round(time.time() - t_all, 1)
        write_json(res_p, out_res)

    print(f"=== EXP 6B (DRaFT V2 DEV-4 ABLATION) DONE in {round(time.time()-t_all, 1)}s -> {res_p} ===", flush=True)


if __name__ == "__main__":
    main()
