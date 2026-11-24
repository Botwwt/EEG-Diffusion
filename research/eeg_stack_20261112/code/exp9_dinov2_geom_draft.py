"""Method 2 (exp9): Non-CLIP Self-Supervised DINOv2 Geometric (CLS + 4x4 Spatial Patch) Scorer inside Hybrid SwapDiff + DRaFT-Compete4.

Motivation:
  - Previous DRaFT-Intra4 (exp6/6b/6c) used OpenCLIP-H14/B32 + coarse latent as its 4-way scorer. Contrastive
    text-image CLIP encoders are trained for semantic invariance to camera viewpoint, orientation, and spatial
    framing — the EXACT attributes that distinguish two photos of the same concept!
  - Self-supervised DINOv2-base (facebook/dinov2-base; zero labels, 100% disjoint from EVERY evaluation model:
    OpenAI CLIP ViT-L/14, Inception-v3, AlexNet-5, PixCorr, SSIM!) explicitly preserves fine-grained 3D viewpoint,
    object orientation, part geometry, and 2D spatial layout in its CLS token [768] AND 16x16 spatial patch grid.

Design (Strict Zero Evaluator Leakage):
  1. Precompute & cache exact tr14 DINOv2 targets (CLS [13225, 768] + 4x4 pooled spatial grid [13225, 16, 768] fp16)
     by decoding tr14 VAE latents once (~60-90s).
  2. During 4-step DRaFT-Compete4 (s375, genuine 4-fold OOF z0_oof, shared noise seed across 4 same-concept photos),
     compute the 4x4 generated-vs-GT geometric + semantic matrix:
       S_4 = 0.35 * S_dino_cls + 0.35 * S_dino_patch4x4 + 0.20 * S_h14 + 0.10 * S_b32
     where S_dino_patch4x4 matches each of the 16 spatial 4x4 cells (u, v) at identical locations — directly
     supervising object pose, viewpoint, scale, and quadrant layout through all 4 SDXL-Turbo steps!
  3. Hybrid Two-Stage Stabilization: Every step combines:
       (a) Pure 4-way symmetric geometric competition l_ce (ZERO uncontrasted l_sim blur pull!), PLUS
       (b) Fast low-variance single-step SwapDiff UNet 2x2 denoising swap anchor (8 pairs/step) to preserve
           Arm3b's analytical UNet geometry across all 4 timesteps while absorbing DINOv2 4-step supervision!

Clean 2-Arm Ablation on Dev-4 (200 steps = 3,200 full 4-step rollouts per arm, evaluated at s375 & s625):
  - M2a_DINOv2_GeomCompete_Cond   : Frozen EEG encoder, trains base_cond with DINOv2 CLS + 4x4 patch + H14/B32
                                    4-way competition + SwapDiff UNet anchor (direct upgrade over D2_Compete4_Cond!).
  - M2b_DINOv2_Hybrid_UnfEnc_Z0   : Unfrozen shared EEG trunk (block + head_f + horizon_logits) + base_cond +
                                    compact PCA-256 Delta_z0(E) trained with the exact same Hybrid DINOv2 + SwapDiff objective.
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
from exp6_draft_intra4 import DRAFT_DIR, forward_branches_and_trunk, precompute_or_load_tr14_z0_oof, rollout_4step_step_ips
from exp6c_clean_unfreeze_dev4 import CompactPCAZ0Head
from exp8_trial_precision_swapdiff import sample_same_concept_pairs
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder
from train_flex import shift_batch

from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel

nz = F.normalize
M2_DIR = OUT_DIR / "method2_dinov2_geom"
M2_DIR.mkdir(parents=True, exist_ok=True)
DINO_DIR = ROOT / "models" / "dinov2-base"


def extract_dinov2_cls_and_grid4(dino_model, imgs_01: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """Differentiable DINOv2-base CLS [B, 768] + 4x4 spatial patch grid [B, 16, 768] (L2-normalized)."""
    x = ((F.interpolate(imgs_01, size=224, mode="bilinear", align_corners=False) - mean) / std).to(dino_model.dtype)
    hs = dino_model(pixel_values=x).last_hidden_state.float()  # [B, 1 + 256, 768]
    cls = nz(hs[:, 0, :], dim=-1)  # [B, 768]
    # 16x16 patch grid -> avg_pool2d(4) -> 4x4 = 16 spatial geometry tokens [B, 16, 768]
    patches = hs[:, 1:, :].transpose(1, 2).reshape(-1, hs.shape[-1], 16, 16)
    g4 = F.avg_pool2d(patches, kernel_size=4, stride=4).flatten(2).transpose(1, 2)  # [B, 16, 768]
    g4 = nz(g4, dim=-1)
    return cls, g4


@torch.inference_mode()
def precompute_or_load_tr14_dinov2(tr_idx_np: np.ndarray, dino_model, vae_bf, dev: str = "cuda:0") -> dict:
    """Precompute DINOv2-base CLS [N_tr, 768] and 4x4 spatial grid [N_tr, 16, 768] fp16 for all tr14 images."""
    cache_p = M2_DIR / "dinov2_base_tr14_cls_grid4.pt"
    if cache_p.exists():
        print(f"[CACHED] {cache_p}", flush=True)
        return torch.load(cache_p, map_location="cpu", weights_only=True)

    t0 = time.time()
    print(f"[PRECOMPUTE] DINOv2-base CLS + 4x4 spatial patch targets for {len(tr_idx_np)} tr14 images ...", flush=True)
    lat_tr = np.load(PREP / "lat512_train.npy")[tr_idx_np]
    im_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    N = len(lat_tr)
    cls_all = torch.empty((N, 768), dtype=torch.float16)
    g4_all = torch.empty((N, 16, 768), dtype=torch.float16)
    bs = 8
    for i in range(0, N, bs):
        lb = torch.from_numpy(lat_tr[i:i + bs]).to(dev, torch.bfloat16)
        im = (vae_bf.decode(lb / vae_bf.config.scaling_factor).sample.float() / 2.0 + 0.5).clamp(0.0, 1.0)
        c_b, g_b = extract_dinov2_cls_and_grid4(dino_model, im, im_mean, im_std)
        cls_all[i:i + len(lb)] = c_b.half().cpu()
        g4_all[i:i + len(lb)] = g_b.half().cpu()
        if (i // bs) % 400 == 0:
            print(f"  [DINOv2 precompute] {i}/{N} ({time.time()-t0:.1f}s)", flush=True)
    del lat_tr
    pack = {"cls": cls_all, "grid4": g4_all}
    torch.save(pack, cache_p)
    print(f"[SAVED] {cache_p} ({time.time()-t0:.1f}s)", flush=True)
    torch.cuda.empty_cache()
    return pack


def train_m2_dinov2_arm(
    mode: str,
    init_ckpt: Path,
    res: dict,
    z0_oof_pack: dict,
    dino_tr_pack: dict,
    pipe_unet,
    scheduler,
    vae_bf,
    adapter_bf,
    proj_bf,
    dino_vis,
    h14_vis,
    b32_vis,
    b32_fix,
    pe_bf,
    ppe_bf,
    atid_bf,
    steps: int = 200,
    seed: int = 2025,
    dev: str = "cuda:0",
) -> Path:
    assert mode in ("m2a_dinov2_geom_cond", "m2b_dinov2_hybrid_unfenc_z0")
    out_p = M2_DIR / f"{mode}_st{steps}_s{seed}.pt"
    if out_p.exists():
        print(f"[CACHED] {out_p}", flush=True)
        return out_p

    unfreeze_enc = mode == "m2b_dinov2_hybrid_unfenc_z0"
    use_z0_head = mode == "m2b_dinov2_hybrid_unfenc_z0"

    t0 = time.time()
    pack = torch.load(init_ckpt, map_location="cpu", weights_only=True)
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
    model.eval()
    model.requires_grad_(False)

    enc_params = []
    model_ref = None
    if unfreeze_enc:
        for mod in (model.block, model.head_f):
            for p in mod.parameters():
                p.requires_grad_(True)
                enc_params.append(p)
        model.horizon_logits.requires_grad_(True)
        enc_params.append(model.horizon_logits)
        model_ref = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
        model_ref.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
        model_ref.eval().requires_grad_(False)

    base_cond = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev)
    base_cond.load_state_dict({k: v.float() for k, v in pack["cond_layer"].items()})

    cond_ref = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev)
    cond_ref.load_state_dict(base_cond.state_dict())
    cond_ref.eval().requires_grad_(False)

    V_pca = z0_oof_pack["V_pca"].to(dev)
    my_z0 = z0_oof_pack["my"].to(dev)
    Z_oof = z0_oof_pack["Z_oof"]
    z0_head = CompactPCAZ0Head(V_pca, z0_oof_pack["z0_rms_ref"]).to(dev) if use_z0_head else None

    cond_params = list(base_cond.parameters())
    param_groups = [{"params": cond_params, "lr": 1.2e-4, "weight_decay": 0.05}]
    all_train_params = list(cond_params)
    if unfreeze_enc:
        param_groups.append({"params": enc_params, "lr": 2.5e-5, "weight_decay": 0.02})
        all_train_params += enc_params
    if z0_head is not None:
        z0_params = list(z0_head.parameters())
        param_groups.append({"params": z0_params, "lr": 1.2e-4, "weight_decay": 0.05})
        all_train_params += z0_params

    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.995))

    def lr_lambda(st_i):
        warm = min(12, max(1, steps // 10))
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
    im_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    dino_cls_cpu = dino_tr_pack["cls"]     # [N_tr, 768] fp16
    dino_g4_cpu = dino_tr_pack["grid4"]    # [N_tr, 16, 768] fp16

    c_groups = res["tr14_c_groups"]
    n_c = len(c_groups)
    g = torch.Generator(device=dev).manual_seed(seed * 257 + 41)
    rng = np.random.default_rng(seed * 257 + 41)
    timesteps = scheduler.timesteps.to(dev)
    sigmas = scheduler.sigmas.to(dev)
    dt = pipe_unet.dtype

    n_micro = 4
    tau = 15.0

    print(
        f"[TRAIN {mode}] steps={steps} unfreeze_enc={unfreeze_enc} z0_head={use_z0_head} "
        f"scorer=0.35*DINO_cls+0.35*DINO_4x4+0.20*H14+0.10*B32 + aux SwapDiff UNet",
        flush=True,
    )
    base_cond.train()
    if z0_head is not None:
        z0_head.train()

    run_t1, run_dino_t1, run_ce = [], [], []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        st_t1, st_dt1, st_ce = 0.0, 0.0, 0.0

        # Part A: 4 groups x 4 same-concept photos full 4-step DRaFT with DINOv2 geometric competition
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
                d_cls_gt = dino_cls_cpu[locs].float().to(dev)   # [4, 768]
                d_g4_gt = dino_g4_cpu[locs].float().to(dev)     # [4, 16, 768]
                h14_gt = nz(res["H14_tr"][locs_dev].float(), dim=-1)
                b32_gt = nz(res["B32_tr"][locs_dev].float(), dim=-1)

            if unfreeze_enc:
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

            ips_cur = base_cond(c_ens, ips_proj)
            if z0_head is not None:
                z0_use = z0_head(c_ens, cs_br, z0_oof)
                l_reg = 0.04 * (ips_cur.float() - ips_ref).pow(2).mean() + 0.04 * (z0_use - z0_oof).pow(2).mean()
            else:
                z0_use = z0_oof
                l_reg = 0.04 * (ips_cur.float() - ips_ref).pow(2).mean()

            eps_sh = torch.randn(1, 4, 64, 64, device=dev, generator=g).expand(4, -1, -1, -1)
            sn_sh = [torch.randn(1, 4, 64, 64, device=dev, generator=g).expand(4, -1, -1, -1) for _ in range(4)]

            lat_out = rollout_4step_step_ips(
                pipe_unet, scheduler, [ips_cur, ips_cur, ips_cur, ips_cur], z0_use, eps_sh, sn_sh,
                pe_bf, ppe_bf, atid_bf, smap_c, low_keep,
            )

            imgs = []
            for j0 in range(0, 4, 2):
                im = (vae_bf.decode((lat_out[j0:j0 + 2] / vae_bf.config.scaling_factor).to(torch.bfloat16)).sample.float() / 2.0 + 0.5).clamp(0.0, 1.0)
                imgs.append(im)
            img_gen = torch.cat(imgs, dim=0)

            # Non-CLIP geometric DINOv2 CLS + 4x4 aligned spatial patch similarity
            d_cls_gen, d_g4_gen = extract_dinov2_cls_and_grid4(dino_vis, img_gen, im_mean, im_std)
            S_dino_cls = d_cls_gen @ d_cls_gt.T  # [4, 4]
            # Location-aligned 4x4 patch cosine: mean over 16 cells of <g_gen[i,k], g_gt[j,k]>
            S_dino_p4 = torch.einsum("ikd,jkd->ij", d_g4_gen, d_g4_gt) / 16.0  # [4, 4]

            # Light semantic anchor from Brain-HIVE LAION H14/B32 (ZERO OpenAI ViT-L/14 leakage!)
            x224 = ((F.interpolate(img_gen, size=224, mode="bilinear", align_corners=False) - clip_mean) / clip_std).to(torch.bfloat16)
            e_h14 = nz(h14_vis(pixel_values=x224).image_embeds.float(), dim=-1)
            e_b32 = nz(b32_vis(pixel_values=x224).image_embeds.float() @ b32_fix, dim=-1)
            S_h14 = e_h14 @ h14_gt.T
            S_b32 = e_b32 @ b32_gt.T

            S_4 = 0.35 * S_dino_cls + 0.35 * S_dino_p4 + 0.20 * S_h14 + 0.10 * S_b32
            tgt = torch.arange(4, device=dev)
            l_ce = 0.5 * (F.cross_entropy(tau * S_4, tgt) + F.cross_entropy(tau * S_4.T, tgt))
            loss_mb = (1.0 * l_ce + l_reg) / float(n_micro)
            if not torch.isfinite(loss_mb):
                raise RuntimeError(f"[{mode}] Non-finite loss at step {step} mb {mb}")
            loss_mb.backward()

            st_t1 += float((S_4.detach().argmax(dim=1) == tgt).float().mean()) / n_micro
            st_dt1 += float(((0.5 * S_dino_cls + 0.5 * S_dino_p4).detach().argmax(dim=1) == tgt).float().mean()) / n_micro
            st_ce += float(l_ce.detach()) / n_micro

        # Part B: Hybrid Auxiliary Low-Variance Single-Step SwapDiff UNet 2x2 Anchor (8 pairs/step, ~0.4s!)
        sid_u = int(rng.integers(0, 10))
        loc_a, loc_b = sample_same_concept_pairs(c_groups, 8, rng)
        xa = shift_batch(res["x_tr_all"][sid_u][loc_a].float().mean(1).to(dev), 2, g)
        xb = shift_batch(res["x_tr_all"][sid_u][loc_b].float().mean(1).to(dev), 2, g)
        if unfreeze_enc:
            fp_a, h_a, _ = forward_branches_and_trunk(model, xa, sid_u, r_eff=4.0)
            fp_b, h_b, _ = forward_branches_and_trunk(model, xb, sid_u, r_eff=4.0)
            c_a = per_sample_cone(fp_a, R_tr, a_tr, b_tr, u_tr).to(dt)
            c_b = per_sample_cone(fp_b, R_tr, a_tr, b_tr, u_tr).to(dt)
            ips_pa = proj_bf(adapter_bf(c_a))
            ips_pb = proj_bf(adapter_bf(c_b))
        else:
            with torch.no_grad():
                fp_a, h_a, _ = forward_branches_and_trunk(model, xa, sid_u, r_eff=4.0)
                fp_b, h_b, _ = forward_branches_and_trunk(model, xb, sid_u, r_eff=4.0)
                ips_pa = proj_bf(adapter_bf(per_sample_cone(fp_a, R_tr, a_tr, b_tr, u_tr).to(dt)))
                ips_pb = proj_bf(adapter_bf(per_sample_cone(fp_b, R_tr, a_tr, b_tr, u_tr).to(dt)))
        ips_a = base_cond(h_a, ips_pa).to(dt)
        ips_b = base_cond(h_b, ips_pb).to(dt)

        z_a = res["lat_tr_cpu"][loc_a].float().to(dev)
        z_b = res["lat_tr_cpu"][loc_b].float().to(dev)
        k_idx = torch.randint(0, 4, (8,), generator=g, device=dev)
        t_v = timesteps[k_idx]
        sig_v = sigmas[k_idx].view(8, 1, 1, 1).float()
        eps_a = torch.randn(8, 4, 64, 64, device=dev, generator=g)
        eps_b = torch.randn(8, 4, 64, 64, device=dev, generator=g)
        in_a = ((z_a + sig_v * eps_a) / (sig_v * sig_v + 1.0).sqrt()).to(dt)
        in_b = ((z_b + sig_v * eps_b) / (sig_v * sig_v + 1.0).sqrt()).to(dt)

        m_in = torch.cat([in_a, in_b, in_a, in_b], dim=0)
        t_in = torch.cat([t_v, t_v, t_v, t_v], dim=0)
        ip_in = torch.cat([ips_a, ips_b, ips_b, ips_a], dim=0)
        eps_hat = pipe_unet(
            m_in, t_in,
            encoder_hidden_states=pe_bf.expand(32, -1, -1),
            added_cond_kwargs={"text_embeds": ppe_bf.expand(32, -1), "time_ids": atid_bf.expand(32, -1)},
            cross_attention_kwargs={"ip_hidden_states": ip_in},
        ).sample.float()
        e_aa, e_bb, e_ab, e_ba = eps_hat.chunk(4, dim=0)
        d_aa = (e_aa - eps_a).pow(2).mean(dim=(1, 2, 3))
        d_bb = (e_bb - eps_b).pow(2).mean(dim=(1, 2, 3))
        d_ab = (e_ab - eps_a).pow(2).mean(dim=(1, 2, 3))
        d_ba = (e_ba - eps_b).pow(2).mean(dim=(1, 2, 3))
        margin = (d_ab + d_ba) - (d_aa + d_bb)
        l_aux = 0.25 * (0.75 * (d_aa.mean() + d_bb.mean()) + 0.45 * F.softplus(-18.0 * margin).mean())
        l_aux.backward()

        # Representation stability anchor when unfreezing EEG trunk
        if unfreeze_enc:
            sid_a = int(rng.integers(0, 10))
            c_pick = rng.choice(n_c, size=8, replace=False)
            loc_an = torch.tensor(c_groups[c_pick, :4].reshape(-1), dtype=torch.long)
            xan = res["x_tr_all"][sid_a][loc_an].float().mean(1).to(dev)
            fp_an, c_an, _ = forward_branches_and_trunk(model, xan, sid_a, r_eff=4.0)
            with torch.no_grad():
                fp_ar, c_ar, _ = forward_branches_and_trunk(model_ref, xan, sid_a, r_eff=4.0)
                f_gt_an = nz(res["F_tr"][loc_an.to(dev)].float(), dim=-1)
            f_pr_an = nz(fp_an.float(), dim=-1)
            l_anc = 0.20 * F.cross_entropy(14.0 * (f_pr_an @ f_gt_an.T), torch.arange(32, device=dev))
            l_anc = l_anc + 0.50 * (fp_an.float() - fp_ar.float()).pow(2).mean() + 0.50 * (c_an.float() - c_ar.float()).pow(2).mean()
            l_anc.backward()

        if step == 1:
            g_c = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in cond_params if p.grad is not None))
            g_e = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in enc_params if p.grad is not None)) if enc_params else 0.0
            g_z = float(z0_head.up.weight.grad.float().norm()) if z0_head is not None else 0.0
            print(f"  [{mode} STEP-1 DIAG] g_cond={g_c:.4f} g_enc={g_e:.4f} g_z0={g_z:.4f} mem_MB={torch.cuda.max_memory_allocated()//(1024*1024)}", flush=True)

        torch.nn.utils.clip_grad_norm_(all_train_params, 2.0)
        opt.step()
        sched.step()

        run_t1.append(st_t1)
        run_dino_t1.append(st_dt1)
        run_ce.append(st_ce)
        if step in (1, 10, 40, 80, 120, 160, 200) or step == steps:
            el = time.time() - t0
            eta = (el / step) * (steps - step)
            zn = float(z0_head.up.weight.detach().float().norm()) if z0_head is not None else 0.0
            print(
                f"  [{mode}] step {step:3d}/{steps} 4w_t1={np.mean(run_t1[-20:])*100:.1f}% "
                f"dino_t1={np.mean(run_dino_t1[-20:])*100:.1f}% ce={np.mean(run_ce[-20:]):.3f} z0n={zn:.3f} ({el:.0f}s, ETA {eta:.0f}s)",
                flush=True,
            )

    save_obj = {
        "backbone_ckpt": str(init_ckpt),
        "enc_shared": {
            "block": {k: v.half().cpu() for k, v in model.block.state_dict().items()},
            "head_f": {k: v.half().cpu() for k, v in model.head_f.state_dict().items()},
            "horizon_logits": model.horizon_logits.detach().half().cpu(),
        } if unfreeze_enc else None,
        "cond_layer": {k: v.half().cpu() for k, v in base_cond.state_dict().items()},
        "z0_head": {k: v.half().cpu() for k, v in z0_head.state_dict().items()} if z0_head is not None else None,
        "cone": pack["cone"],
        "mode": mode,
        "steps": steps,
    }
    torch.save(save_obj, out_p)
    print(f"[SAVED] {out_p} ({time.time()-t0:.1f}s)", flush=True)
    return out_p


@torch.inference_mode()
def evaluate_m2_arm(
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
    dino_eval,
    dino_gt_cls,
    dino_gt_g4,
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
    base_cond.load_state_dict({k: v.float() for k, v in pack["cond_layer"].items()})
    base_cond.eval()

    z0_head = None
    if pack.get("z0_head") is not None:
        z0_head = CompactPCAZ0Head(z0_oof_pack["V_pca"].to(dev), z0_oof_pack["z0_rms_ref"]).to(dev)
        z0_head.load_state_dict({k: v.float() for k, v in pack["z0_head"].items()})
        z0_head.eval()

    R_tr, a_tr, b_tr = pack["cone"]["R"], pack["cone"]["alpha"], pack["cone"]["beta"]
    u_tr = pack["cone"]["u_tr"].to(dev).float()

    cids_500 = [i // 10 for i in range(500)]
    eps_map = {ci: torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * ci + 99)) for ci in range(50)}
    sn_map = {
        ci: [torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(1_000_003 * ci + 12345 + 17 * st)) for st in range(4)]
        for ci in range(50)
    }
    im_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    res_by_reg = {}
    for rtag, smap_c, lk in [("s375", 0.375, (8, 0.11, 0.02)), ("s625", 0.625, (8, 0.22, 0.04))]:
        per_sub = {}
        for sub in subs:
            sid = sub - 1
            X_raw = np.load(PREP / f"sub-{sub:02d}_train.npy")
            xm4 = torch.from_numpy(X_raw[sel50_idx].mean(1)).float()
            del X_raw
            z0_e = z0_by_sub[sub]

            ips_l, z0_use_l = [], []
            for i in range(0, 500, 100):
                fp, c_ens, cs_br = forward_branches_and_trunk(model, xm4[i:i + 100].to(dev), sid, r_eff=4.0)
                c_cone = per_sample_cone(fp, R_tr, a_tr, b_tr, u_tr).to(torch.float16)
                ips_p = proj(adapter(c_cone))
                ips_l.append(base_cond(c_ens, ips_p).cpu())
                if z0_head is not None:
                    z0_use_l.append(z0_head(c_ens, cs_br, z0_e[i:i + 100].to(dev)).cpu())
                else:
                    z0_use_l.append(z0_e[i:i + 100])
            ips_all = torch.cat(ips_l, 0)
            z0_final = torch.cat(z0_use_l, 0)

            rec = torch.empty(500, 3, 512, 512, dtype=torch.float32)
            for s0 in range(0, 500, 50):
                q = list(range(s0, min(500, s0 + 50)))
                cids_q = [cids_500[i] for i in q]
                eps_q = torch.stack([eps_map[ci] for ci in cids_q]).to(dev)
                sn_q = [torch.stack([sn_map[ci][st] for ci in cids_q]).to(dev) for st in range(4)]
                ip_q = ips_all[q].to(dev)
                lat_out = rollout_4step_step_ips(
                    pipe_unet, scheduler, [ip_q, ip_q, ip_q, ip_q], z0_final[q].to(dev), eps_q, sn_q,
                    pe, ppe, atid, smap_c, lk,
                )
                for j0 in range(0, len(q), 10):
                    img_b = (vae32.decode(lat_out[j0:j0 + 10].float() / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
                    rec[s0 + j0:s0 + j0 + len(img_b)] = img_b

            m = score_full500(rec, gt_rgb50, gt_feats, nets, dev=dev)
            # Also track training scorer DINOv2 separately for transparency
            c_l, g_l = [], []
            for i in range(0, 500, 50):
                cb, gb = extract_dinov2_cls_and_grid4(dino_eval, rec[i:i + 50].to(dev), im_mean, im_std)
                c_l.append(cb.cpu())
                g_l.append(gb.cpu())
            dc_r = torch.cat(c_l, 0).view(50, 10, -1)
            dg_r = torch.cat(g_l, 0).view(50, 10, 16, -1)
            S_d = 0.5 * torch.bmm(dc_r, dino_gt_cls.transpose(1, 2)) + 0.5 * (torch.einsum("cikd,cjkd->cij", dg_r, dino_gt_g4) / 16.0)
            m["dinov2_train_scorer"] = intra10_and_swap_from_sim(S_d.numpy())
            per_sub[f"sub-{sub:02d}"] = m

        m_avg = {
            "clip_t10": float(np.mean([v["clip"]["top1"] for v in per_sub.values()])),
            "clip_dc10": float(np.mean([v["clip"]["top1_dc"] for v in per_sub.values()])),
            "clip_sw": float(np.mean([v["clip"]["swap2x2"] for v in per_sub.values()])),
            "clip_g500": float(np.mean([v["clip"]["global500_top1"] for v in per_sub.values()])),
            "dino_t10": float(np.mean([v["dinov2_train_scorer"]["top1"] for v in per_sub.values()])),
            "dino_sw": float(np.mean([v["dinov2_train_scorer"]["swap2x2"] for v in per_sub.values()])),
            "incep_sw": float(np.mean([v["inception"]["swap2x2"] for v in per_sub.values()])),
            "alex5_t10": float(np.mean([v["alex5"]["top1"] for v in per_sub.values()])),
            "alex5_sw": float(np.mean([v["alex5"]["swap2x2"] for v in per_sub.values()])),
            "pix_sw": float(np.mean([v["pixcorr"]["swap2x2"] for v in per_sub.values()])),
            "ssim": float(np.mean([v["ssim"] for v in per_sub.values()])),
        }
        res_by_reg[rtag] = {"mean": m_avg, "per_sub": per_sub}
        print(
            f">>> [{arm_name:28s} {rtag}] "
            f"INDEP CLIP-L14: t10={m_avg['clip_t10']*100:5.2f}% dc10={m_avg['clip_dc10']*100:5.2f}% sw={m_avg['clip_sw']*100:5.2f}% g500={m_avg['clip_g500']*100:4.2f}% | "
            f"Inc sw={m_avg['incep_sw']*100:5.2f}% | A5: t10={m_avg['alex5_t10']*100:5.2f}% sw={m_avg['alex5_sw']*100:5.2f}% | "
            f"Px sw={m_avg['pix_sw']*100:5.2f}% | SSIM={m_avg['ssim']:.4f} || (train-DINOv2: t10={m_avg['dino_t10']*100:5.2f}% sw={m_avg['dino_sw']*100:5.2f}%)",
            flush=True,
        )
    del model, base_cond, z0_head
    torch.cuda.empty_cache()
    return res_by_reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--modes", default="m2a_dinov2_geom_cond,m2b_dinov2_hybrid_unfenc_z0")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    dev = "cuda:0"
    t0 = time.time()
    arm3b_ckpt = OUT_DIR / "arm3b_rep_rawswap_condlyr_st600_s2025.pt"
    assert arm3b_ckpt.exists()
    assert DINO_DIR.exists(), f"Missing {DINO_DIR}"

    res = build_tr14_resources(dev=dev)
    z0_oof_pack = precompute_or_load_tr14_z0_oof(res["tr_idx_np"], res["concept_np"], dev=dev)

    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    from transformers import AutoModel, CLIPVisionModelWithProjection
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

    dino_vis = AutoModel.from_pretrained(str(DINO_DIR), dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    dino_vis.gradient_checkpointing_enable()
    h14_vis = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / "models" / EMB_KEYS[0]), torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    h14_vis.gradient_checkpointing_enable()
    b32_vis = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / "models" / EMB_KEYS[1]), torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    b32_vis.gradient_checkpointing_enable()
    b32_fix = torch.load(PREP / "b32_fix.pt", weights_only=True).float().to(dev)

    dino_tr_pack = precompute_or_load_tr14_dinov2(res["tr_idx_np"], dino_vis, vae_bf, dev=dev)

    ckpts = {}
    for m in modes:
        ckpts[m] = train_m2_dinov2_arm(
            m, arm3b_ckpt, res, z0_oof_pack, dino_tr_pack, pipe_tr.unet, pipe_tr.scheduler, vae_bf,
            adapter_bf, proj_bf, dino_vis, h14_vis, b32_vis, b32_fix, pe_bf, ppe_bf, atid_bf,
            steps=args.steps, seed=2025, dev=dev,
        )

    del pipe_tr, vae_bf, adapter_bf, proj_bf, dino_vis, h14_vis, b32_vis, pe_bf, ppe_bf, atid_bf, dino_tr_pack
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
    dino_eval = AutoModel.from_pretrained(str(DINO_DIR), dtype=dt).to(dev).eval().requires_grad_(False)

    lat50 = torch.from_numpy(np.load(PREP / "lat512_train.npy")[res["sel50_idx"]]).float()
    im_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    with torch.inference_mode():
        gt_rgb50 = torch.cat([
            (vae32.decode(lat50[j:j + 20].to(dev, torch.float32) / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
            for j in range(0, 500, 20)
        ])
        nets = _nets()
        gt_feats = extract_eval_feats(gt_rgb50, nets, dev=dev)
        c_gt, g_gt = [], []
        for i in range(0, 500, 50):
            cb, gb = extract_dinov2_cls_and_grid4(dino_eval, gt_rgb50[i:i + 50].to(dev), im_mean, im_std)
            c_gt.append(cb.cpu())
            g_gt.append(gb.cpu())
        dino_gt_cls = torch.cat(c_gt, 0).view(50, 10, -1)
        dino_gt_g4 = torch.cat(g_gt, 0).view(50, 10, 16, -1)

    res_p = M2_DIR / "exp9_m2_dinov2_geom_dev4_results.json"
    out_res = json.loads(res_p.read_text()) if res_p.exists() else {"arms": {}}
    for arm_name, cp in ckpts.items():
        if arm_name in out_res["arms"]:
            print(f"[CACHED EVAL] {arm_name}", flush=True)
            continue
        out_res["arms"][arm_name] = evaluate_m2_arm(
            arm_name, cp, DEV_SUBS, res["sel50_idx"], z0_by_sub, z0_oof_pack,
            pipe.unet, pipe.scheduler, vae32, adapter, proj, pe, ppe, atid,
            gt_rgb50, gt_feats, nets, dino_eval, dino_gt_cls, dino_gt_g4, dev=dev,
        )
        out_res["wall_s"] = round(time.time() - t0, 1)
        write_json(res_p, out_res)

    print(f"=== EXP 9 (METHOD 2 DINOV2 GEOMETRIC DRaFT DEV-4) DONE in {round(time.time()-t0, 1)}s -> {res_p} ===", flush=True)


if __name__ == "__main__":
    main()
