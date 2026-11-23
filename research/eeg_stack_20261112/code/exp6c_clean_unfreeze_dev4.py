"""Step 2c (exp6c): Clean Parameter-Matched Unfrozen-Encoder DRaFT-Intra4 (Removing Noisy step_up[0..3]).

Directly motivated by exp6 + exp6b causal ablation findings:
  - exp6 proved 4-way within-concept competition (D2) beats SimOnly (D1) on 4/4 subjects across all metrics.
  - exp6b proved unfreezing the cross-subject shared EEG trunk (D4 vs D3) boosted CLIP-L14 Intra10 by +1.85%pp
    (18.35% -> 20.20%, new best!) and Alex5 Intra10 by +2.05%pp (26.00% -> 28.05%, new best!), while the 4 heavy
    randomly-initialized step_up[0..3] MLPs (8M new params) in D3 HURT by -1.60%pp vs Arm3b.

Here we drop the bloated step_up[0..3] heads completely and run a crisp 2-arm test:
  1. d2u_compete4_unfreeze_cond   : 100% EXACT same architecture & parameter count as Arm1b / Arm3b (ZERO new params!),
                                    unfreezing cross-subject shared EEG trunk (block + head_f + horizon_logits) +
                                    base_cond under pure 4-way within-concept DRaFT competition (w_sim=0) + within-concept
                                    tr14 representation anchor.
  2. d4c_compete4_unfreeze_condz0 : Exact same as d2u + ONLY the compact 164K-param PCA-256 Delta_z0(E) residual,
                                    directly isolating whether training z0 with the same 4-step competition objective
                                    adds further gain on top of unfrozen EEG trunk + cond_layer.

Zero Evaluator Leakage: training uses ONLY LAION OpenCLIP-H14/B32 + multi-scale VAE latent; evaluation uses
strictly independent OpenAI CLIP ViT-L/14 (primary: R=4 FreeGen Intra10 Top-1), Inception-v3, AlexNet-5, PixCorr, SSIM.
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
from exp6b_draft_v2_dev4 import safe_rms
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder
from train_flex import shift_batch

from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel

nz = F.normalize


class CompactPCAZ0Head(nn.Module):
    """Compact zero-init PCA-256 start-latent residual Delta_z0(E) (~164K params, NaN-safe)."""

    def __init__(self, v_pca_256: torch.Tensor, z0_rms_ref: float, max_rel_z0: float = 0.30):
        super().__init__()
        self.z0_rms_ref = float(z0_rms_ref)
        self.max_rel_z0 = float(max_rel_z0)
        self.register_buffer("V_pca", v_pca_256.float())
        self.ln = nn.LayerNorm(1024)
        self.down = nn.Linear(1024, 128)
        self.up = nn.Linear(128, v_pca_256.shape[1], bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, c_ens: torch.Tensor, cs_branches: tuple, z0_ridge: torch.Tensor) -> torch.Tensor:
        h_z = 0.5 * c_ens.float() + 0.5 * cs_branches[0].float()
        dz = (self.up(F.silu(self.down(self.ln(h_z)))) @ self.V_pca.T).view(-1, 4, 64, 64)
        dz_rms = safe_rms(dz, dim=(1, 2, 3), keepdim=True, eps=1e-12)
        sc = torch.minimum(torch.ones_like(dz_rms), (self.max_rel_z0 * self.z0_rms_ref) / dz_rms)
        return z0_ridge.float() + dz * sc


def train_clean_arm(
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
) -> Path:
    assert mode in ("d2u_compete4_unfreeze_cond", "d4c_compete4_unfreeze_condz0")
    out_p = DRAFT_DIR / f"{mode}_st{steps}_s{seed}.pt"
    if out_p.exists():
        print(f"[CACHED] {out_p}", flush=True)
        return out_p

    use_z0_head = mode == "d4c_compete4_unfreeze_condz0"
    t0 = time.time()
    pack = torch.load(init_ckpt, map_location="cpu", weights_only=True)

    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
    model.eval()
    model.requires_grad_(False)

    enc_params = []
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
    param_groups = [
        {"params": cond_params, "lr": 8.0e-5, "weight_decay": 0.05},
        {"params": enc_params, "lr": 2.5e-5, "weight_decay": 0.02},
    ]
    all_train_params = cond_params + enc_params
    if z0_head is not None:
        z0_params = list(z0_head.parameters())
        param_groups.append({"params": z0_params, "lr": 1.2e-4, "weight_decay": 0.05})
        all_train_params = all_train_params + z0_params

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
    g = torch.Generator(device=dev).manual_seed(seed * 149 + 23)
    rng = np.random.default_rng(seed * 149 + 23)

    n_micro = 4
    tau = 15.0

    print(f"[TRAIN {mode}] steps={steps} use_z0_head={use_z0_head} pure_ce=True s375", flush=True)
    base_cond.train()
    if z0_head is not None:
        z0_head.train()

    run_top1, run_sim, run_ce = [], [], []
    for step in range(1, steps + 1):
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
                fp_r, c_r, _ = forward_branches_and_trunk(model_ref, x4, sid, r_eff=4.0)
                ips_proj_r = proj_bf(adapter_bf(per_sample_cone(fp_r, R_tr, a_tr, b_tr, u_tr).to(torch.bfloat16)))
                ips_ref = cond_ref(c_r, ips_proj_r).detach().float()

            fp, c_ens, cs_br = forward_branches_and_trunk(model, x4, sid, r_eff=4.0)
            c_cone = per_sample_cone(fp, R_tr, a_tr, b_tr, u_tr).to(torch.bfloat16)
            ips_proj = proj_bf(adapter_bf(c_cone))
            ips_cur = base_cond(c_ens, ips_proj)
            ips_steps = [ips_cur, ips_cur, ips_cur, ips_cur]

            if z0_head is not None:
                z0_use = z0_head(c_ens, cs_br, z0_oof)
                l_reg = 0.05 * (ips_cur.float() - ips_ref).pow(2).mean() + 0.05 * (z0_use - z0_oof).pow(2).mean()
            else:
                z0_use = z0_oof
                l_reg = 0.05 * (ips_cur.float() - ips_ref).pow(2).mean()

            eps_sh = torch.randn(1, 4, 64, 64, device=dev, generator=g).expand(4, -1, -1, -1)
            sn_sh = [torch.randn(1, 4, 64, 64, device=dev, generator=g).expand(4, -1, -1, -1) for _ in range(4)]

            lat_out = rollout_4step_step_ips(
                pipe_unet, scheduler, ips_steps, z0_use, eps_sh, sn_sh,
                pe_bf, ppe_bf, atid_bf, smap_c, low_keep,
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

            loss_mb = (1.0 * l_ce + l_reg) / float(n_micro)
            if not torch.isfinite(loss_mb):
                raise RuntimeError(f"[{mode}] Non-finite loss at step {step} mb {mb}")
            loss_mb.backward()

            st_top1 += float((S_4.detach().argmax(dim=1) == tgt).float().mean()) / n_micro
            st_sim += float(S_4.detach().diag().mean()) / n_micro
            st_ce += float(l_ce.detach()) / n_micro

        # Fast tr14 representation stability anchor preserving BOTH global & within-concept geometry!
        sid_a = int(rng.integers(0, 10))
        c_pick = rng.choice(n_c, size=8, replace=False)
        loc_a = torch.tensor(c_groups[c_pick, :4].reshape(-1), dtype=torch.long)  # [8*4 = 32]
        xa = res["x_tr_all"][sid_a][loc_a].float().mean(1).to(dev)
        fp_a, c_a, _ = forward_branches_and_trunk(model, xa, sid_a, r_eff=4.0)
        with torch.no_grad():
            fp_ar, c_ar, _ = forward_branches_and_trunk(model_ref, xa, sid_a, r_eff=4.0)
            f_gt_a = nz(res["F_tr"][loc_a.to(dev)].float(), dim=-1)
        f_pr_a = nz(fp_a.float(), dim=-1)
        # Global 32-way + within-concept 4-way InfoNCE + trust-region to Arm3b
        l_glob = F.cross_entropy(14.0 * (f_pr_a @ f_gt_a.T), torch.arange(32, device=dev))
        S_in = torch.bmm(f_pr_a.view(8, 4, -1), f_gt_a.view(8, 4, -1).transpose(1, 2))
        t4 = torch.arange(4, device=dev).unsqueeze(0).expand(8, -1).reshape(-1)
        l_intra = 0.5 * (
            F.cross_entropy((14.0 * S_in).reshape(32, 4), t4)
            + F.cross_entropy((14.0 * S_in.transpose(1, 2)).reshape(32, 4), t4)
        )
        l_anc = 0.20 * l_glob + 0.30 * l_intra + 0.60 * (fp_a.float() - fp_ar.float()).pow(2).mean() + 0.60 * (c_a.float() - c_ar.float()).pow(2).mean()
        l_anc.backward()

        if step == 1:
            g_cond = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in cond_params if p.grad is not None))
            g_enc = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in enc_params if p.grad is not None))
            g_z0 = float(z0_head.up.weight.grad.float().norm()) if z0_head is not None else 0.0
            print(f"  [{mode} STEP-1 DIAG] g_cond={g_cond:.4f} g_enc={g_enc:.4f} g_z0={g_z0:.4f} mem_MB={torch.cuda.max_memory_allocated()//(1024*1024)}", flush=True)

        torch.nn.utils.clip_grad_norm_(all_train_params, 2.0)
        opt.step()
        sched.step()

        run_top1.append(st_top1)
        run_sim.append(st_sim)
        run_ce.append(st_ce)
        if step in (1, 10, 30, 60, 90, 120, 150) or step == steps:
            el = time.time() - t0
            eta = (el / step) * (steps - step)
            zn = float(z0_head.up.weight.detach().float().norm()) if z0_head is not None else 0.0
            print(
                f"  [{mode}] step {step:3d}/{steps} 4w_t1={np.mean(run_top1[-20:])*100:.1f}% "
                f"sim={np.mean(run_sim[-20:]):.4f} ce={np.mean(run_ce[-20:]):.3f} z0n={zn:.3f} ({el:.0f}s, ETA {eta:.0f}s)",
                flush=True,
            )

    save_obj = {
        "backbone_ckpt": str(init_ckpt),
        "enc_shared": {
            "block": {k: v.half().cpu() for k, v in model.block.state_dict().items()},
            "head_f": {k: v.half().cpu() for k, v in model.head_f.state_dict().items()},
            "horizon_logits": model.horizon_logits.detach().half().cpu(),
        },
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
def evaluate_clean_arm(
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
            f">>> [{arm_name:28s} {rtag}] "
            f"INDEP CLIP-L14: t10={m_avg['clip_t10']*100:5.2f}% dc10={m_avg['clip_dc10']*100:5.2f}% sw={m_avg['clip_sw']*100:5.2f}% g500={m_avg['clip_g500']*100:4.2f}% | "
            f"Inc sw={m_avg['incep_sw']*100:5.2f}% | A5: t10={m_avg['alex5_t10']*100:5.2f}% sw={m_avg['alex5_sw']*100:5.2f}% | "
            f"Px sw={m_avg['pix_sw']*100:5.2f}% | SSIM={m_avg['ssim']:.4f} || (train-H14: t10={m_avg['h14_t10']*100:5.2f}% sw={m_avg['h14_sw']*100:5.2f}%)",
            flush=True,
        )
    del model, base_cond, z0_head
    torch.cuda.empty_cache()
    return res_by_reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--modes", default="d2u_compete4_unfreeze_cond,d4c_compete4_unfreeze_condz0")
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
        ckpts[m] = train_clean_arm(
            m, arm3b_ckpt, res, z0_oof_pack, pipe_tr.unet, pipe_tr.scheduler, vae_bf,
            adapter_bf, proj_bf, h14_vis, b32_vis, b32_fix, pe_bf, ppe_bf, atid_bf,
            steps=args.steps, seed=2025, dev=dev,
        )

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

    res_p = DRAFT_DIR / "exp6c_clean_unfreeze_dev4_results.json"
    out_res = json.loads(res_p.read_text()) if res_p.exists() else {"arms": {}}
    for arm_name, cp in ckpts.items():
        if arm_name in out_res["arms"]:
            print(f"[CACHED EVAL] {arm_name}", flush=True)
            continue
        out_res["arms"][arm_name] = evaluate_clean_arm(
            arm_name, cp, DEV_SUBS, res["sel50_idx"], z0_by_sub, z0_oof_pack,
            pipe.unet, pipe.scheduler, vae32, adapter, proj, pe, ppe, atid,
            gt_rgb50, gt_feats, nets, h14_eval, H14_gt50, dev=dev,
        )
        out_res["wall_s"] = round(time.time() - t_all, 1)
        write_json(res_p, out_res)

    print(f"=== EXP 6C (CLEAN UNFREEZE DRaFT DEV-4) DONE in {round(time.time()-t_all, 1)}s -> {res_p} ===", flush=True)


if __name__ == "__main__":
    main()
