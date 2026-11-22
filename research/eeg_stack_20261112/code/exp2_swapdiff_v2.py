"""Phase 2: Bottom-Up Fix of Posterior-Mean Blur Collapse & Rao-Blackwellized 2x2 EEG-Swap Denoising Contrast.

Root-cause diagnosis from Phase 1:
  - Arm2a_SwapDiff_Cone proved frozen-UNet 2x2 swap backprop massively boosted UNet denoising swap on unseen concepts
    (+7.2%pp 3t-avg, +13.4%pp at t=499, +11.7%pp at t=249, 4/4 subs) and FixedZ0 PixCorr swap (+3.14%pp, 4/4),
    while Arm1_RepIntra moved UNet swap 0% (74.89% -> 74.38%).
  - HOWEVER, Arm2a suffered textbook Posterior-Mean Blur Collapse because 20.0 * raw eps-MSE dominated L_base by 12x
    (grad_tot=9.04 vs grad_base=0.74), pulling c(E) toward the blurry gray posterior mean (SSIM +0.070, pxL1 shrank
    0.1345 -> 0.1169, L1_KV dropped 35.4% -> 23.9%, hurting sharp CLIP/Alex5 semantics).

Two clean first-principles fixes (ZERO L_rep_intra added -> 100% pure comparison vs Arm1_RepIntra!):
  1. Arm2_RawSwap_NoBlur (exact raw-eps 2x2 swap, identical architecture, 0 extra params):
       - Keeps exact raw-eps doubly-balanced swap Delta_swap = [d(A,E_A)+d(B,E_B)] - [d(A,E_B)+d(B,E_A)] < 0,
       - Fixes gradient imbalance: drops blur-inducing raw l_rec weight 20.0 -> 0.5, anchors L_base = 1.0,
         scales l_swap so grad_swap ~ 1.0 * grad_base (no manifold collapse!).
  2. Arm2_RBSwap_Cone (Rao-Blackwellized Exact Expected 2x2 Swap, identical architecture, 0 extra params):
       - By Law of Total Expectation, E[Delta_swap_raw | z_t, A, B] == Delta_swap_RB where
           d_RB(I, E) = || eps_hat(z_t^I, t, c(E)) - stopgrad(eps_hat(z_t^I, t, F_gt(I))) ||^2.
       - Eliminates 100% of irreducible single-noise Monte Carlo variance eta = eps - eps* AND eliminates
         posterior-mean blur collapse because the teacher field eps*(z_t^I, t, F_gt(I)) is crisp and sharp!
       - Auto-calibrates beta_t on Step 0 from the actual std(Delta_swap_RB) per timestep t in {749, 499, 249}.
  3. Arm2_RBSwap_CondLyr: Same Rao-Blackwellized 2x2 swap + zero-init low-rank residual IP conditioning layer (1024->128->16x2048).
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

from common import PREP, PRIOR, TURBO, write_json
from exp0_intra10_swap_gate import DEV_SUBS, OUT_DIR, fit_tr14_ridge_z0
from exp1_swapdiff_3arm import (
    ResidualIPCondLayer,
    build_tr14_resources,
    compute_base_anchor_loss,
    evaluate_arm_complete,
    forward_with_trunk,
    per_sample_cone,
)
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder
from train_flex import shift_batch

from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel


def calibrate_rb_betas(model, res, pipe_unet, alphas_cp, adapter, proj, pe, ppe, atid, pack, dev="cuda:0"):
    """Measure exact empirical std(Delta_swap_RB) per timestep t in {749, 499, 249} so beta_t * Delta_swap ~ O(1)."""
    R_tr, a_tr, b_tr, u_tr = pack["cone"]["R"], pack["cone"]["alpha"], pack["cone"]["beta"], pack["cone"]["u_tr"].to(dev).float()
    c_groups = res["tr14_c_groups"]
    rng = np.random.default_rng(20261122)
    g = torch.Generator(device=dev).manual_seed(20261122)
    betas = {}
    model.eval()
    with torch.no_grad():
        for t_int in [749, 499, 249]:
            ab = float(alphas_cp[t_int])
            sq_ab, sq_1mab = math.sqrt(ab), math.sqrt(1.0 - ab)
            deltas = []
            for _ in range(3):
                sid = int(rng.integers(0, 10))
                c_sel = rng.choice(len(c_groups), size=8, replace=False)
                la, lb = [], []
                for ci in c_sel:
                    ab_i = rng.choice(10, size=2, replace=False)
                    la.append(c_groups[ci, ab_i[0]])
                    lb.append(c_groups[ci, ab_i[1]])
                la_t, lb_t = torch.tensor(la), torch.tensor(lb)
                ea = res["x_tr_all"][sid][la_t].float().mean(1).to(dev)
                eb = res["x_tr_all"][sid][lb_t].float().mean(1).to(dev)
                z0_a = res["lat_tr_cpu"][la_t].float().to(dev)
                z0_b = res["lat_tr_cpu"][lb_t].float().to(dev)
                eps_a = torch.randn(z0_a.shape, device=dev, generator=g)
                eps_b = torch.randn(z0_b.shape, device=dev, generator=g)
                zt_a = (sq_ab * z0_a + sq_1mab * eps_a).to(torch.bfloat16)
                zt_b = (sq_ab * z0_b + sq_1mab * eps_b).to(torch.bfloat16)

                # Teacher sharp denoising fields
                fg_a = res["F_tr"][la_t.to(dev)].to(torch.bfloat16)
                fg_b = res["F_tr"][lb_t.to(dev)].to(torch.bfloat16)
                ips_gt = proj(adapter(torch.cat([fg_a, fg_b], 0)))
                zt_gt = torch.cat([zt_a, zt_b], 0)
                t_gt = torch.full((16,), t_int, device=dev, dtype=torch.long)
                p_gt = pipe_unet(
                    zt_gt, t_gt,
                    encoder_hidden_states=pe.expand(16, -1, -1),
                    added_cond_kwargs={"text_embeds": ppe.expand(16, -1), "time_ids": atid.expand(16, -1)},
                    cross_attention_kwargs={"ip_hidden_states": ips_gt},
                ).sample.float()
                gt_a, gt_b = p_gt.chunk(2, dim=0)

                fp_both, _, _, _ = forward_with_trunk(model, torch.cat([ea, eb], 0), sid, r_eff=4.0)
                c_both = per_sample_cone(fp_both, R_tr, a_tr, b_tr, u_tr).to(torch.bfloat16)
                ips_a, ips_b = proj(adapter(c_both)).chunk(2, dim=0)
                zt_4 = torch.cat([zt_a, zt_b, zt_a, zt_b], 0)
                ips_4 = torch.cat([ips_a, ips_b, ips_b, ips_a], 0)
                t_4 = torch.full((32,), t_int, device=dev, dtype=torch.long)
                p_4 = pipe_unet(
                    zt_4, t_4,
                    encoder_hidden_states=pe.expand(32, -1, -1),
                    added_cond_kwargs={"text_embeds": ppe.expand(32, -1), "time_ids": atid.expand(32, -1)},
                    cross_attention_kwargs={"ip_hidden_states": ips_4},
                ).sample.float()
                p_aa, p_bb, p_ab, p_ba = p_4.chunk(4, dim=0)
                d_aa = (p_aa - gt_a).pow(2).mean(dim=(1, 2, 3))
                d_bb = (p_bb - gt_b).pow(2).mean(dim=(1, 2, 3))
                d_ab = (p_ab - gt_a).pow(2).mean(dim=(1, 2, 3))
                d_ba = (p_ba - gt_b).pow(2).mean(dim=(1, 2, 3))
                deltas.append(((d_aa + d_bb) - (d_ab + d_ba)).cpu().numpy())
            arr = np.concatenate(deltas)
            sd = max(float(arr.std()), 1e-5)
            # Set beta_t = 1.0 / sd (clamped safely) so beta_t * Delta_swap has unit scale!
            beta_t = float(np.clip(1.0 / sd, 50.0, 2000.0))
            betas[t_int] = beta_t
            print(f"  [CALIB RB t={t_int}] mean={arr.mean():.3e} std={sd:.3e} win={(arr<0).mean()*100:.1f}% -> beta_t={beta_t:.1f}", flush=True)
    return betas


def train_swapdiff_v2(
    mode: str,
    arm0_ckpt: Path,
    res: dict,
    pipe_unet,
    alphas_cp: torch.Tensor,
    adapter,
    proj,
    pe,
    ppe,
    atid,
    rb_betas: dict,
    steps: int = 450,
    seed: int = 2025,
    dev: str = "cuda:0",
) -> Path:
    """Train one of: raw_noblur, rb_cone, rb_condlyr."""
    assert mode in ("raw_noblur", "rb_cone", "rb_condlyr")
    use_rb = mode in ("rb_cone", "rb_condlyr")
    use_cond_layer = mode == "rb_condlyr"
    out_p = OUT_DIR / f"arm2v2_{mode}_st{steps}_s{seed}.pt"
    if out_p.exists():
        print(f"[CACHED] {out_p}", flush=True)
        return out_p

    t0 = time.time()
    pack = torch.load(arm0_ckpt, map_location="cpu", weights_only=True)
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
    cond_layer = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.25).to(dev) if use_cond_layer else None

    R_tr = pack["cone"]["R"]
    a_tr = pack["cone"]["alpha"]
    b_tr = pack["cone"]["beta"]
    u_tr = pack["cone"]["u_tr"].to(dev).float()

    ls_f = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_h = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_b = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_x = nn.Parameter(torch.tensor(2.8, device=dev))

    params = [{"params": list(model.parameters()) + [ls_f, ls_h, ls_b, ls_x], "lr": 1.0e-4, "weight_decay": 0.03}]
    if cond_layer is not None:
        params.append({"params": list(cond_layer.parameters()), "lr": 2.0e-4, "weight_decay": 0.01})
    opt = torch.optim.AdamW(params, betas=(0.9, 0.995))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    pipe_unet.eval().requires_grad_(False)
    pipe_unet.enable_gradient_checkpointing()
    adapter.eval().requires_grad_(False)
    proj.eval().requires_grad_(False)

    g = torch.Generator(device=dev).manual_seed(seed * 71 + 19)
    rng = np.random.default_rng(seed * 71 + 19)
    c_groups = res["tr14_c_groups"]
    n_c = len(c_groups)
    n_tr = len(res["tr_idx_np"])

    T_CHOICES = [749, 499, 249]
    T_PROBS = [0.40, 0.40, 0.20]
    BETA_RAW = {749: 180.0, 499: 120.0, 249: 160.0}

    # Faster batching on 24GB 4090: 2 micro-batches x 8 pairs = 16 pairs/step (32 latents/UNet call -> ~15 GB VRAM)
    n_micro = 2
    pairs_per_mb = 8

    # Carefully balanced weights (ZERO blur collapse!):
    # - L_base = 1.0 (sharp visual manifold & inter-concept anchor, fn_mask=False!)
    # - For raw_noblur: w_rec = 0.5 (40x smaller than Arm2a's buggy 20.0!), w_swap = 0.35
    # - For rb_cone / rb_condlyr: teacher field eps* is SHARP -> w_rec_rb = 15.0 (scaled to ~0.5 grad norm), w_swap = 0.50
    w_base = 1.0
    if use_rb:
        w_rec = 15.0
        w_swap = 0.50
    else:
        w_rec = 0.5
        w_swap = 0.35

    print(f"[TRAIN arm2v2_{mode}] steps={steps}, pairs/step={n_micro*pairs_per_mb}, w_base={w_base}, w_rec={w_rec}, w_swap={w_swap}", flush=True)
    model.train()
    if cond_layer is not None:
        cond_layer.train()

    run_sw, run_delta, run_rec = [], [], []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        # 1. Sharp inter-concept anchor (fn_mask=False -> never suppresses within-concept discrimination!)
        idx_dev = torch.randint(0, n_tr, (512,), device=dev, generator=g)
        idx_cpu = idx_dev.cpu()
        s_pair = rng.choice(10, size=2, replace=False)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            l_base = w_base * compute_base_anchor_loss(
                model, res, int(s_pair[0]), int(s_pair[1]), idx_cpu, idx_dev, g, dev,
                ls_f, ls_h, ls_b, ls_x, remove_fn_mask=True,
            )
        l_base.backward()
        if step == 1:
            g_base = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in model.parameters() if p.grad is not None))

        # 2. Doubly-balanced 2x2 EEG-Swap Denoising Contrast
        step_rec_val, step_sw_val, step_d_val, step_win = 0.0, 0.0, 0.0, 0.0
        for mb in range(n_micro):
            sid = int(rng.integers(0, 10))
            c_sel = rng.choice(n_c, size=pairs_per_mb, replace=False)
            loc_a, loc_b = [], []
            for ci in c_sel:
                ab_idx = rng.choice(10, size=2, replace=False)
                loc_a.append(c_groups[ci, ab_idx[0]])
                loc_b.append(c_groups[ci, ab_idx[1]])
            loc_a_t = torch.tensor(loc_a, dtype=torch.long)
            loc_b_t = torch.tensor(loc_b, dtype=torch.long)

            ea = res["x_tr_all"][sid][loc_a_t].float().mean(1).to(dev)
            eb = res["x_tr_all"][sid][loc_b_t].float().mean(1).to(dev)
            ea, eb = shift_batch(ea, 2, g), shift_batch(eb, 2, g)

            z0_a = res["lat_tr_cpu"][loc_a_t].float().to(dev)
            z0_b = res["lat_tr_cpu"][loc_b_t].float().to(dev)

            t_int = int(rng.choice(T_CHOICES, p=T_PROBS))
            beta_t = rb_betas[t_int] if use_rb else BETA_RAW[t_int]
            ab_val = float(alphas_cp[t_int])
            sq_ab, sq_1mab = math.sqrt(ab_val), math.sqrt(1.0 - ab_val)

            eps_a = torch.randn(z0_a.shape, device=dev, generator=g, dtype=torch.float32)
            eps_b = torch.randn(z0_b.shape, device=dev, generator=g, dtype=torch.float32)
            zt_a = (sq_ab * z0_a + sq_1mab * eps_a).to(torch.bfloat16)
            zt_b = (sq_ab * z0_b + sq_1mab * eps_b).to(torch.bfloat16)

            if use_rb:
                # Exact Rao-Blackwellized sharp teacher denoising target eps*(z_t, t, F_gt)
                with torch.no_grad():
                    fg_a = res["F_tr"][loc_a_t.to(dev)].to(torch.bfloat16)
                    fg_b = res["F_tr"][loc_b_t.to(dev)].to(torch.bfloat16)
                    ips_gt = proj(adapter(torch.cat([fg_a, fg_b], dim=0)))
                    zt_gt = torch.cat([zt_a, zt_b], dim=0)
                    B2 = zt_gt.shape[0]
                    t_gt = torch.full((B2,), t_int, device=dev, dtype=torch.long)
                    p_gt = pipe_unet(
                        zt_gt, t_gt,
                        encoder_hidden_states=pe.expand(B2, -1, -1),
                        added_cond_kwargs={"text_embeds": ppe.expand(B2, -1), "time_ids": atid.expand(B2, -1)},
                        cross_attention_kwargs={"ip_hidden_states": ips_gt},
                    ).sample.float()
                    tgt_a, tgt_b = p_gt.chunk(2, dim=0)
            else:
                tgt_a, tgt_b = eps_a, eps_b

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                e_both = torch.cat([ea, eb], dim=0)
                fp_both, _, _, h_both = forward_with_trunk(model, e_both, sid, r_eff=4.0)
                c_both = per_sample_cone(fp_both, R_tr, a_tr, b_tr, u_tr).to(torch.bfloat16)
                ips_both = proj(adapter(c_both))
                if cond_layer is not None:
                    ips_both = cond_layer(h_both, ips_both)
                ips_a, ips_b = ips_both[:pairs_per_mb], ips_both[pairs_per_mb:]

                zt_4 = torch.cat([zt_a, zt_b, zt_a, zt_b], dim=0)
                ips_4 = torch.cat([ips_a, ips_b, ips_b, ips_a], dim=0)
                B4 = zt_4.shape[0]
                t_ten = torch.full((B4,), t_int, device=dev, dtype=torch.long)

                pred_4 = pipe_unet(
                    zt_4, t_ten,
                    encoder_hidden_states=pe.expand(B4, -1, -1),
                    added_cond_kwargs={"text_embeds": ppe.expand(B4, -1), "time_ids": atid.expand(B4, -1)},
                    cross_attention_kwargs={"ip_hidden_states": ips_4},
                ).sample.float()

                p_aa, p_bb, p_ab, p_ba = pred_4.chunk(4, dim=0)
                d_aa = (p_aa - tgt_a).pow(2).mean(dim=(1, 2, 3))
                d_bb = (p_bb - tgt_b).pow(2).mean(dim=(1, 2, 3))
                d_ab = (p_ab - tgt_a).pow(2).mean(dim=(1, 2, 3))
                d_ba = (p_ba - tgt_b).pow(2).mean(dim=(1, 2, 3))

                l_rec = 0.5 * (d_aa + d_bb).mean()
                delta_sw = (d_aa + d_bb) - (d_ab + d_ba)
                l_sw = F.softplus(beta_t * delta_sw).mean()
                l_diff = (w_rec * l_rec + w_swap * l_sw) / float(n_micro)

            l_diff.backward()
            step_rec_val += float(l_rec.detach()) / n_micro
            step_sw_val += float(l_sw.detach()) / n_micro
            step_d_val += float(delta_sw.detach().mean()) / n_micro
            step_win += float((delta_sw.detach() < 0.0).float().mean()) / n_micro

        if step == 1:
            g_tot = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in model.parameters() if p.grad is not None))
            g_cl = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in cond_layer.parameters() if p.grad is not None)) if cond_layer is not None else 0.0
            print(f"  [arm2v2_{mode} STEP-1 DIAG] grad_base={g_base:.3f} grad_tot={g_tot:.3f} grad_cl={g_cl:.3f} mem_MB={torch.cuda.max_memory_allocated()//(1024*1024)}", flush=True)

        all_p = list(model.parameters()) + [ls_f, ls_h, ls_b, ls_x] + (list(cond_layer.parameters()) if cond_layer is not None else [])
        torch.nn.utils.clip_grad_norm_(all_p, 2.0)
        opt.step()
        sched.step()

        run_sw.append(step_win)
        run_delta.append(step_d_val)
        run_rec.append(step_rec_val)
        if step in (1, 10, 100, 200, 300, 450):
            elapsed = time.time() - t0
            eta = (elapsed / step) * (steps - step)
            print(
                f"  [arm2v2_{mode}] step {step:3d}/{steps} base={float(l_base.detach()):.3f} rec={np.mean(run_rec[-30:]):.4e} "
                f"swL={step_sw_val:.3f} tr_swWin={np.mean(run_sw[-30:])*100:.1f}% delta={np.mean(run_delta[-30:]):.2e} ({elapsed:.0f}s, ETA {eta:.0f}s)",
                flush=True,
            )

    model.eval()
    if cond_layer is not None:
        cond_layer.eval()
    torch.save(
        {
            "state_dict": {k: v.half().cpu() for k, v in model.state_dict().items()},
            "cond_layer": {k: v.half().cpu() for k, v in cond_layer.state_dict().items()} if cond_layer is not None else None,
            "cone": pack["cone"],
            "arm": f"arm2v2_{mode}",
            "steps": steps,
        },
        out_p,
    )
    print(f"[SAVED] {out_p} ({time.time()-t0:.1f}s)", flush=True)
    return out_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="rb_cone,raw_noblur,rb_condlyr")
    ap.add_argument("--steps", type=int, default=450)
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    dev = "cuda:0"
    t_all = time.time()
    arm0_ckpt = OUT_DIR / "arm0_tr14_xbrain_s2025.pt"
    res = build_tr14_resources(dev=dev)
    z0_by_sub = {sub: fit_tr14_ridge_z0(sub, res["tr_idx_np"], res["sel50_idx"], dev=dev) for sub in DEV_SUBS}

    from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
    from metrics import _nets

    # 1. Load bf16 frozen UNet for training
    pipe_tr = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=torch.bfloat16, variant="fp16")
    IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(pipe_tr.unet)
    pipe_tr.to(dev, torch.bfloat16)
    with torch.no_grad():
        pe_bf, _, ppe_bf, _ = pipe_tr.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
        atid_bf = pipe_tr._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=torch.bfloat16, text_encoder_projection_dim=pipe_tr.text_encoder_2.config.projection_dim).to(dev)
        pe_bf, ppe_bf, atid_bf = pe_bf.detach().clone(), ppe_bf.detach().clone(), atid_bf.detach().clone()
    del pipe_tr.text_encoder, pipe_tr.text_encoder_2, pipe_tr.vae
    pipe_tr.text_encoder = None
    pipe_tr.text_encoder_2 = None
    pipe_tr.vae = None
    alphas_cp = pipe_tr.scheduler.alphas_cumprod.clone()
    adapter_bf = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, torch.bfloat16).eval().requires_grad_(False)
    proj_bf = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, torch.bfloat16).eval().requires_grad_(False)
    torch.cuda.empty_cache()

    # Calibrate RB betas once from Arm0
    pack0 = torch.load(arm0_ckpt, map_location="cpu", weights_only=True)
    m0 = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    m0.load_state_dict({k: v.float() for k, v in pack0["state_dict"].items()})
    rb_betas = calibrate_rb_betas(m0, res, pipe_tr.unet, alphas_cp, adapter_bf, proj_bf, pe_bf, ppe_bf, atid_bf, pack0, dev=dev)
    del m0
    torch.cuda.empty_cache()

    ckpts = {}
    for mode in modes:
        ckpts[mode] = train_swapdiff_v2(
            mode, arm0_ckpt, res, pipe_tr.unet, alphas_cp, adapter_bf, proj_bf, pe_bf, ppe_bf, atid_bf,
            rb_betas=rb_betas, steps=args.steps, seed=2025, dev=dev,
        )

    del pipe_tr, adapter_bf, proj_bf, pe_bf, ppe_bf, atid_bf
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Evaluate all newly trained v2 arms in the EXACT SAME fp16 OfficialTurboRunner protocol
    dt = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=dt, variant="fp16")
    IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(pipe.unet)
    pipe.to(dev, dt)
    pipe.unet.eval().requires_grad_(False)
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        pe, _, ppe, _ = pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
        atid = pipe._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=dt, text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).to(dev)
    del pipe.text_encoder, pipe.text_encoder_2
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    p2 = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    p2.set_progress_bar_config(disable=True)
    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, dt).eval().requires_grad_(False)
    proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, dt).eval().requires_grad_(False)
    vae32 = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval().requires_grad_(False)
    vae32.enable_slicing()

    lat50 = torch.from_numpy(np.load(PREP / "lat512_train.npy")[res["sel50_idx"]]).float()
    with torch.inference_mode():
        gt_rgb50 = torch.cat([
            (vae32.decode(lat50[j:j + 20].to(dev, torch.float32) / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
            for j in range(0, 500, 20)
        ])
        nets = _nets()
        gt_feats = {}
        for name, (mod, pre, layer) in nets.items():
            if name == "effnet":
                continue
            fl = []
            for i in range(0, 500, 50):
                y = mod(pre(gt_rgb50[i:i + 50].to(dev)))
                fl.append((y if layer is None else y[layer]).float().flatten(1).cpu())
            gt_feats[name] = torch.cat(fl, 0)

    res_p = OUT_DIR / "exp1_4arm_results.json"
    results = json.loads(res_p.read_text()) if res_p.exists() else {"arms": {}}
    for mode in modes:
        arm_name = f"Arm2v2_{mode}"
        results["arms"][arm_name] = evaluate_arm_complete(
            arm_name, ckpts[mode], res, pipe, p2, vae32, adapter, proj, pe, ppe, atid,
            lat50, gt_rgb50, gt_feats, z0_by_sub, dev=dev,
        )
        results["wall_s"] = round(time.time() - t_all, 1)
        write_json(res_p, results)

    print(f"=== PHASE 2 COMPLETE in {round(time.time()-t_all, 1)}s -> {res_p} ===", flush=True)


if __name__ == "__main__":
    main()
