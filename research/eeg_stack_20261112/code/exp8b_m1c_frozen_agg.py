"""Method 1c (exp8b): Zero-Overfitting Frozen-Backbone TrialPrecisionSetAggregator.

Diagnosis from exp8:
  - M1a_ZeroRetrain_PrecDual (frozen Arm3b + inv_ch_rms2(p=0.5) ERP + per-trial feature mean + prec z0_ridge)
    IMMEDIATELY jumped on Dev-4 with ZERO retraining:
      F-rep 3270: 38.69% -> 39.74% (+1.05%pp), Swap 88.17% -> 89.16% (+0.99%pp)
      s375 FreeGen: CLIP Swap 67.84% -> 69.50% (+1.66%pp!), Alex5 Intra10 27.55% -> 28.90% (+1.35%pp!),
                    Alex5 Swap 78.26% -> 79.14% (+0.88%pp), g500 2.95% -> 3.25%
      s625 FreeGen: CLIP Intra10 18.45% -> 19.50% (+1.05%pp, 4/4 subjects UP!)
  - M1b fine-tuned the whole EEG encoder without Arm3b's multi-view cross-subject losses, causing tr14
    overfitting (rep4w=97.3%, fold-0 F-rep dropped 39.74% -> 38.26%).

Here (M1c) we keep Arm3b's EEG backbone AND cond_layer 100% FROZEN (zero encoder drift!) and train ONLY
the compact, permutation-invariant TrialPrecisionSetAggregator (~100K params) on tr14 4-trial EEG using
global + within-concept InfoNCE + MSE so it learns optimal channel-window precision weights, latent
trial-agreement attention, and adaptive dual-stream fusion without touching a single backbone weight!
"""
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
from exp0_intra10_swap_gate import ALL_SUBS, DEV_SUBS, OUT_DIR, intra10_and_swap_from_sim
from exp1_swapdiff_3arm import ResidualIPCondLayer, build_tr14_resources, per_sample_cone
from exp5_z0_2x2_diag import extract_eval_feats, score_full500
from exp6_draft_intra4 import forward_branches_and_trunk, rollout_4step_step_ips
from exp8_trial_precision_swapdiff import (
    M1_DIR,
    TrialPrecisionSetAggregator,
    evaluate_m1_variant,
    fit_tr14_ridge_z0_prec,
    prec_erp_analytic,
)
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder
from train_flex import shift_batch

from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel

nz = F.normalize


def train_m1c_frozen_agg(init_ckpt: Path, res: dict, steps: int = 800, seed: int = 2025, dev: str = "cuda:0") -> Path:
    out_p = M1_DIR / f"m1c_frozen_backbone_agg_st{steps}_s{seed}.pt"
    if out_p.exists():
        print(f"[CACHED] {out_p}", flush=True)
        return out_p

    t0 = time.time()
    pack = torch.load(init_ckpt, map_location="cpu", weights_only=True)
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
    model.eval().requires_grad_(False)

    agg = TrialPrecisionSetAggregator(n_ch=17, n_win=5, d_lat=1024).to(dev)
    # Initialize dual-stream weights right at the verified zero-retrain sweet spot (0.72 ERP / 0.28 Set)
    with torch.no_grad():
        agg.gate_net[-1].bias.fill_(-0.95)   # sigmoid ~= 0.28
        agg.fp_set_logit.fill_(-0.95)        # sigmoid ~= 0.28

    opt = torch.optim.AdamW(agg.parameters(), lr=1.0e-3, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-5)

    c_groups = res["tr14_c_groups"]
    n_c = len(c_groups)
    g = torch.Generator(device=dev).manual_seed(seed * 313 + 7)
    rng = np.random.default_rng(seed * 313 + 7)

    print(f"[TRAIN M1c_FrozenBackbone_Agg] steps={steps}, ONLY ~100K aggregator params trained!", flush=True)
    agg.train()
    run_t1 = []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        sid = int(rng.integers(0, 10))
        c_pick = rng.choice(n_c, size=16, replace=False)
        loc_r = torch.tensor(c_groups[c_pick, :4].reshape(-1), dtype=torch.long)  # [64]
        x4 = res["x_tr_all"][sid][loc_r].float().to(dev)  # [64, 4, 17, 250]
        # Random trial permutation + light independent per-trial jitter (+/-1 sample)
        x4 = torch.stack([shift_batch(x4[:, r], 1, g) for r in range(4)], dim=1)

        fp, h, fp_erp, fp_set = agg.forward_dual(model, x4, sid)
        F_gt = res["F_tr"][loc_r.to(dev)].float()
        f_pr_n = nz(fp.float(), dim=-1)
        f_gt_n = nz(F_gt, dim=-1)

        l_glob = F.cross_entropy(14.0 * (f_pr_n @ f_gt_n.T), torch.arange(64, device=dev))
        S_in = torch.bmm(f_pr_n.view(16, 4, -1), f_gt_n.view(16, 4, -1).transpose(1, 2))
        t4 = torch.arange(4, device=dev).unsqueeze(0).expand(16, -1).reshape(-1)
        l_intra = 0.5 * (
            F.cross_entropy((14.0 * S_in).reshape(64, 4), t4)
            + F.cross_entropy((14.0 * S_in.transpose(1, 2)).reshape(64, 4), t4)
        )
        l_mse = (fp.float() - F_gt).pow(2).mean()
        loss = 0.35 * l_glob + 0.65 * l_intra + 0.50 * l_mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agg.parameters(), 2.0)
        opt.step()
        sched.step()

        run_t1.append(float((S_in.detach().argmax(dim=2) == torch.arange(4, device=dev).view(1, 4)).float().mean()))
        if step in (1, 100, 300, 500, 800) or step == steps:
            a_m = float(F.softplus(agg.log_alpha.detach()).mean())
            b_m = float(agg.beta_agree.detach().mean())
            wf = float(torch.sigmoid(agg.fp_set_logit.detach()))
            print(
                f"  [M1c] step {step:3d}/{steps} rep4w={np.mean(run_t1[-40:])*100:5.1f}% "
                f"alpha={a_m:.2f} beta={b_m:.2f} w_fp={wf:.2f} ({time.time()-t0:.0f}s)",
                flush=True,
            )

    save_obj = {
        "state_dict": pack["state_dict"],
        "cond_layer": pack["cond_layer"],
        "agg": {k: v.half().cpu() for k, v in agg.state_dict().items()},
        "cone": pack["cone"],
        "steps": steps,
    }
    torch.save(save_obj, out_p)
    print(f"[SAVED] {out_p} ({time.time()-t0:.1f}s)", flush=True)
    return out_p


def main():
    dev = "cuda:0"
    t0 = time.time()
    arm3b_ckpt = OUT_DIR / "arm3b_rep_rawswap_condlyr_st600_s2025.pt"
    res = build_tr14_resources(dev=dev)
    m1c_ckpt = train_m1c_frozen_agg(arm3b_ckpt, res, steps=800, seed=2025, dev=dev)

    z0_prec_dev4 = {sub: fit_tr14_ridge_z0_prec(sub, res["tr_idx_np"], res["sel50_idx"], dev=dev, p=0.5) for sub in DEV_SUBS}

    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    from metrics import _nets

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

    res_p = M1_DIR / "exp8b_m1c_frozen_agg_dev4_results.json"
    out_res = {
        "wall_s": 0.0,
        "arms": {
            "M1c_FrozenBackbone_LearnedAgg": evaluate_m1_variant(
                "M1c_FrozenBackbone_LearnedAgg", "trained_agg", m1c_ckpt, DEV_SUBS, res["sel50_idx"], res["f0_idx"], F_f0,
                z0_prec_dev4, pipe.unet, pipe.scheduler, vae32, adapter, proj, pe, ppe, atid, gt_rgb50, gt_feats, nets, dev=dev,
            )
        },
    }
    out_res["wall_s"] = round(time.time() - t0, 1)
    write_json(res_p, out_res)
    print(f"=== EXP 8B (M1c FROZEN-BACKBONE LEARNED AGG DEV-4) DONE in {out_res['wall_s']}s -> {res_p} ===", flush=True)


if __name__ == "__main__":
    main()
