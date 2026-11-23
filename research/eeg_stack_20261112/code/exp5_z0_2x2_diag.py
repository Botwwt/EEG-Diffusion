"""Step 1 Diagnostic (exp5): Exact 2x2 Factorial (z0 x cond) Swap Experiment & Anchoring Strength Sweep
using existing saved weights (Arm0_Base, Arm1b_RepIntra_CondLyr, Arm3b_RepRawSwap_CondLyr) on sealed fold-0 (R=4 EEG).

Also builds & validates our self-contained differentiable 4-step SDXL-Turbo Euler-Ancestral rollout
(rollout_4step_diff) across all 4 anchoring regimes:
  - s625: smap_c=0.625, low_keep=(8, 0.22, 0.04)  [current O14 default: hold=1.25 -> step0 100% z0, step1 25% z0]
  - s375: smap_c=0.375, low_keep=(8, 0.11, 0.02)  [weak: hold=0.75 -> step0 75% z0, step1..3 100% free]
  - s150: smap_c=0.150, low_keep=None             [very weak: hold=0.30 -> step0 30% z0, step1..3 100% free, no tether]
  - s000: smap_c=0.000, low_keep=None             [pure noise: zero z0, 100% condition-driven 4-step rollout]

Outputs:
  A. 4-Cell (2x2) Cross-Combination (z0_A/B x cond_A/B) on 50 unseen concept pairs x 4 dev subs = 200 pairs:
       - Effect(z0) holding cond fixed
       - Effect(cond) holding z0 fixed
       - Conflict Winner: on (z0_A, c_B) vs (z0_B, c_A), % following cond vs % following z0
       - True target rank (/10) for matched vs cond-swapped vs z0-swapped
  B. Full 500-image FreeGen Intra10 & 2x2 Swap across all 4 anchoring regimes for Oracle, Arm1b, and Arm3b.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity

ROOT = Path(os.environ["RAD_ROOT"])
for d in [ROOT / "code/exp_w7", ROOT / "code/exp_w6", ROOT / "code/rad", ROOT / "code/exp", ROOT / "code/Brain-HIVE"]:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

from common import PREP, PRIOR, TURBO, write_json
from exp0_intra10_swap_gate import DEV_SUBS, OUT_DIR, fit_tr14_ridge_z0, get_concept_splits, intra10_and_swap_from_sim
from exp1_swapdiff_3arm import ResidualIPCondLayer, forward_with_trunk, per_sample_cone
from o11_tri_stream_bigg import _lp_op
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder

from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel

nz = F.normalize

REGIMES = [
    ("s625", 0.625, (8, 0.22, 0.04)),
    ("s375", 0.375, (8, 0.11, 0.02)),
    ("s150", 0.150, None),
    ("s000", 0.000, None),
]


def rollout_4step_diff(
    unet,
    scheduler,
    ips: torch.Tensor,
    z0: torch.Tensor,
    eps_init: torch.Tensor,
    step_noises: list,
    pe: torch.Tensor,
    ppe: torch.Tensor,
    atid: torch.Tensor,
    smap_c: float,
    low_keep,
) -> torch.Tensor:
    """Self-contained differentiable 4-step SDXL-Turbo Euler-Ancestral rollout (works with or without grad!)."""
    B = ips.shape[0]
    dev = ips.device
    dt = unet.dtype
    steps = 4
    timesteps = scheduler.timesteps  # [999, 749, 499, 249]
    sigmas = scheduler.sigmas        # [14.6146, 4.0817, 1.6129, 0.6932, 0.0]

    hold_val = 2.0 * smap_c if smap_c > 0.0 else 0.0  # 4 - 4*(1 - 0.5*smap_c) = 2*smap_c
    sig0 = float(sigmas[0])
    if smap_c > 0.0:
        lt = z0.float() + sig0 * eps_init.float()
    else:
        lt = sig0 * eps_init.float()

    pe_b = pe.expand(B, -1, -1).to(dt)
    ppe_b = ppe.expand(B, -1).to(dt)
    atid_b = atid.expand(B, -1).to(dt)
    ips_b = ips.to(dt)

    for i_step in range(steps):
        sigma = float(sigmas[i_step])
        sigma_next = float(sigmas[i_step + 1])
        t_cur = timesteps[i_step].expand(B)

        # Skip UNet call at i_step=0 ONLY if a==1.0 AND we are in no_grad? No, keep exact flow when grad enabled,
        # and at i_step=0 when smap_c=0.625, a=1.0 overwrites lt anyway (matches OfficialTurboRunner identically!).
        model_in = (lt / math.sqrt(sigma * sigma + 1.0)).to(dt)
        eps_pred = unet(
            model_in,
            t_cur,
            encoder_hidden_states=pe_b,
            added_cond_kwargs={"text_embeds": ppe_b, "time_ids": atid_b},
            cross_attention_kwargs={"ip_hidden_states": ips_b},
        ).sample.float()

        # Euler-Ancestral step (exact diffusers formula for prediction_type="epsilon")
        sigma_up = min(sigma_next, math.sqrt((sigma_next ** 2) * (sigma ** 2 - sigma_next ** 2) / (sigma ** 2))) if sigma_next > 0 else 0.0
        sigma_down = math.sqrt(max(0.0, sigma_next ** 2 - sigma_up ** 2))
        lt = lt + eps_pred * (sigma_down - sigma)
        if sigma_up > 0.0:
            lt = lt + sigma_up * step_noises[i_step].float()

        # Exact fractional start-latent hold & step-2/3 coarse tether
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


@torch.inference_mode()
def get_arm_ips_sel50(ckpt_p: Path, subs, sel50_idx, adapter, proj, dev="cuda:0"):
    pack = torch.load(ckpt_p, map_location="cpu", weights_only=True)
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
    model.eval()
    cl = None
    if pack.get("cond_layer") is not None:
        cl = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev)
        cl.load_state_dict({k: v.float() for k, v in pack["cond_layer"].items()})
        cl.eval()
    R_tr, a_tr, b_tr = pack["cone"]["R"], pack["cone"]["alpha"], pack["cone"]["beta"]
    u_tr = pack["cone"]["u_tr"].to(dev).float()

    ips_by_sub = {}
    for sub in subs:
        sid = sub - 1
        X_raw = np.load(PREP / f"sub-{sub:02d}_train.npy")
        xm4 = torch.from_numpy(X_raw[sel50_idx].mean(1)).float()
        del X_raw
        ips_l = []
        for i in range(0, len(xm4), 100):
            fp, _, _, h_tr = forward_with_trunk(model, xm4[i:i + 100].to(dev), sid, r_eff=4.0)
            c_cone = per_sample_cone(fp, R_tr, a_tr, b_tr, u_tr).to(torch.float16)
            ip = proj(adapter(c_cone))
            if cl is not None:
                ip = cl(h_tr, ip)
            ips_l.append(ip.cpu())
        ips_by_sub[sub] = torch.cat(ips_l, 0)  # [500, 4, 2048]
    del model, cl
    torch.cuda.empty_cache()
    return ips_by_sub


@torch.inference_mode()
def generate_batch_rollout(unet, scheduler, vae32, pe, ppe, atid, ips_all, z0_all, concept_ids, smap_c, low_keep, dev="cuda:0"):
    """Generate N images using exact concept-shared noise seeds for eps_init and 3 ancestral step noises."""
    N = len(ips_all)
    rec = torch.empty(N, 3, 512, 512, dtype=torch.float32)
    # Pre-generate deterministic noise per concept_id so A & B of the same concept share IDENTICAL noise!
    uniq_c = sorted(set(int(x) for x in concept_ids))
    eps_map = {}
    sn_map = {}
    for ci in uniq_c:
        eps_map[ci] = torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * ci + 99))
        sn_map[ci] = [
            torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(1_000_003 * ci + 12345 + 17 * st))
            for st in range(4)
        ]

    bs = 50
    for s0 in range(0, N, bs):
        q = list(range(s0, min(N, s0 + bs)))
        cids_q = [int(concept_ids[i]) for i in q]
        eps_q = torch.stack([eps_map[ci] for ci in cids_q]).to(dev)
        sn_q = [torch.stack([sn_map[ci][st] for ci in cids_q]).to(dev) for st in range(4)]
        lat_out = rollout_4step_diff(
            unet, scheduler, ips_all[q].to(dev), z0_all[q].to(dev), eps_q, sn_q,
            pe, ppe, atid, smap_c, low_keep,
        )
        for j0 in range(0, len(q), 10):
            img_b = (vae32.decode(lat_out[j0:j0 + 10].float() / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
            rec[s0 + j0:s0 + j0 + len(img_b)] = img_b
    return rec


@torch.inference_mode()
def extract_eval_feats(rec, nets, dev="cuda:0"):
    out = {}
    N = len(rec)
    for name in ("clip", "alex5", "inception"):
        mod, pre, layer = nets[name]
        fl = []
        for i in range(0, N, 50):
            y = mod(pre(rec[i:i + 50].to(dev)))
            fl.append((y if layer is None else y[layer]).float().flatten(1).cpu())
        out[name] = torch.cat(fl, 0)
    out["pix128"] = F.interpolate(rec, size=128, mode="bilinear", align_corners=False).flatten(1)
    return out


def score_full500(rec, gt_rgb50, gt_feats, nets, dev="cuda:0"):
    rf = extract_eval_feats(rec, nets, dev=dev)
    res = {}
    for name in ("clip", "alex5", "inception"):
        Fr = nz(rf[name] - rf[name].mean(0, keepdim=True), dim=1).view(50, 10, -1)
        Fg = nz(gt_feats[name] - gt_feats[name].mean(0, keepdim=True), dim=1).view(50, 10, -1)
        m = intra10_and_swap_from_sim(torch.bmm(Fr, Fg.transpose(1, 2)).numpy())
        if name == "clip":
            Fr_g = nz(rf[name] - rf[name].mean(0, keepdim=True), dim=1)
            Fg_g = nz(gt_feats[name] - gt_feats[name].mean(0, keepdim=True), dim=1)
            m["global500_top1"] = float(np.mean((Fr_g @ Fg_g.T).numpy().argmax(1) == np.arange(500)))
        res[name] = m
    rv = rf["pix128"].view(50, 10, -1)
    gv = gt_feats["pix128"].view(50, 10, -1)
    psim = torch.bmm(nz(rv - rv.mean(-1, keepdim=True), dim=-1), nz(gv - gv.mean(-1, keepdim=True), dim=-1).transpose(1, 2)).numpy()
    res["pixcorr"] = intra10_and_swap_from_sim(psim)
    res["pixcorr"]["diag_mean"] = float(np.mean(psim[:, np.arange(10), np.arange(10)]))
    r256 = F.interpolate(rec, size=256, mode="bilinear", align_corners=False).permute(0, 2, 3, 1).numpy()
    g256 = F.interpolate(gt_rgb50, size=256, mode="bilinear", align_corners=False).permute(0, 2, 3, 1).numpy()
    rg, gg = rgb2gray(r256), rgb2gray(g256)
    res["ssim"] = float(np.mean([
        structural_similarity(rg[i], gg[i], gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
        for i in range(500)
    ]))
    return res


def run_2x2_factorial(unet, scheduler, vae32, pe, ppe, atid, nets, gt_feats, ips_500, z0_500, pairs_ab, smap_c, low_keep, dev="cuda:0"):
    """Exact 4-cell (2x2) cross-combination: (z0_A, c_A), (z0_A, c_B), (z0_B, c_A), (z0_B, c_B) on 50 unseen pairs."""
    # pairs_ab: list of 50 tuples (ci, idx_a, idx_b) where idx_a = ci*10 + ia, idx_b = ci*10 + ib
    n_p = len(pairs_ab)
    ia = [p[1] for p in pairs_ab]
    ib = [p[2] for p in pairs_ab]
    cids = [p[0] for p in pairs_ab]

    # Stack 4*50 = 200 requests: [AA(50), AB(50), BA(50), BB(50)]
    z0_4 = torch.cat([z0_500[ia], z0_500[ia], z0_500[ib], z0_500[ib]], dim=0)
    ip_4 = torch.cat([ips_500[ia], ips_500[ib], ips_500[ia], ips_500[ib]], dim=0)
    cid_4 = cids * 4
    rec_4 = generate_batch_rollout(unet, scheduler, vae32, pe, ppe, atid, ip_4, z0_4, cid_4, smap_c, low_keep, dev=dev)
    rf = extract_eval_feats(rec_4, nets, dev=dev)

    out = {}
    for name in ("clip", "alex5", "pix128"):
        F_gt = gt_feats[name]
        if name == "pix128":
            Fr = nz(rf[name] - rf[name].mean(-1, keepdim=True), dim=-1)
            Fg = nz(F_gt - F_gt.mean(-1, keepdim=True), dim=-1)
        else:
            mu_g = F_gt.mean(0, keepdim=True)
            Fr = nz(rf[name] - mu_g, dim=-1)
            Fg = nz(F_gt - mu_g, dim=-1)

        f_aa, f_ab, f_ba, f_bb = Fr.chunk(4, dim=0)  # each [50, D]
        g_a, g_b = Fg[ia], Fg[ib]                    # each [50, D]

        # Cosine similarities to GT A and GT B for all 4 cells
        s_AA_A = (f_aa * g_a).sum(-1).numpy()
        s_AA_B = (f_aa * g_b).sum(-1).numpy()
        s_BB_B = (f_bb * g_b).sum(-1).numpy()
        s_BB_A = (f_bb * g_a).sum(-1).numpy()
        s_AB_A = (f_ab * g_a).sum(-1).numpy()
        s_AB_B = (f_ab * g_b).sum(-1).numpy()
        s_BA_A = (f_ba * g_a).sum(-1).numpy()
        s_BA_B = (f_ba * g_b).sum(-1).numpy()

        # 1. Pure z0 main effect (holding cond fixed): compare AA+BB vs BA+AB on matched cond?
        # Specifically: for cond=A, changing z0 A->B shifts sim to A vs B by (s_AA_A - s_AA_B) - (s_BA_A - s_BA_B)
        # Doubly-balanced z0 effect: [(s_AA_A + s_BB_B) - (s_BA_A + s_AB_B)] + [(s_AB_A + s_BA_B) - (s_BB_A + s_AA_B)]
        # Simpler & exact: holding cond fixed at A (AA vs BA) & at B (BB vs AB):
        eff_z0_vec = ((s_AA_A - s_BA_A) + (s_BB_B - s_AB_B)) - ((s_AA_B - s_BA_B) + (s_BB_A - s_AB_A))
        # Holding z0 fixed at A (AA vs AB) & at B (BB vs BA):
        eff_cond_vec = ((s_AA_A - s_AB_A) + (s_BB_B - s_BA_B)) - ((s_AA_B - s_AB_B) + (s_BB_A - s_BA_A))
        # Direct Conflict Test on mismatched cells (AB = z0_A + c_B) and (BA = z0_B + c_A):
        # Does the image follow cond (AB->B & BA->A) or follow z0 (AB->A & BA->B)?
        conflict_cond_minus_z0 = (s_AB_B + s_BA_A) - (s_AB_A + s_BA_B)
        # Full matched swap (both z0 and cond swapped: AA+BB vs AB_or_BA -> here AA->A + BB->B vs AA->B + BB->A)
        full_swap = (s_AA_A + s_BB_B) - (s_AA_B + s_BB_A)

        # True rank (/10) of GT A for AA vs AB (swapped cond) vs BA (swapped z0), and GT B for BB vs BA vs AB
        Fg_10 = Fg.view(50, 10, -1)
        def _mean_rank_and_top1(feat_50, target_local):
            # feat_50: [50, D], Fg_10: [50, 10, D]
            sims = torch.bmm(feat_50.unsqueeze(1), Fg_10.transpose(1, 2)).squeeze(1).numpy()  # [50, 10]
            ranks, t1 = [], []
            for ci_i in range(50):
                tgt = target_local[ci_i]
                r = int((sims[ci_i] > sims[ci_i, tgt]).sum()) + 1
                ranks.append(r)
                t1.append(1.0 if r == 1 else 0.0)
            return float(np.mean(ranks)), float(np.mean(t1))

        la = [p[1] % 10 for p in pairs_ab]
        lb = [p[2] % 10 for p in pairs_ab]
        rk_match_a, t1_match_a = _mean_rank_and_top1(f_aa, la)
        rk_match_b, t1_match_b = _mean_rank_and_top1(f_bb, lb)
        rk_swC_a, t1_swC_a = _mean_rank_and_top1(f_ab, la)   # kept z0_A, swapped cond to B
        rk_swC_b, t1_swC_b = _mean_rank_and_top1(f_ba, lb)
        rk_swZ_a, t1_swZ_a = _mean_rank_and_top1(f_ba, la)   # kept cond_A, swapped z0 to B
        rk_swZ_b, t1_swZ_b = _mean_rank_and_top1(f_ab, lb)

        out[name] = {
            "eff_z0_win": float((eff_z0_vec > 0).mean()),
            "eff_z0_delta": float(eff_z0_vec.mean()),
            "eff_cond_win": float((eff_cond_vec > 0).mean()),
            "eff_cond_delta": float(eff_cond_vec.mean()),
            "conflict_cond_wins": float((conflict_cond_minus_z0 > 0).mean()),
            "full_swap_win": float((full_swap > 0).mean()),
            "rank_matched": 0.5 * (rk_match_a + rk_match_b),
            "rank_swapCond": 0.5 * (rk_swC_a + rk_swC_b),
            "rank_swapZ0": 0.5 * (rk_swZ_a + rk_swZ_b),
            "top1_matched": 0.5 * (t1_match_a + t1_match_b),
            "top1_swapCond": 0.5 * (t1_swC_a + t1_swC_b),
            "top1_swapZ0": 0.5 * (t1_swZ_a + t1_swZ_b),
        }
    return out


def main():
    dev = "cuda:0"
    t0 = time.time()
    _, _, tr_idx, _, _, sel50_cids, sel50_idx = get_concept_splits()
    z0_by_sub = {sub: fit_tr14_ridge_z0(sub, tr_idx, sel50_idx, dev=dev) for sub in DEV_SUBS}

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
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    pipe.vae = None
    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, dt).eval().requires_grad_(False)
    proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, dt).eval().requires_grad_(False)
    vae32 = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval().requires_grad_(False)
    vae32.enable_slicing()

    lat50 = torch.from_numpy(np.load(PREP / "lat512_train.npy")[sel50_idx]).float()
    F_gt50 = torch.from_numpy(np.load(PREP / "F_train.npy")[sel50_idx]).float()
    with torch.inference_mode():
        ips_oracle = proj(adapter(F_gt50.to(dev, dt))).cpu()
        gt_rgb50 = torch.cat([
            (vae32.decode(lat50[j:j + 20].to(dev, torch.float32) / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
            for j in range(0, 500, 20)
        ])
        nets = _nets()
        gt_feats = extract_eval_feats(gt_rgb50, nets, dev=dev)

    # Pre-register 50 deterministic pairs (A, B) (photo #0 vs #5 of each concept — 100% unbiased!)
    pairs_ab = [(ci, ci * 10 + 0, ci * 10 + 5) for ci in range(50)]
    cids_500 = [i // 10 for i in range(500)]

    # 1. Sanity check our differentiable 4-step rollout with Oracle F_gt across all 4 anchoring regimes!
    print("=== SANITY CHECK: Oracle F_gt across 4 anchoring regimes (z0 = concept mean) ===", flush=True)
    z0_gt_cm = lat50.view(50, 10, 4, 64, 64).mean(1, keepdim=True).expand(-1, 10, -1, -1, -1).reshape(500, 4, 64, 64)
    oracle_res = {}
    for rtag, smap_c, lk in REGIMES:
        rec = generate_batch_rollout(pipe.unet, pipe.scheduler, vae32, pe, ppe, atid, ips_oracle, z0_gt_cm, cids_500, smap_c, lk, dev=dev)
        m = score_full500(rec, gt_rgb50, gt_feats, nets, dev=dev)
        oracle_res[rtag] = m
        print(
            f"  [Oracle {rtag}] CLIP t10={m['clip']['top1']*100:.2f}% sw={m['clip']['swap2x2']*100:.2f}% g500={m['clip']['global500_top1']*100:.2f}% | "
            f"Alex5 t10={m['alex5']['top1']*100:.2f}% sw={m['alex5']['swap2x2']*100:.2f}% | PixC sw={m['pixcorr']['swap2x2']*100:.2f}% ({m['pixcorr']['diag_mean']:.3f}) | SSIM={m['ssim']:.3f}",
            flush=True,
        )

    # 2. Load existing checkpoints' ips for Dev-4
    arm_ckpts = {
        "Arm0_Base": OUT_DIR / "arm0_tr14_xbrain_s2025.pt",
        "Arm1b_RepIntra_CondLyr": OUT_DIR / "arm1b_rep_condlyr_st600_s2025.pt",
        "Arm3b_RepRawSwap_CondLyr": OUT_DIR / "arm3b_rep_rawswap_condlyr_st600_s2025.pt",
    }
    ips_arms = {an: get_arm_ips_sel50(cp, DEV_SUBS, sel50_idx, adapter, proj, dev=dev) for an, cp in arm_ckpts.items()}

    # 3. PART A: 4-Cell (2x2) z0 x cond Factorial across 4 anchoring regimes
    print("\n=== PART A: 2x2 FACTORIAL (z0_A/B x cond_A/B) across anchoring strengths (DEV-4 mean, 200 unseen pairs) ===", flush=True)
    fact_res = {}
    for an in ("Arm0_Base", "Arm1b_RepIntra_CondLyr", "Arm3b_RepRawSwap_CondLyr"):
        fact_res[an] = {}
        for rtag, smap_c, lk in REGIMES:
            per_s = [
                run_2x2_factorial(pipe.unet, pipe.scheduler, vae32, pe, ppe, atid, nets, gt_feats, ips_arms[an][sub], z0_by_sub[sub], pairs_ab, smap_c, lk, dev=dev)
                for sub in DEV_SUBS
            ]
            # Average across 4 dev subs
            avg = {}
            for mn in ("clip", "alex5", "pix128"):
                avg[mn] = {k: float(np.mean([ps[mn][k] for ps in per_s])) for k in per_s[0][mn]}
            fact_res[an][rtag] = avg
            c, a5, px = avg["clip"], avg["alex5"], avg["pix128"]
            print(
                f"  [{an:24s} {rtag}] "
                f"CLIP: effZ={c['eff_z0_win']*100:4.1f}% effC={c['eff_cond_win']*100:4.1f}% cond>z0={c['conflict_cond_wins']*100:4.1f}% rk(match/swC/swZ)={c['rank_matched']:.2f}/{c['rank_swapCond']:.2f}/{c['rank_swapZ0']:.2f} | "
                f"A5: effZ={a5['eff_z0_win']*100:4.1f}% effC={a5['eff_cond_win']*100:4.1f}% cond>z0={a5['conflict_cond_wins']*100:4.1f}% | "
                f"Px: effZ={px['eff_z0_win']*100:4.1f}% effC={px['eff_cond_win']*100:4.1f}% cond>z0={px['conflict_cond_wins']*100:4.1f}%",
                flush=True,
            )

    # 4. PART B: Full 500-Image FreeGen Anchoring Sweep (Arm1b vs Arm3b across s625 / s375 / s150 / s000)
    print("\n=== PART B: FULL 500-IMG FREE-GEN ANCHORING SWEEP (DEV-4 mean, 4-NFE nsel=1, R=4 EEG) ===", flush=True)
    sweep_res = {}
    for an in ("Arm0_Base", "Arm1b_RepIntra_CondLyr", "Arm3b_RepRawSwap_CondLyr"):
        sweep_res[an] = {}
        for rtag, smap_c, lk in REGIMES:
            per_s = {}
            for sub in DEV_SUBS:
                rec = generate_batch_rollout(pipe.unet, pipe.scheduler, vae32, pe, ppe, atid, ips_arms[an][sub], z0_by_sub[sub], cids_500, smap_c, lk, dev=dev)
                per_s[f"sub-{sub:02d}"] = score_full500(rec, gt_rgb50, gt_feats, nets, dev=dev)
            m4 = {
                "clip_t10": float(np.mean([v["clip"]["top1"] for v in per_s.values()])),
                "clip_dc10": float(np.mean([v["clip"]["top1_dc"] for v in per_s.values()])),
                "clip_sw": float(np.mean([v["clip"]["swap2x2"] for v in per_s.values()])),
                "clip_g500": float(np.mean([v["clip"]["global500_top1"] for v in per_s.values()])),
                "incep_sw": float(np.mean([v["inception"]["swap2x2"] for v in per_s.values()])),
                "alex5_t10": float(np.mean([v["alex5"]["top1"] for v in per_s.values()])),
                "alex5_sw": float(np.mean([v["alex5"]["swap2x2"] for v in per_s.values()])),
                "pix_sw": float(np.mean([v["pixcorr"]["swap2x2"] for v in per_s.values()])),
                "pix_diag": float(np.mean([v["pixcorr"]["diag_mean"] for v in per_s.values()])),
                "ssim": float(np.mean([v["ssim"] for v in per_s.values()])),
            }
            sweep_res[an][rtag] = {"mean4": m4, "per_sub": per_s}
            print(
                f"  [{an:24s} {rtag}] "
                f"CLIP: t10={m4['clip_t10']*100:5.2f}% dc10={m4['clip_dc10']*100:5.2f}% sw={m4['clip_sw']*100:5.2f}% g500={m4['clip_g500']*100:4.2f}% | "
                f"Inc sw={m4['incep_sw']*100:5.2f}% | A5: t10={m4['alex5_t10']*100:5.2f}% sw={m4['alex5_sw']*100:5.2f}% | "
                f"Px: sw={m4['pix_sw']*100:5.2f}% ({m4['pix_diag']:.3f}) | SSIM={m4['ssim']:.4f}",
                flush=True,
            )

    out_p = OUT_DIR / "exp5_z0_2x2_diag.json"
    write_json(out_p, {"wall_s": round(time.time() - t0, 1), "oracle": oracle_res, "factorial_2x2": fact_res, "free_sweep": sweep_res})
    print(f"=== EXP 5 (z0 2x2 FACTORIAL + ANCHORING SWEEP) DONE in {round(time.time()-t0, 1)}s -> {out_p} ===", flush=True)


if __name__ == "__main__":
    main()
