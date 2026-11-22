"""Phase 4: Parameter-Matched Rep-Only Control (Arm1b_CondLyr), Full 10-Subject & Untouched Held-Out-6
Confirmation ({1, 3, 4, 6, 7, 10}), Paired Significance Tests, and Visual 2x2 EEG-Swap Proof Gallery.

Strict Scientific Controls:
  - Arm0_Base               : Original O13/O14 shared-trunk tr14 baseline.
  - Arm1_RepIntra_1400      : 0-extra-param, step-matched (1,400 steps) representation-only within-concept control.
  - Arm1b_RepIntra_CondLyr  : 100% ARCHITECTURE-MATCHED & PARAMETER-MATCHED control to Arm3b (identical
                              ResidualIPCondLayer 4x2048, identical 800+600 steps, trained strictly in linear
                              70-layer UNet L_KV representation space, ZERO nonlinear UNet denoising backprop!).
  - Arm2v3_RawSwap_CondLyr  : Pure Raw-eps 2x2 Denoising Swap + 4x2048 CondLyr (ZERO L_rep_intra).
  - Arm3b_RepRawSwap_CondLyr: Full SwapDiff (Rep + Raw-eps 2x2 Denoising Swap + 4x2048 CondLyr).

All 5 arms are evaluated across ALL 10 subjects (22,500 within-concept photo pairs total; 13,500 pairs on
the 6 completely untouched held-out subjects {1, 3, 4, 6, 7, 10}) under R=4 real EEG, 4-step SDXL-Turbo (nsel=1).
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
from PIL import Image, ImageDraw
from scipy import stats
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity

ROOT = Path(os.environ["RAD_ROOT"])
for d in [ROOT / "code/exp_w7", ROOT / "code/exp_w6", ROOT / "code/rad", ROOT / "code/exp", ROOT / "code/Brain-HIVE"]:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

from common import EMB_KEYS, PREP, PRIOR, TURBO, write_json
from exp0_intra10_swap_gate import ALL_SUBS, DEV_SUBS, OUT_DIR, fit_tr14_ridge_z0, intra10_and_swap_from_sim
from exp1_swapdiff_3arm import (
    ResidualIPCondLayer,
    build_tr14_resources,
    compute_base_anchor_loss,
    forward_with_trunk,
    per_sample_cone,
)
from o10_factorized import _perp_dir
from o11_tri_stream_bigg import _lp_op
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder
from train_flex import shift_batch

from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel

nz = F.normalize
HELDOUT_SUBS = (1, 3, 4, 6, 7, 10)


def train_arm1b_rep_condlyr(arm1_ckpt: Path, res: dict, steps: int = 600, seed: int = 2025, dev: str = "cuda:0") -> Path:
    """Exact architecture- & parameter-matched Rep-Only control to Arm3b (4x2048 CondLyr trained ONLY in linear L_KV space)."""
    out_p = OUT_DIR / f"arm1b_rep_condlyr_st{steps}_s{seed}.pt"
    if out_p.exists():
        print(f"[CACHED] {out_p}", flush=True)
        return out_p

    t0 = time.time()
    pack = torch.load(arm1_ckpt, map_location="cpu", weights_only=True)
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
    cond_layer = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev)

    kv = res["kv_full"].kv
    R_tr, a_tr, b_tr = pack["cone"]["R"], pack["cone"]["alpha"], pack["cone"]["beta"]
    u_tr = pack["cone"]["u_tr"].to(dev).float()

    ls_f = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_h = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_b = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_x = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_in = nn.Parameter(torch.tensor(2.8, device=dev))

    opt = torch.optim.AdamW(
        [
            {"params": list(model.parameters()) + [ls_f, ls_h, ls_b, ls_x, ls_in], "lr": 1.0e-4, "weight_decay": 0.03},
            {"params": list(cond_layer.parameters()), "lr": 3.0e-4, "weight_decay": 0.01},
        ],
        betas=(0.9, 0.995),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    g = torch.Generator(device=dev).manual_seed(seed * 89 + 23)
    rng = np.random.default_rng(seed * 89 + 23)
    c_groups = res["tr14_c_groups"]
    n_c = len(c_groups)
    n_tr = len(res["tr_idx_np"])

    print(f"[TRAIN Arm1b_RepIntra_CondLyr] steps={steps}, exact 4x2048 CondLyr in linear L_KV space (0 UNet backprop) ...", flush=True)
    model.train()
    cond_layer.train()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        idx_dev = torch.randint(0, n_tr, (512,), device=dev, generator=g)
        idx_cpu = idx_dev.cpu()
        s_pair = rng.choice(10, size=2, replace=False)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            l_base = compute_base_anchor_loss(
                model, res, int(s_pair[0]), int(s_pair[1]), idx_cpu, idx_dev, g, dev,
                ls_f, ls_h, ls_b, ls_x, remove_fn_mask=True,
            )

            # Same-subject 8 concepts x 10 photos (exact R=4 mean)
            sid_c = int(rng.integers(0, 10))
            c_sel = rng.choice(n_c, size=8, replace=False)
            g_locs = c_groups[c_sel].reshape(-1)
            g_locs_t = torch.from_numpy(g_locs).long()
            x_c = res["x_tr_all"][sid_c][g_locs_t].float().mean(1).to(dev)
            x_c = shift_batch(x_c, 2, g)
            fp, hp, bp, h_tr = forward_with_trunk(model, x_c, sid_c, r_eff=4.0)

            idx_c_dev = g_locs_t.to(dev)
            with torch.no_grad():
                ff = res["F_tr"][idx_c_dev].float()
                uf = nz(ff, dim=-1)
                df_t = nz(uf - (uf @ u_tr.T) * u_tr, dim=-1)
                ips_gt = kv.proj(kv.adapter(ff)).float()
                ykv_t = nz((ips_gt @ kv.L_kv_t).flatten(1) - kv.mu_kv, dim=-1)
                ht = nz(res["H14_tr"][idx_c_dev].float(), dim=-1)
                bt = nz(res["B32_tr"][idx_c_dev].float(), dim=-1)

            c_cone = per_sample_cone(fp, R_tr, a_tr, b_tr, u_tr)
            with torch.amp.autocast("cuda", enabled=False):
                ips_base = kv.proj(kv.adapter(c_cone.float())).float()
                ips_pred = cond_layer(h_tr.float(), ips_base)
                zk_p = nz((ips_pred @ kv.L_kv_t).flatten(1) - kv.mu_kv, dim=-1)
                l_ips_mse = F.mse_loss(ips_pred, ips_gt)

            df_p = nz(nz(fp.float(), dim=-1) - (nz(fp.float(), dim=-1) @ u_tr.T) * u_tr, dim=-1)
            hp_n = nz(hp.float(), dim=-1)
            bp_n = nz(bp.float(), dim=-1)
            sc_in = ls_in.float().exp().clamp(5.0, 50.0)

            def _intra(P, T):
                Pb, Tb = P.view(8, 10, -1), T.view(8, 10, -1)
                S = torch.bmm(Pb, Tb.transpose(1, 2))
                tgt = torch.arange(10, device=dev).unsqueeze(0).expand(8, 10)
                ce = 0.5 * (
                    F.cross_entropy((sc_in * S).reshape(80, 10), tgt.reshape(80))
                    + F.cross_entropy((sc_in * S.transpose(1, 2)).reshape(80, 10), tgt.reshape(80))
                )
                iu, ju = torch.triu_indices(10, 10, offset=1, device=dev)
                delta = (S[:, iu, ju] + S[:, ju, iu]) - (S[:, iu, iu] + S[:, ju, ju])
                return ce + 0.5 * F.softplus(8.0 * delta).mean()

            # Strong supervision directly onto cond_layer's L_KV output + ips_gt MSE + cone/H14/B32
            l_intra = 0.25 * _intra(df_p, df_t) + 0.50 * _intra(zk_p, ykv_t) + 0.15 * _intra(hp_n, ht) + 0.10 * _intra(bp_n, bt)
            loss = l_base + 1.0 * l_intra + 0.5 * l_ips_mse

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(cond_layer.parameters()) + [ls_f, ls_h, ls_b, ls_x, ls_in], 2.0)
        opt.step()
        sched.step()
        if step in (1, 200, 400, 600):
            print(f"  [Arm1b_CondLyr] step {step:3d}/{steps} base={float(l_base.detach()):.3f} intra={float(l_intra.detach()):.3f} ips_mse={float(l_ips_mse.detach()):.4f} ({time.time()-t0:.1f}s)", flush=True)

    model.eval()
    cond_layer.eval()
    torch.save(
        {
            "state_dict": {k: v.half().cpu() for k, v in model.state_dict().items()},
            "cond_layer": {k: v.half().cpu() for k, v in cond_layer.state_dict().items()},
            "cone": pack["cone"],
            "arm": "Arm1b_RepIntra_CondLyr",
            "steps": steps,
        },
        out_p,
    )
    print(f"[SAVED] {out_p} ({time.time()-t0:.1f}s)", flush=True)
    return out_p


@torch.inference_mode()
def evaluate_arm_subs_fast(
    arm_name: str,
    ckpt_p: Path,
    subs_to_eval: list,
    res: dict,
    pipe,
    p2,
    vae32,
    adapter,
    proj,
    pe,
    ppe,
    atid,
    lat50: torch.Tensor,
    gt_rgb50: torch.Tensor,
    gt_feats: dict,
    z0_by_sub: dict,
    init_fix_by_sub: dict,
    init_free_by_sub: dict,
    dev: str = "cuda:0",
):
    """Exact-parity fast evaluator (caches start-image VAE decodes, bs=50 UNet)."""
    from metrics import _nets

    pack = torch.load(ckpt_p, map_location="cpu", weights_only=True)
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
    model.eval()

    cond_layer = None
    if pack.get("cond_layer") is not None:
        cond_layer = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev)
        cond_layer.load_state_dict({k: v.float() for k, v in pack["cond_layer"].items()})
        cond_layer.eval()

    R_tr, a_tr, b_tr = pack["cone"]["R"], pack["cone"]["alpha"], pack["cone"]["beta"]
    u_tr = pack["cone"]["u_tr"].to(dev).float()
    kv_full = res["kv_full"]

    F_all = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    H14_all = emb_tr[EMB_KEYS[0]].float()

    f0_idx = res["f0_idx"]
    sel50_idx = res["sel50_idx"]
    f0_pos = {int(g): i for i, g in enumerate(f0_idx)}
    loc50 = np.array([f0_pos[int(g)] for g in sel50_idx], dtype=np.int64)

    def _cos_blk(P, G, nc):
        return torch.bmm(nz(P.float(), dim=-1).view(nc, 10, -1), nz(G.float(), dim=-1).view(nc, 10, -1).transpose(1, 2)).cpu().numpy()

    nets = _nets()
    per_sub = {}
    for sub in subs_to_eval:
        sid = sub - 1
        X_raw = np.load(PREP / f"sub-{sub:02d}_train.npy")
        xm4_f0 = torch.from_numpy(X_raw[f0_idx].mean(1)).float()
        del X_raw

        fp_l, hp_l, h_l = [], [], []
        for i in range(0, len(xm4_f0), 256):
            fp, hp, _, h_tr = forward_with_trunk(model, xm4_f0[i:i + 256].to(dev), sid, r_eff=4.0)
            fp_l.append(fp.cpu())
            hp_l.append(hp.cpu())
            h_l.append(h_tr.cpu())
        fp_f0, hp_f0, h_f0 = torch.cat(fp_l, 0), torch.cat(hp_l, 0), torch.cat(h_l, 0)

        df_p = nz(nz(fp_f0.to(dev), dim=-1) - (nz(fp_f0.to(dev), dim=-1) @ u_tr.T) * u_tr, dim=-1)
        df_g = _perp_dir(F_all[f0_idx].to(dev), u_tr)
        m_f327 = intra10_and_swap_from_sim(_cos_blk(df_p, df_g, len(res["f0_all10"])))
        c_f0 = per_sample_cone(fp_f0.to(dev), R_tr, a_tr, b_tr, u_tr)
        zk_p = nz(kv_full.kv.project_kv(c_f0) - kv_full.kv.mu_kv, dim=-1)
        zk_g = nz(kv_full.kv.project_kv(F_all[f0_idx].to(dev)) - kv_full.kv.mu_kv, dim=-1)
        m_kv327 = intra10_and_swap_from_sim(_cos_blk(zk_p, zk_g, len(res["f0_all10"])))
        m_h327 = intra10_and_swap_from_sim(_cos_blk(hp_f0.to(dev), H14_all[f0_idx].to(dev), len(res["f0_all10"])))

        c_50 = c_f0[loc50].to(torch.float16)
        ips_50 = proj(adapter(c_50))
        if cond_layer is not None:
            ips_50 = cond_layer(h_f0[loc50].to(dev), ips_50)
        ips_50 = ips_50.cpu()

        # Layer 2 (25 concepts x 100 pairs x 3 active timesteps, chunked by 100!)
        pipe.scheduler.set_timesteps(4, device=dev)
        n_c2 = 25
        lat_sub = lat50[:n_c2 * 10].to(dev, torch.float32)
        ips_sub = ips_50[:n_c2 * 10].to(dev, torch.float16)
        l2_t = {}
        err_sum = np.zeros((n_c2, 10, 10), dtype=np.float64)
        for t_int in [749, 499, 249]:
            ab = float(pipe.scheduler.alphas_cumprod[t_int])
            sq_ab, sq_1mab = math.sqrt(ab), math.sqrt(1.0 - ab)
            gen = torch.Generator(device=dev).manual_seed(20261122 + t_int)
            eps = torch.randn(lat_sub.shape, generator=gen, device=dev, dtype=torch.float32)
            zt_all = (sq_ab * lat_sub + sq_1mab * eps).to(torch.float16)
            t_ten = torch.tensor([t_int], device=dev, dtype=torch.long)
            D_mat = np.zeros((n_c2, 10, 10), dtype=np.float64)
            for ci in range(n_c2):
                sl = slice(ci * 10, (ci + 1) * 10)
                zt_p = zt_all[sl].repeat_interleave(10, dim=0)
                ep_p = eps[sl].repeat_interleave(10, dim=0)
                ip_p = ips_sub[sl].repeat(10, 1, 1)
                pr = pipe.unet(
                    zt_p, t_ten.expand(100),
                    encoder_hidden_states=pe.expand(100, -1, -1),
                    added_cond_kwargs={"text_embeds": ppe.expand(100, -1), "time_ids": atid.expand(100, -1)},
                    cross_attention_kwargs={"ip_hidden_states": ip_p},
                ).sample.float()
                D_mat[ci] = (pr - ep_p).pow(2).mean(dim=(1, 2, 3)).cpu().numpy().reshape(10, 10)
            l2_t[f"t{t_int}"] = intra10_and_swap_from_sim(-D_mat)
            err_sum += (D_mat - D_mat.mean(axis=(1, 2), keepdims=True)) / max(float(D_mat.std()), 1e-8)
        l2_t["avg_3t"] = intra10_and_swap_from_sim(-err_sum)

        # Layer 3 (bs=50, cached init!)
        def _gen_score(z0_all, init_rgb):
            N, steps = 500, 4
            s_rel, s_unrel, smap_c = 0.5, 1.0, 0.625
            hold_val = steps - steps * (s_unrel - (s_unrel - s_rel) * smap_c)
            low_keep = (8, 0.22, 0.04)
            rec = torch.empty(N, 3, 512, 512, dtype=torch.float32)
            for s0 in range(0, N, 50):
                q = list(range(s0, min(N, s0 + 50)))
                Q = len(q)
                ip_b = ips_50[q].to(dev, torch.float16)
                gens = [torch.Generator(dev).manual_seed(1_000_003 * (i // 10) + 12345) for i in q]
                eps_q = torch.stack([
                    torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * (i // 10) + 99))
                    for i in q
                ]).to(dev)
                z0_q = z0_all[q].to(dev, torch.float32)
                i0_q = init_rgb[q].to(dev, torch.float16)

                def cb(pp, i_step, t_cur, ck):
                    lt = ck["latents"].float()
                    si = int(pp.scheduler.step_index)
                    sig = pp.scheduler.sigmas[si].to(dev)
                    a = max(0.0, min(1.0, hold_val - i_step))
                    if a > 0:
                        lt = a * (z0_q + sig * eps_q) + (1.0 - a) * lt
                    if i_step in (steps - 2, steps - 1):
                        f_lp, w2, w3 = low_keep
                        g_t = w2 if i_step == steps - 2 else w3
                        if g_t > 0:
                            lt = lt + g_t * _lp_op((z0_q + sig * eps_q) - lt, f_lp)
                    ck["latents"] = lt.to(torch.float16)
                    return ck

                lat_out = p2(
                    image=i0_q,
                    prompt_embeds=pe.expand(Q, -1, -1),
                    pooled_prompt_embeds=ppe.expand(Q, -1),
                    num_inference_steps=steps,
                    strength=s_unrel,
                    guidance_scale=0.0,
                    output_type="latent",
                    generator=gens,
                    cross_attention_kwargs={"ip_hidden_states": ip_b},
                    callback_on_step_end=cb,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images
                for j0 in range(0, Q, 10):
                    img_b = (vae32.decode(lat_out[j0:j0 + 10].float() / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
                    rec[s0 + j0:s0 + j0 + len(img_b)] = img_b

            out = {}
            for name, (mod, pre, layer) in nets.items():
                if name == "effnet":
                    continue
                fl = []
                for i in range(0, N, 50):
                    y = mod(pre(rec[i:i + 50].to(dev)))
                    fl.append((y if layer is None else y[layer]).float().flatten(1).cpu())
                F_rec = torch.cat(fl, 0)
                F_gt = gt_feats[name]
                Fr = nz(F_rec - F_rec.mean(0, keepdim=True), dim=1).view(50, 10, -1)
                Fg = nz(F_gt - F_gt.mean(0, keepdim=True), dim=1).view(50, 10, -1)
                m = intra10_and_swap_from_sim(torch.bmm(Fr, Fg.transpose(1, 2)).numpy())
                if name == "clip":
                    Fr_g = nz(F_rec - F_rec.mean(0, keepdim=True), dim=1)
                    Fg_g = nz(F_gt - F_gt.mean(0, keepdim=True), dim=1)
                    m["global500_top1"] = float(np.mean((Fr_g @ Fg_g.T).numpy().argmax(1) == np.arange(N)))
                    iu, ju = np.triu_indices(10, k=1)
                    ss = torch.bmm(Fr, Fr.transpose(1, 2)).numpy()
                    m["swap_clip_disp"] = float(np.mean(1.0 - ss[:, iu, ju]))
                out[name] = m

            r128 = F.interpolate(rec, size=128, mode="bilinear", align_corners=False).view(50, 10, -1)
            g128 = F.interpolate(gt_rgb50, size=128, mode="bilinear", align_corners=False).view(50, 10, -1)
            pix_sim = torch.bmm(nz(r128 - r128.mean(-1, keepdim=True), dim=-1), nz(g128 - g128.mean(-1, keepdim=True), dim=-1).transpose(1, 2)).numpy()
            out["pixcorr"] = intra10_and_swap_from_sim(pix_sim)
            out["pixcorr"]["diag_mean"] = float(np.mean(pix_sim[:, np.arange(10), np.arange(10)]))
            iu, ju = np.triu_indices(10, k=1)
            out["swap_pixel_l1"] = float((r128[:, iu] - r128[:, ju]).abs().mean())

            r256 = F.interpolate(rec, size=256, mode="bilinear", align_corners=False).permute(0, 2, 3, 1).numpy()
            g256 = F.interpolate(gt_rgb50, size=256, mode="bilinear", align_corners=False).permute(0, 2, 3, 1).numpy()
            rg, gg = rgb2gray(r256), rgb2gray(g256)
            out["ssim_diag"] = float(np.mean([
                structural_similarity(rg[i], gg[i], gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
                for i in range(N)
            ]))
            return out

        z0_e = z0_by_sub[sub]
        z0_cmean = z0_e.view(50, 10, 4, 64, 64).mean(dim=1, keepdim=True).expand(-1, 10, -1, -1, -1).reshape(500, 4, 64, 64)
        l3_fix = _gen_score(z0_cmean, init_fix_by_sub[sub])
        l3_free = _gen_score(z0_e, init_free_by_sub[sub])

        per_sub[f"sub-{sub:02d}"] = {
            "L1_327": {"F": m_f327, "KV": m_kv327, "H14": m_h327},
            "L2_unet": l2_t,
            "L3_fixedZ0": l3_fix,
            "L3_freeGen": l3_free,
        }
        print(
            f"[EVAL10 {arm_name:24s} sub-{sub:02d}] "
            f"L1(F/KV)={m_f327['top1']*100:.1f}/{m_kv327['top1']*100:.1f}% | "
            f"L2(sw/dc10)={l2_t['avg_3t']['swap2x2']*100:.1f}/{l2_t['avg_3t']['top1_dc']*100:.1f}% | "
            f"FixZ0(CLIP t10/sw, A5 t10/sw, Px sw)={l3_fix['clip']['top1']*100:.1f}/{l3_fix['clip']['swap2x2']*100:.1f}%, "
            f"{l3_fix['alex5']['top1']*100:.1f}/{l3_fix['alex5']['swap2x2']*100:.1f}%, {l3_fix['pixcorr']['swap2x2']*100:.1f}% | "
            f"Free(CLIP t10/sw/g500, Inc sw, A5 t10/sw, SSIM)={l3_free['clip']['top1']*100:.1f}/{l3_free['clip']['swap2x2']*100:.1f}/{l3_free['clip']['global500_top1']*100:.1f}%, "
            f"{l3_free['inception']['swap2x2']*100:.1f}%, {l3_free['alex5']['top1']*100:.1f}/{l3_free['alex5']['swap2x2']*100:.1f}%, {l3_free['ssim_diag']:.3f}",
            flush=True,
        )

    del model, cond_layer
    torch.cuda.empty_cache()
    return per_sub


@torch.inference_mode()
def build_swap_visual_gallery(
    ckpts: dict,
    res: dict,
    p2,
    vae32,
    adapter,
    proj,
    pe,
    ppe,
    lat50: torch.Tensor,
    gt_rgb50: torch.Tensor,
    z0_by_sub: dict,
    dev: str = "cuda:0",
):
    """Build deterministic visual 2x2 EEG-swap gallery on untouched held-out subjects (sub-10, sub-03, sub-04).
    Each row shows a real same-concept photo pair (A, B) under IDENTICAL fixed z0 & seed:
      [Photo A | Photo B | Arm0(E_A) | Arm0(E_B) | Arm1b(E_A) | Arm1b(E_B) | Arm3b(E_A) | Arm3b(E_B)]
    """
    meta = json.loads((PREP / "train_ids.json").read_text())
    sel50_cids = res["sel50_cids"]
    sel50_idx = res["sel50_idx"]

    # Pick 8 deterministic visually concrete concepts from sel50 (preferring vehicles/animals/everyday objects if present, else evenly spaced)
    c_names = []
    for ci, cid in enumerate(sel50_cids):
        j = int(sel50_idx[ci * 10])
        nm = Path(meta["paths"][j]).parent.name
        c_names.append((ci, int(cid), nm))
    print("  [GALLERY] sel50 sample concepts:", [x[2] for x in c_names[:15]], flush=True)

    # Select 8 diverse concepts evenly across sel50 + any bike/car/boat/animal if in sel50
    priority_kw = ["bike", "car", "boat", "truck", "bus", "dog", "cat", "bird", "horse", "chair", "guitar", "clock", "shoe", "hat"]
    chosen = []
    for ci, cid, nm in c_names:
        if any(kw in nm.lower() for kw in priority_kw) and len(chosen) < 5:
            chosen.append((ci, cid, nm))
    for ci in range(0, 50, 6):
        if len(chosen) >= 8:
            break
        if all(x[0] != ci for x in chosen):
            chosen.append(c_names[ci])

    # For each chosen concept, pick the most visually distinct pair (A, B) by GT pixel/latent L2 distance (zero cherry-picking on model outputs!)
    gt_flat = gt_rgb50.view(50, 10, -1)
    rows_spec = []
    ho_cycle = [10, 3, 4, 6, 7, 10, 3, 4]
    for r_i, (ci, cid, nm) in enumerate(chosen[:8]):
        sub = ho_cycle[r_i]
        # Pick pair (a, b) with max GT visual distance (e.g. different viewpoint/color/composition)
        g10 = gt_flat[ci]
        dmat = torch.cdist(g10, g10)
        flat_idx = int(dmat.argmax())
        ia, ib = divmod(flat_idx, 10)
        rows_spec.append((sub, ci, nm, ia, ib))

    arm_keys = ["Arm0_Base", "Arm1b_RepIntra_CondLyr", "Arm3b_RepRawSwap_CondLyr"]
    gen_imgs = {ak: [] for ak in arm_keys}

    for ak in arm_keys:
        pack = torch.load(ckpts[ak], map_location="cpu", weights_only=True)
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

        for sub, ci, nm, ia, ib in rows_spec:
            sid = sub - 1
            ga, gb = int(sel50_idx[ci * 10 + ia]), int(sel50_idx[ci * 10 + ib])
            X_raw = np.load(PREP / f"sub-{sub:02d}_train.npy")
            x_ab = torch.from_numpy(X_raw[[ga, gb]].mean(1)).float().to(dev)  # [2, 17, 250] R=4 exact!
            del X_raw
            fp, _, _, h_tr = forward_with_trunk(model, x_ab, sid, r_eff=4.0)
            c_ab = per_sample_cone(fp, R_tr, a_tr, b_tr, u_tr).to(torch.float16)
            ips_ab = proj(adapter(c_ab))
            if cl is not None:
                ips_ab = cl(h_tr, ips_ab)

            # IDENTICAL fixed start latent (concept-mean EEG z0) & IDENTICAL noise seed for A and B!
            z0_cm = z0_by_sub[sub][ci * 10:(ci + 1) * 10].mean(0, keepdim=True).expand(2, -1, -1, -1).to(dev, torch.float32)
            i0 = (vae32.decode(z0_cm[:1] / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).to(torch.float16).expand(2, -1, -1, -1)
            eps_shared = torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * ci + 99)).to(dev).expand(2, -1, -1, -1)
            gens = [torch.Generator(dev).manual_seed(1_000_003 * ci + 12345) for _ in range(2)]
            steps, smap_c = 4, 0.625
            hold_val = steps - steps * (1.0 - 0.5 * smap_c)

            def cb(pp, i_step, t_cur, ck):
                lt = ck["latents"].float()
                si = int(pp.scheduler.step_index)
                sig = pp.scheduler.sigmas[si].to(dev)
                a = max(0.0, min(1.0, hold_val - i_step))
                if a > 0:
                    lt = a * (z0_cm + sig * eps_shared) + (1.0 - a) * lt
                if i_step in (2, 3):
                    g_t = 0.22 if i_step == 2 else 0.04
                    lt = lt + g_t * _lp_op((z0_cm + sig * eps_shared) - lt, 8)
                ck["latents"] = lt.to(torch.float16)
                return ck

            lat_out = p2(
                image=i0,
                prompt_embeds=pe.expand(2, -1, -1),
                pooled_prompt_embeds=ppe.expand(2, -1),
                num_inference_steps=4,
                strength=1.0,
                guidance_scale=0.0,
                output_type="latent",
                generator=gens,
                cross_attention_kwargs={"ip_hidden_states": ips_ab},
                callback_on_step_end=cb,
                callback_on_step_end_tensor_inputs=["latents"],
            ).images
            imgs = (vae32.decode(lat_out.float() / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
            gen_imgs[ak].append((imgs[0], imgs[1]))
        del model, cl
        torch.cuda.empty_cache()

    # Render clean 8-column contact sheet: [GT A | GT B | Arm0 A | Arm0 B | Arm1b A | Arm1b B | Arm3b A | Arm3b B]
    cell = 192
    pad = 8
    hdr_h = 46
    row_lbl_h = 22
    W = 8 * cell + 9 * pad
    H = hdr_h + len(rows_spec) * (cell + row_lbl_h + pad) + pad
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    cols_title = ["GT Photo A", "GT Photo B", "Arm0 (E_A)", "Arm0 (E_B)", "Arm1b Rep (E_A)", "Arm1b Rep (E_B)", "Arm3b SwapDiff (E_A)", "Arm3b SwapDiff (E_B)"]
    draw.text((pad, 6), "Held-Out Subjects R=4 EEG 2x2 Swap Test (IDENTICAL fixed z0 & seed within each pair; 4-step SDXL-Turbo)", fill=(20, 20, 20))
    for c_i, t_str in enumerate(cols_title):
        draw.text((pad + c_i * (cell + pad) + 4, 26), t_str, fill=(30, 30, 120) if c_i >= 6 else (40, 40, 40))

    def _to_pil(t_img):
        arr = (t_img.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).resize((cell, cell), Image.BICUBIC)

    for r_i, (sub, ci, nm, ia, ib) in enumerate(rows_spec):
        y0 = hdr_h + r_i * (cell + row_lbl_h + pad)
        draw.text((pad, y0 + 2), f"sub-{sub:02d} (held-out) | concept: {nm} | photos #{ia} vs #{ib} (R=4 EEG, fixed z0 & seed)", fill=(20, 20, 20))
        y_img = y0 + row_lbl_h
        imgs_row = [
            gt_rgb50[ci * 10 + ia],
            gt_rgb50[ci * 10 + ib],
            gen_imgs["Arm0_Base"][r_i][0],
            gen_imgs["Arm0_Base"][r_i][1],
            gen_imgs["Arm1b_RepIntra_CondLyr"][r_i][0],
            gen_imgs["Arm1b_RepIntra_CondLyr"][r_i][1],
            gen_imgs["Arm3b_RepRawSwap_CondLyr"][r_i][0],
            gen_imgs["Arm3b_RepRawSwap_CondLyr"][r_i][1],
        ]
        for c_i, im in enumerate(imgs_row):
            canvas.paste(_to_pil(im), (pad + c_i * (cell + pad), y_img))

    out_jpg = OUT_DIR / "swap_gallery_heldout.jpg"
    canvas.save(out_jpg, quality=92)
    print(f"[GALLERY SAVED] {out_jpg}", flush=True)


def print_paired_stats_10sub(all_arms: dict):
    """Print All-10 and Untouched Held-Out-6 summary table & two-sided paired t-tests."""
    metrics_spec = [
        ("L1_F_top1", ["L1_327", "F", "top1"]),
        ("L2_unet_sw", ["L2_unet", "avg_3t", "swap2x2"]),
        ("L2_unet_dc10", ["L2_unet", "avg_3t", "top1_dc"]),
        ("L2_t499_sw", ["L2_unet", "t499", "swap2x2"]),
        ("L2_t249_sw", ["L2_unet", "t249", "swap2x2"]),
        ("FixZ0_clip_t10", ["L3_fixedZ0", "clip", "top1"]),
        ("FixZ0_clip_sw", ["L3_fixedZ0", "clip", "swap2x2"]),
        ("FixZ0_alex5_t10", ["L3_fixedZ0", "alex5", "top1"]),
        ("FixZ0_alex5_sw", ["L3_fixedZ0", "alex5", "swap2x2"]),
        ("FixZ0_pix_sw", ["L3_fixedZ0", "pixcorr", "swap2x2"]),
        ("Free_clip_t10", ["L3_freeGen", "clip", "top1"]),
        ("Free_clip_sw", ["L3_freeGen", "clip", "swap2x2"]),
        ("Free_clip_g500", ["L3_freeGen", "clip", "global500_top1"]),
        ("Free_incep_sw", ["L3_freeGen", "inception", "swap2x2"]),
        ("Free_alex5_t10", ["L3_freeGen", "alex5", "top1"]),
        ("Free_alex5_sw", ["L3_freeGen", "alex5", "swap2x2"]),
        ("Free_alex2_sw", ["L3_freeGen", "alex2", "swap2x2"]),
        ("Free_pix_sw", ["L3_freeGen", "pixcorr", "swap2x2"]),
        ("Free_ssim", ["L3_freeGen", "ssim_diag"]),
    ]

    def _vec(arm, path, subs):
        out = []
        for s in subs:
            d = all_arms[arm]["per_sub"][f"sub-{s:02d}"]
            for k in path:
                d = d[k]
            out.append(float(d))
        return np.array(out, dtype=np.float64)

    for split_name, subs in [("ALL-10", ALL_SUBS), ("HELDOUT-6", HELDOUT_SUBS), ("DEV-4", DEV_SUBS)]:
        print(f"\n================ {split_name} SUMMARY ({len(subs)} subjects, R=4 EEG, sealed fold-0) ================", flush=True)
        for arm in ["Arm0_Base", "Arm1_RepIntra_1400", "Arm1b_RepIntra_CondLyr", "Arm2v3_RawSwap_CondLyr", "Arm3b_RepRawSwap_CondLyr"]:
            vals = {m: float(_vec(arm, p, subs).mean()) for m, p in metrics_spec}
            print(
                f"  {arm:24s} | L2(sw/dc10)={vals['L2_unet_sw']*100:5.2f}/{vals['L2_unet_dc10']*100:5.2f}% | "
                f"FixZ0(CLIP t10/sw, A5 t10/sw, Px sw)={vals['FixZ0_clip_t10']*100:5.2f}/{vals['FixZ0_clip_sw']*100:5.2f}%, "
                f"{vals['FixZ0_alex5_t10']*100:5.2f}/{vals['FixZ0_alex5_sw']*100:5.2f}%, {vals['FixZ0_pix_sw']*100:5.2f}% | "
                f"Free(CLIP t10/sw/g500, Inc sw, A5 t10/sw, SSIM)={vals['Free_clip_t10']*100:5.2f}/{vals['Free_clip_sw']*100:5.2f}/{vals['Free_clip_g500']*100:4.2f}%, "
                f"{vals['Free_incep_sw']*100:5.2f}%, {vals['Free_alex5_t10']*100:5.2f}/{vals['Free_alex5_sw']*100:5.2f}%, {vals['Free_ssim']:.4f}",
                flush=True,
            )

    print("\n================ PAIRED TWO-SIDED T-TESTS (ALL-10 & UNTOUCHED HELDOUT-6) ================", flush=True)
    pairs_cmp = [
        ("Arm3b_RepRawSwap_CondLyr", "Arm0_Base"),
        ("Arm3b_RepRawSwap_CondLyr", "Arm1_RepIntra_1400"),
        ("Arm3b_RepRawSwap_CondLyr", "Arm1b_RepIntra_CondLyr"),
        ("Arm2v3_RawSwap_CondLyr", "Arm1b_RepIntra_CondLyr"),
    ]
    key_m = ["L2_unet_sw", "L2_unet_dc10", "FixZ0_clip_t10", "FixZ0_clip_sw", "FixZ0_alex5_t10", "FixZ0_alex5_sw", "FixZ0_pix_sw", "Free_clip_t10", "Free_clip_sw", "Free_incep_sw", "Free_alex5_t10", "Free_ssim"]
    m_map = dict(metrics_spec)
    stats_out = {}
    for a, b in pairs_cmp:
        stats_out[f"{a}_vs_{b}"] = {}
        print(f"\n--- {a} vs {b} ---", flush=True)
        for split_name, subs in [("ALL-10", ALL_SUBS), ("HELDOUT-6", HELDOUT_SUBS)]:
            stats_out[f"{a}_vs_{b}"][split_name] = {}
            for m in key_m:
                va = _vec(a, m_map[m], subs)
                vb = _vec(b, m_map[m], subs)
                diff = va - vb
                t_val, p_val = stats.ttest_rel(va, vb)
                wins = int((diff > 0).sum())
                stats_out[f"{a}_vs_{b}"][split_name][m] = {
                    "a": float(va.mean()), "b": float(vb.mean()), "d": float(diff.mean()),
                    "wins": wins, "n": len(subs), "t": float(t_val), "p": float(p_val),
                }
                if split_name == "ALL-10" or m in ("L2_unet_sw", "FixZ0_clip_sw", "FixZ0_alex5_sw", "FixZ0_pix_sw", "Free_clip_t10", "Free_clip_sw", "Free_incep_sw", "Free_ssim"):
                    sc = 100.0 if m != "Free_ssim" else 1.0
                    u = "%pp" if m != "Free_ssim" else ""
                    print(f"  [{split_name:9s}] {m:16s}: {va.mean()*sc:6.3f} vs {vb.mean()*sc:6.3f} | d={diff.mean()*sc:+6.3f}{u:3s} | wins={wins}/{len(subs)} | t={t_val:+6.2f}, p={p_val:.4f}", flush=True)
    return stats_out


def main():
    dev = "cuda:0"
    t_all = time.time()
    res = build_tr14_resources(dev=dev)

    ckpts = {
        "Arm0_Base": OUT_DIR / "arm0_tr14_xbrain_s2025.pt",
        "Arm1_RepIntra_1400": OUT_DIR / "arm1_rep_intra_st1400_s2025.pt",
        "Arm2v3_RawSwap_CondLyr": OUT_DIR / "arm2v3_rawswap_condlyr_st600_s2025.pt",
        "Arm3b_RepRawSwap_CondLyr": OUT_DIR / "arm3b_rep_rawswap_condlyr_st600_s2025.pt",
    }
    arm1_800 = OUT_DIR / "arm1_rep_intra_st800_s2025.pt"
    ckpts["Arm1b_RepIntra_CondLyr"] = train_arm1b_rep_condlyr(arm1_800, res, steps=600, seed=2025, dev=dev)

    # Fit tr14-only ridge z0 for ALL 10 subjects (cached for 2,5,8,9; fits 1,3,4,6,7,10 in ~90s!)
    z0_by_sub = {sub: fit_tr14_ridge_z0(sub, res["tr_idx_np"], res["sel50_idx"], dev=dev) for sub in ALL_SUBS}

    # Load fp16 OfficialTurboRunner pipeline once
    from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
    from metrics import _nets

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

        # Pre-decode init_fix & init_free ONCE per subject (saves 80% of VAE decode time!)
        init_fix_by_sub, init_free_by_sub = {}, {}
        for sub in ALL_SUBS:
            z0_e = z0_by_sub[sub]
            z0_cm = z0_e.view(50, 10, 4, 64, 64).mean(dim=1, keepdim=True).expand(-1, 10, -1, -1, -1).reshape(500, 4, 64, 64)
            init_free_by_sub[sub] = torch.cat([
                (vae32.decode(z0_e[j:j + 20].to(dev, torch.float32) / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
                for j in range(0, 500, 20)
            ])
            # Concept-mean only has 50 unique latents! Decode 50 and repeat_interleave(10) -> 10x faster!
            z0_cm50 = z0_e.view(50, 10, 4, 64, 64).mean(dim=1)
            rgb50 = torch.cat([
                (vae32.decode(z0_cm50[j:j + 25].to(dev, torch.float32) / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
                for j in range(0, 50, 25)
            ])
            init_fix_by_sub[sub] = rgb50.repeat_interleave(10, dim=0)

    # Load existing dev-4 results and expand cleanly to full 10-subject master file
    dev_p = OUT_DIR / "exp1_4arm_results.json"
    full_p = OUT_DIR / "exp4_10sub_master_results.json"
    if full_p.exists():
        master = json.loads(full_p.read_text())
    else:
        prev = json.loads(dev_p.read_text()) if dev_p.exists() else {"arms": {}}
        master = {"arms": {k: {"per_sub": dict(v["per_sub"])} for k, v in prev["arms"].items() if k in ckpts}}

    for arm_name in ["Arm1b_RepIntra_CondLyr", "Arm0_Base", "Arm1_RepIntra_1400", "Arm2v3_RawSwap_CondLyr", "Arm3b_RepRawSwap_CondLyr"]:
        master["arms"].setdefault(arm_name, {"per_sub": {}})
        missing_subs = [s for s in ALL_SUBS if f"sub-{s:02d}" not in master["arms"][arm_name]["per_sub"]]
        if not missing_subs:
            print(f"[CACHED 10-SUB] {arm_name}", flush=True)
            continue
        new_res = evaluate_arm_subs_fast(
            arm_name, ckpts[arm_name], missing_subs, res, pipe, p2, vae32, adapter, proj, pe, ppe, atid,
            lat50, gt_rgb50, gt_feats, z0_by_sub, init_fix_by_sub, init_free_by_sub, dev=dev,
        )
        master["arms"][arm_name]["per_sub"].update(new_res)
        master["wall_s"] = round(time.time() - t_all, 1)
        write_json(full_p, master)

    # Print full 10-subject & Held-Out-6 paired significance tests
    master["paired_stats"] = print_paired_stats_10sub(master["arms"])
    write_json(full_p, master)

    # Build visual 2x2 EEG-swap comparison gallery on untouched held-out subjects
    build_swap_visual_gallery(ckpts, res, p2, vae32, adapter, proj, pe, ppe, lat50, gt_rgb50, z0_by_sub, dev=dev)
    print(f"=== PHASE 4 (FULL 10-SUBJECT CONFIRMATION + VISUAL SWAP GALLERY) DONE in {round(time.time()-t_all, 1)}s ===", flush=True)


if __name__ == "__main__":
    main()
