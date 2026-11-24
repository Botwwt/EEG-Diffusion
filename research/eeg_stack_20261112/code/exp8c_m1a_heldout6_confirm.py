"""Method 1 Held-Out-6 Confirmation (exp8c): Zero-Parameter Dual-Stream Trial-Precision Aggregation (M1a).

Evaluates M1a_PrecDual (inv_ch_rms2(p=0.5) ERP + per-trial feature fusion + precision-weighted z0_ridge,
ZERO learned parameters / ZERO retraining) on the 6 completely untouched Held-Out subjects {1, 3, 4, 6, 7, 10}
for:
  1. M1a_on_SwapDiff_Arm3b  (direct paired test vs naive-mean SwapDiff_Arm3b & Arm1b!)
  2. M1a_on_D2_Compete4     (plug-and-play synergy with 4-step DRaFT-Compete4!)
Also computes All-10 (N=10) & Untouched Held-Out-6 (N=6) paired two-sided t-tests vs naive-mean baselines
and F-rep 3,270 unseen fold-0 representation Intra10/Swap across all 10 subjects.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

ROOT = Path(os.environ["RAD_ROOT"])
for d in [ROOT / "code/exp_w7", ROOT / "code/exp_w6", ROOT / "code/rad", ROOT / "code/exp", ROOT / "code/Brain-HIVE"]:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

from common import PREP, PRIOR, TURBO, write_json
from exp0_intra10_swap_gate import ALL_SUBS, DEV_SUBS, OUT_DIR, get_concept_splits, intra10_and_swap_from_sim
from exp1_swapdiff_3arm import ResidualIPCondLayer, per_sample_cone
from exp5_z0_2x2_diag import extract_eval_feats, score_full500
from exp6_draft_intra4 import DRAFT_DIR, forward_branches_and_trunk, rollout_4step_step_ips
from exp7_heldout6_confirm import HELDOUT_SUBS, extract_metric
from exp8_trial_precision_swapdiff import M1_DIR, fit_tr14_ridge_z0_prec, prec_erp_analytic
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder

from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel

nz = F.normalize


@torch.inference_mode()
def eval_m1a_on_ckpt(
    tag: str,
    ckpt_p: Path,
    subs,
    sel50_idx,
    f0_idx,
    F_f0,
    z0_prec_by_sub,
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
    compute_frep: bool = True,
    dev: str = "cuda:0",
):
    pack = torch.load(ckpt_p, map_location="cpu", weights_only=True)
    sd_b = pack["state_dict"] if "state_dict" in pack else torch.load(pack["backbone_ckpt"], map_location="cpu", weights_only=True)["state_dict"]
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in sd_b.items()})
    model.eval()
    base_cond = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev)
    base_cond.load_state_dict({k: v.float() for k, v in pack["cond_layer"].items()})
    base_cond.eval()
    R_tr, a_tr, b_tr = pack["cone"]["R"], pack["cone"]["alpha"], pack["cone"]["beta"]
    u_tr = pack["cone"]["u_tr"].to(dev).float()

    def encode_dual(x4_cpu, sid):
        fps, hs = [], []
        for i in range(0, len(x4_cpu), 64):
            xb = x4_cpu[i:i + 64].float().to(dev)
            x_pr = prec_erp_analytic(xb, p=0.5)
            fp_e, h_e, _ = forward_branches_and_trunk(model, x_pr, sid, r_eff=4.0)
            B, R, C, T = xb.shape
            fp_t, h_t, _ = forward_branches_and_trunk(model, xb.reshape(B * R, C, T), sid, r_eff=1.0)
            fp = 0.72 * fp_e + 0.28 * fp_t.view(B, R, -1).mean(1)
            h = 0.72 * h_e + 0.28 * h_t.view(B, R, -1).mean(1)
            fps.append(fp.cpu())
            hs.append(h.cpu())
        return torch.cat(fps, 0), torch.cat(hs, 0)

    out = {}
    if compute_frep:
        Fg = nz(F_f0 - F_f0.mean(0, keepdim=True), dim=1).view(-1, 10, F_f0.shape[1])
        rep_naive, rep_m1a = {}, {}
        for sub in subs:
            sid = sub - 1
            X4_f0 = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_train.npy")[f0_idx])
            # Naive mean
            fps_n = [forward_branches_and_trunk(model, X4_f0[i:i + 256].float().mean(1).to(dev), sid, r_eff=4.0)[0].cpu() for i in range(0, len(X4_f0), 256)]
            fp_n = torch.cat(fps_n, 0)
            Fp_n = nz(fp_n - fp_n.mean(0, keepdim=True), dim=1).view(-1, 10, fp_n.shape[1])
            rep_naive[f"sub-{sub:02d}"] = intra10_and_swap_from_sim(torch.bmm(Fp_n, Fg.transpose(1, 2)).numpy())
            # M1a dual
            fp_m, _ = encode_dual(X4_f0, sid)
            Fp_m = nz(fp_m - fp_m.mean(0, keepdim=True), dim=1).view(-1, 10, fp_m.shape[1])
            rep_m1a[f"sub-{sub:02d}"] = intra10_and_swap_from_sim(torch.bmm(Fp_m, Fg.transpose(1, 2)).numpy())
        out["rep_f0_naive"] = rep_naive
        out["rep_f0_m1a"] = rep_m1a
        nt10 = np.mean([v["top1"] for v in rep_naive.values()]) * 100
        mt10 = np.mean([v["top1"] for v in rep_m1a.values()]) * 100
        nsw = np.mean([v["swap2x2"] for v in rep_naive.values()]) * 100
        msw = np.mean([v["swap2x2"] for v in rep_m1a.values()]) * 100
        print(f">>> [{tag:26s} F-REP 3270] Naive={nt10:5.2f}%/{nsw:5.2f}% -> M1a={mt10:5.2f}%/{msw:5.2f}% ({mt10-nt10:+.2f}pp/{msw-nsw:+.2f}pp)", flush=True)

    cids_500 = [i // 10 for i in range(500)]
    eps_map = {ci: torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * ci + 99)) for ci in range(50)}
    sn_map = {
        ci: [torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(1_000_003 * ci + 12345 + 17 * st)) for st in range(4)]
        for ci in range(50)
    }

    for rtag, smap_c, lk in [("s375", 0.375, (8, 0.11, 0.02)), ("s625", 0.625, (8, 0.22, 0.04))]:
        per_sub = {}
        for sub in subs:
            sid = sub - 1
            X4_50 = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_train.npy")[sel50_idx])
            fp50, h50 = encode_dual(X4_50, sid)
            ips_l = []
            for i in range(0, 500, 100):
                c_cone = per_sample_cone(fp50[i:i + 100].to(dev), R_tr, a_tr, b_tr, u_tr).to(torch.float16)
                ips_p = proj(adapter(c_cone))
                ips_l.append(base_cond(h50[i:i + 100].to(dev), ips_p).cpu())
            ips_all = torch.cat(ips_l, 0)
            z0_e = z0_prec_by_sub[sub]

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
        out[rtag] = {"mean": m_avg, "per_sub": per_sub}
        print(
            f">>> [{tag:26s} {rtag}] "
            f"CLIP t10={m_avg['clip_t10']*100:5.2f}% dc10={m_avg['clip_dc10']*100:5.2f}% sw={m_avg['clip_sw']*100:5.2f}% g500={m_avg['clip_g500']*100:4.2f}% | "
            f"Inc sw={m_avg['incep_sw']*100:5.2f}% | A5 t10={m_avg['alex5_t10']*100:5.2f}% sw={m_avg['alex5_sw']*100:5.2f}% | "
            f"Px sw={m_avg['pix_sw']*100:5.2f}% | SSIM={m_avg['ssim']:.4f}",
            flush=True,
        )
    del model, base_cond
    torch.cuda.empty_cache()
    return out


def main():
    dev = "cuda:0"
    t0 = time.time()
    concept_np, ok_np, tr_idx_np, f0_all10, f0_idx, sel50_cids, sel50_idx = get_concept_splits()
    # Precompute & cache precision z0_ridge for ALL 10 subjects (Dev-4 already cached!)
    z0_prec_all = {sub: fit_tr14_ridge_z0_prec(sub, tr_idx_np, sel50_idx, dev=dev, p=0.5) for sub in ALL_SUBS}

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

    lat50 = torch.from_numpy(np.load(PREP / "lat512_train.npy")[sel50_idx]).float()
    F_f0 = torch.from_numpy(np.load(PREP / "F_train.npy")[f0_idx]).float()
    with torch.inference_mode():
        gt_rgb50 = torch.cat([
            (vae32.decode(lat50[j:j + 20].to(dev, torch.float32) / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
            for j in range(0, 500, 20)
        ])
        nets = _nets()
        gt_feats = extract_eval_feats(gt_rgb50, nets, dev=dev)

    arm3b_ckpt = OUT_DIR / "arm3b_rep_rawswap_condlyr_st600_s2025.pt"
    d2_ckpt = DRAFT_DIR / "d2_compete4_cond_st120_s2025.pt"

    # Evaluate M1a_on_Arm3b on Held-Out-6 (Dev-4 already in exp8) + F-rep on ALL 10 subs!
    ho6_m1a_arm3b = eval_m1a_on_ckpt(
        "HO6 M1a_on_SwapDiff", arm3b_ckpt, HELDOUT_SUBS, sel50_idx, f0_idx, F_f0, z0_prec_all,
        pipe.unet, pipe.scheduler, vae32, adapter, proj, pe, ppe, atid, gt_rgb50, gt_feats, nets,
        compute_frep=True, dev=dev,
    )
    # Also evaluate M1a_on_D2_Compete4 on ALL 10 subjects (fast ~12 min!)
    all10_m1a_d2 = eval_m1a_on_ckpt(
        "ALL10 M1a_on_D2_Compete4", d2_ckpt, ALL_SUBS, sel50_idx, f0_idx, F_f0, z0_prec_all,
        pipe.unet, pipe.scheduler, vae32, adapter, proj, pe, ppe, atid, gt_rgb50, gt_feats, nets,
        compute_frep=False, dev=dev,
    )

    # Merge Dev-4 (exp8) + Held-Out-6 -> All-10 M1a_on_SwapDiff and compare paired vs exp7 baselines!
    d7 = json.loads((DRAFT_DIR / "exp7_heldout6_and_all10_results.json").read_text())
    d8 = json.loads((M1_DIR / "exp8_m1_trial_prec_dev4_results.json").read_text())

    m1a_arm3b_all10 = {}
    for rtag in ("s375", "s625"):
        merged = {}
        merged.update(d8["arms"]["M1a_ZeroRetrain_PrecDual"][rtag]["per_sub"])
        merged.update(ho6_m1a_arm3b[rtag]["per_sub"])
        m1a_arm3b_all10[rtag] = merged

    def paired_stats(ps_a, ps_b, slist):
        keys = [f"sub-{s:02d}" for s in slist]
        res = {}
        for mk in ["clip_t10", "clip_dc10", "clip_sw", "clip_g500", "incep_sw", "alex5_t10", "alex5_sw", "pix_sw", "ssim"]:
            va = np.array([extract_metric(ps_a[k], mk) for k in keys])
            vb = np.array([extract_metric(ps_b[k], mk) for k in keys])
            diff = va - vb
            t_s, p_v = stats.ttest_rel(va, vb)
            res[mk] = {"mean_a": float(va.mean()), "mean_b": float(vb.mean()), "delta": float(diff.mean()), "wins": int((diff > 0).sum()), "n": len(keys), "p": float(p_v)}
        return res

    ttests = {}
    for split, slist in [("heldout6", HELDOUT_SUBS), ("all10", ALL_SUBS)]:
        ttests[split] = {}
        for rtag in ("s375", "s625"):
            ttests[split][rtag] = {
                "M1a_Arm3b_vs_Naive_Arm3b": paired_stats(m1a_arm3b_all10[rtag], d7["all10_per_sub"]["SwapDiff_Arm3b"][rtag], slist),
                "M1a_Arm3b_vs_Naive_Arm1b": paired_stats(m1a_arm3b_all10[rtag], d7["all10_per_sub"]["Arm1b_RepIntra_CondLyr"][rtag], slist),
                "M1a_D2_vs_Naive_D2": paired_stats(all10_m1a_d2[rtag]["per_sub"], d7["all10_per_sub"]["D2_Compete4_Cond"][rtag], slist),
                "M1a_D2_vs_Naive_Arm1b": paired_stats(all10_m1a_d2[rtag]["per_sub"], d7["all10_per_sub"]["Arm1b_RepIntra_CondLyr"][rtag], slist),
            }

    out_p = M1_DIR / "exp8c_m1a_heldout6_and_all10_results.json"
    write_json(out_p, {
        "wall_s": round(time.time() - t0, 1),
        "ho6_rep_f0_naive": ho6_m1a_arm3b["rep_f0_naive"],
        "ho6_rep_f0_m1a": ho6_m1a_arm3b["rep_f0_m1a"],
        "m1a_arm3b_all10": m1a_arm3b_all10,
        "m1a_d2_all10": {rtag: all10_m1a_d2[rtag]["per_sub"] for rtag in ("s375", "s625")},
        "ttests": ttests,
    })
    print(f"=== EXP 8C (M1a HELD-OUT-6 & ALL-10 CONFIRM) DONE in {round(time.time()-t0, 1)}s -> {out_p} ===", flush=True)


if __name__ == "__main__":
    main()
