"""Final Confirmation (exp9b): Held-Out-6 & All-10 Matrix for Method 2 (DINOv2) and Method 1 + Method 2 Combo + Visual Gallery.

Evaluates on all 10 subjects at s375 (primary operating point):
  - M2a_DINOv2_Naive (Held-Out-6, merging Dev-4 from exp9 -> All-10)
  - M1a_on_M2a_DINOv2 (All-10: combining Method 1 Trial-Precision Dual-Stream + Method 2 DINOv2 geometric DRaFT!)
Also builds an uncherry-picked Held-Out sub-01 s375 comparison gallery:
  [GT | Naive Arm1b | M1a+Arm1b | Naive SwapDiff | M1a+SwapDiff | M1a+D2_Compete4 | M1a+M2a_DINOv2]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import stats

ROOT = Path(os.environ["RAD_ROOT"])
for d in [ROOT / "code/exp_w7", ROOT / "code/exp_w6", ROOT / "code/rad", ROOT / "code/exp", ROOT / "code/Brain-HIVE"]:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

from common import PREP, PRIOR, TURBO, write_json
from exp0_intra10_swap_gate import ALL_SUBS, DEV_SUBS, OUT_DIR, fit_tr14_ridge_z0, get_concept_splits
from exp1_swapdiff_3arm import ResidualIPCondLayer, per_sample_cone
from exp5_z0_2x2_diag import extract_eval_feats, score_full500
from exp6_draft_intra4 import DRAFT_DIR, forward_branches_and_trunk, rollout_4step_step_ips
from exp7_heldout6_confirm import HELDOUT_SUBS, extract_metric
from exp8_trial_precision_swapdiff import M1_DIR, fit_tr14_ridge_z0_prec, prec_erp_analytic
from exp9_dinov2_geom_draft import M2_DIR
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder

from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel


@torch.inference_mode()
def run_s375_eval(
    tag: str,
    ckpt_p: Path,
    use_m1a: bool,
    subs,
    sel50_idx,
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
    gallery_sub: int = 1,
    gallery_indices=None,
    gallery_store: dict = None,
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

    cids_500 = [i // 10 for i in range(500)]
    eps_map = {ci: torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * ci + 99)) for ci in range(50)}
    sn_map = {
        ci: [torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(1_000_003 * ci + 12345 + 17 * st)) for st in range(4)]
        for ci in range(50)
    }

    per_sub = {}
    for sub in subs:
        sid = sub - 1
        X4_50 = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_train.npy")[sel50_idx])
        fps, hs = [], []
        for i in range(0, 500, 64):
            xb = X4_50[i:i + 64].float().to(dev)
            if use_m1a:
                x_pr = prec_erp_analytic(xb, p=0.5)
                fp_e, h_e, _ = forward_branches_and_trunk(model, x_pr, sid, r_eff=4.0)
                B, R, C, T = xb.shape
                fp_t, h_t, _ = forward_branches_and_trunk(model, xb.reshape(B * R, C, T), sid, r_eff=1.0)
                fp = 0.72 * fp_e + 0.28 * fp_t.view(B, R, -1).mean(1)
                h = 0.72 * h_e + 0.28 * h_t.view(B, R, -1).mean(1)
            else:
                fp, h, _ = forward_branches_and_trunk(model, xb.mean(1), sid, r_eff=4.0)
            fps.append(fp.cpu())
            hs.append(h.cpu())
        fp50, h50 = torch.cat(fps, 0), torch.cat(hs, 0)
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
                pe, ppe, atid, 0.375, (8, 0.11, 0.02),
            )
            for j0 in range(0, len(q), 10):
                img_b = (vae32.decode(lat_out[j0:j0 + 10].float() / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
                rec[s0 + j0:s0 + j0 + len(img_b)] = img_b
        if gallery_store is not None and sub == gallery_sub and gallery_indices is not None:
            gallery_store[tag] = rec[gallery_indices].clone()
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
    print(
        f">>> [{tag:28s} s375] "
        f"CLIP t10={m_avg['clip_t10']*100:5.2f}% dc10={m_avg['clip_dc10']*100:5.2f}% sw={m_avg['clip_sw']*100:5.2f}% g500={m_avg['clip_g500']*100:4.2f}% | "
        f"Inc sw={m_avg['incep_sw']*100:5.2f}% | A5 t10={m_avg['alex5_t10']*100:5.2f}% sw={m_avg['alex5_sw']*100:5.2f}% | "
        f"Px sw={m_avg['pix_sw']*100:5.2f}% | SSIM={m_avg['ssim']:.4f}",
        flush=True,
    )
    del model, base_cond
    torch.cuda.empty_cache()
    return {"mean": m_avg, "per_sub": per_sub}


def build_m1_m2_gallery(gt_rgb50: torch.Tensor, gallery_store: dict, gallery_indices: list, out_jpg: Path):
    cols = [
        ("GT Photo", None),
        ("Naive Arm1b", "Naive_Arm1b"),
        ("M1a + Arm1b", "M1a_Arm1b"),
        ("Naive SwapDiff", "Naive_SwapDiff"),
        ("M1a + SwapDiff", "M1a_SwapDiff"),
        ("M1a + D2_Comp4", "M1a_D2_Compete4"),
        ("M1a + M2a_DINO", "M1a_M2a_DINOv2"),
    ]
    tile = 176
    pad = 7
    header_h = 32
    label_w = 104
    W = label_w + len(cols) * (tile + pad) + pad
    H = header_h + len(gallery_indices) * (tile + pad) + pad
    canvas = Image.new("RGB", (W, H), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    for c_i, (title, _) in enumerate(cols):
        x0 = label_w + pad + c_i * (tile + pad)
        draw.text((x0 + 4, 9), title, fill=(20, 20, 20))
    for r_i, idx in enumerate(gallery_indices):
        y0 = header_h + pad + r_i * (tile + pad)
        draw.text((8, y0 + tile // 2 - 12), f"Concept #{idx//10:02d}\n{'Photo A' if idx%10==0 else 'Photo B'}", fill=(30, 30, 30))
        for c_i, (_, key) in enumerate(cols):
            x0 = label_w + pad + c_i * (tile + pad)
            ten = gt_rgb50[idx] if key is None else gallery_store[key][r_i]
            arr = (ten.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            canvas.paste(Image.fromarray(arr).resize((tile, tile), Image.BICUBIC), (x0, y0))
        if r_i % 2 == 1 and r_i < len(gallery_indices) - 1:
            draw.line([(4, y0 + tile + pad // 2), (W - 4, y0 + tile + pad // 2)], fill=(170, 170, 170), width=2)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_jpg, quality=92)
    print(f"[SAVED GALLERY] {out_jpg} ({W}x{H})", flush=True)


def main():
    dev = "cuda:0"
    t0 = time.time()
    _, _, tr_idx_np, _, _, _, sel50_idx = get_concept_splits()
    z0_naive_all = {sub: fit_tr14_ridge_z0(sub, tr_idx_np, sel50_idx, dev=dev) for sub in ALL_SUBS}
    z0_prec_all = {sub: fit_tr14_ridge_z0_prec(sub, tr_idx_np, sel50_idx, dev=dev, p=0.5) for sub in ALL_SUBS}

    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    from metrics import _nets

    dt = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=dt, variant="fp16")
    IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(pipe.unet)
    pipe.to(dev, dt); pipe.unet.eval().requires_grad_(False); pipe.scheduler.set_timesteps(4, device=dev)
    with torch.inference_mode():
        pe, _, ppe, _ = pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
        atid = pipe._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=dt, text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).to(dev)
    del pipe.text_encoder, pipe.text_encoder_2, pipe.vae
    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, dt).eval().requires_grad_(False)
    proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, dt).eval().requires_grad_(False)
    vae32 = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval().requires_grad_(False)
    vae32.enable_slicing()

    lat50 = torch.from_numpy(np.load(PREP / "lat512_train.npy")[sel50_idx]).float()
    with torch.inference_mode():
        gt_rgb50 = torch.cat([(vae32.decode(lat50[j:j + 20].to(dev, torch.float32) / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu() for j in range(0, 500, 20)])
        nets = _nets()
        gt_feats = extract_eval_feats(gt_rgb50, nets, dev=dev)

    arm1b_ckpt = OUT_DIR / "arm1b_rep_condlyr_st600_s2025.pt"
    arm3b_ckpt = OUT_DIR / "arm3b_rep_rawswap_condlyr_st600_s2025.pt"
    d2_ckpt = DRAFT_DIR / "d2_compete4_cond_st120_s2025.pt"
    m2a_ckpt = M2_DIR / "m2a_dinov2_geom_cond_st200_s2025.pt"

    gallery_indices = [c * 10 + p for c in range(6) for p in (0, 1)]
    gallery_store = {}

    # 1. Held-Out-6 Naive M2a_DINOv2
    ho6_m2a = run_s375_eval(
        "HO6 Naive_M2a_DINOv2", m2a_ckpt, False, HELDOUT_SUBS, sel50_idx, z0_naive_all,
        pipe.unet, pipe.scheduler, vae32, adapter, proj, pe, ppe, atid, gt_rgb50, gt_feats, nets, dev=dev,
    )
    # 2. All-10 M1a + M2a_DINOv2 (also captures sub-01 gallery!)
    all10_m1a_m2a = run_s375_eval(
        "M1a_M2a_DINOv2", m2a_ckpt, True, ALL_SUBS, sel50_idx, z0_prec_all,
        pipe.unet, pipe.scheduler, vae32, adapter, proj, pe, ppe, atid, gt_rgb50, gt_feats, nets,
        gallery_sub=1, gallery_indices=gallery_indices, gallery_store=gallery_store, dev=dev,
    )
    # 3. Quick sub-01-only renders for the remaining 5 gallery columns (~30s each!)
    for gtag, ck, use_m1, zmap in [
        ("Naive_Arm1b", arm1b_ckpt, False, z0_naive_all),
        ("M1a_Arm1b", arm1b_ckpt, True, z0_prec_all),
        ("Naive_SwapDiff", arm3b_ckpt, False, z0_naive_all),
        ("M1a_SwapDiff", arm3b_ckpt, True, z0_prec_all),
        ("M1a_D2_Compete4", d2_ckpt, True, z0_prec_all),
    ]:
        run_s375_eval(
            gtag, ck, use_m1, [1], sel50_idx, zmap,
            pipe.unet, pipe.scheduler, vae32, adapter, proj, pe, ppe, atid, gt_rgb50, gt_feats, nets,
            gallery_sub=1, gallery_indices=gallery_indices, gallery_store=gallery_store, dev=dev,
        )

    gal_p = M2_DIR / "m1_m2_heldout_sub01_s375_gallery.jpg"
    build_m1_m2_gallery(gt_rgb50, gallery_store, gallery_indices, gal_p)

    # Merge Dev-4 (exp9) + Held-Out-6 -> All-10 Naive M2a_DINOv2
    d9 = json.loads((M2_DIR / "exp9_m2_dinov2_geom_dev4_results.json").read_text())
    m2a_naive_all10 = {}
    m2a_naive_all10.update(d9["arms"]["m2a_dinov2_geom_cond"]["s375"]["per_sub"])
    m2a_naive_all10.update(ho6_m2a["per_sub"])

    d7 = json.loads((DRAFT_DIR / "exp7_heldout6_and_all10_results.json").read_text())
    d8c = json.loads((M1_DIR / "exp8c_m1a_heldout6_and_all10_results.json").read_text())

    def summarize(ps, slist):
        keys = [f"sub-{s:02d}" for s in slist]
        return {mk: float(np.mean([extract_metric(ps[k], mk) for k in keys])) for mk in ["clip_t10", "clip_dc10", "clip_sw", "clip_g500", "incep_sw", "alex5_t10", "alex5_sw", "pix_sw", "ssim"]}

    def paired(ps_a, ps_b, slist):
        keys = [f"sub-{s:02d}" for s in slist]
        out = {}
        for mk in ["clip_t10", "clip_dc10", "clip_sw", "clip_g500", "incep_sw", "alex5_t10", "alex5_sw", "ssim"]:
            va = np.array([extract_metric(ps_a[k], mk) for k in keys])
            vb = np.array([extract_metric(ps_b[k], mk) for k in keys])
            t_s, p_v = stats.ttest_rel(va, vb)
            out[mk] = {"mean_a": float(va.mean()), "mean_b": float(vb.mean()), "delta": float((va - vb).mean()), "wins": int(((va - vb) > 0).sum()), "n": len(keys), "p": float(p_v)}
        return out

    res_out = {
        "wall_s": round(time.time() - t0, 1),
        "summary": {
            "heldout6": {
                "M2a_DINOv2_Naive": summarize(m2a_naive_all10, HELDOUT_SUBS),
                "M1a_M2a_DINOv2": summarize(all10_m1a_m2a["per_sub"], HELDOUT_SUBS),
            },
            "all10": {
                "M2a_DINOv2_Naive": summarize(m2a_naive_all10, ALL_SUBS),
                "M1a_M2a_DINOv2": summarize(all10_m1a_m2a["per_sub"], ALL_SUBS),
            },
        },
        "ttests": {
            split: {
                "M2a_Naive_vs_Arm3b_Naive": paired(m2a_naive_all10, d7["all10_per_sub"]["SwapDiff_Arm3b"]["s375"], slist),
                "M1a_M2a_vs_M2a_Naive": paired(all10_m1a_m2a["per_sub"], m2a_naive_all10, slist),
                "M1a_M2a_vs_Arm1b_Naive": paired(all10_m1a_m2a["per_sub"], d7["all10_per_sub"]["Arm1b_RepIntra_CondLyr"]["s375"], slist),
                "M1a_M2a_vs_M1a_D2": paired(all10_m1a_m2a["per_sub"], d8c["m1a_d2_all10"]["s375"], slist),
            }
            for split, slist in [("heldout6", HELDOUT_SUBS), ("all10", ALL_SUBS)]
        },
        "m2a_naive_all10": m2a_naive_all10,
        "m1a_m2a_all10": all10_m1a_m2a["per_sub"],
    }
    out_p = M2_DIR / "exp9b_m2_and_m1m2_all10_results.json"
    write_json(out_p, res_out)
    print(f"=== EXP 9B (M2 & M1+M2 ALL-10 CONFIRM + GALLERY) DONE in {res_out['wall_s']}s -> {out_p} ===", flush=True)


if __name__ == "__main__":
    main()
