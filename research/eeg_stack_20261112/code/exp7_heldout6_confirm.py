"""Step 3 (exp7): Untouched Held-Out-6 Confirmation & All-10 Statistical Matrix + Uncherry-Picked Visual Gallery.

Evaluates the 6 core arms on the 6 completely untouched Held-Out subjects {1, 3, 4, 6, 7, 10}
under BOTH anchoring regimes (s375 and s625, 4-NFE nsel=1, R=4 FreeGen, sealed fold-0 unseen concepts):
  1. Arm0_Base               : O13/O14 shared-trunk baseline
  2. Arm1b_RepIntra_CondLyr  : Architecture- & param-matched representation-level intra-concept control
  3. SwapDiff_Arm3b          : Single-step 2x2 EEG-swap denoising contrast
  4. D1_SimOnly_Cond         : 4-step DRaFT positive similarity ONLY (NO competition)
  5. D2_Compete4_Cond        : 4-step DRaFT within-concept 4-way competition (cond_layer only)
  6. D4_UnfreezeEnc_StepZ0   : Best Dev-4 DRaFT-Compete4 arm (unfrozen shared EEG trunk + step-routed cond + PCA-256 Delta_z0)

Also computes exact All-10 (N=10) and Untouched Held-Out-6 (N=6) paired two-sided t-tests + Holm-Bonferroni
adjusted p-values across strictly independent evaluators (OpenAI CLIP ViT-L/14 primary Intra10 Top-1,
dc10, Swap, g500; Inception-v3; AlexNet-5; PixCorr; SSIM) and generates an uncherry-picked Held-Out visual gallery.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy import stats

ROOT = Path(os.environ["RAD_ROOT"])
for d in [ROOT / "code/exp_w7", ROOT / "code/exp_w6", ROOT / "code/rad", ROOT / "code/exp", ROOT / "code/Brain-HIVE"]:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

from common import EMB_KEYS, PREP, PRIOR, TURBO, write_json
from exp0_intra10_swap_gate import ALL_SUBS, DEV_SUBS, OUT_DIR, fit_tr14_ridge_z0, get_concept_splits, intra10_and_swap_from_sim
from exp1_swapdiff_3arm import ResidualIPCondLayer, per_sample_cone
from exp5_z0_2x2_diag import extract_eval_feats, score_full500
from exp6_draft_intra4 import DRAFT_DIR, forward_branches_and_trunk, rollout_4step_step_ips
from exp6b_draft_v2_dev4 import StepRouteZ0DualCA, rollout_4step_dualca
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder

from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel

nz = F.normalize
HELDOUT_SUBS = [s for s in ALL_SUBS if s not in DEV_SUBS]  # [1, 3, 4, 6, 7, 10]


def load_arm_components(arm_kind: str, ckpt_p: Path, z0_oof_pack: dict, dev: str):
    """Load (model, base_cond, head_mod, cone) uniformly across Arm0/Arm1b/Arm3b/D1/D2/D4."""
    pack = torch.load(ckpt_p, map_location="cpu", weights_only=True)
    if "state_dict" in pack:
        sd_backbone = pack["state_dict"]
    else:
        sd_backbone = torch.load(pack["backbone_ckpt"], map_location="cpu", weights_only=True)["state_dict"]

    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in sd_backbone.items()})
    if pack.get("enc_shared") is not None:
        es = pack["enc_shared"]
        model.block.load_state_dict({k: v.float() for k, v in es["block"].items()})
        model.head_f.load_state_dict({k: v.float() for k, v in es["head_f"].items()})
        model.horizon_logits.copy_(es["horizon_logits"].float().to(dev))
    model.eval()

    base_cond = None
    head_mod = None
    if arm_kind == "v2_head":
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
    elif pack.get("cond_layer") is not None:
        base_cond = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev)
        base_cond.load_state_dict({k: v.float() for k, v in pack["cond_layer"].items()})
        base_cond.eval()

    return model, base_cond, head_mod, pack["cone"]


@torch.inference_mode()
def eval_arm_on_subs(
    arm_name: str,
    arm_kind: str,
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
    h14_eval,
    H14_gt50,
    gallery_sub: int = 1,
    gallery_indices=None,
    gallery_store: dict = None,
    dev: str = "cuda:0",
):
    model, base_cond, head_mod, cone = load_arm_components(arm_kind, ckpt_p, z0_oof_pack, dev)
    R_tr, a_tr, b_tr = cone["R"], cone["alpha"], cone["beta"]
    u_tr = cone["u_tr"].to(dev).float()

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
                if head_mod is not None:
                    st_l = head_mod.forward_ips_steps(c_ens, cs_br, ips_p)
                    for k in range(4):
                        ips_steps_all[k].append(st_l[k].cpu())
                    z0_use_l.append(head_mod.forward_z0(c_ens, cs_br, z0_e[i:i + 100].to(dev)).cpu())
                    pe_u, ppe_u = head_mod.forward_dual_ca(c_ens, pe, ppe)
                    pe_use_l.append(pe_u.cpu())
                    ppe_use_l.append(ppe_u.cpu())
                else:
                    ip = base_cond(c_ens, ips_p).cpu() if base_cond is not None else ips_p.cpu()
                    for k in range(4):
                        ips_steps_all[k].append(ip)
                    z0_use_l.append(z0_e[i:i + 100])

            ips_steps_cat = [torch.cat(lst, 0) for lst in ips_steps_all]
            z0_final = torch.cat(z0_use_l, 0)
            pe_final = torch.cat(pe_use_l, 0) if pe_use_l else None
            ppe_final = torch.cat(ppe_use_l, 0) if ppe_use_l else None

            rec = torch.empty(500, 3, 512, 512, dtype=torch.float32)
            for s0 in range(0, 500, 50):
                q = list(range(s0, min(500, s0 + 50)))
                cids_q = [cids_500[i] for i in q]
                eps_q = torch.stack([eps_map[ci] for ci in cids_q]).to(dev)
                sn_q = [torch.stack([sn_map[ci][st] for ci in cids_q]).to(dev) for st in range(4)]
                ips_q = [ips_steps_cat[st][q].to(dev) for st in range(4)]
                if pe_final is not None:
                    lat_out = rollout_4step_dualca(
                        pipe_unet, scheduler, ips_q, z0_final[q].to(dev), eps_q, sn_q,
                        pe_final[q].to(dev), ppe_final[q].to(dev), atid, smap_c, lk,
                    )
                else:
                    lat_out = rollout_4step_step_ips(
                        pipe_unet, scheduler, ips_q, z0_final[q].to(dev), eps_q, sn_q,
                        pe, ppe, atid, smap_c, lk,
                    )
                for j0 in range(0, len(q), 10):
                    img_b = (vae32.decode(lat_out[j0:j0 + 10].float() / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
                    rec[s0 + j0:s0 + j0 + len(img_b)] = img_b

            if gallery_store is not None and sub == gallery_sub and rtag == "s375" and gallery_indices is not None:
                gallery_store[arm_name] = rec[gallery_indices].clone()

            m = score_full500(rec, gt_rgb50, gt_feats, nets, dev=dev)
            h_l = []
            for i in range(0, 500, 50):
                x224 = ((F.interpolate(rec[i:i + 50].to(dev), size=224, mode="bilinear", align_corners=False) - clip_mean) / clip_std).to(torch.float16)
                h_l.append(h14_eval(pixel_values=x224).image_embeds.float().cpu())
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
            f">>> [HO-6 {arm_name:24s} {rtag}] "
            f"INDEP CLIP-L14: t10={m_avg['clip_t10']*100:5.2f}% dc10={m_avg['clip_dc10']*100:5.2f}% sw={m_avg['clip_sw']*100:5.2f}% g500={m_avg['clip_g500']*100:4.2f}% | "
            f"Inc sw={m_avg['incep_sw']*100:5.2f}% | A5: t10={m_avg['alex5_t10']*100:5.2f}% sw={m_avg['alex5_sw']*100:5.2f}% | "
            f"Px sw={m_avg['pix_sw']*100:5.2f}% | SSIM={m_avg['ssim']:.4f} || (H14: t10={m_avg['h14_t10']*100:5.2f}% sw={m_avg['h14_sw']*100:5.2f}%)",
            flush=True,
        )
    del model, base_cond, head_mod
    torch.cuda.empty_cache()
    return res_by_reg


def build_heldout_gallery(gt_rgb50: torch.Tensor, gallery_store: dict, gallery_indices: list, out_jpg: Path):
    """Build uncherry-picked side-by-side Held-Out sub-01 s375 gallery: pairs (A, B) from first 6 sealed concepts."""
    arm_order = [
        ("GT Photo", None),
        ("Arm1b (RepIntra)", "Arm1b_RepIntra_CondLyr"),
        ("SwapDiff (Arm3b)", "SwapDiff_Arm3b"),
        ("D1 (SimOnly)", "D1_SimOnly_Cond"),
        ("D2 (Compete4)", "D2_Compete4_Cond"),
        ("D4 (UnfEnc+Z0)", "D4_UnfreezeEnc_StepZ0"),
    ]
    tile = 192
    pad = 8
    header_h = 34
    label_w = 108
    n_rows = len(gallery_indices)
    n_cols = len(arm_order)
    W = label_w + n_cols * (tile + pad) + pad
    H = header_h + n_rows * (tile + pad) + pad
    canvas = Image.new("RGB", (W, H), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)

    for c_i, (title, _) in enumerate(arm_order):
        x0 = label_w + pad + c_i * (tile + pad)
        draw.text((x0 + 6, 10), title, fill=(20, 20, 20))

    for r_i, idx in enumerate(gallery_indices):
        y0 = header_h + pad + r_i * (tile + pad)
        c_num = idx // 10
        p_ab = "Photo A" if (idx % 10 == 0) else "Photo B"
        draw.text((8, y0 + tile // 2 - 12), f"Concept #{c_num:02d}\n{p_ab} (s375)", fill=(30, 30, 30))
        for c_i, (_, key) in enumerate(arm_order):
            x0 = label_w + pad + c_i * (tile + pad)
            ten = gt_rgb50[idx] if key is None else gallery_store[key][r_i]
            arr = (ten.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            im = Image.fromarray(arr).resize((tile, tile), Image.BICUBIC)
            canvas.paste(im, (x0, y0))
        # Subtle divider between concept pairs
        if r_i % 2 == 1 and r_i < n_rows - 1:
            draw.line([(4, y0 + tile + pad // 2), (W - 4, y0 + tile + pad // 2)], fill=(170, 170, 170), width=2)

    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_jpg, quality=92)
    print(f"[SAVED GALLERY] {out_jpg} ({W}x{H})", flush=True)


def extract_metric(sub_dict: dict, mkey: str) -> float:
    if mkey == "clip_t10": return sub_dict["clip"]["top1"]
    if mkey == "clip_dc10": return sub_dict["clip"]["top1_dc"]
    if mkey == "clip_sw": return sub_dict["clip"]["swap2x2"]
    if mkey == "clip_g500": return sub_dict["clip"]["global500_top1"]
    if mkey == "incep_sw": return sub_dict["inception"]["swap2x2"]
    if mkey == "alex5_t10": return sub_dict["alex5"]["top1"]
    if mkey == "alex5_sw": return sub_dict["alex5"]["swap2x2"]
    if mkey == "pix_sw": return sub_dict["pixcorr"]["swap2x2"]
    if mkey == "ssim": return sub_dict["ssim"]
    raise KeyError(mkey)


def paired_ttest_block(all_per_sub: dict, arm_a: str, arm_b: str, reg: str, subs_list: list):
    """Return paired two-sided t-test stats for arm_a - arm_b across subs_list."""
    keys = [f"sub-{s:02d}" for s in subs_list]
    metrics = ["clip_t10", "clip_dc10", "clip_sw", "clip_g500", "incep_sw", "alex5_t10", "alex5_sw", "pix_sw", "ssim"]
    out = {}
    for mk in metrics:
        va = np.array([extract_metric(all_per_sub[arm_a][reg][k], mk) for k in keys], dtype=np.float64)
        vb = np.array([extract_metric(all_per_sub[arm_b][reg][k], mk) for k in keys], dtype=np.float64)
        diff = va - vb
        t_stat, p_val = stats.ttest_rel(va, vb)
        out[mk] = {
            "mean_a": float(va.mean()),
            "mean_b": float(vb.mean()),
            "delta": float(diff.mean()),
            "sd_diff": float(diff.std(ddof=1)),
            "wins": int((diff > 0).sum()),
            "ties": int((diff == 0).sum()),
            "n": len(keys),
            "t": float(t_stat),
            "p": float(p_val),
        }
    return out


def main():
    dev = "cuda:0"
    t0 = time.time()

    concept_np, ok_np, tr_idx_np, f0_all10, f0_idx, sel50_cids, sel50_idx = get_concept_splits()
    z0_oof_pack = torch.load(DRAFT_DIR / "z0_oof_tr14_256.pt", map_location="cpu", weights_only=True)
    z0_by_sub = {sub: fit_tr14_ridge_z0(sub, tr_idx_np, sel50_idx, dev=dev) for sub in HELDOUT_SUBS}

    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    from transformers import CLIPVisionModelWithProjection
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
    h14_eval = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / "models" / EMB_KEYS[0]), torch_dtype=dt).to(dev).eval().requires_grad_(False)

    lat50 = torch.from_numpy(np.load(PREP / "lat512_train.npy")[sel50_idx]).float()
    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    H14_gt50 = emb_tr[EMB_KEYS[0]][sel50_idx].float()
    with torch.inference_mode():
        gt_rgb50 = torch.cat([
            (vae32.decode(lat50[j:j + 20].to(dev, torch.float32) / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
            for j in range(0, 500, 20)
        ])
        nets = _nets()
        gt_feats = extract_eval_feats(gt_rgb50, nets, dev=dev)

    arm_specs = [
        ("Arm0_Base", "std", OUT_DIR / "arm0_tr14_xbrain_s2025.pt"),
        ("Arm1b_RepIntra_CondLyr", "std", OUT_DIR / "arm1b_rep_condlyr_st600_s2025.pt"),
        ("SwapDiff_Arm3b", "std", OUT_DIR / "arm3b_rep_rawswap_condlyr_st600_s2025.pt"),
        ("D1_SimOnly_Cond", "std", DRAFT_DIR / "d1_simonly_cond_st120_s2025.pt"),
        ("D2_Compete4_Cond", "std", DRAFT_DIR / "d2_compete4_cond_st120_s2025.pt"),
        ("D4_UnfreezeEnc_StepZ0", "v2_head", DRAFT_DIR / "d4_unfreeze_enc_condz0_st150_s2025.pt"),
    ]
    for name, _, cp in arm_specs:
        assert cp.exists(), f"Missing checkpoint {cp}"

    # Uncherry-picked gallery indices: first 6 sealed concepts, photos 0 and 1 each (12 rows = 6 within-concept pairs!)
    gallery_indices = [c * 10 + p for c in range(6) for p in (0, 1)]
    gallery_store = {}

    res_p = DRAFT_DIR / "exp7_heldout6_and_all10_results.json"
    ho6_res = {}
    for arm_name, arm_kind, cp in arm_specs:
        ho6_res[arm_name] = eval_arm_on_subs(
            arm_name, arm_kind, cp, HELDOUT_SUBS, sel50_idx, z0_by_sub, z0_oof_pack,
            pipe.unet, pipe.scheduler, vae32, adapter, proj, pe, ppe, atid,
            gt_rgb50, gt_feats, nets, h14_eval, H14_gt50,
            gallery_sub=1, gallery_indices=gallery_indices, gallery_store=gallery_store, dev=dev,
        )

    # Save uncherry-picked Held-Out sub-01 s375 gallery
    gal_p = DRAFT_DIR / "draft_heldout_sub01_s375_gallery.jpg"
    build_heldout_gallery(gt_rgb50, gallery_store, gallery_indices, gal_p)

    # Merge Dev-4 (from exp5/exp6/exp6b) + Held-Out-6 -> complete All-10 per_sub matrix!
    d5 = json.loads((OUT_DIR / "exp5_z0_2x2_diag.json").read_text())
    d6 = json.loads((DRAFT_DIR / "exp6_draft_dev4_results.json").read_text())
    d6b = json.loads((DRAFT_DIR / "exp6b_draft_v2_dev4_results.json").read_text())

    dev4_sources = {
        "Arm0_Base": d5["free_sweep"]["Arm0_Base"],
        "Arm1b_RepIntra_CondLyr": d5["free_sweep"]["Arm1b_RepIntra_CondLyr"],
        "SwapDiff_Arm3b": d6["arms"]["SwapDiff_Arm3b"],
        "D1_SimOnly_Cond": d6["arms"]["d1_simonly_cond"],
        "D2_Compete4_Cond": d6["arms"]["d2_compete4_cond"],
        "D4_UnfreezeEnc_StepZ0": d6b["arms"]["d4_unfreeze_enc_condz0"],
    }

    all10_per_sub = {}
    summary = {"heldout6": {}, "all10": {}}
    for arm_name, _, _ in arm_specs:
        all10_per_sub[arm_name] = {}
        summary["heldout6"][arm_name] = {}
        summary["all10"][arm_name] = {}
        for rtag in ("s375", "s625"):
            merged = {}
            merged.update(dev4_sources[arm_name][rtag]["per_sub"])
            merged.update(ho6_res[arm_name][rtag]["per_sub"])
            assert len(merged) == 10, f"Expected 10 subs for {arm_name} {rtag}, got {len(merged)}"
            all10_per_sub[arm_name][rtag] = merged
            summary["heldout6"][arm_name][rtag] = ho6_res[arm_name][rtag]["mean"]
            summary["all10"][arm_name][rtag] = {
                mk: float(np.mean([extract_metric(merged[f"sub-{s:02d}"], mk) for s in ALL_SUBS]))
                for mk in ["clip_t10", "clip_dc10", "clip_sw", "clip_g500", "incep_sw", "alex5_t10", "alex5_sw", "pix_sw", "ssim"]
            }

    # Key paired two-sided t-tests on BOTH Held-Out-6 and All-10!
    pairs = [
        ("SwapDiff_Arm3b_vs_Arm1b", "SwapDiff_Arm3b", "Arm1b_RepIntra_CondLyr"),
        ("SwapDiff_Arm3b_vs_Arm0", "SwapDiff_Arm3b", "Arm0_Base"),
        ("D2_Compete4_vs_D1_SimOnly", "D2_Compete4_Cond", "D1_SimOnly_Cond"),
        ("D2_Compete4_vs_SwapDiff", "D2_Compete4_Cond", "SwapDiff_Arm3b"),
        ("D2_Compete4_vs_Arm1b", "D2_Compete4_Cond", "Arm1b_RepIntra_CondLyr"),
        ("D4_UnfreezeEnc_vs_D1_SimOnly", "D4_UnfreezeEnc_StepZ0", "D1_SimOnly_Cond"),
        ("D4_UnfreezeEnc_vs_SwapDiff", "D4_UnfreezeEnc_StepZ0", "SwapDiff_Arm3b"),
        ("D4_UnfreezeEnc_vs_Arm1b", "D4_UnfreezeEnc_StepZ0", "Arm1b_RepIntra_CondLyr"),
        ("D4_UnfreezeEnc_vs_Arm0", "D4_UnfreezeEnc_StepZ0", "Arm0_Base"),
    ]
    ttests = {"heldout6": {}, "all10": {}}
    for rtag in ("s375", "s625"):
        ttests["heldout6"][rtag] = {label: paired_ttest_block(all10_per_sub, a, b, rtag, HELDOUT_SUBS) for label, a, b in pairs}
        ttests["all10"][rtag] = {label: paired_ttest_block(all10_per_sub, a, b, rtag, ALL_SUBS) for label, a, b in pairs}

    # Also test s375 vs s625 anchoring effect within each arm across All-10 and Held-Out-6!
    anchor_ttests = {"heldout6": {}, "all10": {}}
    for split_name, slist in [("heldout6", HELDOUT_SUBS), ("all10", ALL_SUBS)]:
        keys = [f"sub-{s:02d}" for s in slist]
        for arm_name, _, _ in arm_specs:
            anchor_ttests[split_name][arm_name] = {}
            for mk in ["clip_t10", "clip_dc10", "clip_sw", "incep_sw", "alex5_t10", "ssim"]:
                v375 = np.array([extract_metric(all10_per_sub[arm_name]["s375"][k], mk) for k in keys])
                v625 = np.array([extract_metric(all10_per_sub[arm_name]["s625"][k], mk) for k in keys])
                diff = v375 - v625
                t_s, p_v = stats.ttest_rel(v375, v625)
                anchor_ttests[split_name][arm_name][mk] = {
                    "s375": float(v375.mean()),
                    "s625": float(v625.mean()),
                    "delta": float(diff.mean()),
                    "wins": int((diff > 0).sum()),
                    "n": len(keys),
                    "t": float(t_s),
                    "p": float(p_v),
                }

    final_obj = {
        "wall_s": round(time.time() - t0, 1),
        "dev_subs": DEV_SUBS,
        "heldout_subs": HELDOUT_SUBS,
        "summary": summary,
        "ttests": ttests,
        "anchor_s375_vs_s625": anchor_ttests,
        "all10_per_sub": all10_per_sub,
    }
    write_json(res_p, final_obj)
    print(f"=== EXP 7 (HELD-OUT-6 & ALL-10 CONFIRMATION) DONE in {round(time.time()-t0, 1)}s -> {res_p} ===", flush=True)


if __name__ == "__main__":
    main()
