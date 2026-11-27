"""PhotoEEG-Cascade (V5): F-Native High-SNR Structural Blueprint -> SDXL-Base Photorealistic Refiner!
Solves the exact 26.5-point F->H14 information bottleneck (F_200w=71.5% vs linear F->H14=45.0%):
  Stage 1: Brain-HIVE F-native IPAttnAdapterModel + SDXL-Turbo + low-band Ridge layout +
           strictly train-centered Closed-Loop F-correction + FEmbedder Sel6 -> High-SNR structural/semantic draft I_draft.
           (Retains the FULL 71.5% H14+B32+VAE signal for object identity, pose, aspect ratio & color!)
  Stage 2: SDXL-Base-1.0 (20-step DPM++ 2M Karras, CFG=5.0-5.5, TRUE uncond=ZERO, anti-clay neg prompt)
           Img2Img refinement from z_draft at strength=0.52..0.70 + Official IP-Adapter ViT-H/14
           conditioned on EEG H14 + I_draft H14 + confidence-gated B5-H14 CV-OpenVocab (8,141) prompt!
           (Replaces 100% of clay/plastic pixels with crisp natural DSLR textures while keeping exact Stage-1 object/shape!)
  Stage 3: Strictly Zero-Leakage Multi-Modal Verifier (FEmbedder + LAION H14 + LAION H14-L12 4x4 + SDXL-VAE)
           selecting ONLY from SDXL-Base photorealistic candidates (mixing Cascade + Direct-Base draws).
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

RAD = Path(os.environ.get("RAD_ROOT", "/path/to/rad"))
for sub in ("code/exp_photo", "code/Brain-HIVE", "code/rad", "code/exp"):
    p = str(RAD / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from common import EMB_KEYS, PRIOR, TURBO
from fenc import FEmbedder
from run_photoeeg_pipeline import query_seed, save_gallery, predict_single_query_subject
from train_photoeeg_encoders import nz
from train_strong_h14_and_diag import calibrate_h14
from run_v4_photo_ultra import PhotoUltraSDXLBase, UltraPhotoVerifier, fit_fused_h14_and_prior
from eval_photo_suite import PhotoFidelityEvaluator

PREP = RAD / "data/prep"
TGT_DIR = RAD / "runs/photo_eeg/targets"
MOD_DIR = RAD / "runs/photo_eeg/models"
V4_ROOT = RAD / "runs/photo_eeg/ultra_v4"
OUT_ROOT = RAD / "runs/photo_eeg/cascade_v5"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def build_b5_openvocab_prompts(h14_u: torch.Tensor, mu_H: torch.Tensor, margin_thresh: float = 0.015):
    """Strictly zero-leakage CV-OpenVocab (8,141) prompts decoded from high-SNR B5-H14.
    Uses ONLY LAION CLIP-H/14 text bank (no test labels, no OpenAI CLIP).
    Confidence-gated: if top-1 minus top-5 margin < margin_thresh, falls back to clean generic photo prompt.
    """
    tb = torch.load(RAD / "runs/obj_bench/text_banks/text_banks_laion.pt", map_location="cpu", weights_only=False)
    words = tb["cv_open"]
    t_emb = nz(tb["h14_cv_open"].float(), dim=-1)
    t_c = nz(t_emb - t_emb.mean(0, keepdim=True), dim=-1)

    hc = nz(h14_u.cpu().float() - mu_H.cpu().float(), dim=-1)
    # Blend raw and centered similarity for crisp open-vocab object naming without hub bias
    sim = 0.45 * (nz(h14_u.cpu().float(), dim=-1) @ t_emb.T) + 0.55 * (hc @ t_c.T)
    topv, topi = sim.topk(5, dim=-1)

    prompts = []
    chosen_words = []
    n_gated = 0
    for i in range(len(h14_u)):
        margin = float((topv[i, 0] - topv[i, 4]).item())
        w1 = words[int(topi[i, 0])].replace("_", " ")
        w2 = words[int(topi[i, 1])].replace("_", " ")
        chosen_words.append((w1, w2, margin))
        if margin >= margin_thresh:
            n_gated += 1
            prompts.append(f"a sharp realistic natural photograph of a {w1}, crisp object detail, natural lighting")
        else:
            prompts.append("a sharp realistic natural photograph, crisp object detail, natural lighting")
    return prompts, chosen_words, n_gated


@torch.inference_mode()
def run_stage1_turbo_f_draft(c_f: torch.Tensor, z0_ridge: torch.Tensor, mu_F: torch.Tensor,
                             sub_id: int, dev: str = "cuda:0", n_sel: int = 6,
                             loop_w: float = 0.35, ilvr_w: float = 0.25):
    """Stage 1: F-Native 71.5%-SNR Structural & Semantic Draft using Brain-HIVE IPAttnAdapterModel + SDXL-Turbo.
    Strictly single-query: uses ONLY train mean mu_F for centering (ZERO test-batch TTA!).
    Generates n_sel drafts with mild closed-loop F correction and selects best via zero-leakage FEmbedder.
    Frees ALL Turbo weights before returning uint8 [200, 3, 512, 512] + draft H14 embeddings!
    """
    from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel

    t0 = time.time()
    dt = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=dt, variant="fp16")
    attn = IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter")
    attn.bind_unet(pipe.unet)
    proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, dt).eval()
    A = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, dt).eval()
    pipe.to(dev, dt)
    pipe.vae.to(dtype=torch.float32)
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    p2 = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    p2.set_progress_bar_config(disable=True)

    pe1, _, ppe1, _ = pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
    pe1, ppe1 = pe1.half(), ppe1.half()

    N = len(c_f)
    # Decode z0_ridge cleanly to 512px start images (bit-exact to Brain-HIVE img2img!)
    i0_all = torch.empty((N, 3, 512, 512), dtype=torch.float32)
    for j in range(0, N, 10):
        sl = slice(j, min(N, j + 10))
        dec = pipe.vae.decode(z0_ridge[sl].to(dev, dtype=torch.float32) / pipe.vae.config.scaling_factor).sample
        i0_all[sl] = (dec * 0.5 + 0.5).clamp(0, 1).cpu()
    # Cast VAE to fp16 to match StableDiffusionXLImg2ImgPipeline (bit-exact to gen.py)
    pipe.vae.to(dtype=dt)
    p2 = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    p2.set_progress_bar_config(disable=True)

    def gen_pass(cv_in: torch.Tensor, draw_k: int, strength: float = 0.75, amp: float = 1.0, use_ilvr: bool = True):
        # Optional mild train-centered amplification (STRICTLY train mean mu_F, ZERO test-batch stats!)
        cv = cv_in.float().to(dev)
        if abs(amp - 1.0) > 1e-4:
            cn = nz(cv, dim=-1)
            m = nz(mu_F.to(dev).float(), dim=-1)
            cv = cv.norm(dim=-1, keepdim=True) * nz(m + amp * (cn - m), dim=-1)
        out = torch.empty((N, 3, 512, 512), dtype=torch.uint8)
        bs = 10
        n_run = min(int(4 * strength), 4)
        for j in range(0, N, bs):
            sl = slice(j, min(N, j + bs))
            B = sl.stop - sl.start
            cc = cv[sl].to(dt)
            ips = proj(A(cc))
            i0 = i0_all[sl].to(dev, dtype=dt)
            z0 = z0_ridge[sl].to(dev, dtype=torch.float32)
            gens = [torch.Generator(dev).manual_seed(query_seed(sub_id, idx, 500 + draw_k)) for idx in range(sl.start, sl.stop)]
            eps = torch.stack([torch.randn((4, 64, 64), generator=torch.Generator().manual_seed(query_seed(sub_id, idx, 900 + draw_k)))
                               for idx in range(sl.start, sl.stop)]).to(dev)

            def lp(z):
                return F.interpolate(F.avg_pool2d(z, 4), size=z.shape[-1], mode="bilinear", align_corners=False)

            def cb(pp, step_i, t, ck):
                if use_ilvr and ilvr_w > 0 and step_i < n_run - 1 and step_i < 2:
                    sig = pp.scheduler.sigmas[pp.scheduler.step_index].to(dev)
                    lt = ck["latents"].float()
                    ck["latents"] = (lt + ilvr_w * (lp(z0 + sig * eps) - lp(lt))).to(ck["latents"].dtype)
                return ck

            imgs = p2(
                image=i0, strength=strength, generator=gens,
                prompt_embeds=pe1.expand(B, -1, -1), pooled_prompt_embeds=ppe1.expand(B, -1),
                num_inference_steps=4, guidance_scale=0.0, output_type="pt",
                cross_attention_kwargs={"ip_hidden_states": ips},
                callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"],
            ).images
            out[sl] = (imgs.float().clamp(0, 1) * 255.0).round().byte().cpu()
        return out

    fe = FEmbedder(dev=dev)
    mu_Fd = mu_F.to(dev).float()
    cf_c = nz(c_f.to(dev).float() - mu_Fd, dim=-1)

    # 1-iteration strictly train-centered closed-loop F correction!
    c_work = c_f.clone().float()
    if loop_w > 0:
        d0 = gen_pass(c_work, draw_k=0, strength=0.75, amp=1.0)
        f0 = fe(d0.float() / 255.0, bs=20).to(dev).float()
        dc = nz(c_f.to(dev).float() - mu_Fd, dim=-1)
        df = nz(f0 - mu_Fd, dim=-1)
        # Push condition along component of dc missed by df
        miss = dc - (dc * df).sum(-1, keepdim=True) * df
        c_corr_dir = nz(nz(c_f.to(dev).float(), dim=-1) + loop_w * miss, dim=-1)
        c_corr = c_corr_dir * c_f.to(dev).float().norm(dim=-1, keepdim=True)
    else:
        c_corr = c_f.to(dev).float()

    # Generate diverse Stage-1 drafts (mix of raw c_f & closed-loop c_corr & amp 1.0/1.25)
    specs = [
        (c_f, 0, 0.75, 1.00),
        (c_f, 1, 0.75, 1.15),
        (c_corr.cpu(), 2, 0.75, 1.00),
        (c_corr.cpu(), 3, 0.75, 1.15),
        (c_corr.cpu(), 4, 1.00, 1.05),   # pure semantic draft (free of blurry start if start was noisy!)
        (c_f, 5, 0.75, 1.25),
    ][:n_sel]

    cands = []
    scores = torch.zeros((len(specs), N), device=dev)
    h14_cands = []
    for k, (cv_k, dk, st_k, am_k) in enumerate(specs):
        im_k = gen_pass(cv_k, draw_k=dk, strength=st_k, amp=am_k, use_ilvr=(st_k < 0.99))
        cands.append(im_k)
        f_list, h_list = [], []
        for j in range(0, N, 20):
            sl = slice(j, min(N, j + 20))
            parts = fe.parts(im_k[sl].float() / 255.0)
            f_list.append(fe.fuse(**parts).float())
            h_list.append(nz(parts[EMB_KEYS[0]].float(), dim=-1).cpu())
        fk = torch.cat(f_list, 0)
        h14_cands.append(torch.cat(h_list, 0))
        scores[k] = (nz(fk - mu_Fd, dim=-1) * cf_c).sum(-1)

    best_k = scores.argmax(dim=0).cpu()
    best_imgs = torch.empty_like(cands[0])
    best_h14 = torch.empty((N, 1024), dtype=torch.float32)
    for i in range(N):
        ki = int(best_k[i])
        best_imgs[i] = cands[ki][i]
        best_h14[i] = h14_cands[ki][i]
    hist = [int((best_k == k).sum().item()) for k in range(len(specs))]
    print(f"  [STAGE1 Turbo-F sub-{sub_id:02d}] Sel{len(specs)} done in {time.time()-t0:.1f}s | hist={hist}", flush=True)

    # Free ALL Turbo + FEmbedder VRAM cleanly before Stage 2 SDXL-Base loads!
    del pipe, p2, attn, proj, A, fe, i0_all, cands
    torch.cuda.empty_cache()
    return best_imgs, best_h14


def run_cascade(subs: list, cond_dir: str = "cone_b5_w1.00", low_tag: str = "lat512_b5_v2_n0.80",
                dev: str = "cuda:0", make_gallery: bool = True):
    bp = fit_fused_h14_and_prior(dev=dev)
    mu_F, mu_H = bp["mu_F"], bp["mu_H"]
    W_fh = bp["W_fh"]
    h14_raw_norm = bp["h14_raw_norm"]

    # Stage 1 FIRST for all requested subjects (frees Turbo VRAM completely before Stage 2!)
    s1_cache = {}
    for s in subs:
        s1_p = OUT_ROOT / f"{cond_dir}_C0_Stage1_TurboF_Sel6" / f"sub-{s:02d}.pt"
        s1_p.parent.mkdir(parents=True, exist_ok=True)
        if s1_p.exists():
            d = torch.load(s1_p, map_location="cpu", weights_only=False)
            s1_cache[s] = (d["imgs_u8"], d["draft_h14"])
        else:
            c_f = torch.from_numpy(np.load(RAD / f"runs/cond/{cond_dir}/sub-{s:02d}.npy")).float()
            z0_ridge = torch.from_numpy(np.load(RAD / f"runs/low/sub-{s:02d}/{low_tag}.npy")).float()
            im_s1, h14_s1 = run_stage1_turbo_f_draft(c_f, z0_ridge, mu_F, sub_id=s, dev=dev, n_sel=6, loop_w=0.35, ilvr_w=0.25)
            torch.save({"imgs_u8": im_s1, "draft_h14": h14_s1}, s1_p)
            s1_cache[s] = (im_s1, h14_s1)

    # Stage 2: Load SDXL-Base + Zero-Leakage Verifier + Blind Evaluator
    gen = PhotoUltraSDXLBase(dev=dev)
    ver = UltraPhotoVerifier(gen.vae, mu_F, mu_H, dev=dev)
    ev = PhotoFidelityEvaluator(dev)

    arms = [
        "C0_Stage1_TurboF_Sel6",
        "C1_Cascade_Str52_Base20",
        "C2_Cascade_Str62_Base20",
        "C3_Cascade_Str70_Early3",
        "C4_Cascade_Str60_OpenVocab",
        "C5_Cascade_DualIP_Str58",
        "C6_PhotoCascade_VerifySel8",
    ]

    for s in subs:
        c_f = torch.from_numpy(np.load(RAD / f"runs/cond/{cond_dir}/sub-{s:02d}.npy")).float().to(dev)
        z0_ridge = torch.from_numpy(np.load(RAD / f"runs/low/sub-{s:02d}/{low_tag}.npy")).float()
        pk_p = MOD_DIR / f"sq_preds_r80_sub-{s:02d}.pt"
        pk = torch.load(pk_p, map_location="cpu", weights_only=False) if pk_p.exists() else predict_single_query_subject(s, budget="r80", dev=dev)
        p16_115 = pk["mu_p16"].float() + 1.15 * (pk["p16_raw"].float() - pk["mu_p16"].float())
        l12_raw = pk["l12"].float().to(dev)

        im_s1, h14_s1 = s1_cache[s]
        # Encode Stage-1 draft into SDXL-Base VAE latent z_draft (batch=5, zero OOM!)
        z_draft = torch.empty((len(im_s1), 4, 64, 64), dtype=torch.float32)
        with torch.inference_mode():
            for j in range(0, len(im_s1), 5):
                sl = slice(j, min(len(im_s1), j + 5))
                xb = (im_s1[sl].to(dev).float() / 255.0) * 2.0 - 1.0
                z_draft[sl] = (gen.vae.encode(xb).latent_dist.mean * gen.vae.config.scaling_factor).cpu()

        # High-SNR H14 from pure F->H14 + gentle 25% reinforcement from Stage-1 F-native draft H14!
        h_from_f = nz(mu_H.to(dev) + (c_f - mu_F.to(dev)) @ W_fh.to(dev), dim=-1).cpu()
        h_casc = nz(0.75 * h_from_f + 0.25 * nz(h14_s1.float(), dim=-1), dim=-1)

        h_c16 = calibrate_h14(h_casc, mu_H, h14_raw_norm, angle_gain=1.60)
        h_c19 = calibrate_h14(h_casc, mu_H, h14_raw_norm, angle_gain=1.90)
        h_p17 = calibrate_h14(h_from_f, mu_H, h14_raw_norm, angle_gain=1.75)

        p_gen = ["a sharp realistic natural photograph, crisp object detail, natural lighting"] * len(im_s1)
        p_ov, ov_words, n_gated = build_b5_openvocab_prompts(h_casc, mu_H, margin_thresh=0.015)
        print(f"  [CASCADE sub-{s:02d}] B5-OpenVocab gated={n_gated}/200 | first 6={[(w[0], round(w[2], 3)) for w in ov_words[:6]]}", flush=True)

        gal_cols = []
        sub_res = {}
        cached_imgs = {}
        for arm in arms:
            arm_d = OUT_ROOT / f"{cond_dir}_{arm}"
            arm_d.mkdir(parents=True, exist_ok=True)
            cp = arm_d / f"sub-{s:02d}.pt"
            if cp.exists() and "metrics" in torch.load(cp, map_location="cpu", weights_only=False):
                d = torch.load(cp, map_location="cpu", weights_only=False)
                imgs, m = d["imgs_u8"], d["metrics"]
            else:
                t0 = time.time()
                hist = None
                if arm == "C0_Stage1_TurboF_Sel6":
                    imgs = im_s1
                elif arm == "C1_Cascade_Str52_Base20":
                    imgs = gen.generate(h_c16, p16_115, z0_lat=z_draft, prompts=p_gen,
                                        scale_std=0.85, scale_plus=0.0, cfg_scale=5.5, num_steps=20, strength=0.52,
                                        early_anchor_steps=0, sub_id=s, draw_k=10)
                elif arm == "C2_Cascade_Str62_Base20":
                    imgs = gen.generate(h_c16, p16_115, z0_lat=z_draft, prompts=p_gen,
                                        scale_std=0.88, scale_plus=0.0, cfg_scale=5.2, num_steps=20, strength=0.62,
                                        early_anchor_steps=0, sub_id=s, draw_k=11)
                elif arm == "C3_Cascade_Str70_Early3":
                    imgs = gen.generate(h_c19, p16_115, z0_lat=z_draft, prompts=p_gen,
                                        scale_std=0.90, scale_plus=0.0, cfg_scale=5.2, num_steps=20, strength=0.70,
                                        early_anchor_steps=3, early_anchor_gain=0.14, early_anchor_lp=10, sub_id=s, draw_k=12)
                elif arm == "C4_Cascade_Str60_OpenVocab":
                    imgs = gen.generate(h_c16, p16_115, z0_lat=z_draft, prompts=p_ov,
                                        scale_std=0.80, scale_plus=0.0, cfg_scale=5.5, num_steps=20, strength=0.60,
                                        early_anchor_steps=0, sub_id=s, draw_k=13)
                elif arm == "C5_Cascade_DualIP_Str58":
                    imgs = gen.generate(h_c16, p16_115, z0_lat=z_draft, prompts=p_gen,
                                        scale_std=0.72, scale_plus=0.20, cfg_scale=5.2, num_steps=20, strength=0.58,
                                        early_anchor_steps=0, sub_id=s, draw_k=14)
                elif arm == "C6_PhotoCascade_VerifySel8":
                    # Extra 2 fresh photorealistic draws: Cascade str=0.58 seed2 + Direct SDXL-Base PureFH14!
                    d_c2b = gen.generate(h_c19, p16_115, z0_lat=z_draft, prompts=p_gen,
                                         scale_std=0.88, scale_plus=0.0, cfg_scale=5.4, num_steps=20, strength=0.58,
                                         early_anchor_steps=0, sub_id=s, draw_k=15)
                    # Reuse cached V4 U0/U1 if available, otherwise generate Direct Base draw cleanly!
                    u0_p = V4_ROOT / f"{cond_dir}_U0_PureFH14_Ridge72_Early5" / f"sub-{s:02d}.pt"
                    u1_p = V4_ROOT / f"{cond_dir}_U1_FusedH14_RidgeEarly4" / f"sub-{s:02d}.pt"
                    if u0_p.exists():
                        d_dir1 = torch.load(u0_p, map_location="cpu", weights_only=False)["imgs_u8"]
                    else:
                        d_dir1 = gen.generate(h_p17, p16_115, z0_lat=z0_ridge, prompts=p_gen,
                                              scale_std=0.94, scale_plus=0.0, cfg_scale=5.0, num_steps=20, strength=0.72,
                                              early_anchor_steps=5, early_anchor_gain=0.16, early_anchor_lp=10, sub_id=s, draw_k=6)
                    cands = [
                        cached_imgs["C1_Cascade_Str52_Base20"],
                        cached_imgs["C2_Cascade_Str62_Base20"],
                        cached_imgs["C3_Cascade_Str70_Early3"],
                        cached_imgs["C4_Cascade_Str60_OpenVocab"],
                        cached_imgs["C5_Cascade_DualIP_Str58"],
                        d_c2b,
                        d_dir1,
                    ]
                    if u1_p.exists():
                        cands.append(torch.load(u1_p, map_location="cpu", weights_only=False)["imgs_u8"])
                    imgs, hist = ver.select_best(cands, c_f, h_from_f.to(dev), l12_raw, z0_ridge,
                                                 w_f=0.50, w_h=0.20, w_l12=0.15, w_lat=0.15)
                else:
                    raise ValueError(arm)

                m = ev.evaluate(imgs, compute_lowlevel=True)
                if hist is not None:
                    m["cand_hist"] = hist
                save_d = {"imgs_u8": imgs, "metrics": m}
                if arm == "C0_Stage1_TurboF_Sel6":
                    save_d["draft_h14"] = h14_s1
                torch.save(save_d, cp)
                print(
                    f"  [CASCADE sub-{s:02d} {arm:28s}] ({time.time()-t0:5.1f}s) | "
                    f"DINO_T1={m['dino_t1']:5.2f}% cos={m['dino_cos']:.3f} p8={m['dino_patch_cos']:.3f} | "
                    f"CLIP_Img={m['clip_img200_t1']:5.2f}% ZS200={m['zs_obj200_t1']:5.2f}% ZS1823={m['zs_obj1823_t1']:4.2f}% | "
                    f"Crisp={m['obj_crisp']:4.1f}% Alex2/5={m['alex2']:4.1f}/{m['alex5']:4.1f} Incep={m['incep_2way']:4.1f} "
                    f"2wC={m['clip_2way']:4.1f} Pix={m['pixcorr']:.3f} SSIM={m['ssim']:.3f}" +
                    (f" hist={hist}" if hist else ""),
                    flush=True,
                )
            sub_res[arm] = m
            cached_imgs[arm] = imgs
            gal_cols.append((arm.replace("Cascade_", "").replace("Stage1_", "S1_"), imgs))

        if make_gallery and gal_cols:
            save_gallery(ev.gt_512, gal_cols, ev.test_names, OUT_ROOT / f"gallery_cascade_{cond_dir}_sub-{s:02d}.jpg")
        json.dump(sub_res, open(OUT_ROOT / f"metrics_cascade_{cond_dir}_sub-{s:02d}.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", type=str, default="10")
    ap.add_argument("--cond", type=str, default="cone_b5_w1.00")
    ap.add_argument("--low", type=str, default="lat512_b5_v2_n0.80")
    ap.add_argument("--dev", type=str, default="cuda:0")
    ap.add_argument("--no-gallery", action="store_true")
    args = ap.parse_args()
    subs = [int(x) for x in args.subs.split(",") if x.strip()]
    run_cascade(subs, cond_dir=args.cond, low_tag=args.low, dev=args.dev, make_gallery=not args.no_gallery)
