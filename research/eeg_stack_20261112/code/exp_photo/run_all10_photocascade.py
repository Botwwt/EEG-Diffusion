"""PhotoEEG-Cascade: Full 10-Subject Confirmation & Publication Statistical Evaluation.

Evaluates 5 strictly matched arms across all 10 THINGS-EEG2 subjects (2,000 unseen test photos/arm)
under BOTH Full-Stack B5 (cone_b5_w1.00) and Strict Single-Subject O8 (cone_o8_w1.00):
  - B0_TurboF_1Draw        : Matched Brain-HIVE F-native SDXL-Turbo (4-step, nsel=1)
  - B1_TurboF_Sel6         : Matched Brain-HIVE F-native SDXL-Turbo (train-centered closed-loop + FEmbedder Sel6)
  - P1_Cascade_Base_1Draw  : PhotoEEG-Cascade Stage-2 SDXL-Base-1.0 (18-step DPM++ 2M Karras, CFG=5.3, str=0.58, 1 draw)
  - P2_Cascade_OV_1Draw    : PhotoEEG-Cascade + Confidence-Gated LAION CV-OpenVocab (8,141) prompt (1 draw)
  - P3_Cascade_Sel6        : PhotoEEG-Cascade-Sel6 (all-Cascade SDXL-Base pool ranked strictly zero-leakage by
                             0.70*FEmbedder + 0.15*LAION_H14 + 0.15*SDXL_VAE_Lat16)

100% Zero Evaluator Leakage:
  DINOv2-base, OpenAI CLIP-ViT-L/14, Inception-v3, EfficientNet-B1, AlexNet(2/5), SSIM, and PixCorr
  are NEVER touched during Stage-1, Stage-2, Open-Vocab prompting, or Sel6 verification.
"""
import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import stats
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

RAD = Path(os.environ.get("RAD_ROOT", "/path/to/rad"))
for sub in ("code/exp_photo", "code/Brain-HIVE", "code/rad", "code/exp"):
    p = str(RAD / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from common import EMB_KEYS, PRIOR, TURBO
from fenc import FEmbedder
from run_photoeeg_pipeline import query_seed
from train_photoeeg_encoders import nz
from train_strong_h14_and_diag import calibrate_h14
from run_v4_photo_ultra import PhotoUltraSDXLBase, fit_fused_h14_and_prior
from run_v5_cascade_photo import build_b5_openvocab_prompts
from eval_photo_suite import PhotoFidelityEvaluator

OUT_ROOT = RAD / "runs/photo_eeg/all10_cascade"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

ARMS = [
    "B0_TurboF_1Draw",
    "B1_TurboF_Sel6",
    "P1_Cascade_Base_1Draw",
    "P2_Cascade_OV_1Draw",
    "P3_Cascade_Sel6",
]

SPACED_12 = [15, 32, 48, 65, 82, 99, 115, 132, 148, 165, 182, 195]
FIRST_12 = list(range(12))


def save_panel(gt_512: torch.Tensor, cols: list, names: list, out_path: Path, indices: list, cell: int = 224):
    """Save clean side-by-side comparison gallery for specified stimulus indices (uncherry-picked)."""
    n_rows = len(indices)
    n_cols = 1 + len(cols)
    pad = 4
    top_h = 30
    left_w = 136
    W = left_w + n_cols * (cell + pad) + pad
    H = top_h + n_rows * (cell + pad) + pad
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    headers = ["GT Original"] + [label for label, _ in cols]
    for c, hdr in enumerate(headers):
        x0 = left_w + c * (cell + pad) + pad
        draw.text((x0 + 4, 8), hdr[:26], fill=(20, 20, 20))

    for r, idx in enumerate(indices):
        y0 = top_h + r * (cell + pad) + pad
        clean_name = names[idx].rsplit("_", 1)[0].replace("_", " ")
        draw.text((6, y0 + cell // 2 - 6), f"#{idx:03d} {clean_name}"[:20], fill=(25, 25, 25))
        g_img = Image.fromarray((gt_512[idx].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8))
        canvas.paste(g_img.resize((cell, cell), Image.BICUBIC), (left_w + pad, y0))
        for c, (_, imgs_u8) in enumerate(cols):
            x0 = left_w + (c + 1) * (cell + pad) + pad
            im = Image.fromarray(imgs_u8[idx].permute(1, 2, 0).numpy())
            canvas.paste(im.resize((cell, cell), Image.BICUBIC), (x0, y0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    print(f"[GALLERY SAVED] {out_path}", flush=True)


@torch.inference_mode()
def run_stage1_both(c_f: torch.Tensor, z0_ridge: torch.Tensor, mu_F: torch.Tensor,
                    sub_id: int, dev: str = "cuda:0", loop_w: float = 0.35, ilvr_w: float = 0.25):
    """Generates BOTH B0_TurboF_1Draw (draw 0) and B1_TurboF_Sel6 (train-centered closed-loop + FEmbedder Sel6)
    in a single fast SDXL-Turbo pass (bs=20), then frees ALL Turbo VRAM before Stage 2!
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

    pe1, _, ppe1, _ = pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
    pe1, ppe1 = pe1.half(), ppe1.half()

    N = len(c_f)
    i0_all = torch.empty((N, 3, 512, 512), dtype=torch.float32)
    for j in range(0, N, 10):
        sl = slice(j, min(N, j + 10))
        dec = pipe.vae.decode(z0_ridge[sl].to(dev, dtype=torch.float32) / pipe.vae.config.scaling_factor).sample
        i0_all[sl] = (dec * 0.5 + 0.5).clamp(0, 1).cpu()
    pipe.vae.to(dtype=dt)
    p2 = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    p2.set_progress_bar_config(disable=True)

    def gen_pass(cv_in: torch.Tensor, draw_k: int, strength: float = 0.75, amp: float = 1.0, use_ilvr: bool = True):
        cv = cv_in.float().to(dev)
        if abs(amp - 1.0) > 1e-4:
            cn = nz(cv, dim=-1)
            m = nz(mu_F.to(dev).float(), dim=-1)
            cv = cv.norm(dim=-1, keepdim=True) * nz(m + amp * (cn - m), dim=-1)
        out = torch.empty((N, 3, 512, 512), dtype=torch.uint8)
        bs = 20
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

    # Draw 0 IS B0_TurboF_1Draw (strictly single draw, zero closed-loop!)
    im_b0 = gen_pass(c_f, draw_k=0, strength=0.75, amp=1.0, use_ilvr=True)

    # 1-iteration strictly train-centered closed-loop F correction (ZERO test-batch stats!)
    f0_list, h0_list = [], []
    for j in range(0, N, 20):
        sl = slice(j, min(N, j + 20))
        parts = fe.parts(im_b0[sl].float() / 255.0)
        f0_list.append(fe.fuse(**parts).float())
        h0_list.append(nz(parts[EMB_KEYS[0]].float(), dim=-1).cpu())
    f0 = torch.cat(f0_list, 0)
    h0 = torch.cat(h0_list, 0)

    dc = cf_c
    df = nz(f0 - mu_Fd, dim=-1)
    miss = dc - (dc * df).sum(-1, keepdim=True) * df
    c_corr = nz(nz(c_f.to(dev).float(), dim=-1) + loop_w * miss, dim=-1) * c_f.to(dev).float().norm(dim=-1, keepdim=True)
    c_corr_cpu = c_corr.cpu()

    cands = [im_b0]
    h14_cands = [h0]
    scores = torch.zeros((6, N), device=dev)
    scores[0] = (df * cf_c).sum(-1)

    specs = [
        (c_f, 1, 0.75, 1.15),
        (c_corr_cpu, 2, 0.75, 1.00),
        (c_corr_cpu, 3, 0.75, 1.15),
        (c_corr_cpu, 4, 1.00, 1.05),
        (c_f, 5, 0.75, 1.25),
    ]
    for k, (cv_k, dk, st_k, am_k) in enumerate(specs, start=1):
        im_k = gen_pass(cv_k, draw_k=dk, strength=st_k, amp=am_k, use_ilvr=(st_k < 0.99))
        cands.append(im_k)
        fl, hl = [], []
        for j in range(0, N, 20):
            sl = slice(j, min(N, j + 20))
            parts = fe.parts(im_k[sl].float() / 255.0)
            fl.append(fe.fuse(**parts).float())
            hl.append(nz(parts[EMB_KEYS[0]].float(), dim=-1).cpu())
        fk = torch.cat(fl, 0)
        h14_cands.append(torch.cat(hl, 0))
        scores[k] = (nz(fk - mu_Fd, dim=-1) * cf_c).sum(-1)

    best_k = scores.argmax(dim=0).cpu()
    im_b1 = torch.empty_like(im_b0)
    h14_b1 = torch.empty((N, 1024), dtype=torch.float32)
    for i in range(N):
        ki = int(best_k[i])
        im_b1[i] = cands[ki][i]
        h14_b1[i] = h14_cands[ki][i]
    hist = [int((best_k == k).sum().item()) for k in range(6)]
    print(f"  [STAGE1 sub-{sub_id:02d}] B0+B1(Sel6) done in {time.time()-t0:.1f}s | hist={hist}", flush=True)

    del pipe, p2, attn, proj, A, fe, i0_all, cands
    torch.cuda.empty_cache()
    return im_b0, im_b1, h14_b1


class PureZeroLeakVerifier:
    """Strictly Zero-Leakage Verifier using ONLY FEmbedder (LAION H14+B32+VAE), LAION H14, and SDXL-VAE Lat16.
    Zero cross-subject extra networks, 100% blind to DINOv2 / OpenAI CLIP / CNNs / SSIM / PixCorr.
    """
    def __init__(self, vae, mu_F: torch.Tensor, mu_H: torch.Tensor, dev: str = "cuda:0"):
        self.dev = dev
        self.vae = vae
        self.fe = FEmbedder(dev=dev)
        self.mu_F = mu_F.to(dev, dtype=torch.float32)
        self.mu_H = mu_H.to(dev, dtype=torch.float32)

    @torch.inference_mode()
    def select_best(self, cand_list: list, c_f: torch.Tensor, h14_pred: torch.Tensor, z0_pred: torch.Tensor,
                    w_f: float = 0.70, w_h: float = 0.15, w_lat: float = 0.15):
        K = len(cand_list)
        N = cand_list[0].shape[0]
        cf_c = nz(c_f.to(self.dev).float() - self.mu_F, dim=-1)
        hp_c = nz(h14_pred.to(self.dev).float() - self.mu_H, dim=-1)
        z0_lo = F.avg_pool2d(z0_pred.to(self.dev).float(), 4).flatten(1)
        z0_lo = nz(z0_lo - z0_lo.mean(1, keepdim=True), dim=-1)

        sf = torch.zeros((K, N), device=self.dev)
        sh = torch.zeros((K, N), device=self.dev)
        slat = torch.zeros((K, N), device=self.dev)

        torch.cuda.empty_cache()
        for k, imgs_u8 in enumerate(cand_list):
            imgs = imgs_u8.float() / 255.0
            for j in range(0, N, 5):
                sl = slice(j, min(N, j + 5))
                xb = imgs[sl].to(self.dev)
                parts = self.fe.parts(xb)
                f_img = self.fe.fuse(**parts).float()
                sf[k, sl] = (nz(f_img - self.mu_F, dim=-1) * cf_c[sl]).sum(-1)

                h_img = nz(parts[EMB_KEYS[0]].float(), dim=-1)
                sh[k, sl] = (nz(h_img - self.mu_H, dim=-1) * hp_c[sl]).sum(-1)

                z_enc = self.vae.encode(xb * 2.0 - 1.0).latent_dist.mean * self.vae.config.scaling_factor
                z_lo = F.avg_pool2d(z_enc.float(), 4).flatten(1)
                z_lo = nz(z_lo - z_lo.mean(1, keepdim=True), dim=-1)
                slat[k, sl] = (z_lo * z0_lo[sl]).sum(-1)

        def zp(m):
            return (m - m.mean(dim=0, keepdim=True)) / (m.std(dim=0, keepdim=True) + 1e-6)

        tot = w_f * zp(sf) + w_h * zp(sh) + w_lat * zp(slat)
        best_k = tot.argmax(dim=0).cpu()
        out = torch.empty_like(cand_list[0])
        hist = [int((best_k == k).sum().item()) for k in range(K)]
        for i in range(N):
            out[i] = cand_list[int(best_k[i])][i]
        return out, hist


def run_worker(track: str, subs: list, dev: str = "cuda:0", batch_size: int = 8, num_steps: int = 18):
    if track == "b5":
        cond_dir, low_tag = "cone_b5_w1.00", "lat512_b5_v2_n0.80"
    elif track == "o8":
        cond_dir, low_tag = "cone_o8_w1.00", "lat512_o8_v2_n0.80"
    else:
        raise ValueError(f"Unknown track: {track}")

    bp = fit_fused_h14_and_prior(dev=dev)
    mu_F, mu_H, W_fh = bp["mu_F"], bp["mu_H"], bp["W_fh"]
    h14_raw_norm = bp["h14_raw_norm"]

    t_dir = OUT_ROOT / track
    t_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: Run Stage-1 (B0 + B1) for all uncached subjects FIRST, then completely free Turbo VRAM!
    s1_map = {}
    for s in subs:
        p_b0 = t_dir / "B0_TurboF_1Draw" / f"sub-{s:02d}.pt"
        p_b1 = t_dir / "B1_TurboF_Sel6" / f"sub-{s:02d}.pt"
        p_b0.parent.mkdir(parents=True, exist_ok=True)
        p_b1.parent.mkdir(parents=True, exist_ok=True)
        if p_b0.exists() and p_b1.exists():
            d0 = torch.load(p_b0, map_location="cpu", weights_only=False)
            d1 = torch.load(p_b1, map_location="cpu", weights_only=False)
            s1_map[s] = (d0["imgs_u8"], d1["imgs_u8"], d1["draft_h14"])
        else:
            c_f = torch.from_numpy(np.load(RAD / f"runs/cond/{cond_dir}/sub-{s:02d}.npy")).float()
            z0_ridge = torch.from_numpy(np.load(RAD / f"runs/low/sub-{s:02d}/{low_tag}.npy")).float()
            im_b0, im_b1, h14_b1 = run_stage1_both(c_f, z0_ridge, mu_F, sub_id=s, dev=dev)
            torch.save({"imgs_u8": im_b0}, p_b0)
            torch.save({"imgs_u8": im_b1, "draft_h14": h14_b1}, p_b1)
            s1_map[s] = (im_b0, im_b1, h14_b1)

    # Pass 2: Load SDXL-Base + PureZeroLeakVerifier + Blind Evaluator ONCE for all subjects!
    gen = PhotoUltraSDXLBase(dev=dev)
    ver = PureZeroLeakVerifier(gen.vae, mu_F, mu_H, dev=dev)
    ev = PhotoFidelityEvaluator(dev)
    dummy_p16 = torch.zeros((200, 16, 2048), dtype=torch.float32)

    for s in subs:
        c_f = torch.from_numpy(np.load(RAD / f"runs/cond/{cond_dir}/sub-{s:02d}.npy")).float().to(dev)
        z0_ridge = torch.from_numpy(np.load(RAD / f"runs/low/sub-{s:02d}/{low_tag}.npy")).float()
        im_b0, im_b1, h14_b1 = s1_map[s]

        # Encode Stage-1 B1 draft to SDXL-Base VAE latent z_draft
        z_draft = torch.empty((len(im_b1), 4, 64, 64), dtype=torch.float32)
        with torch.inference_mode():
            for j in range(0, len(im_b1), 5):
                sl = slice(j, min(len(im_b1), j + 5))
                xb = (im_b1[sl].to(dev).float() / 255.0) * 2.0 - 1.0
                z_draft[sl] = (gen.vae.encode(xb).latent_dist.mean * gen.vae.config.scaling_factor).cpu()

        h_from_f = nz(mu_H.to(dev) + (c_f - mu_F.to(dev)) @ W_fh.to(dev), dim=-1).cpu()
        h_casc = nz(0.75 * h_from_f + 0.25 * nz(h14_b1.float(), dim=-1), dim=-1)

        h_c16 = calibrate_h14(h_casc, mu_H, h14_raw_norm, angle_gain=1.60)
        h_c185 = calibrate_h14(h_casc, mu_H, h14_raw_norm, angle_gain=1.85)
        p_gen = ["a sharp realistic natural photograph, crisp object detail, natural lighting"] * len(im_b1)
        p_ov, _, n_gated = build_b5_openvocab_prompts(h_casc, mu_H, margin_thresh=0.015)

        cached = {}
        for arm in ARMS:
            cp = t_dir / arm / f"sub-{s:02d}.pt"
            cp.parent.mkdir(parents=True, exist_ok=True)
            d_ex = torch.load(cp, map_location="cpu", weights_only=False) if cp.exists() else None
            if d_ex is not None and "metrics" in d_ex:
                imgs, m = d_ex["imgs_u8"], d_ex["metrics"]
                print(
                    f"  [CACHED {track.upper()} sub-{s:02d} {arm:24s}] | "
                    f"DINO={m['dino_t1']:5.2f}% p8={m['dino_patch_cos']:.3f} CLIP={m['clip_img200_t1']:5.2f}% "
                    f"ZS200={m['zs_obj200_t1']:5.2f}% ZS1823={m['zs_obj1823_t1']:4.2f}% Crisp={m['obj_crisp']:4.1f}% "
                    f"Alex2/5={m['alex2']:4.1f}/{m['alex5']:4.1f} Incep={m['incep_2way']:4.1f} 2wC={m['clip_2way']:4.1f} "
                    f"Pix={m['pixcorr']:.3f} SSIM={m['ssim']:.3f}",
                    flush=True,
                )
            else:
                t0 = time.time()
                hist = None
                if arm == "B0_TurboF_1Draw":
                    imgs = im_b0
                elif arm == "B1_TurboF_Sel6":
                    imgs = im_b1
                elif arm == "P1_Cascade_Base_1Draw":
                    imgs = gen.generate(h_c16, dummy_p16, z0_lat=z_draft, prompts=p_gen,
                                        scale_std=0.88, scale_plus=0.0, cfg_scale=5.3, num_steps=num_steps, strength=0.58,
                                        early_anchor_steps=0, sub_id=s, draw_k=10, batch_size=batch_size)
                elif arm == "P2_Cascade_OV_1Draw":
                    imgs = gen.generate(h_c16, dummy_p16, z0_lat=z_draft, prompts=p_ov,
                                        scale_std=0.80, scale_plus=0.0, cfg_scale=5.4, num_steps=num_steps, strength=0.58,
                                        early_anchor_steps=0, sub_id=s, draw_k=11, batch_size=batch_size)
                elif arm == "P3_Cascade_Sel6":
                    # Pure all-Cascade SDXL-Base 6-candidate pool (reuses P1 & P2 + 4 fresh diverse Cascade draws!)
                    c_str52 = gen.generate(h_c16, dummy_p16, z0_lat=z_draft, prompts=p_gen,
                                           scale_std=0.86, scale_plus=0.0, cfg_scale=5.5, num_steps=num_steps, strength=0.52,
                                           early_anchor_steps=0, sub_id=s, draw_k=12, batch_size=batch_size)
                    c_str56b = gen.generate(h_c185, dummy_p16, z0_lat=z_draft, prompts=p_gen,
                                            scale_std=0.88, scale_plus=0.0, cfg_scale=5.3, num_steps=num_steps, strength=0.56,
                                            early_anchor_steps=0, sub_id=s, draw_k=13, batch_size=batch_size)
                    c_str64 = gen.generate(h_c16, dummy_p16, z0_lat=z_draft, prompts=p_gen,
                                           scale_std=0.88, scale_plus=0.0, cfg_scale=5.2, num_steps=num_steps, strength=0.64,
                                           early_anchor_steps=0, sub_id=s, draw_k=14, batch_size=batch_size)
                    c_ov62 = gen.generate(h_c185, dummy_p16, z0_lat=z_draft, prompts=p_ov,
                                          scale_std=0.80, scale_plus=0.0, cfg_scale=5.4, num_steps=num_steps, strength=0.62,
                                          early_anchor_steps=0, sub_id=s, draw_k=15, batch_size=batch_size)
                    cands = [
                        cached["P1_Cascade_Base_1Draw"],
                        cached["P2_Cascade_OV_1Draw"],
                        c_str52, c_str56b, c_str64, c_ov62,
                    ]
                    imgs, hist = ver.select_best(cands, c_f, h_casc.to(dev), z0_ridge,
                                                 w_f=0.70, w_h=0.15, w_lat=0.15)
                else:
                    raise ValueError(arm)

                torch.cuda.empty_cache()
                m = ev.evaluate(imgs, compute_lowlevel=True)
                if hist is not None:
                    m["cand_hist"] = hist
                save_d = {"imgs_u8": imgs, "metrics": m}
                if arm == "B1_TurboF_Sel6":
                    save_d["draft_h14"] = h14_b1
                torch.save(save_d, cp)
                print(
                    f"  [ALL10 {track.upper()} sub-{s:02d} {arm:24s}] ({time.time()-t0:5.1f}s) | "
                    f"DINO={m['dino_t1']:5.2f}% cos={m['dino_cos']:.3f} p8={m['dino_patch_cos']:.3f} "
                    f"CLIP={m['clip_img200_t1']:5.2f}% ZS200={m['zs_obj200_t1']:5.2f}% ZS1823={m['zs_obj1823_t1']:4.2f}% "
                    f"Crisp={m['obj_crisp']:4.1f}% Alex2/5={m['alex2']:4.1f}/{m['alex5']:4.1f} "
                    f"Incep={m['incep_2way']:4.1f} 2wC={m['clip_2way']:4.1f} Pix={m['pixcorr']:.3f} SSIM={m['ssim']:.3f}" +
                    (f" hist={hist}" if hist else ""),
                    flush=True,
                )
            cached[arm] = imgs

        # Save publication galleries for canonical subjects (sub-01, sub-08, sub-10)
        if s in (1, 8, 10):
            cols = [
                ("TurboF 1-Draw (B0)", cached["B0_TurboF_1Draw"]),
                ("TurboF Sel6 (B1)", cached["B1_TurboF_Sel6"]),
                ("Cascade 1-Draw (P1)", cached["P1_Cascade_Base_1Draw"]),
                ("Cascade+OV 1-Draw (P2)", cached["P2_Cascade_OV_1Draw"]),
                ("PhotoCascade Sel6 (P3)", cached["P3_Cascade_Sel6"]),
            ]
            save_panel(ev.gt_512, cols, ev.test_names, t_dir / f"gallery_{track}_sub-{s:02d}_first12.jpg", FIRST_12)
            if s in (8, 10):
                save_panel(ev.gt_512, cols, ev.test_names, t_dir / f"gallery_{track}_sub-{s:02d}_spaced12.jpg", SPACED_12)


def holm_bonferroni(pvals: list) -> list:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, pvals[idx] * (m - rank))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


def aggregate_track(track: str):
    t_dir = OUT_ROOT / track
    metric_keys = [
        "dino_t1", "dino_t5", "dino_2way", "dino_cos", "dino_patch_cos",
        "clip_img200_t1", "clip_img200_t5", "zs_obj200_t1", "zs_obj1823_t1", "coarse27_t1",
        "obj_crisp", "alex2", "alex5", "incep_2way", "clip_2way", "eff_dist",
        "color16_corr", "pixcorr", "ssim",
    ]
    per_arm = {}
    for arm in ARMS:
        rows = {}
        for s in range(1, 11):
            cp = t_dir / arm / f"sub-{s:02d}.pt"
            if not cp.exists():
                raise FileNotFoundError(f"Missing {cp}")
            d = torch.load(cp, map_location="cpu", weights_only=False)
            rows[f"sub-{s:02d}"] = {k: float(d["metrics"][k]) for k in metric_keys if k in d["metrics"]}
        summary = {}
        for k in metric_keys:
            vals = np.array([rows[f"sub-{s:02d}"][k] for s in range(1, 11)], dtype=np.float64)
            summary[k] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)), "min": float(vals.min()), "max": float(vals.max())}
        per_arm[arm] = {"summary": summary, "by_subject": rows}

    # Paired 10-subject tests (two-sided t-test + Cohen's d + 95% t-CI + Holm-Bonferroni across metrics)
    pairs = [
        ("P1_Cascade_Base_1Draw", "B1_TurboF_Sel6"),
        ("P2_Cascade_OV_1Draw", "B1_TurboF_Sel6"),
        ("P3_Cascade_Sel6", "B1_TurboF_Sel6"),
        ("P3_Cascade_Sel6", "B0_TurboF_1Draw"),
        ("P1_Cascade_Base_1Draw", "B0_TurboF_1Draw"),
    ]
    paired_stats = {}
    for a, b in pairs:
        p_raw_list = []
        tmp = {}
        for k in metric_keys:
            va = np.array([per_arm[a]["by_subject"][f"sub-{s:02d}"][k] for s in range(1, 11)], dtype=np.float64)
            vb = np.array([per_arm[b]["by_subject"][f"sub-{s:02d}"][k] for s in range(1, 11)], dtype=np.float64)
            diff = va - vb
            mean_d = float(diff.mean())
            sd_d = float(diff.std(ddof=1))
            se_d = sd_d / math.sqrt(len(diff))
            t_stat, p_val = stats.ttest_rel(va, vb)
            t_crit = float(stats.t.ppf(0.975, df=len(diff) - 1))
            cohen_d = mean_d / sd_d if sd_d > 1e-12 else 0.0
            wins = int((diff > 0).sum()) if k != "eff_dist" else int((diff < 0).sum())
            tmp[k] = {
                "delta": mean_d, "sd": sd_d, "ci95": [mean_d - t_crit * se_d, mean_d + t_crit * se_d],
                "t": float(t_stat), "p_raw": float(p_val), "cohen_d": float(cohen_d), "wins_of_10": wins,
            }
            p_raw_list.append(float(p_val))
        p_adj = holm_bonferroni(p_raw_list)
        for k, pa in zip(metric_keys, p_adj):
            tmp[k]["p_holm"] = float(pa)
        paired_stats[f"{a}_vs_{b}"] = tmp

    out = {"track": track, "n_subjects": 10, "n_test_images_per_sub": 200, "arms": per_arm, "paired": paired_stats}
    out_p = t_dir / f"all10_{track}_stats.json"
    out_p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n================ ALL-10 SUMMARY ({track.upper()}) ================")
    for arm in ARMS:
        sm = per_arm[arm]["summary"]
        print(
            f"  {arm:24s} | DINO={sm['dino_t1']['mean']:5.2f}+-{sm['dino_t1']['sd']:4.2f}% "
            f"p8={sm['dino_patch_cos']['mean']:.3f} | CLIP={sm['clip_img200_t1']['mean']:5.2f}% "
            f"ZS200={sm['zs_obj200_t1']['mean']:5.2f}% ZS1823={sm['zs_obj1823_t1']['mean']:4.2f}% "
            f"Crisp={sm['obj_crisp']['mean']:4.1f}% | Alex2/5={sm['alex2']['mean']:4.1f}/{sm['alex5']['mean']:4.1f} "
            f"Incep={sm['incep_2way']['mean']:4.1f} 2wC={sm['clip_2way']['mean']:4.1f} "
            f"Pix={sm['pixcorr']['mean']:.3f} SSIM={sm['ssim']['mean']:.3f}"
        )
    print(f"\n--- Key Paired Deltas ({track.upper()}, N=10, two-sided paired t + Holm-Bonferroni) ---")
    for pname in ("P1_Cascade_Base_1Draw_vs_B1_TurboF_Sel6", "P3_Cascade_Sel6_vs_B1_TurboF_Sel6", "P3_Cascade_Sel6_vs_B0_TurboF_1Draw"):
        print(f"  [{pname}]")
        for k in ("dino_t1", "clip_img200_t1", "zs_obj200_t1", "alex2", "alex5", "incep_2way", "clip_2way", "pixcorr", "ssim"):
            st = paired_stats[pname][k]
            print(f"    {k:16s}: delta={st['delta']:+7.3f} 95%CI=[{st['ci95'][0]:+7.3f},{st['ci95'][1]:+7.3f}] "
                  f"p_raw={st['p_raw']:.4g} p_holm={st['p_holm']:.4g} wins={st['wins_of_10']}/10")
    print(f"[SAVED] {out_p}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=str, default="b5", choices=["b5", "o8"])
    ap.add_argument("--subs", type=str, default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--dev", type=str, default="cuda:0")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate_track(args.track)
    else:
        subs = [int(x) for x in args.subs.split(",") if x.strip()]
        run_worker(args.track, subs, dev=args.dev, batch_size=args.bs, num_steps=args.steps)
