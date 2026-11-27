"""PhotoEEG-Plus: Strictly Single-Query Photorealistic Spatial-Semantic EEG-to-Image Reconstruction.
Uses:
  - CrossBrainPlus16Net (EEG -> 16x2048 IP-Adapter-Plus spatial visual tokens + H14-L12 4x4 + H14-Final)
  - SharpLayoutNet (EEG -> sharp 4x64x64 SDXL-VAE layout + 3x16x16 vivid color map)
  - SQ-ObjSemEnc (strictly single-query CV-OpenVocab 8,141 open-world photographic prompt)
  - SDXL-Base-1.0 (DPM++ 2M Karras, CFG=5.0, photorealistic negative prompt) + IP-Adapter-Plus (16 spatial tokens)
100% STRICTLY SINGLE-QUERY (ZERO test-batch TTA, ZERO test-batch centering, ZERO evaluator leakage!).
"""
import os
import sys
import json
import time
import math
import hashlib
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from transformers import CLIPVisionModelWithProjection

RAD = Path(os.environ.get("RAD_ROOT", "/path/to/rad"))
sys.path.insert(0, str(RAD / "code/exp_photo"))

from train_photoeeg_encoders import (
    CrossBrainPlus16Net, SharpLayoutNet, SQSemNet,
    prec_erp_analytic, nz,
)
from eval_photo_suite import PhotoFidelityEvaluator

PREP = RAD / "data/prep"
TURBO = RAD / "models/sdxl-turbo"
SDXL_BASE = RAD / "models/sdxl-base-unet"
IPA_DIR = RAD / "models/IP-Adapter"
H14_DIR = RAD / "models/CLIP-ViT-H-14-laion2B-s32B-b79K"
TGT_DIR = RAD / "runs/photo_eeg/targets"
MOD_DIR = RAD / "runs/photo_eeg/models"
OUT_ROOT = RAD / "runs/photo_eeg/evals"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


class IdentityImageProj(nn.Module):
    """Bypass frozen Resampler when feeding pre-projected [B, 16, 2048] tokens directly to IP-Adapter-Plus."""
    def forward(self, image_embeds):
        if image_embeds.ndim == 4:
            return image_embeds.reshape(image_embeds.shape[0], -1, image_embeds.shape[-1])
        return image_embeds


def query_seed(sub: int, idx: int, draw_k: int = 0) -> int:
    h = hashlib.sha256(f"photoeeg-20261127/sub-{sub:02d}/q-{idx}/k-{draw_k}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31 - 1)


# ==============================================================================
# Strictly Single-Query EEG Inference (ZERO test-batch stats!)
# ==============================================================================
@torch.inference_mode()
def predict_single_query_subject(sub: int, budget: str = "r80", dev: str = "cuda:0"):
    """Predict strictly single-query plus16, h14_l12, h14_fin, sharp_lat, rgb16, and open-vocab words for 1 subject."""
    cache_p = MOD_DIR / f"sq_preds_{budget}_sub-{sub:02d}.pt"
    if cache_p.exists():
        return torch.load(cache_p, map_location="cpu", weights_only=False)

    stats = torch.load(MOD_DIR / "train_eeg_stats.pt", map_location="cpu", weights_only=True)
    eeg_mean = stats["mean"][sub - 1].to(dev)  # [1, 17, 1]
    eeg_std = stats["std"][sub - 1].to(dev)

    x_test = np.load(PREP / f"sub-{sub:02d}_test.npy", mmap_mode="r")  # [200, 80, 17, 250]
    groups = np.arange(80).reshape(20, 4).tolist() if budget == "r80" else [[0, 1, 2, 3]]

    # 1. Ensemble CrossBrainPlus16Net (all trained seeds)
    p16_ckpts = sorted(MOD_DIR.glob("plus16_seed*.pt"))
    assert p16_ckpts, "No plus16 checkpoints found!"
    p16_models = []
    for cp in p16_ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        m = CrossBrainPlus16Net(n_subs=10, d_model=384, n_queries=16, drop=0.0).to(dev).eval()
        m.load_state_dict(ck["state_dict"])
        p16_models.append((m, ck["mu_p16"].to(dev), ck["spread_p16"].to(dev), ck["mu_h14"].to(dev)))

    # 2. Ensemble SharpLayoutNet
    lay_ckpts = sorted(MOD_DIR.glob("layout_seed*.pt"))
    lay_models = []
    for cp in lay_ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        m = SharpLayoutNet(n_subs=10, drop=0.0).to(dev).eval()
        m.load_state_dict(ck["state_dict"])
        lay_models.append((m, ck["mu_lat"].to(dev), ck["gamma"]))

    # 3. Ensemble SQSemNet
    sem_ckpts = sorted(MOD_DIR.glob("sqsem_seed*.pt"))
    sem_models = []
    for cp in sem_ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        m = SQSemNet(n_subs=10, drop=0.0).to(dev).eval()
        m.load_state_dict(ck["state_dict"])
        sem_models.append(m)

    N = len(x_test)
    out_p16_raw = torch.zeros((N, 16, 2048), device=dev)
    out_p16_cal = torch.zeros((N, 16, 2048), device=dev)
    out_l12 = torch.zeros((N, 16, 1280), device=dev)
    out_h14 = torch.zeros((N, 1024), device=dev)
    out_lat = torch.zeros((N, 4, 64, 64), device=dev)
    out_rgb16 = torch.zeros((N, 3, 16, 16), device=dev)
    out_sem_h = torch.zeros((N, 1024), device=dev)
    out_sem_b = torch.zeros((N, 1280), device=dev)

    # Strictly single-query loop (every query uses ONLY its own EEG + fixed train stats!)
    for start in range(0, N, 40):
        sl = slice(start, min(N, start + 40))
        xb_all = torch.from_numpy(np.array(x_test[sl], copy=True)).to(dev, dtype=torch.float32)  # [B, 80, 17, 250]
        B = xb_all.shape[0]
        sid = torch.full((B,), sub - 1, device=dev, dtype=torch.long)
        sid_t = torch.full((B * 4,), sub - 1, device=dev, dtype=torch.long)

        g_p16_raw, g_p16_cal, g_l12, g_h14 = 0, 0, 0, 0
        g_lat, g_rgb, g_sh, g_sb = 0, 0, 0, 0
        for grp in groups:
            bb = xb_all[:, grp]  # [B, 4, 17, 250]
            # M1a dual-stream: 0.72 * PrecERP + 0.28 * TrialSet (each standardized by TRAIN-ONLY mean/std!)
            x_erp = (prec_erp_analytic(bb, p=0.5) - eeg_mean) / eeg_std
            x_tri = (bb.reshape(B * 4, 17, 250) - eeg_mean) / eeg_std

            # Plus16 ensemble
            for m, mu_p, sp_p, mu_h in p16_models:
                p_e, l_e, h_e, _ = m(x_erp, sid)
                p_t, l_t, h_t, _ = m(x_tri, sid_t)
                p = 0.72 * p_e + 0.28 * p_t.view(B, 4, 16, 2048).mean(1)
                l = 0.72 * l_e + 0.28 * l_t.view(B, 4, 16, 1280).mean(1)
                h = 0.72 * h_e + 0.28 * h_t.view(B, 4, 1024).mean(1)
                g_p16_raw = g_p16_raw + p / len(p16_models)
                # Train-only radial spread calibration (per-token, strictly single-query!)
                p_cal = mu_p + (p - mu_p) * sp_p.view(1, 16, 1)
                g_p16_cal = g_p16_cal + p_cal / len(p16_models)
                g_l12 = g_l12 + nz(l, dim=-1) / len(p16_models)
                g_h14 = g_h14 + nz(h - mu_h, dim=-1) / len(p16_models)

            # SharpLayout ensemble
            for m, mu_l, gam in lay_models:
                pl, pr = m(x_erp, sid)
                g_lat = g_lat + (mu_l + gam * pl) / len(lay_models)
                g_rgb = g_rgb + pr / len(lay_models)

            # SQSem ensemble
            for m in sem_models:
                sh_e, sb_e = m(x_erp, sid)
                sh_t, sb_t = m(x_tri, sid_t)
                sh = 0.72 * nz(sh_e) + 0.28 * nz(sh_t.view(B, 4, 1024).mean(1))
                sb = 0.72 * nz(sb_e) + 0.28 * nz(sb_t.view(B, 4, 1280).mean(1))
                g_sh = g_sh + nz(sh) / len(sem_models)
                g_sb = g_sb + nz(sb) / len(sem_models)

        ng = float(len(groups))
        out_p16_raw[sl] = g_p16_raw / ng
        out_p16_cal[sl] = g_p16_cal / ng
        out_l12[sl] = nz(g_l12 / ng, dim=-1)
        out_h14[sl] = nz(g_h14 / ng, dim=-1)
        out_lat[sl] = g_lat / ng
        out_rgb16[sl] = (g_rgb / ng).clamp(0, 1)
        out_sem_h[sl] = nz(g_sh / ng, dim=-1)
        out_sem_b[sl] = nz(g_sb / ng, dim=-1)

    # Decode strictly single-query top-5 CV-OpenVocab (8,141) words (NO test-batch centering!)
    tb = torch.load(RAD / "runs/obj_bench/text_banks/text_banks_laion.pt", map_location="cpu", weights_only=False)
    tr_cid = tb["tr_img2cid"].long()
    mu_th = tb["h14_train"][tr_cid].float().mean(0, keepdim=True)
    mu_tb = tb["bigg_train"][tr_cid].float().mean(0, keepdim=True)
    H_cv = nz(tb["h14_cv_open"].float() - mu_th, dim=-1).to(dev)
    B_cv = nz(tb["bigg_cv_open"].float() - mu_tb, dim=-1).to(dev)
    words = tb["cv_open"]

    sim_cv = 0.50 * (out_sem_h @ H_cv.T) + 0.50 * (out_sem_b @ B_cv.T)  # [200, 8141]
    top_vals, top_idx = sim_cv.topk(5, dim=-1)
    top_words = [[words[int(i)] for i in row] for row in top_idx.cpu().tolist()]

    # Quick diagnostic: check Plus16 and H14 retrieval on unseen test set (diagnostic print ONLY!)
    gt_p16 = torch.from_numpy(np.load(TGT_DIR / "plus16_test.npy")).float().to(dev)
    mu_p16_tr = p16_models[0][1]
    dp_pred = (out_p16_raw - mu_p16_tr).flatten(1)
    dp_gt = (gt_p16 - mu_p16_tr).flatten(1)
    s_p16 = nz(dp_pred) @ nz(dp_gt).T
    idx200 = torch.arange(200, device=dev)
    p16_t1 = (s_p16.argmax(-1) == idx200).float().mean().item() * 100.0
    p16_t5 = (s_p16.topk(5, -1).indices == idx200[:, None]).any(-1).float().mean().item() * 100.0

    gt_h14 = torch.from_numpy(np.load(TGT_DIR / "h14_fin_test.npy")).float().to(dev)
    mu_h14_tr = p16_models[0][3]
    s_h14 = out_h14 @ nz(gt_h14 - mu_h14_tr).T
    h14_t1 = (s_h14.argmax(-1) == idx200).float().mean().item() * 100.0

    print(f"  [SQ-Preds {budget} sub-{sub:02d}] Plus16-200w={p16_t1:.1f}%/{p16_t5:.1f}% | "
          f"H14-200w={h14_t1:.1f}% | ex0-3 words: {[w[0] for w in top_words[:4]]}", flush=True)

    pack = {
        "p16_raw": out_p16_raw.half().cpu(),
        "p16_cal": out_p16_cal.half().cpu(),
        "mu_p16": mu_p16_tr.half().cpu(),
        "l12": out_l12.half().cpu(),
        "h14": out_h14.half().cpu(),
        "lat": out_lat.half().cpu(),
        "rgb16": out_rgb16.half().cpu(),
        "sem_h": out_sem_h.half().cpu(),
        "sem_b": out_sem_b.half().cpu(),
        "top_words": top_words,
        "top_vals": top_vals.cpu(),
        "diag": {"p16_t1": p16_t1, "p16_t5": p16_t5, "h14_t1": h14_t1},
    }
    torch.save(pack, cache_p)
    return pack


# ==============================================================================
# High-Fidelity SDXL-Base-1.0 + IP-Adapter-Plus (16 Spatial Tokens) Renderer
# ==============================================================================
class PhotoEEGPlusGenerator:
    def __init__(self, backend: str = "sdxl-base", dev: str = "cuda:0"):
        self.dev = dev
        self.backend = backend
        pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=torch.float16, variant="fp16")
        if backend == "sdxl-base":
            pipe.unet = UNet2DConditionModel.from_pretrained(
                str(SDXL_BASE / "unet"), torch_dtype=torch.float16, variant="fp16"
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                str(SDXL_BASE / "scheduler"), use_karras_sigmas=True
            )
        pipe.load_ip_adapter(
            str(IPA_DIR), subfolder="sdxl_models",
            weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
            image_encoder_folder=None,
        )
        # Swap Resampler to Identity so we directly inject [B, 16, 2048] spatial visual tokens!
        pipe.unet.encoder_hid_proj.image_projection_layers[0] = IdentityImageProj()
        pipe.to(dev)
        pipe.vae.to(dtype=torch.float32)  # crisp fp32 VAE decode, zero NaN/banding!
        self.pipe = pipe
        self.unet = pipe.unet.eval()
        self.vae = pipe.vae.eval()
        self.scheduler = pipe.scheduler
        self._prompt_cache = {}

    @torch.inference_mode()
    def encode_prompt_cached(self, text: str):
        if text not in self._prompt_cache:
            pe, _, ppe, _ = self.pipe.encode_prompt(
                prompt=text, device=self.dev, num_images_per_prompt=1, do_classifier_free_guidance=False
            )
            self._prompt_cache[text] = (pe.half(), ppe.half())
        return self._prompt_cache[text]

    @torch.inference_mode()
    def generate(
        self,
        tokens16: torch.Tensor,          # [N, 16, 2048]
        z0_lat: torch.Tensor = None,     # [N, 4, 64, 64] SharpLayoutNet latent (optional SDEdit init)
        prompts: list = None,            # list[str] len N, or None -> natural photo generic prompt
        neg_prompt: str = "clay, plastic, 3d render, blurry, low contrast, monochrome gray, abstract block, distorted, ugly",
        ip_scale: float = 0.85,
        cfg_scale: float = 5.0,
        num_steps: int = 18,
        strength: float = 0.80,          # SDEdit noise strength when z0_lat is provided (0.72-0.88)
        sub_id: int = 1,
        draw_k: int = 0,
        batch_size: int = 10,
        uncond_tokens16: torch.Tensor = None,
    ) -> torch.Tensor:
        """Render [N, 3, 512, 512] uint8 RGB images with SDXL + IP-Adapter-Plus (16 spatial tokens)."""
        self.pipe.set_ip_adapter_scale(ip_scale)
        N = tokens16.shape[0]
        if prompts is None:
            prompts = ["a sharp realistic natural photograph, crisp detail, natural lighting"] * N

        neg_pe_1, neg_ppe_1 = self.encode_prompt_cached(neg_prompt)
        time_ids_1 = torch.tensor([[512.0, 512.0, 0.0, 0.0, 512.0, 512.0]], device=self.dev, dtype=torch.float16)

        if uncond_tokens16 is None:
            uncond_tokens16 = torch.zeros((1, 16, 2048), device=self.dev, dtype=torch.float16)
        else:
            uncond_tokens16 = uncond_tokens16.to(self.dev, dtype=torch.float16).reshape(1, 16, 2048)

        out_u8 = torch.empty((N, 3, 512, 512), dtype=torch.uint8)

        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            B = end - start

            # Deterministic per-query noise
            noises = []
            for qi in range(start, end):
                g = torch.Generator(device="cpu").manual_seed(query_seed(sub_id, qi, draw_k))
                noises.append(torch.randn((4, 64, 64), generator=g))
            eps = torch.stack(noises, 0).to(self.dev, dtype=torch.float16)

            # Encode prompts for batch
            pes, ppes = zip(*[self.encode_prompt_cached(prompts[i]) for i in range(start, end)])
            cond_pe = torch.cat(pes, 0)
            cond_ppe = torch.cat(ppes, 0)
            neg_pe = neg_pe_1.expand(B, -1, -1)
            neg_ppe = neg_ppe_1.expand(B, -1)
            time_ids = time_ids_1.expand(B, -1)

            cond_tok = tokens16[start:end].to(self.dev, dtype=torch.float16)
            uncond_tok = uncond_tokens16.expand(B, -1, -1)

            # Setup scheduler & SDEdit start timestep
            self.scheduler.set_timesteps(num_steps, device=self.dev)
            timesteps = self.scheduler.timesteps
            if z0_lat is not None and strength < 0.999:
                t_start = max(0, min(num_steps - 1, int(round(num_steps * (1.0 - strength)))))
                timesteps = timesteps[t_start:]
                z0_b = z0_lat[start:end].to(self.dev, dtype=torch.float16)
                latents = self.scheduler.add_noise(z0_b, eps, timesteps[:1].expand(B))
            else:
                latents = eps * self.scheduler.init_noise_sigma

            do_cfg = (cfg_scale > 1.01)
            for t in timesteps:
                if do_cfg:
                    lat_in = torch.cat([latents, latents], dim=0)
                    lat_in = self.scheduler.scale_model_input(lat_in, t)
                    pe_in = torch.cat([neg_pe, cond_pe], dim=0)
                    ppe_in = torch.cat([neg_ppe, cond_ppe], dim=0)
                    tid_in = torch.cat([time_ids, time_ids], dim=0)
                    ip_in = torch.cat([uncond_tok, cond_tok], dim=0)
                    noise_pred = self.unet(
                        lat_in, t, encoder_hidden_states=pe_in,
                        added_cond_kwargs={"text_embeds": ppe_in, "time_ids": tid_in, "image_embeds": [ip_in]},
                    ).sample
                    n_uncond, n_cond = noise_pred.chunk(2)
                    noise_pred = n_uncond + cfg_scale * (n_cond - n_uncond)
                else:
                    lat_in = self.scheduler.scale_model_input(latents, t)
                    noise_pred = self.unet(
                        lat_in, t, encoder_hidden_states=cond_pe,
                        added_cond_kwargs={"text_embeds": cond_ppe, "time_ids": time_ids, "image_embeds": [cond_tok]},
                    ).sample
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Crisp fp32 VAE decode
            img = self.vae.decode(latents.float() / self.vae.config.scaling_factor).sample
            img = (img * 0.5 + 0.5).clamp(0, 1).mul(255).round().byte().cpu()
            out_u8[start:end] = img

        return out_u8


# ==============================================================================
# Zero-Leakage Single-Query Spatial-Semantic Photo Verifier (ONLY LAION H14 + RGB16!)
# ==============================================================================
class SpatialPhotoVerifier:
    """Strictly zero-leakage verifier (ONLY LAION CLIP-ViT-H/14 Layer-12 4x4 + Final H14 + RGB16).
    100% BLIND to DINOv2, OpenAI ViT-L/14, SigLIP, Inception, EffNet, AlexNet!
    """
    def __init__(self, dev: str = "cuda:0"):
        self.dev = dev
        self.vis_enc = CLIPVisionModelWithProjection.from_pretrained(str(H14_DIR), torch_dtype=torch.float16).to(dev).eval()
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
        # Train-only H14 mean for strictly single-query centering!
        self.mu_h14 = torch.from_numpy(np.load(TGT_DIR / "h14_fin_train.npy").astype(np.float32).mean(0, keepdims=True)).to(dev)
        self.mu_rgb = torch.from_numpy(np.load(TGT_DIR / "rgb16_train.npy").astype(np.float32).mean(0, keepdims=True)).to(dev)

    @torch.inference_mode()
    def select_best(self, cand_list: list, pred_l12: torch.Tensor, pred_h14: torch.Tensor,
                    pred_sem_h: torch.Tensor, pred_rgb16: torch.Tensor,
                    w_l12: float = 0.40, w_h14: float = 0.35, w_sem: float = 0.10, w_rgb: float = 0.15):
        K = len(cand_list)
        N = cand_list[0].shape[0]
        pl12 = nz(pred_l12.to(self.dev), dim=-1)    # [N, 16, 1280]
        ph14 = nz(pred_h14.to(self.dev), dim=-1)    # [N, 1024]
        psem = nz(pred_sem_h.to(self.dev), dim=-1)  # [N, 1024]
        prgb = (pred_rgb16.to(self.dev).float() - self.mu_rgb).flatten(1)
        prgb = nz(prgb, dim=-1)

        s_l12 = torch.zeros((K, N), device=self.dev)
        s_h14 = torch.zeros((K, N), device=self.dev)
        s_sem = torch.zeros((K, N), device=self.dev)
        s_rgb = torch.zeros((K, N), device=self.dev)

        for k, imgs_u8 in enumerate(cand_list):
            imgs = imgs_u8.float() / 255.0
            for j in range(0, N, 40):
                sl = slice(j, min(N, j + 40))
                xb = imgs[sl].to(self.dev)
                rgb_k = (F.adaptive_avg_pool2d(xb, (16, 16)) - self.mu_rgb).flatten(1)
                s_rgb[k, sl] = (nz(rgb_k) * prgb[sl]).sum(-1)

                x224 = F.interpolate(xb, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
                xn = (x224 - self.mean) / self.std
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = self.vis_enc(pixel_values=xn.half(), output_hidden_states=True)
                l12 = out.hidden_states[12][:, 1:, :].float()
                l12_4 = F.avg_pool2d(l12.transpose(1, 2).reshape(-1, 1280, 16, 16), 4).permute(0, 2, 3, 1).reshape(-1, 16, 1280)
                l12_4 = nz(l12_4, dim=-1)
                s_l12[k, sl] = (l12_4 * pl12[sl]).sum(-1).mean(-1)

                fin = nz(out.image_embeds.float() - self.mu_h14, dim=-1)
                s_h14[k, sl] = (fin * ph14[sl]).sum(-1)
                s_sem[k, sl] = (fin * psem[sl]).sum(-1)

        # Strictly per-query z-score across the K candidates of query i ONLY (zero cross-query dependence!)
        def z_per_query(m):
            return (m - m.mean(dim=0, keepdim=True)) / (m.std(dim=0, keepdim=True) + 1e-6)

        total = w_l12 * z_per_query(s_l12) + w_h14 * z_per_query(s_h14) + w_sem * z_per_query(s_sem) + w_rgb * z_per_query(s_rgb)
        best_k = total.argmax(dim=0).cpu()
        out = torch.empty_like(cand_list[0])
        hist = [int((best_k == k).sum().item()) for k in range(K)]
        for i in range(N):
            out[i] = cand_list[int(best_k[i])][i]
        return out, hist


def save_gallery(gt_512: torch.Tensor, cols: list, names: list, out_path: Path, n_rows: int = 12, cell: int = 224):
    """cols: list of (label, imgs_u8 [200, 3, 512, 512])."""
    ncols = 1 + len(cols)
    hdr, lw = 32, 130
    W = lw + ncols * (cell + 4) + 8
    H = hdr + n_rows * (cell + 4) + 8
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    headers = ["GT Original"] + [c[0] for c in cols]
    for ci, h in enumerate(headers):
        draw.text((lw + ci * (cell + 4) + 6, 8), h, fill=(10, 10, 10))

    for ri in range(n_rows):
        y = hdr + ri * (cell + 4)
        draw.text((6, y + cell // 2 - 8), f"#{ri:03d} {names[ri][:15]}", fill=(20, 20, 20))
        gt_im = Image.fromarray(gt_512[ri].mul(255).byte().permute(1, 2, 0).numpy()).resize((cell, cell), Image.BICUBIC)
        canvas.paste(gt_im, (lw, y))
        for ci, (_, arr) in enumerate(cols):
            im = Image.fromarray(arr[ri].permute(1, 2, 0).numpy()).resize((cell, cell), Image.BICUBIC)
            canvas.paste(im, (lw + (ci + 1) * (cell + 4), y))
    canvas.save(out_path, quality=92)
    print(f"[GALLERY SAVED] {out_path}", flush=True)


def run_oracle(dev: str = "cuda:0"):
    """Verify IP-Adapter-Plus 16-token + SDXL-Base Oracle ceiling on all 200 unseen test photos!"""
    gen = PhotoEEGPlusGenerator("sdxl-base", dev=dev)
    gt_p16 = torch.from_numpy(np.load(TGT_DIR / "plus16_test.npy")).float()
    imgs = gen.generate(gt_p16, z0_lat=None, ip_scale=0.95, cfg_scale=5.0, num_steps=18, sub_id=0, draw_k=0, batch_size=8)
    del gen
    torch.cuda.empty_cache()
    ev = PhotoFidelityEvaluator(dev)
    m = ev.evaluate(imgs, compute_lowlevel=True)
    print(f"[ORACLE Plus16 SDXL-Base] DINO_T1={m['dino_t1']:.2f}% DINO_cos={m['dino_cos']:.3f} DINO_p8={m['dino_patch_cos']:.3f} | "
          f"CLIP_Img={m['clip_img200_t1']:.2f}% ZS200={m['zs_obj200_t1']:.2f}% ZS1823={m['zs_obj1823_t1']:.2f}% | "
          f"Crisp={m['obj_crisp']:.1f}% Alex2/5={m['alex2']:.1f}/{m['alex5']:.1f} Incep={m['incep_2way']:.1f} "
          f"2wC={m['clip_2way']:.1f} Pix={m['pixcorr']:.3f} SSIM={m['ssim']:.3f}", flush=True)
    save_gallery(ev.gt_512, [("Oracle Plus16 SDXL-Base", imgs)], ev.test_names, OUT_ROOT / "oracle_plus16_gallery.jpg")
    json.dump(m, open(OUT_ROOT / "oracle_plus16_metrics.json", "w"), indent=2)


def build_prompt_list(top_words: list, mode: str) -> list:
    if mode == "photo_generic":
        return ["a sharp realistic natural photograph, crisp object detail, natural lighting"] * len(top_words)
    elif mode == "sq_top2":
        return [f"a realistic natural photograph of {w[0]} or {w[1]}, sharp focus, natural lighting" for w in top_words]
    elif mode == "sq_top1":
        return [f"a crisp realistic photograph of a {w[0]}, centered, sharp focus, natural lighting" for w in top_words]
    raise ValueError(mode)


def run_arms(subs: list, arms: list, budget: str = "r80", dev: str = "cuda:0", make_gallery: bool = False):
    ev = PhotoFidelityEvaluator(dev)
    gen_base = PhotoEEGPlusGenerator("sdxl-base", dev=dev)
    ver = SpatialPhotoVerifier(dev=dev) if any("Verify" in a for a in arms) else None

    for s in subs:
        pk = predict_single_query_subject(s, budget=budget, dev=dev)
        mu_p = pk["mu_p16"]
        p16_raw = pk["p16_raw"].float()
        p16_cal = pk["p16_cal"].float()
        # Tunable radial calibration helper (strictly train-only mu_p!)
        def spread_p16(alpha: float):
            return mu_p.float() + alpha * (p16_cal - mu_p.float())

        gal_cols = []
        for arm in arms:
            arm_dir = OUT_ROOT / f"{budget}_{arm}"
            arm_dir.mkdir(parents=True, exist_ok=True)
            cp = arm_dir / f"sub-{s:02d}.pt"
            if cp.exists():
                d = torch.load(cp, map_location="cpu", weights_only=False)
                m = d["metrics"]
                imgs = d["imgs_u8"]
            else:
                t0 = time.time()
                hist = None
                if arm == "P1_Base20_NoLat_Cal":
                    imgs = gen_base.generate(
                        p16_cal, z0_lat=None, prompts=build_prompt_list(pk["top_words"], "photo_generic"),
                        ip_scale=0.90, cfg_scale=5.0, num_steps=18, sub_id=s, draw_k=0, uncond_tokens16=mu_p,
                    )
                elif arm == "P2_Base20_SDE82_Cal":
                    imgs = gen_base.generate(
                        p16_cal, z0_lat=pk["lat"], prompts=build_prompt_list(pk["top_words"], "photo_generic"),
                        ip_scale=0.90, cfg_scale=5.0, num_steps=20, strength=0.82, sub_id=s, draw_k=0, uncond_tokens16=mu_p,
                    )
                elif arm == "P3_Base20_SDE74_Cal":
                    imgs = gen_base.generate(
                        p16_cal, z0_lat=pk["lat"], prompts=build_prompt_list(pk["top_words"], "photo_generic"),
                        ip_scale=0.90, cfg_scale=5.0, num_steps=20, strength=0.74, sub_id=s, draw_k=0, uncond_tokens16=mu_p,
                    )
                elif arm == "P4_Base20_SDE78_SQPrompt2":
                    imgs = gen_base.generate(
                        p16_cal, z0_lat=pk["lat"], prompts=build_prompt_list(pk["top_words"], "sq_top2"),
                        ip_scale=0.80, cfg_scale=5.0, num_steps=20, strength=0.78, sub_id=s, draw_k=0, uncond_tokens16=mu_p,
                    )
                elif arm == "P5_Base20_SDE78_SQPrompt1":
                    imgs = gen_base.generate(
                        p16_cal, z0_lat=pk["lat"], prompts=build_prompt_list(pk["top_words"], "sq_top1"),
                        ip_scale=0.72, cfg_scale=5.5, num_steps=20, strength=0.78, sub_id=s, draw_k=0, uncond_tokens16=mu_p,
                    )
                elif arm == "P6_Base20_SDE78_Spread125":
                    imgs = gen_base.generate(
                        spread_p16(1.25), z0_lat=pk["lat"], prompts=build_prompt_list(pk["top_words"], "photo_generic"),
                        ip_scale=0.92, cfg_scale=5.0, num_steps=20, strength=0.78, sub_id=s, draw_k=0, uncond_tokens16=mu_p,
                    )
                elif arm == "P7_PhotoVerify_Sel4":
                    # 4 diverse photorealistic candidates ranked strictly single-query by zero-leakage LAION H14-L12 + H14 + RGB16!
                    cands = [
                        gen_base.generate(
                            p16_cal, z0_lat=pk["lat"], prompts=build_prompt_list(pk["top_words"], "photo_generic"),
                            ip_scale=0.90, cfg_scale=5.0, num_steps=20, strength=0.78, sub_id=s, draw_k=0, uncond_tokens16=mu_p,
                        ),
                        gen_base.generate(
                            spread_p16(1.20), z0_lat=pk["lat"], prompts=build_prompt_list(pk["top_words"], "photo_generic"),
                            ip_scale=0.92, cfg_scale=5.0, num_steps=20, strength=0.74, sub_id=s, draw_k=1, uncond_tokens16=mu_p,
                        ),
                        gen_base.generate(
                            p16_cal, z0_lat=pk["lat"], prompts=build_prompt_list(pk["top_words"], "sq_top2"),
                            ip_scale=0.80, cfg_scale=5.0, num_steps=20, strength=0.78, sub_id=s, draw_k=2, uncond_tokens16=mu_p,
                        ),
                        gen_base.generate(
                            p16_cal, z0_lat=pk["lat"], prompts=build_prompt_list(pk["top_words"], "sq_top1"),
                            ip_scale=0.72, cfg_scale=5.5, num_steps=20, strength=0.80, sub_id=s, draw_k=3, uncond_tokens16=mu_p,
                        ),
                    ]
                    imgs, hist = ver.select_best(cands, pk["l12"], pk["h14"], pk["sem_h"], pk["rgb16"])
                else:
                    raise ValueError(f"Unknown arm: {arm}")

                m = ev.evaluate(imgs, compute_lowlevel=True)
                if hist is not None:
                    m["cand_hist"] = hist
                torch.save({"imgs_u8": imgs, "metrics": m}, cp)
                print(
                    f"  [{budget}:{arm:24s}] sub-{s:02d} ({time.time()-t0:5.1f}s) | "
                    f"DINO_T1={m['dino_t1']:5.2f}% cos={m['dino_cos']:.3f} p8={m['dino_patch_cos']:.3f} | "
                    f"CLIP_Img={m['clip_img200_t1']:5.2f}% ZS200={m['zs_obj200_t1']:5.2f}% ZS1823={m['zs_obj1823_t1']:4.2f}% | "
                    f"Crisp={m['obj_crisp']:4.1f}% Alex2/5={m['alex2']:4.1f}/{m['alex5']:4.1f} Incep={m['incep_2way']:4.1f} "
                    f"2wC={m['clip_2way']:4.1f} Pix={m['pixcorr']:.3f} SSIM={m['ssim']:.3f}" +
                    (f" hist={hist}" if hist else ""),
                    flush=True,
                )
            gal_cols.append((arm.replace("Base20_", ""), imgs))

        if make_gallery and gal_cols:
            save_gallery(ev.gt_512, gal_cols, ev.test_names, OUT_ROOT / f"gallery_{budget}_sub-{s:02d}.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["oracle", "eval"])
    ap.add_argument("--subs", type=str, default="1,2,8,10")
    ap.add_argument("--arms", type=str, default="P1_Base20_NoLat_Cal,P2_Base20_SDE82_Cal,P3_Base20_SDE74_Cal,P4_Base20_SDE78_SQPrompt2,P5_Base20_SDE78_SQPrompt1,P6_Base20_SDE78_Spread125,P7_PhotoVerify_Sel4")
    ap.add_argument("--budget", type=str, default="r80", choices=["r80", "r4"])
    ap.add_argument("--dev", type=str, default="cuda:0")
    ap.add_argument("--gallery", action="store_true")
    args = ap.parse_args()
    if args.mode == "oracle":
        run_oracle(args.dev)
    else:
        subs = [int(x) for x in args.subs.split(",") if x.strip()]
        arms = [x.strip() for x in args.arms.split(",") if x.strip()]
        run_arms(subs, arms, budget=args.budget, dev=args.dev, make_gallery=args.gallery)


if __name__ == "__main__":
    main()
