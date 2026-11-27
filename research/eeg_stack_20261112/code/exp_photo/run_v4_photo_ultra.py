"""PhotoEEG-Ultra (V4): Attacking the Exact Remaining Visual-Likeness Bottlenecks!
  1. Fused Multi-Stream (F_1024 + Plus16 + H14-L12) -> ON-MANIFOLD CLIP-ViT-H/14 (1024-D) Bridge
  2. High-Precision Spatial Layout (lat512_b5/o8, PixCorr~0.31!) with EARLY-ONLY Steps 1-4 Structure Anchor
     (locks 2D silhouette, aspect ratio & color palette while leaving Steps 5-20 100% free for DSLR textures!)
  3. Continuous Image-Space Flow-Matching Diffusion Prior in 1024-D CLIP-H/14 (ZERO class labels -> snaps blurry
     off-manifold EEG vectors onto crisp natural-photo objects without memorizing training classes!)
  4. Zero-Leakage Multi-Modal Spatial-Semantic-Layout Verifier (Sel8, 100% BLIND to DINOv2/OpenAI-CLIP/CNNs/SSIM/PixCorr!)
"""
import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from transformers import CLIPVisionModelWithProjection

RAD = Path(os.environ.get("RAD_ROOT", "/path/to/rad"))
for sub in ("code/exp_photo", "code/Brain-HIVE", "code/rad", "code/exp"):
    p = str(RAD / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from fenc import FEmbedder
from run_photoeeg_pipeline import IdentityImageProj, query_seed, save_gallery, build_prompt_list, predict_single_query_subject
from train_photoeeg_encoders import nz
from train_strong_h14_and_diag import calibrate_h14
from eval_photo_suite import PhotoFidelityEvaluator

PREP = RAD / "data/prep"
TURBO = RAD / "models/sdxl-turbo"
SDXL_BASE = RAD / "models/sdxl-base-unet"
IPA_DIR = RAD / "models/IP-Adapter"
H14_DIR = RAD / "models/CLIP-ViT-H-14-laion2B-s32B-b79K"
TGT_DIR = RAD / "runs/photo_eeg/targets"
MOD_DIR = RAD / "runs/photo_eeg/models"
OUT_ROOT = RAD / "runs/photo_eeg/ultra_v4"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def _lp_op(x: torch.Tensor, f_lp: int) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    lo = F.interpolate(x, size=(f_lp, f_lp), mode="bilinear", align_corners=False)
    return F.interpolate(lo, size=(h, w), mode="bilinear", align_corners=False)


# ==============================================================================
# 1. Train-Only F + Plus16 + L12 -> H14 Bridge + Continuous H14 Image Flow Prior
# ==============================================================================
class H14FlowDenoiser(nn.Module):
    """Continuous 1024-D CLIP-H/14 Image-Space Rectified Flow Prior (ZERO class labels!).
    Denoises a noisy/blurry EEG-predicted H14 embedding onto the crisp natural-image H14 hypersphere.
    """
    def __init__(self, dim: int = 1024, hidden: int = 1536, drop: float = 0.15):
        super().__init__()
        self.t_emb = nn.Sequential(nn.Linear(64, 256), nn.SiLU(), nn.Linear(256, hidden))
        self.in_proj = nn.Linear(dim * 2, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Dropout(drop),
                nn.Linear(hidden, hidden),
            )
            for _ in range(3)
        ])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim))

    def _sin_t(self, t: torch.Tensor):
        freqs = torch.exp(torch.linspace(0, math.log(100.0), 32, device=t.device))
        ang = t[:, None] * freqs[None, :] * math.pi
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(torch.cat([x_t, cond], dim=-1)) + self.t_emb(self._sin_t(t))
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


def fit_fused_h14_and_prior(dev: str = "cuda:0"):
    cache_p = MOD_DIR / "ultra_v4_bridge_and_flow.pt"
    if cache_p.exists():
        return torch.load(cache_p, map_location="cpu", weights_only=False)

    t0 = time.time()
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    H_tr = nz(torch.from_numpy(np.load(TGT_DIR / "h14_fin_train.npy")).float(), dim=-1)
    P_tr = torch.from_numpy(np.load(TGT_DIR / "plus16_train.npy").astype(np.float32))
    L_tr = torch.from_numpy(np.load(TGT_DIR / "h14_l12_train.npy").astype(np.float32)).reshape(16540, -1)

    mu_F = F_tr.mean(0, keepdim=True)
    mu_H = H_tr.mean(0, keepdim=True)
    mu_P = P_tr.mean(0, keepdim=True)
    mu_L = L_tr.mean(0, keepdim=True)

    # 1. Exact train-only Ridge bridges: F -> H14, Plus16(PCA-512) -> H14, L12(PCA-512) -> H14
    Fc = (F_tr - mu_F).to(dev)
    Hc = (H_tr - mu_H).to(dev)
    W_fh = torch.linalg.solve(Fc.T @ Fc + 10.0 * torch.eye(1024, device=dev), Fc.T @ Hc)

    torch.manual_seed(20261127)
    Pc = (P_tr - mu_P).flatten(1).to(dev)
    _, _, V_p = torch.pca_lowrank(Pc, q=512, center=False, niter=4)
    Zp = Pc @ V_p
    W_ph = torch.linalg.solve(Zp.T @ Zp + 30.0 * torch.eye(512, device=dev), Zp.T @ Hc)

    Lc = (L_tr - mu_L).to(dev)
    _, _, V_l = torch.pca_lowrank(Lc, q=512, center=False, niter=4)
    Zl = Lc @ V_l
    W_lh = torch.linalg.solve(Zl.T @ Zl + 30.0 * torch.eye(512, device=dev), Zl.T @ Hc)

    # 2. Train continuous 1024-D CLIP-H/14 Image-Space Rectified Flow Prior (ZERO class labels!)
    # Simulates EEG corruption with realistic SNR (cosine ~0.50-0.75) + isotropic/structured noise around H_tr
    H_dev = H_tr.to(dev)
    mu_H_d = mu_H.to(dev)
    H_c_u = nz(H_dev - mu_H_d, dim=-1)
    # Match empirical centered norm
    std_hc = float((H_dev - mu_H_d).norm(dim=-1).mean().item())

    flow = H14FlowDenoiser(dim=1024, hidden=1536, drop=0.15).to(dev)
    opt = torch.optim.AdamW(flow.parameters(), lr=4e-4, weight_decay=2e-2)
    B = 256
    n_steps = 1800
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-5)
    print(f"Training Continuous 1024-D CLIP-H/14 Image-Space Flow Prior ({n_steps} steps, ZERO class labels)...", flush=True)
    for st in range(1, n_steps + 1):
        idx = torch.randint(0, 16540, (B,), device=dev)
        x1 = H_dev[idx] - mu_H_d  # clean centered target [B, 1024]
        # Construct realistic EEG-like noisy condition (blend of true signal + random real-image distractor + Gaussian noise)
        snr = torch.empty((B, 1), device=dev).uniform_(0.35, 0.80)
        dist_idx = torch.randint(0, 16540, (B,), device=dev)
        noise_dir = nz(0.5 * (H_dev[dist_idx] - mu_H_d) + 0.5 * torch.randn_like(x1) * (std_hc / math.sqrt(1024)), dim=-1) * std_hc
        cond = snr * x1 + (1.0 - snr) * noise_dir
        cond = cond / cond.norm(dim=-1, keepdim=True).clamp_min(1e-5) * std_hc
        # 15% CFG unconditional drop
        drop_m = (torch.rand((B, 1), device=dev) < 0.15)
        cond_in = torch.where(drop_m, torch.zeros_like(cond), cond)

        x0 = torch.randn_like(x1) * (std_hc / math.sqrt(1024))
        t = torch.rand((B,), device=dev)
        x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1
        v_tgt = x1 - x0

        opt.zero_grad(set_to_none=True)
        v_pred = flow(x_t, cond_in, t)
        loss = F.mse_loss(v_pred, v_tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), 1.5)
        opt.step()
        sched.step()
        if st % 600 == 0:
            print(f"  [H14Flow st={st}/{n_steps}] mse={loss.item():.5f} ({time.time()-t0:.1f}s)", flush=True)

    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=False)
    h14_k = [k for k in emb_tr.keys() if "H" in k or "h14" in k.lower()][0]
    h14_raw_norm = float(emb_tr[h14_k].float().norm(dim=-1).mean().item())
    del emb_tr

    pack = {
        "mu_F": mu_F.cpu(), "mu_H": mu_H.cpu(), "mu_P": mu_P.cpu(), "mu_L": mu_L.cpu(),
        "W_fh": W_fh.cpu(), "V_p": V_p.cpu(), "W_ph": W_ph.cpu(), "V_l": V_l.cpu(), "W_lh": W_lh.cpu(),
        "flow_state": {k: v.cpu() for k, v in flow.state_dict().items()},
        "std_hc": std_hc, "h14_raw_norm": h14_raw_norm,
    }
    torch.save(pack, cache_p)
    print(f"[SAVED] {cache_p} ({time.time()-t0:.1f}s)", flush=True)
    return pack


@torch.inference_mode()
def sample_h14_flow(flow: H14FlowDenoiser, h_cond_u: torch.Tensor, mu_H: torch.Tensor, std_hc: float,
                    n_steps: int = 16, cfg: float = 2.0, n_avg: int = 4, dev: str = "cuda:0") -> torch.Tensor:
    """Sample crisp on-manifold H14 embeddings conditioned on EEG H14 (averaged over n_avg draws for stability)."""
    N = h_cond_u.shape[0]
    cond = (h_cond_u.to(dev) - mu_H.to(dev))
    cond = cond / cond.norm(dim=-1, keepdim=True).clamp_min(1e-5) * std_hc
    zeros = torch.zeros_like(cond)

    outs = []
    for k in range(n_avg):
        g = torch.Generator(device=dev).manual_seed(20261127 + k * 997)
        x = torch.randn((N, 1024), generator=g, device=dev) * (std_hc / math.sqrt(1024))
        dt = 1.0 / n_steps
        for si in range(n_steps):
            t = torch.full((N,), (si + 0.5) * dt, device=dev)
            v_u = flow(x, zeros, t)
            v_c = flow(x, cond, t)
            x = x + dt * (v_u + cfg * (v_c - v_u))
        outs.append(nz(mu_H.to(dev) + x, dim=-1))
    return nz(torch.stack(outs, 0).mean(0), dim=-1)


# ==============================================================================
# 2. SDXL-Base-1.0 Dual-IP Generator WITH Early-Only (Steps 1-4) Structure Anchor!
# ==============================================================================
class PhotoUltraSDXLBase:
    """SDXL-Base-1.0 (20-step DPM++ 2M Karras, CFG=4.5-5.5, TRUE uncond=ZERO)
    with Official Dual IP-Adapter (ip-adapter_sdxl_vit-h + ip-adapter-plus_sdxl_vit-h)
    AND Early-Only Spatial Structure Anchor (locks silhouette/color in steps 0..anchor_steps,
    leaving all remaining steps 100% free for crisp natural DSLR photography!).
    """
    def __init__(self, dev: str = "cuda:0"):
        self.dev = dev
        pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=torch.float16, variant="fp16")
        pipe.unet = UNet2DConditionModel.from_pretrained(str(SDXL_BASE / "unet"), torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(str(SDXL_BASE / "scheduler"), use_karras_sigmas=True)
        pipe.load_ip_adapter(
            [str(IPA_DIR), str(IPA_DIR)],
            subfolder=["sdxl_models", "sdxl_models"],
            weight_name=["ip-adapter_sdxl_vit-h.safetensors", "ip-adapter-plus_sdxl_vit-h.safetensors"],
            image_encoder_folder=None,
        )
        pipe.unet.encoder_hid_proj.image_projection_layers[1] = IdentityImageProj()
        pipe.to(dev)
        pipe.vae.to(dtype=torch.float32)
        pipe.vae.enable_slicing()
        self.pipe = pipe
        self.unet = pipe.unet.eval()
        self.vae = pipe.vae.eval()
        self.scheduler = pipe.scheduler
        self._pcache = {}

    @torch.inference_mode()
    def enc_p(self, txt: str):
        if txt not in self._pcache:
            pe, _, ppe, _ = self.pipe.encode_prompt(prompt=txt, device=self.dev, num_images_per_prompt=1, do_classifier_free_guidance=False)
            self._pcache[txt] = (pe.half(), ppe.half())
        return self._pcache[txt]

    @torch.inference_mode()
    def generate(
        self, h14_emb: torch.Tensor, plus16_tok: torch.Tensor, z0_lat: torch.Tensor = None,
        prompts: list = None,
        neg_prompt: str = "clay, plastic, 3d render, blurry, low contrast, muddy, monochrome, abstract, distorted, ugly, artifact",
        scale_std: float = 0.88, scale_plus: float = 0.0, cfg_scale: float = 4.8,
        num_steps: int = 20, strength: float = 0.78,
        early_anchor_steps: int = 0, early_anchor_gain: float = 0.14, early_anchor_lp: int = 8,
        sub_id: int = 1, draw_k: int = 0, batch_size: int = 5,
    ) -> torch.Tensor:
        self.pipe.set_ip_adapter_scale([scale_std, scale_plus])
        N = h14_emb.shape[0]
        if prompts is None:
            prompts = ["a sharp realistic natural photograph, crisp object detail, natural lighting"] * N
        neg_pe1, neg_ppe1 = self.enc_p(neg_prompt)
        tid1 = torch.tensor([[512.0, 512.0, 0.0, 0.0, 512.0, 512.0]], device=self.dev, dtype=torch.float16)
        u_h1 = torch.zeros((1, 1, 1024), device=self.dev, dtype=torch.float16)
        u_p1 = torch.zeros((1, 16, 2048), device=self.dev, dtype=torch.float16)

        out_u8 = torch.empty((N, 3, 512, 512), dtype=torch.uint8)
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            B = end - start
            noises = [torch.randn((4, 64, 64), generator=torch.Generator().manual_seed(query_seed(sub_id, i, draw_k))) for i in range(start, end)]
            eps = torch.stack(noises, 0).to(self.dev, dtype=torch.float16)

            pes, ppes = zip(*[self.enc_p(prompts[i]) for i in range(start, end)])
            cond_pe, cond_ppe = torch.cat(pes, 0), torch.cat(ppes, 0)
            neg_pe, neg_ppe = neg_pe1.expand(B, -1, -1), neg_ppe1.expand(B, -1)
            tid = tid1.expand(B, -1)

            c_h = h14_emb[start:end].to(self.dev, dtype=torch.float16).unsqueeze(1)
            c_p = plus16_tok[start:end].to(self.dev, dtype=torch.float16)
            u_h = u_h1.expand(B, -1, -1)
            u_p = u_p1.expand(B, -1, -1)

            self.scheduler.set_timesteps(num_steps, device=self.dev)
            timesteps = self.scheduler.timesteps
            z0_b = z0_lat[start:end].to(self.dev, dtype=torch.float16) if z0_lat is not None else None

            if z0_b is not None and strength < 0.999:
                t_start = max(0, min(num_steps - 1, int(round(num_steps * (1.0 - strength)))))
                timesteps = timesteps[t_start:]
                latents = self.scheduler.add_noise(z0_b, eps, timesteps[:1].expand(B))
            else:
                latents = eps * self.scheduler.init_noise_sigma

            for si, t in enumerate(timesteps):
                lat_in = self.scheduler.scale_model_input(torch.cat([latents, latents], 0), t)
                pred = self.unet(
                    lat_in, t, encoder_hidden_states=torch.cat([neg_pe, cond_pe], 0),
                    added_cond_kwargs={
                        "text_embeds": torch.cat([neg_ppe, cond_ppe], 0),
                        "time_ids": torch.cat([tid, tid], 0),
                        "image_embeds": [torch.cat([u_h, c_h], 0), torch.cat([u_p, c_p], 0)],
                    },
                ).sample
                nu, nc = pred.chunk(2)
                latents = self.scheduler.step(nu + cfg_scale * (nc - nu), t, latents).prev_sample

                # EARLY-ONLY soft low-frequency silhouette/color anchor (steps 0..early_anchor_steps-1 ONLY!)
                # Leaves the final ~75% of denoising steps 100% unconstrained for razor-sharp DSLR textures!
                if z0_b is not None and early_anchor_steps > 0 and si < early_anchor_steps and (si + 1) < len(timesteps):
                    t_next = timesteps[si + 1]
                    z_ref = self.scheduler.add_noise(z0_b, eps, t_next.expand(B))
                    latents = (latents.float() + early_anchor_gain * _lp_op(z_ref.float() - latents.float(), early_anchor_lp)).to(torch.float16)

            img = self.vae.decode(latents.float() / self.vae.config.scaling_factor).sample
            out_u8[start:end] = (img * 0.5 + 0.5).clamp(0, 1).mul(255).round().byte().cpu()
        return out_u8


# ==============================================================================
# 3. Zero-Leakage Multi-Modal Spatial + F-Space + Layout Photo Verifier!
# ==============================================================================
class UltraPhotoVerifier:
    """100% BLIND to DINOv2 / OpenAI CLIP / SigLIP / Inception / EffNet / AlexNet / SSIM / PixCorr!
    Uses ONLY LAION FEmbedder (H14+B32+VAE), LAION H14 Layer-12 4x4 Spatial Grid, and SDXL-VAE low-freq layout.
    """
    def __init__(self, vae, mu_F: torch.Tensor, mu_H: torch.Tensor, dev: str = "cuda:0"):
        self.dev = dev
        self.vae = vae
        self.fe = FEmbedder(dev=dev)
        self.vis_enc = self.fe.h14
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
        self.mu_F = mu_F.to(dev, dtype=torch.float32)
        self.mu_H = mu_H.to(dev, dtype=torch.float32)

    @torch.inference_mode()
    def select_best(self, cand_list: list, c_f: torch.Tensor, h14_pred: torch.Tensor,
                    l12_pred: torch.Tensor, z0_pred: torch.Tensor,
                    w_f: float = 0.40, w_h: float = 0.25, w_l12: float = 0.20, w_lat: float = 0.15):
        K = len(cand_list)
        N = cand_list[0].shape[0]
        cf_c = nz(c_f.to(self.dev).float() - self.mu_F, dim=-1)
        hp_c = nz(h14_pred.to(self.dev).float() - self.mu_H, dim=-1)
        lp = nz(l12_pred.to(self.dev).float(), dim=-1)
        z0_lo = F.avg_pool2d(z0_pred.to(self.dev).float(), 4).flatten(1)
        z0_lo = nz(z0_lo - z0_lo.mean(1, keepdim=True), dim=-1)

        sf = torch.zeros((K, N), device=self.dev)
        sh = torch.zeros((K, N), device=self.dev)
        sl12 = torch.zeros((K, N), device=self.dev)
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

                # LAION H14 Layer-12 4x4 spatial + Final H14
                x224 = F.interpolate(xb, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
                xn = (x224 - self.mean) / self.std
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = self.vis_enc(pixel_values=xn.half(), output_hidden_states=True)
                l12 = out.hidden_states[12][:, 1:, :].float()
                l12_4 = F.avg_pool2d(l12.transpose(1, 2).reshape(-1, 1280, 16, 16), 4).permute(0, 2, 3, 1).reshape(-1, 16, 1280)
                sl12[k, sl] = (nz(l12_4, dim=-1) * lp[sl]).sum(-1).mean(-1)
                fin = nz(out.image_embeds.float() - self.mu_H, dim=-1)
                sh[k, sl] = (fin * hp_c[sl]).sum(-1)

                # SDXL-VAE 16x16 low-freq spatial layout agreement
                z_enc = self.vae.encode(xb * 2.0 - 1.0).latent_dist.mean * self.vae.config.scaling_factor
                z_lo = F.avg_pool2d(z_enc.float(), 4).flatten(1)
                z_lo = nz(z_lo - z_lo.mean(1, keepdim=True), dim=-1)
                slat[k, sl] = (z_lo * z0_lo[sl]).sum(-1)

        def zp(m):
            return (m - m.mean(dim=0, keepdim=True)) / (m.std(dim=0, keepdim=True) + 1e-6)

        tot = w_f * zp(sf) + w_h * zp(sh) + w_l12 * zp(sl12) + w_lat * zp(slat)
        best_k = tot.argmax(dim=0).cpu()
        out = torch.empty_like(cand_list[0])
        hist = [int((best_k == k).sum().item()) for k in range(K)]
        for i in range(N):
            out[i] = cand_list[int(best_k[i])][i]
        return out, hist


def run_ultra(subs: list, cond_dir: str = "cone_b5_w1.00", low_tag: str = "lat512_b5_v2_n0.80",
              dev: str = "cuda:0", arms_csv: str = None, make_gallery: bool = True):
    bp = fit_fused_h14_and_prior(dev=dev)
    mu_F, mu_H = bp["mu_F"].to(dev), bp["mu_H"].to(dev)
    mu_P, mu_L = bp["mu_P"].to(dev), bp["mu_L"].to(dev)
    W_fh, V_p, W_ph = bp["W_fh"].to(dev), bp["V_p"].to(dev), bp["W_ph"].to(dev)
    V_l, W_lh = bp["V_l"].to(dev), bp["W_lh"].to(dev)
    h14_raw_norm = bp["h14_raw_norm"]

    flow = H14FlowDenoiser(dim=1024, hidden=1536, drop=0.0).to(dev).eval()
    flow.load_state_dict(bp["flow_state"])

    gen = PhotoUltraSDXLBase(dev=dev)
    ver = UltraPhotoVerifier(gen.vae, mu_F, mu_H, dev=dev)
    ev = PhotoFidelityEvaluator(dev)

    gt_h14 = nz(torch.from_numpy(np.load(TGT_DIR / "h14_fin_test.npy")).float().to(dev), dim=-1)
    idx200 = torch.arange(200, device=dev)

    arms = [a.strip() for a in arms_csv.split(",")] if arms_csv else [
        "U0_PureFH14_Ridge72_Early5",
        "U1_FusedH14_RidgeEarly4",
        "U2_FusedH14_HighAngle_Early4",
        "U3_DualIP_FusedH14_Plus16_Early4",
        "U4_FlowPriorH14_RidgeEarly4",
        "U5_FusedH14_SQPrompt2_Early4",
        "U6_PhotoUltra_VerifySel7",
    ]

    for s in subs:
        c_f = torch.from_numpy(np.load(RAD / f"runs/cond/{cond_dir}/sub-{s:02d}.npy")).float().to(dev)
        z0_ridge = torch.from_numpy(np.load(RAD / f"runs/low/sub-{s:02d}/{low_tag}.npy")).float()

        pk_p = MOD_DIR / f"sq_preds_r80_sub-{s:02d}.pt"
        pk = torch.load(pk_p, map_location="cpu", weights_only=False) if pk_p.exists() else predict_single_query_subject(s, budget="r80", dev=dev)
        z0_sharp = pk["lat"].float()
        # Blend Ridge (PixCorr ~0.31) + SharpLayoutNet (vivid color) -> best spatial layout!
        z0_hyb = 0.70 * z0_ridge + 0.30 * z0_sharp

        p16_raw = pk["p16_raw"].float().to(dev)
        l12_raw = pk["l12"].float().to(dev)

        # Fused multi-stream ON-MANIFOLD H14!
        h_from_f = nz(mu_H + (c_f - mu_F) @ W_fh, dim=-1)
        h_from_p = nz(mu_H + ((p16_raw - mu_P).flatten(1) @ V_p) @ W_ph, dim=-1)
        h_from_l = nz(mu_H + ((l12_raw.flatten(1) - mu_L) @ V_l) @ W_lh, dim=-1)
        h_fused = nz(0.75 * h_from_f + 0.15 * h_from_p + 0.10 * h_from_l, dim=-1)

        # Continuous H14 Image-Space Flow Prior denoising!
        h_flow = sample_h14_flow(flow, h_fused, mu_H, bp["std_hc"], n_steps=16, cfg=2.0, n_avg=4, dev=dev)
        h_flow_mix = nz(0.55 * h_fused + 0.45 * h_flow, dim=-1)

        # Diagnostic print of H14 retrieval (diagnostic ONLY!)
        def r200(h):
            sim = nz(h - mu_H, dim=-1) @ nz(gt_h14 - mu_H, dim=-1).T
            return (sim.argmax(-1) == idx200).float().mean().item() * 100.0, float((h * gt_h14).sum(-1).mean().item())
        rf, cf = r200(h_from_f)
        rm, cm = r200(h_fused)
        rfl, cfl = r200(h_flow_mix)
        print(f"  [ULTRA-H14 {cond_dir} sub-{s:02d}] F->H14={rf:.1f}%({cf:.3f}) | Fused={rm:.1f}%({cm:.3f}) | FlowMix={rfl:.1f}%({cfl:.3f})", flush=True)

        h_p17 = calibrate_h14(h_from_f.cpu(), mu_H.cpu(), h14_raw_norm, angle_gain=1.75)
        h_g17 = calibrate_h14(h_fused.cpu(), mu_H.cpu(), h14_raw_norm, angle_gain=1.70)
        h_g21 = calibrate_h14(h_fused.cpu(), mu_H.cpu(), h14_raw_norm, angle_gain=2.15)
        h_fl17 = calibrate_h14(h_flow_mix.cpu(), mu_H.cpu(), h14_raw_norm, angle_gain=1.65)
        p16_115 = pk["mu_p16"].float() + 1.15 * (pk["p16_raw"].float() - pk["mu_p16"].float())

        p_gen = build_prompt_list(pk["top_words"], "photo_generic")
        p_sq2 = build_prompt_list(pk["top_words"], "sq_top2")
        p_sq1 = build_prompt_list(pk["top_words"], "sq_top1")

        gal_cols = []
        sub_res = {}
        cached_imgs = {}
        for arm in arms:
            arm_d = OUT_ROOT / f"{cond_dir}_{arm}"
            arm_d.mkdir(parents=True, exist_ok=True)
            cp = arm_d / f"sub-{s:02d}.pt"
            if cp.exists():
                d = torch.load(cp, map_location="cpu", weights_only=False)
                imgs, m = d["imgs_u8"], d["metrics"]
                print(
                    f"  [CACHED sub-{s:02d} {arm:32s}] | "
                    f"DINO_T1={m['dino_t1']:5.2f}% cos={m['dino_cos']:.3f} p8={m['dino_patch_cos']:.3f} | "
                    f"CLIP_Img={m['clip_img200_t1']:5.2f}% ZS200={m['zs_obj200_t1']:5.2f}% ZS1823={m['zs_obj1823_t1']:4.2f}% | "
                    f"Crisp={m['obj_crisp']:4.1f}% Alex2/5={m['alex2']:4.1f}/{m['alex5']:4.1f} Incep={m['incep_2way']:4.1f} "
                    f"2wC={m['clip_2way']:4.1f} Pix={m['pixcorr']:.3f} SSIM={m['ssim']:.3f}",
                    flush=True,
                )
            else:
                t0 = time.time()
                hist = None
                if arm == "U0_PureFH14_Ridge72_Early5":
                    imgs = gen.generate(h_p17, p16_115, z0_lat=z0_ridge, prompts=p_gen,
                                        scale_std=0.94, scale_plus=0.0, cfg_scale=5.0, num_steps=20, strength=0.72,
                                        early_anchor_steps=5, early_anchor_gain=0.16, early_anchor_lp=10, sub_id=s, draw_k=6)
                elif arm == "U1_FusedH14_RidgeEarly4":
                    imgs = gen.generate(h_g17, p16_115, z0_lat=z0_hyb, prompts=p_gen,
                                        scale_std=0.92, scale_plus=0.0, cfg_scale=4.8, num_steps=20, strength=0.76,
                                        early_anchor_steps=4, early_anchor_gain=0.14, early_anchor_lp=8, sub_id=s, draw_k=0)
                elif arm == "U2_FusedH14_HighAngle_Early4":
                    imgs = gen.generate(h_g21, p16_115, z0_lat=z0_hyb, prompts=p_gen,
                                        scale_std=0.95, scale_plus=0.0, cfg_scale=5.2, num_steps=20, strength=0.76,
                                        early_anchor_steps=4, early_anchor_gain=0.14, early_anchor_lp=8, sub_id=s, draw_k=1)
                elif arm == "U3_DualIP_FusedH14_Plus16_Early4":
                    imgs = gen.generate(h_g17, p16_115, z0_lat=z0_hyb, prompts=p_gen,
                                        scale_std=0.78, scale_plus=0.22, cfg_scale=4.8, num_steps=20, strength=0.76,
                                        early_anchor_steps=4, early_anchor_gain=0.14, early_anchor_lp=8, sub_id=s, draw_k=2)
                elif arm == "U4_FlowPriorH14_RidgeEarly4":
                    imgs = gen.generate(h_fl17, p16_115, z0_lat=z0_hyb, prompts=p_gen,
                                        scale_std=0.92, scale_plus=0.0, cfg_scale=4.8, num_steps=20, strength=0.76,
                                        early_anchor_steps=4, early_anchor_gain=0.14, early_anchor_lp=8, sub_id=s, draw_k=3)
                elif arm == "U5_FusedH14_SQPrompt2_Early4":
                    imgs = gen.generate(h_g17, p16_115, z0_lat=z0_hyb, prompts=p_sq2,
                                        scale_std=0.82, scale_plus=0.0, cfg_scale=5.0, num_steps=20, strength=0.76,
                                        early_anchor_steps=4, early_anchor_gain=0.14, early_anchor_lp=8, sub_id=s, draw_k=4)
                elif arm == "U6_PhotoUltra_VerifySel7":
                    # Reuse already-computed U0..U5 + 1 extra fresh Pure-F->H14 high-angle draw!
                    extra = gen.generate(h_p17, p16_115, z0_lat=z0_hyb, prompts=p_gen, scale_std=0.92, scale_plus=0.0,
                                         cfg_scale=5.0, num_steps=20, strength=0.76, early_anchor_steps=4, early_anchor_gain=0.14, sub_id=s, draw_k=7)
                    cands = [cached_imgs[k] for k in [
                        "U0_PureFH14_Ridge72_Early5", "U1_FusedH14_RidgeEarly4", "U2_FusedH14_HighAngle_Early4",
                        "U3_DualIP_FusedH14_Plus16_Early4", "U4_FlowPriorH14_RidgeEarly4", "U5_FusedH14_SQPrompt2_Early4",
                    ] if k in cached_imgs] + [extra]
                    imgs, hist = ver.select_best(cands, c_f, h_from_f, l12_raw, z0_ridge,
                                                 w_f=0.45, w_h=0.25, w_l12=0.15, w_lat=0.15)
                else:
                    raise ValueError(arm)

                m = ev.evaluate(imgs, compute_lowlevel=True)
                if hist is not None:
                    m["cand_hist"] = hist
                torch.save({"imgs_u8": imgs, "metrics": m}, cp)
                print(
                    f"  [ULTRA sub-{s:02d} {arm:32s}] ({time.time()-t0:5.1f}s) | "
                    f"DINO_T1={m['dino_t1']:5.2f}% cos={m['dino_cos']:.3f} p8={m['dino_patch_cos']:.3f} | "
                    f"CLIP_Img={m['clip_img200_t1']:5.2f}% ZS200={m['zs_obj200_t1']:5.2f}% ZS1823={m['zs_obj1823_t1']:4.2f}% | "
                    f"Crisp={m['obj_crisp']:4.1f}% Alex2/5={m['alex2']:4.1f}/{m['alex5']:4.1f} Incep={m['incep_2way']:4.1f} "
                    f"2wC={m['clip_2way']:4.1f} Pix={m['pixcorr']:.3f} SSIM={m['ssim']:.3f}" +
                    (f" hist={hist}" if hist else ""),
                    flush=True,
                )
            sub_res[arm] = m
            cached_imgs[arm] = imgs
            gal_cols.append((arm.replace("FusedH14_", "").replace("PureFH14_", "PF_").replace("Ridge", "R"), imgs))

        if make_gallery and gal_cols:
            save_gallery(ev.gt_512, gal_cols, ev.test_names, OUT_ROOT / f"gallery_ultra_{cond_dir}_sub-{s:02d}.jpg")
        json.dump(sub_res, open(OUT_ROOT / f"metrics_ultra_{cond_dir}_sub-{s:02d}.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", type=str, default="10")
    ap.add_argument("--cond", type=str, default="cone_b5_w1.00")
    ap.add_argument("--low", type=str, default="lat512_b5_v2_n0.80")
    ap.add_argument("--arms", type=str, default=None)
    ap.add_argument("--dev", type=str, default="cuda:0")
    ap.add_argument("--no-gallery", action="store_true")
    args = ap.parse_args()
    subs = [int(x) for x in args.subs.split(",") if x.strip()]
    run_ultra(subs, cond_dir=args.cond, low_tag=args.low, dev=args.dev, arms_csv=args.arms, make_gallery=not args.no_gallery)
