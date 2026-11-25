"""Stage 2 & 3 of ObjDRaFT-Bridge: End-to-End 4-Step SDXL-Turbo Object-Recognizability Bridge Training & All-10 Eval.
100% ZERO test data during training. 100% ZERO OpenAI ViT-L/14 during training (only LAION H14 + DINOv2-base!).
"""
import os, sys, gc, math, time, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from PIL import Image

RAD = Path(os.environ.get("RAD_ROOT", "/path/to/rad"))
for sub in ("code/exp_obj", "code/exp_w7", "code/exp_w6", "code/rad", "code/exp", "code/Brain-HIVE"):
    p = str(RAD / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from main.models_adapter import IPProjectionModel, IPAttnAdapterModel, FusionEncoderModel
from eval_obj_suite import ObjReconEvaluator
from run_objeeg_v2 import DualStreamSemanticEngine, LAIONH14Reranker, log_and_save, _lp_op

PREP = RAD / "data/prep"
TURBO = RAD / "models/sdxl-turbo"
PRIOR = RAD / "models/Fusion-Prior-H14_B32_VAE"
H14_DIR = RAD / "models/CLIP-ViT-H-14-laion2B-s32B-b79K"
DINO_DIR = RAD / "models/dinov2-base"
OOF_DIR = RAD / "runs/obj_bench/oof_unseen"
BRIDGE_ROOT = RAD / "runs/obj_bench/objdraft_bridge"


def nz(x, dim=-1):
    return F.normalize(x.float(), dim=dim)


class ObjDRaFTBridge(nn.Module):
    """Lightweight Zero-Init Residual Bridge: (c_eeg, h_sem, b_sem) -> denoised IP tokens + 8 native SDXL dual-text tokens + ppe!"""
    def __init__(self, n_prompt_tokens: int = 8, hidden: int = 1024):
        super().__init__()
        self.n_tok = n_prompt_tokens
        in_dim = 1024 + 1024 + 1280  # c_u (1024) + h_sem (1024) + b_sem (1280) = 3328
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        # Residual denoiser on Fusion-Prior 1024-D conditioning BEFORE frozen adapter+proj (ultra-clean, on-manifold!)
        self.head_c_delta = nn.Linear(hidden, 1024)
        # Residual refinement directly on IP-Adapter 4x2048 tokens
        self.head_ip_delta = nn.Linear(hidden, 4 * 2048)
        # Native SDXL Dual-Text Cross-Attention tokens (positions 1..n_tok inside [B, 77, 2048])
        self.head_pe_tokens = nn.Linear(hidden, n_prompt_tokens * 2048)
        # Native SDXL Pooled Text Embedding delta ([B, 1280])
        self.head_ppe_delta = nn.Linear(hidden, 1280)

        # Zero-init all residual outputs so Step 0 = exact pretrained O14!
        for h in (self.head_c_delta, self.head_ip_delta, self.head_pe_tokens, self.head_ppe_delta):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, c_eeg: torch.Tensor, h_sem: torch.Tensor, b_sem: torch.Tensor,
                adapter_frozen, proj_frozen, pe_empty: torch.Tensor, ppe_empty: torch.Tensor,
                text_only: bool = False):
        c_norm = c_eeg.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        c_u = c_eeg.float() / c_norm
        feat = torch.cat([c_u, nz(h_sem), nz(b_sem)], dim=-1)
        h = self.trunk(feat)

        # 1. Denoised F-space vector -> frozen Fusion-Prior adapter + proj + small residual
        ad_dt = next(adapter_frozen.parameters()).dtype
        if text_only:
            dc = torch.zeros_like(c_u)
            ips = proj_frozen(adapter_frozen(c_eeg.to(ad_dt))).float()
            dip = torch.zeros_like(ips)
        else:
            dc = self.head_c_delta(h)
            c_denoised = nz(c_u + dc, dim=-1) * c_norm
            ips_base = proj_frozen(adapter_frozen(c_denoised.to(ad_dt))).float()
            dip = self.head_ip_delta(h).view(-1, 4, 2048)
            ips = ips_base + dip

        # 2. Learned continuous native SDXL dual-text prompt tokens (positions 1..n_tok)
        B = c_eeg.shape[0]
        pe = pe_empty.float().expand(B, -1, -1).clone()
        dpe = self.head_pe_tokens(h).view(B, self.n_tok, 2048)
        pe[:, 1:1 + self.n_tok, :] = pe[:, 1:1 + self.n_tok, :] + dpe

        # 3. Learned pooled prompt embedding delta
        dppe = self.head_ppe_delta(h)
        ppe = ppe_empty.float().expand(B, -1) + dppe

        reg = (dc.pow(2).mean() + 0.1 * dip.pow(2).mean() + 0.1 * dpe.pow(2).mean() + 0.1 * dppe.pow(2).mean())
        return ips, pe, ppe, reg


class DiffSDXLTurbo4Step(nn.Module):
    """Differentiable 4-step SDXL-Turbo unroll (exact EulerAncestral/Euler deterministic step matching OfficialTurboRunner)."""
    def __init__(self, dev="cuda:0"):
        super().__init__()
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline
        self.dev = dev
        self.dt = torch.bfloat16
        pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=self.dt, variant="fp16")
        self.attn_adapter = IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter")
        self.attn_adapter.bind_unet(pipe.unet)
        pipe.to(dev, self.dt)

        with torch.no_grad():
            pe_e, _, ppe_e, _ = pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False)
        self.register_buffer("pe_empty", pe_e.float())
        self.register_buffer("ppe_empty", ppe_e.float())

        self.unet = pipe.unet.eval().requires_grad_(False)
        self.scheduler = pipe.scheduler
        self.scheduler.set_timesteps(4, device=dev)
        self.timesteps = self.scheduler.timesteps.clone()
        self.sigmas = self.scheduler.sigmas.clone().float().to(dev)
        self.init_noise_sigma = float(self.sigmas[0].item())

        self.proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, self.dt).eval().requires_grad_(False)
        self.adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, self.dt).eval().requires_grad_(False)
        self.vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
        self.vae.enable_slicing()
        self.vae_scale = float(self.vae.config.scaling_factor)

        del pipe.text_encoder, pipe.text_encoder_2
        torch.cuda.empty_cache()

        # Standard SDXL 512x512 time_ids: [orig_h, orig_w, crop_y, crop_x, tgt_h, tgt_w]
        self.register_buffer("base_time_ids", torch.tensor([[512.0, 512.0, 0.0, 0.0, 512.0, 512.0]], device=dev))

    def unroll_4steps(self, ips: torch.Tensor, pe: torch.Tensor, ppe: torch.Tensor,
                      eps_init: torch.Tensor, z0: torch.Tensor = None, smap_c: float = 0.0,
                      use_checkpoint: bool = True):
        """Returns differentiable RGB tensor [B, 3, 512, 512] in [0, 1]."""
        B = ips.shape[0]
        time_ids = self.base_time_ids.expand(B, -1).to(self.dt)
        ips_dt = ips.to(self.dt)
        pe_dt = pe.to(self.dt)
        ppe_dt = ppe.to(self.dt)

        # Initialize latent at sigma_0
        lat = eps_init.float() * self.init_noise_sigma
        if z0 is not None and smap_c > 0.0:
            # Exact OfficialTurboRunner hold schedule
            smap = 1.0 - 0.5 * smap_c
            hold = max(0.0, 4.0 - 4.0 * smap)
            lat = z0.float() + eps_init.float() * self.init_noise_sigma
        else:
            hold = 0.0

        for i_step in range(4):
            t_cur = self.timesteps[i_step]
            sig = self.sigmas[i_step]
            sig_next = self.sigmas[i_step + 1]
            # Scale model input: lat / sqrt(sigma^2 + 1)
            lat_in = (lat / (sig.pow(2) + 1.0).sqrt()).to(self.dt)

            def _unet_fn(x_in, pe_in, ppe_in, ip_in):
                return self.unet(
                    x_in,
                    t_cur,
                    encoder_hidden_states=pe_in,
                    added_cond_kwargs={"text_embeds": ppe_in, "time_ids": time_ids},
                    cross_attention_kwargs={"ip_hidden_states": ip_in},
                    return_dict=False,
                )[0]

            if use_checkpoint and torch.is_grad_enabled():
                eps_pred = checkpoint(_unet_fn, lat_in, pe_dt, ppe_dt, ips_dt, use_reentrant=False).float()
            else:
                eps_pred = _unet_fn(lat_in, pe_dt, ppe_dt, ips_dt).float()

            # Euler EDM / epsilon-prediction update: x0_hat = lat - sig * eps_pred; lat_next = x0_hat + sig_next * eps_pred
            # (Exact deterministic Euler step, differentiable with smooth gradients!)
            lat = lat + (sig_next - sig) * eps_pred

            if z0 is not None and hold > 0.0:
                a = max(0.0, min(1.0, hold - i_step))
                if a > 0.0:
                    lat = a * (z0.float() + sig_next * eps_init.float()) + (1.0 - a) * lat

        def _vae_dec(z):
            return self.vae.decode(z.to(self.dt) / self.vae_scale, return_dict=False)[0]

        if use_checkpoint and torch.is_grad_enabled():
            rgb = checkpoint(_vae_dec, lat, use_reentrant=False).float()
        else:
            rgb = _vae_dec(lat).float()
        return (rgb / 2.0 + 0.5).clamp(0.0, 1.0)


def train_bridge(tag: str = "objdraft_v1", steps: int = 700, bs: int = 4, grad_accum: int = 2, lr: float = 3e-4,
                 w_cls: float = 1.5, w_vis: float = 1.2, w_dino: float = 1.0, use_z0_train: bool = True,
                 text_only: bool = False, dev: str = "cuda:0"):
    out_d = BRIDGE_ROOT / tag
    out_d.mkdir(parents=True, exist_ok=True)
    ckpt_p = out_d / "bridge.pt"
    if ckpt_p.exists():
        print(f"[CACHED] {ckpt_p}", flush=True)
        return ckpt_p

    torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
    t0 = time.time()

    # 1. Load 2-Fold 100% Unseen-Concept OOF EEG (16,540 images x 10 subjects, ZERO test data!)
    f0 = torch.load(OOF_DIR / "oof_fold0.pt", map_location="cpu", weights_only=False)
    f1 = torch.load(OOF_DIR / "oof_fold1.pt", map_location="cpu", weights_only=False)
    N = 16540
    c_oof = torch.empty(10, N, 1024, dtype=torch.float16)
    h_oof = torch.empty(10, N, 1024, dtype=torch.float16)
    b_oof = torch.empty(10, N, 1280, dtype=torch.float16)
    for fk in (f0, f1):
        vi = fk["va_idx"]
        c_oof[:, vi] = fk["c_oof"]
        h_oof[:, vi] = fk["h_oof"]
        b_oof[:, vi] = fk["b_oof"]
    del f0, f1

    # Load train concept labels & LAION H14 train text bank [1627, 1024] & GT train H14 image embeds [16540, 1024]
    tb = torch.load(RAD / "runs/obj_bench/text_banks/text_banks_laion.pt", map_location="cpu", weights_only=False)
    tr_cid = tb["tr_img2cid"].long().to(dev)  # [16540] in 0..1626
    T_h14_1627 = nz(tb["h14_train"].float(), dim=-1).to(dev)  # [1627, 1024]
    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    k_h14 = next(k for k in emb_tr.keys() if "h14" in k.lower() or "h-14" in k.lower())
    GT_h14_tr = nz(emb_tr[k_h14].float(), dim=-1).to(dev)  # [16540, 1024]
    del emb_tr

    # Load train GT VAE latents (for light z0 conditioning during training if enabled)
    lat_tr = np.load(PREP / "lat512_train.npy", mmap_mode="r")  # [16540, 4, 64, 64]

    # 2. Load Frozen Reward Encoders (STRICTLY LAION CLIP-ViT-H-14 + DINOv2-base — ZERO OpenAI ViT-L/14!)
    from transformers import CLIPModel, AutoModel
    h14_vis = CLIPModel.from_pretrained(str(H14_DIR), torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    dino = AutoModel.from_pretrained(str(DINO_DIR), torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False) if w_dino > 0 else None

    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
    dino_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    dino_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    # Pre-load GT training images paths for DINOv2 target on-the-fly
    train_raw = json.load(open(PREP / "train_ids.json"))
    img_root = RAD / "data/things-eeg/Image_set"
    # Check if training images exist locally; if not, compute DINOv2 target from GT VAE latent decoded!
    has_tr_imgs = (img_root / train_raw["paths"][0]).exists()
    print(f"[ObjDRaFT] has_train_images_on_disk={has_tr_imgs} (if False, DINOv2 uses decoded GT lat512_train!)", flush=True)

    # 3. Load Frozen Differentiable 4-Step SDXL-Turbo + Trainable Bridge
    turbo = DiffSDXLTurbo4Step(dev=dev)
    bridge = ObjDRaFTBridge(n_prompt_tokens=8, hidden=1024).to(dev)
    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    rng = np.random.default_rng(2026)
    g_torch = torch.Generator(device=dev).manual_seed(2026)

    print(f"[ObjDRaFT] Starting training {tag}: steps={steps}, bs={bs}x{grad_accum}, w_cls={w_cls}, w_vis={w_vis}, w_dino={w_dino}, text_only={text_only}", flush=True)
    log_cls, log_vis, log_dino, log_acc = [], [], [], []
    bridge.train()
    opt.zero_grad(set_to_none=True)
    for step in range(1, steps + 1):
        for acc_i in range(grad_accum):
            idx_np = rng.integers(0, N, size=bs)
            idx = torch.from_numpy(idx_np).to(dev)

            k_sub = int(rng.choice([2, 4, 6, 8, 10], p=[0.10, 0.20, 0.25, 0.25, 0.20]))
            s_sel = rng.choice(10, size=k_sub, replace=False)
            c_b = c_oof[s_sel][:, idx_np].float().mean(0).to(dev)
            c_b = nz(c_b, dim=-1) * 32.0
            h_b = nz(h_oof[s_sel][:, idx_np].float().mean(0).to(dev), dim=-1)
            b_b = nz(b_oof[s_sel][:, idx_np].float().mean(0).to(dev), dim=-1)

            if use_z0_train and ((step + acc_i) % 2 == 0):
                z_gt = torch.from_numpy(np.array(lat_tr[idx_np], dtype=np.float32)).to(dev)
                z0_b = 0.45 * z_gt + 0.55 * torch.randn_like(z_gt) * z_gt.std(dim=(1, 2, 3), keepdim=True)
                smap_cur = 0.375
            else:
                z0_b = None
                smap_cur = 0.0

            eps_init = torch.randn(bs, 4, 64, 64, device=dev, generator=g_torch)
            ips, pe, ppe, reg = bridge(c_b, h_b, b_b, turbo.adapter, turbo.proj, turbo.pe_empty, turbo.ppe_empty, text_only=text_only)
            rgb = turbo.unroll_4steps(ips, pe, ppe, eps_init, z0=z0_b, smap_c=smap_cur, use_checkpoint=True)

            x_clip = F.interpolate(rgb, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
            x_clip = ((x_clip - clip_mean) / clip_std).to(torch.bfloat16)
            v_out = h14_vis.get_image_features(pixel_values=x_clip)
            if hasattr(v_out, "pooler_output"):
                v_out = v_out.pooler_output
            v_h14 = nz(v_out.float(), dim=-1)

            y_c = tr_cid[idx]
            logits_1627 = (v_h14 @ T_h14_1627.T) / 0.04
            l_cls = F.cross_entropy(logits_1627, y_c)
            l_txt = (1.0 - (v_h14 * T_h14_1627[y_c]).sum(-1)).mean()
            l_vis = (1.0 - (v_h14 * GT_h14_tr[idx]).sum(-1)).mean()

            if dino is not None and w_dino > 0:
                with torch.no_grad():
                    z_gt_d = torch.from_numpy(np.array(lat_tr[idx_np], dtype=np.float32)).to(dev)
                    gt_rgb = (turbo.vae.decode(z_gt_d.to(torch.bfloat16) / turbo.vae_scale, return_dict=False)[0].float() / 2.0 + 0.5).clamp(0, 1)
                    gx_d = ((F.interpolate(gt_rgb, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1) - dino_mean) / dino_std).to(torch.bfloat16)
                    gt_dino = nz(dino(pixel_values=gx_d).last_hidden_state[:, 0].float(), dim=-1)
                px_d = ((F.interpolate(rgb, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1) - dino_mean) / dino_std).to(torch.bfloat16)
                pr_dino = nz(dino(pixel_values=px_d).last_hidden_state[:, 0].float(), dim=-1)
                l_dino = (1.0 - (pr_dino * gt_dino).sum(-1)).mean()
            else:
                l_dino = torch.tensor(0.0, device=dev)

            loss = (w_cls * (0.65 * l_cls + 0.35 * l_txt * 5.0) + w_vis * l_vis * 5.0 + w_dino * l_dino * 5.0 + 0.02 * reg) / grad_accum
            loss.backward()

            with torch.no_grad():
                acc1 = (logits_1627.argmax(-1) == y_c).float().mean().item() * 100.0
                log_cls.append(l_cls.item()); log_vis.append(l_vis.item()); log_dino.append(l_dino.item()); log_acc.append(acc1)

        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

        if step % 100 == 0 or step == 1:
            print(
                f"  [step {step:4d}/{steps}] ({time.time()-t0:5.1f}s) | L_cls={np.mean(log_cls[-50:]):.3f} | "
                f"L_vis={np.mean(log_vis[-50:]):.3f} | L_dino={np.mean(log_dino[-50:]):.3f} | "
                f"Train1627_Top1={np.mean(log_acc[-50:]):4.1f}% | reg={reg.item():.4f}",
                flush=True,
            )

    torch.save({"state_dict": bridge.state_dict(), "tag": tag, "steps": steps, "text_only": text_only}, ckpt_p)
    print(f"[ObjDRaFT] Saved {ckpt_p} in {time.time()-t0:.1f}s!", flush=True)
    return ckpt_p


@torch.inference_mode()
def evaluate_bridge(tag: str, subs: list, smap_c: float = 0.375, sel_k: int = 1, dev: str = "cuda:0"):
    ckpt_p = BRIDGE_ROOT / tag / "bridge.pt"
    ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    text_only = bool(ckpt.get("text_only", False))
    bridge = ObjDRaFTBridge(n_prompt_tokens=8, hidden=1024).to(dev).eval()
    bridge.load_state_dict(ckpt["state_dict"])

    eng = DualStreamSemanticEngine("objsem_v1")
    turbo = DiffSDXLTurbo4Step(dev=dev)
    ev = ObjReconEvaluator(dev)
    reranker = LAIONH14Reranker(eng.H_cv_raw, dev=dev) if sel_k > 1 else None

    name = f"{tag}_s{int(smap_c*1000):03d}_sel{sel_k}"
    out_d = BRIDGE_ROOT / "eval_runs" / name
    out_d.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()

    for s in subs:
        sp = eng.get_subject_pack(s, manifold_beta=0.0)
        c_all = sp["c_o14"].to(dev)
        h_all = sp["h14_sem"].to(dev)
        b_all = sp["bigg_sem"].to(dev)
        z0_all = sp["z0"].to(dev)

        cands = []
        for k in range(sel_k):
            imgs = torch.empty(200, 3, 512, 512, dtype=torch.uint8)
            for st in range(0, 200, 20):
                q = slice(st, min(200, st + 20))
                Q = min(200, st + 20) - st
                ips, pe, ppe, _ = bridge(c_all[q], h_all[q], b_all[q], turbo.adapter, turbo.proj, turbo.pe_empty, turbo.ppe_empty, text_only=text_only)
                eps = torch.stack([
                    torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * (st + i) + k + 99))
                    for i in range(Q)
                ]).to(dev)
                rgb = turbo.unroll_4steps(ips, pe, ppe, eps, z0=(z0_all[q] if smap_c > 0 else None), smap_c=smap_c, use_checkpoint=False)
                imgs[q] = (rgb * 255.0).round().to(torch.uint8).cpu()
            cands.append(imgs)

        imgs_final = cands[0] if sel_k == 1 else reranker.select_best(
            cands, sp["h_tta"], sp["h14_sem"], sp["mu_Th"], w_vis=0.45, w_sem=0.40, w_obj=0.15
        )
        m = ev.evaluate(imgs_final, compute_lowlevel=True)
        rows.append(m)
        torch.save({"k0_rgb_u8": imgs_final, "metrics": m}, out_d / f"sub-{s:02d}_imgs.pt")
        print(
            f"  [{name:28s}] sub-{s:02d} | ZS200={m['zs_obj200_t1']:5.2f}/{m['zs_obj200_t5']:5.2f} | "
            f"ZS1823={m['zs_obj1823_t1']:4.2f}/{m['zs_obj1823_t5']:5.2f} | Cat27={m['coarse27_t1']:4.1f}% | "
            f"IN1K={m['in1k_inc_t1']:4.1f}/{m['in1k_inc_t5']:4.1f} | Crisp={m['obj_crisp']:4.1f}% | "
            f"Img200={m['clip_img200_t1']:5.2f}% | 2wC={m['clip_2way']:5.2f}% | SSIM={m['ssim']:.3f}",
            flush=True,
        )
    return log_and_save(name, subs, rows, out_d, t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["train", "eval_dev4", "eval_all10"])
    ap.add_argument("--tag", type=str, default="objdraft_v1")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w_cls", type=float, default=1.5)
    ap.add_argument("--w_vis", type=float, default=1.2)
    ap.add_argument("--w_dino", type=float, default=1.0)
    ap.add_argument("--text_only", action="store_true")
    ap.add_argument("--smap_c", type=float, default=0.375)
    ap.add_argument("--sel_k", type=int, default=1)
    args = ap.parse_args()

    if args.mode == "train":
        train_bridge(args.tag, steps=args.steps, bs=args.bs, grad_accum=args.grad_accum, lr=args.lr,
                     w_cls=args.w_cls, w_vis=args.w_vis, w_dino=args.w_dino, text_only=args.text_only)
    elif args.mode == "eval_dev4":
        evaluate_bridge(args.tag, [1, 2, 8, 10], smap_c=args.smap_c, sel_k=args.sel_k)
    elif args.mode == "eval_all10":
        evaluate_bridge(args.tag, list(range(1, 11)), smap_c=args.smap_c, sel_k=args.sel_k)
