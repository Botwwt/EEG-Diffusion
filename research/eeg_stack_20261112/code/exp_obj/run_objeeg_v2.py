"""ObjEEG-Diff v2: Manifold-Denoised SoftOpenText + Dual-Space Semantic Verification (Dev-4 & All-10)."""
import os, sys, gc, math, time, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

RAD = Path(os.environ.get("RAD_ROOT", "/path/to/rad"))
for sub in ("code/exp_obj", "code/exp_w7", "code/exp_w6", "code/rad", "code/exp", "code/Brain-HIVE"):
    p = str(RAD / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from main.models_adapter import IPProjectionModel, IPAttnAdapterModel, FusionEncoderModel
from eval_obj_suite import ObjReconEvaluator

PREP = RAD / "data/prep"
TURBO = RAD / "models/sdxl-turbo"
PRIOR = RAD / "models/Fusion-Prior-H14_B32_VAE"
H14_DIR = RAD / "models/CLIP-ViT-H-14-laion2B-s32B-b79K"
OUT_ROOT = RAD / "runs/obj_bench/objeeg_v2"


def nz(x, dim=-1):
    return F.normalize(x.float(), dim=dim)


def zscore_rows(m):
    return (m - m.mean(dim=-1, keepdim=True)) / (m.std(dim=-1, keepdim=True) + 1e-6)


def _lp_op(x: torch.Tensor, f_lp: int) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    lo = F.interpolate(x, size=(f_lp, f_lp), mode="bilinear", align_corners=False)
    return F.interpolate(lo, size=(h, w), mode="bilinear", align_corners=False)


class DualStreamSemanticEngine:
    """Strictly zero-leakage (LAION H14 + SDXL bigG + train-only F_train + CV-OpenVocab 8,141)."""
    def __init__(self, objsem_tag: str = "objsem_v1"):
        self.tb = torch.load(RAD / "runs/obj_bench/text_banks/text_banks_laion.pt", map_location="cpu", weights_only=False)
        self.objsem = torch.load(RAD / f"runs/obj_bench/obj_sem_enc/{objsem_tag}/ens_tta_all10.pt", map_location="cpu", weights_only=False)

        # Train-only F_train [16540, 1024] for manifold denoising & bridge (ZERO test data!)
        self.F_tr_raw = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
        self.F_tr_u = nz(self.F_tr_raw, dim=-1)
        tr_cid = self.tb["tr_img2cid"].long()
        Y_h = self.tb["h14_train"][tr_cid].float()
        Y_b = self.tb["bigg_train"][tr_cid].float()
        self.mu_F = self.F_tr_u.mean(0, keepdim=True)
        self.mu_Th = Y_h.mean(0, keepdim=True)
        self.mu_Tb = Y_b.mean(0, keepdim=True)

        self.T_tr_h = nz(Y_h - self.mu_Th, dim=-1)  # [16540, 1024]
        self.T_tr_b = nz(Y_b - self.mu_Tb, dim=-1)  # [16540, 1280]

        Fc = self.F_tr_u - self.mu_F
        FtF = Fc.T @ Fc
        self.W_fh = torch.linalg.solve(FtF + 50.0 * torch.eye(1024), Fc.T @ (Y_h - self.mu_Th))
        self.W_fb = torch.linalg.solve(FtF + 50.0 * torch.eye(1024), Fc.T @ (Y_b - self.mu_Tb))

        # Open-World CV-OpenVocab (8,141 OpenImages + LVIS + IN1K, ZERO THINGS test provenance!)
        self.words = self.tb["cv_open"]
        self.H_cv = nz(self.tb["h14_cv_open"].float() - self.mu_Th, dim=-1)
        self.B_cv = nz(self.tb["bigg_cv_open"].float() - self.mu_Tb, dim=-1)
        self.H_cv_raw = nz(self.tb["h14_cv_open"].float(), dim=-1)

        # Coarse-27 super-categories (centered)
        self.cat27 = self.tb["cat27"]
        self.H_cat = nz(self.tb["h14_cat27"].float() - self.mu_Th, dim=-1)
        self.B_cat = nz(self.tb["bigg_cat27"].float() - self.mu_Tb, dim=-1)

    def get_subject_pack(self, sub: int, manifold_beta: float = 0.25, manifold_k: int = 64, manifold_tau: float = 0.35):
        pk = torch.load(RAD / f"runs/o14_official/cond/sub-{sub:02d}_o14_pack.pt", map_location="cpu", weights_only=False)
        c_o14 = pk["c_o14"].float()  # [200, 1024], norm ~32
        h_tta = nz(pk["h_tta"].float(), dim=-1)
        p_o14 = nz(pk["ppe_hat"].float() - pk["ppe_hat"].float().mean(0, keepdim=True), dim=-1)
        z0 = torch.from_numpy(np.load(RAD / f"runs/low/sub-{sub:02d}/lat512_o14_triarch_v2_n0.80.npy")).float()

        od = self.objsem[f"sub-{sub:02d}"]
        qh_sem = od["h14_txt"].float()    # [200, 1024], centered normalized
        qb_sem = od["bigg_txt"].float()   # [200, 1280], centered normalized

        # O14 train-only visual->text bridge
        c_u = nz(c_o14, dim=-1)
        cc = c_u - c_u.mean(0, keepdim=True)
        qh_o14 = nz(cc @ self.W_fh, dim=-1)
        qb_o14 = nz(cc @ self.W_fb, dim=-1)
        h_c = nz(pk["h_tta"].float() - pk["h_tta"].float().mean(0, keepdim=True), dim=-1)

        # Fused semantic score over 8,141 CV-OpenVocab words (purely LAION H14 + bigG!)
        s_o14_v = (
            0.35 * zscore_rows(qh_o14 @ self.H_cv.T) +
            0.35 * zscore_rows(qb_o14 @ self.B_cv.T) +
            0.15 * zscore_rows(h_c @ self.H_cv.T) +
            0.15 * zscore_rows(p_o14 @ self.B_cv.T)
        )
        s_sem_v = 0.50 * zscore_rows(qh_sem @ self.H_cv.T) + 0.50 * zscore_rows(qb_sem @ self.B_cv.T)
        s_cv = 0.55 * zscore_rows(s_sem_v) + 0.45 * zscore_rows(s_o14_v)  # [200, 8141]

        # Dual-stream continuous bigG & H14 semantic targets
        bigg_sem = nz(0.55 * qb_sem + 0.45 * qb_o14, dim=-1)
        h14_sem = nz(0.55 * qh_sem + 0.45 * qh_o14, dim=-1)

        # Train-only soft manifold denoising of c_o14 on F_train (16,540 clean training images!)
        if manifold_beta > 0.0:
            sim_f = zscore_rows(c_u @ self.F_tr_u.T)
            sim_t = 0.5 * zscore_rows(qh_sem @ self.T_tr_h.T) + 0.5 * zscore_rows(qb_sem @ self.T_tr_b.T)
            sim_joint = 0.65 * sim_f + 0.35 * zscore_rows(sim_t)  # [200, 16540]
            topv, topi = sim_joint.topk(manifold_k, dim=-1)
            w = F.softmax(topv / manifold_tau, dim=-1)  # [200, K]
            nbrs = self.F_tr_u[topi]  # [200, K, 1024]
            c_knn = nz((w.unsqueeze(-1) * nbrs).sum(dim=1), dim=-1)
            c_norm = c_o14.norm(dim=-1, keepdim=True)
            c_clean = nz((1.0 - manifold_beta) * c_u + manifold_beta * c_knn, dim=-1) * c_norm
        else:
            c_clean = c_o14

        return {
            "c_o14": c_o14,
            "c_clean": c_clean,
            "z0": z0,
            "h_tta": h_tta,
            "h14_sem": h14_sem,
            "bigg_sem": bigg_sem,
            "s_cv": s_cv,
            "mu_Th": self.mu_Th,
        }


class LAIONH14Reranker:
    """Strictly zero-leakage LAION CLIP-ViT-H-14 vision reranker (100% disjoint from OpenAI ViT-L/14 eval!)."""
    def __init__(self, H_cv_raw: torch.Tensor, dev="cuda:0"):
        from transformers import CLIPModel
        self.dev = dev
        self.model = CLIPModel.from_pretrained(str(H14_DIR), torch_dtype=torch.float16).to(dev).eval().requires_grad_(False)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
        self.H_cv_raw = H_cv_raw.to(dev, torch.float32)  # [8141, 1024]

    @torch.inference_mode()
    def encode_images(self, imgs_u8: torch.Tensor) -> torch.Tensor:
        """imgs_u8: [N, 3, H, W] uint8 -> [N, 1024] normalized LAION H14 visual embeddings."""
        out = []
        for i in range(0, len(imgs_u8), 50):
            x = imgs_u8[i:i + 50].to(self.dev).float() / 255.0
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
            x = ((x - self.mean) / self.std).half()
            v = self.model.get_image_features(pixel_values=x)
            if hasattr(v, "pooler_output"):
                v = v.pooler_output
            out.append(F.normalize(v.float(), dim=-1))
        return torch.cat(out, dim=0)

    @torch.inference_mode()
    def select_best(self, cand_imgs_list: list, h_tta: torch.Tensor, h14_sem: torch.Tensor,
                    mu_Th: torch.Tensor, w_vis: float = 0.45, w_sem: float = 0.40, w_obj: float = 0.15) -> torch.Tensor:
        """cand_imgs_list: list of K tensors [200, 3, 512, 512] uint8 -> returns best [200, 3, 512, 512]."""
        K = len(cand_imgs_list)
        hv = h_tta.to(self.dev, torch.float32)
        hs = h14_sem.to(self.dev, torch.float32)
        mu_t = mu_Th.to(self.dev, torch.float32)

        scores = torch.zeros(K, 200, device=self.dev)
        vis_raw = torch.zeros(K, 200, device=self.dev)
        sem_raw = torch.zeros(K, 200, device=self.dev)
        obj_raw = torch.zeros(K, 200, device=self.dev)

        for k, imgs in enumerate(cand_imgs_list):
            vf = self.encode_images(imgs)  # [200, 1024]
            # 1. Visual consistency with O14 h_tta
            vis_raw[k] = (vf * hv).sum(dim=-1)
            # 2. Semantic consistency with ObjSemEnc (centered against train text mean!)
            vf_c = F.normalize(vf - vf.mean(0, keepdim=True), dim=-1)
            sem_raw[k] = (vf_c * hs).sum(dim=-1)
            # 3. Open-world objectness (max cosine to any of 8,141 CV-OpenVocab concrete objects!)
            obj_raw[k] = (vf @ self.H_cv_raw.T).max(dim=-1).values

        # Z-normalize across candidates per image so all 3 terms are scale-balanced
        def z_k(x):
            return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)

        scores = w_vis * z_k(vis_raw) + w_sem * z_k(sem_raw) + w_obj * z_k(obj_raw)
        best_k = scores.argmax(dim=0).cpu()  # [200]
        out = torch.empty_like(cand_imgs_list[0])
        for i in range(200):
            out[i] = cand_imgs_list[int(best_k[i])][i]
        return out


class ObjDiffV2Generator:
    """SDXL-Turbo generator supporting SoftOpenText mixtures, controlled z0 smap_c, ip_scale, and continuous bigG."""
    def __init__(self, dev="cuda:0"):
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
        self.dev = dev
        self.dt = torch.float16
        self.pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=self.dt, variant="fp16")
        self.attn_adapter = IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter")
        self.attn_adapter.bind_unet(self.pipe.unet)
        self.pipe.to(dev, self.dt)
        self.pipe.vae.enable_slicing()
        self.pipe.set_progress_bar_config(disable=True)
        self.p2 = StableDiffusionXLImg2ImgPipeline(**self.pipe.components)
        self.p2.set_progress_bar_config(disable=True)

        self.proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, self.dt).eval().requires_grad_(False)
        self.adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, self.dt).eval().requires_grad_(False)
        self.vae32 = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval().requires_grad_(False)
        self.vae32.enable_slicing()

        with torch.inference_mode():
            self.pe_empty, _, self.ppe_empty, _ = self.pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False)
        self._word_cache = {}

    def set_ip_scale(self, scale: float):
        for proc in self.attn_adapter.attn_processors.values():
            proc.scale = float(scale)

    @torch.inference_mode()
    def build_soft_open_text_embeds(self, s_cv: torch.Tensor, words: list, topk: int = 16,
                                    tau: float = 0.45, gamma: float = 0.55, shuffle_seed: int = -1):
        """Computes posterior-expected SDXL dual-text embeddings over top-K open-world objects (ZERO argmax collapse!)."""
        topv, topi = s_cv.topk(topk, dim=-1)  # [200, K]
        probs = F.softmax(topv / tau, dim=-1)  # [200, K]

        # Collect unique words needed for this subject and batch-encode via SDXL dual encoders
        uniq_ids = sorted(set(topi.reshape(-1).tolist()))
        missing = [idx for idx in uniq_ids if idx not in self._word_cache]
        if missing:
            prompts = [f"a clear photograph of a {words[idx]}" for idx in missing]
            for j in range(0, len(prompts), 64):
                pe, _, ppe, _ = self.pipe.encode_prompt(prompts[j:j + 64], device=self.dev, do_classifier_free_guidance=False)
                for m_i, idx in enumerate(missing[j:j + 64]):
                    self._word_cache[idx] = (pe[m_i].cpu(), ppe[m_i].cpu())

        pe_e = self.pe_empty[0].float().cpu()    # [77, 2048]
        ppe_e = self.ppe_empty[0].float().cpu()  # [1280]
        pe_all = torch.empty(200, pe_e.shape[0], pe_e.shape[1], dtype=torch.float16)
        ppe_all = torch.empty(200, ppe_e.shape[0], dtype=torch.float16)

        for i in range(200):
            pe_mix = torch.zeros_like(pe_e)
            ppe_mix = torch.zeros_like(ppe_e)
            for k in range(topk):
                wk = float(probs[i, k].item())
                pe_k, ppe_k = self._word_cache[int(topi[i, k].item())]
                pe_mix += wk * pe_k.float()
                ppe_mix += wk * ppe_k.float()
            pe_all[i] = ((1.0 - gamma) * pe_e + gamma * pe_mix).half()
            ppe_all[i] = ((1.0 - gamma) * ppe_e + gamma * ppe_mix).half()

        if shuffle_seed >= 0:
            rng = np.random.default_rng(shuffle_seed)
            perm = torch.from_numpy(rng.permutation(200))
            pe_all = pe_all[perm]
            ppe_all = ppe_all[perm]
        return pe_all, ppe_all

    @torch.inference_mode()
    def build_hard_rank_text_embeds(self, s_cv: torch.Tensor, words: list, rank_k: int = 0, gamma: float = 0.85):
        """Encodes the rank_k-th open-world object hypothesis for Hypothesis-Proposal reranking."""
        topi = s_cv.topk(rank_k + 1, dim=-1).indices[:, rank_k].tolist()
        missing = sorted(set(idx for idx in topi if idx not in self._word_cache))
        if missing:
            prompts = [f"a clear photograph of a {words[idx]}" for idx in missing]
            for j in range(0, len(prompts), 64):
                pe, _, ppe, _ = self.pipe.encode_prompt(prompts[j:j + 64], device=self.dev, do_classifier_free_guidance=False)
                for m_i, idx in enumerate(missing[j:j + 64]):
                    self._word_cache[idx] = (pe[m_i].cpu(), ppe[m_i].cpu())
        pe_e = self.pe_empty[0].float().cpu()
        ppe_e = self.ppe_empty[0].float().cpu()
        pe_all = torch.stack([((1.0 - gamma) * pe_e + gamma * self._word_cache[idx][0].float()).half() for idx in topi])
        ppe_all = torch.stack([((1.0 - gamma) * ppe_e + gamma * self._word_cache[idx][1].float()).half() for idx in topi])
        return pe_all, ppe_all

    @torch.inference_mode()
    def generate(self, c_all: torch.Tensor, z0_all: torch.Tensor, pe_all: torch.Tensor = None, ppe_all: torch.Tensor = None,
                 bigg_sem: torch.Tensor = None, alpha_ppe: float = 0.0, ip_scale: float = 1.0,
                 smap_c: float = 0.375, low_keep=(8, 0.0, 0.0), seed_offset: int = 0, bs: int = 20):
        self.set_ip_scale(ip_scale)
        out_imgs = torch.empty(200, 3, 512, 512, dtype=torch.uint8)
        ppe_norm = float(self.ppe_empty.float().norm().item())

        steps = 4
        use_z0 = smap_c > 0.0
        if use_z0:
            init = torch.cat([
                (self.vae32.decode(z0_all[j:j + 10].to(self.dev, torch.float32) / self.vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
                for j in range(0, 200, 10)
            ])
            s_rel, s_unrel = 0.5, 1.0
            rho = torch.full((200, 1, 64, 64), smap_c)
            smap = s_unrel - (s_unrel - s_rel) * rho
            n_run = min(int(round(steps * s_unrel)), steps)
            hold = (n_run - steps * smap).clamp(min=0)

        for s in range(0, 200, bs):
            q = list(range(s, min(200, s + bs)))
            Q = len(q)
            cc = c_all[q].to(self.dev, self.dt)
            ips = self.proj(self.adapter(cc))
            pe_b = pe_all[q].to(self.dev, self.dt) if pe_all is not None else self.pe_empty.expand(Q, -1, -1)
            ppe_b = ppe_all[q].to(self.dev, self.dt) if ppe_all is not None else self.ppe_empty.expand(Q, -1)
            if bigg_sem is not None and alpha_ppe > 0.0:
                pd = bigg_sem[q].to(self.dev, torch.float32)
                ppe_b = (ppe_b.float() + alpha_ppe * ppe_norm * pd).to(self.dt)

            gens = [torch.Generator(self.dev).manual_seed(1_000_003 * i + seed_offset + 12345) for i in q]

            if use_z0:
                i0 = init[q].to(self.dev, self.dt)
                z0 = z0_all[q].to(self.dev).float()
                eps = torch.stack([
                    torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * i + seed_offset + 99))
                    for i in q
                ]).to(self.dev)
                h_ = hold[q].to(self.dev)

                def cb(pp, i_step, t_cur, ck):
                    lt = ck["latents"].float()
                    si = int(pp.scheduler.step_index)
                    sig = pp.scheduler.sigmas[si].to(self.dev)
                    a = (h_ - i_step).clamp(0.0, 1.0)
                    if a.max() > 0:
                        lt = a * (z0 + sig * eps) + (1.0 - a) * lt
                    if i_step in (steps - 2, steps - 1):
                        f_lp, w2, w3 = low_keep
                        g_t = w2 if i_step == steps - 2 else w3
                        if g_t > 0:
                            lt = lt + g_t * _lp_op((z0 + sig * eps) - lt, f_lp)
                    ck["latents"] = lt.to(self.dt)
                    return ck

                lat = self.p2(
                    image=i0,
                    prompt_embeds=pe_b,
                    pooled_prompt_embeds=ppe_b,
                    num_inference_steps=steps,
                    strength=1.0,
                    guidance_scale=0.0,
                    output_type="latent",
                    generator=gens,
                    cross_attention_kwargs={"ip_hidden_states": ips},
                    callback_on_step_end=cb,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images
            else:
                lat = self.pipe(
                    prompt_embeds=pe_b,
                    pooled_prompt_embeds=ppe_b,
                    num_inference_steps=steps,
                    guidance_scale=0.0,
                    output_type="latent",
                    generator=gens,
                    cross_attention_kwargs={"ip_hidden_states": ips},
                ).images

            rgb = (self.vae32.decode(lat.float() / self.vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0, 1)
            out_imgs[q] = (rgb * 255.0).round().to(torch.uint8).cpu()
        return out_imgs


def log_and_save(name, subs, rows, out_d, t0):
    mean_m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    json.dump({"subs": subs, "per_sub": rows, "mean": mean_m}, open(out_d / f"summary_{len(subs)}sub.json", "w"), indent=2)
    print(
        f"==> [{name:30s}] {len(subs)}-SUB ({time.time()-t0:5.1f}s) | "
        f"ZS200={mean_m['zs_obj200_t1']:5.2f}/{mean_m['zs_obj200_t5']:5.2f} | "
        f"ZS1823={mean_m['zs_obj1823_t1']:4.2f}/{mean_m['zs_obj1823_t5']:5.2f} | "
        f"Cat27={mean_m['coarse27_t1']:4.1f}% | IN1K={mean_m['in1k_inc_t1']:4.1f}/{mean_m['in1k_inc_t5']:4.1f} | "
        f"Eff1K={mean_m['in1k_eff_t1']:4.1f}% | Crisp={mean_m['obj_crisp']:4.1f}% | "
        f"Img200={mean_m['clip_img200_t1']:5.2f}% | 2wC={mean_m['clip_2way']:5.2f}% | "
        f"Incep={mean_m['incep_2way']:5.2f}% | SSIM={mean_m['ssim']:.3f}\n",
        flush=True,
    )
    return mean_m


def run_arm(name, subs, eng, gen, ev, cfg, reranker=None):
    out_d = OUT_ROOT / name
    out_d.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for s in subs:
        sp = eng.get_subject_pack(s, manifold_beta=cfg.get("beta", 0.0))
        c_use = sp["c_clean"] if cfg.get("beta", 0.0) > 0 else sp["c_o14"]
        mode = cfg["mode"]

        if mode == "cont":
            imgs = gen.generate(
                c_use, sp["z0"], bigg_sem=sp["bigg_sem"], alpha_ppe=cfg.get("alpha_ppe", 0.45),
                ip_scale=cfg.get("ip_scale", 1.0), smap_c=cfg.get("smap_c", 0.375),
                low_keep=cfg.get("low_keep", (8, 0.0, 0.0)),
            )
        elif mode == "soft":
            pe, ppe = gen.build_soft_open_text_embeds(
                sp["s_cv"], eng.words, topk=cfg.get("topk", 16), tau=cfg.get("tau", 0.45),
                gamma=cfg.get("gamma", 0.55), shuffle_seed=cfg.get("shuffle_seed", -1),
            )
            imgs = gen.generate(
                c_use, sp["z0"], pe_all=pe, ppe_all=ppe, bigg_sem=sp["bigg_sem"],
                alpha_ppe=cfg.get("alpha_ppe", 0.25), ip_scale=cfg.get("ip_scale", 1.0),
                smap_c=cfg.get("smap_c", 0.375), low_keep=cfg.get("low_keep", (8, 0.0, 0.0)),
            )
        elif mode == "rerank_sel4":
            cands = []
            # 2 SoftOpenText draws (different seeds & gammas) + 2 distinct Top-1/Top-2 object hypothesis draws!
            pe_s1, ppe_s1 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=16, tau=0.45, gamma=0.55)
            pe_s2, ppe_s2 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=16, tau=0.45, gamma=0.75)
            pe_h0, ppe_h0 = gen.build_hard_rank_text_embeds(sp["s_cv"], eng.words, rank_k=0, gamma=0.80)
            pe_h1, ppe_h1 = gen.build_hard_rank_text_embeds(sp["s_cv"], eng.words, rank_k=1, gamma=0.80)
            specs = [
                (pe_s1, ppe_s1, 0.25, 1.00, 0),
                (pe_s2, ppe_s2, 0.20, 0.90, 1),
                (pe_h0, ppe_h0, 0.15, 0.85, 2),
                (pe_h1, ppe_h1, 0.15, 0.85, 3),
            ]
            for pe_k, ppe_k, ap_k, ip_k, sd_k in specs:
                cands.append(gen.generate(
                    c_use, sp["z0"], pe_all=pe_k, ppe_all=ppe_k, bigg_sem=sp["bigg_sem"],
                    alpha_ppe=ap_k, ip_scale=ip_k, smap_c=cfg.get("smap_c", 0.375),
                    low_keep=(8, 0.0, 0.0), seed_offset=sd_k,
                ))
            imgs = reranker.select_best(
                cands, sp["h_tta"], sp["h14_sem"], sp["mu_Th"],
                w_vis=cfg.get("w_vis", 0.45), w_sem=cfg.get("w_sem", 0.40), w_obj=cfg.get("w_obj", 0.15),
            )
        elif mode == "rerank_sel8":
            cands = []
            pe_s1, ppe_s1 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=16, tau=0.45, gamma=0.50)
            pe_s2, ppe_s2 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=16, tau=0.45, gamma=0.75)
            # 2 ContSem + 2 SoftOpenText + 4 distinct Top-4 Open-World Object Proposals!
            specs = [
                (None, None, 0.45, 1.00, 0),
                (None, None, 0.45, 1.00, 1),
                (pe_s1, ppe_s1, 0.25, 1.00, 2),
                (pe_s2, ppe_s2, 0.20, 0.90, 3),
            ]
            for rk in range(4):
                pe_h, ppe_h = gen.build_hard_rank_text_embeds(sp["s_cv"], eng.words, rank_k=rk, gamma=0.80)
                specs.append((pe_h, ppe_h, 0.15, 0.85, 4 + rk))
            for pe_k, ppe_k, ap_k, ip_k, sd_k in specs:
                cands.append(gen.generate(
                    c_use, sp["z0"], pe_all=pe_k, ppe_all=ppe_k, bigg_sem=sp["bigg_sem"],
                    alpha_ppe=ap_k, ip_scale=ip_k, smap_c=cfg.get("smap_c", 0.375),
                    low_keep=(8, 0.0, 0.0), seed_offset=sd_k,
                ))
            imgs = reranker.select_best(
                cands, sp["h_tta"], sp["h14_sem"], sp["mu_Th"],
                w_vis=cfg.get("w_vis", 0.45), w_sem=cfg.get("w_sem", 0.40), w_obj=cfg.get("w_obj", 0.15),
            )
        elif mode == "o14_sel4_ctrl":
            cands = []
            for sd_k in range(4):
                cands.append(gen.generate(
                    sp["c_o14"], sp["z0"], bigg_sem=None, alpha_ppe=0.0,
                    ip_scale=1.0, smap_c=0.625, low_keep=(8, 0.22, 0.04), seed_offset=sd_k,
                ))
            imgs = reranker.select_best(cands, sp["h_tta"], sp["h14_sem"], sp["mu_Th"], w_vis=1.0, w_sem=0.0, w_obj=0.0)
        else:
            raise ValueError(mode)

        m = ev.evaluate(imgs, compute_lowlevel=True)
        rows.append(m)
        torch.save({"k0_rgb_u8": imgs, "metrics": m}, out_d / f"sub-{s:02d}_imgs.pt")
        print(
            f"  [{name:30s}] sub-{s:02d} | ZS200={m['zs_obj200_t1']:5.2f}/{m['zs_obj200_t5']:5.2f} | "
            f"ZS1823={m['zs_obj1823_t1']:4.2f}/{m['zs_obj1823_t5']:5.2f} | Cat27={m['coarse27_t1']:4.1f}% | "
            f"IN1K={m['in1k_inc_t1']:4.1f}/{m['in1k_inc_t5']:4.1f} | Crisp={m['obj_crisp']:4.1f}% | "
            f"Img200={m['clip_img200_t1']:5.2f}% | 2wC={m['clip_2way']:5.2f}% | SSIM={m['ssim']:.3f}",
            flush=True,
        )
    return log_and_save(name, subs, rows, out_d, t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=str, required=True, choices=["g0", "g2", "g4", "all10_main", "all10_sel4", "all10_sel8", "all10_ctrl"])
    args = ap.parse_args()

    eng = DualStreamSemanticEngine("objsem_v1")
    gen = ObjDiffV2Generator("cuda:0")
    ev = ObjReconEvaluator("cuda:0")
    dev4 = [1, 2, 8, 10]
    all10 = list(range(1, 11))

    if args.group == "g0":
        run_arm("v2_C0_z0s375_cont", dev4, eng, gen, ev, {"mode": "cont", "beta": 0.0, "alpha_ppe": 0.45, "ip_scale": 1.0, "smap_c": 0.375})
        run_arm("v2_C1_cman25_z0s375_cont", dev4, eng, gen, ev, {"mode": "cont", "beta": 0.25, "alpha_ppe": 0.45, "ip_scale": 1.0, "smap_c": 0.375})
        run_arm("v2_C2_cman40_z0s375_cont", dev4, eng, gen, ev, {"mode": "cont", "beta": 0.40, "alpha_ppe": 0.45, "ip_scale": 1.0, "smap_c": 0.375})
        run_arm("v2_C3_cman25_z0s500_cont", dev4, eng, gen, ev, {"mode": "cont", "beta": 0.25, "alpha_ppe": 0.45, "ip_scale": 1.0, "smap_c": 0.50, "low_keep": (8, 0.08, 0.0)})
    elif args.group == "g2":
        run_arm("v2_S1_soft16_g040_ip100_s375", dev4, eng, gen, ev, {"mode": "soft", "beta": 0.25, "gamma": 0.40, "ip_scale": 1.00, "smap_c": 0.375, "alpha_ppe": 0.25})
        run_arm("v2_S2_soft16_g065_ip090_s375", dev4, eng, gen, ev, {"mode": "soft", "beta": 0.25, "gamma": 0.65, "ip_scale": 0.90, "smap_c": 0.375, "alpha_ppe": 0.20})
        run_arm("v2_S3_soft16_g085_ip085_s375", dev4, eng, gen, ev, {"mode": "soft", "beta": 0.25, "gamma": 0.85, "ip_scale": 0.85, "smap_c": 0.375, "alpha_ppe": 0.15})
        run_arm("v2_S_CTRL_SHUF_g065", dev4, eng, gen, ev, {"mode": "soft", "beta": 0.25, "gamma": 0.65, "ip_scale": 0.90, "smap_c": 0.375, "alpha_ppe": 0.0, "shuffle_seed": 777})
    elif args.group == "g4":
        reranker = LAIONH14Reranker(eng.H_cv_raw, dev="cuda:0")
        run_arm("v2_R0_o14_sel4_ctrl", dev4, eng, gen, ev, {"mode": "o14_sel4_ctrl"}, reranker=reranker)
        run_arm("v2_R1_sel4_hyp_semrerank", dev4, eng, gen, ev, {"mode": "rerank_sel4", "beta": 0.25, "smap_c": 0.375}, reranker=reranker)
        run_arm("v2_R2_sel8_hyp_semrerank", dev4, eng, gen, ev, {"mode": "rerank_sel8", "beta": 0.25, "smap_c": 0.375}, reranker=reranker)


if __name__ == "__main__":
    main()
