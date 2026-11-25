"""ObjEEG-Diff: Semantic-Visual Decoupled Object-Recognizable EEG Diffusion (Dev-4 Ablation & All-10 Confirm)."""
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
OUT_ROOT = RAD / "runs/obj_bench/objeeg_diff"

def nz(x, dim=-1):
    return F.normalize(x.float(), dim=dim)

def zscore_rows(m):
    return (m - m.mean(dim=-1, keepdim=True)) / (m.std(dim=-1, keepdim=True) + 1e-6)

class SemanticDecoder:
    """Zero-leakage Open-World Object Semantic Decoder (purely LAION H14 + SDXL bigG; ZERO OpenAI L14, ZERO test labels)."""
    def __init__(self, use_objsem: bool = True, objsem_tag: str = "objsem_v1", vocab_mode: str = "cv_open"):
        self.tb = torch.load(RAD / "runs/obj_bench/text_banks/text_banks_laion.pt", map_location="cpu", weights_only=False)
        self.use_objsem = use_objsem
        self.vocab_mode = vocab_mode

        # Train-only ridge bridge from O14 F-space -> LAION H14_txt & bigG_txt (100% train-only!)
        F_tr = nz(torch.from_numpy(np.load(PREP / "F_train.npy")).float(), dim=-1)
        tr_cid = self.tb["tr_img2cid"].long()
        Y_h = self.tb["h14_train"][tr_cid].float()
        Y_b = self.tb["bigg_train"][tr_cid].float()
        self.mu_F = F_tr.mean(0, keepdim=True)
        self.mu_Th = Y_h.mean(0, keepdim=True)
        self.mu_Tb = Y_b.mean(0, keepdim=True)
        Fc = F_tr - self.mu_F
        FtF = Fc.T @ Fc
        self.W_fh = torch.linalg.solve(FtF + 50.0 * torch.eye(1024), Fc.T @ (Y_h - self.mu_Th))
        self.W_fb = torch.linalg.solve(FtF + 50.0 * torch.eye(1024), Fc.T @ (Y_b - self.mu_Tb))

        if vocab_mode == "cv_open":
            self.words = self.tb["cv_open"]
            self.H_v = nz(self.tb["h14_cv_open"].float() - self.mu_Th, dim=-1)
            self.B_v = nz(self.tb["bigg_cv_open"].float() - self.mu_Tb, dim=-1)
        elif vocab_mode == "cv_comp":
            self.words = self.tb["cv_lvis_in1k"]
            self.H_v = nz(self.tb["h14_cv_comp"].float() - self.mu_Th, dim=-1)
            self.B_v = nz(self.tb["bigg_cv_comp"].float() - self.mu_Tb, dim=-1)
        else:
            raise ValueError(vocab_mode)

        self.cat27 = self.tb["cat27"]
        self.H_cat = nz(self.tb["h14_cat27"].float() - self.mu_Th, dim=-1)
        self.B_cat = nz(self.tb["bigg_cat27"].float() - self.mu_Tb, dim=-1)

        # Eval-only 1823 text bank for diagnostic readout of EEG->Text (NEVER used to decode prompts!)
        self.H_e1823 = nz(self.tb["h14_eval1823"].float() - self.mu_Th, dim=-1)
        self.B_e1823 = nz(self.tb["bigg_eval1823"].float() - self.mu_Tb, dim=-1)

        self.objsem = None
        if use_objsem:
            p = RAD / f"runs/obj_bench/obj_sem_enc/{objsem_tag}/ens_tta_all10.pt"
            if p.exists():
                self.objsem = torch.load(p, map_location="cpu", weights_only=False)
            else:
                raise FileNotFoundError(f"ObjSemEnc ensemble not found at {p}")

    def decode_subject(self, sub: int, prompt_style: str = "top2_cat", shuffle_seed: int = -1):
        """Returns (prompts_list_200, c_o14, z0_o14, bigg_sem_vec, diag_dict)."""
        pk = torch.load(RAD / f"runs/o14_official/cond/sub-{sub:02d}_o14_pack.pt", map_location="cpu", weights_only=False)
        c_o14 = pk["c_o14"].float()  # [200, 1024]
        # Load O14 z0 latent if present, else fallback
        z0_path = RAD / f"runs/o14_official/cond/sub-{sub:02d}_o14_z0.pt"
        if "z0_v2" in pk:
            z0 = pk["z0_v2"].float()
        elif "z0" in pk:
            z0 = pk["z0"].float()
        elif z0_path.exists():
            z0 = torch.load(z0_path, map_location="cpu", weights_only=False).float()
        else:
            z0 = None

        # O14 visual-to-text stream (train-only ridge bridge + centered h_tta & ppe_hat)
        c_u = nz(c_o14, dim=-1)
        cc = c_u - c_u.mean(0, keepdim=True)
        qh_o14 = nz(cc @ self.W_fh, dim=-1)
        qb_o14 = nz(cc @ self.W_fb, dim=-1)
        h_c = nz(pk["h_tta"].float() - pk["h_tta"].float().mean(0, keepdim=True), dim=-1)
        p_c = nz(pk["ppe_hat"].float() - pk["ppe_hat"].float().mean(0, keepdim=True), dim=-1)

        s_o14_v = (
            0.35 * zscore_rows(qh_o14 @ self.H_v.T) +
            0.35 * zscore_rows(qb_o14 @ self.B_v.T) +
            0.15 * zscore_rows(h_c @ self.H_v.T) +
            0.15 * zscore_rows(p_c @ self.B_v.T)
        )
        s_o14_c = 0.5 * zscore_rows(qh_o14 @ self.H_cat.T) + 0.5 * zscore_rows(qb_o14 @ self.B_cat.T)
        s_o14_e = (
            0.35 * zscore_rows(qh_o14 @ self.H_e1823.T) +
            0.35 * zscore_rows(qb_o14 @ self.B_e1823.T) +
            0.15 * zscore_rows(h_c @ self.H_e1823.T) +
            0.15 * zscore_rows(p_c @ self.B_e1823.T)
        )

        if self.use_objsem and self.objsem is not None:
            od = self.objsem[f"sub-{sub:02d}"]
            qh_sem = od["h14_txt"].float()
            qb_sem = od["bigg_txt"].float()
            cat_p = od["cat27_prob"].float()
            s_sem_v = 0.5 * zscore_rows(qh_sem @ self.H_v.T) + 0.5 * zscore_rows(qb_sem @ self.B_v.T)
            s_sem_e = 0.5 * zscore_rows(qh_sem @ self.H_e1823.T) + 0.5 * zscore_rows(qb_sem @ self.B_e1823.T)
            # Combine ObjSemEnc (0.55) + O14-bridge (0.45)
            s_v = 0.55 * zscore_rows(s_sem_v) + 0.45 * zscore_rows(s_o14_v)
            s_e = 0.55 * zscore_rows(s_sem_e) + 0.45 * zscore_rows(s_o14_e)
            s_sem_c = 0.5 * zscore_rows(qh_sem @ self.H_cat.T) + 0.5 * zscore_rows(qb_sem @ self.B_cat.T)
            cat_score = 0.55 * zscore_rows(s_sem_c) + 0.45 * zscore_rows(s_o14_c)
            bigg_sem = nz(0.55 * qb_sem + 0.45 * qb_o14, dim=-1)
        else:
            s_v = s_o14_v
            s_e = s_o14_e
            cat_score = s_o14_c
            bigg_sem = nz(0.65 * qb_o14 + 0.35 * p_c, dim=-1)

        # Diagnostic readout of EEG -> Eval1823 (purely logging, never feeds generation!)
        diag_idx = torch.arange(200)
        e200_t1 = float((s_e[:, :200].argmax(-1) == diag_idx).float().mean().item() * 100.0)
        e200_t5 = float((s_e[:, :200].topk(5, -1).indices == diag_idx[:, None]).any(-1).float().mean().item() * 100.0)
        e1823_t1 = float((s_e.argmax(-1) == diag_idx).float().mean().item() * 100.0)
        e1823_t5 = float((s_e.topk(5, -1).indices == diag_idx[:, None]).any(-1).float().mean().item() * 100.0)

        # Build natural prompts purely from CV-OpenVocab + Cat27
        prompts = []
        top_words_log = []
        for i in range(200):
            topv = s_v[i].topk(3)
            ws = [self.words[j] for j in topv.indices.tolist()]
            conf = float(topv.values[0].item())
            cat1 = self.cat27[int(cat_score[i].argmax().item())]
            top_words_log.append((ws, cat1, round(conf, 2)))
            if prompt_style == "empty":
                prompts.append("")
            elif prompt_style == "top1":
                prompts.append(f"a clear photograph of a {ws[0]}, centered object")
            elif prompt_style == "top1_clean":
                prompts.append(f"a photo of a {ws[0]}")
            elif prompt_style == "top2_cat":
                if conf >= 3.2:
                    prompts.append(f"a clear photograph of a {ws[0]}")
                else:
                    prompts.append(f"a clear photograph of a {ws[0]}, {ws[1]}, {cat1}")
            elif prompt_style == "top2":
                prompts.append(f"a clear photograph of a {ws[0]}, {ws[1]}")
            elif prompt_style == "cat_only":
                prompts.append(f"a clear photograph of a {cat1}")
            else:
                raise ValueError(prompt_style)

        if shuffle_seed >= 0:
            rng = np.random.default_rng(shuffle_seed + sub * 17)
            perm = rng.permutation(200)
            prompts = [prompts[j] for j in perm]

        return prompts, c_o14, z0, bigg_sem, {
            "eeg_txt200_t1": e200_t1, "eeg_txt200_t5": e200_t5,
            "eeg_txt1823_t1": e1823_t1, "eeg_txt1823_t5": e1823_t5,
            "top_words_sample12": top_words_log[:12],
        }

class ObjDiffGenerator:
    """SDXL-Turbo generator with controllable Native Dual-Text Prompts, IP-Adapter scale, z0 strength, and CFG."""
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

    def set_ip_scale(self, scale: float):
        for proc in self.attn_adapter.attn_processors.values():
            proc.scale = float(scale)

    @torch.inference_mode()
    def generate(self, c_all: torch.Tensor, prompts: list, z0_all: torch.Tensor = None,
                 ip_scale: float = 0.65, z0_strength: float = 0.0, steps: int = 4,
                 guidance_scale: float = 0.0, bigg_sem: torch.Tensor = None, alpha_ppe: float = 0.0,
                 bs: int = 20):
        self.set_ip_scale(ip_scale)
        out_imgs = torch.empty(200, 3, 512, 512, dtype=torch.uint8)

        # Pre-decode z0 init images if z0_strength > 0 (Note: in img2img, strength=1.0-z0_strength? No: strength=0.75 means 25% z0!)
        init_imgs = None
        if z0_strength > 0.0 and z0_all is not None:
            init_imgs = torch.cat([
                (self.vae32.decode(z0_all[j:j + 10].to(self.dev, torch.float32) / self.vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
                for j in range(0, 200, 10)
            ])

        do_cfg = guidance_scale > 1.0
        ppe_norm = float(self.ppe_empty.float().norm().item())

        for s in range(0, 200, bs):
            q = list(range(s, min(200, s + bs)))
            Q = len(q)
            cc = c_all[q].to(self.dev, self.dt)
            ips = self.proj(self.adapter(cc))
            if do_cfg:
                ips_in = torch.cat([torch.zeros_like(ips), ips], dim=0)
            else:
                ips_in = ips

            p_list = [prompts[i] for i in q]
            pe_b, neg_pe_b, ppe_b, neg_ppe_b = self.pipe.encode_prompt(
                prompt=p_list,
                negative_prompt=(["blurry, abstract, distorted, low quality"] * Q if do_cfg else None),
                device=self.dev,
                do_classifier_free_guidance=do_cfg,
            )
            if bigg_sem is not None and alpha_ppe > 0:
                pd = bigg_sem[q].to(self.dev, torch.float32)
                ppe_b = (ppe_b.float() + alpha_ppe * ppe_norm * pd).to(self.dt)

            gens = [torch.Generator(self.dev).manual_seed(1_000_003 * i + 12345) for i in q]

            if init_imgs is not None:
                # Img2img denoising from mild z0: img2img strength = 1.0 - z0_strength (e.g. z0_strength=0.25 -> strength=0.75)
                denoise_strength = max(0.5, min(1.0, 1.0 - z0_strength))
                n_steps_call = max(steps, int(math.ceil(steps / denoise_strength)))
                lat = self.p2(
                    image=init_imgs[q].to(self.dev, self.dt),
                    prompt_embeds=pe_b,
                    negative_prompt_embeds=(neg_pe_b if do_cfg else None),
                    pooled_prompt_embeds=ppe_b,
                    negative_pooled_prompt_embeds=(neg_ppe_b if do_cfg else None),
                    num_inference_steps=n_steps_call,
                    strength=denoise_strength,
                    guidance_scale=guidance_scale,
                    output_type="latent",
                    generator=gens,
                    cross_attention_kwargs={"ip_hidden_states": ips_in},
                ).images
            else:
                lat = self.pipe(
                    prompt_embeds=pe_b,
                    negative_prompt_embeds=(neg_pe_b if do_cfg else None),
                    pooled_prompt_embeds=ppe_b,
                    negative_pooled_prompt_embeds=(neg_ppe_b if do_cfg else None),
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    output_type="latent",
                    generator=gens,
                    cross_attention_kwargs={"ip_hidden_states": ips_in},
                ).images

            rgb = (self.vae32.decode(lat.float() / self.vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0, 1)
            out_imgs[q] = (rgb * 255.0).round().to(torch.uint8).cpu()
        return out_imgs

def run_config(name: str, subs: list, gen: ObjDiffGenerator, ev: ObjReconEvaluator,
               dec: SemanticDecoder, prompt_style: str, ip_scale: float, z0_strength: float,
               steps: int = 4, guidance_scale: float = 0.0, alpha_ppe: float = 0.0,
               shuffle_seed: int = -1, save_imgs: bool = True):
    out_d = OUT_ROOT / name
    out_d.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for s in subs:
        prompts, c_o14, z0, bigg_sem, diag = dec.decode_subject(s, prompt_style=prompt_style, shuffle_seed=shuffle_seed)
        imgs = gen.generate(
            c_o14, prompts, z0_all=z0, ip_scale=ip_scale, z0_strength=z0_strength,
            steps=steps, guidance_scale=guidance_scale, bigg_sem=bigg_sem, alpha_ppe=alpha_ppe
        )
        m = ev.evaluate(imgs, compute_lowlevel=True)
        m.update({k: v for k, v in diag.items() if not isinstance(v, list)})
        rows.append(m)
        if save_imgs:
            torch.save({"k0_rgb_u8": imgs, "prompts": prompts, "metrics": m}, out_d / f"sub-{s:02d}_imgs.pt")
        print(f"  [{name:28s}] sub-{s:02d} | ZS200={m['zs_obj200_t1']:5.2f}/{m['zs_obj200_t5']:5.2f} | ZS1823={m['zs_obj1823_t1']:4.2f}/{m['zs_obj1823_t5']:5.2f} | Cat27={m['coarse27_t1']:4.1f}% | IN1K={m['in1k_inc_t1']:4.1f}/{m['in1k_inc_t5']:4.1f} | Crisp={m['obj_crisp']:4.1f}% | Img200={m['clip_img200_t1']:5.2f}% | 2wC={m['clip_2way']:5.2f}% | SSIM={m['ssim']:.3f}", flush=True)
    mean_m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    json.dump({"subs": subs, "per_sub": rows, "mean": mean_m}, open(out_d / f"summary_{len(subs)}sub.json", "w"), indent=2)
    print(f"==> [{name:28s}] {len(subs)}-SUB MEAN ({time.time()-t0:.1f}s): ZS200={mean_m['zs_obj200_t1']:5.2f}/{mean_m['zs_obj200_t5']:5.2f} | ZS1823={mean_m['zs_obj1823_t1']:4.2f}/{mean_m['zs_obj1823_t5']:5.2f} | Cat27={mean_m['coarse27_t1']:4.1f}% | IN1K={mean_m['in1k_inc_t1']:4.1f}/{mean_m['in1k_inc_t5']:4.1f} | Crisp={mean_m['obj_crisp']:4.1f}% | Img200={mean_m['clip_img200_t1']:5.2f}% | 2wC={mean_m['clip_2way']:5.2f}% | Incep={mean_m['incep_2way']:5.2f}% | SSIM={mean_m['ssim']:.3f}\n", flush=True)
    return mean_m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["dev4_phase1", "dev4_phase2", "all10_confirm"])
    ap.add_argument("--winner_cfg", type=str, default="")
    args = ap.parse_args()

    gen = ObjDiffGenerator("cuda:0")
    ev = ObjReconEvaluator("cuda:0")
    dev4 = [1, 2, 8, 10]

    if args.mode == "dev4_phase1":
        # Phase 1: Pure O14-only ablations (no ObjSemEnc yet): z0 release + O14-bridge CV-OpenVocab prompt!
        dec_o14 = SemanticDecoder(use_objsem=False, vocab_mode="cv_open")
        run_config("dev4_A1_o14_freez0_notxt", dev4, gen, ev, dec_o14, prompt_style="empty", ip_scale=1.0, z0_strength=0.0, steps=4)
        run_config("dev4_A2_o14_bridge_top2cat_ip055", dev4, gen, ev, dec_o14, prompt_style="top2_cat", ip_scale=0.55, z0_strength=0.0, steps=4)
        run_config("dev4_A3_o14_bridge_top2cat_ip075", dev4, gen, ev, dec_o14, prompt_style="top2_cat", ip_scale=0.75, z0_strength=0.0, steps=4)
        run_config("dev4_A4_o14_bridge_top1_ip055", dev4, gen, ev, dec_o14, prompt_style="top1_clean", ip_scale=0.55, z0_strength=0.0, steps=4)
    elif args.mode == "dev4_phase2":
        # Phase 2: Full ObjSemEnc + O14 Dual-Stream ablations (including Continuous-NoDict, Shuffled-Prompt control, ip_scale, z0, CFG)
        dec_sem = SemanticDecoder(use_objsem=True, objsem_tag="objsem_v1", vocab_mode="cv_open")
        dec_comp = SemanticDecoder(use_objsem=True, objsem_tag="objsem_v1", vocab_mode="cv_comp")
        # 1. Zero-Dictionary Continuous Semantic-Text Injection (no open-vocab words!)
        run_config("dev4_B0_objsem_nodict_cont", dev4, gen, ev, dec_sem, prompt_style="empty", ip_scale=1.0, z0_strength=0.0, steps=4, alpha_ppe=0.45)
        # 2. Full ObjEEG-Diff (CV-OpenVocab 8,141) across ip_scale & prompt style
        run_config("dev4_B1_objeeg_top2cat_ip035", dev4, gen, ev, dec_sem, prompt_style="top2_cat", ip_scale=0.35, z0_strength=0.0, steps=4)
        run_config("dev4_B2_objeeg_top2cat_ip055", dev4, gen, ev, dec_sem, prompt_style="top2_cat", ip_scale=0.55, z0_strength=0.0, steps=4)
        run_config("dev4_B3_objeeg_top2cat_ip075", dev4, gen, ev, dec_sem, prompt_style="top2_cat", ip_scale=0.75, z0_strength=0.0, steps=4)
        run_config("dev4_B4_objeeg_top1_ip055", dev4, gen, ev, dec_sem, prompt_style="top1_clean", ip_scale=0.55, z0_strength=0.0, steps=4)
        run_config("dev4_B5_objeeg_cvcomp_ip055", dev4, gen, ev, dec_comp, prompt_style="top2_cat", ip_scale=0.55, z0_strength=0.0, steps=4)
        # 3. Causal Shuffled-Prompt Control (MUST drop sharply!)
        run_config("dev4_B_CTRL_SHUF_ip055", dev4, gen, ev, dec_sem, prompt_style="top2_cat", ip_scale=0.55, z0_strength=0.0, steps=4, shuffle_seed=999)
        # 4. Higher-NFE Mild CFG (8 steps, cfg=2.0, clearly separated budget!)
        run_config("dev4_B6_objeeg_cfg2_8st_ip055", dev4, gen, ev, dec_sem, prompt_style="top2_cat", ip_scale=0.55, z0_strength=0.0, steps=8, guidance_scale=2.0)

if __name__ == "__main__":
    main()
