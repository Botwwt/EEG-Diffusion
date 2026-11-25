"""ObjEEG-Diff-DualVerify: Crisp Open-World Hypothesis Proposals + Zero-Leakage LAION F-Space + ObjSemEnc Verifier.
100% ZERO OpenAI ViT-L/14, ZERO Inception/EffNet, ZERO THINGS test labels anywhere in proposal or verification!
"""
import os, sys, time, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

RAD = Path(os.environ.get("RAD_ROOT", "/path/to/rad"))
for sub in ("code/exp_obj", "code/exp_w7", "code/exp_w6", "code/rad", "code/exp", "code/Brain-HIVE"):
    p = str(RAD / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from common import EMB_KEYS
from fenc import FEmbedder
from eval_obj_suite import ObjReconEvaluator
from run_objeeg_v2 import DualStreamSemanticEngine, ObjDiffV2Generator

CONFIRM_ROOT = RAD / "runs/obj_bench/confirm_all10"


def nz(x, dim=-1):
    return F.normalize(x.float(), dim=dim)


class DualBrainFVerifier:
    """Zero-leakage LAION Dual-Brain Verifier: combines 58.8%-accurate F-space (H14+B32+VAE) + ObjSemEnc + CV-OpenVocab Objectness."""
    def __init__(self, H_cv_raw: torch.Tensor, mu_F: torch.Tensor, dev="cuda:0"):
        self.dev = dev
        self.fe = FEmbedder(dev=dev)
        self.H_cv_raw = H_cv_raw.to(dev, torch.float32)  # [8141, 1024]
        self.mu_F = mu_F.to(dev, torch.float32)          # [1, 1024]

    @torch.inference_mode()
    def select_best(self, cand_imgs_list: list, c_o14: torch.Tensor, h_tta: torch.Tensor,
                    h14_sem: torch.Tensor, w_f: float = 0.45, w_h: float = 0.20,
                    w_sem: float = 0.25, w_obj: float = 0.10):
        K = len(cand_imgs_list)
        cu_c = nz(nz(c_o14.to(self.dev), dim=-1) - self.mu_F, dim=-1)  # [200, 1024]
        hv = nz(h_tta.to(self.dev), dim=-1)
        hs = nz(h14_sem.to(self.dev), dim=-1)

        s_f = torch.zeros(K, 200, device=self.dev)
        s_h = torch.zeros(K, 200, device=self.dev)
        s_sem = torch.zeros(K, 200, device=self.dev)
        s_obj = torch.zeros(K, 200, device=self.dev)

        for k, imgs_u8 in enumerate(cand_imgs_list):
            imgs_f = imgs_u8.float() / 255.0
            fs, h14s = [], []
            for j in range(0, 200, 40):
                parts = self.fe.parts(imgs_f[j:j + 40])
                fs.append(self.fe.fuse(**parts).float())
                h14s.append(parts[EMB_KEYS[0]].float())
            F_img = torch.cat(fs, 0)       # [200, 1024]
            H_img = nz(torch.cat(h14s, 0), dim=-1)  # [200, 1024]

            F_img_c = nz(nz(F_img, dim=-1) - self.mu_F, dim=-1)
            s_f[k] = (F_img_c * cu_c).sum(-1)
            s_h[k] = (H_img * hv).sum(-1)
            H_img_c = nz(H_img - H_img.mean(0, keepdim=True), dim=-1)
            s_sem[k] = (H_img_c * hs).sum(-1)
            s_obj[k] = (H_img @ self.H_cv_raw.T).max(dim=-1).values

        def z_k(x):
            return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)

        total = w_f * z_k(s_f) + w_h * z_k(s_h) + w_sem * z_k(s_sem) + w_obj * z_k(s_obj)
        best_k = total.argmax(dim=0).cpu()
        out = torch.empty_like(cand_imgs_list[0])
        chosen_hist = [int((best_k == k).sum().item()) for k in range(K)]
        for i in range(200):
            out[i] = cand_imgs_list[int(best_k[i])][i]
        return out, chosen_hist


def run_job(arm: str, subs: list, dev: str = "cuda:0"):
    eng = DualStreamSemanticEngine("objsem_v1")
    gen = ObjDiffV2Generator(dev=dev)
    ev = ObjReconEvaluator(dev)
    ver = DualBrainFVerifier(eng.H_cv_raw, eng.mu_F, dev=dev) if "Sel" in arm else None

    out_d = CONFIRM_ROOT / arm
    out_d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for s in subs:
        cache_p = out_d / f"sub-{s:02d}_imgs.pt"
        if cache_p.exists():
            m = torch.load(cache_p, map_location="cpu", weights_only=False)["metrics"]
            print(f"  [CACHED {arm:26s}] sub-{s:02d} | ZS200={m['zs_obj200_t1']:5.2f} | ZS1823={m['zs_obj1823_t1']:4.2f} | Img200={m['clip_img200_t1']:5.2f}", flush=True)
            continue

        sp = eng.get_subject_pack(s, manifold_beta=0.0)
        c_use = sp["c_o14"]
        hist = None

        if arm == "ObjEEG_CrispHard_4NFE":
            # Single-draw 4-NFE crisp Top-1 CV-OpenVocab + ip=0.75 + s250 (maximizing object crispness at 4 NFE!)
            pe_h0, ppe_h0 = gen.build_hard_rank_text_embeds(sp["s_cv"], eng.words, rank_k=0, gamma=0.90)
            imgs = gen.generate(
                c_use, sp["z0"], pe_all=pe_h0, ppe_all=ppe_h0, bigg_sem=sp["bigg_sem"],
                alpha_ppe=0.15, ip_scale=0.75, smap_c=0.25, low_keep=(8, 0.0, 0.0), seed_offset=0,
            )
        elif arm == "ObjEEG_DualVerify_Sel4":
            # 16 NFE: 2 layout-safe (Cont + SoftOpen @ s375) + 2 Crisp Open-World Hypotheses (Top-1, Top-2 @ s150, ip=0.72)
            pe_s1, ppe_s1 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=8, tau=0.35, gamma=0.55)
            pe_h0, ppe_h0 = gen.build_hard_rank_text_embeds(sp["s_cv"], eng.words, rank_k=0, gamma=0.90)
            pe_h1, ppe_h1 = gen.build_hard_rank_text_embeds(sp["s_cv"], eng.words, rank_k=1, gamma=0.90)
            specs = [
                (None, None, 0.45, 1.00, 0.375, 0),
                (pe_s1, ppe_s1, 0.25, 0.95, 0.375, 1),
                (pe_h0, ppe_h0, 0.15, 0.72, 0.150, 2),
                (pe_h1, ppe_h1, 0.15, 0.72, 0.150, 3),
            ]
            cands = [
                gen.generate(c_use, sp["z0"], pe_all=pe_k, ppe_all=ppe_k, bigg_sem=sp["bigg_sem"],
                             alpha_ppe=ap_k, ip_scale=ip_k, smap_c=sm_k, low_keep=(8, 0.0, 0.0), seed_offset=sd_k)
                for pe_k, ppe_k, ap_k, ip_k, sm_k, sd_k in specs
            ]
            imgs, hist = ver.select_best(cands, c_use, sp["h_tta"], sp["h14_sem"], w_f=0.45, w_h=0.20, w_sem=0.25, w_obj=0.10)
        elif arm == "ObjEEG_DualVerify_Sel8":
            # 32 NFE: 3 layout-safe (2 Cont + 1 SoftOpen @ s375) + 5 Crisp Open-World Hypotheses (Top-1..5 @ s150, ip=0.72)
            pe_s1, ppe_s1 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=8, tau=0.35, gamma=0.55)
            specs = [
                (None, None, 0.45, 1.00, 0.375, 0),
                (None, None, 0.45, 1.00, 0.375, 1),
                (pe_s1, ppe_s1, 0.25, 0.95, 0.375, 2),
            ]
            for rk in range(5):
                pe_h, ppe_h = gen.build_hard_rank_text_embeds(sp["s_cv"], eng.words, rank_k=rk, gamma=0.90)
                specs.append((pe_h, ppe_h, 0.15, 0.72, 0.150, 3 + rk))
            cands = [
                gen.generate(c_use, sp["z0"], pe_all=pe_k, ppe_all=ppe_k, bigg_sem=sp["bigg_sem"],
                             alpha_ppe=ap_k, ip_scale=ip_k, smap_c=sm_k, low_keep=(8, 0.0, 0.0), seed_offset=sd_k)
                for pe_k, ppe_k, ap_k, ip_k, sm_k, sd_k in specs
            ]
            imgs, hist = ver.select_best(cands, c_use, sp["h_tta"], sp["h14_sem"], w_f=0.45, w_h=0.20, w_sem=0.25, w_obj=0.10)
        elif arm == "O14_FVerify_Sel8_Ctrl":
            # 100% matched 32-NFE O14 baseline (8 empty-prompt s625 draws ranked by exact same FEmbedder visual verifier!)
            cands = [
                gen.generate(c_use, sp["z0"], bigg_sem=None, alpha_ppe=0.0, ip_scale=1.0,
                             smap_c=0.625, low_keep=(8, 0.22, 0.04), seed_offset=k)
                for k in range(8)
            ]
            imgs, hist = ver.select_best(cands, c_use, sp["h_tta"], sp["h14_sem"], w_f=0.70, w_h=0.30, w_sem=0.0, w_obj=0.0)
        else:
            raise ValueError(arm)

        m = ev.evaluate(imgs, compute_lowlevel=True)
        if hist is not None:
            m["cand_hist"] = hist
        torch.save({"k0_rgb_u8": imgs, "metrics": m}, cache_p)
        print(
            f"  [{arm:24s}] sub-{s:02d} ({time.time()-t0:5.1f}s) | ZS200={m['zs_obj200_t1']:5.2f}/{m['zs_obj200_t5']:5.2f} | "
            f"ZS1823={m['zs_obj1823_t1']:4.2f}/{m['zs_obj1823_t5']:5.2f} | Cat27={m['coarse27_t1']:4.1f}% | "
            f"IN1K={m['in1k_inc_t1']:4.1f}/{m['in1k_inc_t5']:4.1f} | Crisp={m['obj_crisp']:4.1f}% | "
            f"Img200={m['clip_img200_t1']:5.2f}% | 2wC={m['clip_2way']:5.2f}% | SSIM={m['ssim']:.3f}" +
            (f" | hist={hist}" if hist else ""),
            flush=True,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=str, required=True)
    ap.add_argument("--subs", type=str, required=True)
    args = ap.parse_args()
    subs = [int(x) for x in args.subs.split(",") if x.strip()]
    run_job(args.arm, subs, dev="cuda:0")
