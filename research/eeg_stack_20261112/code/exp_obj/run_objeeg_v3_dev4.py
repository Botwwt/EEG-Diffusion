"""ObjEEG-Diff v3 Dev-4: Uncoupled beta=0 SoftOpenText + ObjSemEnc + z0 smap_c + Sel4 Dual-Space Verification."""
import os, sys, time, json
from pathlib import Path
import numpy as np
import torch

RAD = Path(os.environ.get("RAD_ROOT", "/path/to/rad"))
for sub in ("code/exp_obj", "code/exp_w7", "code/exp_w6", "code/rad", "code/exp", "code/Brain-HIVE"):
    p = str(RAD / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_obj_suite import ObjReconEvaluator
from run_objeeg_v2 import DualStreamSemanticEngine, ObjDiffV2Generator, LAIONH14Reranker, OUT_ROOT, log_and_save


def run_v3_arm(name, subs, eng, gen, ev, cfg, reranker=None):
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
                c_use, sp["z0"], bigg_sem=sp["bigg_sem"], alpha_ppe=cfg.get("alpha_ppe", 0.50),
                ip_scale=cfg.get("ip_scale", 1.0), smap_c=cfg.get("smap_c", 0.375),
                low_keep=(8, 0.0, 0.0),
            )
        elif mode == "soft":
            pe, ppe = gen.build_soft_open_text_embeds(
                sp["s_cv"], eng.words, topk=cfg.get("topk", 8), tau=cfg.get("tau", 0.35),
                gamma=cfg.get("gamma", 0.50), shuffle_seed=cfg.get("shuffle_seed", -1),
            )
            imgs = gen.generate(
                c_use, sp["z0"], pe_all=pe, ppe_all=ppe, bigg_sem=sp["bigg_sem"],
                alpha_ppe=cfg.get("alpha_ppe", 0.30), ip_scale=cfg.get("ip_scale", 1.0),
                smap_c=cfg.get("smap_c", 0.375), low_keep=(8, 0.0, 0.0),
            )
        elif mode == "sel4_beta0":
            pe_s1, ppe_s1 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=8, tau=0.35, gamma=0.45)
            pe_s2, ppe_s2 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=8, tau=0.35, gamma=0.65)
            pe_h0, ppe_h0 = gen.build_hard_rank_text_embeds(sp["s_cv"], eng.words, rank_k=0, gamma=0.75)
            specs = [
                (None, None, 0.50, 1.00, 0),
                (pe_s1, ppe_s1, 0.30, 1.00, 1),
                (pe_s2, ppe_s2, 0.25, 0.95, 2),
                (pe_h0, ppe_h0, 0.20, 0.85, 3),
            ]
            cands = []
            for pe_k, ppe_k, ap_k, ip_k, sd_k in specs:
                cands.append(gen.generate(
                    c_use, sp["z0"], pe_all=pe_k, ppe_all=ppe_k, bigg_sem=sp["bigg_sem"],
                    alpha_ppe=ap_k, ip_scale=ip_k, smap_c=cfg.get("smap_c", 0.375),
                    low_keep=(8, 0.0, 0.0), seed_offset=sd_k,
                ))
            imgs = reranker.select_best(
                cands, sp["h_tta"], sp["h14_sem"], sp["mu_Th"],
                w_vis=0.45, w_sem=0.40, w_obj=0.15,
            )
        elif mode == "sel8_beta0":
            pe_s1, ppe_s1 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=8, tau=0.35, gamma=0.45)
            pe_s2, ppe_s2 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=8, tau=0.35, gamma=0.65)
            specs = [
                (None, None, 0.50, 1.00, 0),
                (None, None, 0.50, 1.00, 1),
                (pe_s1, ppe_s1, 0.30, 1.00, 2),
                (pe_s2, ppe_s2, 0.25, 0.95, 3),
            ]
            for rk in range(4):
                pe_h, ppe_h = gen.build_hard_rank_text_embeds(sp["s_cv"], eng.words, rank_k=rk, gamma=0.75)
                specs.append((pe_h, ppe_h, 0.20, 0.85, 4 + rk))
            cands = []
            for pe_k, ppe_k, ap_k, ip_k, sd_k in specs:
                cands.append(gen.generate(
                    c_use, sp["z0"], pe_all=pe_k, ppe_all=ppe_k, bigg_sem=sp["bigg_sem"],
                    alpha_ppe=ap_k, ip_scale=ip_k, smap_c=cfg.get("smap_c", 0.375),
                    low_keep=(8, 0.0, 0.0), seed_offset=sd_k,
                ))
            imgs = reranker.select_best(
                cands, sp["h_tta"], sp["h14_sem"], sp["mu_Th"],
                w_vis=0.45, w_sem=0.40, w_obj=0.15,
            )
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
    eng = DualStreamSemanticEngine("objsem_v1")
    gen = ObjDiffV2Generator("cuda:0")
    ev = ObjReconEvaluator("cuda:0")
    reranker = LAIONH14Reranker(eng.H_cv_raw, dev="cuda:0")
    dev4 = [1, 2, 8, 10]

    run_v3_arm("v3_W1_b0_soft16_g040_ip100_s375", dev4, eng, gen, ev, {"mode": "soft", "beta": 0.0, "topk": 16, "tau": 0.45, "gamma": 0.40, "ip_scale": 1.00, "smap_c": 0.375, "alpha_ppe": 0.30})
    run_v3_arm("v3_W2_b0_soft8_g055_ip095_s375",  dev4, eng, gen, ev, {"mode": "soft", "beta": 0.0, "topk": 8,  "tau": 0.35, "gamma": 0.55, "ip_scale": 0.95, "smap_c": 0.375, "alpha_ppe": 0.25})
    run_v3_arm("v3_W3_b0_soft8_g055_ip095_s250",  dev4, eng, gen, ev, {"mode": "soft", "beta": 0.0, "topk": 8,  "tau": 0.35, "gamma": 0.55, "ip_scale": 0.95, "smap_c": 0.250, "alpha_ppe": 0.25})
    run_v3_arm("v3_W4_b0_cont_s250",              dev4, eng, gen, ev, {"mode": "cont", "beta": 0.0, "alpha_ppe": 0.50, "ip_scale": 1.00, "smap_c": 0.250})
    run_v3_arm("v3_W5_b0_sel4_semrerank",         dev4, eng, gen, ev, {"mode": "sel4_beta0", "beta": 0.0, "smap_c": 0.375}, reranker=reranker)
    run_v3_arm("v3_W6_b0_sel8_semrerank",         dev4, eng, gen, ev, {"mode": "sel8_beta0", "beta": 0.0, "smap_c": 0.375}, reranker=reranker)


if __name__ == "__main__":
    main()
