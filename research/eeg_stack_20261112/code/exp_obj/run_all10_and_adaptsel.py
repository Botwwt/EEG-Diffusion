"""All-10 4-NFE Confirmatory Runner & Dev-4 Adaptive-z0 Crisp Hypothesis-Proposal Sel4/Sel8."""
import os, sys, time, json, argparse
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

CONFIRM_ROOT = RAD / "runs/obj_bench/confirm_all10"


def run_named(name, subs, eng, gen, ev, mode, reranker=None, root=CONFIRM_ROOT):
    out_d = root / name
    out_d.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for s in subs:
        cache_p = out_d / f"sub-{s:02d}_imgs.pt"
        if cache_p.exists():
            m = torch.load(cache_p, map_location="cpu", weights_only=False)["metrics"]
            rows.append(m)
            print(f"  [CACHED {name:28s}] sub-{s:02d} | ZS200={m['zs_obj200_t1']:5.2f} | ZS1823={m['zs_obj1823_t1']:4.2f} | Img200={m['clip_img200_t1']:5.2f}", flush=True)
            continue

        sp = eng.get_subject_pack(s, manifold_beta=0.0)
        c_use = sp["c_o14"]

        if mode == "cont_4nfe":
            imgs = gen.generate(
                c_use, sp["z0"], bigg_sem=sp["bigg_sem"], alpha_ppe=0.45,
                ip_scale=1.0, smap_c=0.375, low_keep=(8, 0.0, 0.0),
            )
        elif mode == "softopen_4nfe":
            pe, ppe = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=8, tau=0.35, gamma=0.55)
            imgs = gen.generate(
                c_use, sp["z0"], pe_all=pe, ppe_all=ppe, bigg_sem=sp["bigg_sem"],
                alpha_ppe=0.25, ip_scale=0.95, smap_c=0.375, low_keep=(8, 0.0, 0.0),
            )
        elif mode == "softopen_shuf_4nfe":
            pe, ppe = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=8, tau=0.35, gamma=0.55, shuffle_seed=777 + s * 13)
            imgs = gen.generate(
                c_use, sp["z0"], pe_all=pe, ppe_all=ppe, bigg_sem=None,
                alpha_ppe=0.0, ip_scale=0.95, smap_c=0.375, low_keep=(8, 0.0, 0.0),
            )
        elif mode == "o14_sel4_ctrl":
            cands = [
                gen.generate(c_use, sp["z0"], bigg_sem=None, alpha_ppe=0.0, ip_scale=1.0, smap_c=0.625, low_keep=(8, 0.22, 0.04), seed_offset=k)
                for k in range(4)
            ]
            imgs = reranker.select_best(cands, sp["h_tta"], sp["h14_sem"], sp["mu_Th"], w_vis=1.0, w_sem=0.0, w_obj=0.0)
        elif mode == "adaptsel4":
            # 2 layout-anchored (Cont + SoftOpen @ s375) + 2 free-z0 crisp object hypotheses (Top-1, Top-2 @ s0, ip=0.70)!
            pe_s1, ppe_s1 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=8, tau=0.35, gamma=0.55)
            pe_h0, ppe_h0 = gen.build_hard_rank_text_embeds(sp["s_cv"], eng.words, rank_k=0, gamma=0.90)
            pe_h1, ppe_h1 = gen.build_hard_rank_text_embeds(sp["s_cv"], eng.words, rank_k=1, gamma=0.90)
            specs = [
                (None, None, 0.45, 1.00, 0.375, 0),
                (pe_s1, ppe_s1, 0.25, 0.95, 0.375, 1),
                (pe_h0, ppe_h0, 0.15, 0.70, 0.000, 2),
                (pe_h1, ppe_h1, 0.15, 0.70, 0.000, 3),
            ]
            cands = [
                gen.generate(c_use, sp["z0"], pe_all=pe_k, ppe_all=ppe_k, bigg_sem=sp["bigg_sem"],
                             alpha_ppe=ap_k, ip_scale=ip_k, smap_c=sm_k, low_keep=(8, 0.0, 0.0), seed_offset=sd_k)
                for pe_k, ppe_k, ap_k, ip_k, sm_k, sd_k in specs
            ]
            imgs = reranker.select_best(cands, sp["h_tta"], sp["h14_sem"], sp["mu_Th"], w_vis=0.45, w_sem=0.35, w_obj=0.20)
        elif mode == "adaptsel8":
            # 4 layout-anchored (2 Cont + 2 SoftOpen @ s375) + 4 free-z0 crisp object hypotheses (Top-1..4 @ s0, ip=0.70)!
            pe_s1, ppe_s1 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=8, tau=0.35, gamma=0.55)
            pe_s2, ppe_s2 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=16, tau=0.45, gamma=0.45)
            specs = [
                (None, None, 0.45, 1.00, 0.375, 0),
                (None, None, 0.45, 1.00, 0.375, 1),
                (pe_s1, ppe_s1, 0.25, 0.95, 0.375, 2),
                (pe_s2, ppe_s2, 0.25, 1.00, 0.375, 3),
            ]
            for rk in range(4):
                pe_h, ppe_h = gen.build_hard_rank_text_embeds(sp["s_cv"], eng.words, rank_k=rk, gamma=0.90)
                specs.append((pe_h, ppe_h, 0.15, 0.70, 0.000, 4 + rk))
            cands = [
                gen.generate(c_use, sp["z0"], pe_all=pe_k, ppe_all=ppe_k, bigg_sem=sp["bigg_sem"],
                             alpha_ppe=ap_k, ip_scale=ip_k, smap_c=sm_k, low_keep=(8, 0.0, 0.0), seed_offset=sd_k)
                for pe_k, ppe_k, ap_k, ip_k, sm_k, sd_k in specs
            ]
            imgs = reranker.select_best(cands, sp["h_tta"], sp["h14_sem"], sp["mu_Th"], w_vis=0.45, w_sem=0.35, w_obj=0.20)
        elif mode == "b0_sel8_s375":
            # 2 Cont + 2 SoftOpen + 4 Top-1..4 ALL @ s375, ip=0.85..1.00
            pe_s1, ppe_s1 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=8, tau=0.35, gamma=0.45)
            pe_s2, ppe_s2 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=8, tau=0.35, gamma=0.65)
            specs = [
                (None, None, 0.50, 1.00, 0.375, 0),
                (None, None, 0.50, 1.00, 0.375, 1),
                (pe_s1, ppe_s1, 0.30, 1.00, 0.375, 2),
                (pe_s2, ppe_s2, 0.25, 0.95, 0.375, 3),
            ]
            for rk in range(4):
                pe_h, ppe_h = gen.build_hard_rank_text_embeds(sp["s_cv"], eng.words, rank_k=rk, gamma=0.75)
                specs.append((pe_h, ppe_h, 0.20, 0.85, 0.375, 4 + rk))
            cands = [
                gen.generate(c_use, sp["z0"], pe_all=pe_k, ppe_all=ppe_k, bigg_sem=sp["bigg_sem"],
                             alpha_ppe=ap_k, ip_scale=ip_k, smap_c=sm_k, low_keep=(8, 0.0, 0.0), seed_offset=sd_k)
                for pe_k, ppe_k, ap_k, ip_k, sm_k, sd_k in specs
            ]
            imgs = reranker.select_best(cands, sp["h_tta"], sp["h14_sem"], sp["mu_Th"], w_vis=0.45, w_sem=0.40, w_obj=0.15)
        elif mode == "b0_sel4_s375":
            pe_s1, ppe_s1 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=8, tau=0.35, gamma=0.45)
            pe_s2, ppe_s2 = gen.build_soft_open_text_embeds(sp["s_cv"], eng.words, topk=8, tau=0.35, gamma=0.65)
            pe_h0, ppe_h0 = gen.build_hard_rank_text_embeds(sp["s_cv"], eng.words, rank_k=0, gamma=0.75)
            specs = [
                (None, None, 0.50, 1.00, 0.375, 0),
                (pe_s1, ppe_s1, 0.30, 1.00, 0.375, 1),
                (pe_s2, ppe_s2, 0.25, 0.95, 0.375, 2),
                (pe_h0, ppe_h0, 0.20, 0.85, 0.375, 3),
            ]
            cands = [
                gen.generate(c_use, sp["z0"], pe_all=pe_k, ppe_all=ppe_k, bigg_sem=sp["bigg_sem"],
                             alpha_ppe=ap_k, ip_scale=ip_k, smap_c=sm_k, low_keep=(8, 0.0, 0.0), seed_offset=sd_k)
                for pe_k, ppe_k, ap_k, ip_k, sm_k, sd_k in specs
            ]
            imgs = reranker.select_best(cands, sp["h_tta"], sp["h14_sem"], sp["mu_Th"], w_vis=0.45, w_sem=0.40, w_obj=0.15)
        else:
            raise ValueError(mode)

        m = ev.evaluate(imgs, compute_lowlevel=True)
        rows.append(m)
        torch.save({"k0_rgb_u8": imgs, "metrics": m}, cache_p)
        print(
            f"  [{name:28s}] sub-{s:02d} | ZS200={m['zs_obj200_t1']:5.2f}/{m['zs_obj200_t5']:5.2f} | "
            f"ZS1823={m['zs_obj1823_t1']:4.2f}/{m['zs_obj1823_t5']:5.2f} | Cat27={m['coarse27_t1']:4.1f}% | "
            f"IN1K={m['in1k_inc_t1']:4.1f}/{m['in1k_inc_t5']:4.1f} | Crisp={m['obj_crisp']:4.1f}% | "
            f"Img200={m['clip_img200_t1']:5.2f}% | 2wC={m['clip_2way']:5.2f}% | SSIM={m['ssim']:.3f}",
            flush=True,
        )
    return log_and_save(name, subs, rows, out_d, t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True)
    args = ap.parse_args()

    eng = DualStreamSemanticEngine("objsem_v1")
    gen = ObjDiffV2Generator("cuda:0")
    ev = ObjReconEvaluator("cuda:0")
    dev4 = [1, 2, 8, 10]
    all10 = list(range(1, 11))

    if args.task == "dev4_adaptsel":
        rr = LAIONH14Reranker(eng.H_cv_raw, dev="cuda:0")
        run_named("dev4_AdaptSel4", dev4, eng, gen, ev, "adaptsel4", reranker=rr, root=OUT_ROOT)
        run_named("dev4_AdaptSel8", dev4, eng, gen, ev, "adaptsel8", reranker=rr, root=OUT_ROOT)
    elif args.task == "all10_4nfe":
        run_named("ObjEEG_Cont_4NFE", all10, eng, gen, ev, "cont_4nfe")
        run_named("ObjEEG_SoftOpen_4NFE", all10, eng, gen, ev, "softopen_4nfe")
        run_named("ObjEEG_SoftOpen_ShufCtrl_4NFE", all10, eng, gen, ev, "softopen_shuf_4nfe")
    elif args.task == "all10_o14_sel4":
        rr = LAIONH14Reranker(eng.H_cv_raw, dev="cuda:0")
        run_named("O14_Sel4_16NFE_Ctrl", all10, eng, gen, ev, "o14_sel4_ctrl", reranker=rr)
    elif args.task == "all10_sel4":
        rr = LAIONH14Reranker(eng.H_cv_raw, dev="cuda:0")
        run_named("ObjEEG_Sel4_16NFE", all10, eng, gen, ev, "b0_sel4_s375", reranker=rr)
    elif args.task == "all10_sel8":
        rr = LAIONH14Reranker(eng.H_cv_raw, dev="cuda:0")
        run_named("ObjEEG_Sel8_32NFE", all10, eng, gen, ev, "b0_sel8_s375", reranker=rr)
    elif args.task == "all10_adaptsel8":
        rr = LAIONH14Reranker(eng.H_cv_raw, dev="cuda:0")
        run_named("ObjEEG_AdaptSel8_32NFE", all10, eng, gen, ev, "adaptsel8", reranker=rr)
    else:
        raise ValueError(args.task)


if __name__ == "__main__":
    main()
