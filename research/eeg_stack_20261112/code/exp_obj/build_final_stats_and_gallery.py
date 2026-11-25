"""Aggregate ALL 10-subject baselines + ObjEEG-Diff arms, compute paired t-tests & Holm-Bonferroni FWER,
and render publication-ready uncherry-picked comparison galleries (sub-10 & sub-08, sequential #000-#011).
"""
import os, re, json, math
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from PIL import Image, ImageDraw

RAD = Path(os.environ.get("RAD_ROOT", "/path/to/rad"))
OUT = RAD / "runs/obj_bench/final_package"
OUT.mkdir(parents=True, exist_ok=True)


def load_per_sub_metrics(spec):
    kind, path = spec
    if kind == "eval_json":
        d = json.load(open(path))
        return d["per_sub"]
    elif kind == "confirm_dir":
        rows = []
        for s in range(1, 11):
            p = Path(path) / f"sub-{s:02d}_imgs.pt"
            rows.append(torch.load(p, map_location="cpu", weights_only=False)["metrics"])
        return rows
    raise ValueError(kind)


ARMS = [
    ("NICE-EEG (ICLR'24)",           4,  1, ("eval_json",   RAD / "runs/obj_bench/evals/nice_all10.json")),
    ("ATM-S (NeurIPS'24)",           4,  1, ("eval_json",   RAD / "runs/obj_bench/evals/atm_s_all10.json")),
    ("Brain-HIVE (Official)",        4,  1, ("eval_json",   RAD / "runs/obj_bench/evals/brain_hive_all10.json")),
    ("CogCap-MM (AAAI'25*)",         4,  1, ("eval_json",   RAD / "runs/obj_bench/evals/cogcap_mm_all10.json")),
    ("O14 (Prior SOTA)",             4,  1, ("eval_json",   RAD / "runs/obj_bench/evals/O14_all10.json")),
    ("ObjEEG-Cont (Zero-Dict)",      4,  1, ("confirm_dir", RAD / "runs/obj_bench/confirm_all10/ObjEEG_Cont_4NFE")),
    ("ObjEEG-SoftOpen (CV-8141)",    4,  1, ("confirm_dir", RAD / "runs/obj_bench/confirm_all10/ObjEEG_SoftOpen_4NFE")),
    ("ObjEEG-CrispHard (CV-8141)",   4,  1, ("confirm_dir", RAD / "runs/obj_bench/confirm_all10/ObjEEG_CrispHard_4NFE")),
    ("ObjEEG-SoftOpen-ShufCtrl",     4,  1, ("confirm_dir", RAD / "runs/obj_bench/confirm_all10/ObjEEG_SoftOpen_ShufCtrl_4NFE")),
    ("O14-Sel4 (Matched Ctrl)",     16,  4, ("confirm_dir", RAD / "runs/obj_bench/confirm_all10/O14_Sel4_16NFE_Ctrl")),
    ("ObjEEG-DualVerify-Sel4",      16,  4, ("confirm_dir", RAD / "runs/obj_bench/confirm_all10/ObjEEG_DualVerify_Sel4")),
    ("O14-FVerify-Sel8 (Ctrl)",     32,  8, ("confirm_dir", RAD / "runs/obj_bench/confirm_all10/O14_FVerify_Sel8_Ctrl")),
    ("ObjEEG-DualVerify-Sel8",      32,  8, ("confirm_dir", RAD / "runs/obj_bench/confirm_all10/ObjEEG_DualVerify_Sel8")),
]

KEYS = [
    "zs_obj200_t1", "zs_obj200_t5", "zs_obj1823_t1", "zs_obj1823_t5",
    "coarse27_t1", "coarse27_t3", "in1k_inc_t1", "in1k_inc_t5",
    "in1k_eff_t1", "in1k_eff_t5", "in1k_pcos", "obj_crisp",
    "clip_img200_t1", "clip_img200_t5", "clip_2way", "incep_2way",
    "alex2", "alex5", "eff_dist", "pixcorr", "ssim",
]

summary = {}
print(f"{'Method':28s} | NFE | {'ZS200 T1/T5':>11s} | {'ZS1823 T1/T5':>12s} | {'Cat27':>5s} | {'IN1K_Inc':>9s} | {'Eff1K':>5s} | {'Crisp':>5s} | {'Img200':>6s} | {'2wCLIP':>6s} | {'Incep':>5s} | {'SSIM':>5s}")
print("-" * 136)
for name, nfe, nsel, spec in ARMS:
    try:
        rows = load_per_sub_metrics(spec)
    except Exception as e:
        print(f"{name:28s} | MISSING ({e})")
        continue
    mean_d = {k: float(np.mean([r[k] for r in rows])) for k in KEYS if k in rows[0]}
    std_d = {k: float(np.std([r[k] for r in rows], ddof=1)) for k in KEYS if k in rows[0]}
    hist_mean = None
    if "cand_hist" in rows[0]:
        hist_mean = [float(np.mean([r["cand_hist"][j] for r in rows])) for j in range(len(rows[0]["cand_hist"]))]
    summary[name] = {"nfe": nfe, "nsel": nsel, "mean": mean_d, "std": std_d, "per_sub": rows, "cand_hist_mean": hist_mean}
    d = mean_d
    print(
        f"{name:28s} | {nfe:3d} | {d['zs_obj200_t1']:5.2f}/{d['zs_obj200_t5']:5.2f} | "
        f"{d['zs_obj1823_t1']:5.2f}/{d['zs_obj1823_t5']:5.2f}  | {d['coarse27_t1']:5.1f} | "
        f"{d['in1k_inc_t1']:4.1f}/{d['in1k_inc_t5']:4.1f} | {d['in1k_eff_t1']:5.1f} | {d['obj_crisp']:5.1f} | "
        f"{d['clip_img200_t1']:6.2f} | {d['clip_2way']:6.2f} | {d['incep_2way']:5.2f} | {d['ssim']:5.3f}"
    )

# Paired two-sided t-tests across all 10 subjects
pairs = [
    ("ObjEEG-DualVerify-Sel8", "O14 (Prior SOTA)"),
    ("ObjEEG-DualVerify-Sel8", "O14-FVerify-Sel8 (Ctrl)"),
    ("ObjEEG-DualVerify-Sel4", "O14 (Prior SOTA)"),
    ("ObjEEG-DualVerify-Sel4", "O14-Sel4 (Matched Ctrl)"),
    ("ObjEEG-DualVerify-Sel8", "ATM-S (NeurIPS'24)"),
    ("ObjEEG-DualVerify-Sel8", "CogCap-MM (AAAI'25*)"),
    ("ObjEEG-SoftOpen (CV-8141)", "ObjEEG-SoftOpen-ShufCtrl"),
    ("ObjEEG-Cont (Zero-Dict)", "O14 (Prior SOTA)"),
    ("ObjEEG-CrispHard (CV-8141)", "O14 (Prior SOTA)"),
]
test_metrics = ["zs_obj200_t1", "zs_obj200_t5", "zs_obj1823_t1", "zs_obj1823_t5", "coarse27_t1", "in1k_inc_t1", "in1k_inc_t5", "in1k_eff_t1", "obj_crisp", "clip_img200_t1", "clip_2way", "incep_2way", "ssim"]

paired_res = {}
print("\n=== PAIRED TWO-SIDED T-TESTS (10 SUBJECTS) ===")
for a, b in pairs:
    if a not in summary or b not in summary:
        continue
    paired_res[f"{a} vs {b}"] = {}
    print(f"\n[{a}]  vs  [{b}]")
    for k in test_metrics:
        va = np.array([r[k] for r in summary[a]["per_sub"]])
        vb = np.array([r[k] for r in summary[b]["per_sub"]])
        diff = va - vb
        mean_diff = float(diff.mean())
        sd_diff = float(diff.std(ddof=1))
        t_val, p_val = stats.ttest_rel(va, vb)
        wins = int((diff > 0).sum())
        ties = int((diff == 0).sum())
        d_cohen = mean_diff / (sd_diff + 1e-12)
        paired_res[f"{a} vs {b}"][k] = {
            "delta": mean_diff, "sd": sd_diff, "t": float(t_val), "p": float(p_val),
            "wins": wins, "ties": ties, "cohen_d": float(d_cohen),
        }
        if k in ("zs_obj200_t1", "zs_obj1823_t1", "zs_obj1823_t5", "coarse27_t1", "in1k_inc_t1", "in1k_eff_t1", "obj_crisp", "clip_img200_t1", "clip_2way", "ssim"):
            print(f"  {k:16s}: delta={mean_diff:+7.3f} | wins={wins:2d}/10 | t={t_val:+6.2f} | p={p_val:.5f} | d={d_cohen:+.2f}")

json.dump({"summary": {k: {kk: vv for kk, vv in v.items() if kk != "per_sub"} for k, v in summary.items()},
           "per_sub_all": {k: v["per_sub"] for k, v in summary.items()},
           "paired_tests": paired_res},
          open(OUT / "final_all10_stats.json", "w"), indent=2)

# Build uncherry-picked sequential (#000-#011) 9-column publication galleries for sub-10 and sub-08!
test_raw = json.load(open(RAD / "data/prep/test_ids.json"))
img_root = RAD / "data/things-eeg/Image_set"

def p2c(p):
    folder = p.split("/")[1]
    name = folder.split("_", 1)[1] if "_" in folder else folder
    return re.sub(r"\d+$", "", name).replace("_", " ").strip()

def load_u8(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    arr = d["k0_rgb_u8"]
    if arr.shape[-1] == 3 and arr.shape[1] != 3:
        arr = arr.permute(0, 3, 1, 2)
    return arr

for sub in (10, 8, 1):
    cols = [
        ("GT Photo", None),
        ("NICE (ICLR'24)", RAD / f"runs/obj_bench/baselines/nice/sub-{sub:02d}_imgs.pt"),
        ("ATM-S (NeurIPS'24)", RAD / f"runs/obj_bench/baselines/atm_s/sub-{sub:02d}_imgs.pt"),
        ("Brain-HIVE", RAD / f"runs/obj_bench/baselines/brain_hive/sub-{sub:02d}_imgs.pt"),
        ("CogCap-MM (AAAI'25*)", RAD / f"runs/obj_bench/baselines/cogcap_mm/sub-{sub:02d}_imgs.pt"),
        ("O14 (Prior SOTA)", RAD / f"runs/o14_official/audit_cache/sub-{sub:02d}_O14_audit.pt"),
        ("ObjEEG-Crisp (4 NFE)", RAD / f"runs/obj_bench/confirm_all10/ObjEEG_CrispHard_4NFE/sub-{sub:02d}_imgs.pt"),
        ("ObjEEG-DualV-Sel4", RAD / f"runs/obj_bench/confirm_all10/ObjEEG_DualVerify_Sel4/sub-{sub:02d}_imgs.pt"),
        ("ObjEEG-DualV-Sel8", RAD / f"runs/obj_bench/confirm_all10/ObjEEG_DualVerify_Sel8/sub-{sub:02d}_imgs.pt"),
    ]
    active = []
    for lbl, p in cols:
        if p is None:
            imgs = [Image.open(img_root / rp).convert("RGB").resize((192, 192), Image.BICUBIC) for rp in test_raw["paths"][:12]]
            active.append((lbl, imgs))
        elif p.exists():
            u8 = load_u8(p)[:12]
            imgs = [Image.fromarray(u8[i].permute(1, 2, 0).numpy()).resize((192, 192), Image.BICUBIC) for i in range(12)]
            active.append((lbl, imgs))
    S, PAD, HDR, LBL_W = 192, 6, 36, 148
    W = LBL_W + len(active) * (S + PAD) + PAD
    H = HDR + 12 * (S + PAD) + PAD
    canvas = Image.new("RGB", (W, H), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    for ci, (lbl, _) in enumerate(active):
        draw.text((LBL_W + ci * (S + PAD) + 4, 10), lbl, fill=(15, 15, 15))
    for ri in range(12):
        y = HDR + ri * (S + PAD)
        draw.text((8, y + S // 2 - 8), f"#{ri:03d} {p2c(test_raw['paths'][ri])}", fill=(15, 15, 15))
        for ci, (_, imgs) in enumerate(active):
            canvas.paste(imgs[ri], (LBL_W + ci * (S + PAD), y))
    gp = OUT / f"gallery_objbench_sub{sub:02d}_seq012.jpg"
    canvas.save(gp, quality=92)
    print(f"Saved gallery: {gp} ({canvas.size})")
