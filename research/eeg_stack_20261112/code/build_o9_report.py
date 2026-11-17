"""Build comprehensive 10-subject & dev ablation report for Stage II O9 (results/o9_streamlined.txt)."""
import json

import numpy as np

from common import PREP, RUNS

KEYS = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "effnet", "clip_top1"]
SOTA = [0.2670, 0.4264, 0.9100, 0.9270, 0.7790, 0.8300]

ARMS = [
    ("Brain-HIVE (repro)", "s{:02d}_t0_s2025_A1"),
    ("ordinary head (per sub)", "s{:02d}_t0_s2025_A1_ord"),
    ("O6 (prior original, 21 ckpts)", "s{:02d}_t0_s2025_A1_ordmw7bag10_lowrktta1_0.5_smap_c0.611_0.5_1.0"),
    ("O7a (w=1.00, indep K=4, 43+ ckpts)", "s{:02d}_w4_O7_w1_notmf"),
    ("O7 (w=1.50, + K=4 tmf)", "s{:02d}_w35_O7_v2b"),
    ("O8 (w=1.00, indep K=4, 92 ckpts)", "s{:02d}_o8_w1_notmf"),
    ("O8b (O8 + K=4 tmf, 92 ckpts)", "s{:02d}_o8_w1_tmf"),
    ("O9-0tta (8 ckpts, 0 TTA, w=1.00)", "s{:02d}_o9_0tta_w1_notmf"),
    ("O9 (8 ckpts, 350 kvTTA, w=1.00)", "s{:02d}_o9_w1_notmf"),
    ("O9b (O9 + K=4 tmf, 8 ckpts, w=1.00)", "s{:02d}_o9_w1_tmf"),
    ("B5a (full-stack sel16, indep K=4)", "s{:02d}_b5_sel16_notmf"),
    ("B5 (full-stack sel16, + K=4 tmf)", "s{:02d}_b5_sel16_tmf"),
]

DEV_ARMS = [
    ("O6  (21 ckpts, old v1)", "s{:02d}_t0_s2025_A1_ordmw7bag10_lowrktta1_0.5_smap_c0.611_0.5_1.0"),
    ("O7a (92 ckpts, 3000+ TTA, kNN)", "s{:02d}_w4_O7_w1_notmf"),
    ("O8  (92 ckpts, 3900 TTA, kNN)", "s{:02d}_o8_w1_notmf"),
    ("S1  (3 enc+ddpm, 0 TTA, v2)", "s{:02d}_o9_S1_uni3_ddpm_v2"),
    ("S2  (3 enc+drd, 0 TTA, v2)", "s{:02d}_o9_S2_uni3_drd_v2"),
    ("S3  (3 enc+drd, 150 TTA, no bag)", "s{:02d}_o9_S3_uni3_tta50_drd_v2"),
    ("S4  (S3 + zero-kNN gauss kalman)", "s{:02d}_o9_S4_zero_knn_gauss_kalman"),
    ("S5  (S3 + zero-kNN gauss lk)", "s{:02d}_o9_S5_zero_knn_gauss_lk"),
    ("R1  (5 enc+L_gn+gn_ordh, 0 TTA)", "s{:02d}_o9_R1_gn5_0tta_v2"),
    ("R2  (R1 + 374-param lead-field)", "s{:02d}_o9_R2_gn5_lf374_v2"),
    ("R3  (R2 + 1-win prec1 start)", "s{:02d}_o9_R3_gn5_lf374_prec1"),
    ("R4  (R3 + 256 proto zero-kNN)", "s{:02d}_o9_R4_gn5_lf374_proto256"),
    ("V1  (7 enc+kvdrd, 0 TTA+bag10)", "s{:02d}_o9_V1_uni7_0tta_kvdrd"),
    ("V2  (7 enc+ddpm, 350 kvTTA -> O9)", "s{:02d}_o9_V2_uni7_kvtta_ddpm"),
    ("V3  (7 enc+kvdrd, 350 kvTTA)", "s{:02d}_o9_V3_uni7_kvtta_kvdrd"),
    ("W1  (v5 dec-hor+ddpmkv, 0 TTA)", "s{:02d}_o9_W1_v5_0tta"),
    ("W2  (v5 dec-hor+ddpmkv, 245 hybTTA)", "s{:02d}_o9_W2_v5_hybtta"),
]


def load_arm_dict(pat, subs):
    arr = {}
    for s in subs:
        run, _, var = pat.format(s).partition("#")
        p = RUNS / "img" / run / "metrics.json"
        if p.exists():
            d = json.loads(p.read_text())
            m = d[var.split(":", 1)[0]][var.split(":", 1)[1]] if var else d["mean"]
            arr[s] = np.array([m[k] for k in KEYS], dtype=float)
    return arr


data = {lbl: load_arm_dict(pat, range(1, 11)) for lbl, pat in ARMS}
data = {k: v for k, v in data.items() if len(v) == 10}

kv_meta_path = PREP / "unet_kv_metric.pt.json"
kv_info = json.loads(kv_meta_path.read_text()) if kv_meta_path.exists() else {}

lines = []
w = lines.append
w("=" * 136)
w("O9-0tta / O9 / O9b STREAMLINED SINGLE-SUBJECT BENCHMARK — 10-SUBJECT EVALUATION & STAGE II ABLATIONS")
w("=" * 136)
w("Dataset : THINGS-EEG2 official test set (200 images x 80 repeats, 17 channels, 0-1000 ms @ 250 Hz, 200 ms RSVP SOA)")
w("Protocol: Strict single-subject training data only, w = 1.00 pure F_train cone restoration, nsel = 1, zero post-hoc blur;")
w("          indep K = 4 draws in O9-0tta / O9; step-1/2 K = 4 Tweedie coupling (--tmf 1,0.10,0.10,0 --tmf_cr 1) in O9b.")
w("Adapter : BoundedConditionAdapter (research/eeg/brainhive_bounded_component_v1.py) imported unchanged in every arm.")
if kv_info:
    w(f"UNet KV : {kv_info.get('n_layers', 70)} IPAttnProcessor2_0 layers (scales={kv_info.get('scales')}, "
      f"eff_rank={kv_info.get('eff_rank', 0.0):.1f}/2048, rel_err={kv_info.get('rel_err', 0.0):.2e}).")
w("")
w("=" * 136)
w("1. ARCHITECTURE & COMPLEXITY COMPARISON")
w("=" * 136)
w(f"{'Arm':<18} {'Enc Ckpts':>10} {'Head':>6} {'Total/Sub':>10} {'TTA Steps':>10} {'Aux Heads':>10}  Architecture Summary")
w("-" * 136)
for row in [
    ("O6", "21", "1", "22", "50", "0", "3-window BrainMLP ensemble + v1 128px start"),
    ("O7a", "42+", "1", "43+", "3,000+", "0", "6-window 42 BrainMLP + 5-win v2 512px start + cone w=1.00"),
    ("O8 / O8b", "90", "2", "92", "3,900", "3 (H14/B32/Spat)", "42 BrainMLP + 36 Flex + 12 Spat512 + OOF simplex blend"),
    ("O9-0tta", "7", "1", "8", "0", "0", "7x KVEBPyrEncoder (TargetSNRGate + CenteredKVLoss) + DDPM head + v2 start"),
    ("O9 / O9b", "7", "1", "8", "350", "0", "O9-0tta + 350-step centered L_kv split-half TTA + bag10 (r_eff=8.0)"),
]:
    w(f"{row[0]:<18} {row[1]:>10} {row[2]:>6} {row[3]:>10} {row[4]:>10} {row[5]:>10}  {row[6]}")

w("")
w("=" * 136)
w("2. 10-SUBJECT SUMMARY (MEAN & BETWEEN-SUBJECT SD OVER sub-01 .. sub-10)")
w("=" * 136)
w(f"{'Arm':<40} {'N':>3}" + "".join(f"{k[:8]:>10}" for k in KEYS) + "   >SOTA")
w(f"{'[Published SOTA max per metric]':<40} {'10':>3}" + "".join(f"{v:>10.4f}" for v in SOTA) + f"{'-':>10}{'-':>10}")
w("-" * 136)
for label, _ in ARMS:
    if label not in data:
        continue
    mat = np.stack([data[label][s] for s in range(1, 11)], axis=0)
    mn, sd = mat.mean(0), mat.std(0, ddof=1)
    beats = sum(int(mn[i] > SOTA[i]) for i in range(6))
    w(f"{label:<40} {'10':>3}" + "".join(f"{v:>10.4f}" for v in mn) + f"   [{beats}/6]")
    w(f"{'  (10-subj SD)':<40} {'':>3}" + "".join(f"{v:>10.4f}" for v in sd))


def fmt_pair(title, subs_list, arm_b, arm_a):
    if arm_a not in data or arm_b not in data:
        return
    a = np.stack([data[arm_a][s] for s in subs_list], 0)
    b = np.stack([data[arm_b][s] for s in subs_list], 0)
    d = b - a
    dm = d.mean(0)
    n = len(subs_list)
    wins = [int((d[:, i] < 0).sum() if KEYS[i] == "effnet" else (d[:, i] > 0).sum()) for i in range(len(KEYS))]
    tvals = dm / (d.std(0, ddof=1) / (n ** 0.5) + 1e-12)
    w(f"\nPaired ({n} subs {title}): {arm_b}  vs  {arm_a}")
    w("  delta : " + " ".join(f"{KEYS[i][:7]}={dm[i]:+.4f}" for i in range(len(KEYS))))
    w("  wins  : " + " ".join(f"{KEYS[i][:7]}={wins[i]}/{n}" for i in range(len(KEYS))))
    w("  t-stat: " + " ".join(f"{KEYS[i][:7]}={tvals[i]:+.2f}" for i in range(len(KEYS))))


S4, S6, S10 = [2, 5, 8, 9], [1, 3, 4, 6, 7, 10], list(range(1, 11))
w("")
w("=" * 136)
w("3. PRIOR SCREENED 4 SUBJECTS {2,5,8,9} VS UNTOUCHED HELD-OUT 6 SUBJECTS {1,3,4,6,7,10}")
w("=" * 136)
for arm_b, arm_a in [
    ("O9 (8 ckpts, 350 kvTTA, w=1.00)", "O6 (prior original, 21 ckpts)"),
    ("O9 (8 ckpts, 350 kvTTA, w=1.00)", "O7a (w=1.00, indep K=4, 43+ ckpts)"),
    ("O9 (8 ckpts, 350 kvTTA, w=1.00)", "O8 (w=1.00, indep K=4, 92 ckpts)"),
    ("O9b (O9 + K=4 tmf, 8 ckpts, w=1.00)", "O8b (O8 + K=4 tmf, 92 ckpts)"),
    ("O9 (8 ckpts, 350 kvTTA, w=1.00)", "O9-0tta (8 ckpts, 0 TTA, w=1.00)"),
]:
    fmt_pair("[screened: 2,5,8,9]", S4, arm_b, arm_a)
    fmt_pair("[held-out: 1,3,4,6,7,10]", S6, arm_b, arm_a)

w("")
w("=" * 136)
w("4. 10-SUBJECT PAIRED COMPARISONS & COMPONENT ISOLATION")
w("=" * 136)
for arm_b, arm_a in [
    ("O9 (8 ckpts, 350 kvTTA, w=1.00)", "O6 (prior original, 21 ckpts)"),
    ("O9 (8 ckpts, 350 kvTTA, w=1.00)", "O7a (w=1.00, indep K=4, 43+ ckpts)"),
    ("O9 (8 ckpts, 350 kvTTA, w=1.00)", "O8 (w=1.00, indep K=4, 92 ckpts)"),
    ("O9b (O9 + K=4 tmf, 8 ckpts, w=1.00)", "O8b (O8 + K=4 tmf, 92 ckpts)"),
    ("O9-0tta (8 ckpts, 0 TTA, w=1.00)", "O7a (w=1.00, indep K=4, 43+ ckpts)"),
    ("O9 (8 ckpts, 350 kvTTA, w=1.00)", "O9-0tta (8 ckpts, 0 TTA, w=1.00)"),
    ("O9b (O9 + K=4 tmf, 8 ckpts, w=1.00)", "O9 (8 ckpts, 350 kvTTA, w=1.00)"),
]:
    fmt_pair("[all 10 subjects]", S10, arm_b, arm_a)

w("")
w("=" * 136)
w("5. PER-SUBJECT BREAKDOWN (sub-01 .. sub-10)")
w("=" * 136)
for label in [
    "O8 (w=1.00, indep K=4, 92 ckpts)",
    "O9-0tta (8 ckpts, 0 TTA, w=1.00)",
    "O9 (8 ckpts, 350 kvTTA, w=1.00)",
    "O9b (O9 + K=4 tmf, 8 ckpts, w=1.00)",
]:
    if label not in data:
        continue
    w(f"\n--- {label} ---")
    w("Sub     " + "".join(f"{k[:8]:>10}" for k in KEYS))
    for s in range(1, 11):
        w(f"sub-{s:02d}  " + "".join(f"{v:>10.4f}" for v in data[label][s]))
    mat = np.stack([data[label][s] for s in range(1, 11)], 0)
    w("MEAN    " + "".join(f"{v:>10.4f}" for v in mat.mean(0)))
    w("SD      " + "".join(f"{v:>10.4f}" for v in mat.std(0, ddof=1)))

w("")
w("=" * 136)
w("6. STAGE II DEV {2,5,8,9} CONTROLLED ABLATIONS & NEGATIVE RESULTS (w=1.00, indep K=4, nsel=1, ZERO BLUR)")
w("=" * 136)
w(f"{'Dev Arm':<36}" + "".join(f"{k[:8]:>10}" for k in KEYS))
w("-" * 136)
for label, pat in DEV_ARMS:
    d_sub = load_arm_dict(pat, S4)
    if len(d_sub) == len(S4):
        mn = np.stack([d_sub[s] for s in S4], 0).mean(0)
        w(f"{label:<36}" + "".join(f"{v:>10.4f}" for v in mn))

# 5-fold OOF metrics observed from runs/enc/o9_uni (dev {2,5,8,9}) and all 10 subs
oof_lines = []
for etag, name_lbl in (("o9_uni", "o9_uni KVEBPyr+CenteredKV (all 10, seed 2025)"), ("o9_v5", "o9_v5 DecoupledHor+TangentKV (dev 4, seed 2025)")):
    recs = []
    for s in (range(1, 11) if etag == "o9_uni" else S4):
        for fo in range(5):
            mp = RUNS / "enc" / etag / f"sub-{s:02d}" / f"fold{fo}_seed2025/metrics.json"
            if mp.exists():
                recs.append(json.loads(mp.read_text()))
    if recs and "oof_perp_200w" in recs[0]:
        oof_lines.append(
            f"  {name_lbl:<48} | n={len(recs)} folds | params={recs[0].get('params', 0):,} | "
            f"Perp_200w={np.mean([r['oof_perp_200w'] for r in recs]):.4f} | KVc_200w={np.mean([r['kv_c_200w'] for r in recs]):.4f}"
        )
if oof_lines:
    w("\n5-Fold Concept-Disjoint Encoder OOF Summary:")
    w("\n".join(oof_lines))

report = "\n".join(lines) + "\n"
out_p = RUNS / "o9_streamlined.txt"
out_p.write_text(report)
print(report, flush=True)
