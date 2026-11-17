"""Build comprehensive 10-subject report for O8 / O8b and B5-1a / B5-1 / B5a / B5 (o8_b5_stack.txt)."""
import json

import numpy as np

from common import RUNS

KEYS = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "effnet", "clip_top1"]
SOTA = [0.2670, 0.4264, 0.9100, 0.9270, 0.7790, 0.8300]

ARMS = [
    ("O6 (prior original)", "s{:02d}_t0_s2025_A1_ordmw7bag10_lowrktta1_0.5_smap_c0.611_0.5_1.0"),
    ("O7a (w=1.00, indep K=4, strict)", "s{:02d}_w4_O7_w1_notmf"),
    ("O7 (w=1.50, + K=4 tmf)", "s{:02d}_w35_O7_v2b"),
    ("O8#w5start (R=4-gated start, not adopted)", "s{:02d}_w5_O8_w1_notmf"),
    ("O8 (w=1.00, indep K=4, strict NEW)", "s{:02d}_o8_w1_notmf"),
    ("O8b (w=1.00, + K=4 tmf, NEW)", "s{:02d}_o8_w1_tmf"),
    ("B''' (prior full-stack)", "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_amp2_sel16#post:0,24,0.8"),
    ("B4-1a (indep K=4, no tmf)", "s{:02d}_w4_B4_n1_notmf"),
    ("B4-1 (+ K=4 tmf)", "s{:02d}_w35_B4_v2b_n1"),
    ("B5-1a#w5start (R=4-gated start, not adopted)", "s{:02d}_w5_B5_n1_notmf"),
    ("B5-1a (indep K=4, no tmf, NEW)", "s{:02d}_b5_n1_notmf"),
    ("B5-1 (+ K=4 tmf, NEW)", "s{:02d}_b5_n1_tmf"),
    ("B4a (best-of-16, indep K=4)", "s{:02d}_w4_B4_sel16_notmf"),
    ("B4 (best-of-16, + K=4 tmf)", "s{:02d}_w35_B4_v2b_sel16"),
    ("B5a (best-of-16, indep K=4, NEW)", "s{:02d}_b5_sel16_notmf"),
    ("B5 (best-of-16, + K=4 tmf, NEW)", "s{:02d}_b5_sel16_tmf"),
]

data = {}
for label, pat in ARMS:
    arr = {}
    for s in range(1, 11):
        run, _, var = pat.format(s).partition("#")
        p = RUNS / "img" / run / "metrics.json"
        if p.exists():
            d = json.loads(p.read_text())
            if var:
                kd, sp = var.split(":", 1)
                m = d[kd][sp]
            else:
                m = d["mean"]
            arr[s] = np.array([m[k] for k in KEYS], dtype=float)
    if len(arr) == 10:
        data[label] = arr

oof_w = {}
oof_p = RUNS / "cond/o8_b5_oof_weights.json"
if oof_p.exists():
    oof_w = json.loads(oof_p.read_text())

lines = []
w = lines.append
w("=" * 136)
w("O8 / O8b (SINGLE-SUBJECT) AND B5-1a / B5-1 / B5a / B5 (FULL-STACK) — 10-SUBJECT EVALUATION (2026-11-16)")
w("=" * 136)
w("Dataset : THINGS-EEG2 official test set (Gifford et al. 2022), 10 subjects (sub-01 .. sub-10), 200 test images x 80 repeats")
w("EEG     : 17 occipito-parietal channels, 0-1000 ms post-stimulus at 250 Hz (4 ms/sample; ATM -200..1000 ms epoch, first 50 dropped);")
w("          THINGS-EEG2 RSVP SOA = 200 ms (100 ms image presentation + 100 ms blank)")
w("Protocol: K = 4 samples/query, unchanged metrics.py (Brain-HIVE / MindEye evaluation), zero post-hoc Gaussian blur (--post \"\")")
w("Adapter : BoundedConditionAdapter (research/eeg/brainhive_bounded_component_v1.py) imported unchanged in every arm")
w(f"5-Fold OOF weights (selected strictly on 16,540 training images, zero test-set tuning): {oof_w}")
w("Methodological boundaries & unchanged O7a/B4 start-latent + sampler controls:")
w("  - O8  (sXX_o8_w1_notmf)   : strict unborrowed single-subject headline — pure F_train cone restoration (w = 1.00, zero")
w("                              directional amplification), strictly independent K = 4 draws (no --tmf), nsel = 1, single-subject")
w("                              FlexBrainEncoder + ord_o8_flex (5-fold OOF alpha_o8 = 0.50), exact O7a v2 start-latent & sampler flags.")
w("  - O8b (sXX_o8_w1_tmf)     : O8 + step-1/2 K = 4 Tweedie coupling (--tmf 1,0.10,0.10,0 --tmf_cr 1; w = 1.00 unchanged).")
w("  - B5-1a (sXX_b5_n1_notmf) : full-stack nsel = 1, strictly independent K = 4 draws (no --tmf), exact B4-1a v2 start & sampler flags.")
w("  - B5-1  (sXX_b5_n1_tmf)   : full-stack nsel = 1 + step-1/2 K = 4 Tweedie coupling (--tmf 1,0.12,0.12,0 --tmf_cr 1).")
w("  - B5a (sXX_b5_sel16_notmf): full-stack best-of-16, strictly independent K = 4 draws (no --tmf), exact B4a scoring weights.")
w("  - B5  (sXX_b5_sel16_tmf)  : full-stack best-of-16 + step-1/2 K = 4 Tweedie coupling, exact B4 scoring weights, zero blur.")
w("  - O8#w5start / B5-1a#w5start (not adopted): negative start-latent ablation calibrating linear r_4^1.0 PCA gating & spat_ms512")
w("                              lp4_rms scaling on R=4 training repeats, which under-weights mid-frequency modes at R=80 test repeats.")
w("")
w("=" * 136)
w("1. 10-SUBJECT SUMMARY (MEAN & BETWEEN-SUBJECT SD OVER sub-01 .. sub-10)")
w("=" * 136)
w(f"{'Arm':<42} {'N':>3}" + "".join(f"{k[:8]:>10}" for k in KEYS) + "   >SOTA")
w(f"{'[Published SOTA max per metric]':<42} {'10':>3}" + "".join(f"{v:>10.4f}" for v in SOTA) + f"{'-':>10}{'-':>10}")
w("-" * 136)

for label, _ in ARMS:
    if label not in data:
        continue
    mat = np.stack([data[label][s] for s in range(1, 11)], axis=0)
    mn = mat.mean(0)
    sd = mat.std(0, ddof=1)
    beats = sum(int(mn[i] > SOTA[i]) for i in range(6))
    w(f"{label:<42} {'10':>3}" + "".join(f"{v:>10.4f}" for v in mn) + f"   [{beats}/6]")
    w(f"{'  (10-subj SD)':<42} {'':>3}" + "".join(f"{v:>10.4f}" for v in sd))


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


w("")
w("=" * 136)
w("2. PRIOR SCREENED 4 SUBJECTS {2,5,8,9} VS UNTOUCHED HELD-OUT 6 SUBJECTS {1,3,4,6,7,10}")
w("=" * 136)
S4 = [2, 5, 8, 9]
S6 = [1, 3, 4, 6, 7, 10]
S10 = list(range(1, 11))

for arm_new, arm_prev in [
    ("O8 (w=1.00, indep K=4, strict NEW)", "O7a (w=1.00, indep K=4, strict)"),
    ("O8 (w=1.00, indep K=4, strict NEW)", "O6 (prior original)"),
    ("B5-1a (indep K=4, no tmf, NEW)", "B4-1a (indep K=4, no tmf)"),
    ("B5a (best-of-16, indep K=4, NEW)", "B4a (best-of-16, indep K=4)"),
    ("B5 (best-of-16, + K=4 tmf, NEW)", "B4 (best-of-16, + K=4 tmf)"),
]:
    fmt_pair("[screened: 2,5,8,9]", S4, arm_new, arm_prev)
    fmt_pair("[held-out: 1,3,4,6,7,10]", S6, arm_new, arm_prev)

w("")
w("=" * 136)
w("3. 10-SUBJECT PAIRED COMPARISONS & COMPONENT ISOLATION")
w("=" * 136)
for arm_b, arm_a in [
    ("O8 (w=1.00, indep K=4, strict NEW)", "O6 (prior original)"),
    ("O8 (w=1.00, indep K=4, strict NEW)", "O7a (w=1.00, indep K=4, strict)"),
    ("O8 (w=1.00, indep K=4, strict NEW)", "O7 (w=1.50, + K=4 tmf)"),
    ("O8b (w=1.00, + K=4 tmf, NEW)", "O8 (w=1.00, indep K=4, strict NEW)"),
    ("O8 (w=1.00, indep K=4, strict NEW)", "O8#w5start (R=4-gated start, not adopted)"),
    ("B5-1a (indep K=4, no tmf, NEW)", "B4-1a (indep K=4, no tmf)"),
    ("B5-1 (+ K=4 tmf, NEW)", "B4-1 (+ K=4 tmf)"),
    ("B5a (best-of-16, indep K=4, NEW)", "B4a (best-of-16, indep K=4)"),
    ("B5a (best-of-16, indep K=4, NEW)", "B''' (prior full-stack)"),
    ("B5 (best-of-16, + K=4 tmf, NEW)", "B4 (best-of-16, + K=4 tmf)"),
    ("B5 (best-of-16, + K=4 tmf, NEW)", "B5a (best-of-16, indep K=4, NEW)"),
]:
    fmt_pair("[all 10 subjects]", S10, arm_b, arm_a)

w("")
w("=" * 136)
w("4. PER-SUBJECT BREAKDOWN (sub-01 .. sub-10)")
w("=" * 136)
for label in [
    "O8 (w=1.00, indep K=4, strict NEW)",
    "O8b (w=1.00, + K=4 tmf, NEW)",
    "B5-1a (indep K=4, no tmf, NEW)",
    "B5-1 (+ K=4 tmf, NEW)",
    "B5a (best-of-16, indep K=4, NEW)",
    "B5 (best-of-16, + K=4 tmf, NEW)",
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

print("\n".join(lines))
