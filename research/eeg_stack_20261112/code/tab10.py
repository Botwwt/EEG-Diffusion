"""Ten-subject table: python tab10.py [seed]  ->  baseline vs recipe per subject and averaged, next to the paper's Table 2."""
import json
import sys

import numpy as np

from common import RUNS

seed = sys.argv[1] if len(sys.argv) > 1 else "2025"
M = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "effnet"]
ARMS = {"baseline (reproduced)": "s{:02d}_t0_s" + seed + "_A1",
        "ordinary head (co-author)": "s{:02d}_t0_s" + seed + "_A1_ord",
        "ordinary head + low-level start": "s{:02d}_t0_s" + seed + "_A1_ord_low",
        "ordinary head + low-level + best of 4": "s{:02d}_t0_s" + seed + "_A1_ord_low_sel4",
        "ordinary head + low-level + amp 2": "s{:02d}_t0_s" + seed + "_A1_ord_low_amp2",
        "joint enc: ordinary head": "s{:02d}_ms_e10_ft10in_A1_ord",
        "joint enc: head + low-level": "s{:02d}_ms_e10_ft10in_A1_ord_low",
        "joint enc: head + low-level + amp 2": "s{:02d}_ms_e10_ft10in_A1_ord_low_amp2",
        "joint enc: head + low-level + amp 1.5": "s{:02d}_ms_e10_ft10in_A1_ord_low_amp1.5",
        "joint enc: head + low-level + best of 4": "s{:02d}_ms_e10_ft10in_A1_ord_low_sel4",
        "joint enc: head + low + amp 2 + best of 4": "s{:02d}_ms_e10_ft10in_A1_ord_low_amp2_sel4",
        "joint enc: SHARED head": "s{:02d}_ms_e10_ft10in_A1_pooled_p1",
        "joint enc: shared head + low-level": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_low",
        "joint enc: shared head + low + best of 4": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_low_sel4",
        "joint enc: shared head + low + amp 2 + best of 4": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_low_amp2_sel4",
        "low-level point: strength 0.5 + best of 4": "s{:02d}_ms_e10_ft10in_A1_ord_low_st0.5_sel4",
        "A': shared head + ridge+retrieval start + best of 4": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_sel4",
        "B': ... + amp 2": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_amp2_sel4",
        "calibration + low-level start": "s{:02d}_t0_s" + seed + "_A1_cal0.5_low",
        "+ best of 4 candidates": "s{:02d}_t0_s" + seed + "_A1_cal0.5_low_sel4"}
PAPER = [0.195, 0.336, 0.843, 0.905, 0.756, 0.808]   # Brain-HIVE Table 2, EEG average (SwAV 0.554 not measured here)
rows = {}
for arm, pat in ARMS.items():
    for s in range(1, 11):
        f = RUNS / "img" / pat.format(s) / "metrics.json"
        if f.exists():
            rows[arm, s] = [json.loads(f.read_text())["mean"][k] for k in M]
print(f"{'':<53}" + "".join(f"{k:>10}" for k in M))
for s in range(1, 11):
    for arm in ARMS:
        if (arm, s) in rows:
            print(f"sub-{s:02d} {arm:<40}" + "".join(f"{v:>10.4f}" for v in rows[arm, s]))
def compare(base, arms):
    for arm in arms:
        both = [s for s in range(1, 11) if (base, s) in rows and (arm, s) in rows]
        if not both:
            continue
        a, b = np.array([rows[base, s] for s in both]), np.array([rows[arm, s] for s in both])
        print(f"\n{arm} vs {base}, mean over {len(both)} subjects {both}")
        print(f"{base:<48}" + "".join(f"{x:>10.4f}" for x in a.mean(0)))
        print(f"{arm:<48}" + "".join(f"{x:>10.4f}" for x in b.mean(0)))
        d = b - a
        print(f"{'difference (mean)':<48}" + "".join(f"{x:>+10.4f}" for x in d.mean(0)))
        print(f"{'subjects improved':<48}" + "".join(f"{int((d[:, i] < 0).sum() if k == 'effnet' else (d[:, i] > 0).sum()):>7}/{len(both)}" for i, k in enumerate(M)))


compare("baseline (reproduced)", list(ARMS)[1:])
print("\n===== gain of the additions over the co-author's head alone =====")
compare("ordinary head (co-author)", [a for a in ARMS if a.startswith("ordinary head +")])
print("\n===== joint encoder vs per-subject encoder, same arm =====")
compare("ordinary head (co-author)", ["joint enc: ordinary head"])
compare("ordinary head + low-level start", ["joint enc: head + low-level"])
compare("ordinary head + low-level + amp 2", ["joint enc: head + low-level + amp 2"])
compare("joint enc: head + low-level", ["joint enc: head + low-level + amp 1.5", "joint enc: head + low-level + amp 2", "joint enc: head + low-level + best of 4", "joint enc: head + low + amp 2 + best of 4"])
print("\n===== one shared head vs one head per subject (joint encoder) =====")
for x, y in [("joint enc: ordinary head", "joint enc: SHARED head"), ("joint enc: head + low-level", "joint enc: shared head + low-level"),
             ("joint enc: head + low-level + best of 4", "joint enc: shared head + low + best of 4"),
             ("joint enc: head + low + amp 2 + best of 4", "joint enc: shared head + low + amp 2 + best of 4")]:
    compare(x, [y])
print("\n===== start image: ridge vs ridge + retrieved training latents (weight 0.5) =====")
compare("joint enc: shared head + low + best of 4", ["A': shared head + ridge+retrieval start + best of 4"])
compare("joint enc: shared head + low + amp 2 + best of 4", ["B': ... + amp 2"])
print(f"\n{'paper Table 2 (10-subject mean)':<48}" + "".join(f"{x:>10.4f}" for x in PAPER))
