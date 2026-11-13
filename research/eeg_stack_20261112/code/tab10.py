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
        "A' best of 8": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_sel8sv",
        "B' best of 8": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_amp2_sel8sv",
        "A'': A' best of 8 + HF shrink (16 px, 0.8)": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_sel8sv#post:0,16,0.8",
        "B'': B' best of 8 + HF shrink (16 px, 0.8)": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_amp2_sel8sv#post:0,16,0.8",
        "A' best of 8, target = corrected condition": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_sel8sv#selvar:head@8",
        "B' best of 8, target = corrected condition": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_amp2_sel8sv#selvar:head@8",
        "A' best of 8, target = kNN training prototype": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_sel8sv#selvar:knn@8",
        "B' best of 16": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_amp2_sel16",
        "B‴: B' best of 16 + HF shrink (24 px, 0.8)": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_amp2_sel16#post:0,24,0.8",
        "B' best of 16 + HF shrink (16 px, 0.8)": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_amp2_sel16#post:0,16,0.8",
        "B' best of 16, target = corrected condition": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_amp2_sel16#selvar:head@16",
        "A' best of 8, candidates alternating amp 1 / 2": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_ampmix12_sel8",
        "A' retrieval weight 0.25": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.25_sel4",
        "A' single draw": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5",
        "A' single draw, 8 steps strength 0.75": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_8st0.75",
        "A' single draw, 8 steps strength 0.625": "s{:02d}_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_8st0.625",
        "A with the 3-pass shared head": "s{:02d}_ms_e10_ft10in_A1_pooled_p3_low_sel4",
        "B with the 3-pass shared head": "s{:02d}_ms_e10_ft10in_A1_pooled_p3_low_amp2_sel4",
        "A' with the 3-pass shared head": "s{:02d}_ms_e10_ft10in_A1_pooled_p3_lowrk0.5_sel4",
        "B' with the 3-pass shared head": "s{:02d}_ms_e10_ft10in_A1_pooled_p3_lowrk0.5_amp2_sel4",
        "O1: per-subject enc + head + retrieval-blended start": "s{:02d}_t0_s2025_A1_ord_lowrkt0_0.5",
        "O2: O1 + inverse-variance repeat aggregation": "s{:02d}_t0_s2025_A1_ordivw_lowrkivw_0.5",
        "O3: O2 + fractional-strength re-injection": "s{:02d}_t0_s2025_A1_ordivw_lowrkivw_0.5_smap_c0.611_0.5_1.0",
        "O3 with the strength set by subject reliability": "s{:02d}_t0_s2025_A1_ordivw_lowrkivw_0.5_smap_const_0.5_1.0",
        "O4: O3 + test-time-adapted start image": "s{:02d}_t0_s2025_A1_ordivw_lowrktta1_0.5_smap_c0.611_0.5_1.0",
        "O3 + TTA for start and condition": "s{:02d}_t0_s2025_A1_ordtta1_lowrktta1_0.5_smap_c0.611_0.5_1.0",
        "O5: O4 + 7-seed ensemble with repeat bagging for the condition": "s{:02d}_t0_s2025_A1_ordens7bag10_lowrktta1_0.5_smap_c0.611_0.5_1.0",
        "O4 + adapted 7-seed ensemble condition": "s{:02d}_t0_s2025_A1_ordens7tta1_lowrktta1_0.5_smap_c0.611_0.5_1.0",
        "O4 + adapted 7-seed ensemble condition, PCA 512": "s{:02d}_t0_s2025_A1_ordens7tta1_p512_lowrktta1_0.5_smap_c0.611_0.5_1.0",
        "O4 with retrieval weight 0.75": "s{:02d}_t0_s2025_A1_ordivw_lowrktta1_0.75_smap_c0.611_0.5_1.0",
        "O4 with retrieval weight 1.0": "s{:02d}_t0_s2025_A1_ordivw_lowrktta1_1.0_smap_c0.611_0.5_1.0",
        "O5 with 20 repeat groups": "s{:02d}_t0_s2025_A1_ordens7bag20_lowrktta1_0.5_smap_c0.611_0.5_1.0",
        "O5 with 5 repeat groups": "s{:02d}_t0_s2025_A1_ordens7bag5_lowrktta1_0.5_smap_c0.611_0.5_1.0",
        "O5 without repeat bagging (7-seed ensemble)": "s{:02d}_t0_s2025_A1_ordens7_lowrktta1_0.5_smap_c0.611_0.5_1.0",
        "O5 with the start retrieved by the adapted 7-seed ensemble": "s{:02d}_t0_s2025_A1_ordens7bag10_lowrkens7tta1_0.5_smap_c0.611_0.5_1.0",
        "O4 with the adapted condition, PCA 256": "s{:02d}_t0_s2025_A1_ordtta1_p256_lowrktta1_0.5_smap_c0.611_0.5_1.0",
        "O4 with the condition projected on PCA 256": "s{:02d}_t0_s2025_A1_ordbase_p256_lowrktta1_0.5_smap_c0.611_0.5_1.0",
        "O3 + IP weight schedule 1.25 / 1.25 / 1.25": "s{:02d}_t0_s2025_A1_ordivw_lowrkivw_0.5_smap_c0.611_0.5_1.0_ips1_1.25_1.25_1.25",
        "O3 + IP weight schedule 1 / 1.25 / 1.25": "s{:02d}_t0_s2025_A1_ordivw_lowrkivw_0.5_smap_c0.611_0.5_1.0_ips1_1_1.25_1.25",
        "O3 + IP weight schedule 1.25 / 1.25 / 1": "s{:02d}_t0_s2025_A1_ordivw_lowrkivw_0.5_smap_c0.611_0.5_1.0_ips1_1.25_1.25_1",
        "O3 + IP weight schedule 1.5 / 1 / 1": "s{:02d}_t0_s2025_A1_ordivw_lowrkivw_0.5_smap_c0.611_0.5_1.0_ips1_1.5_1_1",
        "O3 + IP weight schedule 1 / 1 / 1.5": "s{:02d}_t0_s2025_A1_ordivw_lowrkivw_0.5_smap_c0.611_0.5_1.0_ips1_1_1_1.5",
        "O3 + IP weight schedule 0.75 / 1 / 1.25": "s{:02d}_t0_s2025_A1_ordivw_lowrkivw_0.5_smap_c0.611_0.5_1.0_ips1_0.75_1_1.25",
        "O1 + re-injection (no aggregation)": "s{:02d}_t0_s2025_A1_ord_lowrkt0_0.5_smap_const_0.5_1.0",
        "O1 + strength map from reliability (0.5-0.75)": "s{:02d}_t0_s2025_A1_ord_lowrkt0_0.5_smap_map_0.5_0.75",
        "O1 + shuffled reliability map (0.5-0.75)": "s{:02d}_t0_s2025_A1_ord_lowrkt0_0.5_smap_shuf_0.5_0.75",
        "O1 + constant strength (0.5-0.75)": "s{:02d}_t0_s2025_A1_ord_lowrkt0_0.5_smap_const_0.5_0.75",
        "O1 + per-image reliability strength (0.5-0.75)": "s{:02d}_t0_s2025_A1_ord_lowrkt0_0.5_smap_img_0.5_0.75",
        "O1 + strength map (0.625-0.75)": "s{:02d}_t0_s2025_A1_ord_lowrkt0_0.5_smap_map_0.625_0.75",
        "O1 + constant strength (0.625-0.75)": "s{:02d}_t0_s2025_A1_ord_lowrkt0_0.5_smap_const_0.625_0.75",
        "O1 + strength map (0.5-1.0)": "s{:02d}_t0_s2025_A1_ord_lowrkt0_0.5_smap_map_0.5_1.0",
        "O1 + ILVR f4 w1, strength 1": "s{:02d}_t0_s2025_A1_ord_lowrkt0_0.5_ilvr4_1.0_0_st1.0",
        "O1 + ILVR f4 w1, strength 0.75": "s{:02d}_t0_s2025_A1_ord_lowrkt0_0.5_ilvr4_1.0_0_st0.75",
        "O1 + ILVR f4 w0.5, strength 0.75": "s{:02d}_t0_s2025_A1_ord_lowrkt0_0.5_ilvr4_0.5_0_st0.75",
        "O1 + ILVR f4 w0.5, strength 1": "s{:02d}_t0_s2025_A1_ord_lowrkt0_0.5_ilvr4_0.5_0_st1.0",
        "O1 + ILVR f8 w1, strength 0.75": "s{:02d}_t0_s2025_A1_ord_lowrkt0_0.5_ilvr8_1.0_0_st0.75",
        "O1 + ILVR f4 w1 first step only, strength 1": "s{:02d}_t0_s2025_A1_ord_lowrkt0_0.5_ilvr4_1.0_1_st1.0",
        "calibration + low-level start": "s{:02d}_t0_s" + seed + "_A1_cal0.5_low",
        "+ best of 4 candidates": "s{:02d}_t0_s" + seed + "_A1_cal0.5_low_sel4"}
PAPER = [0.195, 0.336, 0.843, 0.905, 0.756, 0.808]   # Brain-HIVE Table 2, EEG average (SwAV 0.554 not measured here)
rows = {}
for arm, pat in ARMS.items():
    for s in range(1, 11):
        run, _, var = pat.format(s).partition("#")   # '#post:<spec>' / '#selvar:<spec>': a variant scored inside that run
        f = RUNS / "img" / run / "metrics.json"
        if f.exists():
            d = json.loads(f.read_text())
            if var:
                kind, spec = var.split(":", 1)
                if spec not in d.get(kind, {}):
                    continue
                d = {"mean": d[kind][spec]}
            rows[arm, s] = [d["mean"][k] for k in M]
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
print("\n===== best of 8, selection target, high-frequency shrinkage =====")
compare("A': shared head + ridge+retrieval start + best of 4", ["A' best of 8"])
compare("B': ... + amp 2", ["B' best of 8"])
compare("A' best of 8", ["A' best of 8, target = corrected condition", "A' best of 8, target = kNN training prototype", "A'': A' best of 8 + HF shrink (16 px, 0.8)"])
compare("B' best of 8", ["B' best of 8, target = corrected condition", "B'': B' best of 8 + HF shrink (16 px, 0.8)"])
compare("A'': A' best of 8 + HF shrink (16 px, 0.8)", ["B'': B' best of 8 + HF shrink (16 px, 0.8)"])
compare("B' best of 8", ["B' best of 16"])
compare("B' best of 16", ["B' best of 16, target = corrected condition", "B' best of 16 + HF shrink (16 px, 0.8)", "B‴: B' best of 16 + HF shrink (24 px, 0.8)"])
compare("A' best of 8", ["A' best of 8, candidates alternating amp 1 / 2"])
compare("B' best of 8", ["A' best of 8, candidates alternating amp 1 / 2"])
print("\n===== retrieval weight, sampler steps, 3-pass shared head =====")
compare("joint enc: shared head + low + best of 4", ["A' retrieval weight 0.25"])
compare("A' single draw", ["A' single draw, 8 steps strength 0.75", "A' single draw, 8 steps strength 0.625"])
for x, y in [("joint enc: shared head + low + best of 4", "A with the 3-pass shared head"), ("joint enc: shared head + low + amp 2 + best of 4", "B with the 3-pass shared head"),
             ("A': shared head + ridge+retrieval start + best of 4", "A' with the 3-pass shared head"), ("B': ... + amp 2", "B' with the 3-pass shared head")]:
    compare(x, [y])
print("\n===== original-components pipeline (no joint encoder, no best-of-N, no shrinkage, no amplification) =====")
compare("ordinary head + low-level start", ["O1: per-subject enc + head + retrieval-blended start"])
compare("O1: per-subject enc + head + retrieval-blended start", ["O2: O1 + inverse-variance repeat aggregation", "O1 + re-injection (no aggregation)"])
compare("O2: O1 + inverse-variance repeat aggregation", ["O3: O2 + fractional-strength re-injection", "O3 with the strength set by subject reliability"])
compare("O3: O2 + fractional-strength re-injection", ["O4: O3 + test-time-adapted start image", "O3 + TTA for start and condition"])
compare("O4: O3 + test-time-adapted start image", ["O5: O4 + 7-seed ensemble with repeat bagging for the condition", "O4 + adapted 7-seed ensemble condition", "O4 + adapted 7-seed ensemble condition, PCA 512", "O4 with retrieval weight 0.75", "O4 with retrieval weight 1.0"])
compare("O5: O4 + 7-seed ensemble with repeat bagging for the condition", ["O5 with 20 repeat groups", "O5 with 5 repeat groups", "O5 without repeat bagging (7-seed ensemble)", "O5 with the start retrieved by the adapted 7-seed ensemble", "O4 with the adapted condition, PCA 256", "O4 with the condition projected on PCA 256"])
print("\n===== IP-adapter weight schedules on O3 (screening, subjects 2, 5, 8, 9) =====")
compare("O3: O2 + fractional-strength re-injection", [a for a in ARMS if a.startswith("O3 + IP weight")])
print("\n===== reliability-driven strength maps and ILVR (screening, subjects 2, 5, 8, 9 / 2, 8) =====")
compare("O1: per-subject enc + head + retrieval-blended start", [a for a in ARMS if a.startswith("O1 + ") and a != "O1 + re-injection (no aggregation)"])
compare("O1 + shuffled reliability map (0.5-0.75)", ["O1 + strength map from reliability (0.5-0.75)"])
compare("O1 + constant strength (0.625-0.75)", ["O1 + strength map (0.625-0.75)"])
print(f"\n{'paper Table 2 (10-subject mean)':<48}" + "".join(f"{x:>10.4f}" for x in PAPER))
