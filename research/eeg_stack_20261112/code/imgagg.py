"""Tabulate image-level results: python imgagg.py -> stdout table + runs/img/summary.csv

One row per generation run (mean over the K samples per query). Rows that differ only by encoder seed are also
averaged into "<name without seed> (mean of n seeds)" with the SD over seeds.
"""
import csv
import json
import re
from collections import defaultdict

import numpy as np

from common import RUNS

M = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "effnet", "clip_top1"]
rows = []
for f in sorted((RUNS / "img").glob("*/metrics.json")):
    d = json.loads(f.read_text())
    rows.append(dict(name=f.parent.name, K=d["args"]["K"], sub=d["encoder_args"]["sub"], seed=d["encoder_args"]["seed"],
                     **{k: d["mean"][k] for k in M}))
groups = defaultdict(list)
for r in rows:
    groups[re.sub(r"_s20\d\d", "", r["name"])].append(r)
print(f"{'run':<34}" + "".join(f"{k:>11}" for k in M))
for r in rows:
    print(f"{r['name']:<34}" + "".join(f"{r[k]:>11.4f}" for k in M))
print("\nmean ± SD over encoder seeds")
agg = []
for g, rs in sorted(groups.items()):
    if len(rs) > 1:
        a = dict(name=g, n_seeds=len(rs))
        for k in M:
            v = np.array([r[k] for r in rs])
            a[k], a[k + "_sd"] = float(v.mean()), float(v.std(ddof=1))
        agg.append(a)
        print(f"{g + ' (n=' + str(len(rs)) + ')':<34}" + "".join(f"{a[k]:>6.4f}±{a[k + '_sd']:.3f}".rjust(13) for k in M))
with open(RUNS / "img" / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "sub", "seed", "K"] + M)
    w.writeheader()
    w.writerows(rows)
if agg:
    with open(RUNS / "img" / "summary_seedmean.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)
