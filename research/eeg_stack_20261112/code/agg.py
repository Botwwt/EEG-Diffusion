"""Aggregate condition-level results: python agg.py [sub ...] -> table on stdout + runs/enc/summary.csv

Per (tag, subject): held-out-concept metrics are averaged over the 5 folds within a seed, then mean / SD over seeds.
Official-test metrics come from the fold=-1 models (trained on all 1654 concepts).
"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np

from common import RUNS

subs = [int(s) for s in sys.argv[1:]] or None
rows = defaultdict(lambda: defaultdict(dict))  # (tag, sub) -> seed -> fold -> metrics
for f in sorted((RUNS / "enc").glob("*/sub-*/fold*_seed*/metrics.json")):
    tag, sub = f.parts[-4], int(f.parts[-3][4:])
    if subs and sub not in subs:
        continue
    fold, seed = f.parts[-2].split("_")
    rows[(tag, sub)][int(seed[4:])][int(fold[4:])] = json.loads(f.read_text())


def ms(v):
    v = np.asarray(v, dtype=float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"), len(v))


out = []
for (tag, sub), by_seed in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    oof = defaultdict(list)
    test = defaultdict(list)
    for seed, by_fold in sorted(by_seed.items()):
        folds = [m for fo, m in by_fold.items() if fo >= 0]
        if len(folds) == 5:
            for k in ("1", "2", "3", "4"):
                for name in ("ic_top1", "ic_2afc", "cv200_top1", "gallery_top1", "cos_pair"):
                    oof[f"oof_k{k}_{name}"].append(np.mean([m["oof_by_k"][k][name] for m in folds]))
            oof["seen_ic_top1"].append(np.mean([m["train_concepts_r4_ic_top1"] for m in folds]))
        if -1 in by_fold:
            m = by_fold[-1]
            test["test_r80_top1"].append(m["test_r80_top1_top5"][0])
            test["test_r80_top5"].append(m["test_r80_top1_top5"][1])
            test["test_emb20x4_top1"].append(m["test_emb20x4_top1_top5"][0])
            for k, v in m["test_top1_top5_by_k"].items():
                test[f"test_k{k}_top1"].append(v[0])
    rec = dict(tag=tag, sub=sub)
    for name, v in {**oof, **test}.items():
        rec[name], rec[name + "_sd"], rec[name + "_n"] = ms(v)
    out.append(rec)

cols = ["oof_k4_ic_top1", "oof_k4_ic_2afc", "oof_k4_cv200_top1", "oof_k1_ic_top1", "seen_ic_top1",
        "test_r80_top1", "test_r80_top5", "test_emb20x4_top1", "test_k4_top1", "test_k1_top1"]
print(f"{'tag':<16}{'sub':>4} " + " ".join(f"{c.replace('oof_', 'o_').replace('test_', 't_'):>19}" for c in cols))
for r in out:
    print(f"{r['tag']:<16}{r['sub']:>4} " + " ".join(
        (f"{100 * r[c]:6.2f}±{100 * r[c + '_sd']:4.2f} (n={r[c + '_n']})" if c in r else " " * 19).rjust(19) for c in cols))
if out:
    keys = sorted({k for r in out for k in r})
    with open(RUNS / "enc" / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "sub"] + [k for k in keys if k not in ("tag", "sub")])
        w.writeheader()
        w.writerows(out)
