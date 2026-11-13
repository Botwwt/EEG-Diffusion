"""Every generation run (runs/img/*/metrics.json) as one CSV row, plus one row per --selvar / --post variant (name#kind:spec).

python runsum.py > runs_summary.csv
"""
import csv
import json
import sys

from common import RUNS

M = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "effnet", "clip_top1"]
A = ["run", "use_A", "alpha", "nsel", "sel_low", "init_file", "strength", "steps", "blend", "cond_file", "amp", "cfg", "ip_scale",
     "prompt_file", "loop", "proj_ckpt", "reps", "K"]
w = csv.writer(sys.stdout, lineterminator="\n")
w.writerow(["name"] + A + M)
for f in sorted(RUNS.glob("img/*/metrics.json")):
    d = json.loads(f.read_text())
    if "mean" not in d or f.parent.name.startswith("test"):
        continue
    a = d["args"]
    base = [a.get(k, "") for k in A]
    w.writerow([f.parent.name] + base + [round(d["mean"].get(k, float("nan")), 5) for k in M])
    for kind in ("selvar", "post", "post_off"):
        for spec, m in d.get(kind, {}).items():
            w.writerow([f"{f.parent.name}#{kind}:{spec}"] + base + [round(m.get(k, float("nan")), 5) for k in M])
