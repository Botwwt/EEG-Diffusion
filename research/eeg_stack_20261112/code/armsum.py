"""Mean over the subjects that have every listed arm, and the difference to the first arm (with wins per metric).

python armsum.py <run name pattern with {s} for the 2-digit subject> <arm suffix> [<arm suffix> ...]
  e.g. python armsum.py 's{s}_t0_s2025_A1_ord_lowrkt0_0.5' '' _smap_map_0.5_0.75 _smap_const_0.5_0.75
"""
import json
import sys

import numpy as np

from common import RUNS

M = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "effnet"]
pat, arms = sys.argv[1], sys.argv[2:]


def get(s, a):
    f = RUNS / "img" / (pat.format(s=f"{s:02d}") + a) / "metrics.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return [d["mean"][k] for k in M], d.get("smap_mean_strength")


subs = [s for s in range(1, 11) if all(get(s, a) for a in arms)]
print("subjects", subs)
print(f"{'arm':<34}" + "".join(f"{m[:7]:>9}" for m in M) + "  strength")
base = np.array([get(s, arms[0])[0] for s in subs])
for a in arms:
    v = np.array([get(s, a)[0] for s in subs])
    st = [get(s, a)[1] for s in subs]
    print(f"{(a or 'base'):<34}" + "".join(f"{x:>9.4f}" for x in v.mean(0)) + (f"  {np.mean(st):.3f}" if st[0] else ""))
    if a != arms[0]:
        d = v - base
        w = [int(((d[:, i] < 0) if M[i] == "effnet" else (d[:, i] > 0)).sum()) for i in range(len(M))]
        print(f"{'  delta / wins':<34}" + "".join(f"{x:>+9.4f}" for x in d.mean(0)) + "  " + " ".join(map(str, w)))
