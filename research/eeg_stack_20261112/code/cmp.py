"""cmp.py <arm A suffix> <arm B suffix> [prefix]: per-subject paired comparison of two image arms (subjects where both exist)."""
import json, sys
import numpy as np
from common import RUNS
M = ['pixcorr', 'ssim', 'alex2', 'alex5', 'inception', 'clip', 'effnet']
pre = sys.argv[3] if len(sys.argv) > 3 else "_ms_e10_ft10in_A1_"
def ld(n):
    f = RUNS / 'img' / n / 'metrics.json'
    return [json.loads(f.read_text())['mean'][k] for k in M] if f.exists() else None
a, b, ss = [], [], []
for s in range(1, 11):
    x, y = ld(f's{s:02d}{pre}{sys.argv[1]}'), ld(f's{s:02d}{pre}{sys.argv[2]}')
    if x and y:
        a.append(x); b.append(y); ss.append(s)
a, b = np.array(a), np.array(b); d = b - a
print(f"{sys.argv[1]} -> {sys.argv[2]} | subjects {ss}")
print(f"{'':8}" + "".join(f"{k:>9}" for k in M))
print(f"{'A':8}" + "".join(f"{v:9.4f}" for v in a.mean(0)))
print(f"{'B':8}" + "".join(f"{v:9.4f}" for v in b.mean(0)))
print(f"{'B-A':8}" + "".join(f"{v:+9.4f}" for v in d.mean(0)))
print(f"{'better':8}" + "".join(f"{int(((d[:, i] < 0) if k == 'effnet' else (d[:, i] > 0)).sum()):>6}/{len(ss):<2}" for i, k in enumerate(M)))
