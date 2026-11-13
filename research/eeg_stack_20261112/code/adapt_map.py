"""Spatial reliability map for the high-frequency gain (permet.py map:...) vs a uniform gain at the same SSIM gain.

python adapt_map.py <run suffix> [sigma]
"""
import sys

import numpy as np

from common import RUNS

suf = sys.argv[1]
sg = sys.argv[2] if len(sys.argv) > 2 else "24"
M = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "effnet"]
subs = [s for s in range(1, 11) if (RUNS / "img" / f"s{s:02d}{suf}" / f"permap_s{sg}.npz").exists()]
U = {s: np.load(RUNS / "img" / f"s{s:02d}{suf}" / f"permet_s{sg}.npz") for s in subs}
Pm = {s: np.load(RUNS / "img" / f"s{s:02d}{suf}" / f"permap_s{sg}.npz") for s in subs}
G = U[subs[0]]["gains"]
pairs = Pm[subs[0]]["pairs"]
print("subjects", subs)
uni = np.array([[[U[s][m][g].mean() for m in M] for g in range(len(G))] for s in subs])   # [S, G, M]
mp = np.array([[[Pm[s][m][j].mean() for m in M] for j in range(len(pairs))] for s in subs])
d_uni = (uni - uni[:, :1]).mean(0)      # vs gain 1.0
d_map = (mp - uni[:, :1]).mean(0)
order = np.argsort(d_uni[:, 1])
print(f"{'map lo-hi':<12}{'ssim':>8}" + "".join(f"{m[:7]:>9}" for m in M if m != "ssim") + "   (map minus uniform at the same SSIM gain; effnet sign flipped)")
for j, (lo, hi) in enumerate(pairs):
    u = np.array([np.interp(d_map[j, 1], d_uni[order, 1], d_uni[order, i]) for i in range(len(M))])
    a = d_map[j] - u
    a[M.index("effnet")] *= -1
    per = []
    for k, s in enumerate(subs):   # per-subject sign of the CLIP advantage
        ds, du = mp[k, j] - uni[k, 0], uni[k] - uni[k, :1]
        o = np.argsort(du[:, 1])
        per.append(ds[5] - np.interp(ds[1], du[o, 1], du[o, 5]) > 0)
    print(f"{lo:.1f}-{hi:.1f}     {d_map[j, 1]:>+8.4f}" + "".join(f"{a[i]:>+9.4f}" for i in range(len(M)) if M[i] != "ssim") + f"   clip better in {sum(per)}/{len(subs)}")
