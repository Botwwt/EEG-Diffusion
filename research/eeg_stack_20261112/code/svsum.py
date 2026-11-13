"""Summarise --selvar / --post variants over subjects: mean metrics, delta vs a reference, wins per subject,
and a leave-one-subject-out choice of the post variant (chosen on the other subjects by a criterion).

python svsum.py <run suffix> [ref=main|selvar:<spec>] [criterion metric list for LOSO, e.g. ssim,clip]
"""
import json
import sys

import numpy as np

from common import RUNS

KEYS = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "effnet"]
suf = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else "main"
crit = sys.argv[3].split(",") if len(sys.argv) > 3 else []
subs, rows = [], {}
for s in range(1, 11):
    f = RUNS / "img" / f"s{s:02d}{suf}" / "metrics.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    if "post" not in d and "selvar" not in d and "post_off" not in d:
        continue
    subs.append(s)
    v = {"main": d["mean"]}
    v.update({f"selvar:{k}": m for k, m in d.get("selvar", {}).items()})
    v.update({f"post:{k}": m for k, m in d.get("post", {}).items()})
    v.update({f"post:{k}": m for k, m in d.get("post_off", {}).items()})
    for k, m in v.items():
        rows.setdefault(k, {})[s] = [m[x] for x in KEYS]
full = {k: np.array([r[s] for s in subs]) for k, r in rows.items() if len(r) == len(subs)}
part = {k: r for k, r in rows.items() if len(r) < len(subs)}
rows = full
print(f"{suf} | subjects {subs} | reference {ref}")
print(f"{'variant':26}" + "".join(f"{k[:8]:>9}" for k in KEYS))
R = rows[ref]
for k, r in rows.items():
    print(f"{k:26}" + "".join(f"{v:9.4f}" for v in r.mean(0)))
    if k != ref:
        d = r - R
        win = [(int((d[:, j] < 0).sum()) if KEYS[j] == "effnet" else int((d[:, j] > 0).sum())) for j in range(len(KEYS))]
        print(f"{'  delta':26}" + "".join(f"{v:+9.4f}" for v in d.mean(0)) + "   wins " + " ".join(f"{w}" for w in win))
for k, r in part.items():  # variants present on a subset of subjects: compare on that subset
    ss = sorted(r)
    a, b = np.array([r[s] for s in ss]), np.array([rows[ref][subs.index(s)] for s in ss])
    print(f"{k:26}" + "".join(f"{v:9.4f}" for v in a.mean(0)) + f"   (subjects {ss})")
    print(f"{'  delta':26}" + "".join(f"{v:+9.4f}" for v in (a - b).mean(0)))
if crit:
    post = [k for k in rows if k.startswith("post:")]
    idx = [KEYS.index(c) for c in crit]
    pick, out = [], []
    for i in range(len(subs)):
        others = [j for j in range(len(subs)) if j != i]
        sc = {k: (rows[k][others][:, idx] - R[others][:, idx]).sum(1).mean() for k in post}
        b = max(sc, key=sc.get) if max(sc.values()) > 0 else ref
        pick.append(b); out.append(rows[b][i])
    out = np.array(out)
    print(f"LOSO by {'+'.join(crit)}: picks {pick}")
    print(f"{'  loso':26}" + "".join(f"{v:9.4f}" for v in out.mean(0)))
    print(f"{'  delta':26}" + "".join(f"{v:+9.4f}" for v in (out - R).mean(0)))
