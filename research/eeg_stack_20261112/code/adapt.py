"""Reliability-adaptive high-frequency gain, scored offline from permet.py tables and rel.py reliabilities.

Policy 'uniform g': every image gets gain g.  Policy 'rel <signal> lo hi': image i gets lo + (hi - lo) * q_i, q_i = rank
percentile of its split-half reliability within the subject (label-free), rounded to the nearest gain of the grid.
Controls: the same gains assigned in random order ('perm', mean of 20 permutations) and in reversed order ('rev').
Selection rule (fixed in advance, as for B‴): for each held-out subject, among all policies take the one with the
largest SSIM gain over 'uniform 1' on the other subjects with CLIP loss <= 0.001 and Alex2 loss <= 0.005.

python adapt.py <run suffix> [sigma]
"""
import itertools
import sys

import numpy as np

from common import RUNS

suf = sys.argv[1]
sg = sys.argv[2] if len(sys.argv) > 2 else "24"
M = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "effnet"]
subs, T, R = [], {}, {}
for s in range(1, 11):
    f = RUNS / "img" / f"s{s:02d}{suf}" / f"permet_s{sg}.npz"
    if f.exists():
        subs.append(s)
        T[s] = np.load(f)
        R[s] = np.load(RUNS / "rel" / f"sub-{s:02d}.npz")
G = T[subs[0]]["gains"]
print("subjects", subs, "| gains", G.tolist())
rng = np.random.default_rng(0)


def pct(v):
    return (np.argsort(np.argsort(v)) + 0.5) / len(v)


def score(s, gidx):
    """gidx [200] gain index per image -> metric vector (mean over K samples and images)."""
    return np.array([T[s][m][gidx, :, np.arange(200)].mean() for m in M])


def gi(g):
    return np.abs(G[None, :] - np.asarray(g)[:, None]).argmin(1)


# ---------------------------------------------------------------- diagnostics
print("\ncorrelation over images (pooled within subject, then averaged) of reliability with per-image quantities")
i1, i8 = gi([1.0])[0], gi([0.8])[0]
for sig in ("low", "emb"):
    row = []
    for m in ("ssim", "clip", "alex2", "pixcorr"):
        base = np.mean([np.corrcoef(R[s][sig], T[s][m][i1].mean(0))[0, 1] for s in subs])
        dlt = np.mean([np.corrcoef(R[s][sig], (T[s][m][i8] - T[s][m][i1]).mean(0))[0, 1] for s in subs])
        row.append(f"{m}: level {base:+.3f} gain-0.8 delta {dlt:+.3f}")
    print(f"  {sig:<4}", " | ".join(row))

# ---------------------------------------------------------------- policies
P = {}
for g in G:
    P[f"uniform {g:.1f}"] = lambda s, g=g: np.full(200, gi([g])[0])
for sig in ("low", "emb", "lowemb"):
    for lo, hi in itertools.combinations(sorted(G), 2):
        if hi - lo < 0.15:
            continue
        def pol(s, sig=sig, lo=lo, hi=hi, mode="rel"):
            r = (pct(R[s]["low"]) + pct(R[s]["emb"])) / 2 if sig == "lowemb" else R[s][sig]
            q = pct(r)
            if mode == "rev":
                q = 1 - q
            return gi(lo + (hi - lo) * q)
        P[f"rel {sig} {lo:.1f}-{hi:.1f}"] = pol
        P[f"rev {sig} {lo:.1f}-{hi:.1f}"] = lambda s, pol=pol: pol(s, mode="rev")
res = {name: np.array([score(s, f(s)) for s in subs]) for name, f in P.items()}
for name in [n for n in P if n.startswith("rel ")]:   # random-order control: same gains, shuffled
    f = P[name]
    res["perm " + name[4:]] = np.array([np.mean([score(s, rng.permutation(f(s))) for _ in range(20)], 0) for s in subs])
base = res["uniform 1.0"]


def show(names):
    print(f"{'policy':<26}" + "".join(f"{m[:7]:>9}" for m in M))
    for n in names:
        d = (res[n] - base).mean(0)
        print(f"{n:<26}" + "".join(f"{x:>+9.4f}" for x in d))


print("\nmean over subjects, difference to 'uniform 1.0' (images as generated)")
show([n for n in res if n.startswith("uniform")])
# matched-SSIM comparison: for each adaptive policy, the uniform gain interpolated to the same SSIM gain
uni = sorted([(float((res[n] - base).mean(0)[1]), (res[n] - base).mean(0)) for n in res if n.startswith("uniform")], key=lambda t: t[0])
us, ud = np.array([u[0] for u in uni]), np.array([u[1] for u in uni])


def uniform_at(ssim_gain):
    return np.array([np.interp(ssim_gain, us, ud[:, j]) for j in range(len(M))])


print("\nadaptive vs uniform at the SAME mean SSIM gain (positive = adaptive better; effnet sign flipped)")
print(f"{'policy':<26}{'ssim':>8}" + "".join(f"{m[:7]:>9}" for m in M if m != "ssim") + "   ctl-perm clip  ctl-rev clip")
rows = []
for n in [n for n in res if n.startswith("rel ")]:
    d = (res[n] - base).mean(0)
    u = uniform_at(d[1])
    adv = d - u
    adv[M.index("effnet")] *= -1
    dp = (res["perm " + n[4:]] - base).mean(0) - uniform_at((res["perm " + n[4:]] - base).mean(0)[1])
    dr = (res["rev " + n[4:]] - base).mean(0) - uniform_at((res["rev " + n[4:]] - base).mean(0)[1])
    rows.append((adv[M.index("clip")], n, d[1], adv, dp[M.index("clip")], dr[M.index("clip")]))
for a, n, s_, adv, cp, cr in sorted(rows, reverse=True)[:15]:
    print(f"{n:<26}{s_:>+8.4f}" + "".join(f"{adv[j]:>+9.4f}" for j, m in enumerate(M) if m != "ssim") + f"   {cp:>+9.4f}   {cr:>+9.4f}")

# ---------------------------------------------------------------- leave-one-subject-out selection
print("\nleave-one-subject-out choice (largest SSIM gain on the others with CLIP loss <= 0.001, Alex2 loss <= 0.005)")
for fam in ("uniform", "rel", "any"):
    names = [n for n in res if (fam == "any" and (n.startswith("uniform") or n.startswith("rel "))) or n.startswith(fam + " ")]
    got, picks = [], []
    for j, s in enumerate(subs):
        oth = [i for i in range(len(subs)) if i != j]
        ok = [n for n in names if (res[n][oth] - base[oth]).mean(0)[5] >= -0.001 and (res[n][oth] - base[oth]).mean(0)[2] >= -0.005]
        best = max(ok, key=lambda n: (res[n][oth] - base[oth]).mean(0)[1])
        picks.append(best)
        got.append(res[best][j])
    got = np.array(got)
    d = got - base
    wins = [int(((d[:, i] < 0) if M[i] == "effnet" else (d[:, i] > 0)).sum()) for i in range(len(M))]
    print(f"  {fam:<8}" + "".join(f"{x:>9.4f}" for x in got.mean(0)) + "  delta " + " ".join(f"{x:+.4f}" for x in d.mean(0)))
    print(f"           picks: {sorted(set(picks))}  wins vs as-generated {wins}")
