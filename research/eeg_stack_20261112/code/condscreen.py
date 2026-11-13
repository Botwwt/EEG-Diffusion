"""Screening of IP-condition variants on the inverse-variance (IVW) input, per subject, before image generation.

Variants (all label-free; the per-subject ordinary head is applied to every one with the seed-2025 trunk affine):
  base     seed-2025 encoder on the IVW mean of the 80 repeats (= O2 / O3 / O4 condition, ord_ivw)
  mwN, w150N, w100N   the same over encoders trained on the 0-150 / 0-100 sample windows (w*) or on all three
           windows 0-250, 0-150, 0-100 (mw, multi-window ensemble), N seeds each; combine with bagG as for ens
  ensN     mean of the outputs of N independently seeded encoders (deep ensembles, Lakshminarayanan et al. 2017),
           each on the IVW input, rescaled per image to the norm of base (averaging shrinks the norm)
  bagG     seed-2025 encoder on G disjoint IVW group means (80 / G repeats each), outputs averaged, rescaled to base
           norm (test inputs at the noise level the encoder was trained on; bagging over repeats)
  ensNbagG both
  <v>_pK   any variant projected onto the span of the training-target mean and the top-K principal components of
           the training targets F_train (subspace projection / PCA denoising), norm kept
  tta*     conditions saved by tta.py (cond_test_<name>.npy), optionally rescaled to the base norm ('n' suffix)
Prints 200-way top-1 / top-5 and mean cos to F after the head, and the cosine drift from base.
Labels are used only for scoring.

python condscreen.py <subs> <variants, comma-separated> [save: 1 writes runs/cond/ord_<variant>/sub-XX.npy and,
   before the head, cond_test_<variant>.npy in the seed-2025 encoder directory]
"""
import sys

import numpy as np
import torch

from common import HIVE, PREP, ROOT, RUNS

sys.path.insert(0, str(HIVE))
sys.path.insert(0, str(ROOT / "code/EEG-Diffusion/research/eeg"))
from brainhive_bounded_component_v1 import BoundedConditionAdapter  # noqa: E402
from main.models_brain import BrainEncoder  # noqa: E402

subs = [int(s) for s in sys.argv[1].split(",")]
VAR = sys.argv[2].split(",")
save = len(sys.argv) > 3 and sys.argv[3] == "1"
GK, BK = "backbone.block.layers.0.norm.weight", "backbone.block.layers.0.norm.bias"
nz = torch.nn.functional.normalize
Fn = nz(torch.from_numpy(np.load(PREP / "F_test.npy")).float(), dim=1)
t = torch.arange(200)
SEEDS = [2025, 2028, 2029, 2030, 2031, 2032, 2033]   # seeds 2026 / 2027 exist only for some subjects


def ivw(x):
    tot = x.sum(1, keepdim=True)
    v = (x - (tot - x) / (x.shape[1] - 1)).pow(2).mean(dim=(2, 3))
    w = 1 / v
    return ((w / w.sum(1, keepdim=True))[:, :, None, None] * x).sum(1)


SETS = {"ens": ["t0_r4"], "w150": ["t0_r4_w150"], "w100": ["t0_r4_w100"],
        "mw": ["t0_r4", "t0_r4_w150", "t0_r4_w100"]}   # mw: multi-window ensemble (0-250, 0-150, 0-100 samples)


def load(x, seed, tag="t0_r4"):
    d = RUNS / "enc" / tag / f"sub-{x}" / f"fold-1_seed{seed}"
    sd = torch.load(d / "brain_model.pt", weights_only=True)
    a_ = __import__("json").loads((d / "metrics.json").read_text())["args"]
    m = BrainEncoder(backbone="brain_mlp", brain_channels=17, brain_sequence_length=a_["t1"] - a_["t0"], hidden_size=1024, embedding_dim=1440, dropout=0.1)
    m.load_state_dict(sd)
    m.win = (a_["t0"], a_["t1"])
    return m.eval(), sd


def score(c):
    s = nz(c, dim=1) @ Fn.T
    rk = (s > s.diag()[:, None]).sum(1)
    return float((rk == 0).float().mean()), float((rk < 5).float().mean()), float((nz(c, dim=1) * Fn).sum(1).mean())


Ftr = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
muF = Ftr.mean(0)
_, _, Vt = torch.linalg.svd(Ftr - muF, full_matrices=False)
PB = {}


def proj(c, K):
    if K not in PB:
        PB[K] = torch.linalg.qr(torch.cat([muF[:, None], Vt[:K].T], 1))[0]
    q = PB[K]
    p_ = (c @ q) @ q.T
    return p_ * c.norm(dim=1, keepdim=True) / p_.norm(dim=1, keepdim=True)


res = {v: [] for v in VAR}
for s in subs:
    x = f"{s:02d}"
    xt = torch.from_numpy(np.load(PREP / f"sub-{x}_test.npy")).float()
    a = ivw(xt)
    enc0, sd0 = load(x, 2025)
    wg, bg = sd0[GK].float(), sd0[BK].float()
    head = BoundedConditionAdapter(predict_risk=False).eval()
    head.load_state_dict(torch.load(RUNS / "ord" / f"sub-{x}" / "head.pt", weights_only=True))
    d0 = RUNS / "enc" / "t0_r4" / f"sub-{x}" / "fold-1_seed2025"
    cache = {}

    def enc_out(seed, G, tag="t0_r4"):
        if (seed, G, tag) not in cache:
            m = enc0 if (seed, tag) == (2025, "t0_r4") else load(x, seed, tag)[0]
            t0_, t1_ = m.win
            with torch.no_grad():
                if G == 1:
                    cache[(seed, G, tag)] = m.backbone(a[..., t0_:t1_]).float()
                else:
                    k = 80 // G
                    perm = torch.randperm(80, generator=torch.Generator().manual_seed(7 + s))
                    cache[(seed, G, tag)] = torch.stack([m.backbone(ivw(xt[:, perm[i * k:(i + 1) * k]])[..., t0_:t1_]).float() for i in range(G)]).mean(0)
        return cache[(seed, G, tag)]

    base = enc_out(2025, 1)
    with torch.no_grad():
        hb, _ = head(base, wg.expand_as(base), bg.expand_as(base))
    for vfull in VAR:
        v, _, pk = vfull.partition("_p")
        if v == "base":
            c = base
        elif v.startswith("ens") and "tta" in v:
            # ens<N><tta name>: mean of the test-time-adapted outputs of N seeds (tta.py with TTA_SEED), base norm
            N, nm = int(v[3:].split("tta")[0]), "tta" + v.split("tta")[1]
            c = torch.stack([torch.from_numpy(np.load(RUNS / "enc" / "t0_r4" / f"sub-{x}" / f"fold-1_seed{k}" / f"cond_test_{nm}.npy")).float()
                             for k in SEEDS[:N]]).mean(0)
            c = c * base.norm(dim=1, keepdim=True) / c.norm(dim=1, keepdim=True)
        elif v.startswith("tta"):
            # tta<name>[m | a<alpha>]: m = keep the base mean over the 200 images, replace only the deviations;
            # a<alpha> = base + alpha * (adapted - base)
            nm, mode = v, ""
            if v.endswith("m"):
                nm, mode = v[:-1], "m"
            elif "a" in v[3:]:
                nm, alpha = v[:3] + v[3:].split("a")[0], float(v[3:].split("a")[1])
                mode = "a"
            c = torch.from_numpy(np.load(d0 / f"cond_test_{nm}.npy")).float()
            if mode == "m":
                c = c - c.mean(0) + base.mean(0)
            elif mode == "a":
                c = base + alpha * (c - base)
        else:
            N, G, tags = 1, 1, ["t0_r4"]
            if v.startswith("bag"):
                G = int(v[3:])
            else:      # <set><N>[bag<G>], set in SETS
                st = next(k for k in sorted(SETS, key=len, reverse=True) if v.startswith(k))
                tags, rest = SETS[st], v[len(st):]
                N = int(rest.split("bag")[0])
                G = int(rest.split("bag")[1]) if "bag" in rest else 1
            c = torch.stack([enc_out(sd_, G, tg) for tg in tags for sd_ in SEEDS[:N]]).mean(0)
            c = c * base.norm(dim=1, keepdim=True) / c.norm(dim=1, keepdim=True)
        if pk:
            c = proj(c, int(pk))
        with torch.no_grad():
            h, _ = head(c, wg.expand_as(c), bg.expand_as(c))
        r = score(h) + (float((nz(h, dim=1) * nz(hb, dim=1)).sum(1).mean()),)
        res[vfull].append(r)
        if save and vfull != "base":
            np.save(d0 / f"cond_test_{vfull}.npy", c.numpy())   # before the head: for lowsem.py --ctest (retrieval)
            (RUNS / "cond" / f"ord_{vfull}").mkdir(parents=True, exist_ok=True)
            np.save(RUNS / "cond" / f"ord_{vfull}" / f"sub-{x}.npy", h.numpy())
    print(f"sub-{x} " + " | ".join(f"{v} {res[v][-1][0]:.3f}/{res[v][-1][2]:.4f}" for v in VAR), flush=True)
print(f"\nmean over {len(subs)} subjects (after the head): top1 top5 cosF cos-to-base, subjects with higher top1 / cosF than base")
b = np.array(res[VAR[0]])
for v in VAR:
    r = np.array(res[v])
    print(f"{v:<12} {r[:, 0].mean():.4f} {r[:, 1].mean():.4f} {r[:, 2].mean():.4f} {r[:, 3].mean():.4f}   {(r[:, 0] > b[:, 0]).sum()} / {(r[:, 2] > b[:, 2]).sum()}")
