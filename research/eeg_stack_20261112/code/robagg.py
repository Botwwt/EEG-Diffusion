"""Aggregating the 80 test repeats: plain mean vs robust / reliability-weighted estimators (robust statistics).

  mean      plain average (what every method uses)
  trimK     per-feature trimmed mean, K % cut at each end
  median    per-feature median
  ivw       inverse-variance trial weights: w_r = 1 / mean squared residual of trial r against the mean of the other 79
            (one weight per trial and image; down-weights noisy trials, e.g. blinks or movement)
  huber     per-feature Huber M-estimate (IRLS, threshold 1.345 robust SD of the residuals across repeats)
Label-free: only the test EEG of the image itself. Scored with the per-subject encoder (raw 200-way retrieval of F,
top-1 / top-5) and with the repeat-corrected ridge (centred correlation with the true test latent). The labels are
used only for scoring.

python robagg.py [encoder tag] [methods, comma-separated]  ->  runs/rel/robagg_<tag>.json, and runs/rel/agg_<method>/sub-XX_test_agg.npy
"""
import json
import sys

import numpy as np
import torch

from common import HIVE, PREP, RUNS, concept_folds

sys.path.insert(0, str(HIVE))
from main.models_brain import BrainEncoder  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "t0_r4"
dev = "cuda"
nz = torch.nn.functional.normalize
F = nz(torch.from_numpy(np.load(PREP / "F_test.npy")).float(), dim=1).to(dev)
lat_te = torch.load(PREP / "emb_test.pt", weights_only=True)["vae"].double().to(dev)
Y = torch.load(PREP / "emb_train.pt", weights_only=True)["vae"].to(dev).double()
Yc = Y - Y.mean(0)
concept = torch.from_numpy(np.load(PREP / "train_concept.npy"))
fold = torch.from_numpy(concept_folds(1654, 5))[concept].to(dev)


def aggregate(x, method):
    """x [200, 80, 17, 250] -> [200, 17, 250]"""
    if method == "mean":
        return x.mean(1)
    if method.startswith("trim"):
        k = int(round(80 * int(method[4:]) / 100))
        return x.sort(1).values[:, k:80 - k].mean(1)
    if method == "median":
        return x.median(1).values
    if method == "ivw":
        tot = x.sum(1, keepdim=True)
        loo = (tot - x) / 79
        v = (x - loo).pow(2).mean(dim=(2, 3))                         # [200, 80] residual power per trial
        w = 1 / v
        w = w / w.sum(1, keepdim=True)
        return (w[:, :, None, None] * x).sum(1)
    if method.startswith("ivw") and method != "ivw":
        tot = x.sum(1, keepdim=True)
        if method == "ivwc":       # one weight per trial and channel
            v = (x - (tot - x) / 79).pow(2).mean(3)                          # [200, 80, 17]
            w = 1 / v
            return ((w / w.sum(1, keepdim=True))[..., None] * x).sum(1)
        if method == "ivw2":       # stronger: w = 1 / v^2
            v = (x - (tot - x) / 79).pow(2).mean(dim=(2, 3))
            w = 1 / v ** 2
            return ((w / w.sum(1, keepdim=True))[:, :, None, None] * x).sum(1)
        if method == "ivwit":      # iterated: residuals against the current weighted mean, 3 rounds
            mu = x.mean(1)
            for _ in range(3):
                v = (x - mu[:, None]).pow(2).mean(dim=(2, 3))
                w = 1 / v
                mu = ((w / w.sum(1, keepdim=True))[:, :, None, None] * x).sum(1)
            return mu
    if method == "huber":
        mu = x.median(1).values
        for _ in range(10):
            r = x - mu[:, None]
            s = 1.4826 * r.abs().median(1).values.clamp_min(1e-6)       # robust SD across repeats
            u = r.abs() / (1.345 * s[:, None])
            w = torch.where(u <= 1, torch.ones_like(u), 1 / u)
            mu = (w * x).sum(1) / w.sum(1)
        return mu
    raise ValueError(method)


def rowcorr(a, b):
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))).mean().item()


METHODS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["mean", "trim10", "trim20", "median", "ivw", "huber"]
res = {m: [] for m in METHODS}
for s in range(1, 11):
    x = f"{s:02d}"
    xt = torch.from_numpy(np.load(PREP / f"sub-{x}_test.npy")).float().to(dev)
    enc = BrainEncoder(backbone="brain_mlp", brain_channels=17, brain_sequence_length=250, hidden_size=1024, embedding_dim=1440, dropout=0.1)
    enc.load_state_dict(torch.load(RUNS / "enc" / tag / f"sub-{x}" / "fold-1_seed2025" / "brain_model.pt", weights_only=True))
    enc = enc.to(dev).eval()
    # ridge (repeat-corrected, as low.py vae / rcp)
    x_tr = torch.from_numpy(np.load(PREP / f"sub-{x}_train.npy")).to(dev)
    N = len(x_tr)
    xm = x_tr.mean(1).flatten(1).double()
    mu, sd = xm.mean(0), xm.std(0)
    X = (xm - mu) / sd
    S = torch.zeros(X.shape[1], X.shape[1], dtype=torch.float64, device=dev)
    for i in range(0, N, 2000):
        d = ((x_tr[i:i + 2000] - x_tr[i:i + 2000].mean(1, keepdim=True)).flatten(2).double() / sd).flatten(0, 1)
        S += d.T @ d
    S /= 3 * N
    del x_tr
    cv = {}
    for lam in (0.3, 1.0, 3.0):
        r_ = []
        for k in range(5):
            a, b = fold != k, fold == k
            e, V = torch.linalg.eigh(X[a].T @ X[a] / int(a.sum()))
            P = V.T @ (X[a].T @ Yc[a] / int(a.sum()))
            r_.append(rowcorr(X[b] @ (V @ (P / (e + lam * float(e.mean()))[:, None])), Yc[b]))
        cv[lam] = np.mean(r_)
    lam = max(cv, key=cv.get)
    Mx = X.T @ X / N
    pen = lam * float(torch.trace(Mx)) / Mx.shape[0]
    e, V = torch.linalg.eigh(Mx - S / 4)
    Mx = (V * e.clamp_min(0)) @ V.T + S / 80
    W = torch.linalg.solve(Mx + pen * torch.eye(Mx.shape[0], dtype=Mx.dtype, device=dev), X.T @ Yc / N)
    for m in METHODS:
        a = aggregate(xt, m)
        with torch.no_grad():
            c = nz(enc.backbone(a).float(), dim=1)
        sim = c @ F.T
        rk = (sim > sim.diag()[:, None]).sum(1)
        lp = ((a.flatten(1).double() - mu) / sd) @ W
        res[m].append(dict(top1=float((rk == 0).float().mean()), top5=float((rk < 5).float().mean()),
                           lat=rowcorr(lp, lat_te - Y.mean(0))))
        out = RUNS / "rel" / f"agg_{m}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"sub-{x}_test_agg.npy", a.cpu().numpy())
    print(f"sub-{x} " + " | ".join(f"{m} {res[m][-1]['top1']:.3f}/{res[m][-1]['top5']:.3f}/{res[m][-1]['lat']:.4f}" for m in METHODS), flush=True)
print("\nmean over 10 subjects: top1 / top5 / latent corr, and subjects better than the plain mean")
for m in METHODS:
    a = np.array([[r["top1"], r["top5"], r["lat"]] for r in res[m]])
    b = np.array([[r["top1"], r["top5"], r["lat"]] for r in res["mean"]])
    print(f"{m:<8} {a[:, 0].mean():.4f} {a[:, 1].mean():.4f} {a[:, 2].mean():.4f}   better: {(a > b).sum(0).tolist()}")
(RUNS / "rel" / f"robagg_{tag}_{'-'.join(METHODS)}.json").write_text(json.dumps(res, indent=1))
