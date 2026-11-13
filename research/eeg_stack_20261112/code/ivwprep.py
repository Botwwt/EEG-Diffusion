"""Inverse-variance aggregation of the 80 test repeats, carried through the per-subject pipeline.

Trial weight w_r = 1 / mean squared residual of trial r against the mean of the other 79 trials of the same image
(label-free; robagg.py compares it with mean / trimmed / median / Huber). For each subject:
  1. encoder input = weighted mean  -> runs/enc/<tag>/sub-XX/fold-1_seed2025/cond_test_ivw.npy
  2. the co-author's per-subject head applied to it -> runs/cond/ord_ivw/sub-XX.npy
  3. repeat-corrected ridge (as low.py vae / rcp) on the weighted mean -> runs/low/sub-XX/vae_rcp_ivw_pred_z.npy
Then: python lowsem.py <tag> vae_rcp_ivw --ctest cond_test_ivw.npy --space raw --mode knn --w 0.5 --save lowrkivw_0.5
      python decode_low.py lowrkivw_0.5

python ivwprep.py [encoder tag, default t0_r4]
"""
import sys

import numpy as np
import torch

from common import HIVE, PREP, ROOT, RUNS, concept_folds

sys.path.insert(0, str(HIVE))
sys.path.insert(0, str(ROOT / "code/EEG-Diffusion/research/eeg"))
from brainhive_bounded_component_v1 import BoundedConditionAdapter  # noqa: E402
from main.models_brain import BrainEncoder  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "t0_r4"
dev = "cuda"
GK, BK = "backbone.block.layers.0.norm.weight", "backbone.block.layers.0.norm.bias"
nz = torch.nn.functional.normalize
Fn = nz(torch.from_numpy(np.load(PREP / "F_test.npy")).float(), dim=1)
Y = torch.load(PREP / "emb_train.pt", weights_only=True)["vae"].to(dev).double()
ym, ystd = Y.mean(0), Y.std(0)
Yc = Y - ym
concept = torch.from_numpy(np.load(PREP / "train_concept.npy"))
fold = torch.from_numpy(concept_folds(1654, 5))[concept].to(dev)


def ivw(x):
    tot = x.sum(1, keepdim=True)
    v = (x - (tot - x) / (x.shape[1] - 1)).pow(2).mean(dim=(2, 3))
    w = 1 / v
    return ((w / w.sum(1, keepdim=True))[:, :, None, None] * x).sum(1), w


def rowcorr(a, b):
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))).mean().item()


acc = lambda c: float(((nz(c, dim=1) @ Fn.T).argmax(1) == torch.arange(200)).float().mean())  # noqa: E731
for s in range(1, 11):
    x = f"{s:02d}"
    xt = torch.from_numpy(np.load(PREP / f"sub-{x}_test.npy")).float().to(dev)
    a, w = ivw(xt)
    neff = float((w.sum(1) ** 2 / (w ** 2).sum(1)).mean())                 # effective number of trials
    d = RUNS / "enc" / tag / f"sub-{x}" / "fold-1_seed2025"
    sd = torch.load(d / "brain_model.pt", weights_only=True)
    enc = BrainEncoder(backbone="brain_mlp", brain_channels=17, brain_sequence_length=250, hidden_size=1024, embedding_dim=1440, dropout=0.1)
    enc.load_state_dict(sd)
    enc.eval()
    with torch.no_grad():
        c = enc.backbone(a.cpu()).float()
        c80 = torch.from_numpy(np.load(d / "cond_test_r80.npy")).float()
        head = BoundedConditionAdapter(predict_risk=False).eval()
        head.load_state_dict(torch.load(RUNS / "ord" / f"sub-{x}" / "head.pt", weights_only=True))
        wg, bg = sd[GK].float(), sd[BK].float()
        h, _ = head(c, wg.expand_as(c), bg.expand_as(c))
        h80, _ = head(c80, wg.expand_as(c80), bg.expand_as(c80))
    np.save(d / "cond_test_ivw.npy", c.numpy())
    (RUNS / "cond" / "ord_ivw").mkdir(parents=True, exist_ok=True)
    np.save(RUNS / "cond" / "ord_ivw" / f"sub-{x}.npy", h.numpy())
    # ridge, as low.py (vae, rcp)
    x_tr = torch.from_numpy(np.load(PREP / f"sub-{x}_train.npy")).to(dev)
    N = len(x_tr)
    xm = x_tr.mean(1).flatten(1).double()
    mu, sdv = xm.mean(0), xm.std(0)
    X = (xm - mu) / sdv
    S = torch.zeros(X.shape[1], X.shape[1], dtype=torch.float64, device=dev)
    for i in range(0, N, 2000):
        dd = ((x_tr[i:i + 2000] - x_tr[i:i + 2000].mean(1, keepdim=True)).flatten(2).double() / sdv).flatten(0, 1)
        S += dd.T @ dd
    S /= 3 * N
    del x_tr
    cv = {}
    for lam in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        r_ = []
        for k in range(5):
            tr, ho = fold != k, fold == k
            e, V = torch.linalg.eigh(X[tr].T @ X[tr] / int(tr.sum()))
            P = V.T @ (X[tr].T @ Yc[tr] / int(tr.sum()))
            r_.append(rowcorr(X[ho] @ (V @ (P / (e + lam * float(e.mean()))[:, None])), Yc[ho]))
        cv[lam] = np.mean(r_)
    lam = max(cv, key=cv.get)
    M = X.T @ X / N
    pen = lam * float(torch.trace(M)) / M.shape[0]
    e, V = torch.linalg.eigh(M - S / 4)
    M = (V * e.clamp_min(0)) @ V.T + S / 80
    W = torch.linalg.solve(M + pen * torch.eye(M.shape[0], dtype=M.dtype, device=dev), X.T @ Yc / N)
    dev_ = ((a.flatten(1).double() - mu) / sdv) @ W
    (RUNS / "low" / f"sub-{x}").mkdir(parents=True, exist_ok=True)
    np.save(RUNS / "low" / f"sub-{x}" / "vae_rcp_ivw_pred_z.npy", (dev_ / ystd).float().cpu().numpy())
    print(f"sub-{x}: effective trials {neff:.1f}/80 | ridge weight {lam} | top1 raw {acc(c80):.3f} -> ivw {acc(c):.3f}, "
          f"after head {acc(h80):.3f} -> {acc(h):.3f}", flush=True)
