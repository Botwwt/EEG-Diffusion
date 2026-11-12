"""Cross-fitted priors for the EEG condition (CPU only, closed form).

python prior.py <tag> <sub> <seed>  ->  runs/cond/<tag>_s<sub>_<seed>/<variant>.npy, table on stdout

The encoder is trained contrastively, so its output c matches the image embedding F only up to what InfoNCE does
not penalise (offset, scale, energy in directions the EEG knows nothing about). The generator was trained on real
F. Here c is mapped towards the F distribution with maps fitted on out-of-fold pairs: every training image has a
condition from an encoder that never saw its concept (folds 0-4), paired with its true F. Nothing from the test
set enters any map; the variants named diag_* use test images and are diagnostics only, never a method.
"""
import sys

import numpy as np
import torch

from common import PREP, RUNS, concept_folds

tag, sub, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
torch.set_num_threads(8)
nz = torch.nn.functional.normalize
d = RUNS / "enc" / tag / f"sub-{sub:02d}"
out = RUNS / "cond" / f"{tag}_s{sub:02d}_{seed}"
out.mkdir(parents=True, exist_ok=True)

F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).double()
F_te = torch.from_numpy(np.load(PREP / "F_test.npy")).double()
concept = torch.from_numpy(np.load(PREP / "train_concept.npy"))
fold = torch.from_numpy(concept_folds(1654, 5))[concept]
C_oof = torch.zeros_like(F_tr)
for k in range(5):
    idx = torch.from_numpy(np.load(d / f"fold{k}_seed{seed}" / "oof_index.npy"))
    assert (fold[idx] == k).all()
    C_oof[idx] = torch.from_numpy(np.load(d / f"fold{k}_seed{seed}" / "cond_oof_r4.npy")).double()
run = d / f"fold-1_seed{seed}"
c80 = torch.from_numpy(np.load(run / "cond_test_r80.npy")).double()       # one input = mean of 80 repeats
cemb = torch.from_numpy(np.load(run / "cond_test_emb20x4.npy")).double()   # mean of 20 embeddings of 4-repeat inputs

X, Y = nz(C_oof, dim=1), F_tr
mu_x, mu_y = X.mean(0), Y.mean(0)
sF = float(Y.norm(dim=1).mean())
var_y = float(((Y - mu_y) ** 2).sum(1).mean())
print(f"== {tag} sub-{sub:02d} seed {seed}")
print(f"||F|| train {sF:.2f} +- {float(Y.norm(dim=1).std()):.2f} | ||c|| oof {float(C_oof.norm(dim=1).mean()):.2f} "
      f"test r80 {float(c80.norm(dim=1).mean()):.2f} emb20x4 {float(cemb.norm(dim=1).mean()):.2f} | "
      f"||mu_F|| {float(mu_y.norm()):.2f} spread of F around its mean {var_y ** 0.5:.2f}")


def ridge_fit(Xa, Ya, lam_rel):
    mx, my = Xa.mean(0), Ya.mean(0)
    Xc, Yc = Xa - mx, Ya - my
    G = Xc.T @ Xc
    lam = lam_rel * float(torch.trace(G)) / G.shape[0]
    W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype), Xc.T @ Yc)
    return mx, my, W


def ridge_apply(m, x, gamma=1.0):
    mx, my, W = m
    return my + gamma * (x - mx) @ W


def retr(pred, target):  # 200-way identification, cosine
    s = nz(pred, dim=1) @ nz(target, dim=1).T
    r = s.argsort(1, descending=True)
    t = torch.arange(len(s))
    return float((r[:, 0] == t).double().mean()), float((r[:, :5] == t[:, None]).any(1).double().mean())


# ------------------------------------------------------------ nested check on training concepts only
print("\nridge on out-of-fold conditions, checked on the left-out concept fold (4-repeat inputs):")
print(f"{'lam_rel':>8}{'R2':>8}{'spread':>8}{'ccos':>8}{'top1':>8}{'top1 raw c':>12}")
LAMS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
gam_star = {}
for lam in LAMS:
    sse = sst = ssp = 0.0
    cc, t1, t1raw = [], [], []
    for k in range(5):
        a, b = fold != k, fold == k
        m = ridge_fit(X[a], Y[a], lam)
        p = ridge_apply(m, X[b])
        sse += float(((Y[b] - p) ** 2).sum()); sst += float(((Y[b] - m[1]) ** 2).sum()); ssp += float(((p - m[1]) ** 2).sum())
        cc.append(float((nz(p - m[1], dim=1) * nz(Y[b] - m[1], dim=1)).sum(1).mean()))
        cid = concept[b]
        uc = torch.unique(cid)[:200]
        for j in range(0, 10, 3):
            pos = torch.stack([torch.nonzero(cid == u).squeeze(1)[j] for u in uc])
            t1.append(retr(p[pos], Y[b][pos])[0])
            t1raw.append(retr(X[b][pos], Y[b][pos])[0])
    gam_star[lam] = (sst / ssp) ** 0.5
    print(f"{lam:>8}{1 - sse / sst:>8.3f}{(ssp / sst) ** 0.5:>8.3f}{np.mean(cc):>8.3f}{np.mean(t1):>8.3f}{np.mean(t1raw):>12.3f}")

# ------------------------------------------------------------ test conditions
Fn_tr = nz(F_tr, dim=1)
mu_fn = Fn_tr.mean(0)
g = mu_fn - mu_x
x80, xemb = nz(c80, dim=1), nz(cemb, dim=1)
V = {}
V["base_r80"] = c80
V["base_emb"] = cemb * (c80.norm(dim=1, keepdim=True) / cemb.norm(dim=1, keepdim=True))
V["shift0.5_r80"] = c80.norm(dim=1, keepdim=True) * nz(x80 + 0.5 * g, dim=1)
V["normF_r80"] = sF * x80
V["shift0.5_normF_r80"] = sF * nz(x80 + 0.5 * g, dim=1)
for lam in (0.03, 0.1, 0.3, 1.0):
    m = ridge_fit(X, Y, lam)
    gs = gam_star[lam]
    for gam, gname in ((1.0, "g1"), (gs ** 0.5, "ghalf"), (gs, "gfull")):
        V[f"ridge{lam}_{gname}_r80"] = ridge_apply(m, x80, gam)
    V[f"ridge{lam}_gfull_emb"] = ridge_apply(m, xemb, gs)
    if lam == 0.1:
        for beta in (0.25, 0.5, 0.75):  # blend of the calibrated raw direction and the ridge output, on the F norm scale
            a = nz(x80 + 0.5 * g, dim=1)
            b = nz(ridge_apply(m, x80, gs), dim=1)
            V[f"blend{beta}_ridge0.1_r80"] = sF * nz((1 - beta) * a + beta * b, dim=1)
# nearest real training images (other concepts by construction), searched with centred cosine
m = ridge_fit(X, Y, 0.1)
q = nz(ridge_apply(m, x80) - mu_y, dim=1)
sim = q @ nz(F_tr - mu_y, dim=1).T
order = sim.argsort(1, descending=True)
for k in (1, 5, 20):
    V[f"knn{k}_r80"] = F_tr[order[:, :k]].mean(1)
V["blend0.5_knn5_r80"] = 0.5 * V["knn5_r80"] + 0.5 * ridge_apply(m, x80, gam_star[0.1])
# diagnostics that use test images (ceilings, not methods)
V["diag_oracle"] = F_te
V["diag_snaptest"] = F_te[(x80 @ nz(F_te, dim=1).T).argmax(1)]

print(f"\ntest conditions (200 images){'':<8}{'norm':>8}{'top1':>8}{'top5':>8}{'cos':>8}{'ccos':>8}{'spread':>8}")
Fte_c = nz(F_te - mu_y, dim=1)
for name, c in V.items():
    t1, t5 = retr(c, F_te)
    cos = float((nz(c, dim=1) * nz(F_te, dim=1)).sum(1).mean())
    scale = sF / c.norm(dim=1, keepdim=True)                       # compare shapes on the F norm scale
    ccos = float((nz(c * scale - mu_y, dim=1) * Fte_c).sum(1).mean())
    spread = float((((c * scale - mu_y) ** 2).sum(1).mean() / var_y) ** 0.5)
    print(f"{name:<36}{float(c.norm(dim=1).mean()):>8.2f}{t1:>8.3f}{t5:>8.3f}{cos:>8.3f}{ccos:>8.3f}{spread:>8.2f}")
    np.save(out / f"{name}.npy", c.float().numpy())
print("gamma (variance match on left-out folds):", {k: round(v, 2) for k, v in gam_star.items()})
