"""Low-level latents from the semantic EEG condition, as a complement to the EEG -> latent decoder (low_ms.py).

  knn   : softmax-weighted mean of the training latents whose fused target F is closest to the condition c
          (centred cosine; training images only, their concepts are disjoint from the test concepts)
  ridge : ridge from the out-of-fold training conditions (4-repeat inputs, encoders that never saw the concept)
          to the z-scored latents, ridge weight chosen on fold 0
Both are compared, alone and blended with the decoder prediction <low tag>_pred_z.npy, on the centred correlation
with the real test latents (z-scored). With --save, the best blend is written as <out tag>_pred_z.npy.

python lowsem.py <encoder tag> <low tag> [--save <out tag> --mode knn|ridge --w <blend weight>]
"""
import argparse

import numpy as np
import torch

from common import PREP, RUNS, concept_folds

p = argparse.ArgumentParser()
p.add_argument("enc")
p.add_argument("low")
p.add_argument("--save", default="")
p.add_argument("--mode", default="")
p.add_argument("--w", type=float, default=0.0)
p.add_argument("--k", type=int, default=20)
p.add_argument("--temp", type=float, default=0.05)
p.add_argument("--knn_pool", type=int, default=1, help="keep only the coarse part of the retrieved latents: average-pool 16x16 by this factor, upsample back")
p.add_argument("--ctest", default="cond_test_r80.npy", help="test-condition file in the encoder's fold-1 directory (ivwprep.py: cond_test_ivw.npy)")
p.add_argument("--space", default="z", choices=["z", "raw"], help="raw: correlations and blends on unstandardised latent deviations")
args = p.parse_args()
nz = torch.nn.functional.normalize
Y = torch.load(PREP / "emb_train.pt", weights_only=True)["vae"].float()
YT = torch.load(PREP / "emb_test.pt", weights_only=True)["vae"].float()
ym, ys = Y.mean(0), Y.std(0)
ys_dim = ys.clone()
if args.space == "raw":
    ys = Y.std(0).pow(2).mean().sqrt().expand_as(ys).clone()
Yz, YTz = (Y - ym) / ys, (YT - ym) / ys
Ftr = nz(torch.from_numpy(np.load(PREP / "F_train.npy")).float(), dim=1)
Fc = nz(Ftr - Ftr.mean(0), dim=1)
concept = torch.from_numpy(np.load(PREP / "train_concept.npy"))
fold = torch.from_numpy(concept_folds(1654, 5))[concept]


def rowcorr(a, b):
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return float(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)).mean())


def knn(c, k=args.k, temp=args.temp):
    q = nz(nz(c, dim=1) - nz(c, dim=1).mean(0), dim=1)
    sim = q @ Fc.T
    v, i = sim.topk(k, dim=1)
    w = torch.softmax(v / temp, dim=1)
    out = (w[:, :, None] * Yz[i]).sum(1)
    if args.knn_pool > 1:   # pooled in latent units, so the coarse layout is kept and fine structure dropped
        lat = (out * ys).view(-1, 4, 16, 16)
        lat = torch.nn.functional.interpolate(torch.nn.functional.avg_pool2d(lat, args.knn_pool), size=16, mode="bilinear", align_corners=False)
        out = lat.flatten(1) / ys
    return out


rows = []
for s in range(1, 11):
    d = RUNS / "enc" / args.enc / f"sub-{s:02d}"
    ct = torch.from_numpy(np.load(d / "fold-1_seed2025" / args.ctest)).float()
    oof = torch.zeros(len(Y), ct.shape[1])
    for k in range(5):
        idx = torch.from_numpy(np.load(d / f"fold{k}_seed2025" / "oof_index.npy"))
        oof[idx] = torch.from_numpy(np.load(d / f"fold{k}_seed2025" / "cond_oof_r4.npy")).float()
    on = nz(oof, dim=1)
    mu, sd = on.mean(0), on.std(0)
    Z, Zt = (on - mu) / sd, (nz(ct, dim=1) - mu) / sd
    tr, ho = fold != 0, fold == 0
    best = None
    for lam in (1.0, 10.0, 100.0, 1000.0):
        W = torch.linalg.solve(Z[tr].T @ Z[tr] + lam * torch.eye(Z.shape[1]), Z[tr].T @ Yz[tr])
        c_ = rowcorr(Z[ho] @ W, Yz[ho])
        best = max(best or (c_, lam), (c_, lam))
    W = torch.linalg.solve(Z.T @ Z + best[1] * torch.eye(Z.shape[1]), Z.T @ Yz)
    pr = Zt @ W
    pk = knn(ct)
    net = torch.from_numpy(np.load(RUNS / "low" / f"sub-{s:02d}" / f"{args.low}_pred_z.npy")).float() * ys_dim / ys   # to this space
    std = lambda z: z / z.std()  # noqa: E731
    r = dict(net=rowcorr(net, YTz), knn=rowcorr(pk, YTz), ridge=rowcorr(pr, YTz))
    for w in (0.25, 0.5, 1.0):
        r[f"net+{w}knn"] = rowcorr(std(net) + w * std(pk), YTz)
        r[f"net+{w}ridge"] = rowcorr(std(net) + w * std(pr), YTz)
    rows.append(r)
    if args.save:
        extra = pk if args.mode == "knn" else pr
        np.save(RUNS / "low" / f"sub-{s:02d}" / f"{args.save}_pred_z.npy", ((std(net) + args.w * std(extra)) * float(net.std()) * ys / ys_dim).numpy())
keys = list(rows[0])
print("mean centred test-latent correlation over 10 subjects:")
for k in keys:
    v = [r[k] for r in rows]
    print(f"  {k:<16} {np.mean(v):.4f}   better than net in {sum(x > r['net'] for x, r in zip(v, rows))}/10")
