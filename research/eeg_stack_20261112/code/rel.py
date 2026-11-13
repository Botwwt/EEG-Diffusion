"""Per-image reliability of the test EEG from its own repeats (label-free; no image, no metric network is used).

Split-half Wiener gain: the 80 repeats of a test image are split at random into two halves of 40; both half-means go
through the same decoder. Noise is independent between halves, so for decoded vectors a, b (centred on the test-batch
mean of their half) E[a.b] estimates the signal power and E[(|a|^2 + |b|^2) / 2] signal + noise at 40 repeats.
rho40 = signal / (signal + noise) is the Wiener gain at 40 repeats; Spearman-Brown gives the 80-repeat gain
rho80 = 2 rho40 / (1 + rho40). Averaged over random splits.

Two decoders:
  emb  the joint encoder (the EEG condition that drives the IP adapter)
  low  the repeat-corrected ridge to the SDXL-VAE latent of the start image (same fit as low.py, vae / rcp);
       besides the image-level gain, a map over the 16 x 16 latent positions (signal and total summed over 4 channels)

python rel.py <sub> [encoder tag] [splits]  ->  runs/rel/sub-XX.npz  (emb[200], low[200], low_map[200, 16, 16], emb_id[200])
emb_id: split-half self-identification, fraction of the other 199 half-B conditions less similar to half A than its own.
"""
import sys

import numpy as np
import torch

from common import HIVE, PREP, RUNS, concept_folds

sys.path.insert(0, str(HIVE))
from main.models_brain import BrainEncoder  # noqa: E402

sub = int(sys.argv[1])
tag = sys.argv[2] if len(sys.argv) > 2 else "ms_e10_ft10in"
NSPL = int(sys.argv[3]) if len(sys.argv) > 3 else 32
dev = "cuda"
x = f"{sub:02d}"
g = torch.Generator().manual_seed(20261113 + sub)
xt = torch.from_numpy(np.load(PREP / f"sub-{x}_test.npy")).float().to(dev)          # [200, 80, 17, 250]
splits = [torch.randperm(80, generator=g) for _ in range(NSPL)]


def gain(A, B):
    """A, B: [splits, 200, D] decoded half-means -> per-image rho80 and per-dimension signal / total [200, D]."""
    A, B = A - A.mean(1, keepdim=True), B - B.mean(1, keepdim=True)
    sig = (A * B).mean(0)                         # [200, D]
    tot = ((A * A + B * B) / 2).mean(0)
    r = (sig.sum(1) / tot.sum(1)).clamp(-0.99, 1)
    return (2 * r / (1 + r)).clamp(min=0), sig, tot


# ---------------------------------------------------------------- joint encoder
enc = BrainEncoder(backbone="brain_mlp", brain_channels=17, brain_sequence_length=250, hidden_size=1024, embedding_dim=1440, dropout=0.1)
enc.load_state_dict(torch.load(RUNS / "enc" / tag / f"sub-{x}" / "fold-1_seed2025" / "brain_model.pt", weights_only=True))
enc = enc.to(dev).eval()
with torch.no_grad():
    EA = torch.stack([enc.backbone(xt[:, p[:40]].mean(1)).float() for p in splits])
    EB = torch.stack([enc.backbone(xt[:, p[40:]].mean(1)).float() for p in splits])
r_emb, _, _ = gain(EA, EB)
nz = torch.nn.functional.normalize
ida = []
for a, b in zip(EA, EB):
    s_ = nz(a - a.mean(0), dim=1) @ nz(b - b.mean(0), dim=1).T
    ida.append((s_ < s_.diag()[:, None]).float().sum(1) / 199)
emb_id = torch.stack(ida).mean(0)

# ---------------------------------------------------------------- ridge to the VAE latent (as low.py: vae, rcp)
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
concept = torch.from_numpy(np.load(PREP / "train_concept.npy"))
fold = torch.from_numpy(concept_folds(1654, 5))[concept].to(dev)
Y = torch.load(PREP / "emb_train.pt", weights_only=True)["vae"].to(dev).double()
Yc = Y - Y.mean(0)
LAMS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def rowcorr(a, b):
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return float(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))).mean())


cv = {lam: [] for lam in LAMS}
for k in range(5):
    a, b = fold != k, fold == k
    e, V = torch.linalg.eigh(X[a].T @ X[a] / int(a.sum()))
    P = V.T @ (X[a].T @ Yc[a] / int(a.sum()))
    for lam in LAMS:
        cv[lam].append(rowcorr(X[b] @ (V @ (P / (e + lam * float(e.mean()))[:, None])), Yc[b]))
lam = max(LAMS, key=lambda l_: np.mean(cv[l_]))
M = X.T @ X / N
pen = lam * float(torch.trace(M)) / M.shape[0]
e, V = torch.linalg.eigh(M - S / 4)
M = (V * e.clamp_min(0)) @ V.T + S / 80
W = torch.linalg.solve(M + pen * torch.eye(M.shape[0], dtype=M.dtype, device=dev), X.T @ Yc / N)


def lowpred(xh):
    return (((xh.flatten(1).double() - mu) / sd) @ W).float()


LA = torch.stack([lowpred(xt[:, p[:40]].mean(1)) for p in splits])
LB = torch.stack([lowpred(xt[:, p[40:]].mean(1)) for p in splits])
r_low, sig, tot = gain(LA, LB)
sig_m, tot_m = sig.view(200, 4, 16, 16).sum(1), tot.view(200, 4, 16, 16).sum(1)
rm = (sig_m / tot_m).clamp(-0.99, 1)
low_map = (2 * rm / (1 + rm)).clamp(min=0)
out = RUNS / "rel"
out.mkdir(exist_ok=True)
np.savez(out / f"sub-{x}.npz", emb=r_emb.cpu().numpy(), emb_id=emb_id.cpu().numpy(), low=r_low.cpu().numpy(),
         low_map=low_map.cpu().numpy(), low_sig_map=sig_m.cpu().numpy(), low_tot_map=tot_m.cpu().numpy())
c = lambda u, v: float(np.corrcoef(u.cpu().numpy(), v.cpu().numpy())[0, 1])  # noqa: E731
print(f"sub-{x} ridge weight {lam} | rho80 emb mean {float(r_emb.mean()):.3f} (sd {float(r_emb.std()):.3f}) | low mean "
      f"{float(r_low.mean()):.3f} (sd {float(r_low.std()):.3f}) | split-half id {float(emb_id.mean()):.3f} | "
      f"corr(emb, low) {c(r_emb, r_low):.3f} corr(emb, id) {c(r_emb, emb_id):.3f} | map mean {float(low_map.mean()):.3f}", flush=True)
