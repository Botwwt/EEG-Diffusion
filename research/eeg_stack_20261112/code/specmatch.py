"""Spectral variance matching of the IP condition (closed-form adaptation in front of the generator's IP projection).

Transferred from correlation alignment in domain adaptation (CORAL, Sun & Saenko 2016), applied per principal axis of the
generator's own input distribution: in the PCA basis of the training-image embeddings F_train, the spread of the 200 test
conditions along axis k (sd_h,k, label-free, transductive) is rescaled towards that of F (sd_F,k):
    u_k <- u_k * (sd_F,k / sd_h,k) ** g      for the first K axes (centred on the test-condition mean), then each condition
is rescaled to its original norm. g = 1: full matching; g = 0.5: half way; mode 'down' only shrinks axes with sd_h > sd_F.
The head output concentrates energy on the first axes (about 2x the spread of F on axes 1-4, 0.55x on axes 129-512).

python specmatch.py <subs> <src cond dir> <K> <g> <mode: all|down> -> runs/cond/<src>_sm<K>g<g><mode>/sub-XX.npy
"""
import sys

import numpy as np
import torch

from common import PREP, RUNS

subs = [int(v) for v in sys.argv[1].split(",")]
src, K, g, mode = sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
Ftr = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
_, sv, Vt = torch.linalg.svd(Ftr - Ftr.mean(0), full_matrices=False)
V, sdF = Vt[:K].T, sv[:K] / np.sqrt(len(Ftr) - 1)
out = RUNS / "cond" / f"{src}_sm{K}g{g:g}{mode}"
out.mkdir(parents=True, exist_ok=True)
for s in subs:
    h = torch.from_numpy(np.load(RUNS / "cond" / src / f"sub-{s:02d}.npy")).float()
    m = h.mean(0)
    u = (h - m) @ V
    f = (sdF / u.std(0)) ** g
    if mode == "down":
        f = f.clamp(max=1)
    h2 = h + (u * (f - 1)) @ V.T
    h2 = h2 * h.norm(dim=1, keepdim=True) / h2.norm(dim=1, keepdim=True)
    np.save(out / f"sub-{s:02d}.npy", h2.numpy().astype(np.float32))
    print(f"sub-{s:02d} {out.name}: factor axes 1-4 {f[:4].numpy().round(2)}, mean 5-32 {float(f[4:32].mean()):.2f}, "
          f"33-128 {float(f[32:128].mean()):.2f}, cos to original {float(torch.nn.functional.cosine_similarity(h, h2).mean()):.3f}", flush=True)
