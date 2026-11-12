"""Test conditions at the training repeat level: encode 20 disjoint 4-repeat means and average the 20 embeddings
(instead of encoding the 80-repeat mean once), rescale each to the norm of its 80-repeat condition, then apply the
subject's ordinary head.

python cond20x4.py <encoder tag> <sub> [head name: sub-XX (default) or a shared head under runs/ord_<tag>/]
  -> runs/cond/ord_<tag>_e20x4/sub-XX.npy
Prints 200-way top-1 and cos to F for R80 vs 20x4, before and after the head. CPU is enough.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from common import HIVE, PREP, RUNS

sys.path.insert(0, str(HIVE))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eeg"))  # research/eeg of this repository
from brainhive_bounded_component_v1 import BoundedConditionAdapter  # noqa: E402
from main.models_brain import BrainEncoder  # noqa: E402

GK, BK = "backbone.block.layers.0.norm.weight", "backbone.block.layers.0.norm.bias"
tag, sub = sys.argv[1], int(sys.argv[2])
x = f"{sub:02d}"
hname = sys.argv[3] if len(sys.argv) > 3 else f"sub-{x}"
d = RUNS / "enc" / tag / f"sub-{x}" / "fold-1_seed2025"
enc = BrainEncoder(backbone="brain_mlp", brain_channels=17, brain_sequence_length=250, hidden_size=1024, embedding_dim=1440, dropout=0.1)
sd = torch.load(d / "brain_model.pt", weights_only=True)
enc.load_state_dict(sd)
enc.eval()
w, b = sd[GK].float(), sd[BK].float()
xt = torch.from_numpy(np.load(PREP / f"sub-{x}_test.npy")).float()          # [200, 80, 17, 250]
with torch.no_grad():
    c80 = enc.backbone(xt.mean(1)).float()
    c4 = torch.stack([enc.backbone(xt[:, s:s + 4].mean(1)).float() for s in range(0, 80, 4)]).mean(0)
    n4 = float(c4.norm(dim=1).mean())
    c4 = c4 * (c80.norm(dim=1, keepdim=True) / c4.norm(dim=1, keepdim=True))   # averaging shrinks the norm; keep the R80 norm
    head = BoundedConditionAdapter(predict_risk=False).eval()
    head.load_state_dict(torch.load(RUNS / f"ord_{tag}" / hname / "head.pt", weights_only=True))
    h80, _ = head(c80, w.expand_as(c80), b.expand_as(c80))
    h4, _ = head(c4, w.expand_as(c4), b.expand_as(c4))
nz = torch.nn.functional.normalize
Fn = nz(torch.from_numpy(np.load(PREP / "F_test.npy")).float(), dim=1)
t = torch.arange(200)


def rep(c):
    return f"top1 {float(((nz(c, dim=1) @ Fn.T).argmax(1) == t).float().mean()):.3f} cos {float((nz(c, dim=1) * Fn).sum(1).mean()):.3f}"


out = RUNS / "cond" / f"ord_{tag}_e20x4"
out.mkdir(parents=True, exist_ok=True)
np.save(out / f"sub-{x}.npy", h4.numpy())
print(f"sub-{x} | R80 {rep(c80)} -> head {rep(h80)} | 20x4 {rep(c4)} -> head {rep(h4)} | raw 20x4 norm {n4:.2f} (rescaled to {float(c4.norm(dim=1).mean()):.2f}) | head out norm {float(h4.norm(dim=1).mean()):.2f} vs {float(h80.norm(dim=1).mean()):.2f}", flush=True)
