"""Per-image semantic reliability of the EEG condition (label-free), for confidence-gated img2img strength (gen.py --smap sem<d>).

For each test image the 80 repeats are split into 10 disjoint groups (same permutation as condscreen.py bag10), each group is
inverse-variance averaged and passed through the O6 encoders (0-250, 0-150, 0-100 sample windows, seed 2025). Group outputs are
centred over the 200 test images; the reliability of an image is the mean pairwise cosine between its 10 group outputs, averaged
over the three encoders, then z-scored within the subject. Analogy: a Kalman / inverse-variance gain, where the img2img strength
is the weight of the semantic (generated) estimate against the low-level start image. Labels are not used.

python semrel.py <subs>   -> runs/rel/semz_sub-XX.npy [200]
"""
import sys

import numpy as np
import torch

from common import HIVE, PREP, RUNS

sys.path.insert(0, str(HIVE))
from main.models_brain import BrainEncoder  # noqa: E402

nz = torch.nn.functional.normalize


def ivw(x):
    tot = x.sum(1, keepdim=True)
    v = (x - (tot - x) / (x.shape[1] - 1)).pow(2).mean(dim=(2, 3))
    w = 1 / v
    return ((w / w.sum(1, keepdim=True))[:, :, None, None] * x).sum(1)


for s in [int(v) for v in sys.argv[1].split(",")]:
    x = f"{s:02d}"
    xt = torch.from_numpy(np.load(PREP / f"sub-{x}_test.npy")).float()
    perm = torch.randperm(80, generator=torch.Generator().manual_seed(7 + s))
    groups = [ivw(xt[:, perm[i * 8:(i + 1) * 8]]) for i in range(10)]
    rel = []
    for tag in ["t0_r4", "t0_r4_w150", "t0_r4_w100"]:
        d = RUNS / "enc" / tag / f"sub-{x}" / "fold-1_seed2025"
        a_ = __import__("json").loads((d / "metrics.json").read_text())["args"]
        m = BrainEncoder(backbone="brain_mlp", brain_channels=17, brain_sequence_length=a_["t1"] - a_["t0"], hidden_size=1024, embedding_dim=1440, dropout=0.1)
        m.load_state_dict(torch.load(d / "brain_model.pt", weights_only=True))
        m.eval()
        with torch.no_grad():
            o = torch.stack([m.backbone(g[..., a_["t0"]:a_["t1"]]).float() for g in groups])   # [10, 200, 1024]
        o = nz(o - o.mean(1, keepdim=True), dim=2)
        c = torch.einsum("gid,hid->igh", o, o)                                             # [200, 10, 10]
        rel.append(((c.sum((1, 2)) - 10) / 90).numpy())
    r = np.mean(rel, 0)
    z = (r - r.mean()) / r.std()
    (RUNS / "rel").mkdir(exist_ok=True)
    np.save(RUNS / "rel" / f"semz_sub-{x}.npy", z.astype(np.float32))
    print(f"sub-{x} semantic split-group reliability mean {r.mean():.4f} sd {r.std():.4f}; corr between encoders "
          f"{np.corrcoef(rel)[0, 1]:.3f} {np.corrcoef(rel)[0, 2]:.3f} {np.corrcoef(rel)[1, 2]:.3f}", flush=True)
