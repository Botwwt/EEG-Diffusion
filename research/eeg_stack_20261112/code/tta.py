"""Split-half contrastive test-time adaptation of the encoder (transductive, label-free; test-time adaptation from domain
adaptation). The 80 repeats of every test image are split at random into two halves; the two half-means of one image are
positives, the other 199 images negatives (symmetric InfoNCE, temperature tau). Only the subject's input layer is
updated, with an anchor lam * mean ||c - c_orig||^2 / ||c_orig||^2 to the unadapted output of the full input. No image,
label or test target is used. Scored by raw 200-way retrieval of F (labels used only for scoring).

python tta.py <tag> <subs> <steps> <lr> <tau> <lam> [save name]     (env TTA_SEED: encoder seed, default 2025)
   -> prints top1/top5 before / after (input = inverse-variance weighted mean of the 80 repeats);
      with a save name: runs/enc/<tag>/sub-XX/fold-1_seed2025/cond_test_<name>.npy and the head-corrected conditions
      runs/cond/ord_<name>/sub-XX.npy (per-subject head, as ivwprep.py)
"""
import copy
import sys

import numpy as np
import torch

from common import HIVE, PREP, ROOT, RUNS

sys.path.insert(0, str(HIVE))
sys.path.insert(0, str(ROOT / "code/EEG-Diffusion/research/eeg"))
from brainhive_bounded_component_v1 import BoundedConditionAdapter  # noqa: E402
from main.models_brain import BrainEncoder  # noqa: E402

tag, subs = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
steps, lr, tau, lam = int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
name = sys.argv[7] if len(sys.argv) > 7 else ""
dev = __import__("os").environ.get("TTA_DEV", "cuda")
SEED = int(__import__("os").environ.get("TTA_SEED", "2025"))   # encoder seed (condscreen.py ensembles several)
nz = torch.nn.functional.normalize
GK, BK = "backbone.block.layers.0.norm.weight", "backbone.block.layers.0.norm.bias"
Fn = nz(torch.from_numpy(np.load(PREP / "F_test.npy")).float(), dim=1).to(dev)
t = torch.arange(200, device=dev)


def ivw(x):
    tot = x.sum(1, keepdim=True)
    v = (x - (tot - x) / (x.shape[1] - 1)).pow(2).mean(dim=(2, 3))
    w = 1 / v
    return ((w / w.sum(1, keepdim=True))[:, :, None, None] * x).sum(1)


def score(c):
    s = nz(c, dim=1) @ Fn.T
    rk = (s > s.diag()[:, None]).sum(1)
    return float((rk == 0).float().mean()), float((rk < 5).float().mean())


res = []
for s in subs:
    x = f"{s:02d}"
    xt = torch.from_numpy(np.load(PREP / f"sub-{x}_test.npy")).float().to(dev)
    d = RUNS / "enc" / tag / f"sub-{x}" / f"fold-1_seed{SEED}"
    sd = torch.load(d / "brain_model.pt", weights_only=True)
    enc = BrainEncoder(backbone="brain_mlp", brain_channels=17, brain_sequence_length=250, hidden_size=1024, embedding_dim=1440, dropout=0.1)
    enc.load_state_dict(sd)
    enc = enc.to(dev).eval()
    full = ivw(xt)
    with torch.no_grad():
        c0 = enc.backbone(full).float()
    before = score(c0)
    m = copy.deepcopy(enc)
    m.requires_grad_(False)
    pin = [p_ for n_, p_ in m.named_parameters() if "proj_in" in n_]
    assert pin, [n_ for n_, _ in m.named_parameters()][:5]
    for p_ in pin:
        p_.requires_grad_(True)
    opt = torch.optim.Adam(pin, lr=lr)
    g = torch.Generator(device=dev).manual_seed(1000 + s)
    for it in range(steps):
        perm = torch.rand(200, 80, device=dev, generator=g).argsort(1)
        a = torch.gather(xt, 1, perm[:, :40, None, None].expand(-1, -1, 17, 250))
        b = torch.gather(xt, 1, perm[:, 40:, None, None].expand(-1, -1, 17, 250))
        za, zb = m.backbone(ivw(a)).float(), m.backbone(ivw(b)).float()
        za, zb = za - za.mean(0), zb - zb.mean(0)
        lg = nz(za, dim=1) @ nz(zb, dim=1).T / tau
        loss = (torch.nn.functional.cross_entropy(lg, t) + torch.nn.functional.cross_entropy(lg.T, t)) / 2
        cf = m.backbone(full).float()
        anc = ((cf - c0).pow(2).sum(1) / c0.pow(2).sum(1)).mean()
        (loss + lam * anc).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        c1 = m.backbone(full).float()
    after = score(c1)
    res.append((before, after))
    msg = f"sub-{x}: top1/top5 {before[0]:.3f}/{before[1]:.3f} -> {after[0]:.3f}/{after[1]:.3f} | anchor {float(anc):.4f} loss {float(loss):.3f}"
    if name and SEED != 2025:
        np.save(d / f"cond_test_{name}.npy", c1.cpu().numpy())
    elif name:
        m = m.cpu()
        with torch.no_grad():
            c1c = c1.cpu()
            head = BoundedConditionAdapter(predict_risk=False).eval()
            head.load_state_dict(torch.load(RUNS / "ord" / f"sub-{x}" / "head.pt", weights_only=True))
            wg, bg = sd[GK].float(), sd[BK].float()      # the head sees the LayerNorm affine of the (unchanged) trunk
            h, _ = head(c1c, wg.expand_as(c1c), bg.expand_as(c1c))
        np.save(d / f"cond_test_{name}.npy", c1c.numpy())
        (RUNS / "cond" / f"ord_{name}").mkdir(parents=True, exist_ok=True)
        np.save(RUNS / "cond" / f"ord_{name}" / f"sub-{x}.npy", h.numpy())
        msg += f" | after head {score(h.to(dev))[0]:.3f}"
    print(msg, flush=True)
r = np.array([[b[0], b[1], a[0], a[1]] for b, a in res])
print(f"mean over {len(subs)}: {r[:, 0].mean():.4f}/{r[:, 1].mean():.4f} -> {r[:, 2].mean():.4f}/{r[:, 3].mean():.4f} "
      f"| top1 better in {(r[:, 2] > r[:, 0]).sum()}, worse in {(r[:, 2] < r[:, 0]).sum()}")
