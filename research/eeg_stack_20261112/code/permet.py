"""Per-image metrics of a dumped run (gen.py --dump 1) on a grid of high-frequency gains, for offline policy analysis.

Every metric of metrics.score is a mean over images of a per-image quantity (two-way identification of image b only
depends on the prediction for b), so a policy that picks a gain per image is scored exactly by indexing this table.

python permet.py <run> [sigma] [gains, comma-separated]  ->  runs/img/<run>/permet_s<sigma>.npz
   arrays <metric>[n_gains, K, 200]; 'gains'
python permet.py <run> <sigma> map:<lo>-<hi>,<lo>-<hi>,...  ->  runs/img/<run>/permap_s<sigma>.npz
   spatial version: the gain at each pixel is lo + (hi - lo) * rho(x, y), rho = split-half reliability map of the
   start-image latent (rel.py low_map, 16 x 16, bilinear to 512 px); arrays <metric>[n_pairs, K, 200]; 'pairs'
"""
import sys

import numpy as np
import torch
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torchvision import transforms as T
from torchvision.transforms.functional import gaussian_blur

from common import RUNS
from metrics import _nets, dev, load_gt

run = sys.argv[1]
sg = float(sys.argv[2]) if len(sys.argv) > 2 else 24.0
spec = sys.argv[3] if len(sys.argv) > 3 else "1,0.9,0.8,0.7,0.6,0.5,0.4,0.3"
spatial = spec.startswith("map:")
gains = [tuple(float(u) for u in v.split("-")) for v in spec[4:].split(",")] if spatial else [float(v) for v in spec.split(",")]
out = RUNS / "img" / run
tgt = out / f"{'permap' if spatial else 'permet'}_s{int(sg)}.npz"
mark = tgt.with_suffix(".running")
if tgt.exists() or mark.exists():   # done, or another worker has it
    print("skip", run, flush=True)
    sys.exit(0)
mark.touch()
rec = torch.from_numpy(np.load(out / "rec_u8.npy"))          # [K, 200, 3, 512, 512] uint8
K, N = rec.shape[:2]
_, gt = load_gt()
nets = _nets()


def rs(n):
    return T.Resize(n, interpolation=T.InterpolationMode.BILINEAR)


@torch.no_grad()
def feats(model, pre, x, layer=None):
    o = []
    for i in range(0, len(x), 50):
        y = model(pre(x[i:i + 50].to(dev)))
        o.append((y if layer is None else y[layer]).float().flatten(1).cpu())
    return torch.cat(o).numpy()


def lowpass(x, s):
    ks = int(6 * s) | 1
    return torch.cat([gaussian_blur(x[i:i + 50].to(dev), [ks, ks], [s, s]).cpu() for i in range(0, len(x), 50)])


gt_f = {k: feats(m, p, gt, l_) for k, (m, p, l_) in nets.items()}
gt425 = rs(425)(gt)
gt_gray = rgb2gray(gt425.permute(0, 2, 3, 1).numpy())
gflat = gt425.reshape(N, -1).numpy()


def per_image(x):
    """x [N, 3, 512, 512] float -> dict metric -> [N]."""
    r425 = rs(425)(x)
    rf = r425.reshape(N, -1).numpy()
    m = {"pixcorr": np.array([np.corrcoef(gflat[i], rf[i])[0, 1] for i in range(N)])}
    rg = rgb2gray(r425.permute(0, 2, 3, 1).numpy())
    m["ssim"] = np.array([structural_similarity(rg[i], gt_gray[i], gaussian_weights=True, sigma=1.5,
                                                use_sample_covariance=False, data_range=1.0) for i in range(N)])
    for name, (mod, pre, layer) in nets.items():
        f = feats(mod, pre, x, layer)
        if name == "effnet":
            rc, pc = gt_f[name] - gt_f[name].mean(1, keepdims=True), f - f.mean(1, keepdims=True)
            m[name] = 1 - (rc * pc).sum(1) / (np.linalg.norm(rc, axis=1) * np.linalg.norm(pc, axis=1))
        else:
            r = np.corrcoef(gt_f[name], f)[:N, N:]
            m[name] = np.sum(r < np.diag(r), 0) / (N - 1)
    return m


res = {}
if spatial:
    sub_ = int(run[1:3])
    rmap = torch.from_numpy(np.load(RUNS / "rel" / f"sub-{sub_:02d}.npz")["low_map"]).float()[:, None]   # [200, 1, 16, 16]
    rmap = torch.nn.functional.interpolate(rmap, size=512, mode="bilinear", align_corners=False).clamp(0, 1)
for k in range(K):
    x = rec[k].float() / 255
    lg = lowpass(x, sg)
    for gi, b in enumerate(gains):
        bb = (b[0] + (b[1] - b[0]) * rmap) if spatial else b
        y = (lg + bb * (x - lg)).clamp(0, 1)
        for name, v in per_image(y).items():
            res.setdefault(name, np.zeros((len(gains), K, N)))[gi, k] = v
    print(f"{run} sample {k} done", flush=True)
np.savez(tgt, **({"pairs": np.array(gains)} if spatial else {"gains": np.array(gains)}), **res)
mark.unlink()
print(run, "gain", " ".join(f"{b}:{res['ssim'][i].mean():.4f}/{res['clip'][i].mean():.4f}" for i, b in enumerate(gains)), "(ssim/clip)")
