"""Low-level branch: a blurry start image predicted from EEG by closed-form ridge (idea of MindEye / Perceptogram).

python low.py <sub>  ->  runs/low/sub-XX/<variant>.npy  ([200, 3, h, h] in [0, 1]) and a table of their own metrics

Targets
  vae  published SDXL-VAE latents of the 128-px image (4x16x16, scaled), decoded with the VAE
  pix  the image itself at 32x32 px
Repeat correction (rc): training inputs average 4 repeats, test inputs 80. With G = X'X from 4-repeat inputs and S the
single-trial noise covariance (from the spread of the 4 repeats around their mean, training data only),
G - N*S*(1/4 - 1/80) estimates what X'X would be for 80-repeat inputs; the cross term X'Y is unaffected by noise.
The ridge weight is chosen on held-out training concepts (4-repeat inputs); the test set is never used for fitting.
"""
import json
import sys

import numpy as np
import torch
from PIL import Image

from common import IMG, PREP, RUNS, TURBO, concept_folds
from metrics import load_gt, score

sub = int(sys.argv[1])
dev = "cuda"
out = RUNS / "low" / f"sub-{sub:02d}"
out.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------ data
x_tr = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_train.npy")).to(dev)          # [16540, 4, 17, 250]
x_te = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_test.npy")).to(dev).mean(1)   # [200, 17, 250]
N = len(x_tr)
xm = x_tr.mean(1).flatten(1).double()
mu, sd = xm.mean(0), xm.std(0)
X = (xm - mu) / sd
Xt = (x_te.flatten(1).double() - mu) / sd
S = torch.zeros(X.shape[1], X.shape[1], dtype=torch.float64, device=dev)              # single-trial noise covariance
for i in range(0, N, 2000):
    d = ((x_tr[i:i + 2000] - x_tr[i:i + 2000].mean(1, keepdim=True)).flatten(2).double() / sd).flatten(0, 1)
    S += d.T @ d
S /= 3 * N
del x_tr
concept = torch.from_numpy(np.load(PREP / "train_concept.npy"))
fold = torch.from_numpy(concept_folds(1654, 5))[concept].to(dev)

pix_f = PREP / "pix32_train.npy"
if not pix_f.exists():
    ids = json.loads((PREP / "train_ids.json").read_text())["ids"]
    files = {f.stem: f for f in (IMG / "training_images").glob("*/*.jpg")}
    arr = np.zeros((N, 3, 32, 32), np.uint8)
    ok = np.ones(N, bool)
    for j, i in enumerate(ids):
        try:
            im = Image.open(files[i]).convert("RGB")
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((32, 32), Image.BILINEAR)
            arr[j] = np.asarray(im).transpose(2, 0, 1)
        except OSError:
            ok[j] = False
    np.save(pix_f, arr)
    np.save(PREP / "pix32_train_ok.npy", ok)
pix = torch.from_numpy(np.load(pix_f)).to(dev).double().flatten(1) / 255
pix_ok = torch.from_numpy(np.load(PREP / "pix32_train_ok.npy")).to(dev)
TARGETS = {"vae": torch.load(PREP / "emb_train.pt", weights_only=True)["vae"].to(dev).double(), "pix": pix}
lat_te = torch.load(PREP / "emb_test.pt", weights_only=True)["vae"].to(dev).double()
print(f"sub-{sub:02d}: {N} training images ({int((~pix_ok).sum())} unreadable, left out of the pixel target)", flush=True)

LAMS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]   # relative to the mean eigenvalue of X'X / N


def solve(G, B, lam, n, rc):
    """Ridge with the same absolute penalty for both versions. rc: replace the input second moment (signal + noise/4) by
    signal + noise/80, where signal = PSD part of (second moment - noise/4)."""
    M = G / n
    pen = lam * float(torch.trace(M)) / M.shape[0]
    if rc:
        e, V = torch.linalg.eigh(M - S / 4)
        M = (V * e.clamp_min(0)) @ V.T + S / 80
    return torch.linalg.solve(M + pen * torch.eye(M.shape[0], dtype=M.dtype, device=M.device), B / n)


def rowcorr(a, b):
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return float(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))).mean())


W = {}
for name, Y in TARGETS.items():
    keep = pix_ok if name == "pix" else torch.ones(N, dtype=torch.bool, device=dev)
    my = Y[keep].mean(0)
    Yc = Y - my
    cv = {lam: [] for lam in LAMS}
    for k in range(5):
        a, b = keep & (fold != k), keep & (fold == k)
        G, B = X[a].T @ X[a], X[a].T @ Yc[a]
        e, V = torch.linalg.eigh(G / int(a.sum()))
        P = V.T @ (B / int(a.sum()))
        for lam in LAMS:
            pred = X[b] @ (V @ (P / (e + lam * float(e.mean()))[:, None]))
            cv[lam].append(rowcorr(pred, Yc[b]))
    best = max(LAMS, key=lambda lam: np.mean(cv[lam]))
    print(f"[{name}] held-out-concept correlation of the centred prediction by ridge weight: "
          + "  ".join(f"{lam}:{np.mean(cv[lam]):.4f}" for lam in LAMS) + f"  -> {best}", flush=True)
    G, B, n = X[keep].T @ X[keep], X[keep].T @ Yc[keep], int(keep.sum())
    W[name] = dict(my=my, lam=best, plain=solve(G, B, best, n, False), rcp=solve(G, B, best, n, True),
                   rms=float(Yc[keep].pow(2).sum(1).mean().sqrt()))

# ------------------------------------------------------------ test predictions -> images
from diffusers import AutoencoderKL  # noqa: E402

vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval()


@torch.no_grad()
def decode(lat):
    z = lat.float().view(-1, 4, 16, 16) / vae.config.scaling_factor
    return (vae.decode(z).sample / 2 + 0.5).clamp(0, 1).cpu()


ids, gt = load_gt()
V = {}
# name -> (target, weights, spread): spread None keeps the ridge output; a number rescales the deviations from the mean target
# so that their RMS over the test batch is that fraction of the RMS deviation of real training targets (label-free).
CFG = {"vae_plain": ("vae", "plain", None), "vae_plain_n0.6": ("vae", "plain", 0.6), "vae_rcp": ("vae", "rcp", None),
       "vae_rcp_n0.45": ("vae", "rcp", 0.45), "vae_rcp_n0.6": ("vae", "rcp", 0.6), "vae_rcp_n0.8": ("vae", "rcp", 0.8), "vae_rcp_n1.0": ("vae", "rcp", 1.0), "vae_rcp_n1.2": ("vae", "rcp", 1.2)}
for name, (tgt, wn, spread) in CFG.items():
    dev_ = Xt @ W[tgt][wn]
    nat = float(dev_.pow(2).sum(1).mean().sqrt()) / W[tgt]["rms"]
    if spread is not None:
        dev_ = dev_ * (spread / nat)
    p = W[tgt]["my"] + dev_
    print(f"{name}: natural spread {nat:.2f} of real targets | test latent correlation {rowcorr(p, lat_te):.4f} "
          f"(centred {rowcorr(dev_, lat_te - W['vae']['my']):.4f})")
    V[name] = decode(p)
print(f"\n{'start image alone':<20}" + "".join(f"{k:>10}" for k in ("pixcorr", "ssim", "alex2", "alex5", "incep", "clip")))
for name, im in V.items():
    np.save(out / f"{name}.npy", im.numpy().astype(np.float16))
    big = torch.nn.functional.interpolate(im.float(), size=512, mode="bicubic").clamp(0, 1)
    m = score(big[None], gt)[0]
    print(f"{name:<20}" + "".join(f"{m[k]:>10.4f}" for k in ("pixcorr", "ssim", "alex2", "alex5", "inception", "clip")), flush=True)
