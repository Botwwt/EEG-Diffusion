"""Joint low-level decoder: EEG -> SDXL-VAE latents of the 128-px image (4x16x16), shared by the ten subjects.

Per-subject input layer (Linear 4250 -> H) on per-subject z-scored EEG (4-repeat mean), shared residual MLP trunk,
linear read-out to the 1024 standardised latent dimensions, MSE loss. One epoch = ten rounds; in each round every
training image appears once, paired with a different subject (as train_ms.py). The number of epochs is chosen on
held-out concepts (fold 0: train on the other folds, pick the epoch with the best centred latent correlation), then
the model is refitted on all training concepts for that many epochs. Optional per-subject fine-tuning of the input
layer afterwards. A per-subject plain ridge (same inputs, ridge weight chosen on fold 0) is fitted as reference.

python low_ms.py --tag lowms1 [--hidden 2048 --layers 2 --dropout 0.3 --lr 1e-3 --wd 0.05 --max_epochs 40]
  -> runs/low/sub-XX/<tag>_n0.8.npy (decoded start images, spread 0.8 as low.py) and a summary line per subject
"""
import argparse
import json
import math

import numpy as np
import torch

from common import PREP, RUNS, TURBO, concept_folds, write_json

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--hidden", type=int, default=2048)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--in_dropout", type=float, default=0.0, help="dropout on the EEG input features")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=0.05)
p.add_argument("--bs", type=int, default=512)
p.add_argument("--max_epochs", type=int, default=40)
p.add_argument("--epochs", type=int, default=0, help=">0: skip the selection on fold 0")
p.add_argument("--ft_epochs", type=int, default=0)
p.add_argument("--spread", type=float, default=0.8)
p.add_argument("--seed", type=int, default=2025)
p.add_argument("--space", default="z", choices=["z", "raw"], help="z: every latent dimension standardised; raw: one global scale, "
               "so high-variance (coarse layout) dimensions dominate the loss and the correlation, as in the ridge of low.py")
p.add_argument("--no_images", type=int, default=0)
args = p.parse_args()
dev = "cuda"
subs = list(range(1, 11))
S = len(subs)
torch.manual_seed(args.seed)
g = torch.Generator(device=dev).manual_seed(args.seed)

# ------------------------------------------------------------ data
X, XT = [], []
for s in subs:
    xm = torch.from_numpy(np.load(PREP / f"sub-{s:02d}_train.npy")).mean(1).flatten(1).to(dev)   # [16540, 4250]
    xt = torch.from_numpy(np.load(PREP / f"sub-{s:02d}_test.npy")).mean(1).flatten(1).to(dev)    # [200, 4250]
    mu, sd = xm.mean(0), xm.std(0)
    X.append((xm - mu) / sd)
    XT.append((xt - mu) / sd)
X, XT = torch.stack(X), torch.stack(XT)                                                          # [S, N, D], [S, 200, D]
Y = torch.load(PREP / "emb_train.pt", weights_only=True)["vae"].float().to(dev)                    # [16540, 1024]
YT = torch.load(PREP / "emb_test.pt", weights_only=True)["vae"].float().to(dev)
ym, ys = Y.mean(0), Y.std(0)
if args.space == "raw":
    ys = Y.std(0).pow(2).mean().sqrt().expand_as(ys).clone()          # one global scale for all dimensions
Yz = (Y - ym) / ys
N, D = X.shape[1], X.shape[2]
concept = torch.from_numpy(np.load(PREP / "train_concept.npy"))
fold = torch.from_numpy(concept_folds(1654, 5))[concept].to(dev)


def rowcorr(a, b):
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)).mean().item()


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        H = args.hidden
        self.inp = torch.nn.ModuleList([torch.nn.Linear(D, H) for _ in range(S)])
        self.ind = torch.nn.Dropout(args.in_dropout)
        self.blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.LayerNorm(H), torch.nn.GELU(), torch.nn.Dropout(args.dropout), torch.nn.Linear(H, H))
                                           for _ in range(args.layers)])
        self.out = torch.nn.Sequential(torch.nn.LayerNorm(H), torch.nn.Dropout(args.dropout), torch.nn.Linear(H, Y.shape[1]))

    def forward(self, x, sid):
        x = self.ind(x)
        h = torch.empty(len(x), self.inp[0].out_features, device=x.device)
        for j in torch.unique(sid).tolist():
            m = sid == j
            h[m] = self.inp[j](x[m]).float()
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


def fit(tr_idx, epochs, ev_idx=None, net=None, only_sub=None, lr=None):
    """Train on images tr_idx (all subjects, or only `only_sub`); with ev_idx, return the per-epoch held-out correlation."""
    net = net or Net().to(dev)
    params = [q for n, q in net.named_parameters() if only_sub is None or n.startswith(f"inp.{only_sub}.")]
    opt = torch.optim.AdamW(params, lr=lr or args.lr, weight_decay=args.wd)
    n = len(tr_idx)
    spr = math.ceil(n / args.bs)
    rounds = 1 if only_sub is not None else S
    total = spr * rounds * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(t, total) / total)))
    curve = []
    for ep in range(epochs):
        net.train()
        off = torch.randint(0, S, (n,), device=dev, generator=g)
        for r in range(rounds):
            perm = torch.randperm(n, device=dev, generator=g)
            for b in range(spr):
                j = perm[b * args.bs:(b + 1) * args.bs]
                i = tr_idx[j]
                sid = torch.full_like(j, only_sub) if only_sub is not None else (off[j] + r) % S
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = torch.nn.functional.mse_loss(net(X[sid, i], sid).float(), Yz[i])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
        if ev_idx is not None:
            curve.append(np.mean([rowcorr(predict(net, X[j_, ev_idx], j_), Yz[ev_idx]) for j_ in range(S)]))
    return net, curve


@torch.no_grad()
def predict(net, x, j):
    net.eval()
    return torch.cat([net(x[k:k + 4096], torch.full((len(x[k:k + 4096]),), j, device=dev)).float() for k in range(0, len(x), 4096)])


# ------------------------------------------------------------ epoch selection on held-out concepts (fold 0)
tr0, ho0 = torch.nonzero(fold != 0).squeeze(1), torch.nonzero(fold == 0).squeeze(1)
info = dict(args=vars(args))
if args.epochs > 0:
    E = args.epochs
else:
    _, curve = fit(tr0, args.max_epochs, ho0)
    E = int(np.argmax(curve)) + 1
    info["fold0_curve"] = [round(float(c), 4) for c in curve]
    print(f"{args.tag}: held-out-concept centred latent correlation by epoch: " + " ".join(f"{c:.3f}" for c in curve) + f" -> {E} epochs", flush=True)

# ridge reference per subject (plain, weight chosen on fold 0)
ridge_te = {}
for j, s in enumerate(subs):
    A = X[j, tr0]
    G_, B_ = A.T @ A / len(tr0), A.T @ Yz[tr0] / len(tr0)
    e, V = torch.linalg.eigh(G_.double())
    Pm = V.T @ B_.double()
    best = (-1, None)
    for lam in (0.3, 1.0, 3.0, 10.0):
        Wl = V @ (Pm / (e + lam * e.mean())[:, None])
        c = rowcorr(X[j, ho0].double() @ Wl, Yz[ho0].double())
        best = max(best, (c, lam), key=lambda t: t[0])
    Af = X[j]
    G_, B_ = (Af.T @ Af / N).double(), (Af.T @ Yz / N).double()
    e, V = torch.linalg.eigh(G_)
    Wf = V @ ((V.T @ B_) / (e + best[1] * e.mean())[:, None])
    ridge_te[s] = (XT[j].double() @ Wf).float()

# ------------------------------------------------------------ final fit on all training concepts
net, _ = fit(torch.arange(N, device=dev), E)
res = {}
preds = {}
for j, s in enumerate(subs):
    pz = predict(net, XT[j], j)
    if args.ft_epochs:
        import copy
        n2 = copy.deepcopy(net)
        n2, _ = fit(torch.arange(N, device=dev), args.ft_epochs, net=n2, only_sub=j, lr=args.lr * 0.3)
        pz = predict(n2, XT[j], j)
    preds[s] = pz
    ys_dim = Y.std(0)                                                       # saved in per-dimension z units for decode_low.py
    np.save(RUNS / "low" / f"sub-{s:02d}" / f"{args.tag}_pred_z.npy", (pz * ys / ys_dim).cpu().numpy())
    yz_te = (YT - ym) / ys
    res[s] = dict(net=rowcorr(pz, yz_te), ridge=rowcorr(ridge_te[s], yz_te), avg=rowcorr(pz / pz.std() + ridge_te[s] / ridge_te[s].std(), yz_te))
print(f"{args.tag} | test centred latent correlation (z-scored latents)  net / ridge / net+ridge average")
for s in subs:
    print(f"  sub-{s:02d}  {res[s]['net']:.4f}  {res[s]['ridge']:.4f}  {res[s]['avg']:.4f}")
print(f"  mean    {np.mean([r['net'] for r in res.values()]):.4f}  {np.mean([r['ridge'] for r in res.values()]):.4f}  {np.mean([r['avg'] for r in res.values()]):.4f}", flush=True)
info.update(epochs=E, test=res)
write_json(RUNS / "low" / f"{args.tag}_summary.json", info)

if not args.no_images:
    from diffusers import AutoencoderKL  # noqa: E402
    vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval()
    vae.enable_slicing()
    del X
    torch.cuda.empty_cache()
    rms_real = float(((Y - ym) ** 2).sum(1).mean().sqrt())
    for s in subs:
        for kind, pz in (("", preds[s]), ("avg", preds[s] / preds[s].std() + ridge_te[s] / ridge_te[s].std())):
            dev_ = pz * ys                                                       # back to latent units, deviation from the mean
            dev_ = dev_ * (args.spread * rms_real / float(dev_.pow(2).sum(1).mean().sqrt()))   # label-free spread normalisation
            lat = (ym + dev_).view(-1, 4, 16, 16) / vae.config.scaling_factor
            with torch.no_grad():
                im = torch.cat([(vae.decode(lat[k:k + 20]).sample / 2 + 0.5).clamp(0, 1).cpu() for k in range(0, len(lat), 20)])
            np.save(RUNS / "low" / f"sub-{s:02d}" / f"{args.tag}{'_' + kind if kind else ''}_n{args.spread}.npy", im.numpy().astype(np.float16))
    print("IMAGES_DONE", flush=True)
