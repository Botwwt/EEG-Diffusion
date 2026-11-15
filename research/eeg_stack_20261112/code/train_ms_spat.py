"""Multi-subject multi-window 4-orientation VAE spatial-contour contrastive BrainMLP (train_ms_spat.py).

Two modes:
  1. --mode build:
     Builds a 1024-dim coarse 16x16 layout + 4-orientation (0, 45, 90, 135 deg) finite-difference contour target S_train [16540, 1024]
     strictly from training-set 4x16x16 SDXL-VAE latents (emb_train.pt['vae']), saving PCA basis + stats to data/prep/spat1024.pt.
     No evaluation-network weights and no test-image statistics are used in the basis.

  2. --mode train --tag <tag> --t0 <t0> --t1 <t1> --seed <seed>:
     Trains a 10-subject BrainEncoder (per-subject MSLinear input + shared BrainMLP trunk) with symmetric InfoNCE
     against S_train (12 joint epochs + 10 per-subject input-layer FT epochs), exporting plain single-subject BrainEncoder
     checkpoints to runs/enc/<tag>/sub-XX/fold-1_seed<seed>/brain_model.pt so aw_tta.py adapts them seamlessly.
"""
import argparse
import copy
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import HIVE, PREP, RUNS, write_json

nz = F.normalize


def steerable_raw_blocks(v16):
    """Extract coarse 16x16 layout + 4-orientation finite-difference contour blocks from [N, 4, 16, 16] SDXL-VAE latents."""
    b0 = v16.flatten(1)
    bx = (v16[:, :, :, 1:] - v16[:, :, :, :-1]).flatten(1)
    by = (v16[:, :, 1:, :] - v16[:, :, :-1, :]).flatten(1)
    bd1 = (v16[:, :, 1:, 1:] - v16[:, :, :-1, :-1]).flatten(1)
    bd2 = (v16[:, :, 1:, :-1] - v16[:, :, :-1, 1:]).flatten(1)
    return [b0, bx, by, bd1, bd2]


BLOCK_WEIGHTS = [1.00, 0.85, 0.85, 0.65, 0.65]


def apply_spat_proj(v16, sp_pack):
    blks = steerable_raw_blocks(v16)
    parts = []
    for b, mu, rms, w in zip(blks, sp_pack["mu"], sp_pack["rms"], BLOCK_WEIGHTS):
        parts.append(w * (b - mu.to(b.device)) / rms)
    return torch.cat(parts, dim=1) @ sp_pack["P_spat"].to(v16.device)


p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["build", "train"], required=True)
p.add_argument("--tag", default="ms_spat_w0_150")
p.add_argument("--t0", type=int, default=0)
p.add_argument("--t1", type=int, default=150)
p.add_argument("--seed", type=int, default=2025)
p.add_argument("--epochs", type=int, default=12)
p.add_argument("--ft_epochs", type=int, default=10)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--ft_lr", type=float, default=5e-4)
p.add_argument("--bs", type=int, default=1024)
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"

if args.mode == "build":
    out_pt = PREP / "spat1024.pt"
    v_tr = torch.load(PREP / "emb_train.pt", weights_only=True)["vae"].float().to(dev).view(-1, 4, 16, 16)
    v_te = torch.load(PREP / "emb_test.pt", weights_only=True)["vae"].float().to(dev).view(-1, 4, 16, 16)

    blks_tr = steerable_raw_blocks(v_tr)
    mu_list = [b.mean(0, keepdim=True) for b in blks_tr]
    rms_list = [float((b - m).pow(2).sum(1).mean().sqrt()) for b, m in zip(blks_tr, mu_list)]
    D_tr = torch.cat([w * (b - m) / r for b, m, r, w in zip(blks_tr, mu_list, rms_list, BLOCK_WEIGHTS)], dim=1)

    torch.manual_seed(20261115)
    _, S_vals, V_p = torch.pca_lowrank(D_tr, q=1056, center=False, niter=5)
    P_spat = V_p[:, :1024].contiguous()

    pack = dict(
        mu=[m.cpu() for m in mu_list],
        rms=rms_list,
        weights=BLOCK_WEIGHTS,
        P_spat=P_spat.cpu(),
    )
    pack["S_train"] = apply_spat_proj(v_tr, pack).cpu()
    pack["S_test"] = apply_spat_proj(v_te, pack).cpu()
    torch.save(pack, out_pt)
    expl = float(S_vals[:1024].pow(2).sum() / D_tr.pow(2).sum())
    print(f"Built {out_pt} | D={D_tr.shape[1]} -> 1024 | explained_var={expl:.4f}", flush=True)
    sys.exit(0)

subs = list(range(1, 11))
S = len(subs)
T_len = args.t1 - args.t0
if all((RUNS / "enc" / args.tag / f"sub-{s:02d}" / f"fold-1_seed{args.seed}" / "brain_model.pt").exists() for s in subs):
    print(f"already done: {args.tag} seed {args.seed}", flush=True)
    sys.exit(0)

sys.path.insert(0, str(HIVE))
import transformers  # noqa: E402
from main.models_brain import BrainEncoder  # noqa: E402

transformers.set_seed(args.seed)
sp_pack = torch.load(PREP / "spat1024.pt", map_location="cpu", weights_only=True)
S_tr_u = nz(sp_pack["S_train"].float().to(dev), dim=1)
S_te_u = nz(sp_pack["S_test"].float().to(dev), dim=1)

X = torch.stack([torch.from_numpy(np.load(PREP / f"sub-{s:02d}_train.npy")[..., args.t0:args.t1]).mean(1) for s in subs]).to(dev)
XT = torch.stack([torch.from_numpy(np.load(PREP / f"sub-{s:02d}_test.npy")[..., args.t0:args.t1]).mean(1) for s in subs]).to(dev)
n_tr = X.shape[1]


class MSLinear(torch.nn.Module):
    def __init__(self, base, n):
        super().__init__()
        self.lin = torch.nn.ModuleList([copy.deepcopy(base) for _ in range(n)])
        self.sid = None

    def forward(self, x):
        out = torch.empty(len(x), self.lin[0].out_features, device=x.device, dtype=torch.promote_types(x.dtype, self.lin[0].weight.dtype))
        for j in torch.unique(self.sid).tolist():
            m = self.sid == j
            out[m] = self.lin[j](x[m]).to(out.dtype)
        return out


class SpatBrainCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.brain_model = BrainEncoder(backbone="brain_mlp", brain_channels=17, brain_sequence_length=T_len,
                                        hidden_size=1024, embedding_dim=1440, dropout=0.1)
        self.brain_model.backbone.proj_in = MSLinear(self.brain_model.backbone.proj_in, S)
        self.logit_scale = torch.nn.Parameter(torch.tensor(2.6592))


model = SpatBrainCLIP().to(dev)


def make_opt(params_model, lr, total, only=None):
    decay, no_decay = [], []
    for n, q in params_model.named_parameters():
        if q.requires_grad and (only is None or only(n)):
            (no_decay if (n.endswith("bias") or ".norm." in n or n == "logit_scale") else decay).append(q)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.995), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 0.5 * (1.0 + math.cos(math.pi * s / total))))
    return opt, sched


def step(m, opt, sched, idx, sid):
    m.brain_model.backbone.proj_in.sid = sid
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = m.brain_model.backbone(X[sid, idx]).float()
        z = nz(z - z.mean(0, keepdim=True), dim=1)
        scale = m.logit_scale.exp().clamp(max=100.0)
        logits = scale * (z @ S_tr_u[idx].T)
        tgt_b = torch.arange(len(idx), device=dev)
        loss = 0.5 * (F.cross_entropy(logits, tgt_b) + F.cross_entropy(logits.T, tgt_b))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    return float(loss)


g = torch.Generator(device=dev).manual_seed(args.seed)
spr = math.ceil(n_tr / args.bs)
opt, sched = make_opt(model, args.lr, spr * S * args.epochs)
t_start = time.time()
model.train()
for _ in range(args.epochs):
    off = torch.randint(0, S, (n_tr,), device=dev, generator=g)
    for r in range(S):
        perm = torch.randperm(n_tr, device=dev, generator=g)
        for b in range(spr):
            j = perm[b * args.bs:(b + 1) * args.bs]
            step(model, opt, sched, j, (off[j] + r) % S)
joint_s = time.time() - t_start


@torch.no_grad()
def encode(m, x, j):
    m.eval()
    m.brain_model.backbone.proj_in.sid = torch.full((len(x),), j, device=dev)
    return m.brain_model.backbone(x).float()


def export(m, j):
    sd = {}
    for k, v in m.brain_model.state_dict().items():
        if ".proj_in.lin." in k:
            pre, rest = k.split(".proj_in.lin.")
            jj, name = rest.split(".", 1)
            if int(jj) == j:
                sd[f"{pre}.proj_in.{name}"] = v.cpu()
        else:
            sd[k] = v.cpu()
    return sd


tgt = torch.arange(200, device=dev)
t1s = []
for j, s in enumerate(subs):
    m = copy.deepcopy(model)
    m.train()
    m.requires_grad_(False)
    m.brain_model.backbone.proj_in.lin[j].requires_grad_(True)
    o2, s2 = make_opt(m, args.ft_lr, spr * args.ft_epochs, lambda n: f".proj_in.lin.{j}." in n)
    for _ in range(args.ft_epochs):
        perm = torch.randperm(n_tr, device=dev, generator=g)
        for b in range(spr):
            idx = perm[b * args.bs:(b + 1) * args.bs]
            step(m, o2, s2, idx, torch.full((len(idx),), j, device=dev))
    out = RUNS / "enc" / args.tag / f"sub-{s:02d}" / f"fold-1_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    c80 = encode(m, XT[j], j)
    c80_c = nz(c80 - c80.mean(0, keepdim=True), dim=1)
    sim = c80_c @ S_te_u.T
    t1 = float((sim.argmax(1) == tgt).float().mean())
    t1s.append(t1)
    met = dict(args=dict(vars(args), sub=s, chan="correct", inorm=0, mode="r4", fold=-1, target="spat1024"),
               joint_seconds=joint_s, test_spat200_top1=t1)
    np.save(out / "cond_test_r80.npy", c80.cpu().numpy())
    torch.save(export(m, j), out / "brain_model.pt")
    write_json(out / "metrics.json", met)
    del m

print(f"{args.tag} seed{args.seed} [t={args.t0}:{args.t1}] done in {time.time() - t_start:.0f}s | mean spat-200 top1={np.mean(t1s):.4f}", flush=True)
