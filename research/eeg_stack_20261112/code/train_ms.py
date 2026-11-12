"""Multi-subject encoder: one BrainMLP trunk shared by the ten subjects, one input layer (Linear 4250 -> 1024) per subject.

All ten subjects saw the same 16,540 training images, so the shared trunk (residual MLP + LayerNorm) gets ten times
the data of a per-subject encoder. Recipe otherwise as train.py (symmetric InfoNCE to the frozen Fusion-Prior target,
AdamW 5e-4, betas (0.9, 0.995), wd 0.05, cosine, batch 1024, clip 1, bf16, dropout 0.1, 4-repeat mean inputs).
One epoch = ten rounds; in each round every training image appears once, paired with a different subject, so a batch
never holds the same image twice (no false negatives in InfoNCE).
Optional --ft_epochs: afterwards fine-tune a copy on each subject alone (lr --ft_lr), one model per subject.

Output per subject, in the layout of train.py (so gen.py / ordh.py / calibration use it unchanged):
runs/enc/<tag>/sub-XX/fold{k}_seed{s}/ brain_model.pt (a plain single-subject BrainEncoder state dict),
metrics.json, cond_test_r80.npy and, for k >= 0, cond_oof_r4.npy + oof_index.npy.

  python train_ms.py --tag ms1 [--fold -1] [--seed 2025] [--epochs 25] [--ft_epochs 0]
"""
import argparse
import copy
import math
import sys
import time

import numpy as np
import torch

from common import HIVE, PREP, PRIOR, RUNS, concept_folds, write_json

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--fold", type=int, default=-1)
p.add_argument("--seed", type=int, default=2025)
p.add_argument("--subs", default="1,2,3,4,5,6,7,8,9,10")
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--bs", type=int, default=1024)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--layers", type=int, default=1, help="residual blocks in the shared trunk (official 1)")
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--aug", default="", choices=["", "subset_u"], help="subset_u: training input = mean of a random subset of the 4 repeats (size uniform 1..4); evaluation unchanged")
p.add_argument("--ft_epochs", type=int, default=0)
p.add_argument("--ft_lr", type=float, default=1e-4)
p.add_argument("--ft_trunk", type=int, default=1, help="0: fine-tune only the subject's input layer")
args = p.parse_args()
subs = [int(s) for s in args.subs.split(",")]
S = len(subs)
dev = "cuda"

sys.path.insert(0, str(HIVE))
import transformers  # noqa: E402
import transformers.models.clip.modeling_clip as _mc  # noqa: E402

if not hasattr(_mc, "clip_loss"):
    def _ce(logits):
        return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
    _mc.clip_loss = lambda sim: (_ce(sim) + _ce(sim.t())) / 2.0
from main.models_adapter import FusionEncoderModel  # noqa: E402
from main.models_brain import BrainEncoder  # noqa: E402
from main.models_clip import BrainCLIPModel, CLIPConfig  # noqa: E402

BrainEncoder.forward = lambda self, brain_signals, subject_ids=None: self.backbone(brain_signals)
transformers.set_seed(args.seed)
nz = torch.nn.functional.normalize

# ---------------------------------------------------------------- data (4-repeat / 80-repeat means only)
if args.aug:   # keep the single repeats (fp16 to fit), means computed on the fly
    X4 = torch.stack([torch.from_numpy(np.load(PREP / f"sub-{s:02d}_train.npy")).half() for s in subs]).to(dev)  # [S, 16540, 4, 17, 250]
    X = None
else:
    X = torch.stack([torch.from_numpy(np.load(PREP / f"sub-{s:02d}_train.npy")).mean(1) for s in subs]).to(dev)  # [S, 16540, 17, 250]
XT = torch.stack([torch.from_numpy(np.load(PREP / f"sub-{s:02d}_test.npy")).mean(1) for s in subs]).to(dev)  # [S, 200, 17, 250]
emb_tr = {k: v.float().to(dev) for k, v in torch.load(PREP / "emb_train.pt", weights_only=True).items()}
F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).to(dev)
F_te = torch.from_numpy(np.load(PREP / "F_test.npy")).to(dev)
concept = torch.from_numpy(np.load(PREP / "train_concept.npy"))
img_fold = torch.from_numpy(concept_folds(1654, 5))[concept]
tr_idx = (torch.arange(len(concept)) if args.fold < 0 else torch.nonzero(img_fold != args.fold).squeeze(1)).to(dev)
ho_idx = (torch.empty(0, dtype=torch.long) if args.fold < 0 else torch.nonzero(img_fold == args.fold).squeeze(1)).to(dev)
n_tr = len(tr_idx)


class MSLinear(torch.nn.Module):
    """One Linear per subject; the subject of every row is set in .sid before the forward pass."""

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


pcfg = FusionEncoderModel.load_config(str(PRIOR), subfolder="fusion_encoder")
cfg = dict(hidden_size=1024, intermediate_size=1024, logit_scale_init_value=2.6592, dropout=args.dropout,
           brain_backbone="brain_mlp", brain_channels=17, brain_sequence_length=250, brain_embedding_dim=1440, proj_meta={})
cfg.update({k: v for k, v in pcfg.items() if k in cfg})
model = BrainCLIPModel(CLIPConfig.from_dict(cfg))
model.vision_model.load_state_dict(FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.state_dict())
model.vision_model.requires_grad_(False)
bb = model.brain_model.backbone
if args.layers != 1:
    from main.models_brain import BrainMLP  # noqa: E402
    model.brain_model.backbone = bb = BrainMLP(17 * 250, 1024, num_layers=args.layers, dropout=args.dropout)
bb.proj_in = MSLinear(bb.proj_in, S)
model.to(dev)


def make_opt(params_model, lr, total, only=None):
    decay, no_decay = [], []
    for n, q in params_model.named_parameters():
        if q.requires_grad and (only is None or only(n)):
            (no_decay if (n.endswith("bias") or ".norm." in n) else decay).append(q)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.995), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 0.5 * (1.0 + math.cos(math.pi * s / total))))
    return opt, sched


def xin(sid, idx, train=True):
    if not args.aug:
        return X[sid, idx]
    e = X4[sid, idx].float()                                                   # [B, 4, 17, 250]
    if not train:
        return e.mean(1)
    k = torch.randint(1, 5, (len(idx),), device=dev, generator=g)
    m_ = (torch.rand(len(idx), 4, device=dev, generator=g).argsort(1).argsort(1) < k[:, None]).float()
    return (e * m_[:, :, None, None]).sum(1) / k[:, None, None]


def step(m, opt, sched, idx, sid):
    m.brain_model.backbone.proj_in.sid = sid
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(brain_signals=xin(sid, idx), embs={k: v[idx] for k, v in emb_tr.items()}, return_loss=True).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    return float(loss)


g = torch.Generator(device=dev).manual_seed(args.seed)
spr = math.ceil(n_tr / args.bs)                     # steps per round
total = spr * S * args.epochs
opt, sched = make_opt(model, args.lr, total)
t0 = time.time()
model.train()
log = []
for ep in range(args.epochs):
    tot, n = 0.0, 0
    off = torch.randint(0, S, (n_tr,), device=dev, generator=g)
    for r in range(S):                                # round r: image i goes with subject (off_i + r) mod S
        perm = torch.randperm(n_tr, device=dev, generator=g)
        for b in range(spr):
            j = perm[b * args.bs:(b + 1) * args.bs]
            tot += step(model, opt, sched, tr_idx[j], (off[j] + r) % S) * len(j)
            n += len(j)
    log.append(tot / n)
joint_seconds = time.time() - t0
print(f"{args.tag} fold{args.fold}: joint training {joint_seconds:.0f}s, loss {log[0]:.3f} -> {log[-1]:.3f}", flush=True)


@torch.no_grad()
def encode(m, x, j):
    m.eval()
    m.brain_model.backbone.proj_in.sid = torch.full((len(x),), j, device=dev)
    out = m.get_brain_features(x).float()
    return out


def export(m, j):
    """Plain single-subject BrainEncoder state dict (same keys as train.py writes)."""
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
summary = []
for j, s in enumerate(subs):
    m = model
    ft_log = []
    if args.ft_epochs:
        m = copy.deepcopy(model)
        m.train()
        only = None if args.ft_trunk else (lambda n: f".proj_in.lin.{j}." in n)
        if not args.ft_trunk:
            m.requires_grad_(False)
            m.brain_model.backbone.proj_in.lin[j].requires_grad_(True)
        ft_total = spr * args.ft_epochs
        o2, s2 = make_opt(m, args.ft_lr, ft_total, only)
        for ep in range(args.ft_epochs):
            perm = torch.randperm(n_tr, device=dev, generator=g)
            tot = 0.0
            for b in range(spr):
                idx = tr_idx[perm[b * args.bs:(b + 1) * args.bs]]
                tot += step(m, o2, s2, idx, torch.full((len(idx),), j, device=dev)) * len(idx)
            ft_log.append(tot / n_tr)
    out = RUNS / "enc" / args.tag / f"sub-{s:02d}" / f"fold{args.fold}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    c80 = encode(m, XT[j], j)
    sim = nz(c80, dim=1) @ nz(F_te, dim=1).T
    r = sim.argsort(1, descending=True)
    met = dict(args=dict(vars(args), sub=s, chan="correct", t0=0, t1=250, inorm=0, mode="r4"), joint_seconds=joint_seconds,
               epoch_loss=log, ft_loss=ft_log, logit_scale=float(m.logit_scale.exp()),
               test_r80_top1_top5=[float((r[:, :k] == tgt[:, None]).any(1).float().mean()) for k in (1, 5)],
               test_r80_cos=float((nz(c80, dim=1) * nz(F_te, dim=1)).sum(1).mean()))
    np.save(out / "cond_test_r80.npy", c80.cpu().numpy())
    if args.fold >= 0:
        c = torch.cat([encode(m, xin(torch.full((len(ho_idx[i:i + 4096]),), j, device=dev), ho_idx[i:i + 4096], False), j) for i in range(0, len(ho_idx), 4096)])
        np.save(out / "cond_oof_r4.npy", c.cpu().numpy())
        np.save(out / "oof_index.npy", ho_idx.cpu().numpy())
        met["oof_r4_cos"] = float((nz(c, dim=1) * nz(F_tr[ho_idx], dim=1)).sum(1).mean())
    torch.save(export(m, j), out / "brain_model.pt")
    write_json(out / "metrics.json", met)
    summary.append((s, met["test_r80_top1_top5"][0], met["test_r80_top1_top5"][1], met["test_r80_cos"], met.get("oof_r4_cos", float("nan"))))
    if m is not model:
        del m
print(f"{args.tag} fold{args.fold} seed{args.seed} | " + " ".join(f"s{s}:{t1:.3f}/{t5:.3f}/cos{cs:.3f}" for s, t1, t5, cs, _ in summary)
      + f" | mean top1 {np.mean([x[1] for x in summary]):.4f} top5 {np.mean([x[2] for x in summary]):.4f} cos {np.mean([x[3] for x in summary]):.4f}"
      + (f" oof_cos {np.mean([x[4] for x in summary]):.4f}" if args.fold >= 0 else ""), flush=True)
