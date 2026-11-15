"""Multi-window joint encoder (extends train_ms.py with --t0 / --t1 window slicing, identical recipe to ms_e10_ft10in).

Trains strictly on training EEG (16,540 images x 10 subjects, 10 joint epochs + 10 per-subject input-layer FT epochs, lr=5e-4).
Exports plain single-subject BrainEncoder state dicts to runs/enc/<tag>/sub-XX/fold-1_seed<seed>/brain_model.pt.
"""
import argparse
import copy
import math
import sys
import time

import numpy as np
import torch

from common import HIVE, PREP, PRIOR, RUNS, write_json

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--t0", type=int, default=0)
p.add_argument("--t1", type=int, required=True)
p.add_argument("--seed", type=int, default=2025)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--ft_epochs", type=int, default=10)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--ft_lr", type=float, default=5e-4)
p.add_argument("--bs", type=int, default=1024)
args = p.parse_args()

subs = list(range(1, 11))
S = len(subs)
dev = "cuda"
T_len = args.t1 - args.t0

if all((RUNS / "enc" / args.tag / f"sub-{s:02d}" / f"fold-1_seed{args.seed}" / "brain_model.pt").exists() for s in subs):
    print(f"already done: {args.tag} seed {args.seed}", flush=True)
    sys.exit(0)

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

X = torch.stack([torch.from_numpy(np.load(PREP / f"sub-{s:02d}_train.npy")[..., args.t0:args.t1]).mean(1) for s in subs]).to(dev)
XT = torch.stack([torch.from_numpy(np.load(PREP / f"sub-{s:02d}_test.npy")[..., args.t0:args.t1]).mean(1) for s in subs]).to(dev)
emb_tr = {k: v.float().to(dev) for k, v in torch.load(PREP / "emb_train.pt", weights_only=True).items()}
F_te = torch.from_numpy(np.load(PREP / "F_test.npy")).to(dev)
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


pcfg = FusionEncoderModel.load_config(str(PRIOR), subfolder="fusion_encoder")
cfg = dict(hidden_size=1024, intermediate_size=1024, logit_scale_init_value=2.6592, dropout=0.1,
           brain_backbone="brain_mlp", brain_channels=17, brain_sequence_length=T_len, brain_embedding_dim=1440, proj_meta={})
cfg.update({k: v for k, v in pcfg.items() if k in cfg})
model = BrainCLIPModel(CLIPConfig.from_dict(cfg))
model.vision_model.load_state_dict(FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.state_dict())
model.vision_model.requires_grad_(False)
model.brain_model.backbone.proj_in = MSLinear(model.brain_model.backbone.proj_in, S)
model.to(dev)


def make_opt(params_model, lr, total, only=None):
    decay, no_decay = [], []
    for n, q in params_model.named_parameters():
        if q.requires_grad and (only is None or only(n)):
            (no_decay if (n.endswith("bias") or ".norm." in n) else decay).append(q)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.995), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 0.5 * (1.0 + math.cos(math.pi * s / total))))
    return opt, sched


def step(m, opt, sched, idx, sid):
    m.brain_model.backbone.proj_in.sid = sid
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(brain_signals=X[sid, idx], embs={k: v[idx] for k, v in emb_tr.items()}, return_loss=True).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    return float(loss)


g = torch.Generator(device=dev).manual_seed(args.seed)
spr = math.ceil(n_tr / args.bs)
opt, sched = make_opt(model, args.lr, spr * S * args.epochs)
t0 = time.time()
model.train()
log = []
for ep in range(args.epochs):
    tot, n = 0.0, 0
    off = torch.randint(0, S, (n_tr,), device=dev, generator=g)
    for r in range(S):
        perm = torch.randperm(n_tr, device=dev, generator=g)
        for b in range(spr):
            j = perm[b * args.bs:(b + 1) * args.bs]
            tot += step(model, opt, sched, j, (off[j] + r) % S) * len(j)
            n += len(j)
    log.append(tot / n)
joint_s = time.time() - t0


@torch.no_grad()
def encode(m, x, j):
    m.eval()
    m.brain_model.backbone.proj_in.sid = torch.full((len(x),), j, device=dev)
    return m.get_brain_features(x).float()


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
    sim = nz(c80, dim=1) @ nz(F_te, dim=1).T
    r = sim.argsort(1, descending=True)
    t1 = float((r[:, :1] == tgt[:, None]).any(1).float().mean())
    t1s.append(t1)
    met = dict(args=dict(vars(args), sub=s, chan="correct", inorm=0, mode="r4", fold=-1),
               joint_seconds=joint_s, test_r80_top1=t1)
    np.save(out / "cond_test_r80.npy", c80.cpu().numpy())
    torch.save(export(m, j), out / "brain_model.pt")
    write_json(out / "metrics.json", met)
    del m

print(f"{args.tag} [t={args.t0}:{args.t1}] done in {time.time() - t0:.0f}s | mean R80 top1={np.mean(t1s):.4f}", flush=True)
