"""Train one EEG encoder against the frozen Fusion-Prior target and evaluate it at the condition level.

Recipe = Brain-HIVE reconstruction stage (configs/train_clip.yaml): BrainMLP, symmetric InfoNCE with learnable
temperature (init 1/0.07), AdamW lr 5e-4, betas (0.9, 0.995), wd 0.05 (not on bias / LayerNorm), cosine schedule
without warmup, 25 epochs, batch 1024, grad-norm clip 1.0, bf16 autocast, dropout 0.1 (teacher kept in train mode
as in the official script). Differences from the official entry point: the seed is set before the model is built,
and data live on the GPU instead of a HF Dataset.

Training-input modes (how the 4 single-trial repeats of an image become one input):
  r4        mean of all 4 repeats (official)
  subset_u  mean of a random subset, size k ~ Uniform{1,2,3,4}
  subset_k  mean of a random subset, size k with p(k) proportional to k
  noise     mean of 4 + Gaussian noise whose variance matches a size-k subset (control: generic noise, no real trials)
"""
import argparse
import itertools
import math
import sys
import time

import numpy as np
import torch

from common import CH17, HIVE, PREP, PRIOR, RUNS, concept_folds, write_json

p = argparse.ArgumentParser()
p.add_argument("--sub", type=int, required=True)
p.add_argument("--seed", type=int, default=2025)
p.add_argument("--fold", type=int, default=-1, help="-1: train on all 1654 concepts; k: hold out concept fold k")
p.add_argument("--nfold", type=int, default=5)
p.add_argument("--mode", default="r4", choices=["r4", "subset_u", "subset_k", "noise", "ivw4"],
               help="ivw4: inverse-variance weighted mean of the 4 repeats (weight 1 / residual power against the other 3)")
p.add_argument("--inorm", type=int, default=0, help="1: divide every input by its RMS (train and test)")
p.add_argument("--t0", type=int, default=0)
p.add_argument("--t1", type=int, default=250)
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--bs", type=int, default=1024)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--chan", default="correct", choices=["correct", "typo"])
p.add_argument("--emb", default="", help="target variant: '' clean images (official), deg4 / blur = degraded-image targets from deg.py")
p.add_argument("--cos_w", type=float, default=0.0, help="weight of an absolute alignment term 1 - cos(c, F) added to InfoNCE")
p.add_argument("--chkeep", type=int, default=0,
               help="k > 0: random-subspace encoder (Ho 1998): only k of the 17 channels, chosen by --seed, are kept; the rest "
                    "are zeroed at train and test time")
p.add_argument("--tag", required=True)
p.add_argument("--save_model", type=int, default=1)
p.add_argument("--save_cond", type=int, default=1, help="0: keep only metrics.json and the small mean vectors")
args = p.parse_args()

dev = "cuda"
out = RUNS / "enc" / args.tag / f"sub-{args.sub:02d}" / f"fold{args.fold}_seed{args.seed}"
out.mkdir(parents=True, exist_ok=True)
if (out / "metrics.json").exists():
    print("already done:", out)
    sys.exit(0)

sys.path.insert(0, str(HIVE))
import transformers  # noqa: E402
import transformers.models.clip.modeling_clip as _mc  # noqa: E402

if not hasattr(_mc, "clip_loss"):  # removed from newer transformers; same definition as the one the official code imported
    def _ce(logits):
        return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
    _mc.clip_loss = lambda sim: (_ce(sim) + _ce(sim.t())) / 2.0
from main.models_adapter import FusionEncoderModel  # noqa: E402
from main.models_brain import BrainEncoder  # noqa: E402
from main.models_clip import BrainCLIPModel, CLIPConfig  # noqa: E402

# Official bug at commit f06db69: BrainEncoder.forward requires subject_ids but the brain_mlp path never passes it.
BrainEncoder.forward = lambda self, brain_signals, subject_ids=None: self.backbone(brain_signals)

transformers.set_seed(args.seed)

# ---------------------------------------------------------------- data
x_tr = torch.from_numpy(np.load(PREP / f"sub-{args.sub:02d}_train.npy"))  # [16540, 4, 17, 250]
x_te = torch.from_numpy(np.load(PREP / f"sub-{args.sub:02d}_test.npy"))   # [200, 80, 17, 250]
if args.chan == "typo":  # official reconstruction script: Oz listed where POz belongs
    a, b = CH17.index("POz"), CH17.index("Oz")
    x_tr[:, :, a], x_te[:, :, a] = x_tr[:, :, b].clone(), x_te[:, :, b].clone()
if args.chkeep:  # same subset for every subject and fold of a seed
    args.chlist = np.sort(np.random.default_rng(args.seed).choice(17, args.chkeep, replace=False)).tolist()
    drop = [c for c in range(17) if c not in args.chlist]
    x_tr[:, :, drop], x_te[:, :, drop] = 0, 0
x_tr = x_tr[..., args.t0:args.t1].contiguous().to(dev)
x_te = x_te[..., args.t0:args.t1].contiguous().to(dev)
sfx = f"_{args.emb}" if args.emb else ""
emb_tr = {k: v.float().to(dev) for k, v in torch.load(PREP / f"emb_train{sfx}.pt", weights_only=True).items()}
F_tr = torch.from_numpy(np.load(PREP / f"F_train{sfx}.npy")).to(dev)   # the encoder's own target space (candidates for identification)
F_te = torch.from_numpy(np.load(PREP / f"F_test{sfx}.npy")).to(dev)
F_te_clean = torch.from_numpy(np.load(PREP / "F_test.npy")).to(dev)
concept = torch.from_numpy(np.load(PREP / "train_concept.npy"))
img_fold = torch.from_numpy(concept_folds(1654, args.nfold))[concept]
tr_idx = (torch.arange(len(concept)) if args.fold < 0 else torch.nonzero(img_fold != args.fold).squeeze(1)).to(dev)
ho_idx = (torch.empty(0, dtype=torch.long) if args.fold < 0 else torch.nonzero(img_fold == args.fold).squeeze(1)).to(dev)
n_tr = len(tr_idx)

g = torch.Generator(device=dev).manual_seed(args.seed)
if args.mode == "noise":  # per-dimension single-trial noise std, from the training images only
    e = x_tr[tr_idx]
    sigma = e.var(dim=1, unbiased=True).mean(0).sqrt()  # [17, T]
    del e


def rms_norm(x):
    return x / x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8) if args.inorm else x


def ivw_mean(e):
    """[B, R, C, T] -> inverse-variance weighted mean over the R repeats (weight 1 / residual power against the other R - 1)."""
    tot = e.sum(1, keepdim=True)
    v = (e - (tot - e) / (e.shape[1] - 1)).pow(2).mean(dim=(2, 3))
    w = 1 / v.clamp_min(1e-12)
    return ((w / w.sum(1, keepdim=True))[:, :, None, None] * e).sum(1)


def train_input(idx):
    e = x_tr[idx]  # [B, 4, C, T]
    B = len(idx)
    if args.mode == "r4":
        return rms_norm(e.mean(1))
    if args.mode == "ivw4":
        return rms_norm(ivw_mean(e))
    if args.mode in ("subset_u", "subset_k"):
        w = torch.ones(4, device=dev) if args.mode == "subset_u" else torch.arange(1., 5., device=dev)
        k = torch.multinomial(w, B, replacement=True, generator=g) + 1                      # [B] in 1..4
        rank = torch.rand(B, 4, device=dev, generator=g).argsort(1).argsort(1)
        m = (rank < k[:, None]).float()
        return rms_norm((e * m[:, :, None, None]).sum(1) / k[:, None, None])
    k = torch.randint(1, 5, (B,), device=dev, generator=g).float()                          # noise control
    scale = (1. / k - 0.25).clamp_min(0).sqrt()[:, None, None]
    return rms_norm(e.mean(1) + torch.randn(e.shape[0], *e.shape[2:], device=dev, generator=g) * sigma * scale)


# ---------------------------------------------------------------- model
pcfg = FusionEncoderModel.load_config(str(PRIOR), subfolder="fusion_encoder")
cfg = dict(hidden_size=1024, intermediate_size=1024, logit_scale_init_value=2.6592, dropout=0.1,
           brain_backbone="brain_mlp", brain_channels=17, brain_sequence_length=args.t1 - args.t0,
           brain_embedding_dim=1440, proj_meta={})
cfg.update({k: v for k, v in pcfg.items() if k in cfg})  # same override rule as the official train_clip.py
model = BrainCLIPModel(CLIPConfig.from_dict(cfg))
model.vision_model.load_state_dict(
    FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.state_dict())
model.vision_model.requires_grad_(False)
model.to(dev)

decay, no_decay = [], []
for n, q in model.named_parameters():
    if q.requires_grad:
        (no_decay if (n.endswith("bias") or ".norm." in n) else decay).append(q)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=args.lr, betas=(0.9, 0.995), eps=1e-8)
steps_per_epoch = math.ceil(n_tr / args.bs)
total = steps_per_epoch * args.epochs
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 0.5 * (1.0 + math.cos(math.pi * s / total))))

# ---------------------------------------------------------------- train
t_start = time.time()
model.train()
log = []
for ep in range(args.epochs):
    perm = torch.randperm(n_tr, device=dev, generator=g)
    tot = 0.0
    for b in range(steps_per_epoch):
        idx = tr_idx[perm[b * args.bs:(b + 1) * args.bs]]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(brain_signals=train_input(idx), embs={k: v[idx] for k, v in emb_tr.items()}, return_loss=True)
            loss = o.loss
            if args.cos_w:  # InfoNCE only aligns relatively; this pulls c onto the image embedding itself
                loss = loss + args.cos_w * (1 - (o.brain_embeds * o.image_embeds).sum(-1)).float().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        tot += float(loss) * len(idx)
    log.append(tot / n_tr)
train_seconds = time.time() - t_start

# ---------------------------------------------------------------- condition-level evaluation
model.eval()
nz = torch.nn.functional.normalize


@torch.no_grad()
def encode(x):  # raw encoder output c (what goes to the generator), float32
    return torch.cat([model.get_brain_features(rms_norm(x[i:i + 2048])).float() for i in range(0, len(x), 2048)])


def topk(sim, target, ks=(1, 5)):
    r = sim.argsort(1, descending=True)
    return [float((r[:, :k] == target[:, None]).any(1).float().mean()) for k in ks]


def within_concept(c, f, cid):
    """10-way identification among the images of the same concept (chance 0.1) and its 2AFC version."""
    order = cid.argsort(stable=True)
    c, f = nz(c[order], dim=1), nz(f[order], dim=1)
    n = len(order) // 10
    s = torch.einsum("nid,njd->nij", c.view(n, 10, -1), f.view(n, 10, -1))
    d = torch.diagonal(s, dim1=1, dim2=2)
    top1 = (s.argmax(2) == torch.arange(10, device=s.device)[None]).float().mean()
    win = ((d[:, :, None] > s).float() + 0.5 * (d[:, :, None] == s).float()).sum(2) - 0.5  # drop the self-tie
    return float(top1), float((win / 9).mean())


m = dict(args=vars(args), n_train=n_tr, steps=total, train_seconds=train_seconds, epoch_loss=log,
         logit_scale=float(model.logit_scale.exp()))

# official test set: 200 concepts x 1 image, 80 repeats
Fn_te = nz(F_te, dim=1)
tgt = torch.arange(200, device=dev)
test_by_k, mu_test = {}, {}
c4_sum = torch.zeros(200, 1024, device=dev)
for k in (1, 2, 4, 8, 16, 40, 80):
    accs, mus = [], []
    for s in range(0, 80, k):  # disjoint groups of k consecutive repeats
        c = encode(x_te[:, s:s + k].mean(1))
        accs.append(topk(nz(c, dim=1) @ Fn_te.T, tgt))
        mus.append(nz(c, dim=1).mean(0))
        if k == 4:
            c4_sum += c
    test_by_k[k] = np.mean(accs, 0).tolist()
    mu_test[k] = torch.stack(mus).mean(0).cpu().numpy()
c80 = encode(x_te.mean(1))
c_ivw = encode(ivw_mean(x_te))
m["test_ivw_top1_top5"] = topk(nz(c_ivw, dim=1) @ Fn_te.T, tgt)
np.save(out / "cond_test_ivw.npy", c_ivw.cpu().numpy())
c_emb = c4_sum / 20  # embedding-space average of 20 disjoint 4-repeat inputs (matches the training repeat level)
m["test_top1_top5_by_k"] = test_by_k
m["test_r80_top1_top5"] = topk(nz(c80, dim=1) @ Fn_te.T, tgt)
m["test_emb20x4_top1_top5"] = topk(nz(c_emb, dim=1) @ Fn_te.T, tgt)
m["test_r80_cos"] = float((nz(c80, dim=1) * Fn_te).sum(1).mean())
m["test_r80_top1_top5_cleanF"] = topk(nz(c80, dim=1) @ nz(F_te_clean, dim=1).T, tgt)
np.save(out / "cond_test_r80.npy", c80.cpu().numpy())
np.save(out / "cond_test_emb20x4.npy", c_emb.cpu().numpy())
np.save(out / "mu_test_by_k.npy", np.stack([mu_test[k] for k in (1, 2, 4, 8, 16, 40, 80)]))

# one fold of training concepts seen by this model (memorisation reference), 4-repeat mean
m["train_concepts_r4_ic_top1"] = None
if args.fold >= 0:
    seen = torch.nonzero(img_fold.to(dev) == (args.fold + 1) % args.nfold).squeeze(1)
    c = encode(x_tr[seen].mean(1))
    m["train_concepts_r4_ic_top1"], m["train_concepts_r4_ic_2afc"] = within_concept(c, F_tr[seen], concept.to(dev)[seen])

# held-out concepts (never seen by this encoder): all 15 repeat subsets
if args.fold >= 0:
    cid = concept.to(dev)[ho_idx]
    f_ho = F_tr[ho_idx]
    Fn_ho = nz(f_ho, dim=1)
    n_ho = len(ho_idx)
    tgt_ho = torch.arange(n_ho, device=dev)
    uc = torch.unique(cid)[:200]                                         # first 200 held-out concepts -> 200-way galleries
    by_k = {k: [] for k in (1, 2, 3, 4)}
    mu_k = {k: [] for k in (1, 2, 3, 4)}
    for k in (1, 2, 3, 4):
        for S in itertools.combinations(range(4), k):
            c = encode(x_tr[ho_idx][:, list(S)].mean(1))
            cn = nz(c, dim=1)
            ic1, ic2 = within_concept(c, f_ho, cid)
            g1, g5 = topk(cn @ Fn_ho.T, tgt_ho)
            cv = []
            for j in range(10):                                           # j-th image of each of the 200 concepts
                pos = torch.stack([torch.nonzero(cid == u).squeeze(1)[j] for u in uc])
                cv.append(topk(cn[pos] @ Fn_ho[pos].T, torch.arange(200, device=dev)))
            cv1, cv5 = np.mean(cv, 0).tolist()
            cos = float((cn * Fn_ho).sum(1).mean())
            by_k[k].append([ic1, ic2, g1, g5, cv1, cv5, cos])
            mu_k[k].append(cn.mean(0))
            if k == 4 and args.save_cond:
                np.save(out / "cond_oof_r4.npy", c.cpu().numpy())
    names = ["ic_top1", "ic_2afc", "gallery_top1", "gallery_top5", "cv200_top1", "cv200_top5", "cos_pair"]
    m["oof_by_k"] = {k: dict(zip(names, np.mean(v, 0).tolist())) for k, v in by_k.items()}
    m["oof_n_images"] = n_ho
    np.save(out / "mu_oof_by_k.npy", torch.stack([torch.stack(mu_k[k]).mean(0) for k in (1, 2, 3, 4)]).cpu().numpy())
    np.save(out / "oof_index.npy", ho_idx.cpu().numpy())

if args.save_model:
    torch.save({k: v.cpu() for k, v in model.brain_model.state_dict().items()}, out / "brain_model.pt")
write_json(out / "metrics.json", m)
r = m.get("oof_by_k", {}).get(4, {})
print(f"{args.tag} sub-{args.sub:02d} fold{args.fold} seed{args.seed} | {train_seconds:.0f}s loss {log[0]:.3f}->{log[-1]:.3f} | "
      f"test R80 top1/5 {m['test_r80_top1_top5'][0]:.3f}/{m['test_r80_top1_top5'][1]:.3f} emb20x4 {m['test_emb20x4_top1_top5'][0]:.3f} cos {m['test_r80_cos']:.3f} cleanF top1 {m['test_r80_top1_top5_cleanF'][0]:.3f} | "
      + (f"OOF R4 ic_top1 {r['ic_top1']:.4f} cv200 {r['cv200_top1']:.4f} | seen-concept ic_top1 {m['train_concepts_r4_ic_top1']:.4f}" if r else ""),
      flush=True)
