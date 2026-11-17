"""Single-subject latency-tolerant multi-scale & multi-task EEG encoder (train_flex.py, for strict O8).

Strictly single-subject (zero cross-subject data, zero evaluation-network weights, zero test-image usage in training):
  - Supports --arch {mlp, pyr} and --num_layers {1, 2} with a single Linear proj_in stem (100% compatible with split-half TTA).
  - Supports --target {fused, spat_ms512}:
      * fused      : Brain-HIVE Fusion-Prior F_train with temporal micro-jitter (--jitter), soft teacher-similarity KL (--soft_w),
                     training-concept centroid alignment (--conc_w), direct cosine alignment (--cos_w), and auxiliary multi-scale
                     visual heads (--aux_w on Brain-HIVE's own emb_train[H14, B32] + spat_ms512['S_train']).
      * spat_ms512 : Native 512px multi-scale 4-orientation contour target S_train (from spat_ms512.pt, built strictly on lat512_train).
  - Supports --fold -1 (full training set) and --fold 0..4 (5-fold concept-disjoint OOF, exporting cond_oof_r4.npy + oof_index.npy).
"""
import argparse
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import EMB_KEYS, HIVE, PREP, PRIOR, RUNS, concept_folds, write_json

sys.path.insert(0, str(HIVE))
import transformers  # noqa: E402
from main.layers import ResNet1DBlock  # noqa: E402
from main.models_adapter import FusionEncoderModel  # noqa: E402

nz = F.normalize


def pyramid_feats(x):
    """Linear multi-scale temporal pyramid + smoothed first differences on [B, 17, T]."""
    s2 = F.avg_pool1d(x, kernel_size=3, stride=2, padding=1)
    s4 = F.avg_pool1d(x, kernel_size=7, stride=4, padding=3)
    s8 = F.avg_pool1d(x, kernel_size=15, stride=8, padding=7)
    dx2 = s2[..., 1:] - s2[..., :-1]
    dx4 = s4[..., 1:] - s4[..., :-1]
    return torch.cat([
        x.flatten(1),
        s2.flatten(1) * 0.85,
        s4.flatten(1) * 0.75,
        s8.flatten(1) * 0.65,
        dx2.flatten(1) * 0.70,
        dx4.flatten(1) * 0.60,
    ], dim=1)


class FlexBackbone(nn.Module):
    def __init__(self, t_len, arch="mlp", hidden_size=1024, num_layers=1, dropout=0.12):
        super().__init__()
        self.arch = arch
        in_dim = pyramid_feats(torch.zeros(1, 17, t_len)).shape[1] if arch == "pyr" else 17 * t_len
        self.proj_in = nn.Linear(in_dim, hidden_size)
        self.block = ResNet1DBlock(hidden_size=hidden_size, intermediate_size=hidden_size,
                                   num_layers=num_layers, dropout=dropout)

    def forward(self, x):
        f = pyramid_feats(x) if self.arch == "pyr" else x.flatten(1)
        return self.block(self.proj_in(f))


class FlexBrainEncoder(nn.Module):
    def __init__(self, t_len, arch="mlp", hidden_size=1024, num_layers=1, dropout=0.12):
        super().__init__()
        self.backbone = FlexBackbone(t_len, arch=arch, hidden_size=hidden_size,
                                     num_layers=num_layers, dropout=dropout)

    def forward(self, x, subject_ids=None):
        return self.backbone(x)


def shift_batch(xb, jitter, g):
    if jitter <= 0:
        return xb
    t_len = xb.shape[-1]
    sh = torch.randint(-jitter, jitter + 1, (len(xb),), device=xb.device, generator=g)
    if not (sh != 0).any():
        return xb
    xb_pad = F.pad(xb, (jitter, jitter), mode="replicate")
    t_grid = torch.arange(t_len, device=xb.device)[None, None, :] + (sh + jitter)[:, None, None]
    return torch.gather(xb_pad, 2, t_grid.expand(-1, 17, -1))


def ivw_mean(e):
    tot = e.sum(1, keepdim=True)
    v = (e - (tot - e) / (e.shape[1] - 1)).pow(2).mean(dim=(2, 3))
    w = 1.0 / v.clamp_min(1e-12)
    return ((w / w.sum(1, keepdim=True))[:, :, None, None] * e).sum(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sub", type=int, required=True)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--fold", type=int, default=-1)
    p.add_argument("--t0", type=int, default=0)
    p.add_argument("--t1", type=int, default=250)
    p.add_argument("--arch", choices=["mlp", "pyr"], default="mlp")
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.12)
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--bs", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--jitter", type=int, default=2)
    p.add_argument("--sub_aug", type=int, default=0)
    p.add_argument("--soft_w", type=float, default=0.25)
    p.add_argument("--conc_w", type=float, default=0.20)
    p.add_argument("--cos_w", type=float, default=0.20)
    p.add_argument("--aux_w", type=float, default=0.30)
    p.add_argument("--target", choices=["fused", "spat_ms512"], default="fused")
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    dev = "cuda"
    out = RUNS / "enc" / args.tag / f"sub-{args.sub:02d}" / f"fold{args.fold}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists() and (out / "brain_model.pt").exists():
        print("already done:", out, flush=True)
        sys.exit(0)

    transformers.set_seed(args.seed)
    t_len = args.t1 - args.t0
    x_raw = torch.from_numpy(np.load(PREP / f"sub-{args.sub:02d}_train.npy")[..., args.t0:args.t1]).float().to(dev)
    x_mean = x_raw.mean(1).contiguous()
    x_te = torch.from_numpy(np.load(PREP / f"sub-{args.sub:02d}_test.npy")[..., args.t0:args.t1]).float().to(dev)

    concept = torch.from_numpy(np.load(PREP / "train_concept.npy")).long().to(dev)
    img_fold = torch.from_numpy(concept_folds(1654, 5))[concept.cpu()].to(dev)

    sp_pack = torch.load(PREP / "spat_ms512.pt", map_location="cpu", weights_only=True)
    ok_mask = sp_pack["ok"].to(dev)
    S_tr_raw = sp_pack["S_train"].float().to(dev)
    S_tr_u = nz(S_tr_raw, dim=1)
    # Center S_tr_u on valid training images
    s_mu = S_tr_u[ok_mask].mean(0, keepdim=True)
    Sc_tr_u = nz(S_tr_u - s_mu, dim=1)

    if args.target == "spat_ms512":
        valid_base = ok_mask
    else:
        valid_base = torch.ones(len(concept), dtype=torch.bool, device=dev)

    if args.fold < 0:
        tr_idx = torch.nonzero(valid_base).squeeze(1)
        ho_idx = torch.empty(0, dtype=torch.long, device=dev)
    else:
        tr_idx = torch.nonzero(valid_base & (img_fold != args.fold)).squeeze(1)
        ho_idx = torch.nonzero(img_fold == args.fold).squeeze(1)
    n_tr = len(tr_idx)

    brain_model = FlexBrainEncoder(t_len, arch=args.arch, hidden_size=1024,
                                   num_layers=args.num_layers, dropout=args.dropout).to(dev)
    logit_scale = nn.Parameter(torch.tensor(2.6592, device=dev))
    aux_heads = nn.ModuleDict()

    if args.target == "fused":
        emb_tr = {k: v.float().to(dev) for k, v in torch.load(PREP / "emb_train.pt", weights_only=True).items()}
        F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float().to(dev)
        F_tr_u = nz(F_tr, dim=1)
        H14_u = nz(emb_tr[EMB_KEYS[0]], dim=1)
        B32_u = nz(emb_tr[EMB_KEYS[1]], dim=1)

        # Compute concept centroids strictly on tr_idx (zero cross-fold information)
        C_cent = torch.zeros((1654, 1024), dtype=torch.float32, device=dev)
        C_cent.index_add_(0, concept[tr_idx], F_tr_u[tr_idx])
        C_cent = nz(C_cent, dim=1)

        vision_teacher = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.to(dev).eval()
        vision_teacher.requires_grad_(False)
        if args.aux_w > 0:
            aux_heads["h14"] = nn.Linear(1024, 1024, device=dev)
            aux_heads["b32"] = nn.Linear(1024, 512, device=dev)
            aux_heads["spat"] = nn.Linear(1024, 1024, device=dev)

    params = list(brain_model.parameters()) + [logit_scale] + list(aux_heads.parameters())
    decay, no_decay = [], []
    for n, q in list(brain_model.named_parameters()) + [("logit_scale", logit_scale)] + list(aux_heads.named_parameters()):
        (no_decay if (n.endswith("bias") or ".norm." in n or n == "logit_scale") else decay).append(q)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.995), eps=1e-8)
    spr = math.ceil(n_tr / args.bs)
    total = spr * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 0.5 * (1.0 + math.cos(math.pi * s / total))))
    g = torch.Generator(device=dev).manual_seed(args.seed)

    t_start = time.time()
    brain_model.train()
    if args.target == "fused":
        vision_teacher.train()
    loss_log = []
    for _ in range(args.epochs):
        perm = torch.randperm(n_tr, device=dev, generator=g)
        tot_l = 0.0
        for b in range(spr):
            idx = tr_idx[perm[b * args.bs:(b + 1) * args.bs]]
            if args.sub_aug:
                use3 = torch.rand(len(idx), device=dev, generator=g) < 0.5
                drop_r = torch.randint(0, 4, (len(idx),), device=dev, generator=g)
                mask = torch.ones((len(idx), 4), device=dev)
                mask[use3, drop_r[use3]] = 0.0
                xb = (x_raw[idx] * mask[:, :, None, None]).sum(1) / mask.sum(1)[:, None, None]
            else:
                xb = x_mean[idx]
            xb = shift_batch(xb, args.jitter, g)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                c_raw = brain_model(xb)
                if args.target == "fused":
                    z = nz(c_raw, dim=-1)
                    with torch.no_grad():
                        y = nz(vision_teacher(**{ek: ev[idx] for ek, ev in emb_tr.items()}), dim=-1)
                    scale = logit_scale.exp().clamp(max=100.0)
                    lg = scale * (z @ y.T)
                    tb = torch.arange(len(idx), device=dev)
                    loss_nce = 0.5 * (F.cross_entropy(lg, tb) + F.cross_entropy(lg.T, tb))
                    loss = (1.0 - args.soft_w) * loss_nce
                    if args.soft_w > 0:
                        with torch.no_grad():
                            p_t = F.softmax((F_tr_u[idx] @ F_tr_u[idx].T) / 0.07, dim=-1)
                        loss_kl = 0.5 * (F.kl_div(F.log_softmax(lg, dim=-1), p_t, reduction="batchmean")
                                         + F.kl_div(F.log_softmax(lg.T, dim=-1), p_t, reduction="batchmean"))
                        loss = loss + args.soft_w * loss_kl
                    if args.conc_w > 0:
                        loss = loss + args.conc_w * (1.0 - (z * C_cent[concept[idx]]).sum(-1)).mean()
                    if args.cos_w > 0:
                        loss = loss + args.cos_w * (1.0 - (z * F_tr_u[idx]).sum(-1)).mean()
                    if args.aux_w > 0:
                        valid_s = ok_mask[idx].float()
                        la_h = (1.0 - (nz(aux_heads["h14"](c_raw), dim=-1) * H14_u[idx]).sum(-1)).mean()
                        la_b = (1.0 - (nz(aux_heads["b32"](c_raw), dim=-1) * B32_u[idx]).sum(-1)).mean()
                        la_s = ((1.0 - (nz(aux_heads["spat"](c_raw), dim=-1) * Sc_tr_u[idx]).sum(-1)) * valid_s).sum() / valid_s.sum().clamp_min(1.0)
                        loss = loss + args.aux_w * (la_h + la_b + la_s) / 3.0
                else:
                    zc = nz(c_raw - c_raw.mean(0, keepdim=True), dim=-1)
                    yc = Sc_tr_u[idx]
                    scale = logit_scale.exp().clamp(max=100.0)
                    lg = scale * (zc @ yc.T)
                    tb = torch.arange(len(idx), device=dev)
                    loss_nce = 0.5 * (F.cross_entropy(lg, tb) + F.cross_entropy(lg.T, tb))
                    if args.soft_w > 0:
                        with torch.no_grad():
                            p_t = F.softmax((yc @ yc.T) / 0.08, dim=-1)
                        loss_kl = 0.5 * (F.kl_div(F.log_softmax(lg, dim=-1), p_t, reduction="batchmean")
                                         + F.kl_div(F.log_softmax(lg.T, dim=-1), p_t, reduction="batchmean"))
                        loss = (1.0 - args.soft_w) * loss_nce + args.soft_w * loss_kl
                    else:
                        loss = loss_nce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            tot_l += float(loss) * len(idx)
        loss_log.append(tot_l / n_tr)
    train_s = time.time() - t_start

    brain_model.eval()
    with torch.no_grad():
        c80 = brain_model(x_te.mean(1)).float()
        c_ivw = brain_model(ivw_mean(x_te)).float()

    met = dict(args=dict(vars(args), chan="correct", inorm=0, mode="r4"),
               n_train=n_tr, train_seconds=train_s, logit_scale=float(logit_scale.exp()))
    if args.target == "fused":
        F_te = torch.from_numpy(np.load(PREP / "F_test.npy")).float().to(dev)
        Fn_te = nz(F_te, dim=1)
        tgt = torch.arange(200, device=dev)
        sim80 = nz(c80, dim=1) @ Fn_te.T
        sim_ivw = nz(c_ivw, dim=1) @ Fn_te.T
        met["test_r80_top1"] = float((sim80.argmax(1) == tgt).float().mean())
        met["test_ivw_top1"] = float((sim_ivw.argmax(1) == tgt).float().mean())

    if args.fold >= 0 and len(ho_idx) > 0:
        with torch.no_grad():
            c_oof = torch.cat([brain_model(x_mean[ho_idx[i:i + 2048]]).float() for i in range(0, len(ho_idx), 2048)])
        np.save(out / "cond_oof_r4.npy", c_oof.cpu().numpy())
        np.save(out / "oof_index.npy", ho_idx.cpu().numpy())

    np.save(out / "cond_test_r80.npy", c80.cpu().numpy())
    np.save(out / "cond_test_ivw.npy", c_ivw.cpu().numpy())
    torch.save({k: v.cpu() for k, v in brain_model.state_dict().items()}, out / "brain_model.pt")
    write_json(out / "metrics.json", met)
    print(f"{args.tag} sub-{args.sub:02d} fold{args.fold} s{args.seed} ({train_s:.1f}s) "
          + (f"ivw_t1={met['test_ivw_top1']:.3f}" if "test_ivw_top1" in met else "spat_ms512"), flush=True)
