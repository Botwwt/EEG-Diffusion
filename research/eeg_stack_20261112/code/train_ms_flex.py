"""Multi-subject multi-window latency-tolerant & multi-task joint EEG encoder (train_ms_flex.py, for full-stack B5).

Trains on training EEG across all 10 subjects (per-subject MSLinear input stem + shared ResNet1DBlock trunk),
followed by per-subject input-stem fine-tuning, exporting plain per-subject FlexBrainEncoder checkpoints to:
  runs/enc/<tag>/sub-XX/fold<fold>_seed<seed>/brain_model.pt + metrics.json

Supports --target {fused, h14, spat_ms512}:
  - fused      : Brain-HIVE Fusion-Prior F_train with temporal micro-jitter (--jitter), soft teacher-similarity KL (--soft_w),
                 training-concept centroid alignment (--conc_w), direct cosine (--cos_w), and auxiliary H14/B32/Spat heads (--aux_w).
  - h14        : Brain-HIVE's own CLIP-ViT-H-14-laion2B-s32B-b79K image target (emb_train.pt), zero evaluation-network weights.
  - spat_ms512 : Native 512px multi-scale 4-orientation contour target S_train (from spat_ms512.pt, built strictly on lat512_train).
"""
import argparse
import copy
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import EMB_KEYS, HIVE, PREP, PRIOR, RUNS, concept_folds, write_json
from train_flex import FlexBrainEncoder, ivw_mean, pyramid_feats, shift_batch

sys.path.insert(0, str(HIVE))
import transformers  # noqa: E402
from main.models_adapter import FusionEncoderModel  # noqa: E402

nz = F.normalize


class MSLinear(nn.Module):
    def __init__(self, base, n):
        super().__init__()
        self.lin = nn.ModuleList([copy.deepcopy(base) for _ in range(n)])
        self.sid = None

    def forward(self, x):
        out = torch.empty(len(x), self.lin[0].out_features, device=x.device,
                          dtype=torch.promote_types(x.dtype, self.lin[0].weight.dtype))
        for j in torch.unique(self.sid).tolist():
            m = self.sid == j
            out[m] = self.lin[j](x[m]).to(out.dtype)
        return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--t0", type=int, default=0)
    p.add_argument("--t1", type=int, required=True)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--fold", type=int, default=-1)
    p.add_argument("--arch", choices=["mlp", "pyr"], default="mlp")
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.14)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--ft_epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--ft_lr", type=float, default=4.5e-4)
    p.add_argument("--bs", type=int, default=1024)
    p.add_argument("--jitter", type=int, default=2)
    p.add_argument("--sub_aug", type=int, default=1)
    p.add_argument("--soft_w", type=float, default=0.25)
    p.add_argument("--conc_w", type=float, default=0.20)
    p.add_argument("--cos_w", type=float, default=0.20)
    p.add_argument("--aux_w", type=float, default=0.30)
    p.add_argument("--target", choices=["fused", "h14", "spat_ms512"], default="fused")
    args = p.parse_args()

    subs = list(range(1, 11))
    S = len(subs)
    dev = "cuda"
    t_len = args.t1 - args.t0

    if all((RUNS / "enc" / args.tag / f"sub-{s:02d}" / f"fold{args.fold}_seed{args.seed}" / "brain_model.pt").exists() for s in subs):
        print(f"already done: {args.tag} fold {args.fold} seed {args.seed}", flush=True)
        sys.exit(0)

    transformers.set_seed(args.seed)
    X_raw = torch.stack([torch.from_numpy(np.load(PREP / f"sub-{s:02d}_train.npy")[..., args.t0:args.t1]).half() for s in subs]).to(dev)
    X = X_raw.float().mean(2).contiguous()
    if not args.sub_aug:
        del X_raw
    XT_ivw = torch.stack([ivw_mean(torch.from_numpy(np.load(PREP / f"sub-{s:02d}_test.npy")[..., args.t0:args.t1]).float().to(dev)) for s in subs])

    concept = torch.from_numpy(np.load(PREP / "train_concept.npy")).long().to(dev)
    img_fold = torch.from_numpy(concept_folds(1654, 5))[concept.cpu()].to(dev)

    sp_pack = torch.load(PREP / "spat_ms512.pt", map_location="cpu", weights_only=True)
    ok_mask = sp_pack["ok"].to(dev)
    S_tr_u = nz(sp_pack["S_train"].float().to(dev), dim=1)
    Sc_tr_u = nz(S_tr_u - S_tr_u[ok_mask].mean(0, keepdim=True), dim=1)

    valid_base = ok_mask if args.target == "spat_ms512" else torch.ones(len(concept), dtype=torch.bool, device=dev)
    if args.fold < 0:
        tr_idx = torch.nonzero(valid_base).squeeze(1)
        ho_idx = torch.empty(0, dtype=torch.long, device=dev)
    else:
        tr_idx = torch.nonzero(valid_base & (img_fold != args.fold)).squeeze(1)
        ho_idx = torch.nonzero(img_fold == args.fold).squeeze(1)
    n_tr = len(tr_idx)

    emb_tr = {k: v.float().to(dev) for k, v in torch.load(PREP / "emb_train.pt", weights_only=True).items()}
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float().to(dev)
    F_tr_u = nz(F_tr, dim=1)
    H14_u = nz(emb_tr[EMB_KEYS[0]], dim=1)
    B32_u = nz(emb_tr[EMB_KEYS[1]], dim=1)

    C_cent = torch.zeros((1654, 1024), dtype=torch.float32, device=dev)
    tgt_vec_for_cent = H14_u if args.target == "h14" else F_tr_u
    C_cent.index_add_(0, concept[tr_idx], tgt_vec_for_cent[tr_idx])
    C_cent = nz(C_cent, dim=1)

    vision_teacher = None
    if args.target == "fused":
        vision_teacher = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.to(dev).eval()
        vision_teacher.requires_grad_(False)

    brain_model = FlexBrainEncoder(t_len, arch=args.arch, hidden_size=1024,
                                   num_layers=args.num_layers, dropout=args.dropout)
    brain_model.backbone.proj_in = MSLinear(brain_model.backbone.proj_in, S)
    brain_model = brain_model.to(dev)
    logit_scale = nn.Parameter(torch.tensor(2.6592, device=dev))
    aux_heads = nn.ModuleDict()
    if args.target in ("fused", "h14") and args.aux_w > 0:
        aux_heads["h14"] = nn.Linear(1024, 1024, device=dev)
        aux_heads["b32"] = nn.Linear(1024, 512, device=dev)
        aux_heads["spat"] = nn.Linear(1024, 1024, device=dev)

    def make_opt(bm, ls, ah, lr, total, only=None):
        decay, no_decay, trainable = [], [], []
        for n, q in list(bm.named_parameters()) + [("logit_scale", ls)] + list(ah.named_parameters()):
            if q.requires_grad and (only is None or only(n)):
                trainable.append(q)
                (no_decay if (n.endswith("bias") or ".norm." in n or n == "logit_scale") else decay).append(q)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
                                lr=lr, betas=(0.9, 0.995), eps=1e-8)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 0.5 * (1.0 + math.cos(math.pi * s / total))))
        return opt, sched, trainable

    def train_step(bm, ls, ah, opt, sched, trainable, idx_global, sid, g_rng):
        bm.backbone.proj_in.sid = sid
        if args.sub_aug:
            use3 = torch.rand(len(idx_global), device=dev, generator=g_rng) < 0.5
            drop_r = torch.randint(0, 4, (len(idx_global),), device=dev, generator=g_rng)
            mask = torch.ones((len(idx_global), 4), device=dev)
            mask[use3, drop_r[use3]] = 0.0
            xr = X_raw[sid, idx_global].float()
            xb_in = (xr * mask[:, :, None, None]).sum(1) / mask.sum(1)[:, None, None]
        else:
            xb_in = X[sid, idx_global]
        xb = shift_batch(xb_in, args.jitter, g_rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c_raw = bm(xb)
            if args.target in ("fused", "h14"):
                z = nz(c_raw, dim=-1)
                if args.target == "fused":
                    with torch.no_grad():
                        y = nz(vision_teacher(**{ek: ev[idx_global] for ek, ev in emb_tr.items()}), dim=-1)
                    ref_u = F_tr_u[idx_global]
                else:
                    y = H14_u[idx_global]
                    ref_u = H14_u[idx_global]
                scale = ls.exp().clamp(max=100.0)
                lg = scale * (z @ y.T)
                tb = torch.arange(len(idx_global), device=dev)
                loss_nce = 0.5 * (F.cross_entropy(lg, tb) + F.cross_entropy(lg.T, tb))
                loss = (1.0 - args.soft_w) * loss_nce
                if args.soft_w > 0:
                    with torch.no_grad():
                        p_t = F.softmax((ref_u @ ref_u.T) / 0.07, dim=-1)
                    loss_kl = 0.5 * (F.kl_div(F.log_softmax(lg, dim=-1), p_t, reduction="batchmean")
                                     + F.kl_div(F.log_softmax(lg.T, dim=-1), p_t, reduction="batchmean"))
                    loss = loss + args.soft_w * loss_kl
                if args.conc_w > 0:
                    loss = loss + args.conc_w * (1.0 - (z * C_cent[concept[idx_global]]).sum(-1)).mean()
                if args.cos_w > 0:
                    loss = loss + args.cos_w * (1.0 - (z * ref_u).sum(-1)).mean()
                if args.aux_w > 0 and len(ah) > 0:
                    valid_s = ok_mask[idx_global].float()
                    la_h = (1.0 - (nz(ah["h14"](c_raw), dim=-1) * H14_u[idx_global]).sum(-1)).mean()
                    la_b = (1.0 - (nz(ah["b32"](c_raw), dim=-1) * B32_u[idx_global]).sum(-1)).mean()
                    la_s = ((1.0 - (nz(ah["spat"](c_raw), dim=-1) * Sc_tr_u[idx_global]).sum(-1)) * valid_s).sum() / valid_s.sum().clamp_min(1.0)
                    loss = loss + args.aux_w * (la_h + la_b + la_s) / 3.0
            else:
                zc = nz(c_raw - c_raw.mean(0, keepdim=True), dim=-1)
                yc = Sc_tr_u[idx_global]
                lg = ls.exp().clamp(max=100.0) * (zc @ yc.T)
                tb = torch.arange(len(idx_global), device=dev)
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
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        return float(loss)

    g = torch.Generator(device=dev).manual_seed(args.seed)
    spr = math.ceil(n_tr / args.bs)
    opt, sched, tr_params = make_opt(brain_model, logit_scale, aux_heads, args.lr, spr * S * args.epochs)
    t0_wall = time.time()
    brain_model.train()
    if vision_teacher is not None:
        vision_teacher.train()
    for _ in range(args.epochs):
        off = torch.randint(0, S, (n_tr,), device=dev, generator=g)
        for r in range(S):
            perm = torch.randperm(n_tr, device=dev, generator=g)
            for b in range(spr):
                loc = perm[b * args.bs:(b + 1) * args.bs]
                train_step(brain_model, logit_scale, aux_heads, opt, sched, tr_params, tr_idx[loc], (off[loc] + r) % S, g)
    joint_s = time.time() - t0_wall

    def export_sd(bm, j):
        sd = {}
        for k, v in bm.state_dict().items():
            if ".proj_in.lin." in k:
                pre, rest = k.split(".proj_in.lin.")
                jj, name = rest.split(".", 1)
                if int(jj) == j:
                    sd[f"{pre}.proj_in.{name}"] = v.cpu()
            else:
                sd[k] = v.cpu()
        return sd

    tgt200 = torch.arange(200, device=dev)
    ref_te_c = None
    if args.target == "fused":
        ref_te = nz(torch.from_numpy(np.load(PREP / "F_test.npy")).float().to(dev), dim=1)
        ref_te_c = nz(ref_te - ref_te.mean(0, keepdim=True), dim=1)
    elif args.target == "h14":
        ref_te = nz(torch.load(PREP / "emb_test.pt", weights_only=True)[EMB_KEYS[0]].float().to(dev), dim=1)
        ref_te_c = nz(ref_te - ref_te.mean(0, keepdim=True), dim=1)
    t1s = []
    for j, s in enumerate(subs):
        m = copy.deepcopy(brain_model)
        ls_j = copy.deepcopy(logit_scale)
        ls_j.requires_grad_(False)
        ah_j = copy.deepcopy(aux_heads)
        ah_j.requires_grad_(False)
        m.train()
        m.requires_grad_(False)
        m.backbone.proj_in.lin[j].requires_grad_(True)
        o2, s2, tp2 = make_opt(m, ls_j, ah_j, args.ft_lr, spr * args.ft_epochs, lambda n: f".proj_in.lin.{j}." in n)
        for _ in range(args.ft_epochs):
            perm = torch.randperm(n_tr, device=dev, generator=g)
            for b in range(spr):
                loc = perm[b * args.bs:(b + 1) * args.bs]
                train_step(m, ls_j, ah_j, o2, s2, tp2, tr_idx[loc], torch.full((len(loc),), j, device=dev), g)

        m.eval()
        out = RUNS / "enc" / args.tag / f"sub-{s:02d}" / f"fold{args.fold}_seed{args.seed}"
        out.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            m.backbone.proj_in.sid = torch.full((200,), j, device=dev)
            c_ivw = m(XT_ivw[j]).float()
        met = dict(args=dict(vars(args), sub=s, chan="correct", inorm=0, mode="r4"), joint_seconds=joint_s)
        if ref_te_c is not None:
            c_c = nz(nz(c_ivw, dim=1) - nz(c_ivw, dim=1).mean(0, keepdim=True), dim=1)
            t1 = float(((c_c @ ref_te_c.T).argmax(1) == tgt200).float().mean())
            t1s.append(t1)
            met["test_ivw_cen_top1"] = t1

        if args.fold >= 0 and len(ho_idx) > 0:
            with torch.no_grad():
                oof_chunks = []
                for i in range(0, len(ho_idx), 2048):
                    sl = ho_idx[i:i + 2048]
                    m.backbone.proj_in.sid = torch.full((len(sl),), j, device=dev)
                    oof_chunks.append(m(X[j, sl]).float())
                c_oof = torch.cat(oof_chunks)
            np.save(out / "cond_oof_r4.npy", c_oof.cpu().numpy())
            np.save(out / "oof_index.npy", ho_idx.cpu().numpy())

        np.save(out / "cond_test_ivw.npy", c_ivw.cpu().numpy())
        torch.save(export_sd(m, j), out / "brain_model.pt")
        write_json(out / "metrics.json", met)
        del m

    msg = f"mean cen top1={np.mean(t1s):.4f}" if t1s else "spat_ms512"
    print(f"{args.tag} fold{args.fold} seed{args.seed} [t={args.t0}:{args.t1}] done in {time.time() - t0_wall:.0f}s | {msg}", flush=True)
