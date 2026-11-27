"""Phase 1: Strictly Controlled 4-Arm Comparison for Within-Concept 2x2 EEG-Swap Denoising Contrast.

All arms start from the EXACT SAME tr14 baseline checkpoint (arm0_tr14_xbrain_s2025.pt, ZERO fold-0 access),
train strictly on tr14 (folds 1..4 = 1,318 concepts x 10 photos x 10 subjects = 131,800 pairs),
and are evaluated on sealed unseen fold-0 concepts (R=4 real EEG trials, 4-step SDXL-Turbo, nsel=1):
  - Arm0_Base              : Unmodified O13/O14 shared-trunk tr14 baseline.
  - Arm1_RepIntra          : Same architecture, 800 steps with representation-only within-concept 2x2 swap
                             + 10-way Intra-Concept InfoNCE on (F_perp, 70-layer L_KV, H14, B32), NO fn_mask.
  - Arm2a_SwapDiff_Cone    : Exact same architecture (0 extra params!), 800 steps backpropagating frozen
                             SDXL-Turbo UNet's doubly-balanced 2x2 EEG-swap denoising contrast:
                               Delta_swap = [d(A, E_A) + d(B, E_B)] - [d(A, E_B) + d(B, E_A)] < 0
                             plus matched denoising loss d(A, E_A) + d(B, E_B) + inter-concept anchor L_base.
  - Arm2b_SwapDiff_CondLyr : Same as Arm2a + zero-initialized low-rank conditioning residual Delta_ips
                             (1024 -> 128 -> 16x2048) steering SDXL-Turbo's 16 IP cross-attention tokens directly.

Evaluation on D_intra50 (50 sealed fold-0 concepts x 10 photos = 500 unseen photos, 2,250 pairs/sub, DEV_SUBS={2,5,8,9}):
  - Layer 1: Representation Intra10 & 2x2 Swap (327 concepts & 50 concepts).
  - Layer 2: Frozen UNet Denoising-Error 2x2 Swap Win-Rate & Intra10 (t in {749, 499, 249}).
  - Layer 3a (Controlled Swap): Fixed concept-mean z0 + fixed concept seed, swap EEG condition only (smap=0.625).
  - Layer 3b (Full Free-EEG Generation): EEG-predicted z0(E) + EEG-predicted condition(E), 4 NFE, nsel=1, ZERO GT input.
    Measures Intra10 Top-1, doubly-balanced 2x2 Swap Win-Rate across CLIP-L14, Inception, Alex5, Alex2, PixCorr, SSIM,
    plus Global 500-way CLIP Top-1 (inter-concept preservation check).
"""
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity

ROOT = Path(os.environ["RAD_ROOT"])
for d in [ROOT / "code/exp_w7", ROOT / "code/exp_w6", ROOT / "code/rad", ROOT / "code/exp", ROOT / "code/Brain-HIVE"]:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

from common import EMB_KEYS, HIVE, PREP, PRIOR, RUNS, TURBO, write_json
from deep_enc import sample_sub_repeats
from exp0_intra10_swap_gate import (
    ALL_SUBS,
    DEV_SUBS,
    OUT_DIR,
    fit_tr14_ridge_z0,
    get_concept_splits,
    intra10_and_swap_from_sim,
)
from o10_factorized import _perp_dir
from o11_tri_stream_bigg import OnConeKVProjector, _hybrid_infonce, _lp_op, _simple_infonce, _to_cone
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder, _xbrain_cone_nce, _xbrain_nce
from train_flex import pyramid_feats, shift_batch

from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel

nz = F.normalize


def per_sample_cone(fp: torch.Tensor, R_tr: float, alpha_tr: float, beta_tr: float, u_tr: torch.Tensor) -> torch.Tensor:
    """Strictly per-sample deterministic projection onto the valid F_train cone (ZERO batch-mean dependency)."""
    u_hat = nz(fp.float(), dim=-1)
    u = u_tr.to(u_hat.device, dtype=u_hat.dtype)
    d_perp = nz(u_hat - (u_hat @ u.T) * u, dim=-1)
    return R_tr * nz(alpha_tr * u + beta_tr * d_perp, dim=-1)


class ResidualIPCondLayer(nn.Module):
    """Zero-initialized low-rank residual conditioning layer: h_eeg [1024] -> Delta_ips [4, 2048]."""

    def __init__(self, d_in: int = 1024, rank: int = 128, n_tok: int = 4, d_tok: int = 2048, max_rel_norm: float = 0.35):
        super().__init__()
        self.n_tok = n_tok
        self.d_tok = d_tok
        self.max_rel_norm = max_rel_norm
        self.ln = nn.LayerNorm(d_in)
        self.down = nn.Linear(d_in, rank)
        self.up = nn.Linear(rank, n_tok * d_tok, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, h_eeg: torch.Tensor, ips_base: torch.Tensor) -> torch.Tensor:
        delta = self.up(F.silu(self.down(self.ln(h_eeg.float())))).view(-1, self.n_tok, self.d_tok)
        # Soft-bound relative norm so residual never destabilizes frozen SDXL-Turbo cross-attention
        base_n = ips_base.float().norm(dim=(1, 2), keepdim=True).clamp_min(1e-4)
        del_n = delta.norm(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        scale = torch.minimum(torch.ones_like(del_n), (self.max_rel_norm * base_n) / del_n)
        return ips_base + (delta * scale).to(ips_base.dtype)


def forward_with_trunk(model: CrossSubjectHorizonEncoder, x: torch.Tensor, sid: int, r_eff=4.0):
    """Exact CrossSubjectHorizonEncoder.forward_sub returning (fp, hp, bp, c_ens)."""
    xg = model.ebs[sid](x, r_eff=r_eff)
    hs = [
        model.stem_norms[i](model.sub_stems[sid][i](model.in_drop(pyramid_feats(xg[..., :t_h])))) + model.sub_emb.weight[sid]
        for i, t_h in enumerate(model.horizons)
    ]
    cs = model.block(torch.cat(hs, dim=0)).chunk(len(model.horizons), dim=0)
    w_h = F.softmax(model.horizon_logits, dim=0)
    b_af = model.block.layers[-1].norm.bias.to(cs[0].dtype)
    d0 = cs[0] - b_af
    d_mix = sum(w_h[i] * (cs[i] - b_af) for i in range(len(cs)))
    scale = d0.norm(dim=-1, keepdim=True) / d_mix.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    c_ens = b_af + d_mix * scale
    return model.head_f(c_ens), model.head_h14(c_ens), model.head_b32(c_ens), c_ens


def build_tr14_resources(dev: str, subjects=ALL_SUBS, *, full_training=False):
    concept_np, ok_np, tr_idx_np, f0_all10, f0_idx, sel50_cids, sel50_idx = get_concept_splits(full_training)
    # Index complete ten-photo groups from the active training scope.
    g2loc = {int(g): i for i, g in enumerate(tr_idx_np)}
    tr14_c_groups = []
    for cid in np.unique(concept_np[tr_idx_np]):
        g_ids = np.where(concept_np == cid)[0]
        if len(g_ids) == 10 and ok_np[g_ids].all():
            tr14_c_groups.append([g2loc[int(g)] for g in g_ids])
    tr14_c_groups = np.array(tr14_c_groups, dtype=np.int64)  # [1318, 10]

    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")[tr_idx_np]).float().to(dev)
    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    H14_tr = emb_tr[EMB_KEYS[0]][tr_idx_np].float().to(dev)
    B32_tr = emb_tr[EMB_KEYS[1]][tr_idx_np].float().to(dev)
    lat_tr = np.load(PREP / "lat512_train.npy")[tr_idx_np]  # keep fp16/fp32 on CPU (~430MB)
    lat_tr_cpu = torch.from_numpy(lat_tr).half()
    del lat_tr

    x_tr_all = {}
    for s in subjects:
        arr = np.load(PREP / f"sub-{s:02d}_train.npy")[tr_idx_np]
        x_tr_all[s - 1] = torch.from_numpy(arr).half()

    kv_full = OnConeKVProjector(str(PREP / "unet_kv_metric.pt"), device=dev)
    kv_full.calibrate_fold(F_tr)

    return {
        "concept_np": concept_np,
        "tr_idx_np": tr_idx_np,
        "f0_all10": f0_all10,
        "f0_idx": f0_idx,
        "sel50_cids": sel50_cids,
        "sel50_idx": sel50_idx,
        "tr14_c_groups": tr14_c_groups,
        "F_tr": F_tr,
        "H14_tr": H14_tr,
        "B32_tr": B32_tr,
        "lat_tr_cpu": lat_tr_cpu,
        "x_tr_all": x_tr_all,
        "subject_ids": np.array(list(x_tr_all)),
        "kv_full": kv_full,
    }


def compute_base_anchor_loss(model, res, sid_a, sid_b, idx_cpu, idx_dev, g, dev, ls_f, ls_h, ls_b, ls_x, remove_fn_mask=False):
    """Compact inter-concept anchor loss on random tr14 batch (preserves global 200-way/500-way semantics)."""
    kv_full = res["kv_full"]
    xa, ka = sample_sub_repeats(res["x_tr_all"][sid_a][idx_cpu].float().to(dev), g)
    xb, kb = sample_sub_repeats(res["x_tr_all"][sid_b][idx_cpu].float().to(dev), g)
    xa, xb = shift_batch(xa, 2, g), shift_batch(xb, 2, g)

    sc_f = ls_f.float().exp().clamp(5.0, 100.0)
    sc_h = ls_h.float().exp().clamp(5.0, 100.0)
    sc_b = ls_b.float().exp().clamp(5.0, 100.0)
    sc_x = ls_x.float().exp().clamp(5.0, 100.0)

    with torch.no_grad():
        ff = res["F_tr"][idx_dev].float()
        uf = nz(ff, dim=-1)
        df_t = nz(uf - (uf @ kv_full.u_tr.T) * kv_full.u_tr, dim=-1)
        ykv_f = nz(kv_full.kv.project_kv(ff) - kv_full.kv.mu_kv, dim=-1)
        ht = nz(res["H14_tr"][idx_dev].float(), dim=-1)
        bt = nz(res["B32_tr"][idx_dev].float(), dim=-1)
        eye = torch.eye(len(idx_dev), dtype=torch.bool, device=dev)
        off_diag = ~eye
        if remove_fn_mask:
            fn_mask = None
        else:
            c_b = torch.from_numpy(res["concept_np"][res["tr_idx_np"][idx_cpu.numpy()]]).long().to(dev)
            fn_mask = (c_b[:, None] == c_b[None, :]) & off_diag
        sf_sim = df_t @ df_t.T
        sh, sb = ht @ ht.T, bt @ bt.T
        mf = 0.15 * sc_x.detach() * F.relu(sf_sim - sf_sim[off_diag].mean()) * off_diag.float()
        mh = 0.15 * sc_x.detach() * F.relu(sh - sh[off_diag].mean()) * off_diag.float()
        mb = 0.15 * sc_x.detach() * F.relu(sb - sb[off_diag].mean()) * off_diag.float()

    def _one(fp, hp, bp):
        dfp, zk_f = kv_full.cone_and_kv(fp)
        lf = _hybrid_infonce(dfp, zk_f, df_t, ykv_f, sc_f)
        lh = _simple_infonce(nz(hp.float(), dim=-1), ht, sc_h, fn_mask=fn_mask)
        lb = _simple_infonce(nz(bp.float(), dim=-1), bt, sc_b, fn_mask=fn_mask)
        return 0.50 * lf + 0.30 * lh + 0.20 * lb

    fa, ha, ba, _ = forward_with_trunk(model, xa, sid_a, r_eff=ka)
    fb, hb, bb, _ = forward_with_trunk(model, xb, sid_b, r_eff=kb)
    l_vis = 0.5 * (_one(fa, ha, ba) + _one(fb, hb, bb))
    l_xb = (
        0.35 * _xbrain_cone_nce(fa, fb, kv_full.u_tr, sc_x, mf)
        + 0.40 * _xbrain_nce(ha, hb, sc_x, mh)
        + 0.25 * _xbrain_nce(ba, bb, sc_x, mb)
    )
    return l_vis + res.get("crossbrain_weight", 0.25) * l_xb


def train_arm1_rep_intra(arm0_ckpt: Path, res: dict, steps: int = 800, seed: int = 2025, dev: str = "cuda:0") -> Path:
    """Arm 1: Representation-only within-concept 2x2 swap + 10-way Intra-Concept InfoNCE (F_perp, L_KV, H14, B32)."""
    out_p = OUT_DIR / f"arm1_rep_intra_st{steps}_s{seed}.pt"
    crossbrain_weight = res.get("crossbrain_weight", 0.25)
    if out_p.exists():
        saved = torch.load(out_p, map_location="meta", weights_only=True)
        if saved.get("crossbrain_weight", 0.25) != crossbrain_weight:
            raise ValueError("Cached representation fit uses a different cross-person loss weight")
        print(f"[CACHED] {out_p}", flush=True)
        return out_p

    t0 = time.time()
    pack = torch.load(arm0_ckpt, map_location="cpu", weights_only=True)
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
    kv_full = res["kv_full"]
    u_tr = kv_full.u_tr

    ls_f = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_h = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_b = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_x = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_in = nn.Parameter(torch.tensor(2.8, device=dev))

    opt = torch.optim.AdamW(list(model.parameters()) + [ls_f, ls_h, ls_b, ls_x, ls_in], lr=1.5e-4, weight_decay=0.03, betas=(0.9, 0.995))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    g = torch.Generator(device=dev).manual_seed(seed * 31 + 7)
    rng = np.random.default_rng(seed * 31 + 7)
    c_groups = res["tr14_c_groups"]  # [1318, 10]
    n_c = len(c_groups)
    n_tr = len(res["tr_idx_np"])

    print(f"[TRAIN Arm1_RepIntra] steps={steps}, crossbrain_weight={crossbrain_weight}, "
          "16 same-concept pairs + 8 full 10-img concepts/step ...", flush=True)
    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        # 1. Inter-concept anchor (bs=512, fn_mask REMOVED)
        idx_dev = torch.randint(0, n_tr, (512,), device=dev, generator=g)
        idx_cpu = idx_dev.cpu()
        s_pair = rng.choice(res["subject_ids"], size=2, replace=len(res["subject_ids"]) == 1)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            l_base = compute_base_anchor_loss(
                model, res, int(s_pair[0]), int(s_pair[1]), idx_cpu, idx_dev, g, dev,
                ls_f, ls_h, ls_b, ls_x, remove_fn_mask=True,
            )

            # 2. Within-concept 10-way InfoNCE + exact 2x2 representation swap on 8 concepts (80 images = 360 pairs!)
            # Same subject per concept, R=4 mean EEG!
            sid_c = int(rng.choice(res["subject_ids"]))
            c_sel = rng.choice(n_c, size=8, replace=False)
            g_locs = c_groups[c_sel].reshape(-1)  # [80]
            g_locs_t = torch.from_numpy(g_locs).long()
            x_c = res["x_tr_all"][sid_c][g_locs_t].float().mean(1).to(dev)  # [80, 17, 250] R=4 exact!
            x_c = shift_batch(x_c, 2, g)
            fp, hp, bp, _ = forward_with_trunk(model, x_c, sid_c, r_eff=4.0)

            idx_c_dev = g_locs_t.to(dev)
            with torch.no_grad():
                ff = res["F_tr"][idx_c_dev].float()
                uf = nz(ff, dim=-1)
                df_t = nz(uf - (uf @ u_tr.T) * u_tr, dim=-1)
                ykv_t = nz(kv_full.kv.project_kv(ff) - kv_full.kv.mu_kv, dim=-1)
                ht = nz(res["H14_tr"][idx_c_dev].float(), dim=-1)
                bt = nz(res["B32_tr"][idx_c_dev].float(), dim=-1)

            # Per-sample cone & KV (matching inference)
            c_cone = per_sample_cone(fp, pack["cone"]["R"], pack["cone"]["alpha"], pack["cone"]["beta"], u_tr)
            df_p = nz(nz(fp.float(), dim=-1) - (nz(fp.float(), dim=-1) @ u_tr.T) * u_tr, dim=-1)
            zk_p = nz(kv_full.kv.project_kv(c_cone) - kv_full.kv.mu_kv, dim=-1)
            hp_n = nz(hp.float(), dim=-1)
            bp_n = nz(bp.float(), dim=-1)
            sc_in = ls_in.float().exp().clamp(5.0, 50.0)

            # Within each of the 8 concepts (10x10), compute symmetric 10-way InfoNCE + 2x2 swap logistic loss
            def _intra_loss(P, T):
                Pb = P.view(8, 10, -1)
                Tb = T.view(8, 10, -1)
                S = torch.bmm(Pb, Tb.transpose(1, 2))  # [8, 10, 10]
                tgt = torch.arange(10, device=dev).unsqueeze(0).expand(8, 10)
                ce = 0.5 * (
                    F.cross_entropy((sc_in * S).reshape(80, 10), tgt.reshape(80))
                    + F.cross_entropy((sc_in * S.transpose(1, 2)).reshape(80, 10), tgt.reshape(80))
                )
                iu, ju = torch.triu_indices(10, 10, offset=1, device=dev)
                # 2x2 swap margin: S[i,i] + S[j,j] > S[i,j] + S[j,i]
                delta = (S[:, iu, ju] + S[:, ju, iu]) - (S[:, iu, iu] + S[:, ju, ju])
                sw = F.softplus(8.0 * delta).mean()
                return ce + 0.5 * sw

            l_intra = (
                0.35 * _intra_loss(df_p, df_t)
                + 0.35 * _intra_loss(zk_p, ykv_t)
                + 0.20 * _intra_loss(hp_n, ht)
                + 0.10 * _intra_loss(bp_n, bt)
            )
            loss = 0.5 * l_base + 1.0 * l_intra

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()
        if step in (1, 200, 400, 600, 800):
            print(f"  [Arm1] step {step:4d}/{steps} base={float(l_base.detach()):.4f} intra={float(l_intra.detach()):.4f} ({time.time()-t0:.1f}s)", flush=True)

    model.eval()
    torch.save(
        {
            "state_dict": {k: v.half().cpu() for k, v in model.state_dict().items()},
            "cond_layer": None,
            "cone": pack["cone"],
            "arm": "Arm1_RepIntra",
            "steps": steps,
            "crossbrain_weight": crossbrain_weight,
        },
        out_p,
    )
    print(f"[SAVED] {out_p} ({time.time()-t0:.1f}s)", flush=True)
    return out_p


def train_arm2_swapdiff(
    arm0_ckpt: Path,
    res: dict,
    pipe_unet,
    alphas_cumprod: torch.Tensor,
    adapter,
    proj,
    pe,
    ppe,
    add_time_ids,
    use_cond_layer: bool = False,
    steps: int = 800,
    seed: int = 2025,
    dev: str = "cuda:0",
) -> Path:
    """Arm 2a / 2b: Backpropagate frozen SDXL-Turbo UNet's doubly-balanced 2x2 EEG-swap denoising contrast."""
    tag = "arm2b_swapdiff_condlyr" if use_cond_layer else "arm2a_swapdiff_cone"
    out_p = OUT_DIR / f"{tag}_st{steps}_s{seed}.pt"
    if out_p.exists():
        print(f"[CACHED] {out_p}", flush=True)
        return out_p

    t0 = time.time()
    pack = torch.load(arm0_ckpt, map_location="cpu", weights_only=True)
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
    cond_layer = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=16, d_tok=2048, max_rel_norm=0.35).to(dev) if use_cond_layer else None

    R_tr = pack["cone"]["R"]
    a_tr = pack["cone"]["alpha"]
    b_tr = pack["cone"]["beta"]
    u_tr = pack["cone"]["u_tr"].to(dev).float()

    ls_f = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_h = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_b = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_x = nn.Parameter(torch.tensor(2.8, device=dev))

    params = [{"params": list(model.parameters()) + [ls_f, ls_h, ls_b, ls_x], "lr": 1.0e-4, "weight_decay": 0.03}]
    if cond_layer is not None:
        params.append({"params": list(cond_layer.parameters()), "lr": 3.0e-4, "weight_decay": 0.01})
    opt = torch.optim.AdamW(params, betas=(0.9, 0.995))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    # Enable gradient checkpointing on frozen bf16 UNet!
    pipe_unet.eval().requires_grad_(False)
    pipe_unet.enable_gradient_checkpointing()
    adapter.eval().requires_grad_(False)
    proj.eval().requires_grad_(False)

    g = torch.Generator(device=dev).manual_seed(seed * 53 + 11)
    rng = np.random.default_rng(seed * 53 + 11)
    c_groups = res["tr14_c_groups"]  # [1318, 10]
    n_c = len(c_groups)
    n_tr = len(res["tr_idx_np"])

    # Exact active 4-step SDXL-Turbo timesteps (under smap=0.625, t=999 is overwritten by z0)
    # Calibrated from Phase 0 Layer 2 Delta_swap SDs:
    T_CHOICES = [749, 499, 249]
    T_PROBS = [0.40, 0.40, 0.20]
    BETA_BY_T = {749: 200.0, 499: 150.0, 249: 200.0}

    n_micro = 4        # 4 micro-batches x 4 pairs = 16 same-subject same-concept pairs per optimizer step!
    pairs_per_mb = 4   # 4 pairs -> 16 UNet forwards per micro-batch (fits ~12 GB VRAM in bf16 with GC!)

    print(f"[TRAIN {tag}] steps={steps}, pairs/step={n_micro*pairs_per_mb}, use_cond_layer={use_cond_layer} ...", flush=True)
    model.train()
    if cond_layer is not None:
        cond_layer.train()

    run_sw, run_delta, run_rec = [], [], []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        # 1. Inter-concept anchor (weight 0.5, fn_mask=False so it doesn't fight within-concept discrimination!)
        idx_dev = torch.randint(0, n_tr, (512,), device=dev, generator=g)
        idx_cpu = idx_dev.cpu()
        s_pair = rng.choice(10, size=2, replace=False)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            l_base = 0.5 * compute_base_anchor_loss(
                model, res, int(s_pair[0]), int(s_pair[1]), idx_cpu, idx_dev, g, dev,
                ls_f, ls_h, ls_b, ls_x, remove_fn_mask=True,
            )
        l_base.backward()
        if step == 1:
            g_base = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in model.parameters() if p.grad is not None))

        # 2. Doubly-balanced 2x2 EEG-Swap Denoising Contrast through FROZEN SDXL-Turbo UNet
        step_rec_val, step_sw_val, step_d_val, step_win = 0.0, 0.0, 0.0, 0.0
        for mb in range(n_micro):
            # Sample 1 subject & 4 distinct concepts, 2 distinct photos (A, B) per concept -> same person, same concept!
            sid = int(rng.integers(0, 10))
            c_sel = rng.choice(n_c, size=pairs_per_mb, replace=False)
            loc_a, loc_b = [], []
            for ci in c_sel:
                ab_idx = rng.choice(10, size=2, replace=False)
                loc_a.append(c_groups[ci, ab_idx[0]])
                loc_b.append(c_groups[ci, ab_idx[1]])
            loc_a_t = torch.tensor(loc_a, dtype=torch.long)
            loc_b_t = torch.tensor(loc_b, dtype=torch.long)

            # Exact R=4 real EEG mean for photo A and photo B from the SAME subject
            ea = res["x_tr_all"][sid][loc_a_t].float().mean(1).to(dev)  # [P, 17, 250]
            eb = res["x_tr_all"][sid][loc_b_t].float().mean(1).to(dev)
            ea, eb = shift_batch(ea, 2, g), shift_batch(eb, 2, g)

            z0_a = res["lat_tr_cpu"][loc_a_t].float().to(dev)  # [P, 4, 64, 64]
            z0_b = res["lat_tr_cpu"][loc_b_t].float().to(dev)

            # Sample one active timestep t per micro-batch (shared across A & B so scales match identically!)
            t_int = int(rng.choice(T_CHOICES, p=T_PROBS))
            beta_t = BETA_BY_T[t_int]
            ab_val = float(alphas_cumprod[t_int])
            sq_ab, sq_1mab = math.sqrt(ab_val), math.sqrt(1.0 - ab_val)

            eps_a = torch.randn(z0_a.shape, device=dev, generator=g, dtype=torch.float32)
            eps_b = torch.randn(z0_b.shape, device=dev, generator=g, dtype=torch.float32)
            zt_a = (sq_ab * z0_a + sq_1mab * eps_a).to(torch.bfloat16)
            zt_b = (sq_ab * z0_b + sq_1mab * eps_b).to(torch.bfloat16)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                e_both = torch.cat([ea, eb], dim=0)  # [2P, 17, 250]
                fp_both, _, _, h_both = forward_with_trunk(model, e_both, sid, r_eff=4.0)
                c_both = per_sample_cone(fp_both, R_tr, a_tr, b_tr, u_tr).to(torch.bfloat16)
                ips_both = proj(adapter(c_both))  # [2P, 16, 2048]
                if cond_layer is not None:
                    ips_both = cond_layer(h_both, ips_both)
                ips_a, ips_b = ips_both[:pairs_per_mb], ips_both[pairs_per_mb:]

                # Pack all 4 combinations into 1 UNet call [4P, 4, 64, 64]: (A,E_A), (B,E_B), (A,E_B), (B,E_A)
                zt_4 = torch.cat([zt_a, zt_b, zt_a, zt_b], dim=0)
                ips_4 = torch.cat([ips_a, ips_b, ips_b, ips_a], dim=0)
                B4 = zt_4.shape[0]
                t_ten = torch.full((B4,), t_int, device=dev, dtype=torch.long)

                pred_4 = pipe_unet(
                    zt_4,
                    t_ten,
                    encoder_hidden_states=pe.expand(B4, -1, -1),
                    added_cond_kwargs={"text_embeds": ppe.expand(B4, -1), "time_ids": add_time_ids.expand(B4, -1)},
                    cross_attention_kwargs={"ip_hidden_states": ips_4},
                ).sample.float()

                p_aa, p_bb, p_ab, p_ba = pred_4.chunk(4, dim=0)
                d_aa = (p_aa - eps_a).pow(2).mean(dim=(1, 2, 3))
                d_bb = (p_bb - eps_b).pow(2).mean(dim=(1, 2, 3))
                d_ab = (p_ab - eps_a).pow(2).mean(dim=(1, 2, 3))
                d_ba = (p_ba - eps_b).pow(2).mean(dim=(1, 2, 3))

                l_rec = 0.5 * (d_aa + d_bb).mean()
                delta_sw = (d_aa + d_bb) - (d_ab + d_ba)  # want < 0!
                l_sw = F.softplus(beta_t * delta_sw).mean()
                l_diff = (20.0 * l_rec + 1.0 * l_sw) / float(n_micro)

            l_diff.backward()
            step_rec_val += float(l_rec.detach()) / n_micro
            step_sw_val += float(l_sw.detach()) / n_micro
            step_d_val += float(delta_sw.detach().mean()) / n_micro
            step_win += float((delta_sw.detach() < 0.0).float().mean()) / n_micro

        if step == 1:
            g_tot = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in model.parameters() if p.grad is not None))
            g_cl = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in cond_layer.parameters() if p.grad is not None)) if cond_layer is not None else 0.0
            print(f"  [{tag} STEP-1 DIAG] grad_base={g_base:.3f} grad_tot={g_tot:.3f} grad_condlyr={g_cl:.3f} mem_MB={torch.cuda.max_memory_allocated()//(1024*1024)}", flush=True)

        all_p = list(model.parameters()) + [ls_f, ls_h, ls_b, ls_x] + (list(cond_layer.parameters()) if cond_layer is not None else [])
        torch.nn.utils.clip_grad_norm_(all_p, 2.0)
        opt.step()
        sched.step()

        run_sw.append(step_win)
        run_delta.append(step_d_val)
        run_rec.append(step_rec_val)
        if step in (1, 10, 100, 200, 400, 600, 800):
            w_rec = np.mean(run_rec[-50:])
            w_sw = np.mean(run_sw[-50:]) * 100.0
            w_dl = np.mean(run_delta[-50:])
            elapsed = time.time() - t0
            eta = (elapsed / step) * (steps - step)
            print(
                f"  [{tag}] step {step:4d}/{steps} base={float(l_base.detach()):.3f} rec={w_rec:.4f} "
                f"sw_loss={step_sw_val:.3f} tr_swWin={w_sw:.1f}% delta={w_dl:.2e} ({elapsed:.0f}s, ETA {eta:.0f}s)",
                flush=True,
            )

    model.eval()
    if cond_layer is not None:
        cond_layer.eval()
    torch.save(
        {
            "state_dict": {k: v.half().cpu() for k, v in model.state_dict().items()},
            "cond_layer": {k: v.half().cpu() for k, v in cond_layer.state_dict().items()} if cond_layer is not None else None,
            "cone": pack["cone"],
            "arm": tag,
            "steps": steps,
        },
        out_p,
    )
    print(f"[SAVED] {out_p} ({time.time()-t0:.1f}s)", flush=True)
    return out_p


@torch.inference_mode()
def evaluate_arm_complete(
    arm_name: str,
    ckpt_p: Path,
    res: dict,
    pipe,
    p2,
    vae32,
    adapter,
    proj,
    pe,
    ppe,
    add_time_ids,
    lat50: torch.Tensor,
    gt_rgb50: torch.Tensor,
    gt_feats: dict,
    z0_by_sub: dict,
    dev: str = "cuda:0",
):
    """Evaluate one arm across Layer 1 (Rep), Layer 2 (UNet Denoise), Layer 3a (FixedZ0 Swap), Layer 3b (FreeGen O14)."""
    from metrics import _nets

    pack = torch.load(ckpt_p, map_location="cpu", weights_only=True)
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
    model.eval()

    cond_layer = None
    if pack.get("cond_layer") is not None:
        cond_layer = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev)
        cond_layer.load_state_dict({k: v.float() for k, v in pack["cond_layer"].items()})
        cond_layer.eval()

    R_tr = pack["cone"]["R"]
    a_tr = pack["cone"]["alpha"]
    b_tr = pack["cone"]["beta"]
    u_tr = pack["cone"]["u_tr"].to(dev).float()
    kv_full = res["kv_full"]

    F_all = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    H14_all = emb_tr[EMB_KEYS[0]].float()

    f0_idx = res["f0_idx"]
    sel50_idx = res["sel50_idx"]
    f0_pos = {int(g): i for i, g in enumerate(f0_idx)}
    loc50 = np.array([f0_pos[int(g)] for g in sel50_idx], dtype=np.int64)

    def _cos_blk(P, G, nc):
        return torch.bmm(nz(P.float(), dim=-1).view(nc, 10, -1), nz(G.float(), dim=-1).view(nc, 10, -1).transpose(1, 2)).cpu().numpy()

    per_sub = {}
    nets = _nets()

    for sub in DEV_SUBS:
        sid = sub - 1
        X_raw = np.load(PREP / f"sub-{sub:02d}_train.npy")
        xm4_f0 = torch.from_numpy(X_raw[f0_idx].mean(1)).float()  # [3270, 17, 250] R=4 exact!
        del X_raw

        fp_l, hp_l, h_l = [], [], []
        for i in range(0, len(xm4_f0), 256):
            fp, hp, _, h_tr = forward_with_trunk(model, xm4_f0[i:i + 256].to(dev), sid, r_eff=4.0)
            fp_l.append(fp.cpu())
            hp_l.append(hp.cpu())
            h_l.append(h_tr.cpu())
        fp_f0 = torch.cat(fp_l, 0)
        hp_f0 = torch.cat(hp_l, 0)
        h_f0 = torch.cat(h_l, 0)

        # Layer 1 on all 327 concepts (3,270 imgs)
        df_p = nz(nz(fp_f0.to(dev), dim=-1) - (nz(fp_f0.to(dev), dim=-1) @ u_tr.T) * u_tr, dim=-1)
        df_g = _perp_dir(F_all[f0_idx].to(dev), u_tr)
        m_f327 = intra10_and_swap_from_sim(_cos_blk(df_p, df_g, len(res["f0_all10"])))
        c_f0 = per_sample_cone(fp_f0.to(dev), R_tr, a_tr, b_tr, u_tr)
        zk_p = nz(kv_full.kv.project_kv(c_f0) - kv_full.kv.mu_kv, dim=-1)
        zk_g = nz(kv_full.kv.project_kv(F_all[f0_idx].to(dev)) - kv_full.kv.mu_kv, dim=-1)
        m_kv327 = intra10_and_swap_from_sim(_cos_blk(zk_p, zk_g, len(res["f0_all10"])))
        m_h327 = intra10_and_swap_from_sim(_cos_blk(hp_f0.to(dev), H14_all[f0_idx].to(dev), len(res["f0_all10"])))

        # Build ips for sel50 [500, 16, 2048]
        c_50 = c_f0[loc50].to(torch.float16)
        ips_50 = proj(adapter(c_50))
        if cond_layer is not None:
            ips_50 = cond_layer(h_f0[loc50].to(dev), ips_50)
        ips_50 = ips_50.cpu()

        # Layer 2: Frozen UNet Denoising-Error 2x2 Swap on first 25 concepts across active t in {749, 499, 249}
        pipe.scheduler.set_timesteps(4, device=dev)
        n_c2 = 25
        lat_sub = lat50[:n_c2 * 10].to(dev, torch.float32)
        ips_sub = ips_50[:n_c2 * 10].to(dev, torch.float16)
        l2_t = {}
        err_sum = np.zeros((n_c2, 10, 10), dtype=np.float64)
        for t_int in [749, 499, 249]:
            ab = float(pipe.scheduler.alphas_cumprod[t_int])
            sq_ab, sq_1mab = math.sqrt(ab), math.sqrt(1.0 - ab)
            gen = torch.Generator(device=dev).manual_seed(20261122 + t_int)
            eps = torch.randn(lat_sub.shape, generator=gen, device=dev, dtype=torch.float32)
            zt_all = (sq_ab * lat_sub + sq_1mab * eps).to(torch.float16)
            t_ten = torch.tensor([t_int], device=dev, dtype=torch.long)
            D_mat = np.zeros((n_c2, 10, 10), dtype=np.float64)
            for ci in range(n_c2):
                sl = slice(ci * 10, (ci + 1) * 10)
                zt_p = zt_all[sl].repeat_interleave(10, dim=0)
                ep_p = eps[sl].repeat_interleave(10, dim=0)
                ip_p = ips_sub[sl].repeat(10, 1, 1)
                errs = []
                for b0 in range(0, 100, 50):
                    b1 = min(100, b0 + 50)
                    B = b1 - b0
                    pr = pipe.unet(
                        zt_p[b0:b1], t_ten.expand(B),
                        encoder_hidden_states=pe.expand(B, -1, -1),
                        added_cond_kwargs={"text_embeds": ppe.expand(B, -1), "time_ids": add_time_ids.expand(B, -1)},
                        cross_attention_kwargs={"ip_hidden_states": ip_p[b0:b1]},
                    ).sample.float()
                    errs.append((pr - ep_p[b0:b1]).pow(2).mean(dim=(1, 2, 3)).cpu().numpy())
                D_mat[ci] = np.concatenate(errs, 0).reshape(10, 10)
            l2_t[f"t{t_int}"] = intra10_and_swap_from_sim(-D_mat)
            err_sum += (D_mat - D_mat.mean(axis=(1, 2), keepdims=True)) / max(float(D_mat.std()), 1e-8)
        l2_t["avg_3t"] = intra10_and_swap_from_sim(-err_sum)

        # Layer 3: Generate 500 images for FixedZ0_Swap & FreeGen_O14 (4-step SDXL-Turbo, nsel=1, fixed concept seed)
        def _gen_score(z0_all):
            N = 500
            steps = 4
            s_rel, s_unrel, smap_c = 0.5, 1.0, 0.625
            hold_val = steps - steps * (s_unrel - (s_unrel - s_rel) * smap_c)
            low_keep = (8, 0.22, 0.04)
            init = torch.cat([
                (vae32.decode(z0_all[j:j + 20].to(dev, torch.float32) / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
                for j in range(0, N, 20)
            ])
            rec = torch.empty(N, 3, 512, 512, dtype=torch.float32)
            for s0 in range(0, N, 25):
                q = list(range(s0, min(N, s0 + 25)))
                Q = len(q)
                ip_b = ips_50[q].to(dev, torch.float16)
                gens = [torch.Generator(dev).manual_seed(1_000_003 * (i // 10) + 12345) for i in q]
                eps_q = torch.stack([
                    torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * (i // 10) + 99))
                    for i in q
                ]).to(dev)
                z0_q = z0_all[q].to(dev, torch.float32)
                i0_q = init[q].to(dev, torch.float16)

                def cb(pp, i_step, t_cur, ck):
                    lt = ck["latents"].float()
                    si = int(pp.scheduler.step_index)
                    sig = pp.scheduler.sigmas[si].to(dev)
                    a = max(0.0, min(1.0, hold_val - i_step))
                    if a > 0:
                        lt = a * (z0_q + sig * eps_q) + (1.0 - a) * lt
                    if i_step in (steps - 2, steps - 1):
                        f_lp, w2, w3 = low_keep
                        g_t = w2 if i_step == steps - 2 else w3
                        if g_t > 0:
                            lt = lt + g_t * _lp_op((z0_q + sig * eps_q) - lt, f_lp)
                    ck["latents"] = lt.to(torch.float16)
                    return ck

                lat_out = p2(
                    image=i0_q,
                    prompt_embeds=pe.expand(Q, -1, -1),
                    pooled_prompt_embeds=ppe.expand(Q, -1),
                    num_inference_steps=steps,
                    strength=s_unrel,
                    guidance_scale=0.0,
                    output_type="latent",
                    generator=gens,
                    cross_attention_kwargs={"ip_hidden_states": ip_b},
                    callback_on_step_end=cb,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images
                for j0 in range(0, Q, 10):
                    img_b = (vae32.decode(lat_out[j0:j0 + 10].float() / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
                    rec[s0 + j0:s0 + j0 + len(img_b)] = img_b

            out = {}
            for name, (mod, pre, layer) in nets.items():
                if name == "effnet":
                    continue
                fl = []
                for i in range(0, N, 50):
                    y = mod(pre(rec[i:i + 50].to(dev)))
                    fl.append((y if layer is None else y[layer]).float().flatten(1).cpu())
                F_rec = torch.cat(fl, 0)
                F_gt = gt_feats[name]
                Fr = nz(F_rec - F_rec.mean(0, keepdim=True), dim=1).view(50, 10, -1)
                Fg = nz(F_gt - F_gt.mean(0, keepdim=True), dim=1).view(50, 10, -1)
                m = intra10_and_swap_from_sim(torch.bmm(Fr, Fg.transpose(1, 2)).numpy())
                if name == "clip":
                    Fr_g = nz(F_rec - F_rec.mean(0, keepdim=True), dim=1)
                    Fg_g = nz(F_gt - F_gt.mean(0, keepdim=True), dim=1)
                    S_all = (Fr_g @ Fg_g.T).numpy()
                    m["global500_top1"] = float(np.mean(S_all.argmax(1) == np.arange(N)))
                    iu, ju = np.triu_indices(10, k=1)
                    ss = torch.bmm(Fr, Fr.transpose(1, 2)).numpy()
                    m["swap_clip_disp"] = float(np.mean(1.0 - ss[:, iu, ju]))
                out[name] = m

            r128 = F.interpolate(rec, size=128, mode="bilinear", align_corners=False).view(50, 10, -1)
            g128 = F.interpolate(gt_rgb50, size=128, mode="bilinear", align_corners=False).view(50, 10, -1)
            pix_sim = torch.bmm(nz(r128 - r128.mean(-1, keepdim=True), dim=-1), nz(g128 - g128.mean(-1, keepdim=True), dim=-1).transpose(1, 2)).numpy()
            out["pixcorr"] = intra10_and_swap_from_sim(pix_sim)
            out["pixcorr"]["diag_mean"] = float(np.mean(pix_sim[:, np.arange(10), np.arange(10)]))
            iu, ju = np.triu_indices(10, k=1)
            out["swap_pixel_l1"] = float((r128[:, iu] - r128[:, ju]).abs().mean())

            r256 = F.interpolate(rec, size=256, mode="bilinear", align_corners=False).permute(0, 2, 3, 1).numpy()
            g256 = F.interpolate(gt_rgb50, size=256, mode="bilinear", align_corners=False).permute(0, 2, 3, 1).numpy()
            rg, gg = rgb2gray(r256), rgb2gray(g256)
            out["ssim_diag"] = float(np.mean([
                structural_similarity(rg[i], gg[i], gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
                for i in range(N)
            ]))
            return out

        z0_e = z0_by_sub[sub]
        z0_cmean = z0_e.view(50, 10, 4, 64, 64).mean(dim=1, keepdim=True).expand(-1, 10, -1, -1, -1).reshape(500, 4, 64, 64)
        l3_fix = _gen_score(z0_cmean)
        l3_free = _gen_score(z0_e)

        per_sub[f"sub-{sub:02d}"] = {
            "L1_327": {"F": m_f327, "KV": m_kv327, "H14": m_h327},
            "L2_unet": l2_t,
            "L3_fixedZ0": l3_fix,
            "L3_freeGen": l3_free,
        }
        print(
            f"[EVAL {arm_name:22s} sub-{sub:02d}] "
            f"L1(F/KV)={m_f327['top1']*100:.2f}/{m_kv327['top1']*100:.2f}% | "
            f"L2(3t sw/dc10)={l2_t['avg_3t']['swap2x2']*100:.2f}/{l2_t['avg_3t']['top1_dc']*100:.2f}% | "
            f"FixZ0(CLIP t10/sw, A5 sw, Px sw)={l3_fix['clip']['top1']*100:.1f}/{l3_fix['clip']['swap2x2']*100:.1f}%, {l3_fix['alex5']['swap2x2']*100:.1f}%, {l3_fix['pixcorr']['swap2x2']*100:.1f}% | "
            f"Free(CLIP t10/sw/g500, A5 t10/sw, Px sw, SSIM)={l3_free['clip']['top1']*100:.1f}/{l3_free['clip']['swap2x2']*100:.1f}/{l3_free['clip']['global500_top1']*100:.1f}%, "
            f"{l3_free['alex5']['top1']*100:.1f}/{l3_free['alex5']['swap2x2']*100:.1f}%, {l3_free['pixcorr']['swap2x2']*100:.1f}%, {l3_free['ssim_diag']:.3f}",
            flush=True,
        )

    # 4-subject summary
    def _m(path):
        vals = []
        for sub in DEV_SUBS:
            d = per_sub[f"sub-{sub:02d}"]
            for k in path:
                d = d[k]
            vals.append(float(d))
        return float(np.mean(vals))

    mean4 = {
        "L1_F_top1": _m(["L1_327", "F", "top1"]),
        "L1_KV_top1": _m(["L1_327", "KV", "top1"]),
        "L2_unet_sw": _m(["L2_unet", "avg_3t", "swap2x2"]),
        "L2_unet_dc10": _m(["L2_unet", "avg_3t", "top1_dc"]),
        "L2_t749_sw": _m(["L2_unet", "t749", "swap2x2"]),
        "L2_t499_sw": _m(["L2_unet", "t499", "swap2x2"]),
        "L2_t249_sw": _m(["L2_unet", "t249", "swap2x2"]),
        "FixZ0_clip_t10": _m(["L3_fixedZ0", "clip", "top1"]),
        "FixZ0_clip_sw": _m(["L3_fixedZ0", "clip", "swap2x2"]),
        "FixZ0_incep_sw": _m(["L3_fixedZ0", "inception", "swap2x2"]),
        "FixZ0_alex5_t10": _m(["L3_fixedZ0", "alex5", "top1"]),
        "FixZ0_alex5_sw": _m(["L3_fixedZ0", "alex5", "swap2x2"]),
        "FixZ0_alex2_sw": _m(["L3_fixedZ0", "alex2", "swap2x2"]),
        "FixZ0_pix_sw": _m(["L3_fixedZ0", "pixcorr", "swap2x2"]),
        "FixZ0_pxL1": _m(["L3_fixedZ0", "swap_pixel_l1"]),
        "Free_clip_t10": _m(["L3_freeGen", "clip", "top1"]),
        "Free_clip_dc10": _m(["L3_freeGen", "clip", "top1_dc"]),
        "Free_clip_sw": _m(["L3_freeGen", "clip", "swap2x2"]),
        "Free_clip_g500": _m(["L3_freeGen", "clip", "global500_top1"]),
        "Free_incep_t10": _m(["L3_freeGen", "inception", "top1"]),
        "Free_incep_sw": _m(["L3_freeGen", "inception", "swap2x2"]),
        "Free_alex5_t10": _m(["L3_freeGen", "alex5", "top1"]),
        "Free_alex5_sw": _m(["L3_freeGen", "alex5", "swap2x2"]),
        "Free_alex2_sw": _m(["L3_freeGen", "alex2", "swap2x2"]),
        "Free_pix_sw": _m(["L3_freeGen", "pixcorr", "swap2x2"]),
        "Free_pix_diag": _m(["L3_freeGen", "pixcorr", "diag_mean"]),
        "Free_ssim": _m(["L3_freeGen", "ssim_diag"]),
    }
    print(
        f">>> MEAN4 [{arm_name:22s}] "
        f"L1(F/KV)={mean4['L1_F_top1']*100:.2f}/{mean4['L1_KV_top1']*100:.2f}% | "
        f"L2(sw/dc10)={mean4['L2_unet_sw']*100:.2f}/{mean4['L2_unet_dc10']*100:.2f}% "
        f"(t749={mean4['L2_t749_sw']*100:.1f} t499={mean4['L2_t499_sw']*100:.1f} t249={mean4['L2_t249_sw']*100:.1f}) | "
        f"FixZ0: CLIP={mean4['FixZ0_clip_t10']*100:.2f}%({mean4['FixZ0_clip_sw']*100:.2f}%) "
        f"A5={mean4['FixZ0_alex5_t10']*100:.2f}%({mean4['FixZ0_alex5_sw']*100:.2f}%) Px={mean4['FixZ0_pix_sw']*100:.2f}% | "
        f"Free: CLIP={mean4['Free_clip_t10']*100:.2f}%({mean4['Free_clip_sw']*100:.2f}%) g500={mean4['Free_clip_g500']*100:.2f}% "
        f"Inc={mean4['Free_incep_sw']*100:.2f}% A5={mean4['Free_alex5_t10']*100:.2f}%({mean4['Free_alex5_sw']*100:.2f}%) "
        f"A2={mean4['Free_alex2_sw']*100:.2f}% Px={mean4['Free_pix_sw']*100:.2f}% SSIM={mean4['Free_ssim']:.4f}",
        flush=True,
    )
    del model, cond_layer
    torch.cuda.empty_cache()
    return {"per_sub": per_sub, "mean4": mean4}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_arm2b", action="store_true")
    ap.add_argument("--only_arm2b", action="store_true")
    args = ap.parse_args()

    dev = "cuda:0"
    t_all = time.time()
    arm0_ckpt = OUT_DIR / "arm0_tr14_xbrain_s2025.pt"
    assert arm0_ckpt.exists(), f"Missing {arm0_ckpt}; run exp0 first."

    res = build_tr14_resources(dev=dev)
    z0_by_sub = {sub: fit_tr14_ridge_z0(sub, res["tr_idx_np"], res["sel50_idx"], dev=dev) for sub in DEV_SUBS}

    # 1. Train Arm1_RepIntra (cached)
    arm1_ckpt = train_arm1_rep_intra(arm0_ckpt, res, steps=800, seed=2025, dev=dev)
    arm2a_ckpt = OUT_DIR / "arm2a_swapdiff_cone_st800_s2025.pt"
    arm2b_ckpt = OUT_DIR / "arm2b_swapdiff_condlyr_st800_s2025.pt"

    need_train_2a = not arm2a_ckpt.exists()
    need_train_2b = (not args.skip_arm2b) and (not arm2b_ckpt.exists())

    from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
    from metrics import _nets

    if need_train_2a or need_train_2b:
        pipe_tr = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=torch.bfloat16, variant="fp16")
        IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(pipe_tr.unet)
        pipe_tr.to(dev, torch.bfloat16)
        with torch.no_grad():
            pe_bf, _, ppe_bf, _ = pipe_tr.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
            atid_bf = pipe_tr._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=torch.bfloat16, text_encoder_projection_dim=pipe_tr.text_encoder_2.config.projection_dim).to(dev)
            pe_bf, ppe_bf, atid_bf = pe_bf.detach().clone(), ppe_bf.detach().clone(), atid_bf.detach().clone()
        del pipe_tr.text_encoder, pipe_tr.text_encoder_2, pipe_tr.vae
        pipe_tr.text_encoder = None
        pipe_tr.text_encoder_2 = None
        pipe_tr.vae = None
        alphas_cp = pipe_tr.scheduler.alphas_cumprod.clone()
        adapter_bf = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, torch.bfloat16).eval().requires_grad_(False)
        proj_bf = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, torch.bfloat16).eval().requires_grad_(False)
        torch.cuda.empty_cache()

        if need_train_2a:
            arm2a_ckpt = train_arm2_swapdiff(
                arm0_ckpt, res, pipe_tr.unet, alphas_cp, adapter_bf, proj_bf, pe_bf, ppe_bf, atid_bf,
                use_cond_layer=False, steps=800, seed=2025, dev=dev,
            )
        if need_train_2b:
            arm2b_ckpt = train_arm2_swapdiff(
                arm0_ckpt, res, pipe_tr.unet, alphas_cp, adapter_bf, proj_bf, pe_bf, ppe_bf, atid_bf,
                use_cond_layer=True, steps=800, seed=2025, dev=dev,
            )
        del pipe_tr, adapter_bf, proj_bf, pe_bf, ppe_bf, atid_bf
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Full Controlled Evaluation in identical fp16 OfficialTurboRunner pipeline
    dt = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=dt, variant="fp16")
    IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(pipe.unet)
    pipe.to(dev, dt)
    pipe.unet.eval().requires_grad_(False)
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        pe, _, ppe, _ = pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
        atid = pipe._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=dt, text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).to(dev)
    del pipe.text_encoder, pipe.text_encoder_2
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    p2 = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    p2.set_progress_bar_config(disable=True)
    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, dt).eval().requires_grad_(False)
    proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, dt).eval().requires_grad_(False)
    vae32 = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval().requires_grad_(False)
    vae32.enable_slicing()

    lat50 = torch.from_numpy(np.load(PREP / "lat512_train.npy")[res["sel50_idx"]]).float()
    with torch.inference_mode():
        gt_rgb50 = torch.cat([
            (vae32.decode(lat50[j:j + 20].to(dev, torch.float32) / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
            for j in range(0, 500, 20)
        ])
        nets = _nets()
        gt_feats = {}
        for name, (mod, pre, layer) in nets.items():
            if name == "effnet":
                continue
            fl = []
            for i in range(0, 500, 50):
                y = mod(pre(gt_rgb50[i:i + 50].to(dev)))
                fl.append((y if layer is None else y[layer]).float().flatten(1).cpu())
            gt_feats[name] = torch.cat(fl, 0)

    res_p = OUT_DIR / "exp1_4arm_results.json"
    results = json.loads(res_p.read_text()) if res_p.exists() else {"arms": {}}
    eval_list = [
        ("Arm0_Base", arm0_ckpt),
        ("Arm1_RepIntra", arm1_ckpt),
        ("Arm2a_SwapDiff_Cone", arm2a_ckpt),
    ]
    if not args.skip_arm2b:
        eval_list.append(("Arm2b_SwapDiff_CondLyr", arm2b_ckpt))
    if args.only_arm2b:
        eval_list = [("Arm2b_SwapDiff_CondLyr", arm2b_ckpt)]

    for arm_name, ckpt in eval_list:
        if arm_name in results["arms"]:
            print(f"[CACHED EVAL] {arm_name}", flush=True)
            continue
        results["arms"][arm_name] = evaluate_arm_complete(
            arm_name, ckpt, res, pipe, p2, vae32, adapter, proj, pe, ppe, atid,
            lat50, gt_rgb50, gt_feats, z0_by_sub, dev=dev,
        )
        results["wall_s"] = round(time.time() - t_all, 1)
        write_json(res_p, results)

    print(f"=== PHASE 1 EVAL DONE in {round(time.time()-t_all, 1)}s -> {res_p} ===", flush=True)


if __name__ == "__main__":
    main()
