"""Phase 0 Go/No-Go Gate: Within-Concept (Intra10) Representation vs UNet Denoising vs 4-Step Generation Swap Sensitivity.

Strictly concept-disjoint (zero leakage):
  - Training: tr14 = folds 1..4 (1,323 concepts x 10 photos = 13,230 imgs, 132,300 subject-image pairs).
  - Sealed evaluation: fold 0 (327 valid concepts x 10 photos = 3,270 unseen photos, R=4 real EEG trials).
  - Generation & UNet swap benchmark: D_intra50 = 50 deterministically sealed fold-0 concepts x 10 photos = 500 photos
    (2,250 within-concept pairs (A, B) per subject).

Outputs exact bottom-up diagnostic metrics across 3 layers:
  Layer 1: Raw R=4 EEG 2v2 Intra10, Arm0 (O13/O14 shared-trunk tr14 baseline) F / F_kv / H14 / B32 / z0_ridge Intra10 & 2x2 Swap.
  Layer 2: Frozen SDXL-Turbo UNet denoising-error 2x2 Swap Win-Rate & Intra10 across t in {999, 749, 499, 249}
           for BOTH Oracle F_gt and Baseline Arm0 c_eeg(R=4).
  Layer 3: 4-step SDXL-Turbo generated image Intra10, 2x2 Swap Win-Rate, and Swap Displacement under:
           (a) Oracle F_gt + FixedZ0 (smap=0.625) and Oracle F_gt + PureNoise (smap=0.0)
           (b) EEG_FixedZ0_smap625 (fixed z0_bar & seed, swap c(E) only)
           (c) EEG_FixedSeed_PureNoise (smap=0.0, fixed seed, swap c(E) only)
           (d) EEG_FreeGen_O14 (full free-EEG z0(E) + c(E), smap=0.625, low_keep=(8,0.22,0.04), 4 NFE, nsel=1)
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
CODE_DIRS = [
    ROOT / "code/repo_code",
    ROOT / "code",
    Path(__file__).resolve().parent,
]
for d in CODE_DIRS:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

from common import EMB_KEYS, HIVE, PREP, PRIOR, RUNS, TURBO, concept_folds, write_json
from deep_enc import TargetSNRGate, sample_sub_repeats
from o10_factorized import _cone_params, _perp_dir
from o11_tri_stream_bigg import OnConeKVProjector, _hybrid_infonce, _lp_op, _simple_infonce, _to_cone
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder, _xbrain_cone_nce, _xbrain_nce
from train_flex import shift_batch

sys.path.insert(0, str(HIVE))
from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel

nz = F.normalize
OUT_DIR = RUNS / "swap_diff"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALL_SUBS = tuple(range(1, 11))
DEV_SUBS = (2, 5, 8, 9)


def get_concept_splits(full_training=False):
    concept = np.load(PREP / "train_concept.npy")
    ok = np.load(PREP / "lat512_train_ok.npy").astype(bool)
    c_fold = concept_folds(1654, 5, seed=20261112)
    f0_all10 = np.array([cid for cid in np.where(c_fold == 0)[0] if ok[concept == cid].all()], dtype=np.int64)
    tr14_all10 = np.array([cid for cid in np.where(c_fold != 0)[0] if ok[concept == cid].all()], dtype=np.int64)
    rng = np.random.default_rng(20261122)
    sel50_cids = np.sort(rng.choice(f0_all10, size=50, replace=False))
    tr_idx = np.flatnonzero(ok) if full_training else np.flatnonzero((c_fold[concept] != 0) & ok)
    f0_idx = np.concatenate([np.where(concept == cid)[0] for cid in f0_all10])
    sel50_idx = np.concatenate([np.where(concept == cid)[0] for cid in sel50_cids])
    return concept, ok, tr_idx, f0_all10, f0_idx, sel50_cids, sel50_idx


def train_or_load_arm0_tr14(dev: str = "cuda:0", seed: int = 2025, epochs: int = 150, subjects=ALL_SUBS,
                           *, full_training=False, crossbrain_weight=0.25) -> Path:
    """Train the shared encoder under the held-concept or official training scope."""
    if crossbrain_weight not in (0.0, 0.25):
        raise ValueError("Use the original cross-person objective or the matched zero-weight control")
    scope = "full" if full_training else "tr14"
    ckpt_p = OUT_DIR / f"arm0_{scope}_xbrain_s{seed}.pt"
    if ckpt_p.exists():
        saved = torch.load(ckpt_p, map_location="meta", weights_only=True)
        if saved.get("crossbrain_weight", 0.25) != crossbrain_weight:
            raise ValueError(f"Cross-person objective differs from the checkpoint: {ckpt_p}")
        print(f"[CACHED] {ckpt_p}", flush=True)
        return ckpt_p

    t0 = time.time()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    concept_np, ok_np, tr_idx_np, _, _, _, _ = get_concept_splits(full_training)
    tr_idx = torch.from_numpy(tr_idx_np).long()
    n_tr = len(tr_idx)

    concept = torch.from_numpy(concept_np[tr_idx_np]).long().to(dev)
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")[tr_idx_np]).float().to(dev)
    F_sem_all = torch.load(PREP / "F_sem_train.pt", map_location="cpu", weights_only=True)["F_sem"].float()
    F_sem = F_sem_all[tr_idx].to(dev)
    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    H14_tr = emb_tr[EMB_KEYS[0]][tr_idx].float().to(dev)
    B32_tr = emb_tr[EMB_KEYS[1]][tr_idx].float().to(dev)
    VAE_tr = emb_tr[EMB_KEYS[2]][tr_idx].float().to(dev)
    h14_norm = float(H14_tr.norm(dim=1).mean())
    b32_norm = float(B32_tr.norm(dim=1).mean())

    x_tr_all = {}
    for s in subjects:
        arr = np.load(PREP / f"sub-{s:02d}_train.npy")[tr_idx_np]
        x_tr_all[s - 1] = torch.from_numpy(arr).half()

    kv_full = OnConeKVProjector(str(PREP / "unet_kv_metric.pt"), device=dev)
    kv_full.calibrate_fold(F_tr)
    kv_sem = OnConeKVProjector(str(PREP / "unet_kv_metric.pt"), device=dev)
    kv_sem.calibrate_fold(F_sem)
    fe_vis = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.to(dev).float().eval().requires_grad_(False)

    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    for sid in x_tr_all:
        model.ebs[sid].fit_from_fold(x_tr_all[sid].float().to(dev))

    log_sf = nn.Parameter(torch.tensor(2.6592, device=dev))
    log_sh = nn.Parameter(torch.tensor(2.6592, device=dev))
    log_sb = nn.Parameter(torch.tensor(2.6592, device=dev))
    log_ss = nn.Parameter(torch.tensor(2.6592, device=dev))
    log_sx = nn.Parameter(torch.tensor(2.6592, device=dev))
    extra = [("ls_f", log_sf), ("ls_h", log_sh), ("ls_b", log_sb), ("ls_s", log_ss), ("ls_x", log_sx)]
    dec, ndec = [], []
    for n, p_ in list(model.named_parameters()) + extra:
        (ndec if (n.endswith("bias") or "norm" in n or n.startswith("ls_") or "emb" in n or "logit" in n) else dec).append(p_)
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.05}, {"params": ndec, "weight_decay": 0.0}], lr=5e-4, betas=(0.9, 0.995))
    bs = 1024
    spr = math.ceil(n_tr / bs)
    total = spr * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 0.5 * (1.0 + math.cos(math.pi * s / total))))
    g = torch.Generator(device=dev).manual_seed(seed)
    rng_np = np.random.default_rng(seed)
    lam_x = crossbrain_weight

    print(f"[TRAIN Arm0 {scope}] n_tr={n_tr}, epochs={epochs}, bs={bs}, seed={seed} ...", flush=True)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=dev, generator=g)
        sub_order = rng_np.permutation(list(x_tr_all))
        for b in range(spr):
            idx = perm[b * bs:(b + 1) * bs]
            if len(idx) < 32:
                continue
            idx_c = idx.cpu()
            sid_a = int(sub_order[(2 * b) % len(sub_order)])
            sid_b = int(sub_order[(2 * b + 1) % len(sub_order)])
            raw_a = x_tr_all[sid_a][idx_c].float().to(dev)
            raw_b = x_tr_all[sid_b][idx_c].float().to(dev)
            xa, ka = sample_sub_repeats(raw_a, g)
            xb, kb = sample_sub_repeats(raw_b, g)
            xa, xb = shift_batch(xa, 2, g), shift_batch(xb, 2, g)
            c_b = concept[idx]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                sc_f = log_sf.float().exp().clamp(max=100.0)
                sc_h = log_sh.float().exp().clamp(max=100.0)
                sc_b = log_sb.float().exp().clamp(max=100.0)
                sc_s = log_ss.float().exp().clamp(max=100.0)
                sc_x = log_sx.float().exp().clamp(max=100.0)

                with torch.no_grad():
                    ff = F_tr[idx].float()
                    uf = nz(ff, dim=-1)
                    df_t = nz(uf - (uf @ kv_full.u_tr.T) * kv_full.u_tr, dim=-1)
                    ykv_f = nz(kv_full.kv.project_kv(ff) - kv_full.kv.mu_kv, dim=-1)

                    fs = F_sem[idx].float()
                    us = nz(fs, dim=-1)
                    ds_t = nz(us - (us @ kv_sem.u_tr.T) * kv_sem.u_tr, dim=-1)
                    ykv_s = nz(kv_sem.kv.project_kv(fs) - kv_sem.kv.mu_kv, dim=-1)

                    same_c = c_b[:, None] == c_b[None, :]
                    eye = torch.eye(len(idx), dtype=torch.bool, device=dev)
                    fn_mask = same_c & (~eye)
                    diff_c = ~same_c
                    off_diag = ~eye

                    ht = nz(H14_tr[idx], dim=-1)
                    bt = nz(B32_tr[idx], dim=-1)
                    sh, sb = ht @ ht.T, bt @ bt.T
                    sf_sim = df_t @ df_t.T
                    sh_bar, sb_bar, sf_bar = sh[diff_c].mean(), sb[diff_c].mean(), sf_sim[diff_c].mean()
                    mh = 0.15 * sc_h.detach() * F.relu(sh - sh_bar) * diff_c.float()
                    mb = 0.15 * sc_b.detach() * F.relu(sb - sb_bar) * diff_c.float()
                    si = 0.5 * (ds_t @ ds_t.T) + 0.5 * (ykv_s @ ykv_s.T)
                    ms = 0.15 * sc_s.detach() * F.relu(si - si[diff_c].mean()) * diff_c.float()
                    vae_r = VAE_tr[torch.randint(0, n_tr, (len(idx),), device=dev, generator=g)]

                def _one_sub_loss(fp, hp, bp):
                    dfp, zk_f = kv_full.cone_and_kv(fp)
                    lf = _hybrid_infonce(dfp, zk_f, df_t, ykv_f, sc_f)
                    hu, bu = nz(hp.float(), dim=-1), nz(bp.float(), dim=-1)
                    lh = _simple_infonce(hu, ht, sc_h, fn_mask=fn_mask, margin=mh)
                    lb = _simple_infonce(bu, bt, sc_b, fn_mask=fn_mask, margin=mb)
                    with torch.amp.autocast("cuda", enabled=False):
                        fsyn = fe_vis(**{EMB_KEYS[0]: hu * h14_norm, EMB_KEYS[1]: bu * b32_norm, EMB_KEYS[2]: vae_r.float()}).float()
                    dfs, zks = kv_sem.cone_and_kv(fsyn)
                    lkvs = _hybrid_infonce(dfs, zks, ds_t, ykv_s, sc_s, fn_mask=fn_mask, margin=ms)
                    return 0.35 * lf + 0.30 * lh + 0.15 * lb + 0.20 * lkvs

                def _br_loss(brs):
                    tot = 0.0
                    for fp, hp, bp in brs:
                        dfp, zk_f = kv_full.cone_and_kv(fp)
                        lf = _hybrid_infonce(dfp, zk_f, df_t, ykv_f, sc_f)
                        lh = _simple_infonce(nz(hp.float(), dim=-1), ht, sc_h, fn_mask=fn_mask, margin=mh)
                        lb = _simple_infonce(nz(bp.float(), dim=-1), bt, sc_b, fn_mask=fn_mask, margin=mb)
                        tot = tot + (0.45 * lf + 0.35 * lh + 0.20 * lb)
                    return tot / float(len(brs))

                (fa, ha, ba), bra = model.forward_sub(xa, sid_a, r_eff=ka, return_branches=True)
                (fb_, hb, bb), brb = model.forward_sub(xb, sid_b, r_eff=kb, return_branches=True)
                l_vis = 0.5 * (0.65 * _one_sub_loss(fa, ha, ba) + 0.35 * _br_loss(bra) + 0.65 * _one_sub_loss(fb_, hb, bb) + 0.35 * _br_loss(brb))
                mf_all = 0.15 * sc_x.detach() * F.relu(sf_sim - sf_bar) * off_diag.float()
                mh_all = 0.15 * sc_x.detach() * F.relu(sh - sh_bar) * off_diag.float()
                mb_all = 0.15 * sc_x.detach() * F.relu(sb - sb_bar) * off_diag.float()
                l_xb = (
                    0.35 * _xbrain_cone_nce(fa, fb_, kv_full.u_tr, sc_x, mf_all)
                    + 0.40 * _xbrain_nce(ha, hb, sc_x, mh_all)
                    + 0.25 * _xbrain_nce(ba, bb, sc_x, mb_all)
                )
                loss = l_vis + lam_x * l_xb

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
        if (ep + 1) in (50, 100, 150):
            print(f"  ep {ep+1:3d}/{epochs} loss={float(loss):.4f} ({time.time()-t0:.1f}s)", flush=True)

    model.eval()
    torch.save(
        {
            "state_dict": {k: v.half().cpu() for k, v in model.state_dict().items()},
            "cone": {"R": float(kv_full.kv.nu_bar), "alpha": float(kv_full.alpha_tr), "beta": float(kv_full.beta_tr), "u_tr": kv_full.u_tr.cpu()},
            "h14_norm": h14_norm,
            "b32_norm": b32_norm,
            "seed": seed,
            "epochs": epochs,
            "crossbrain_weight": crossbrain_weight,
        },
        ckpt_p,
    )
    print(f"[SAVED] {ckpt_p} ({time.time()-t0:.1f}s)", flush=True)
    del x_tr_all, model, fe_vis, kv_full, kv_sem
    gc.collect()
    torch.cuda.empty_cache()
    return ckpt_p


def fit_tr14_ridge_z0(sub: int, tr_idx: np.ndarray, eval_idx: np.ndarray, dev: str = "cuda:0") -> torch.Tensor:
    """Strictly tr14-only multi-window QR-PCA ridge predictor for 512px start latents on sealed fold-0 R=4 EEG."""
    cache_p = OUT_DIR / f"z0_tr14_ridge_sub{sub:02d}_sel50.pt"
    if cache_p.exists():
        return torch.load(cache_p, map_location="cpu", weights_only=True)

    lat_all = np.load(PREP / "lat512_train.npy")
    Y_tr = torch.from_numpy(lat_all[tr_idx]).float().to(dev).flatten(1)
    my = Y_tr.mean(0)
    Yc = Y_tr - my
    del lat_all, Y_tr

    def lp_flat(y_flat, f):
        v = y_flat.view(-1, 4, 64, 64)
        return F.interpolate(F.avg_pool2d(v, f), size=64, mode="bilinear", align_corners=False).flatten(1)

    torch.manual_seed(20261115)
    n_pca = 768
    Yc_w = Yc + 1.5 * lp_flat(Yc, 4) + 0.75 * lp_flat(Yc, 2)
    _, _, V_p = torch.pca_lowrank(Yc_w, q=n_pca + 32, center=False, niter=4)
    del Yc_w
    V_pca, _ = torch.linalg.qr(V_p[:, :n_pca].double(), mode="reduced")
    Z_tr = Yc.double() @ V_pca  # [N_tr, 768]
    rms_real_lp4 = float(F.avg_pool2d(Yc.view(-1, 4, 64, 64), 4).flatten(1).pow(2).sum(1).mean().sqrt())
    del Yc

    X_raw = np.load(PREP / f"sub-{sub:02d}_train.npy")  # [16540, 4, 17, 250]
    X_m4 = torch.from_numpy(X_raw.mean(1)).float()
    del X_raw
    X_tr = X_m4[tr_idx]
    X_ev = X_m4[eval_idx]

    WINDOWS = [(12, 62), (10, 75), (15, 90), (10, 125), (0, 250)]
    preds = []
    for t0, t1 in WINDOWS:
        A_tr = X_tr[:, :, t0:t1].flatten(1).to(dev)
        A_ev = X_ev[:, :, t0:t1].flatten(1).to(dev)
        mu = A_tr.mean(0, keepdim=True)
        sd = A_tr.std(0, keepdim=True).clamp_min(1e-6)
        A_tr = ((A_tr - mu) / sd).double()
        A_ev = ((A_ev - mu) / sd).double()
        Cxx = (A_tr.T @ A_tr) / float(len(A_tr))
        Cxy = (A_tr.T @ Z_tr) / float(len(A_tr))
        lam = 3.0 * float(torch.trace(Cxx) / Cxx.shape[0])
        W = torch.linalg.solve(Cxx + lam * torch.eye(Cxx.shape[0], dtype=torch.float64, device=dev), Cxy)
        preds.append(A_ev @ W)
    Z_pred = torch.stack(preds, dim=0).mean(0)  # [N_ev, 768]
    Y_pred = (Z_pred @ V_pca.T).float()
    # Match coarse 16x16 RMS scale to 0.80 * real (same spread=0.80 as O7..O14 v2)
    cur_rms = float(F.avg_pool2d(Y_pred.view(-1, 4, 64, 64), 4).flatten(1).pow(2).sum(1).mean().sqrt())
    if cur_rms > 1e-6:
        Y_pred = Y_pred * (0.80 * rms_real_lp4 / cur_rms)
    z0_ev = (Y_pred + my).view(-1, 4, 64, 64).cpu()
    torch.save(z0_ev, cache_p)
    del Z_tr, V_pca, my
    torch.cuda.empty_cache()
    return z0_ev


def intra10_and_swap_from_sim(sim_mat: np.ndarray):
    """Given [C, 10, 10] similarity matrix S[c, i, j] = sim(pred_i, gt_j), compute Intra10 Top-1 & doubly-balanced 2x2 Swap."""
    C = sim_mat.shape[0]
    eye_idx = np.arange(10)
    top1_raw = float(np.mean(sim_mat.argmax(axis=2) == eye_idx[None, :]))
    # Doubly-centered top-1 (removes image/pred bias)
    sc = sim_mat - sim_mat.mean(axis=1, keepdims=True) - sim_mat.mean(axis=2, keepdims=True)
    top1_dc = float(np.mean(sc.argmax(axis=2) == eye_idx[None, :]))
    # Doubly-balanced 2x2 swap win rate: S[i,i] + S[j,j] > S[i,j] + S[j,i] for all i < j
    iu, ju = np.triu_indices(10, k=1)
    margin = (sim_mat[:, iu, iu] + sim_mat[:, ju, ju]) - (sim_mat[:, iu, ju] + sim_mat[:, ju, iu])
    swap_win = float(np.mean(margin > 0.0))
    return {"top1": top1_raw, "top1_dc": top1_dc, "swap2x2": swap_win, "margin_mean": float(margin.mean())}


def eval_rep_layer1(ckpt_p: Path, concept_np, tr_idx, f0_all10, f0_idx, sel50_cids, sel50_idx, dev="cuda:0"):
    """Layer 1: Raw R=4 EEG 2v2 Intra10 + Arm0 Representation Intra10 & 2x2 Swap."""
    pack = torch.load(ckpt_p, map_location="cpu", weights_only=True)
    model = CrossSubjectHorizonEncoder(n_subs=10).to(dev)
    model.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
    model.eval()
    u_tr = pack["cone"]["u_tr"].to(dev).float()

    F_all = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    H14_all = emb_tr[EMB_KEYS[0]].float()
    B32_all = emb_tr[EMB_KEYS[1]].float()
    lat_all = np.load(PREP / "lat512_train.npy")
    kv_full = OnConeKVProjector(str(PREP / "unet_kv_metric.pt"), device=dev)
    kv_full.calibrate_fold(F_all[tr_idx].to(dev))

    def _cosine_blocks(P: torch.Tensor, G: torch.Tensor, n_concepts: int):
        Pn = nz(P.float(), dim=-1).view(n_concepts, 10, -1)
        Gn = nz(G.float(), dim=-1).view(n_concepts, 10, -1)
        return torch.bmm(Pn, Gn.transpose(1, 2)).cpu().numpy()

    res = {}
    c_sel50_by_sub = {}
    z0_sel50_by_sub = {}
    for sub in DEV_SUBS:
        sid = sub - 1
        X_raw = np.load(PREP / f"sub-{sub:02d}_train.npy")  # [16540, 4, 17, 250]
        # Raw R=4 EEG 2v2 split-half Intra10 on all 327 fold-0 concepts
        xf0_raw = torch.from_numpy(X_raw[f0_idx]).float()  # [3270, 4, 17, 250]
        ha = xf0_raw[:, :2].mean(1).flatten(1)
        hb = xf0_raw[:, 2:].mean(1).flatten(1)
        # Center within fold-0 to remove DC offset
        ha = ha - ha.mean(0, keepdim=True)
        hb = hb - hb.mean(0, keepdim=True)
        raw_sim = _cosine_blocks(ha, hb, len(f0_all10))
        raw_m = intra10_and_swap_from_sim(raw_sim)

        # Arm0 forward on all 3270 fold-0 images (R=4 mean)
        xm4 = xf0_raw.mean(1)
        fp_l, hp_l, bp_l = [], [], []
        with torch.no_grad():
            for i in range(0, len(xm4), 256):
                fp, hp, bp = model.forward_sub(xm4[i:i + 256].to(dev), sid, r_eff=4.0)
                fp_l.append(fp.cpu())
                hp_l.append(hp.cpu())
                bp_l.append(bp.cpu())
        fp_f0 = torch.cat(fp_l, 0)
        hp_f0 = torch.cat(hp_l, 0)
        bp_f0 = torch.cat(bp_l, 0)

        # Evaluate on all 327 fold-0 concepts AND sel50
        def _eval_subset(idx_sub, fp_s, hp_s, bp_s, n_c):
            df_p = _perp_dir(fp_s.to(dev), u_tr)
            df_g = _perp_dir(F_all[idx_sub].to(dev), u_tr)
            m_f = intra10_and_swap_from_sim(_cosine_blocks(df_p, df_g, n_c))
            # In UNet L_KV space
            _, zk_p = kv_full.cone_and_kv(fp_s.to(dev))
            zk_g = nz(kv_full.kv.project_kv(F_all[idx_sub].to(dev)) - kv_full.kv.mu_kv, dim=-1)
            m_kv = intra10_and_swap_from_sim(_cosine_blocks(zk_p, zk_g, n_c))
            m_h14 = intra10_and_swap_from_sim(_cosine_blocks(hp_s.to(dev), H14_all[idx_sub].to(dev), n_c))
            m_b32 = intra10_and_swap_from_sim(_cosine_blocks(bp_s.to(dev), B32_all[idx_sub].to(dev), n_c))
            return {"F_perp": m_f, "F_kv": m_kv, "H14": m_h14, "B32": m_b32}

        sub_327 = _eval_subset(f0_idx, fp_f0, hp_f0, bp_f0, len(f0_all10))

        # Map sel50 inside f0
        f0_pos = {int(g): i for i, g in enumerate(f0_idx)}
        loc50 = np.array([f0_pos[int(g)] for g in sel50_idx], dtype=np.int64)
        sub_50 = _eval_subset(sel50_idx, fp_f0[loc50], hp_f0[loc50], bp_f0[loc50], 50)

        # Build on-cone c_eeg for sel50 (exact O13/O14 w=1.00 cone!)
        df_50 = _perp_dir(fp_f0[loc50].to(dev), u_tr)
        c_50 = _to_cone(df_50, pack["cone"]["R"], pack["cone"]["alpha"], pack["cone"]["beta"], u_tr).cpu()
        c_sel50_by_sub[sub] = c_50

        # Also fit tr14-only ridge z0 and measure its Intra10
        z0_50 = fit_tr14_ridge_z0(sub, tr_idx, sel50_idx, dev=dev)
        z0_sel50_by_sub[sub] = z0_50
        z0_gt50 = torch.from_numpy(lat_all[sel50_idx]).float()
        z0_p_c = z0_50.flatten(1) - z0_50.flatten(1).mean(0, keepdim=True)
        z0_g_c = z0_gt50.flatten(1) - z0_gt50.flatten(1).mean(0, keepdim=True)
        sub_50["z0_ridge"] = intra10_and_swap_from_sim(_cosine_blocks(z0_p_c, z0_g_c, 50))

        res[f"sub-{sub:02d}"] = {"raw_eeg_2v2_327": raw_m, "rep_327": sub_327, "rep_50": sub_50}
        print(
            f"[Layer1 sub-{sub:02d}] RawEEG2v2={raw_m['top1']*100:.2f}% (sw={raw_m['swap2x2']*100:.2f}%) | "
            f"327-c: F={sub_327['F_perp']['top1']*100:.2f}% KV={sub_327['F_kv']['top1']*100:.2f}% "
            f"H14={sub_327['H14']['top1']*100:.2f}% B32={sub_327['B32']['top1']*100:.2f}% | "
            f"50-c: F={sub_50['F_perp']['top1']*100:.2f}% (sw={sub_50['F_perp']['swap2x2']*100:.2f}%) "
            f"z0={sub_50['z0_ridge']['top1']*100:.2f}% (sw={sub_50['z0_ridge']['swap2x2']*100:.2f}%)",
            flush=True,
        )

    del model, kv_full
    torch.cuda.empty_cache()
    return res, c_sel50_by_sub, z0_sel50_by_sub


@torch.inference_mode()
def eval_unet_denoising_layer2(pipe, adapter, proj, pe, ppe, add_time_ids, lat50: torch.Tensor, c_dict: dict, dev="cuda:0"):
    """Layer 2: Frozen SDXL-Turbo UNet denoising-error 2x2 Swap Win-Rate & Intra10 across t in {999, 749, 499, 249}."""
    pipe.scheduler.set_timesteps(4, device=dev)
    t_list = [int(t) for t in pipe.scheduler.timesteps.tolist()]  # [999, 749, 499, 249]
    # Evaluate on first 25 concepts (25 * 10 = 250 imgs, 25 * 45 = 1,125 pairs) for speed (~15s per arm)
    n_c = 25
    lat_sub = lat50[:n_c * 10].to(dev, torch.float32)  # [250, 4, 64, 64]
    out_res = {}

    for arm_name, c_all in c_dict.items():
        c_sub = c_all[:n_c * 10].to(dev, torch.float16)
        ips_all = proj(adapter(c_sub))  # [250, 16, 2048]
        per_t = {}
        err_sum_norm = np.zeros((n_c, 10, 10), dtype=np.float64)

        for t_int in t_list:
            ab = float(pipe.scheduler.alphas_cumprod[t_int])
            sq_ab, sq_1mab = math.sqrt(ab), math.sqrt(1.0 - ab)
            gen = torch.Generator(device=dev).manual_seed(20261122 + t_int)
            eps = torch.randn(lat_sub.shape, generator=gen, device=dev, dtype=torch.float32)
            zt_all = (sq_ab * lat_sub + sq_1mab * eps).to(torch.float16)  # exact scale_model_input!
            t_ten = torch.tensor([t_int], device=dev, dtype=torch.long)

            D_mat = np.zeros((n_c, 10, 10), dtype=np.float64)
            # Process 5 concepts at a time: 5 concepts * 100 pairs = 500 -> chunk by 50
            for ci in range(n_c):
                sl = slice(ci * 10, (ci + 1) * 10)
                zt_c = zt_all[sl]       # [10, 4, 64, 64]
                ep_c = eps[sl]          # [10, 4, 64, 64]
                ip_c = ips_all[sl]      # [10, 16, 2048]
                # All 10x10 pairs: image i repeated 10x, cond j tiled 10x
                zt_pair = zt_c.repeat_interleave(10, dim=0)   # [100, 4, 64, 64]
                ep_pair = ep_c.repeat_interleave(10, dim=0)
                ip_pair = ip_c.repeat(10, 1, 1)               # [100, 16, 2048]
                errs = []
                for b0 in range(0, 100, 50):
                    b1 = min(100, b0 + 50)
                    B = b1 - b0
                    pred = pipe.unet(
                        zt_pair[b0:b1],
                        t_ten.expand(B),
                        encoder_hidden_states=pe.expand(B, -1, -1),
                        added_cond_kwargs={"text_embeds": ppe.expand(B, -1), "time_ids": add_time_ids.expand(B, -1)},
                        cross_attention_kwargs={"ip_hidden_states": ip_pair[b0:b1]},
                    ).sample.float()
                    e_mse = (pred - ep_pair[b0:b1]).pow(2).mean(dim=(1, 2, 3))
                    errs.append(e_mse.cpu().numpy())
                D_mat[ci] = np.concatenate(errs, 0).reshape(10, 10)

            # Note: smaller error = higher similarity -> pass -D_mat
            m_t = intra10_and_swap_from_sim(-D_mat)
            # Also report raw scale of Delta_swap = (D_ii + D_jj) - (D_ij + D_ji)
            iu, ju = np.triu_indices(10, k=1)
            delta_sw = (D_mat[:, iu, iu] + D_mat[:, ju, ju]) - (D_mat[:, iu, ju] + D_mat[:, ju, iu])
            m_t["delta_swap_mean"] = float(delta_sw.mean())
            m_t["delta_swap_std"] = float(delta_sw.std())
            m_t["d_diag_mean"] = float(np.mean(D_mat[:, np.arange(10), np.arange(10)]))
            per_t[f"t{t_int}"] = m_t
            err_sum_norm += (D_mat - D_mat.mean(axis=(1, 2), keepdims=True)) / max(float(D_mat.std()), 1e-8)

        m_all = intra10_and_swap_from_sim(-err_sum_norm)
        per_t["all_4t_avg"] = m_all
        out_res[arm_name] = per_t
        print(
            f"[Layer2 UNet {arm_name:14s}] "
            + " | ".join(
                f"t{t}: sw={per_t[f't{t}']['swap2x2']*100:.1f}% dc10={per_t[f't{t}']['top1_dc']*100:.1f}% (d={per_t[f't{t}']['delta_swap_mean']:.2e}±{per_t[f't{t}']['delta_swap_std']:.2e})"
                for t in t_list
            )
            + f" || 4t-avg: sw={m_all['swap2x2']*100:.1f}% dc10={m_all['top1_dc']*100:.1f}%",
            flush=True,
        )
    return out_res


@torch.inference_mode()
def generate_and_score_layer3(pipe, p2, vae32, adapter, proj, pe, ppe, gt_rgb50, gt_feats, c_all, z0_all, smap_c: float, low_keep, dev="cuda:0"):
    """Generate 500 images (50 concepts x 10, 4-step SDXL-Turbo, K=1 fixed seed) & score Intra10 + 2x2 Swap + Displacement."""
    from metrics import _nets

    N = len(c_all)
    assert N == 500
    steps = 4
    s_rel, s_unrel = 0.5, 1.0
    smap_val = s_unrel - (s_unrel - s_rel) * smap_c  # smap_c=0 -> strength=1.0 (pure noise); smap_c=0.625 -> O14 exact!
    hold_val = max(0.0, steps - steps * smap_val) if smap_c > 0 else 0.0

    # Decode start latents if smap_c > 0
    if smap_c > 0:
        init = torch.cat([
            (vae32.decode(z0_all[j:j + 20].to(dev, torch.float32) / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
            for j in range(0, N, 20)
        ])
    else:
        init = torch.full((N, 3, 512, 512), 0.5)

    rec = torch.empty(N, 3, 512, 512, dtype=torch.float32)
    bs = 25
    for s in range(0, N, bs):
        q = list(range(s, min(N, s + bs)))
        Q = len(q)
        cc = c_all[q].to(dev, torch.float16)
        ips = proj(adapter(cc))
        pe_b = pe.expand(Q, -1, -1)
        ppe_b = ppe.expand(Q, -1)
        # IMPORTANT: Within each concept c, use IDENTICAL seed across all 10 photos when testing swap sensitivity!
        # Wait: for concept-shared fixed seed, seed depends ONLY on concept index (i // 10)!
        # Let's check: does fixing seed per concept also keep free-gen completely fair? YES!
        # Because GT images are independent of the noise seed, fixing seed per concept eliminates seed-noise
        # between photo A and photo B in the SAME concept while using 50 independent seeds across the 50 concepts!
        gens = [torch.Generator(dev).manual_seed(1_000_003 * (i // 10) + 12345) for i in q]
        eps = torch.stack([
            torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * (i // 10) + 99))
            for i in q
        ]).to(dev)
        z0 = z0_all[q].to(dev, torch.float32)
        i0 = init[q].to(dev, torch.float16)

        if smap_c > 0:
            def cb(pp, i_step, t_cur, ck):
                lt = ck["latents"].float()
                si = int(pp.scheduler.step_index)
                sig = pp.scheduler.sigmas[si].to(dev)
                a = max(0.0, min(1.0, hold_val - i_step))
                if a > 0:
                    lt = a * (z0 + sig * eps) + (1.0 - a) * lt
                if low_keep is not None and i_step in (steps - 2, steps - 1):
                    f_lp, w2, w3 = low_keep
                    g_t = w2 if i_step == steps - 2 else w3
                    if g_t > 0:
                        lt = lt + g_t * _lp_op((z0 + sig * eps) - lt, f_lp)
                ck["latents"] = lt.to(torch.float16)
                return ck

            lat_out = p2(
                image=i0,
                prompt_embeds=pe_b,
                pooled_prompt_embeds=ppe_b,
                num_inference_steps=steps,
                strength=s_unrel,
                guidance_scale=0.0,
                output_type="latent",
                generator=gens,
                cross_attention_kwargs={"ip_hidden_states": ips},
                callback_on_step_end=cb,
                callback_on_step_end_tensor_inputs=["latents"],
            ).images
        else:
            # Pure noise start (smap = 0.0, 4-step txt2img/IP-only) with identical concept-level initial noise!
            lat_out = pipe(
                prompt_embeds=pe_b,
                pooled_prompt_embeds=ppe_b,
                num_inference_steps=steps,
                guidance_scale=0.0,
                output_type="latent",
                latents=(eps * float(pipe.scheduler.init_noise_sigma)).to(torch.float16),
                generator=gens,
                cross_attention_kwargs={"ip_hidden_states": ips},
            ).images

        for j0 in range(0, Q, 10):
            img_b = (vae32.decode(lat_out[j0:j0 + 10].float() / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
            rec[s + j0:s + j0 + len(img_b)] = img_b

    # Score Intra10 & 2x2 Swap across CLIP, Inception, Alex5, Alex2, PixCorr, SSIM
    nets = _nets()
    out = {}
    for name, (mod, pre, layer) in nets.items():
        if name == "effnet":
            continue
        feats = []
        for i in range(0, N, 50):
            y = mod(pre(rec[i:i + 50].to(dev)))
            feats.append((y if layer is None else y[layer]).float().flatten(1).cpu())
        F_rec = torch.cat(feats, 0)
        F_gt = gt_feats[name]
        # Cosine similarity on centered features per 10-img concept
        Fr = nz(F_rec - F_rec.mean(0, keepdim=True), dim=1).view(50, 10, -1)
        Fg = nz(F_gt - F_gt.mean(0, keepdim=True), dim=1).view(50, 10, -1)
        sim = torch.bmm(Fr, Fg.transpose(1, 2)).numpy()
        m = intra10_and_swap_from_sim(sim)
        if name == "clip":
            # Also measure mean pairwise 1 - cos(G_A, G_B) within same concept (Swap Feature Displacement!)
            iu, ju = np.triu_indices(10, k=1)
            self_sim = torch.bmm(Fr, Fr.transpose(1, 2)).numpy()
            m["swap_clip_disp"] = float(np.mean(1.0 - self_sim[:, iu, ju]))
            # Also compute 500-way global CLIP top-1 for reference!
            Fr_g = nz(F_rec - F_rec.mean(0, keepdim=True), dim=1)
            Fg_g = nz(F_gt - F_gt.mean(0, keepdim=True), dim=1)
            S_all = (Fr_g @ Fg_g.T).numpy()
            m["global500_top1"] = float(np.mean(S_all.argmax(1) == np.arange(N)))
        out[name] = m

    # Low-level PixCorr & Pixel L1 swap displacement (fast at 128x128)
    r128 = F.interpolate(rec, size=128, mode="bilinear", align_corners=False)
    g128 = F.interpolate(gt_rgb50, size=128, mode="bilinear", align_corners=False)
    rv = r128.view(50, 10, -1)
    gv = g128.view(50, 10, -1)
    rv_c = nz(rv - rv.mean(dim=-1, keepdim=True), dim=-1)
    gv_c = nz(gv - gv.mean(dim=-1, keepdim=True), dim=-1)
    pix_sim = torch.bmm(rv_c, gv_c.transpose(1, 2)).numpy()
    out["pixcorr"] = intra10_and_swap_from_sim(pix_sim)
    out["pixcorr"]["diag_mean"] = float(np.mean(pix_sim[:, np.arange(10), np.arange(10)]))
    iu, ju = np.triu_indices(10, k=1)
    out["swap_pixel_l1"] = float((rv[:, iu] - rv[:, ju]).abs().mean())

    # SSIM diag mean (500 imgs at 256px for speed)
    r256 = F.interpolate(rec, size=256, mode="bilinear", align_corners=False).permute(0, 2, 3, 1).numpy()
    g256 = F.interpolate(gt_rgb50, size=256, mode="bilinear", align_corners=False).permute(0, 2, 3, 1).numpy()
    rg = rgb2gray(r256)
    gg = rgb2gray(g256)
    out["ssim_diag"] = float(np.mean([
        structural_similarity(rg[i], gg[i], gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
        for i in range(N)
    ]))
    return out


def main():
    dev = "cuda:0"
    t_start = time.time()
    concept_np, ok_np, tr_idx, f0_all10, f0_idx, sel50_cids, sel50_idx = get_concept_splits()
    print(f"=== PHASE 0 GATE === tr14={len(tr_idx)} imgs, fold0_all10={len(f0_all10)} concepts ({len(f0_idx)} imgs), D_intra50={len(sel50_idx)} imgs", flush=True)

    # 1. Train/Load clean tr14 Arm0 baseline
    ckpt_p = train_or_load_arm0_tr14(dev=dev, seed=2025, epochs=150)

    # 2. Layer 1: Raw EEG & Arm0 Representation Intra10
    layer1_res, c_sel50_by_sub, z0_sel50_by_sub = eval_rep_layer1(ckpt_p, concept_np, tr_idx, f0_all10, f0_idx, sel50_cids, sel50_idx, dev=dev)

    # 3. Load SDXL-Turbo + IP-Adapter + VAE32
    from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
    from metrics import _nets

    dt = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=dt, variant="fp16")
    IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(pipe.unet)
    pipe.to(dev, dt)
    pipe.unet.eval().requires_grad_(False)
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        pe, _, ppe, _ = pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
        add_time_ids = pipe._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=dt, text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).to(dev)
    del pipe.text_encoder, pipe.text_encoder_2
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    p2 = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    p2.set_progress_bar_config(disable=True)
    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, dt).eval().requires_grad_(False)
    proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, dt).eval().requires_grad_(False)
    vae32 = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval().requires_grad_(False)
    vae32.enable_slicing()

    # Load ground-truth latents & decode 500 GT images for D_intra50
    lat50 = torch.from_numpy(np.load(PREP / "lat512_train.npy")[sel50_idx]).float()  # [500, 4, 64, 64]
    F_gt50 = torch.from_numpy(np.load(PREP / "F_train.npy")[sel50_idx]).float()      # [500, 1024]

    # 4. Layer 2: Frozen SDXL-Turbo UNet Denoising-Error 2x2 Swap Test
    unet_arms = {
        "Oracle_Fgt": F_gt50,
        "Arm0_sub02": c_sel50_by_sub[2],
        "Arm0_sub08": c_sel50_by_sub[8],
    }
    layer2_res = eval_unet_denoising_layer2(pipe, adapter, proj, pe, ppe, add_time_ids, lat50, unet_arms, dev=dev)

    # 5. Layer 3: 4-Step Generated Image Intra10 & Controlled Swap Sensitivity
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

    # Shared concept-mean z0 (same z0 across all 10 photos of a concept, isolating condition swap!)
    z0_concept_mean_gt = lat50.view(50, 10, 4, 64, 64).mean(dim=1, keepdim=True).expand(-1, 10, -1, -1, -1).reshape(500, 4, 64, 64)
    z0_zero = torch.zeros_like(lat50)

    layer3_res = {}
    def _run_l3(tag, c_use, z0_use, smap_c, low_keep):
        m = generate_and_score_layer3(pipe, p2, vae32, adapter, proj, pe, ppe, gt_rgb50, gt_feats, c_use, z0_use, smap_c, low_keep, dev=dev)
        layer3_res[tag] = m
        print(
            f"[Layer3 {tag:28s}] "
            f"CLIP: t10={m['clip']['top1']*100:.2f}% dc10={m['clip']['top1_dc']*100:.2f}% sw={m['clip']['swap2x2']*100:.2f}% g500={m['clip']['global500_top1']*100:.2f}% | "
            f"Alex5: t10={m['alex5']['top1']*100:.2f}% sw={m['alex5']['swap2x2']*100:.2f}% | "
            f"Alex2: sw={m['alex2']['swap2x2']*100:.2f}% | PixC: sw={m['pixcorr']['swap2x2']*100:.2f}% ({m['pixcorr']['diag_mean']:.3f}) | "
            f"SSIM={m['ssim_diag']:.3f} | Disp: pxL1={m['swap_pixel_l1']:.4f} clip={m['clip']['swap_clip_disp']:.4f}",
            flush=True,
        )

    # Oracle ceilings
    _run_l3("Oracle_Fgt_FixedZ0_s625", F_gt50, z0_concept_mean_gt, 0.625, (8, 0.22, 0.04))
    _run_l3("Oracle_Fgt_PureNoise_s000", F_gt50, z0_zero, 0.0, None)

    # 4 Dev Subjects under R=4 EEG
    for sub in DEV_SUBS:
        c_e = c_sel50_by_sub[sub]
        z0_e = z0_sel50_by_sub[sub]
        z0_e_cmean = z0_e.view(50, 10, 4, 64, 64).mean(dim=1, keepdim=True).expand(-1, 10, -1, -1, -1).reshape(500, 4, 64, 64)
        _run_l3(f"EEG_s{sub:02d}_FixedZ0_s625", c_e, z0_e_cmean, 0.625, (8, 0.22, 0.04))
        _run_l3(f"EEG_s{sub:02d}_PureNoise_s000", c_e, z0_zero, 0.0, None)
        _run_l3(f"EEG_s{sub:02d}_FreeGen_O14", c_e, z0_e, 0.625, (8, 0.22, 0.04))

    # Aggregate 4-subject means for Layer 3
    summary = {"wall_s": round(time.time() - t_start, 1), "sel50_cids": sel50_cids.tolist(), "layer1": layer1_res, "layer2": layer2_res, "layer3": layer3_res}
    write_json(OUT_DIR / "exp0_gate_results.json", summary)
    print(f"=== PHASE 0 DONE in {summary['wall_s']}s -> {OUT_DIR / 'exp0_gate_results.json'} ===", flush=True)


if __name__ == "__main__":
    main()
