"""Stage 1 of ObjDRaFT-Bridge: 2-Fold Cross-Concept Unseen-Concept OOF EEG Predictions (100% train-only, zero test data)."""
import os, sys, gc, math, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RAD = Path(os.environ.get("RAD_ROOT", "/path/to/rad"))
for sub in ("code/exp_obj", "code/exp_w7", "code/exp_w6", "code/rad", "code/exp", "code/Brain-HIVE"):
    p = str(RAD / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from common import EMB_KEYS, PREP
from deep_enc import TargetSNRGate, sample_sub_repeats
from train_flex import ivw_mean, pyramid_feats, shift_batch
from o14_triarch_xbrain_bigg import CrossSubSingleTargetKVPyr
from o11_tri_stream_bigg import OnConeKVProjector, _hybrid_infonce
from o13_xbrain_shared_trunk import _xbrain_cone_nce
from train_obj_sem_enc import CrossSubObjSemEncoder, supcon_concept_loss, xbrain_sem_loss

nz = F.normalize
ALL_SUBS = tuple(range(1, 11))
OOF_DIR = RAD / "runs/obj_bench/oof_unseen"


def train_fold(fold: int, dev: str = "cuda:0", epochs: int = 90):
    """Fold 0 trains on cid % 2 == 0, predicts on UNSEEN cid % 2 == 1; Fold 1 vice-versa."""
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    out_p = OOF_DIR / f"oof_fold{fold}.pt"
    if out_p.exists():
        print(f"[CACHED] {out_p}", flush=True)
        return

    torch.manual_seed(4200 + fold); torch.cuda.manual_seed_all(4200 + fold)
    tb = torch.load(RAD / "runs/obj_bench/text_banks/text_banks_laion.pt", map_location="cpu", weights_only=False)
    tr_cid = tb["tr_img2cid"].long()  # [16540] in 0..1626
    tr_mask = (tr_cid % 2 == fold)
    va_mask = ~tr_mask
    tr_idx = torch.where(tr_mask)[0]
    va_idx = torch.where(va_mask)[0]
    print(f"[OOF Fold {fold}] train imgs={len(tr_idx)} (concepts={tr_cid[tr_idx].unique().numel()}), unseen val imgs={len(va_idx)} (concepts={tr_cid[va_idx].unique().numel()})", flush=True)

    F_all = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    F_tr = F_all[tr_idx].to(dev)
    c_tr = tr_cid[tr_idx].to(dev)
    n_tr = len(tr_idx)

    # Load all 10 subs train EEG
    x_all = [torch.from_numpy(np.load(PREP / f"sub-{s:02d}_train.npy")).half() for s in ALL_SUBS]

    # 1. Train Cross-Subject Visual F-encoder on fold train ONLY
    t0 = time.time()
    kv_p = OnConeKVProjector(str(PREP / "unet_kv_metric.pt"), device=dev)
    kv_p.calibrate_fold(F_tr)
    m_vis = CrossSubSingleTargetKVPyr(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    for sid in range(10):
        m_vis.ebs[sid].fit_from_fold(x_all[sid][tr_idx].float().to(dev))

    ls1 = nn.Parameter(torch.tensor(2.6592, device=dev))
    lsx = nn.Parameter(torch.tensor(2.6592, device=dev))
    dec, ndec = [], []
    for n, p_ in list(m_vis.named_parameters()) + [("ls1", ls1), ("lsx", lsx)]:
        (ndec if (n.endswith("bias") or "norm" in n or n.startswith("ls") or "emb" in n or "logit" in n) else dec).append(p_)
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.05}, {"params": ndec, "weight_decay": 0.0}], lr=5e-4, betas=(0.9, 0.995))

    bs = 1024
    spr = math.ceil(n_tr / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 0.5 * (1.0 + math.cos(math.pi * s / (spr * epochs)))))
    g = torch.Generator(device=dev).manual_seed(4200 + fold)
    rng = np.random.default_rng(4200 + fold)

    m_vis.train()
    for ep in range(epochs):
        perm = torch.randperm(n_tr, device=dev, generator=g)
        sub_order = rng.permutation(10)
        for b in range(spr):
            loc = perm[b * bs:(b + 1) * bs]
            g_idx = tr_idx[loc.cpu()]
            c_b = c_tr[loc]
            sid_a = int(sub_order[(2 * b) % 10])
            sid_b = int(sub_order[(2 * b + 1) % 10])
            if sid_b == sid_a:
                sid_b = (sid_a + 1) % 10
            xa, ka = sample_sub_repeats(x_all[sid_a][g_idx].float().to(dev), g)
            xb, kb = sample_sub_repeats(x_all[sid_b][g_idx].float().to(dev), g)
            xa, xb = shift_batch(xa, 2, g), shift_batch(xb, 2, g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sc1 = ls1.float().exp().clamp(max=100.0)
                scx = lsx.float().exp().clamp(max=100.0)
                with torch.no_grad():
                    ft = F_tr[loc].float()
                    ut = nz(ft, dim=-1)
                    dt = nz(ut - (ut @ kv_p.u_tr.T) * kv_p.u_tr, dim=-1)
                    yk = nz(kv_p.kv.project_kv(ft) - kv_p.kv.mu_kv, dim=-1)
                    si = 0.5 * (dt @ dt.T) + 0.5 * (yk @ yk.T)
                    diff_c = c_b[:, None] != c_b[None, :]
                    eye = torch.eye(len(loc), dtype=torch.bool, device=dev)
                    sbar = si[diff_c].mean()
                    m_a = 0.15 * scx.detach() * F.relu(si - sbar) * (~eye).float()
                fa = m_vis.forward_sub(xa, sid_a, r_eff=ka)
                fb = m_vis.forward_sub(xb, sid_b, r_eff=kb)
                da, zka = kv_p.cone_and_kv(fa)
                db, zkb = kv_p.cone_and_kv(fb)
                l_vis = 0.5 * (_hybrid_infonce(da, zka, dt, yk, sc1) + _hybrid_infonce(db, zkb, dt, yk, sc1))
                l_xb = _xbrain_cone_nce(fa, fb, kv_p.u_tr, scx, m_a)
                loss = l_vis + 0.25 * l_xb
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec + ndec, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

    print(f"[OOF Fold {fold}] Vis F-encoder trained in {time.time()-t0:.1f}s", flush=True)

    # Predict on UNSEEN fold va_idx across all 10 subjects (R=4 ivw_mean per subject!)
    m_vis.eval()
    c_oof = torch.empty(10, len(va_idx), 1024, dtype=torch.float16)
    f_norm_ref = F_tr.norm(dim=-1).mean().item()
    with torch.no_grad():
        for sid in range(10):
            preds = []
            for j in range(0, len(va_idx), 1024):
                sub_idx = va_idx[j:j + 1024]
                xv = ivw_mean(x_all[sid][sub_idx].float().to(dev))
                pv = m_vis.forward_sub(xv, sid, r_eff=4.0).float()
                preds.append((nz(pv, dim=-1) * f_norm_ref).half().cpu())
            c_oof[sid] = torch.cat(preds, dim=0)
    # Check OOF unseen-concept cosine rho (single-sub R=4 vs 4-sub avg!)
    gt_va_u = nz(F_all[va_idx], dim=-1)
    rho_1sub = (nz(c_oof[0].float(), dim=-1) * gt_va_u).sum(-1).mean().item()
    rho_4sub = (nz(c_oof[:4].float().mean(0), dim=-1) * gt_va_u).sum(-1).mean().item()
    rho_10sub = (nz(c_oof.float().mean(0), dim=-1) * gt_va_u).sum(-1).mean().item()
    print(f"[OOF Fold {fold}] UNSEEN-CONCEPT F-rho: 1-sub(R4)={rho_1sub:.3f} | 4-sub_avg={rho_4sub:.3f} | 10-sub_avg={rho_10sub:.3f}", flush=True)
    del m_vis, kv_p, F_tr; torch.cuda.empty_cache()

    # 2. Train Cross-Subject Semantic Encoder (ObjSemEnc) on fold train ONLY
    t1 = time.time()
    h14_raw = tb["h14_train"][tr_cid[tr_idx]].float()
    bg_raw = tb["bigg_train"][tr_cid[tr_idx]].float()
    mu_h = h14_raw.mean(0, keepdim=True)
    mu_b = bg_raw.mean(0, keepdim=True)
    T_h = nz(h14_raw - mu_h, dim=-1).to(dev)
    T_b = nz(bg_raw - mu_b, dim=-1).to(dev)

    m_sem = CrossSubObjSemEncoder(n_subs=10, hidden_size=1024, num_layers=3, dropout=0.25, in_drop=0.15).to(dev)
    for sid in range(10):
        m_sem.ebs[sid].fit_from_fold(x_all[sid][tr_idx].float().to(dev))

    ls_h = nn.Parameter(torch.tensor(2.6592, device=dev))
    ls_b = nn.Parameter(torch.tensor(2.6592, device=dev))
    ls_x = nn.Parameter(torch.tensor(2.6592, device=dev))
    dec, ndec = [], []
    for n, p_ in list(m_sem.named_parameters()) + [("ls_h", ls_h), ("ls_b", ls_b), ("ls_x", ls_x)]:
        (ndec if (n.endswith("bias") or "norm" in n or n.startswith("ls") or "emb" in n or "logit" in n) else dec).append(p_)
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.05}, {"params": ndec, "weight_decay": 0.0}], lr=5e-4, betas=(0.9, 0.995))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 0.5 * (1.0 + math.cos(math.pi * s / (spr * epochs)))))

    m_sem.train()
    for ep in range(epochs):
        perm = torch.randperm(n_tr, device=dev, generator=g)
        sub_order = rng.permutation(10)
        for b in range(spr):
            loc = perm[b * bs:(b + 1) * bs]
            g_idx = tr_idx[loc.cpu()]
            c_b = c_tr[loc]
            sid_a = int(sub_order[(2 * b) % 10])
            sid_b = int(sub_order[(2 * b + 1) % 10])
            if sid_b == sid_a:
                sid_b = (sid_a + 1) % 10
            xa, ka = sample_sub_repeats(x_all[sid_a][g_idx].float().to(dev), g)
            xb, kb = sample_sub_repeats(x_all[sid_b][g_idx].float().to(dev), g)
            xa, xb = shift_batch(xa, 2, g), shift_batch(xb, 2, g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sh = ls_h.float().exp().clamp(max=100.0)
                sb = ls_b.float().exp().clamp(max=100.0)
                sx = ls_x.float().exp().clamp(max=100.0)
                th, tb_t = T_h[loc], T_b[loc]
                ha, ba, _, _ = m_sem.forward_sub(xa, sid_a, r_eff=ka)
                hb, bb, _, _ = m_sem.forward_sub(xb, sid_b, r_eff=kb)
                def _l(h_, b_):
                    hu = nz(h_.float() - h_.float().mean(0, keepdim=True), dim=-1)
                    bu = nz(b_.float() - b_.float().mean(0, keepdim=True), dim=-1)
                    return 0.5 * (supcon_concept_loss(hu, th, c_b, sh) + 0.25 * (1.0 - (hu * th).sum(-1)).mean()) + \
                           0.5 * (supcon_concept_loss(bu, tb_t, c_b, sb) + 0.25 * (1.0 - (bu * tb_t).sum(-1)).mean())
                loss = 0.5 * (_l(ha, ba) + _l(hb, bb)) + 0.25 * (0.5 * xbrain_sem_loss(ha, hb, c_b, sx) + 0.5 * xbrain_sem_loss(ba, bb, c_b, sx))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec + ndec, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

    print(f"[OOF Fold {fold}] Sem encoder trained in {time.time()-t1:.1f}s", flush=True)
    m_sem.eval()
    h_oof = torch.empty(10, len(va_idx), 1024, dtype=torch.float16)
    b_oof = torch.empty(10, len(va_idx), 1280, dtype=torch.float16)
    with torch.no_grad():
        for sid in range(10):
            hs, bs_ = [], []
            for j in range(0, len(va_idx), 1024):
                sub_idx = va_idx[j:j + 1024]
                xv = ivw_mean(x_all[sid][sub_idx].float().to(dev))
                hp, bp, _, _ = m_sem.forward_sub(xv, sid, r_eff=4.0)
                hs.append(hp.float().cpu()); bs_.append(bp.float().cpu())
            hc = torch.cat(hs, 0); bc = torch.cat(bs_, 0)
            h_oof[sid] = nz(hc - hc.mean(0, keepdim=True), dim=-1).half()
            b_oof[sid] = nz(bc - bc.mean(0, keepdim=True), dim=-1).half()

    torch.save({"va_idx": va_idx, "c_oof": c_oof, "h_oof": h_oof, "b_oof": b_oof}, out_p)
    print(f"[OOF Fold {fold}] Saved {out_p}!", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=90)
    args = ap.parse_args()
    train_fold(args.fold, dev="cuda:0", epochs=args.epochs)
