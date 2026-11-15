"""Native 512px (4x64x64) 5-window frequency-weighted QR-PCA + layout/contour-disambiguated start latents (lat512_start.py).

Components:
  1. Frequency-weighted orthonormal QR-PCA basis on lat512_train (up-weighting coarse 16x16 & 32x32 spatial modes).
  2. 5 post-stimulus EEG windows on 250 Hz (0-1000 ms, 4 ms/sample; THINGS-EEG2 RSVP SOA = 200 ms = 100 ms on + 100 ms blank):
     (12:62) = 48-248 ms, (10:75) = 40-300 ms, (15:90) = 60-360 ms, (10:125) = 40-500 ms, (0:250) = 0-1000 ms,
     fused per PCA mode strictly by 5-fold concept-disjoint OOF r^2.
  3. Layout- & contour-disambiguated kNN prototype:
     - Single-Subject Track (O7): disambiguates semantic kNN exemplars using single-subject 16x16 ridge layout (0.82 sem + 0.18 eeg16).
     - Full-Stack Track (B4): disambiguates semantic kNN exemplars using the 8-encoder multi-window spatial-contour query
       spat_ms_mwtta (0.72 sem + 0.28 spat) and blends a 0.25-weight cross-subject spatial-contour latent readout.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common import PREP, RUNS, concept_folds

p = argparse.ArgumentParser()
p.add_argument("subs", help="Comma-separated subject ints or 'all'")
p.add_argument("--cond_dirs", default="cone_mw6tta_w1.00")
p.add_argument("--n_pca", type=int, default=768)
p.add_argument("--w_sem", type=float, default=0.58)
p.add_argument("--k", type=int, default=30)
p.add_argument("--temp", type=float, default=0.05)
p.add_argument("--knn_pool", type=int, default=2)
p.add_argument("--spreads", default="0.80")
p.add_argument("--tag", default="lat512_o7_v2")
p.add_argument("--also_b4", default="", help="Optional 'cond_dir:spat_dir:tag' for B4 export")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
subs = list(range(1, 11)) if args.subs == "all" else [int(s) for s in args.subs.split(",")]
spreads = [float(x) for x in args.spreads.split(",")]
cdirs = [x.strip() for x in args.cond_dirs.split(",") if x.strip()]
nz = F.normalize


def lp_flat(y_flat, f):
    v = y_flat.view(-1, 4, 64, 64)
    return F.interpolate(F.avg_pool2d(v, f), size=64, mode="bilinear", align_corners=False).flatten(1)


print(f"Loading lat512_train.npy on {dev}...", flush=True)
ok_np = np.load(PREP / "lat512_train_ok.npy")
lat_np = np.load(PREP / "lat512_train.npy")
N = len(lat_np)
ok = torch.from_numpy(ok_np).to(dev)
Y_full = torch.from_numpy(lat_np).float().to(dev).flatten(1)
del lat_np

my = Y_full[ok].mean(0)
Yc_ok = Y_full[ok] - my
rms_real_lp4 = float(F.avg_pool2d(Yc_ok.view(-1, 4, 64, 64), 4).flatten(1).pow(2).sum(1).mean().sqrt())

print(f"Computing frequency-weighted orthonormal top-{args.n_pca} QR-PCA basis...", flush=True)
torch.manual_seed(20261115)
Yc_weighted = Yc_ok + 1.5 * lp_flat(Yc_ok, 4) + 0.75 * lp_flat(Yc_ok, 2)
_, _, V_p = torch.pca_lowrank(Yc_weighted, q=args.n_pca + 32, center=False, niter=4)
del Yc_weighted
V_pca, _ = torch.linalg.qr(V_p[:, :args.n_pca].double(), mode="reduced")
Z_all = torch.zeros((N, args.n_pca), dtype=torch.float64, device=dev)
Z_all[ok] = Yc_ok.double() @ V_pca

F_tr = nz(torch.from_numpy(np.load(PREP / "F_train.npy")).float().to(dev), dim=1)
Fc_ok = nz(F_tr[ok] - F_tr[ok].mean(0, keepdim=True), dim=1)

V16_ok = F.avg_pool2d(Yc_ok.view(-1, 4, 64, 64), 4).flatten(1)
V16c_ok = nz(V16_ok - V16_ok.mean(0, keepdim=True), dim=1)

Sc_ok = None
W_sp2lat = None
if (PREP / "spat1024.pt").exists():
    sp_pack = torch.load(PREP / "spat1024.pt", map_location="cpu", weights_only=True)
    S_tr_ok = nz(sp_pack["S_train"].float().to(dev)[ok], dim=1)
    Sc_ok = nz(S_tr_ok - S_tr_ok.mean(0, keepdim=True), dim=1)
    A_sp = Sc_ok.double()
    e_sp, V_sp = torch.linalg.eigh((A_sp.T @ A_sp) / len(A_sp))
    W_sp2z = V_sp @ ((V_sp.T @ ((A_sp.T @ Z_all[ok, :256]) / len(A_sp))) / (e_sp + 0.5 * e_sp.mean())[:, None])
    W_sp2lat = (W_sp2z @ V_pca[:, :256].T).float()  # [1024, 16384]

concept = torch.from_numpy(np.load(PREP / "train_concept.npy"))
fold = torch.from_numpy(concept_folds(1654, 5))[concept].to(dev)
LAMS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
WINDOWS = [(12, 62), (10, 75), (15, 90), (10, 125), (0, 250)]


def ivw(x):
    tot = x.sum(1, keepdim=True)
    v = (x - (tot - x) / (x.shape[1] - 1)).pow(2).mean(dim=(2, 3))
    w = 1.0 / v.clamp_min(1e-12)
    return ((w / w.sum(1, keepdim=True))[:, :, None, None] * x).sum(1)


def fit_window_rc(x_tr_w, x_te_w, Z_ok, fold_ok, ok_mask):
    xm = x_tr_w.mean(1).flatten(1).double()
    mu, sd = xm.mean(0), xm.std(0).clamp_min(1e-8)
    X = (xm - mu) / sd
    Xt = (x_te_w.flatten(1).double() - mu) / sd
    S = torch.zeros((X.shape[1], X.shape[1]), dtype=torch.float64, device=dev)
    for i in range(0, len(x_tr_w), 2000):
        d = ((x_tr_w[i:i + 2000] - x_tr_w[i:i + 2000].mean(1, keepdim=True)).flatten(2).double() / sd).flatten(0, 1)
        S += d.T @ d
    S /= 3 * len(x_tr_w)

    X_ok = X[ok_mask]
    n_ok = int(ok_mask.sum())
    oof_preds = {lam: torch.zeros((n_ok, args.n_pca), dtype=torch.float64, device=dev) for lam in LAMS}
    for k in range(5):
        tr, ho = fold_ok != k, fold_ok == k
        ntr = int(tr.sum())
        e, V = torch.linalg.eigh((X_ok[tr].T @ X_ok[tr]) / ntr)
        P = V.T @ ((X_ok[tr].T @ Z_ok[tr]) / ntr)
        emean = float(e.mean())
        for lam in LAMS:
            oof_preds[lam][ho] = X_ok[ho] @ (V @ (P / (e + lam * emean)[:, None]))

    Zc = Z_ok - Z_ok.mean(0, keepdim=True)
    Z_norm = Zc.norm(dim=0).clamp_min(1e-8)
    mode_corrs = torch.zeros((len(LAMS), args.n_pca), dtype=torch.float64, device=dev)
    for li, lam in enumerate(LAMS):
        pc = oof_preds[lam] - oof_preds[lam].mean(0, keepdim=True)
        mode_corrs[li] = (pc * Zc).sum(0) / (pc.norm(dim=0).clamp_min(1e-8) * Z_norm)

    best_li = mode_corrs.argmax(dim=0)
    best_r = mode_corrs.gather(0, best_li[None])[0].clamp(min=0.0)

    M = (X_ok.T @ X_ok) / n_ok
    pen_unit = float(torch.trace(M)) / M.shape[0]
    e_s, V_s = torch.linalg.eigh(M - S / 4.0)
    M_rc = (V_s * e_s.clamp_min(0.0)) @ V_s.T + S / 80.0
    e_rc, V_rc = torch.linalg.eigh(M_rc)
    P_rc = V_rc.T @ ((X_ok.T @ Z_ok) / n_ok)

    Z_test = torch.zeros((200, args.n_pca), dtype=torch.float64, device=dev)
    for li, lam in enumerate(LAMS):
        mask_m = best_li == li
        if mask_m.any():
            W_li = V_rc @ (P_rc[:, mask_m] / (e_rc + lam * pen_unit)[:, None])
            Z_test[:, mask_m] = Xt @ W_li
    Z_test = Z_test / Z_test.std(0, keepdim=True).clamp_min(1e-8) * Z_ok.std(0, keepdim=True)
    return Z_test, best_r


def lp4_rms(flat_lat):
    return float(F.avg_pool2d(flat_lat.view(200, 4, 64, 64), 4).flatten(1).pow(2).sum(1).mean().sqrt())


def save_o7(sub_idx, dev_eeg, cond_list, tag_name):
    ds = []
    for cd in cond_list:
        c = torch.from_numpy(np.load(RUNS / "cond" / cd / f"sub-{sub_idx:02d}.npy")).float().to(dev)
        cn = nz(c, dim=1)
        ds.append(nz(cn - cn.mean(0, keepdim=True), dim=1))
    q_sem = nz(torch.stack(ds).mean(0), dim=1)

    eeg16 = F.avg_pool2d(dev_eeg.view(200, 4, 64, 64), 4).flatten(1)
    q_eeg16 = nz(eeg16 - eeg16.mean(0, keepdim=True), dim=1)
    sim = 0.82 * (q_sem @ Fc_ok.T) + 0.18 * (q_eeg16 @ V16c_ok.T)
    v, idx = sim.topk(args.k, dim=1)
    w = torch.softmax(v / args.temp, dim=1)
    dev_sem = (w[:, :, None] * Yc_ok[idx]).sum(1)
    if args.knn_pool > 1:
        dev_sem = lp_flat(dev_sem, args.knn_pool)

    dev_blend = (dev_eeg / max(lp4_rms(dev_eeg), 1e-6)) + args.w_sem * (dev_sem / max(lp4_rms(dev_sem), 1e-6))
    b_rms = lp4_rms(dev_blend)
    out_dir = RUNS / "low" / f"sub-{sub_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in spreads:
        lat_out = (my + dev_blend * (sp * rms_real_lp4 / b_rms)).view(200, 4, 64, 64).cpu().numpy().astype(np.float32)
        np.save(out_dir / f"{tag_name}_n{sp:.2f}.npy", lat_out)


def save_b4(sub_idx, dev_eeg, cond_dir, spat_dir, tag_name):
    c = torch.from_numpy(np.load(RUNS / "cond" / cond_dir / f"sub-{sub_idx:02d}.npy")).float().to(dev)
    cn = nz(c, dim=1)
    q_sem = nz(cn - cn.mean(0, keepdim=True), dim=1)

    spat_path = RUNS / "cond" / spat_dir / f"sub-{sub_idx:02d}.npy"
    if spat_path.exists() and Sc_ok is not None:
        q_spat = torch.from_numpy(np.load(spat_path)).float().to(dev)
        q_spat = nz(q_spat - q_spat.mean(0, keepdim=True), dim=1)
        sim = 0.72 * (q_sem @ Fc_ok.T) + 0.28 * (q_spat @ Sc_ok.T)
        dev_spat = q_spat @ W_sp2lat
        dev_eeg_use = 0.75 * (dev_eeg / max(lp4_rms(dev_eeg), 1e-6)) + 0.25 * (dev_spat / max(lp4_rms(dev_spat), 1e-6))
    else:
        sim = q_sem @ Fc_ok.T
        dev_eeg_use = dev_eeg / max(lp4_rms(dev_eeg), 1e-6)

    v, idx = sim.topk(args.k, dim=1)
    w = torch.softmax(v / args.temp, dim=1)
    dev_sem = (w[:, :, None] * Yc_ok[idx]).sum(1)
    if args.knn_pool > 1:
        dev_sem = lp_flat(dev_sem, args.knn_pool)

    dev_blend = (dev_eeg_use / max(lp4_rms(dev_eeg_use), 1e-6)) + args.w_sem * (dev_sem / max(lp4_rms(dev_sem), 1e-6))
    b_rms = lp4_rms(dev_blend)
    out_dir = RUNS / "low" / f"sub-{sub_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in spreads:
        lat_out = (my + dev_blend * (sp * rms_real_lp4 / b_rms)).view(200, 4, 64, 64).cpu().numpy().astype(np.float32)
        np.save(out_dir / f"{tag_name}_n{sp:.2f}.npy", lat_out)


for sub in subs:
    x_tr = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_train.npy")).to(dev)
    x_te = ivw(torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_test.npy")).float().to(dev))
    fold_ok = fold[ok]
    Z_ok = Z_all[ok]

    z_preds, r_list = [], []
    for t0, t1 in WINDOWS:
        zw, rw = fit_window_rc(x_tr[..., t0:t1].contiguous(), x_te[..., t0:t1].contiguous(), Z_ok, fold_ok, ok)
        z_preds.append(zw)
        r_list.append(rw)
    del x_tr

    R_mat = torch.stack(r_list, dim=0)
    W_mat = R_mat.pow(2) / (R_mat.pow(2).sum(0, keepdim=True) + 1e-10)
    Z_test_pred = (W_mat[:, None, :] * torch.stack(z_preds, dim=0)).sum(0)
    Z_test_pred = Z_test_pred / Z_test_pred.std(0, keepdim=True).clamp_min(1e-8) * Z_ok.std(0, keepdim=True)
    best_r = R_mat.max(0).values

    gate = (best_r / (best_r + 0.03)) * (best_r > 0.015).double()
    dev_eeg = ((Z_test_pred * gate[None, :]) @ V_pca.T).float()

    save_o7(sub, dev_eeg, cdirs, args.tag)
    extra_msg = ""
    if args.also_b4:
        b4_cd, b4_sp, b4_tag = args.also_b4.split(":")
        if (RUNS / "cond" / b4_cd / f"sub-{sub:02d}.npy").exists():
            save_b4(sub, dev_eeg, b4_cd, b4_sp, b4_tag)
            extra_msg = f" + {b4_tag}"

    print(f"sub-{sub:02d} | active_modes={(best_r > 0.02).sum().item()}/{args.n_pca} max_r5={best_r.max():.3f} "
          f"mean_top64_r5={best_r[:64].mean():.3f} -> saved {args.tag}{extra_msg}", flush=True)
