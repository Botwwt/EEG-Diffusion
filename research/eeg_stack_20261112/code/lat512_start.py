"""Native 512px (4x64x64) frequency-weighted QR-PCA + layout/contour-disambiguated start latents (lat512_start.py).

Modes (--mode_v):
  - v2 (default, exact backward-compatibility for O7a / O7 / B4-1a / B4a / B4):
      5 post-stimulus windows, raw flattened EEG, gate = r / (r + 0.03), spat1024.pt for B4.
  - w5 (5-fold OOF-calibrated for O8 and B5; zero test-image or evaluation-network usage):
      8 post-stimulus windows on 250 Hz (0-1000 ms, 4 ms/sample; THINGS-EEG2 RSVP SOA = 200 ms = 100 ms on + 100 ms blank):
        (8:48)=32-192 ms, (12:62)=48-248 ms, (10:75)=40-300 ms, (15:90)=60-360 ms,
        (20:110)=80-440 ms, (10:125)=40-500 ms, (25:165)=100-660 ms, (0:250)=0-1000 ms,
      with linear multi-scale temporal-pyramid + smoothed-derivative features, exact repeat-attenuation correction
      (R=4 -> R=80), Bayes-MMSE linear r^1.0 gating, native 512px multi-scale contour readout (spat_ms512.pt:
      single-subject spat_ss_o8 for O8, cross-subject spat_ms512_b5 for B5), and band-preserving kNN fusion.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common import PREP, RUNS, concept_folds

p = argparse.ArgumentParser()
p.add_argument("subs", help="Comma-separated subject ints or 'all'")
p.add_argument("--mode_v", choices=["v2", "w5"], default="v2")
p.add_argument("--cond_dirs", default="cone_mw6tta_w1.00")
p.add_argument("--spat_ss", default="", help="Optional single-subject contour cond dir for O8 (e.g. spat_ss_o8)")
p.add_argument("--n_pca", type=int, default=768)
p.add_argument("--w_sem", type=float, default=0.58)
p.add_argument("--k", type=int, default=30)
p.add_argument("--temp", type=float, default=0.05)
p.add_argument("--knn_pool", type=int, default=2)
p.add_argument("--spreads", default="0.80")
p.add_argument("--tag", default="lat512_o7_v2")
p.add_argument("--also_b4", default="", help="Optional 'cond_dir:spat_dir:tag' for B4/B5 export")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
subs = list(range(1, 11)) if args.subs == "all" else [int(s) for s in args.subs.split(",")]
spreads = [float(x) for x in args.spreads.split(",")]
cdirs = [x.strip() for x in args.cond_dirs.split(",") if x.strip()]
nz = F.normalize


def lp_flat(y_flat, f):
    v = y_flat.view(-1, 4, 64, 64)
    return F.interpolate(F.avg_pool2d(v, f), size=64, mode="bilinear", align_corners=False).flatten(1)


print(f"Loading lat512_train.npy on {dev} (mode_v={args.mode_v})...", flush=True)
ok_np = np.load(PREP / "lat512_train_ok.npy")
lat_np = np.load(PREP / "lat512_train.npy")
N = len(lat_np)
ok = torch.from_numpy(ok_np).to(dev)
Y_full = torch.from_numpy(lat_np).float().to(dev).flatten(1)
del lat_np

my = Y_full[ok].mean(0)
Yc_ok = Y_full[ok] - my
del Y_full
rms_real_lp4 = float(F.avg_pool2d(Yc_ok.view(-1, 4, 64, 64), 4).flatten(1).pow(2).sum(1).mean().sqrt())

n_pca_use = 1024 if (args.mode_v == "w5" and args.n_pca == 768) else args.n_pca
print(f"Computing frequency-weighted orthonormal top-{n_pca_use} QR-PCA basis...", flush=True)
torch.manual_seed(20261115)
Yc_weighted = Yc_ok + 1.5 * lp_flat(Yc_ok, 4) + 0.75 * lp_flat(Yc_ok, 2)
_, _, V_p = torch.pca_lowrank(Yc_weighted, q=n_pca_use + 32, center=False, niter=4)
del Yc_weighted
V_pca, _ = torch.linalg.qr(V_p[:, :n_pca_use].double(), mode="reduced")
Z_all = torch.zeros((N, n_pca_use), dtype=torch.float64, device=dev)
Z_all[ok] = Yc_ok.double() @ V_pca

F_tr = nz(torch.from_numpy(np.load(PREP / "F_train.npy")).float().to(dev), dim=1)
Fc_ok = nz(F_tr[ok] - F_tr[ok].mean(0, keepdim=True), dim=1)

V16_ok = F.avg_pool2d(Yc_ok.view(-1, 4, 64, 64), 4).flatten(1)
V16c_ok = nz(V16_ok - V16_ok.mean(0, keepdim=True), dim=1)

sp_file = PREP / ("spat_ms512.pt" if args.mode_v == "w5" else "spat1024.pt")
Sc_ok = None
W_sp2lat = None
if sp_file.exists():
    sp_pack = torch.load(sp_file, map_location="cpu", weights_only=True)
    S_tr_ok = nz(sp_pack["S_train"].float().to(dev)[ok], dim=1)
    Sc_ok = nz(S_tr_ok - S_tr_ok.mean(0, keepdim=True), dim=1)
    A_sp = Sc_ok.double()
    e_sp, V_sp = torch.linalg.eigh((A_sp.T @ A_sp) / len(A_sp))
    k_sp = 768 if args.mode_v == "w5" else 256
    W_sp2z = V_sp @ ((V_sp.T @ ((A_sp.T @ Z_all[ok, :k_sp]) / len(A_sp))) / (e_sp + 0.5 * e_sp.mean())[:, None])
    W_sp2lat = (W_sp2z @ V_pca[:, :k_sp].T).float()

concept = torch.from_numpy(np.load(PREP / "train_concept.npy"))
fold = torch.from_numpy(concept_folds(1654, 5))[concept].to(dev)
LAMS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
WINDOWS = (
    [(8, 48), (12, 62), (10, 75), (15, 90), (20, 110), (10, 125), (25, 165), (0, 250)]
    if args.mode_v == "w5"
    else [(12, 62), (10, 75), (15, 90), (10, 125), (0, 250)]
)


def ivw(x):
    tot = x.sum(1, keepdim=True)
    v = (x - (tot - x) / (x.shape[1] - 1)).pow(2).mean(dim=(2, 3))
    w = 1.0 / v.clamp_min(1e-12)
    return ((w / w.sum(1, keepdim=True))[:, :, None, None] * x).sum(1)


def pyr_linear_flat(x3d):
    """Linear multi-scale temporal pyramid + smoothed derivative on [M, 17, T]."""
    s2 = F.avg_pool1d(x3d, kernel_size=3, stride=2, padding=1)
    s4 = F.avg_pool1d(x3d, kernel_size=7, stride=4, padding=3)
    dx2 = s2[..., 1:] - s2[..., :-1]
    return torch.cat([
        x3d.flatten(1),
        s2.flatten(1) * 0.85,
        s4.flatten(1) * 0.75,
        dx2.flatten(1) * 0.70,
    ], dim=1)


def fit_window_rc(x_tr_w, x_te_w, Z_ok, fold_ok, ok_mask):
    if args.mode_v == "w5":
        N_, R_, C_, T_ = x_tr_w.shape
        xm_list = []
        for i in range(0, N_, 2048):
            xb = x_tr_w[i:i + 2048].float()
            xm_list.append(pyr_linear_flat(xb.reshape(-1, C_, T_)).view(len(xb), R_, -1).mean(1).double())
        xm = torch.cat(xm_list, dim=0)
        mu, sd = xm.mean(0), xm.std(0).clamp_min(1e-8)
        X = (xm - mu) / sd
        Xt = (pyr_linear_flat(x_te_w.float()).double() - mu) / sd
        S = torch.zeros((X.shape[1], X.shape[1]), dtype=torch.float64, device=dev)
        for i in range(0, N_, 1024):
            xb = x_tr_w[i:i + 1024].float()
            pf = pyr_linear_flat(xb.reshape(-1, C_, T_)).view(len(xb), R_, -1).double()
            d = ((pf - pf.mean(1, keepdim=True)) / sd).flatten(0, 1)
            S += d.T @ d
        S /= 3 * N_
    else:
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
    oof_preds = {lam: torch.zeros((n_ok, n_pca_use), dtype=torch.float64, device=dev) for lam in LAMS}
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
    mode_corrs = torch.zeros((len(LAMS), n_pca_use), dtype=torch.float64, device=dev)
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

    Z_test = torch.zeros((200, n_pca_use), dtype=torch.float64, device=dev)
    for li, lam in enumerate(LAMS):
        mask_m = best_li == li
        if mask_m.any():
            W_li = V_rc @ (P_rc[:, mask_m] / (e_rc + lam * pen_unit)[:, None])
            Z_test[:, mask_m] = Xt @ W_li
    Z_test = Z_test / Z_test.std(0, keepdim=True).clamp_min(1e-8) * Z_ok.std(0, keepdim=True)
    return Z_test, best_r


def lp4_rms(flat_lat):
    return float(F.avg_pool2d(flat_lat.view(200, 4, 64, 64), 4).flatten(1).pow(2).sum(1).mean().sqrt())


def save_single_sub(sub_idx, dev_eeg, cond_list, tag_name):
    ds = []
    for cd in cond_list:
        c = torch.from_numpy(np.load(RUNS / "cond" / cd / f"sub-{sub_idx:02d}.npy")).float().to(dev)
        cn = nz(c, dim=1)
        ds.append(nz(cn - cn.mean(0, keepdim=True), dim=1))
    q_sem = nz(torch.stack(ds).mean(0), dim=1)

    if args.mode_v == "w5" and args.spat_ss and Sc_ok is not None:
        sp_p = RUNS / "cond" / args.spat_ss / f"sub-{sub_idx:02d}.npy"
        if sp_p.exists():
            q_sp_ss = torch.from_numpy(np.load(sp_p)).float().to(dev)
            q_sp_ss = nz(nz(q_sp_ss, dim=1) - nz(q_sp_ss, dim=1).mean(0, keepdim=True), dim=1)
            dev_sp_ss = q_sp_ss @ W_sp2lat
            dev_eeg_use = 0.88 * (dev_eeg / max(lp4_rms(dev_eeg), 1e-6)) + 0.12 * (dev_sp_ss / max(lp4_rms(dev_sp_ss), 1e-6))
            eeg16 = F.avg_pool2d(dev_eeg_use.view(200, 4, 64, 64), 4).flatten(1)
            q_eeg16 = nz(eeg16 - eeg16.mean(0, keepdim=True), dim=1)
            sim = 0.65 * (q_sem @ Fc_ok.T) + 0.25 * (q_sp_ss @ Sc_ok.T) + 0.10 * (q_eeg16 @ V16c_ok.T)
        else:
            dev_eeg_use = dev_eeg / max(lp4_rms(dev_eeg), 1e-6)
            eeg16 = F.avg_pool2d(dev_eeg_use.view(200, 4, 64, 64), 4).flatten(1)
            q_eeg16 = nz(eeg16 - eeg16.mean(0, keepdim=True), dim=1)
            sim = 0.82 * (q_sem @ Fc_ok.T) + 0.18 * (q_eeg16 @ V16c_ok.T)
    else:
        dev_eeg_use = dev_eeg / max(lp4_rms(dev_eeg), 1e-6)
        eeg16 = F.avg_pool2d(dev_eeg.view(200, 4, 64, 64), 4).flatten(1)
        q_eeg16 = nz(eeg16 - eeg16.mean(0, keepdim=True), dim=1)
        sim = 0.82 * (q_sem @ Fc_ok.T) + 0.18 * (q_eeg16 @ V16c_ok.T)

    v, idx = sim.topk(args.k, dim=1)
    w = torch.softmax(v / args.temp, dim=1)
    dev_sem = (w[:, :, None] * Yc_ok[idx]).sum(1)
    if args.knn_pool > 1:
        dev_sem = lp_flat(dev_sem, args.knn_pool)

    u_normed = dev_eeg_use / max(lp4_rms(dev_eeg_use), 1e-6)
    w_sem_use = 0.55 if (args.mode_v == "w5" and args.w_sem == 0.58) else args.w_sem
    dev_low = u_normed + w_sem_use * (dev_sem / max(lp4_rms(dev_sem), 1e-6))
    b_rms = lp4_rms(dev_low)
    out_dir = RUNS / "low" / f"sub-{sub_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in spreads:
        if args.mode_v == "w5":
            # Band-preserving fusion: blend low/mid frequencies while preserving MMSE-gated high-pass edge residual
            u_hi = u_normed - lp_flat(u_normed, 2)
            base_scaled = dev_low * (sp * rms_real_lp4 / b_rms) + 0.35 * u_hi * (sp * rms_real_lp4)
        else:
            base_scaled = dev_low * (sp * rms_real_lp4 / b_rms)
        lat_out = (my + base_scaled).view(200, 4, 64, 64).cpu().numpy().astype(np.float32)
        np.save(out_dir / f"{tag_name}_n{sp:.2f}.npy", lat_out)


def save_full_stack(sub_idx, dev_eeg, cond_dir, spat_dir, tag_name):
    c = torch.from_numpy(np.load(RUNS / "cond" / cond_dir / f"sub-{sub_idx:02d}.npy")).float().to(dev)
    cn = nz(c, dim=1)
    q_sem = nz(cn - cn.mean(0, keepdim=True), dim=1)

    spat_path = RUNS / "cond" / spat_dir / f"sub-{sub_idx:02d}.npy"
    if spat_path.exists() and Sc_ok is not None:
        q_spat = torch.from_numpy(np.load(spat_path)).float().to(dev)
        q_spat = nz(nz(q_spat, dim=1) - nz(q_spat, dim=1).mean(0, keepdim=True), dim=1)
        dev_spat = q_spat @ W_sp2lat
        if args.mode_v == "w5":
            dev_eeg_use = 0.80 * (dev_eeg / max(lp4_rms(dev_eeg), 1e-6)) + 0.20 * (dev_spat / max(lp4_rms(dev_spat), 1e-6))
            eeg16 = F.avg_pool2d(dev_eeg_use.view(200, 4, 64, 64), 4).flatten(1)
            q_eeg16 = nz(eeg16 - eeg16.mean(0, keepdim=True), dim=1)
            sim = 0.62 * (q_sem @ Fc_ok.T) + 0.28 * (q_spat @ Sc_ok.T) + 0.10 * (q_eeg16 @ V16c_ok.T)
        else:
            sim = 0.72 * (q_sem @ Fc_ok.T) + 0.28 * (q_spat @ Sc_ok.T)
            dev_eeg_use = 0.75 * (dev_eeg / max(lp4_rms(dev_eeg), 1e-6)) + 0.25 * (dev_spat / max(lp4_rms(dev_spat), 1e-6))
    else:
        sim = q_sem @ Fc_ok.T
        dev_eeg_use = dev_eeg / max(lp4_rms(dev_eeg), 1e-6)

    v, idx = sim.topk(args.k, dim=1)
    w = torch.softmax(v / args.temp, dim=1)
    dev_sem = (w[:, :, None] * Yc_ok[idx]).sum(1)
    if args.knn_pool > 1:
        dev_sem = lp_flat(dev_sem, args.knn_pool)

    u_normed = dev_eeg_use / max(lp4_rms(dev_eeg_use), 1e-6)
    w_sem_use = 0.55 if (args.mode_v == "w5" and args.w_sem == 0.58) else args.w_sem
    dev_low = u_normed + w_sem_use * (dev_sem / max(lp4_rms(dev_sem), 1e-6))
    b_rms = lp4_rms(dev_low)
    out_dir = RUNS / "low" / f"sub-{sub_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in spreads:
        if args.mode_v == "w5":
            u_hi = u_normed - lp_flat(u_normed, 2)
            base_scaled = dev_low * (sp * rms_real_lp4 / b_rms) + 0.35 * u_hi * (sp * rms_real_lp4)
        else:
            base_scaled = dev_low * (sp * rms_real_lp4 / b_rms)
        lat_out = (my + base_scaled).view(200, 4, 64, 64).cpu().numpy().astype(np.float32)
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

    if args.mode_v == "w5":
        gate = (best_r.clamp_min(0.0) ** 1.0) * (best_r > 0.015).double()
    else:
        gate = (best_r / (best_r + 0.03)) * (best_r > 0.015).double()
    dev_eeg = ((Z_test_pred * gate[None, :]) @ V_pca.T).float()

    save_single_sub(sub, dev_eeg, cdirs, args.tag)
    extra_msg = ""
    if args.also_b4:
        b4_cd, b4_sp, b4_tag = args.also_b4.split(":")
        if (RUNS / "cond" / b4_cd / f"sub-{sub:02d}.npy").exists():
            save_full_stack(sub, dev_eeg, b4_cd, b4_sp, b4_tag)
            extra_msg = f" + {b4_tag}"

    print(f"sub-{sub:02d} [{args.mode_v}] | active_modes={(best_r > 0.02).sum().item()}/{n_pca_use} max_r5={best_r.max():.3f} "
          f"mean_top64_r5={best_r[:64].mean():.3f} -> saved {args.tag}{extra_msg}", flush=True)
