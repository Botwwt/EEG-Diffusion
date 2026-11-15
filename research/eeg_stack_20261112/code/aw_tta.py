"""All-window split-half TTA + repeat bagging + BoundedConditionAdapter + Conical Optimal Transport (aw_tta.py).

Produces strictly separated condition sets across subjects:
  1. Single-Subject Original Track (O7):
     42 single-subject encoders (6 windows x 7 seeds) + split-half TTA + 10-group IVW bagging +
     per-subject BoundedConditionAdapter + F_train Conical Optimal Transport:
       -> runs/cond/cone_mw6tta_w{1.00, 1.45, 1.50}/sub-XX.npy  (no cross-subject parameters)

  2. Full-Stack Semantic Condition (B4-1 / B4):
     Up to 28 multi-window multi-seed joint semantic encoders + split-half TTA + 10-group IVW bagging +
     shared BoundedConditionAdapter (ord_ms_e10_ft10in/pooled_p1/head.pt) + 0.80/0.20 blend with mw6tta +
     F_train Conical Optimal Transport:
       -> runs/cond/cone_ms28_mwtta_w{1.00, 1.55, 1.65, 1.75}/sub-XX.npy

  3. Full-Stack Retinotopic-Steerable-Contour Query (B4):
     8 multi-window multi-seed joint spatial-contour encoders (ms_spat_*) + split-half TTA + 10-group IVW bagging:
       -> runs/cond/spat_ms_mwtta/sub-XX.npy
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import HIVE, PREP, ROOT, RUNS

sys.path.insert(0, str(HIVE))
_eeg_dir = Path(__file__).resolve().parents[2] / "eeg"  # research/eeg of this repository
if not _eeg_dir.exists():
    _eeg_dir = ROOT / "code/EEG-Diffusion/research/eeg"
sys.path.insert(0, str(_eeg_dir))
from brainhive_bounded_component_v1 import BoundedConditionAdapter  # noqa: E402
from main.models_brain import BrainEncoder  # noqa: E402

subs = list(range(1, 11)) if sys.argv[1] == "all" else [int(x) for x in sys.argv[1].split(",")]
dev = "cuda" if torch.cuda.is_available() else "cpu"
nz = F.normalize
GK, BK = "backbone.block.layers.0.norm.weight", "backbone.block.layers.0.norm.bias"

SS_TAGS = ["t0_r4", "t0_r4_w200", "t0_r4_w175", "t0_r4_w150", "t0_r4_w125", "t0_r4_w100"]
SS_SEEDS = [2025, 2028, 2029, 2030, 2031, 2032, 2033]

MS_SEM_SPECS = [
    ("ms_e10_ft10in", [2025]),
    ("ms_w250", [2026, 2027, 2028]),
    ("ms_w200", [2025, 2026, 2027, 2028]),
    ("ms_w175", [2025, 2026, 2027, 2028]),
    ("ms_w150", [2025, 2026, 2027, 2028]),
    ("ms_w125", [2025, 2026, 2027, 2028]),
    ("ms_w100", [2025, 2026, 2027, 2028]),
    ("ms_w10_135", [2025, 2026]),
    ("ms_w10_185", [2025, 2026]),
]

MS_SPAT_SPECS = [
    ("ms_spat_w10_70", [2025, 2026]),
    ("ms_spat_w10_105", [2025, 2026]),
    ("ms_spat_w0_150", [2025, 2026]),
    ("ms_spat_w0_225", [2025, 2026]),
]

STEPS, LR, TAU, LAM = 50, 1e-4, 0.1, 1.0

F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
R_tr = float(F_tr.norm(dim=1).mean())
mu_tr = nz(F_tr, dim=1).mean(0, keepdim=True)
alpha_tr = float(mu_tr.norm())
beta_tr = float((1.0 - alpha_tr ** 2) ** 0.5)
u_tr = mu_tr / alpha_tr

F_te = torch.from_numpy(np.load(PREP / "F_test.npy")).float()
F_te_c = nz(nz(F_te, dim=1) - nz(F_te, dim=1).mean(0, keepdim=True), dim=1)

S_te_c = None
if (PREP / "spat1024.pt").exists():
    sp_pack = torch.load(PREP / "spat1024.pt", map_location="cpu", weights_only=True)
    st = nz(sp_pack["S_test"].float(), dim=1)
    S_te_c = nz(st - st.mean(0, keepdim=True), dim=1)

tgt = torch.arange(200, device=dev)


def ivw(x):
    tot = x.sum(1, keepdim=True)
    v = (x - (tot - x) / (x.shape[1] - 1)).pow(2).mean(dim=(2, 3))
    w = 1.0 / v.clamp_min(1e-12)
    return ((w / w.sum(1, keepdim=True))[:, :, None, None] * x).sum(1)


def cen_acc(c_cpu, ref_c=None):
    target_c = F_te_c if ref_c is None else ref_c
    cu = nz(c_cpu.float(), dim=1)
    cc = nz(cu - cu.mean(0, keepdim=True), dim=1)
    return float(((cc @ target_c.T).argmax(1) == torch.arange(200)).float().mean())


def adapt_and_bag(xt_full, enc_dir, sub_id, seed_id):
    ea = json.loads((enc_dir / "metrics.json").read_text())["args"]
    t0, t1 = int(ea.get("t0", 0)), int(ea.get("t1", 250))
    T_len = t1 - t0
    xt = xt_full[..., t0:t1].contiguous()
    sd = torch.load(enc_dir / "brain_model.pt", map_location="cpu", weights_only=True)

    enc = BrainEncoder(backbone="brain_mlp", brain_channels=17, brain_sequence_length=T_len,
                       hidden_size=1024, embedding_dim=1440, dropout=0.1)
    enc.load_state_dict(sd)
    enc = enc.to(dev).eval()
    full_in = ivw(xt)
    with torch.no_grad():
        c0 = enc.backbone(full_in).float()

    perm_bag = torch.randperm(80, generator=torch.Generator().manual_seed(7 + sub_id)).to(dev)
    bag_ins = torch.cat([ivw(xt[:, perm_bag[g_ * 8:(g_ + 1) * 8]]) for g_ in range(10)], dim=0)
    with torch.no_grad():
        c_bag = enc.backbone(bag_ins).float().view(10, 200, 1024).mean(0).cpu()

    m = copy.deepcopy(enc)
    m.requires_grad_(False)
    pin = [p_ for n_, p_ in m.named_parameters() if "proj_in" in n_]
    for p_ in pin:
        p_.requires_grad_(True)
    opt = torch.optim.Adam(pin, lr=LR)
    g = torch.Generator(device=dev).manual_seed(1000 + sub_id * 37 + seed_id)
    for _ in range(STEPS):
        perm = torch.rand(200, 80, device=dev, generator=g).argsort(1)
        a = torch.gather(xt, 1, perm[:, :40, None, None].expand(-1, -1, 17, T_len))
        b = torch.gather(xt, 1, perm[:, 40:, None, None].expand(-1, -1, 17, T_len))
        za, zb = m.backbone(ivw(a)).float(), m.backbone(ivw(b)).float()
        za, zb = za - za.mean(0), zb - zb.mean(0)
        lg = nz(za, dim=1) @ nz(zb, dim=1).T / TAU
        loss = 0.5 * (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.T, tgt))
        cf = m.backbone(full_in).float()
        anc = ((cf - c0).pow(2).sum(1) / c0.pow(2).sum(1)).mean()
        (loss + LAM * anc).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    with torch.no_grad():
        c_tta_full = m.backbone(full_in).float().cpu()
        c_tta_bag = m.backbone(bag_ins).float().view(10, 200, 1024).mean(0).cpu()
        c_tta = 0.65 * c_tta_full + 0.35 * c_tta_bag
    return c_tta, c_bag, c0.norm(dim=1, keepdim=True).cpu(), sd[GK].float(), sd[BK].float()


def perp_dir(c_vec):
    cu = nz(c_vec.float(), dim=1)
    dc = nz(cu - cu.mean(0, keepdim=True), dim=1)
    return nz(dc - (dc @ u_tr.T) * u_tr, dim=1)


for s in subs:
    s2 = f"{s:02d}"
    xt_full = torch.from_numpy(np.load(PREP / f"sub-{s2}_test.npy")).float().to(dev)

    # ---------------- Track 1: Single-Subject Original (mw6tta) ----------------
    ss_tta_list, ss_bag_list = [], []
    ref_norm, ref_wg, ref_bg = None, None, None
    for ti, tg in enumerate(SS_TAGS):
        for sd_ in SS_SEEDS:
            d = RUNS / "enc" / tg / f"sub-{s2}" / f"fold-1_seed{sd_}"
            if not (d / "brain_model.pt").exists():
                continue
            c_tta, c_bag, n0, wg, bg = adapt_and_bag(xt_full, d, s, sd_ + ti * 101)
            if (tg, sd_) == ("t0_r4", 2025):
                ref_norm, ref_wg, ref_bg = n0, wg, bg
            ss_tta_list.append(c_tta)
            ss_bag_list.append(c_bag)

    assert len(ss_tta_list) > 0 and ref_wg is not None
    m_tta = torch.stack(ss_tta_list).mean(0)
    m_bag = torch.stack(ss_bag_list).mean(0)
    c_ss_tta_in = m_tta * (ref_norm / m_tta.norm(dim=1, keepdim=True))
    c_ss_bag_in = m_bag * (ref_norm / m_bag.norm(dim=1, keepdim=True))

    h_ss_path = RUNS / "ord_mw7" / f"sub-{s2}" / "head.pt"
    if not h_ss_path.exists():
        h_ss_path = RUNS / "ord" / f"sub-{s2}" / "head.pt"
    head_ss = BoundedConditionAdapter(predict_risk=False).eval()
    head_ss.load_state_dict(torch.load(h_ss_path, map_location="cpu", weights_only=True))
    with torch.no_grad():
        ord_ss_tta, _ = head_ss(c_ss_tta_in, ref_wg.expand_as(c_ss_tta_in), ref_bg.expand_as(c_ss_tta_in))
        ord_ss_bag, _ = head_ss(c_ss_bag_in, ref_wg.expand_as(c_ss_bag_in), ref_bg.expand_as(c_ss_bag_in))

    d_ss = nz(0.85 * perp_dir(ord_ss_tta) + 0.15 * perp_dir(ord_ss_bag), dim=1)
    for w in [1.00, 1.45, 1.50]:
        c_cone = R_tr * nz(alpha_tr * u_tr + (w * beta_tr) * d_ss, dim=1)
        od = RUNS / "cond" / f"cone_mw6tta_w{w:.2f}"
        od.mkdir(parents=True, exist_ok=True)
        np.save(od / f"sub-{s2}.npy", c_cone.numpy().astype(np.float32))

    # ---------------- Track 2: Full-Stack Multi-Window Multi-Seed Semantic (ms28_mwtta) ----------------
    ms_tta_list, ms_bag_list = [], []
    ms_norm, ms_wg, ms_bg = None, None, None
    for ti, (tg, seeds) in enumerate(MS_SEM_SPECS):
        for sd_ in seeds:
            d = RUNS / "enc" / tg / f"sub-{s2}" / f"fold-1_seed{sd_}"
            if not (d / "brain_model.pt").exists():
                continue
            c_tta, c_bag, n0, wg, bg = adapt_and_bag(xt_full, d, s, sd_ + ti * 313)
            if (tg, sd_) == ("ms_e10_ft10in", 2025):
                ms_norm, ms_wg, ms_bg = n0, wg, bg
            ms_tta_list.append(c_tta)
            ms_bag_list.append(c_bag)

    ms_msg = "skipped"
    if ms_tta_list and ms_wg is not None:
        mm_tta = torch.stack(ms_tta_list).mean(0)
        mm_bag = torch.stack(ms_bag_list).mean(0)
        c_ms_tta_in = mm_tta * (ms_norm / mm_tta.norm(dim=1, keepdim=True))
        c_ms_bag_in = mm_bag * (ms_norm / mm_bag.norm(dim=1, keepdim=True))

        h_ms_candidates = [
            RUNS / "ord_ms_e10_ft10in/pooled_p1/head.pt",
            RUNS / "ord_ms_e10_ft10in_pooled_p1/pooled/head.pt",
            RUNS / "ord/pooled_p1/head.pt",
        ]
        h_ms_path = next((p_ for p_ in h_ms_candidates if p_.exists()), h_ss_path)
        head_ms = BoundedConditionAdapter(predict_risk=False).eval()
        head_ms.load_state_dict(torch.load(h_ms_path, map_location="cpu", weights_only=True))
        with torch.no_grad():
            ord_ms_tta, _ = head_ms(c_ms_tta_in, ms_wg.expand_as(c_ms_tta_in), ms_bg.expand_as(c_ms_tta_in))
            ord_ms_bag, _ = head_ms(c_ms_bag_in, ms_wg.expand_as(c_ms_bag_in), ms_bg.expand_as(c_ms_bag_in))

        d_ms_only = nz(0.85 * perp_dir(ord_ms_tta) + 0.15 * perp_dir(ord_ms_bag), dim=1)
        d_full = nz(0.80 * d_ms_only + 0.20 * d_ss, dim=1)
        for w in [1.00, 1.55, 1.65, 1.75]:
            c_cone = R_tr * nz(alpha_tr * u_tr + (w * beta_tr) * d_full, dim=1)
            od = RUNS / "cond" / f"cone_ms28_mwtta_w{w:.2f}"
            od.mkdir(parents=True, exist_ok=True)
            np.save(od / f"sub-{s2}.npy", c_cone.numpy().astype(np.float32))
        ms_msg = (f"ms({len(ms_tta_list)})={cen_acc(R_tr * nz(alpha_tr * u_tr + beta_tr * d_ms_only, dim=1)):.4f}"
                  f" -> full={cen_acc(R_tr * nz(alpha_tr * u_tr + beta_tr * d_full, dim=1)):.4f}")

    # ---------------- Track 3: Full-Stack Retinotopic-Steerable-Contour Query (spat_ms_mwtta) ----------------
    sp_tta_list, sp_bag_list = [], []
    for ti, (tg, seeds) in enumerate(MS_SPAT_SPECS):
        for sd_ in seeds:
            d = RUNS / "enc" / tg / f"sub-{s2}" / f"fold-1_seed{sd_}"
            if not (d / "brain_model.pt").exists():
                continue
            c_tta, c_bag, _, _, _ = adapt_and_bag(xt_full, d, s, sd_ + ti * 521)
            sp_tta_list.append(nz(c_tta - c_tta.mean(0, keepdim=True), dim=1))
            sp_bag_list.append(nz(c_bag - c_bag.mean(0, keepdim=True), dim=1))

    sp_msg = "spat=none"
    if sp_tta_list:
        q_sp_tta = nz(torch.stack(sp_tta_list).mean(0), dim=1)
        q_sp_bag = nz(torch.stack(sp_bag_list).mean(0), dim=1)
        q_spat = nz(0.85 * q_sp_tta + 0.15 * q_sp_bag, dim=1)
        q_spat = nz(q_spat - q_spat.mean(0, keepdim=True), dim=1)
        od = RUNS / "cond/spat_ms_mwtta"
        od.mkdir(parents=True, exist_ok=True)
        np.save(od / f"sub-{s2}.npy", q_spat.numpy().astype(np.float32))
        sp_t1 = cen_acc(q_spat, S_te_c) if S_te_c is not None else float("nan")
        sp_msg = f"spat({len(sp_tta_list)})={sp_t1:.4f}"

    print(f"sub-{s2}: ss({len(ss_tta_list)})={cen_acc(R_tr * nz(alpha_tr * u_tr + beta_tr * d_ss, dim=1)):.4f} | {ms_msg} | {sp_msg}", flush=True)
