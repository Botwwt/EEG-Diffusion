"""All-window split-half TTA + repeat bagging + BoundedConditionAdapter + F_train Cone Restoration/Amplification (aw_tta.py).

Produces strictly separated condition sets across subjects:
  1. Single-Subject Track (O7a / O7 / O8):
     - mw6tta (42 single-subject BrainMLP encoders + per-subject ord_mw7) -> cone_mw6tta_w{1.00, 1.45, 1.50}
     - O8 (36 single-subject FlexBrainEncoder models + per-subject ord_o8_flex, blended with mw6tta using the
       5-fold OOF weight from o8_b5_oof_weights.json, strictly w = 1.00 pure F_train cone restoration with
       zero directional amplification and zero cross-subject parameters) -> cone_o8_w1.00
     - O8 single-subject native 512px contour query (12 single-subject spat_ms512 encoders) -> spat_ss_o8

  2. Full-Stack Semantic Condition (B4-1 / B4 / B5-1a / B5a / B5):
     - B4: 28 joint BrainMLP encoders + ord_ms_e10_ft10in/pooled_p1 + 0.20 mw6tta -> cone_ms28_mwtta_w{1.00, 1.55, 1.65, 1.75}
     - B5: 28 joint FlexBrainEncoder models + ord_b5_flex/pooled_p1, blended with ms28 and O8 using the
       5-fold OOF weights from o8_b5_oof_weights.json -> cone_b5_w{1.00, 1.75}

  3. Full-Stack Auxiliary Condition Queries (B4 / B5):
     - B4 4-orientation 16x16 contour query -> spat_ms_mwtta
     - B5 native 512px multi-scale 4-orientation contour query -> spat_ms512_b5
     - B5 Brain-HIVE LAION-H14 semantic query -> h14_ms_b5
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import HIVE, PREP, ROOT, RUNS
from train_flex import FlexBrainEncoder

sys.path.insert(0, str(HIVE))
_eeg_dir = Path(__file__).resolve().parents[2] / "eeg"  # research/eeg of this repository
if not _eeg_dir.exists():
    _eeg_dir = ROOT / "code/EEG-Diffusion/research/eeg"
sys.path.insert(0, str(_eeg_dir))
from brainhive_bounded_component_v1 import BoundedConditionAdapter  # noqa: E402
from main.models_brain import BrainEncoder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subs", help="Comma-separated subject ints or 'all'")
ap.add_argument("--tracks", choices=["all", "w5", "legacy"], default="w5",
                help="'w5' computes O8/B5 (reusing cone_mw6tta_w1.00 if present); 'all' recomputes legacy O7/B4 too")
args = ap.parse_args()

subs = list(range(1, 11)) if args.subs == "all" else [int(x) for x in args.subs.split(",")]
dev = "cuda" if torch.cuda.is_available() else "cpu"
nz = F.normalize

SS_TAGS = ["t0_r4", "t0_r4_w200", "t0_r4_w175", "t0_r4_w150", "t0_r4_w125", "t0_r4_w100"]
SS_SEEDS = [2025, 2028, 2029, 2030, 2031, 2032, 2033]

O8_FLEX_SEM_SPECS = [
    ("o8_pyr2_w250", [2025, 2026, 2027]),
    ("o8_pyr2_w200", [2025, 2026, 2027]),
    ("o8_pyr2_w185", [2025, 2026, 2027]),
    ("o8_pyr2_w165", [2025, 2026, 2027]),
    ("o8_pyr2_w150", [2025, 2026, 2027]),
    ("o8_pyr2_w125", [2025, 2026, 2027]),
    ("o8_pyr2_w115", [2025, 2026, 2027]),
    ("o8_pyr2_w10_160", [2025, 2026, 2027]),
    ("o8_mlp2_w250", [2025, 2026]),
    ("o8_mlp2_w185", [2025, 2026]),
    ("o8_mlp2_w150", [2025, 2026]),
    ("o8_mlp2_w115", [2025, 2026]),
    ("o8_mlp1_w200", [2025, 2026]),
    ("o8_mlp1_w175", [2025, 2026]),
]

O8_SS_SPAT_SPECS = [
    ("o8_spat_w10_65", [2025, 2026, 2027]),
    ("o8_spat_w10_95", [2025, 2026, 2027]),
    ("o8_spat_w0_140", [2025, 2026, 2027]),
    ("o8_spat_w0_210", [2025, 2026, 2027]),
]

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

B5_MS_SEM_SPECS = [
    ("msf_pyr2_w250", [2025, 2026, 2027]),
    ("msf_pyr2_w200", [2025, 2026, 2027]),
    ("msf_pyr2_w185", [2025, 2026, 2027]),
    ("msf_pyr2_w165", [2025, 2026]),
    ("msf_pyr2_w145", [2025, 2026, 2027]),
    ("msf_pyr2_w120", [2025, 2026]),
    ("msf_pyr2_w10_140", [2025, 2026]),
    ("msf_pyr2_w10_185", [2025, 2026]),
    ("msf_mlp2_w250", [2025, 2026]),
    ("msf_mlp2_w200", [2025, 2026]),
    ("msf_mlp2_w165", [2025, 2026]),
    ("msf_mlp2_w130", [2025, 2026]),
]

MS_SPAT_SPECS = [
    ("ms_spat_w10_70", [2025, 2026]),
    ("ms_spat_w10_105", [2025, 2026]),
    ("ms_spat_w0_150", [2025, 2026]),
    ("ms_spat_w0_225", [2025, 2026]),
]

B5_MS_SPAT_SPECS = [
    ("msf_spat_w10_65", [2025, 2026, 2027]),
    ("msf_spat_w10_95", [2025, 2026, 2027]),
    ("msf_spat_w0_150", [2025, 2026, 2027]),
    ("msf_spat_w0_215", [2025, 2026, 2027]),
]

B5_MS_H14_SPECS = [
    ("msf_h14_w225", [2025, 2026]),
    ("msf_h14_w185", [2025, 2026]),
    ("msf_h14_w150", [2025, 2026]),
    ("msf_h14_w10_170", [2025, 2026]),
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
tgt = torch.arange(200, device=dev)

oof_w_path = RUNS / "cond/o8_b5_oof_weights.json"
if oof_w_path.exists():
    oof_w = json.loads(oof_w_path.read_text())
else:
    oof_w = dict(alpha_o8=0.75, w_b5_flex=0.60, w_ms_old=0.20, w_ss_o8=0.20)


def last_norm_keys(sd):
    idxs = [int(k.split(".")[3]) for k in sd if k.startswith("backbone.block.layers.") and k.endswith(".norm.weight")]
    li = max(idxs)
    return f"backbone.block.layers.{li}.norm.weight", f"backbone.block.layers.{li}.norm.bias"


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


def perp_dir(c_vec):
    cu = nz(c_vec.float(), dim=1)
    dc = nz(cu - cu.mean(0, keepdim=True), dim=1)
    return nz(dc - (dc @ u_tr.T) * u_tr, dim=1)


def adapt_and_bag(xt_full, enc_dir, sub_id, seed_id):
    ea = json.loads((enc_dir / "metrics.json").read_text())["args"]
    t0, t1 = int(ea.get("t0", 0)), int(ea.get("t1", 250))
    T_len = t1 - t0
    xt = xt_full[..., t0:t1].contiguous()
    sd = torch.load(enc_dir / "brain_model.pt", map_location="cpu", weights_only=True)

    if "arch" in ea:
        enc = FlexBrainEncoder(T_len, arch=ea["arch"], hidden_size=1024,
                               num_layers=int(ea.get("num_layers", 1)), dropout=float(ea.get("dropout", 0.12)))
    else:
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
    gk, bk = last_norm_keys(sd)
    return c_tta, c_bag, c0.norm(dim=1, keepdim=True).cpu(), sd[gk].float(), sd[bk].float()


def run_ord_ensemble(xt_full, s, specs, ref_key, head_path, seed_mul):
    tta_l, bag_l = [], []
    ref_n, ref_w, ref_b = None, None, None
    for ti, (tg, seeds) in enumerate(specs):
        for sd_ in seeds:
            d = RUNS / "enc" / tg / f"sub-{s:02d}" / f"fold-1_seed{sd_}"
            if not (d / "brain_model.pt").exists():
                continue
            c_tta, c_bag, n0, wg, bg = adapt_and_bag(xt_full, d, s, sd_ + ti * seed_mul)
            if (tg, sd_) == ref_key or ref_w is None:
                if (tg, sd_) == ref_key or ref_w is None:
                    ref_n, ref_w, ref_b = n0, wg, bg
            tta_l.append(c_tta)
            bag_l.append(c_bag)
    if not tta_l or ref_w is None:
        return None, 0
    m_tta = torch.stack(tta_l).mean(0)
    m_bag = torch.stack(bag_l).mean(0)
    c_tta_in = m_tta * (ref_n / m_tta.norm(dim=1, keepdim=True))
    c_bag_in = m_bag * (ref_n / m_bag.norm(dim=1, keepdim=True))
    head = BoundedConditionAdapter(predict_risk=False).eval()
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
    with torch.no_grad():
        o_tta, _ = head(c_tta_in, ref_w.expand_as(c_tta_in), ref_b.expand_as(c_tta_in))
        o_bag, _ = head(c_bag_in, ref_w.expand_as(c_bag_in), ref_b.expand_as(c_bag_in))
    return nz(0.85 * perp_dir(o_tta) + 0.15 * perp_dir(o_bag), dim=1), len(tta_l)


def run_centered_query(xt_full, s, specs, seed_mul):
    tta_l, bag_l = [], []
    for ti, (tg, seeds) in enumerate(specs):
        for sd_ in seeds:
            d = RUNS / "enc" / tg / f"sub-{s:02d}" / f"fold-1_seed{sd_}"
            if not (d / "brain_model.pt").exists():
                continue
            c_tta, c_bag, _, _, _ = adapt_and_bag(xt_full, d, s, sd_ + ti * seed_mul)
            tta_l.append(nz(nz(c_tta, dim=1) - nz(c_tta, dim=1).mean(0, keepdim=True), dim=1))
            bag_l.append(nz(nz(c_bag, dim=1) - nz(c_bag, dim=1).mean(0, keepdim=True), dim=1))
    if not tta_l:
        return None, 0
    q_tta = nz(torch.stack(tta_l).mean(0), dim=1)
    q_bag = nz(torch.stack(bag_l).mean(0), dim=1)
    q = nz(0.85 * q_tta + 0.15 * q_bag, dim=1)
    return nz(q - q.mean(0, keepdim=True), dim=1), len(tta_l)


for s in subs:
    s2 = f"{s:02d}"
    xt_full = torch.from_numpy(np.load(PREP / f"sub-{s2}_test.npy")).float().to(dev)

    # 1a. Legacy single-subject mw6tta (reuse exact saved cone_mw6tta_w1.00 unless --tracks all/legacy)
    mw6_file = RUNS / "cond/cone_mw6tta_w1.00" / f"sub-{s2}.npy"
    h_ss_path = RUNS / "ord_mw7" / f"sub-{s2}" / "head.pt"
    if not h_ss_path.exists():
        h_ss_path = RUNS / "ord" / f"sub-{s2}" / "head.pt"
    if args.tracks in ("all", "legacy") or not mw6_file.exists():
        ss_specs = [(tg, SS_SEEDS) for tg in SS_TAGS]
        d_ss, n_ss = run_ord_ensemble(xt_full, s, ss_specs, ("t0_r4", 2025), h_ss_path, 101)
        for w in [1.00, 1.45, 1.50]:
            c_cone = R_tr * nz(alpha_tr * u_tr + (w * beta_tr) * d_ss, dim=1)
            od = RUNS / "cond" / f"cone_mw6tta_w{w:.2f}"
            od.mkdir(parents=True, exist_ok=True)
            np.save(od / f"sub-{s2}.npy", c_cone.numpy().astype(np.float32))
    else:
        d_ss = perp_dir(torch.from_numpy(np.load(mw6_file)).float())
        n_ss = 42

    # 1b. Legacy B4 ms28 + spat_ms_mwtta
    h_ms_old = next((p_ for p_ in [
        RUNS / "ord_ms_e10_ft10in/pooled_p1/head.pt",
        RUNS / "ord_ms_e10_ft10in_pooled_p1/pooled/head.pt",
        RUNS / "ord/pooled_p1/head.pt",
    ] if p_.exists()), h_ss_path)
    d_ms_old, n_ms_old = run_ord_ensemble(xt_full, s, MS_SEM_SPECS, ("ms_e10_ft10in", 2025), h_ms_old, 313)
    if args.tracks in ("all", "legacy") and d_ms_old is not None:
        d_b4 = nz(0.80 * d_ms_old + 0.20 * d_ss, dim=1)
        for w in [1.00, 1.55, 1.65, 1.75]:
            c_cone = R_tr * nz(alpha_tr * u_tr + (w * beta_tr) * d_b4, dim=1)
            od = RUNS / "cond" / f"cone_ms28_mwtta_w{w:.2f}"
            od.mkdir(parents=True, exist_ok=True)
            np.save(od / f"sub-{s2}.npy", c_cone.numpy().astype(np.float32))
        q_sp_old, _ = run_centered_query(xt_full, s, MS_SPAT_SPECS, 521)
        if q_sp_old is not None:
            od = RUNS / "cond/spat_ms_mwtta"
            od.mkdir(parents=True, exist_ok=True)
            np.save(od / f"sub-{s2}.npy", q_sp_old.numpy().astype(np.float32))

    if args.tracks == "legacy":
        print(f"sub-{s2} [legacy]: ss={cen_acc(R_tr * nz(alpha_tr * u_tr + beta_tr * d_ss, dim=1)):.4f}", flush=True)
        continue

    # 2a. Strict Single-Subject O8 Semantic Condition (w = 1.00 ONLY, zero cross-subject data, zero amp)
    h_o8 = RUNS / "ord_o8_flex" / f"sub-{s2}" / "head.pt"
    d_o8_flex, n_o8f = run_ord_ensemble(xt_full, s, O8_FLEX_SEM_SPECS, ("o8_pyr2_w250", 2025), h_o8, 137)
    assert d_o8_flex is not None, f"Missing O8 Flex encoders for sub-{s2}"
    a_o8 = float(oof_w["alpha_o8"])
    d_o8 = nz(a_o8 * d_o8_flex + (1.0 - a_o8) * d_ss, dim=1)
    c_o8 = R_tr * nz(alpha_tr * u_tr + beta_tr * d_o8, dim=1)
    od_o8 = RUNS / "cond/cone_o8_w1.00"
    od_o8.mkdir(parents=True, exist_ok=True)
    np.save(od_o8 / f"sub-{s2}.npy", c_o8.numpy().astype(np.float32))

    # 2b. Strict Single-Subject O8 Native 512px Contour Query (spat_ss_o8)
    q_sp_ss, n_spss = run_centered_query(xt_full, s, O8_SS_SPAT_SPECS, 211)
    if q_sp_ss is not None:
        od_spss = RUNS / "cond/spat_ss_o8"
        od_spss.mkdir(parents=True, exist_ok=True)
        np.save(od_spss / f"sub-{s2}.npy", q_sp_ss.numpy().astype(np.float32))

    # 3a. Full-Stack B5 Semantic Condition
    h_b5 = RUNS / "ord_b5_flex/pooled_p1/head.pt"
    d_b5_flex, n_b5f = run_ord_ensemble(xt_full, s, B5_MS_SEM_SPECS, ("msf_pyr2_w250", 2025), h_b5, 409)
    b5_msg = "b5=none"
    if d_b5_flex is not None:
        wf, wm, ws = float(oof_w["w_b5_flex"]), float(oof_w["w_ms_old"]), float(oof_w["w_ss_o8"])
        if d_ms_old is not None:
            d_b5 = nz(wf * d_b5_flex + wm * d_ms_old + ws * d_o8, dim=1)
        else:
            d_b5 = nz((wf + wm) * d_b5_flex + ws * d_o8, dim=1)
        for w in [1.00, 1.75]:
            c_b5 = R_tr * nz(alpha_tr * u_tr + (w * beta_tr) * d_b5, dim=1)
            od_b5 = RUNS / "cond" / f"cone_b5_w{w:.2f}"
            od_b5.mkdir(parents=True, exist_ok=True)
            np.save(od_b5 / f"sub-{s2}.npy", c_b5.numpy().astype(np.float32))
        b5_msg = f"b5({n_b5f}+{n_ms_old})={cen_acc(R_tr * nz(alpha_tr * u_tr + beta_tr * d_b5, dim=1)):.4f}"

    # 3b. Full-Stack B5 Native 512px Multi-Scale Contour Query + LAION-H14 Verifier Query
    q_sp_ms, n_spms = run_centered_query(xt_full, s, B5_MS_SPAT_SPECS, 607)
    if q_sp_ms is not None:
        q_sp_b5 = nz(0.85 * q_sp_ms + 0.15 * q_sp_ss, dim=1) if q_sp_ss is not None else q_sp_ms
        q_sp_b5 = nz(q_sp_b5 - q_sp_b5.mean(0, keepdim=True), dim=1)
        od_spb5 = RUNS / "cond/spat_ms512_b5"
        od_spb5.mkdir(parents=True, exist_ok=True)
        np.save(od_spb5 / f"sub-{s2}.npy", q_sp_b5.numpy().astype(np.float32))

    q_h14, n_h14 = run_centered_query(xt_full, s, B5_MS_H14_SPECS, 733)
    if q_h14 is not None:
        od_h14 = RUNS / "cond/h14_ms_b5"
        od_h14.mkdir(parents=True, exist_ok=True)
        np.save(od_h14 / f"sub-{s2}.npy", q_h14.numpy().astype(np.float32))

    print(f"sub-{s2}: ss_old={cen_acc(R_tr * nz(alpha_tr * u_tr + beta_tr * d_ss, dim=1)):.4f} "
          f"-> o8_flex({n_o8f})={cen_acc(R_tr * nz(alpha_tr * u_tr + beta_tr * d_o8_flex, dim=1)):.4f} "
          f"-> o8_blend={cen_acc(c_o8):.4f} | {b5_msg} | sp_ss={n_spss} sp_ms={n_spms} h14={n_h14}", flush=True)
