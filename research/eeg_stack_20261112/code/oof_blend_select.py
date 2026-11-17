"""5-fold concept-disjoint OOF blend weight calibration for O8 and B5 (oof_blend_select.py).

Strictly uses ONLY 5-fold OOF training-set predictions (16,540 images) and F_train.npy:
  - Zero test images or test embeddings are opened or touched.
  - Selects:
      1. alpha_o8 : weight on d_o8_flex vs (1 - alpha_o8) on d_mw7 (strictly single-subject O8)
      2. w_b5     : (w_b5_flex, w_ms_old, w_ss_o8) on simplex for full-stack B5
  - Saves runs/cond/o8_b5_oof_weights.json for aw_tta.py.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import PREP, ROOT, RUNS, concept_folds, write_json
from ordh import ENS_SETS, last_norm_keys

_eeg_dir = Path(__file__).resolve().parents[2] / "eeg"
if not _eeg_dir.exists():
    _eeg_dir = ROOT / "code/EEG-Diffusion/research/eeg"
sys.path.insert(0, str(_eeg_dir))
from brainhive_bounded_component_v1 import BoundedConditionAdapter  # noqa: E402

nz = F.normalize
dev = "cuda" if torch.cuda.is_available() else "cpu"

F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float().to(dev)
F_u = nz(F_tr, dim=1)
mu_tr = F_u.mean(0, keepdim=True)
u_tr = nz(mu_tr, dim=1)
F_perp = nz(F_u - (F_u @ u_tr.T) * u_tr, dim=1)

concept = torch.from_numpy(np.load(PREP / "train_concept.npy")).long()
img_fold = torch.from_numpy(concept_folds(1654, 5))[concept]


def perp_dir(c_vec):
    cu = nz(c_vec.float(), dim=1)
    dc = nz(cu - cu.mean(0, keepdim=True), dim=1)
    return nz(dc - (dc @ u_tr.T) * u_tr, dim=1)


def oof_200way_acc(d_pred):
    accs = []
    t200 = torch.arange(200, device=dev)
    for k in range(5):
        ho = torch.nonzero(img_fold == k).squeeze(1).to(dev)
        for ch in range(0, len(ho) - 199, 200):
            sl = ho[ch:ch + 200]
            accs.append(float(((d_pred[sl] @ F_perp[sl].T).argmax(1) == t200).float().mean()))
    return float(np.mean(accs))


def load_oof_ord_dir(sub, ens_key, head_path):
    tags, seeds = ENS_SETS[ens_key]
    ref_tg, ref_sd = tags[0], seeds[0]
    C_out = torch.zeros((16540, 1024), dtype=torch.float32)
    head = BoundedConditionAdapter(predict_risk=False).eval()
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))

    for k in range(5):
        if ens_key == "mw7":
            cand = [RUNS / "enc" / tg0 / f"sub-{sub:02d}" / f"fold{k}_seed2025" for tg0 in ("t0_r4", "t0_r4f")]
            d_ref = next(p_ for p_ in cand if (p_ / "brain_model.pt").exists())
        else:
            d_ref = RUNS / "enc" / ref_tg / f"sub-{sub:02d}" / f"fold{k}_seed{ref_sd}"
        idx = torch.from_numpy(np.load(d_ref / "oof_index.npy"))
        c0 = torch.from_numpy(np.load(d_ref / "cond_oof_r4.npy")).float()
        sd = torch.load(d_ref / "brain_model.pt", map_location="cpu", weights_only=True)
        gk, bk = last_norm_keys(sd)
        w, b = sd[gk].float(), sd[bk].float()
        cs = []
        for tg in tags:
            for sd_ in seeds:
                if (tg, sd_) == (ref_tg, ref_sd):
                    cs.append(c0)
                else:
                    e = RUNS / "enc" / tg / f"sub-{sub:02d}" / f"fold{k}_seed{sd_}"
                    cs.append(torch.from_numpy(np.load(e / "cond_oof_r4.npy")).float())
        m = torch.stack(cs).mean(0)
        cin = m * (c0.norm(dim=1, keepdim=True) / m.norm(dim=1, keepdim=True))
        with torch.no_grad():
            cc, _ = head(cin, w.expand_as(cin), b.expand_as(cin))
        C_out[idx] = cc
    return perp_dir(C_out.to(dev))


def load_oof_single_tag_ord_dir(sub, tag, head_path):
    C_out = torch.zeros((16540, 1024), dtype=torch.float32)
    head = BoundedConditionAdapter(predict_risk=False).eval()
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
    for k in range(5):
        d_ref = RUNS / "enc" / tag / f"sub-{sub:02d}" / f"fold{k}_seed2025"
        idx = torch.from_numpy(np.load(d_ref / "oof_index.npy"))
        cin = torch.from_numpy(np.load(d_ref / "cond_oof_r4.npy")).float()
        sd = torch.load(d_ref / "brain_model.pt", map_location="cpu", weights_only=True)
        gk, bk = last_norm_keys(sd)
        w, b = sd[gk].float(), sd[bk].float()
        with torch.no_grad():
            cc, _ = head(cin, w.expand_as(cin), b.expand_as(cin))
        C_out[idx] = cc
    return perp_dir(C_out.to(dev))


subs = list(range(1, 11))
d_mw7_all, d_o8_all, d_b5f_all, d_mso_all = {}, {}, {}, {}
for s in subs:
    d_mw7_all[s] = load_oof_ord_dir(s, "mw7", RUNS / f"ord_mw7/sub-{s:02d}/head.pt")
    d_o8_all[s] = load_oof_ord_dir(s, "o8_flex", RUNS / f"ord_o8_flex/sub-{s:02d}/head.pt")
    d_b5f_all[s] = load_oof_ord_dir(s, "b5_flex", RUNS / "ord_b5_flex/pooled_p1/head.pt")
    d_mso_all[s] = load_oof_single_tag_ord_dir(s, "ms_e10_ft10in", RUNS / "ord_ms_e10_ft10in/pooled_p1/head.pt")

print("=== 10-Subject 5-Fold OOF Single-Subject O8 Blend Sweep (train only) ===", flush=True)
best_a, best_sc = 0.75, -1.0
for a in [0.00, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.00]:
    accs, coss = [], []
    for s in subs:
        d = nz(a * d_o8_all[s] + (1.0 - a) * d_mw7_all[s], dim=1)
        accs.append(oof_200way_acc(d))
        coss.append(float((d * F_perp).sum(1).mean()))
    sc = float(np.mean(accs)) + 0.25 * float(np.mean(coss))
    print(f"  alpha_o8={a:.2f} | OOF_200way={np.mean(accs):.4f} OOF_cos_perp={np.mean(coss):.4f}", flush=True)
    if sc > best_sc:
        best_sc, best_a = sc, a

d_ss_best = {s: nz(best_a * d_o8_all[s] + (1.0 - best_a) * d_mw7_all[s], dim=1) for s in subs}

print("=== 10-Subject 5-Fold OOF Full-Stack B5 Blend Sweep (train only) ===", flush=True)
best_w, best_b5_sc = (0.60, 0.20, 0.20), -1.0
for wf in [0.45, 0.55, 0.60, 0.65, 0.70, 0.75]:
    for wm in [0.10, 0.15, 0.20, 0.25, 0.30]:
        ws = round(1.0 - wf - wm, 4)
        if ws < 0.05 or ws > 0.35:
            continue
        accs, coss = [], []
        for s in subs:
            d = nz(wf * d_b5f_all[s] + wm * d_mso_all[s] + ws * d_ss_best[s], dim=1)
            accs.append(oof_200way_acc(d))
            coss.append(float((d * F_perp).sum(1).mean()))
        sc = float(np.mean(accs)) + 0.25 * float(np.mean(coss))
        print(f"  w_b5=({wf:.2f},{wm:.2f},{ws:.2f}) | OOF_200way={np.mean(accs):.4f} OOF_cos_perp={np.mean(coss):.4f}", flush=True)
        if sc > best_b5_sc:
            best_b5_sc, best_w = sc, (wf, wm, ws)

out_json = RUNS / "cond/o8_b5_oof_weights.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
res = dict(alpha_o8=best_a, w_b5_flex=best_w[0], w_ms_old=best_w[1], w_ss_o8=best_w[2])
write_json(out_json, res)
print("Selected strictly on 10-subject 5-fold OOF (train):", res, flush=True)
