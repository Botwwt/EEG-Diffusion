"""O14: Tri-Architecture Cross-Brain Contrastive Synthesis & Continuous OpenCLIP-bigG/14 Awakening
with 100% STRICTLY SINGLE-SUBJECT TEST INFERENCE on THINGS-EEG2 (10 subjects, R=80, 4 UNet NFE/img).

Grounded strictly on D_dev200 (tr14 folds 1..4 -> fold 0, sub-{02,05,08,09}; ZERO official test access):
  1. Winner 1 (probe_xkv_dual -> +0.72%pp CLIP_t1, +1.50%pp F_200w, +0.0006 SSIM over O13_ref):
       Dedicated two-trunk Cross-Subject KVEBPyrEncoder (O14_XKV_Ffull + O14_XKV_Fsem, both trained
       with L_xbrain = 0.25 across 165,400 pairs) fused in a Tri-Architecture visual cone:
         d_O14 = normalize(0.40 d_XKV + 0.40 d_O13_xbrain + 0.20 d_O10)
       uniting two distinct cross-subject L_xbrain architectures with single-subject O10 diversity.
  2. Winner 2 (probe_bigg -> +0.84%pp on r40, 4/4 subs beat noBigG & 4/4 beat ctrl_shuf by +1.09%pp;
       single-image r4: +0.31%pp Top-1, +0.37%pp CLIP 2w, +0.47%pp Incep, 4/4 beat ctrl_shuf):
       Continuous Cross-Subject L_xbrain OpenCLIP-ViT-bigG/14 text awakening (text_encoder_2 ONLY;
       OpenAI CLIP ViT-L/14 text_encoder_1 remains strictly empty ""). Combines direct cross-brain
       EEG -> (ppe, seq16) encoder (o14_xbrain_bigg) 50/50 with continuous ridge (O13 H14/B32 -> bigG)
       fitted strictly on the 1,654 training concepts (zero discrete medoid collapse) at alpha = 0.35.
  3. Falsified & dropped on D_dev200 (NOT included):
       - Cross-sub spat1024 auxiliary head (probe_spat: -0.22%pp CLIP_t1, -0.0026 SSIM).
       - Shared-trunk auxiliary fsem head (probe_xkv: -0.12%pp CLIP_t1).
       - Unbounded 6-layer Rectified-Flow Prior in 1024-D F (probe_flow_prior: smooths identity, -3%pp F_200w).
  4. Strict Official Parity & Zero-Leakage Guardrail:
       - 100% STRICTLY SINGLE-SUBJECT TEST INFERENCE for O14_noBigG, O14, O14_ctrl_shuf, and O14_sel4.
       - 0 DDPM heads, pure F_train[ok] cone (w = 1.00), unchanged lat512_start.py (--mode_v v2),
         SDXL-Turbo smap c=0.625, low_keep=(8, 0.22, 0.04), 4 UNet NFE/img (nsel=1), strictly independent
         K=4 draws, zero blur, unchanged metrics.py, ZERO OpenAI CLIP ViT-L/14 weights anywhere in pipeline.
       - Explicitly separated higher-budget row O14_sel4 (16 NFE, nsel=4) ranks 4 independent draws using
         ONLY Brain-HIVE's own FEmbedder (F + LAION OpenCLIP-H14 + OpenCLIP-B32; ZERO OpenAI ViT-L/14).
       - Explicitly separated multi-brain diagnostic ceiling O14_diag_cohort10 (10-brain test avg, zero labels).
"""
import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import EMB_KEYS, HIVE, PREP, PRIOR, RUNS, write_json
from deep_enc import TargetSNRGate, sample_sub_repeats
from o10_factorized import _cone_params, _perp_dir
from o11_tri_stream_bigg import (
    OfficialTurboRunner,
    OnConeKVProjector,
    _hybrid_infonce,
    _simple_infonce,
    _to_cone,
    build_bigG_embeds,
    build_or_load_bigG_train1654,
)
from o13_xbrain_shared_trunk import O13_ENC, _xbrain_cone_nce, _xbrain_nce
from train_flex import ivw_mean, pyramid_feats, shift_batch

sys.path.insert(0, str(HIVE))
from main.layers import ResNet1DBlock  # noqa: E402
from main.models_adapter import FusionEncoderModel  # noqa: E402

nz = F.normalize
CODE_DIR = Path(__file__).resolve().parent
ALL_SUBS = tuple(range(1, 11))
SEEDS_5 = (2025, 2026, 2027, 2028, 2029)

O14_ROOT = RUNS / "o14_official"
O14_ENC = O14_ROOT / "enc"
O14_COND = O14_ROOT / "cond"
O14_EVAL = O14_ROOT / "eval"
O14_AUDIT = O14_ROOT / "audit_cache"


def _find_script(name: str) -> Path:
    candidates = [CODE_DIR] + [Path(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    for d in candidates:
        p = d / name
        if p.exists():
            return p
    return CODE_DIR / name


class CrossSubSingleTargetKVPyr(nn.Module):
    """Cross-subject shared-trunk KVEBPyrEncoder (nested 0-1000 / 0-700 / 0-480 ms -> shared ResNet1D -> 1024-D)."""

    def __init__(self, n_subs: int = 10, t_len: int = 250, c_in: int = 17, hidden_size: int = 1024,
                 num_layers: int = 3, dropout: float = 0.20, in_drop: float = 0.15, horizons=(250, 175, 120)):
        super().__init__()
        self.n_subs = n_subs
        self.horizons = horizons
        dims = [pyramid_feats(torch.zeros(1, c_in, t_h)).shape[1] for t_h in horizons]
        self.ebs = nn.ModuleList([TargetSNRGate(c_in=c_in, t_len=t_len) for _ in range(n_subs)])
        self.in_drop = nn.Dropout(in_drop)
        self.sub_stems = nn.ModuleList([
            nn.ModuleList([nn.Linear(d, hidden_size) for d in dims]) for _ in range(n_subs)
        ])
        self.stem_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in dims])
        self.horizon_logits = nn.Parameter(torch.zeros(len(horizons)))
        self.sub_emb = nn.Embedding(n_subs, hidden_size)
        nn.init.zeros_(self.sub_emb.weight)
        self.block = ResNet1DBlock(hidden_size=hidden_size, intermediate_size=hidden_size * 2, num_layers=num_layers, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1024))

    def forward_sub(self, x: torch.Tensor, sid: int, r_eff=4.0):
        xg = self.ebs[sid](x, r_eff=r_eff)
        w = F.softmax(self.horizon_logits, dim=0)
        h = sum(
            w[i] * self.stem_norms[i](self.sub_stems[sid][i](self.in_drop(pyramid_feats(xg[..., :t_h]))))
            for i, t_h in enumerate(self.horizons)
        ) + self.sub_emb.weight[sid]
        return self.head(self.block(h))


class CrossSubBigGEncoder(nn.Module):
    """Cross-subject shared-trunk encoder predicting continuous OpenCLIP-ViT-bigG/14 (ppe 1280-D + seq16 128-D)."""

    def __init__(self, n_subs: int = 10, t_len: int = 250, c_in: int = 17, hidden_size: int = 1024,
                 num_layers: int = 3, dropout: float = 0.20, in_drop: float = 0.15, horizons=(115, 160, 225)):
        super().__init__()
        self.n_subs = n_subs
        self.horizons = horizons
        dims = [pyramid_feats(torch.zeros(1, c_in, t_h)).shape[1] for t_h in horizons]
        self.ebs = nn.ModuleList([TargetSNRGate(c_in=c_in, t_len=t_len) for _ in range(n_subs)])
        self.in_drop = nn.Dropout(in_drop)
        self.sub_stems = nn.ModuleList([
            nn.ModuleList([nn.Linear(d, hidden_size) for d in dims]) for _ in range(n_subs)
        ])
        self.stem_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in dims])
        self.horizon_logits = nn.Parameter(torch.zeros(len(horizons)))
        self.sub_emb = nn.Embedding(n_subs, hidden_size)
        nn.init.zeros_(self.sub_emb.weight)
        self.block = ResNet1DBlock(hidden_size=hidden_size, intermediate_size=hidden_size * 2, num_layers=num_layers, dropout=dropout)
        self.head_ppe = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1280))
        self.head_seq = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 128))

    def forward_sub(self, x: torch.Tensor, sid: int, r_eff=4.0):
        xg = self.ebs[sid](x, r_eff=r_eff)
        hs = [
            self.stem_norms[i](self.sub_stems[sid][i](self.in_drop(pyramid_feats(xg[..., :t_h])))) + self.sub_emb.weight[sid]
            for i, t_h in enumerate(self.horizons)
        ]
        cs = self.block(torch.cat(hs, dim=0)).chunk(len(self.horizons), dim=0)
        w_h = F.softmax(self.horizon_logits, dim=0)
        b_af = self.block.layers[-1].norm.bias.to(cs[0].dtype)
        d0 = cs[0] - b_af
        d_mix = sum(w_h[i] * (cs[i] - b_af) for i in range(len(cs)))
        scale = d0.norm(dim=-1, keepdim=True) / d_mix.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        c_ens = b_af + d_mix * scale
        return self.head_ppe(c_ens), self.head_seq(c_ens)


def train_one_o14_seed(model_name: str, seed: int, bigG_pack: dict, dev: str = "cuda:0", epochs: int = 140):
    """Train one full-train (16,540 imgs x 10 subs = 165,400 pairs, ZERO test data) cross-subject L_xbrain seed."""
    assert model_name in ("o14_xkv_ffull", "o14_xkv_fsem", "o14_xbrain_bigg")
    out_d = O14_ENC / model_name / f"s{seed}"
    if (out_d / "brain_model_fp16.pt").exists() and (out_d / "test_0tta_all10.pt").exists():
        print(f"[CACHED_TRAIN] {model_name} seed={seed}", flush=True)
        return
    out_d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    concept = torch.from_numpy(np.load(PREP / "train_concept.npy")).long().to(dev)
    n_tr = len(concept)
    x_tr_all = [torch.from_numpy(np.load(PREP / f"sub-{s:02d}_train.npy")).half() for s in ALL_SUBS]

    is_bigg = (model_name == "o14_xbrain_bigg")
    use_fnm = (model_name == "o14_xkv_fsem")
    kv_p = None
    target_F = P_u = Z_u = None

    if is_bigg:
        emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
        X_hb = torch.cat([nz(emb_tr[EMB_KEYS[0]].float(), dim=1), nz(emb_tr[EMB_KEYS[1]].float(), dim=1)], dim=1)
        P_all = X_hb @ bigG_pack["W_ppe"].float()
        Z_all = X_hb @ bigG_pack["W_seq"].float()
        P_u = nz(P_all - P_all.mean(0, keepdim=True), dim=1).to(dev)
        Z_u = nz(Z_all, dim=1).to(dev)
        del emb_tr, X_hb, P_all, Z_all
        model = CrossSubBigGEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    else:
        target_F = (
            torch.load(PREP / "F_sem_train.pt", map_location="cpu", weights_only=True)["F_sem"].float().to(dev)
            if use_fnm else torch.from_numpy(np.load(PREP / "F_train.npy")).float().to(dev)
        )
        kv_p = OnConeKVProjector(str(PREP / "unet_kv_metric.pt"), device=dev)
        kv_p.calibrate_fold(target_F)
        model = CrossSubSingleTargetKVPyr(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)

    for sid in range(10):
        model.ebs[sid].fit_from_fold(x_tr_all[sid].float().to(dev))

    ls1 = nn.Parameter(torch.tensor(2.6592, device=dev))
    ls2 = nn.Parameter(torch.tensor(2.6592, device=dev))
    lsx = nn.Parameter(torch.tensor(2.6592, device=dev))
    extra = [("ls1", ls1), ("ls2", ls2), ("lsx", lsx)]
    dec, ndec = [], []
    for n, p_ in list(model.named_parameters()) + extra:
        (ndec if (n.endswith("bias") or "norm" in n or n.startswith("ls") or "emb" in n or "logit" in n) else dec).append(p_)
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.05}, {"params": ndec, "weight_decay": 0.0}], lr=5e-4, betas=(0.9, 0.995))

    bs = 1024
    spr = math.ceil(n_tr / bs)
    total = spr * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 0.5 * (1.0 + math.cos(math.pi * s / total))))
    g = torch.Generator(device=dev).manual_seed(seed)
    rng_np = np.random.default_rng(seed)

    t0 = time.time()
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n_tr, device=dev, generator=g)
        sub_order = rng_np.permutation(10)
        for b in range(spr):
            idx = perm[b * bs:(b + 1) * bs]
            c_b = concept[idx]
            idx_cpu = idx.cpu()
            sid_a = int(sub_order[(2 * b) % 10])
            sid_b = int(sub_order[(2 * b + 1) % 10])
            if sid_b == sid_a:
                sid_b = (sid_a + 1) % 10

            xa, ka = sample_sub_repeats(x_tr_all[sid_a][idx_cpu].float().to(dev), g)
            xb, kb = sample_sub_repeats(x_tr_all[sid_b][idx_cpu].float().to(dev), g)
            xa, xb = shift_batch(xa, 2, g), shift_batch(xb, 2, g)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                sc1 = ls1.float().exp().clamp(max=100.0)
                sc2 = ls2.float().exp().clamp(max=100.0)
                scx = lsx.float().exp().clamp(max=100.0)
                same_c = c_b[:, None] == c_b[None, :]
                eye = torch.eye(len(idx), dtype=torch.bool, device=dev)
                fn_mask = same_c & (~eye)
                diff_c = ~same_c
                off_diag = ~eye

                if is_bigg:
                    with torch.no_grad():
                        pt, st = P_u[idx], Z_u[idx]
                        sp = pt @ pt.T
                        mp_d = 0.15 * sc1.detach() * F.relu(sp - sp[diff_c].mean()) * diff_c.float()
                        mp_a = 0.15 * scx.detach() * F.relu(sp - sp[diff_c].mean()) * off_diag.float()
                    pa, sa = model.forward_sub(xa, sid_a, r_eff=ka)
                    pb, sb = model.forward_sub(xb, sid_b, r_eff=kb)
                    def _bg_l(p_h, s_h):
                        pu = nz(p_h.float() - p_h.float().mean(0, keepdim=True), dim=-1)
                        su = nz(s_h.float(), dim=-1)
                        return 0.60 * _simple_infonce(pu, pt, sc1, fn_mask=fn_mask, margin=mp_d) + 0.40 * _simple_infonce(su, st, sc2, fn_mask=fn_mask)
                    l_vis = 0.5 * (_bg_l(pa, sa) + _bg_l(pb, sb))
                    l_xb = 0.60 * _xbrain_nce(pa, pb, scx, mp_a) + 0.40 * _xbrain_nce(sa, sb, scx, mp_a)
                    loss = l_vis + 0.25 * l_xb
                else:
                    with torch.no_grad():
                        ft = target_F[idx].float()
                        ut = nz(ft, dim=-1)
                        dt = nz(ut - (ut @ kv_p.u_tr.T) * kv_p.u_tr, dim=-1)
                        yk = nz(kv_p.kv.project_kv(ft) - kv_p.kv.mu_kv, dim=-1)
                        si = 0.5 * (dt @ dt.T) + 0.5 * (yk @ yk.T)
                        sbar = si[diff_c].mean()
                        m_v = (0.15 * sc1.detach() * F.relu(si - sbar) * diff_c.float()) if use_fnm else None
                        m_a = 0.15 * scx.detach() * F.relu(si - sbar) * off_diag.float()
                    fa = model.forward_sub(xa, sid_a, r_eff=ka)
                    fb = model.forward_sub(xb, sid_b, r_eff=kb)
                    da, zka = kv_p.cone_and_kv(fa)
                    db, zkb = kv_p.cone_and_kv(fb)
                    fn_use = fn_mask if use_fnm else None
                    l_vis = 0.5 * (_hybrid_infonce(da, zka, dt, yk, sc1, fn_mask=fn_use, margin=m_v) +
                                   _hybrid_infonce(db, zkb, dt, yk, sc1, fn_mask=fn_use, margin=m_v))
                    l_xb = _xbrain_cone_nce(fa, fb, kv_p.u_tr, scx, m_a)
                    loss = l_vis + 0.25 * l_xb

            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec + ndec, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

    train_s = time.time() - t0
    model.eval()
    pack_0tta = {"train_s": train_s}
    with torch.no_grad():
        for s in ALL_SUBS:
            sid = s - 1
            x_te = torch.from_numpy(np.load(PREP / f"sub-{s:02d}_test.npy")).float().to(dev)
            full_in = ivw_mean(x_te)
            perm_bag = torch.randperm(80, generator=torch.Generator().manual_seed(7 + s)).to(dev)
            bag_ins = torch.cat([ivw_mean(x_te[:, perm_bag[g_ * 8:(g_ + 1) * 8]]) for g_ in range(10)], dim=0)
            if is_bigg:
                pf, sf = model.forward_sub(full_in, sid, r_eff=72.0)
                pb, sb = model.forward_sub(bag_ins, sid, r_eff=8.0)
                pack_0tta[f"sub-{s:02d}"] = {
                    "p_full": pf.float().cpu(), "s_full": sf.float().cpu(),
                    "p_bag10": pb.float().view(10, 200, -1).cpu(), "s_bag10": sb.float().view(10, 200, -1).cpu(),
                }
            else:
                ff = model.forward_sub(full_in, sid, r_eff=72.0)
                fb = model.forward_sub(bag_ins, sid, r_eff=8.0)
                pack_0tta[f"sub-{s:02d}"] = {
                    "f_full": ff.float().cpu(), "f_bag10": fb.float().view(10, 200, -1).cpu(),
                }
            del x_te, full_in, bag_ins

    tmp_m = out_d / f".brain_model_fp16.{os.getpid()}.tmp"
    torch.save({k: (v.half().cpu() if v.is_floating_point() else v.cpu()) for k, v in model.state_dict().items()}, tmp_m)
    tmp_m.replace(out_d / "brain_model_fp16.pt")
    tmp_0 = out_d / f".test_0tta.{os.getpid()}.tmp"
    torch.save(pack_0tta, tmp_0)
    tmp_0.replace(out_d / "test_0tta_all10.pt")
    print(f"[O14_TRAIN] {model_name:<16} seed={seed} ep={epochs} done in {train_s:.1f}s", flush=True)
    del model, x_tr_all, kv_p, target_F, P_u, Z_u
    gc.collect()
    torch.cuda.empty_cache()


def run_o14_single_sub_tta(sub: int, model_name: str, dev: str = "cuda:0", steps: int = 50) -> dict:
    """100% STRICTLY SINGLE-SUBJECT split-half TTA (updates ONLY sub_stems[sid] on sub-XX_test.npy!)."""
    cache_p = O14_ENC / model_name / f"sub-{sub:02d}_test_agg.pt"
    if cache_p.exists():
        return torch.load(cache_p, map_location="cpu", weights_only=True)

    is_bigg = (model_name == "o14_xbrain_bigg")
    p0_list = [torch.load(O14_ENC / model_name / f"s{sd}" / "test_0tta_all10.pt", map_location="cpu", weights_only=True)[f"sub-{sub:02d}"] for sd in SEEDS_5]

    ok_mask = torch.from_numpy(np.load(PREP / "lat512_train_ok.npy")).bool()
    F_ref = (
        torch.load(PREP / "F_sem_train.pt", map_location="cpu", weights_only=True)["F_sem"].float()
        if model_name == "o14_xkv_fsem" else torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    )
    _, _, _, u_ref = _cone_params(F_ref[ok_mask])
    u_ref = u_ref.to(dev)
    del F_ref

    t0 = time.time()
    sid = sub - 1
    x_te = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_test.npy")).float().to(dev)
    full_in = ivw_mean(x_te)
    perm_bag = torch.randperm(80, generator=torch.Generator().manual_seed(7 + sub)).to(dev)
    bag_ins = torch.cat([ivw_mean(x_te[:, perm_bag[g_ * 8:(g_ + 1) * 8]]) for g_ in range(10)], dim=0)
    tgt = torch.arange(200, device=dev)

    tta_a, tta_b = [], []
    for sd in SEEDS_5:
        ckpt = torch.load(O14_ENC / model_name / f"s{sd}" / "brain_model_fp16.pt", map_location="cpu", weights_only=True)
        ckpt = {k: (v.float() if v.is_floating_point() else v) for k, v in ckpt.items()}
        m = (CrossSubBigGEncoder if is_bigg else CrossSubSingleTargetKVPyr)(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.0).to(dev)
        m.load_state_dict(ckpt)
        m.eval().requires_grad_(False)
        for p in m.sub_stems[sid].parameters():
            p.requires_grad_(True)
        opt = torch.optim.Adam(m.sub_stems[sid].parameters(), lr=1e-4)
        g = torch.Generator(device=dev).manual_seed(7000 + sid * 37 + sd)

        with torch.no_grad():
            o0 = m.forward_sub(full_in, sid, r_eff=72.0)
            o0 = (o0[0].float(), o0[1].float()) if is_bigg else o0.float()

        for _ in range(steps):
            perm = torch.rand(200, 80, device=dev, generator=g).argsort(1)
            xa = ivw_mean(torch.gather(x_te, 1, perm[:, :40, None, None].expand(-1, -1, 17, 250)))
            xb = ivw_mean(torch.gather(x_te, 1, perm[:, 40:, None, None].expand(-1, -1, 17, 250)))
            if is_bigg:
                pa, sa = m.forward_sub(xa, sid, r_eff=36.0)
                pb, sb = m.forward_sub(xb, sid, r_eff=36.0)
                pac = nz(nz(pa.float(), dim=1) - nz(pa.float(), dim=1).mean(0, keepdim=True), dim=1)
                pbc = nz(nz(pb.float(), dim=1) - nz(pb.float(), dim=1).mean(0, keepdim=True), dim=1)
                sac = nz(nz(sa.float(), dim=1) - nz(sa.float(), dim=1).mean(0, keepdim=True), dim=1)
                sbc = nz(nz(sb.float(), dim=1) - nz(sb.float(), dim=1).mean(0, keepdim=True), dim=1)
                lg = (0.60 * (pac @ pbc.T) + 0.40 * (sac @ sbc.T)) / 0.1
                pf, sf = m.forward_sub(full_in, sid, r_eff=72.0)
                anc = 0.5 * (((pf.float() - o0[0]).pow(2).sum(1) / o0[0].pow(2).sum(1).clamp_min(1e-6)).mean() +
                             ((sf.float() - o0[1]).pow(2).sum(1) / o0[1].pow(2).sum(1).clamp_min(1e-6)).mean())
            else:
                dfa = _perp_dir(m.forward_sub(xa, sid, r_eff=36.0).float(), u_ref)
                dfb = _perp_dir(m.forward_sub(xb, sid, r_eff=36.0).float(), u_ref)
                lg = (dfa @ dfb.T) / 0.1
                ff = m.forward_sub(full_in, sid, r_eff=72.0).float()
                anc = ((ff - o0).pow(2).sum(1) / o0.pow(2).sum(1).clamp_min(1e-6)).mean()
            (0.5 * (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.T, tgt)) + 1.0 * anc).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            if is_bigg:
                pf, sf = m.forward_sub(full_in, sid, r_eff=72.0)
                pb, sb = m.forward_sub(bag_ins, sid, r_eff=8.0)
                tta_a.append((0.70 * pf.float() + 0.30 * pb.float().view(10, 200, -1).mean(0)).cpu())
                tta_b.append((0.70 * sf.float() + 0.30 * sb.float().view(10, 200, -1).mean(0)).cpu())
            else:
                ff = m.forward_sub(full_in, sid, r_eff=72.0).float()
                fb = m.forward_sub(bag_ins, sid, r_eff=8.0).float().view(10, 200, -1).mean(0)
                tta_a.append((0.70 * ff + 0.30 * fb).cpu())
        del m

    if is_bigg:
        pack = {
            "p_0tta": 0.85 * torch.stack([p["p_full"] for p in p0_list], 0).mean(0) + 0.15 * torch.stack([p["p_bag10"] for p in p0_list], 0).mean(0).mean(0),
            "s_0tta": 0.85 * torch.stack([p["s_full"] for p in p0_list], 0).mean(0) + 0.15 * torch.stack([p["s_bag10"] for p in p0_list], 0).mean(0).mean(0),
            "p_tta": torch.stack(tta_a, 0).mean(0),
            "s_tta": torch.stack(tta_b, 0).mean(0),
            "tta_s": time.time() - t0,
        }
    else:
        pack = {
            "f_0tta": 0.85 * torch.stack([p["f_full"] for p in p0_list], 0).mean(0) + 0.15 * torch.stack([p["f_bag10"] for p in p0_list], 0).mean(0).mean(0),
            "f_tta": torch.stack(tta_a, 0).mean(0),
            "tta_s": time.time() - t0,
        }
    tmp_p = cache_p.with_name(f".{cache_p.name}.{os.getpid()}.tmp")
    torch.save(pack, tmp_p)
    tmp_p.replace(cache_p)
    print(f"[O14_TTA] s{sub:02d} {model_name:<16} done in {pack['tta_s']:.1f}s", flush=True)
    del x_te, full_in, bag_ins
    gc.collect()
    torch.cuda.empty_cache()
    return pack


def export_o14_conditions_and_latents(subs, bigG_pack: dict, dev: str = "cuda:0"):
    """Build O14 Tri-Arch cones, continuous cross-brain OpenCLIP-bigG/14 targets, retrieval audit, and lat512_v2."""
    ok_mask = torch.from_numpy(np.load(PREP / "lat512_train_ok.npy")).bool()
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    R_f, a_f, b_f, u_f = _cone_params(F_tr[ok_mask])
    F_te = torch.from_numpy(np.load(PREP / "F_test.npy")).float()
    d_F_te = _perp_dir(F_te, u_f)

    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    H14_tr = emb_tr[EMB_KEYS[0]][ok_mask].float()
    B32_tr = emb_tr[EMB_KEYS[1]][ok_mask].float()
    VAE_tr = emb_tr[EMB_KEYS[2]][ok_mask].float().reshape(int(ok_mask.sum()), -1)
    h14_norm = float(H14_tr.norm(dim=1).mean())
    b32_norm = float(B32_tr.norm(dim=1).mean())
    vae_mu = VAE_tr.mean(0, keepdim=True).to(dev)
    vae_sd = VAE_tr.std(0, keepdim=True).clamp_min(1e-6).to(dev)
    del F_tr, H14_tr, B32_tr, VAE_tr

    fe_vis = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.to(dev).eval().requires_grad_(False)
    W_ppe = bigG_pack["W_ppe"].float()
    W_seq = bigG_pack["W_seq"].float()

    O14_COND.mkdir(parents=True, exist_ok=True)
    cdir_o14 = RUNS / "cond" / "cone_o14_triarch_kvtta_w1.00"
    cdir_o14.mkdir(parents=True, exist_ok=True)

    ret_audit = {}
    d_coh_list, ppe_coh_list, seq_coh_list = [], [], []

    for s in subs:
        # Load cached O10, O13_xbrain, O14_xkv_ffull, O14_xkv_fsem, O14_xbrain_bigg (100% strictly single-subject sub-XX!)
        d_o10 = _perp_dir(torch.from_numpy(np.load(RUNS / "cond" / "cone_o10_bifurc_kvtta_w1.00" / f"sub-{s:02d}.npy")).float(), u_f)
        z0_o10 = torch.from_numpy(np.load(RUNS / "low" / f"sub-{s:02d}" / "lat512_o10_bifurc_v2_n0.80.npy")).float()
        px13 = torch.load(O13_ENC / "o13_xbrain" / f"sub-{s:02d}_test_agg.pt", map_location="cpu", weights_only=True)
        p_ff = torch.load(O14_ENC / "o14_xkv_ffull" / f"sub-{s:02d}_test_agg.pt", map_location="cpu", weights_only=True)
        p_fs = torch.load(O14_ENC / "o14_xkv_fsem" / f"sub-{s:02d}_test_agg.pt", map_location="cpu", weights_only=True)
        p_bg = torch.load(O14_ENC / "o14_xbrain_bigg" / f"sub-{s:02d}_test_agg.pt", map_location="cpu", weights_only=True)

        # O13_xbrain cone (using 0.70 d_f + 0.30 d_syn per probe_xkv insight!)
        z16 = F.avg_pool2d(z0_o10.to(dev), 4).reshape(200, -1)
        vae_hat = vae_mu + 0.65 * (z16 - z16.mean(0, keepdim=True)) * (vae_sd / z16.std(0, keepdim=True).clamp_min(1e-6))
        with torch.no_grad():
            f_syn = fe_vis(**{EMB_KEYS[0]: nz(px13["h_tta"].float(), dim=1).to(dev) * h14_norm,
                              EMB_KEYS[1]: nz(px13["b_tta"].float(), dim=1).to(dev) * b32_norm,
                              EMB_KEYS[2]: vae_hat}).float().cpu()
        d_o13_sa = nz(0.70 * _perp_dir(px13["f_tta"].float(), u_f) + 0.30 * _perp_dir(f_syn, u_f), dim=1)
        d_xkv = nz(0.50 * _perp_dir(p_ff["f_tta"].float(), u_f) + 0.50 * _perp_dir(p_fs["f_tta"].float(), u_f), dim=1)
        d_o14 = nz(0.40 * d_xkv + 0.40 * d_o13_sa + 0.20 * d_o10, dim=1)
        c_o14 = _to_cone(d_o14, R_f, a_f, b_f, u_f)
        np.save(cdir_o14 / f"sub-{s:02d}.npy", c_o14.numpy().astype(np.float32))

        # Continuous OpenCLIP-ViT-bigG/14 embeddings (50% direct O14_xbrain_bigg + 50% ridge from O13 H14/B32)
        xh = torch.cat([nz(px13["h_tta"].float(), dim=1), nz(px13["b_tta"].float(), dim=1)], dim=1)
        ppe_hat = 0.50 * nz(xh @ W_ppe, dim=1) + 0.50 * nz(p_bg["p_tta"].float(), dim=1)
        seq_hat = 0.50 * nz(xh @ W_seq, dim=1) + 0.50 * nz(p_bg["s_tta"].float(), dim=1)

        torch.save({
            "c_o14": c_o14, "d_o14": d_o14,
            "h_tta": px13["h_tta"].float(), "b_tta": px13["b_tta"].float(),
            "ppe_hat": ppe_hat, "seq_hat": seq_hat,
        }, O14_COND / f"sub-{s:02d}_o14_pack.pt")

        d_coh_list.append(d_o14)
        ppe_coh_list.append(ppe_hat)
        seq_coh_list.append(seq_hat)

        f_xkv = float(((d_xkv @ d_F_te.T).argmax(1) == torch.arange(200)).float().mean())
        f_o14 = float(((d_o14 @ d_F_te.T).argmax(1) == torch.arange(200)).float().mean())
        rho_o14 = float((d_o14 * d_F_te).sum(1).mean())
        ret_audit[f"sub-{s:02d}"] = {"XKV_F_200w": f_xkv, "O14_F_200w": f_o14, "O14_rho_F": rho_o14}
        print(f"[O14_RET] s{s:02d} | XKV_F={f_xkv:.3f} | O14_TriArch F={f_o14:.3f} rhoF={rho_o14:.3f}", flush=True)

    if len(subs) == 10:
        d_coh = nz(torch.stack(d_coh_list, 0).mean(0), dim=1)
        ppe_coh = nz(torch.stack(ppe_coh_list, 0).mean(0), dim=1)
        seq_coh = nz(torch.stack(seq_coh_list, 0).mean(0), dim=1)
        c_coh = _to_cone(d_coh, R_f, a_f, b_f, u_f)
        torch.save({"c_coh": c_coh, "ppe_coh": ppe_coh, "seq_coh": seq_coh}, O14_COND / "cohort10_o14_pack.pt")
        f_c = float(((d_coh @ d_F_te.T).argmax(1) == torch.arange(200)).float().mean())
        r_c = float((d_coh * d_F_te).sum(1).mean())
        ret_audit["DIAG_COHORT10"] = {"O14_F_200w": f_c, "O14_rho_F": r_c}
        ret_audit["mean10"] = {
            "XKV_F_200w": float(np.mean([ret_audit[f"sub-{s:02d}"]["XKV_F_200w"] for s in ALL_SUBS])),
            "O14_F_200w": float(np.mean([ret_audit[f"sub-{s:02d}"]["O14_F_200w"] for s in ALL_SUBS])),
            "O14_rho_F":  float(np.mean([ret_audit[f"sub-{s:02d}"]["O14_rho_F"] for s in ALL_SUBS])),
        }
        print(f"\n[O14_RET] MEAN10: XKV_F={ret_audit['mean10']['XKV_F_200w']:.4f} | O14_F={ret_audit['mean10']['O14_F_200w']:.4f} rhoF={ret_audit['mean10']['O14_rho_F']:.4f}", flush=True)
        print(f"[O14_RET] DIAG_COHORT10: O14_F={f_c:.4f} rhoF={r_c:.4f}", flush=True)
        write_json(O14_ROOT / "official_retrieval_audit.json", ret_audit)

    del fe_vis
    gc.collect()
    torch.cuda.empty_cache()

    lat_script = str(_find_script("lat512_start.py"))
    sub_csv = "all" if len(subs) == 10 else ",".join(map(str, subs))
    ltag = "lat512_o14_triarch_v2"
    if not all((RUNS / "low" / f"sub-{s:02d}" / f"{ltag}_n0.80.npy").exists() for s in subs):
        subprocess.run([sys.executable, "-u", lat_script, sub_csv, "--mode_v", "v2", "--cond_dirs", "cone_o14_triarch_kvtta_w1.00", "--tag", ltag], check=True)
    print(f"[O14_COND] Exported O14 cones, bigG targets & {ltag} for {len(subs)} subjects.", flush=True)


@torch.inference_mode()
def evaluate_sel4_arm(runner: OfficialTurboRunner, c_all: torch.Tensor, z0_all: torch.Tensor,
                      d_q: torch.Tensor, h_q: torch.Tensor, b_q: torch.Tensor, u_f: torch.Tensor,
                      pe_all=None, ppe_all=None, bs: int = 4, save_audit_path: Path = None) -> dict:
    """Explicit higher-budget row (nsel=4, 16 UNet NFE/img) using ONLY Brain-HIVE FEmbedder (ZERO OpenAI ViT-L/14!)."""
    from fenc import FEmbedder
    from metrics import _nets, score as eval_metrics_score

    t0 = time.time()
    dev = runner.dev
    init = torch.cat([
        (runner.vae32.decode(z0_all[j:j + 10].to(dev, torch.float32) / runner.vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
        for j in range(0, 200, 10)
    ])
    steps = 4
    smap = 1.0 - 0.5 * torch.full((200, 1, 64, 64), 0.625)
    hold = (steps - steps * smap).clamp(min=0)

    # Generate 16 independent draws (4 groups x 4 candidates) in chunks of 4
    rec16 = torch.empty(16, 200, 3, 512, 512)
    for k_off in range(0, 16, 4):
        for s0 in range(0, 200, bs):
            q = list(range(s0, min(200, s0 + bs)))
            Q = len(q)
            cc = c_all[q].to(dev, runner.dt).repeat_interleave(4, dim=0)
            ips = runner.proj(runner.adapter(cc))
            pe_b = pe_all[q].to(dev, runner.dt).repeat_interleave(4, dim=0) if pe_all is not None else runner.pe.expand(Q * 4, -1, -1)
            ppe_b = ppe_all[q].to(dev, runner.dt).repeat_interleave(4, dim=0) if ppe_all is not None else runner.ppe.expand(Q * 4, -1)
            gens = [torch.Generator(dev).manual_seed(1_000_003 * i + (k_off + k) + 12345) for i in q for k in range(4)]
            i0 = init[q].repeat_interleave(4, dim=0).to(dev, runner.dt)
            z0 = z0_all[q].to(dev).float().repeat_interleave(4, dim=0)
            eps = torch.stack([torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * i + (k_off + k) + 99)) for i in q for k in range(4)]).to(dev)
            h_ = hold[q].repeat_interleave(4, dim=0).to(dev)
            def cb(pp, i_step, t_cur, ck):
                lt = ck["latents"].float()
                sig = pp.scheduler.sigmas[int(pp.scheduler.step_index)].to(dev)
                a = (h_ - i_step).clamp(0.0, 1.0)
                if a.max() > 0:
                    lt = a * (z0 + sig * eps) + (1.0 - a) * lt
                if i_step in (2, 3):
                    g_t = 0.22 if i_step == 2 else 0.04
                    d_ = (z0 + sig * eps) - lt
                    lt = lt + g_t * F.interpolate(F.avg_pool2d(d_, 8), size=64, mode="bilinear", align_corners=False)
                ck["latents"] = lt.to(runner.dt)
                return ck
            lat_out = runner.p2(
                image=i0, prompt_embeds=pe_b, pooled_prompt_embeds=ppe_b,
                num_inference_steps=4, strength=1.0, guidance_scale=0.0, output_type="latent",
                generator=gens, cross_attention_kwargs={"ip_hidden_states": ips},
                callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"],
            ).images
            dec_l = [
                (runner.vae32.decode(lat_out[j:j + 4].float() / runner.vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
                for j in range(0, len(lat_out), 4)
            ]
            img = (torch.cat(dec_l, 0) * 255.0).round().div(255.0)
            rec16[k_off:k_off + 4, q] = img.view(Q, 4, 3, 512, 512).transpose(0, 1)

    # Offload runner temporarily while scoring with Brain-HIVE FEmbedder (ZERO OpenAI ViT-L/14!)
    runner.pipe.to("cpu"); runner.vae32.to("cpu"); runner.proj.to("cpu"); runner.adapter.to("cpu")
    torch.cuda.empty_cache()

    femb = FEmbedder(dev=dev)
    d_qg = d_q.to(dev)
    h_qg = nz(h_q.to(dev) - h_q.to(dev).mean(0, keepdim=True), dim=1)
    b_qg = nz(b_q.to(dev) - b_q.to(dev).mean(0, keepdim=True), dim=1)
    u_fg = u_f.to(dev)
    scores = torch.empty(16, 200)
    for k_ in range(16):
        fc, hc, bc = [], [], []
        for j in range(0, 200, 40):
            pts = femb.parts(rec16[k_, j:j + 40])
            fc.append(femb.fuse(**pts).float())
            hc.append(pts[EMB_KEYS[0]].float())
            bc.append(pts[EMB_KEYS[1]].float())
        fc = _perp_dir(torch.cat(fc, 0), u_fg)
        hc = torch.cat(hc, 0); hc = nz(hc - hc.mean(0, keepdim=True), dim=1)
        bc = torch.cat(bc, 0); bc = nz(bc - bc.mean(0, keepdim=True), dim=1)
        scores[k_] = (0.45 * (fc * d_qg).sum(1) + 0.40 * (hc * h_qg).sum(1) + 0.15 * (bc * b_qg).sum(1)).cpu()
    del femb
    torch.cuda.empty_cache()

    rec = torch.empty(4, 200, 3, 512, 512)
    for k_ in range(4):
        best_c = scores[4 * k_:4 * k_ + 4].argmax(0)
        for i_ in range(200):
            rec[k_, i_] = rec16[4 * k_ + int(best_c[i_]), i_]
    del rec16
    gen_s = float(time.time() - t0)

    ps = eval_metrics_score(rec, runner.gt)
    mean_m = {k: float(np.mean([r[k] for r in ps])) for k in ps[0]}
    mean_m["per_draw"] = ps
    mean_m["nfe_per_img"] = 16
    mean_m["nsel"] = 4
    mean_m["gen_wall_s"] = gen_s
    mean_m["total_wall_s"] = float(time.time() - t0)

    if save_audit_path is not None:
        clip_fn, clip_pre, _ = _nets()["clip"]
        def _clip_np(x_rgb):
            feats = []
            for j in range(0, len(x_rgb), 50):
                feats.append(clip_fn(clip_pre(x_rgb[j:j + 50].to(dev))).float().flatten(1).cpu())
            return torch.cat(feats, dim=0).numpy()
        gt_f = _clip_np(runner.gt)
        N = len(gt_f)
        ranks = torch.empty(4, N, dtype=torch.int16)
        top1_idx = torch.empty(4, N, dtype=torch.int16)
        for k_i in range(4):
            re_f = _clip_np(rec[k_i])
            r_mat = np.corrcoef(gt_f, re_f)[:N, N:]
            diag = np.diag(r_mat)[None, :]
            ranks[k_i] = torch.from_numpy((r_mat > diag).sum(axis=0).astype(np.int16) + 1)
            top1_idx[k_i] = torch.from_numpy(r_mat.argmax(axis=0).astype(np.int16))
        save_audit_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "gt_ids": runner.gt_ids, "ranks": ranks, "top1_idx": top1_idx,
            "k0_rgb_u8": (rec[0].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8),
            "metrics": mean_m,
        }, save_audit_path)

    # Restore runner to GPU cleanly for next subject
    runner.pipe.to(dev, runner.dt); runner.vae32.to(dev); runner.proj.to(dev, runner.dt); runner.adapter.to(dev, runner.dt)
    torch.cuda.empty_cache()
    return mean_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "train", "tta", "cond", "eval"], default="all")
    ap.add_argument("--models", default="o14_xkv_ffull,o14_xkv_fsem,o14_xbrain_bigg")
    ap.add_argument("--seeds", default="2025,2026,2027,2028,2029")
    ap.add_argument("--subs", default="all")
    ap.add_argument("--arms", default="O14_noBigG,O14,O14_ctrl_shuf,O14_sel4,O14_diag_cohort10")
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--dev", default="cuda:0")
    args = ap.parse_args()

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    subs = list(ALL_SUBS) if args.subs == "all" else [int(x) for x in args.subs.split(",")]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    dev = args.dev

    bigG_pack = build_or_load_bigG_train1654(dev=dev)

    if args.phase in ("all", "train"):
        for mn in models:
            for sd in seeds:
                train_one_o14_seed(mn, sd, bigG_pack, dev=dev, epochs=args.epochs)
        if args.phase == "train":
            return

    if args.phase in ("all", "tta"):
        for mn in models:
            for s in subs:
                run_o14_single_sub_tta(s, mn, dev=dev, steps=50)
        if args.phase == "tta":
            return

    if args.phase in ("all", "cond"):
        export_o14_conditions_and_latents(subs, bigG_pack, dev=dev)
        if args.phase == "cond":
            return

    O14_EVAL.mkdir(parents=True, exist_ok=True)
    ok_mask = torch.from_numpy(np.load(PREP / "lat512_train_ok.npy")).bool()
    _, _, _, u_f = _cone_params(torch.from_numpy(np.load(PREP / "F_train.npy")).float()[ok_mask])
    runner = OfficialTurboRunner(dev=dev)
    perm_shuf = torch.roll(torch.arange(200), shifts=73)

    for aname in arms:
        if aname == "O14_diag_cohort10":
            out_j = O14_EVAL / "DIAG_COHORT10_O14.json"
            audit_p = O14_AUDIT / "DIAG_COHORT10_O14_audit.pt"
            if out_j.exists() and audit_p.exists():
                m = json.loads(out_j.read_text())
                print(f"[CACHED] DIAG_COHORT10_O14 | top1={m['clip_top1']:.4f} 2w={m['clip']:.4f} Inc={m['inception']:.4f}", flush=True)
                continue
            cp = torch.load(O14_COND / "cohort10_o14_pack.pt", map_location="cpu", weights_only=True)
            z0_coh = torch.stack([
                torch.from_numpy(np.load(RUNS / "low" / f"sub-{s:02d}" / "lat512_o14_triarch_v2_n0.80.npy")).float()
                for s in ALL_SUBS
            ], dim=0).mean(0)
            pe_c, ppe_c = build_bigG_embeds(cp["ppe_coh"], cp["seq_coh"], bigG_pack, runner.pe, runner.ppe, alpha=0.35)
            m = runner.evaluate_arm(cp["c_coh"], z0_coh, pe_all=pe_c, ppe_all=ppe_c, K=4, bs=args.bs, save_audit_path=audit_p)
            write_json(out_j, m)
            print(
                f"[O14_DIAG_COHORT10] top1={m['clip_top1']:.4f} 2w={m['clip']:.4f} "
                f"SSIM={m['ssim']:.4f} PixC={m['pixcorr']:.4f} Inc={m['inception']:.4f} Eff={m['effnet']:.4f} ({m['gen_wall_s']:.1f}s)",
                flush=True,
            )
            continue

        for s in subs:
            out_j = O14_EVAL / f"s{s:02d}_{aname}.json"
            save_aud = aname in ("O14_noBigG", "O14", "O14_sel4")
            audit_p = (O14_AUDIT / f"sub-{s:02d}_{aname}_audit.pt") if save_aud else None
            if out_j.exists() and (audit_p is None or audit_p.exists()):
                m = json.loads(out_j.read_text())
                print(f"[CACHED] {aname:<16} s{s:02d} | top1={m['clip_top1']:.4f} 2w={m['clip']:.4f} SSIM={m['ssim']:.4f} Inc={m['inception']:.4f}", flush=True)
                continue

            spack = torch.load(O14_COND / f"sub-{s:02d}_o14_pack.pt", map_location="cpu", weights_only=True)
            c_use = spack["c_o14"]
            z0 = torch.from_numpy(np.load(RUNS / "low" / f"sub-{s:02d}" / "lat512_o14_triarch_v2_n0.80.npy")).float()

            if aname == "O14_noBigG":
                m = runner.evaluate_arm(c_use, z0, K=4, bs=args.bs, save_audit_path=audit_p)
                if audit_p is not None and audit_p.exists():
                    ap_d = torch.load(audit_p, map_location="cpu", weights_only=True)
                    ap_d.pop("k0_rgb_u8", None)
                    torch.save(ap_d, audit_p)
            elif aname == "O14":
                pe_in, ppe_in = build_bigG_embeds(spack["ppe_hat"], spack["seq_hat"], bigG_pack, runner.pe, runner.ppe, alpha=0.35)
                m = runner.evaluate_arm(c_use, z0, pe_all=pe_in, ppe_all=ppe_in, K=4, bs=args.bs, save_audit_path=audit_p)
            elif aname == "O14_ctrl_shuf":
                pe_t, ppe_t = build_bigG_embeds(spack["ppe_hat"], spack["seq_hat"], bigG_pack, runner.pe, runner.ppe, alpha=0.35)
                m = runner.evaluate_arm(c_use, z0, pe_all=pe_t[perm_shuf], ppe_all=ppe_t[perm_shuf], K=4, bs=args.bs)
            elif aname == "O14_sel4":
                pe_in, ppe_in = build_bigG_embeds(spack["ppe_hat"], spack["seq_hat"], bigG_pack, runner.pe, runner.ppe, alpha=0.35)
                m = evaluate_sel4_arm(
                    runner, c_use, z0, spack["d_o14"], spack["h_tta"], spack["b_tta"], u_f,
                    pe_all=pe_in, ppe_all=ppe_in, bs=4, save_audit_path=audit_p,
                )
            else:
                raise ValueError(aname)

            write_json(out_j, m)
            print(
                f"[O14_OFFICIAL] {aname:<15} s{s:02d} | top1={m['clip_top1']:.4f} 2w={m['clip']:.4f} "
                f"SSIM={m['ssim']:.4f} PixC={m['pixcorr']:.4f} Inc={m['inception']:.4f} Eff={m['effnet']:.4f} ({m['gen_wall_s']:.1f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
