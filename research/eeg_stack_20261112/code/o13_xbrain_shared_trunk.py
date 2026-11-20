"""O13: Cross-Subject Shared-Trunk & Cross-Brain Contrastive Consistency (L_xbrain)
with 100% STRICTLY SINGLE-SUBJECT TEST INFERENCE on THINGS-EEG2 (10 subjects, R=80, 4 UNet NFE/img).

Why O13 directly solves the Part I training-SNR bottleneck:
  - Part I 10-subject audit proved raw test EEG (R=80) already achieves 96.67% +/- 2.98% 200-way
    cross-session identification (Fano lower bound I >= 7.20 bits / 7.64 bits), whereas training-time
    R=4 EEG achieves only 18.85% +/- 10.25% cross-block identification (single-trial SNR = 0.0091).
  - Single-subject encoders (O9..O12) therefore suffer severe training-time variance on 16,540 noisy
    R=4 examples. All 10 subjects viewed the exact same 16,540 training images (165,400 pairs = 10x).
  - CrossSubjectHorizonEncoder gives each subject an independent anatomical front-end
    (TargetSNRGate + DecoupledHorizon sub_stems[sid] + sub_emb[sid]) feeding a SHARED 3-layer
    ResNet1D semantic trunk and (F_full, H14, B32) heads.
  - Exact Part II Fair Prior (L_xbrain): two DISTINCT subjects viewing the SAME training image
    have strictly independent trial/session noise Cov(eps^(s_a), eps^(s_b)) == 0 (avoiding D23's
    within-session noise correlation counterexample). Symmetric cross-brain InfoNCE with NO fn_mask
    and visual-similarity hard-negative margin enforces cross-brain semantic consistency while
    preserving fine-grained intra-concept discrimination (+3.46%pp Intra10 & +4.38%pp Hard10 on D_dev200).

Strict Protocol & Compute-Matched Controls:
  - Exact compute-matched control: o13_ctrl_xsub (lambda_xbrain = 0.00) vs o13_xbrain (lambda_xbrain = 0.25),
    identical architecture, identical 150 meta-epochs (30 passes/sub), identical 5 seeds (2025..2029),
    identical dual-subject batching, RNG streams, and 50-step single-subject split-half kvTTA.
  - 100% STRICTLY SINGLE-SUBJECT TEST INFERENCE for all headline/ablation arms: when evaluating sub-XX,
    both 0-TTA and 50-step split-half kvTTA (updating ONLY sub_stems[sid]) touch ONLY sub-XX_test.npy.
  - Strictly zero OpenAI CLIP ViT-L/14 leakage (text_encoder_1 = "", text_encoder_2 = ""), 0 DDPM heads,
    pure F_train[ok] cone (w = 1.00), unchanged lat512_start.py (--mode_v v2), SDXL-Turbo smap c=0.625,
    low_keep=(8, 0.22, 0.04), 4 UNet NFE/img, nsel=1, strictly independent K=4 draws, zero blur,
    unchanged metrics.py.
  - Track B (clearly labeled non-single-subject diagnostic ceiling): O13_diag_cohort10 averages the
    10 subjects' official test conditions (ZERO test labels/images) to quantify the multi-brain SNR ceiling.
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
)
from train_flex import ivw_mean, pyramid_feats, shift_batch

sys.path.insert(0, str(HIVE))
from main.layers import ResNet1DBlock  # noqa: E402
from main.models_adapter import FusionEncoderModel  # noqa: E402

nz = F.normalize
CODE_DIR = Path(__file__).resolve().parent
ALL_SUBS = tuple(range(1, 11))
SEEDS_5 = (2025, 2026, 2027, 2028, 2029)

O13_ROOT = RUNS / "o13_official"
O13_ENC = O13_ROOT / "enc"
O13_COND = O13_ROOT / "cond"
O13_EVAL = O13_ROOT / "eval"
O13_AUDIT = O13_ROOT / "audit_cache"


def _find_script(name: str) -> Path:
    candidates = [CODE_DIR] + [Path(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    for d in candidates:
        p = d / name
        if p.exists():
            return p
    return CODE_DIR / name


class CrossSubjectHorizonEncoder(nn.Module):
    """Per-subject EB gate + decoupled-horizon stems -> SHARED 3-layer ResNet1D trunk -> (F_full, H14, B32)."""

    def __init__(self, n_subs: int = 10, t_len: int = 250, c_in: int = 17, hidden_size: int = 1024,
                 num_layers: int = 3, dropout: float = 0.20, in_drop: float = 0.15, horizons=(115, 160, 225)):
        super().__init__()
        self.n_subs = n_subs
        self.horizons = horizons
        dims = [pyramid_feats(torch.zeros(1, c_in, t_h)).shape[1] for t_h in horizons]
        self.ebs = nn.ModuleList([TargetSNRGate(c_in=c_in, t_len=t_len) for _ in range(n_subs)])
        self.in_drop = nn.Dropout(in_drop)
        self.sub_stems = nn.ModuleList([
            nn.ModuleList([nn.Linear(dim_h, hidden_size) for dim_h in dims])
            for _ in range(n_subs)
        ])
        self.stem_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in dims])
        self.horizon_logits = nn.Parameter(torch.zeros(len(horizons)))
        self.sub_emb = nn.Embedding(n_subs, hidden_size)
        nn.init.zeros_(self.sub_emb.weight)

        self.block = ResNet1DBlock(hidden_size=hidden_size, intermediate_size=hidden_size * 2, num_layers=num_layers, dropout=dropout)
        self.head_f = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1024))
        self.head_h14 = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1024))
        self.head_b32 = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 512))

    def forward_sub(self, x: torch.Tensor, sid: int, r_eff=4.0, return_branches: bool = False):
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
        out = (self.head_f(c_ens), self.head_h14(c_ens), self.head_b32(c_ens))
        if return_branches:
            br = [(self.head_f(c), self.head_h14(c), self.head_b32(c)) for c in cs]
            return out, br
        return out


def _xbrain_nce(ua: torch.Tensor, ub: torch.Tensor, scale_x: torch.Tensor, margin_all: torch.Tensor):
    ca = nz(ua.float() - ua.float().mean(0, keepdim=True), dim=-1)
    cb = nz(ub.float() - ub.float().mean(0, keepdim=True), dim=-1)
    lg = scale_x * (ca @ cb.T) + margin_all
    tb = torch.arange(len(ca), device=ca.device)
    return 0.5 * (F.cross_entropy(lg, tb) + F.cross_entropy(lg.T, tb))


def _xbrain_cone_nce(fa: torch.Tensor, fb: torch.Tensor, u_f: torch.Tensor, scale_x: torch.Tensor, margin_all: torch.Tensor):
    da = _perp_dir(fa.float(), u_f)
    db = _perp_dir(fb.float(), u_f)
    lg = scale_x * (da @ db.T) + margin_all
    tb = torch.arange(len(da), device=da.device)
    return 0.5 * (F.cross_entropy(lg, tb) + F.cross_entropy(lg.T, tb))


def train_one_xsub_seed(model_name: str, seed: int, dev: str = "cuda:0", epochs: int = 150):
    """Train one full-train (16,540 imgs x 10 subs = 165,400 pairs, ZERO test data) cross-subject seed."""
    assert model_name in ("o13_ctrl_xsub", "o13_xbrain")
    lam_x = 0.0 if model_name == "o13_ctrl_xsub" else 0.25
    out_d = O13_ENC / model_name / f"s{seed}"
    if (out_d / "brain_model.pt").exists() and (out_d / "test_0tta_all10.pt").exists():
        print(f"[CACHED_TRAIN] {model_name} seed={seed}", flush=True)
        return
    out_d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    concept = torch.from_numpy(np.load(PREP / "train_concept.npy")).long().to(dev)
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float().to(dev)
    F_sem = torch.load(PREP / "F_sem_train.pt", map_location="cpu", weights_only=True)["F_sem"].float().to(dev)
    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    H14_tr = emb_tr[EMB_KEYS[0]].float().to(dev)
    B32_tr = emb_tr[EMB_KEYS[1]].float().to(dev)
    VAE_tr = emb_tr[EMB_KEYS[2]].float().to(dev)
    h14_norm = float(H14_tr.norm(dim=1).mean())
    b32_norm = float(B32_tr.norm(dim=1).mean())
    n_tr = len(F_tr)

    # Load all 10 subjects' training EEG strictly (fp16 on CPU = ~5.6 GB RAM)
    x_tr_all = []
    for s in ALL_SUBS:
        x_tr_all.append(torch.from_numpy(np.load(PREP / f"sub-{s:02d}_train.npy")).half())

    kv_full = OnConeKVProjector(str(PREP / "unet_kv_metric.pt"), device=dev)
    kv_full.calibrate_fold(F_tr)
    kv_sem = OnConeKVProjector(str(PREP / "unet_kv_metric.pt"), device=dev)
    kv_sem.calibrate_fold(F_sem)
    fe_vis = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.to(dev).eval().requires_grad_(False)

    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    for sid in range(10):
        model.ebs[sid].fit_from_fold(x_tr_all[sid].float().to(dev))

    ls_f = nn.Parameter(torch.tensor(2.6592, device=dev))
    ls_h = nn.Parameter(torch.tensor(2.6592, device=dev))
    ls_b = nn.Parameter(torch.tensor(2.6592, device=dev))
    ls_s = nn.Parameter(torch.tensor(2.6592, device=dev))
    ls_x = nn.Parameter(torch.tensor(2.6592, device=dev))
    extra = [("ls_f", ls_f), ("ls_h", ls_h), ("ls_b", ls_b), ("ls_s", ls_s)]
    if lam_x > 0.0:
        extra.append(("ls_x", ls_x))

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
            xa = shift_batch(xa, 2, g)
            xb = shift_batch(xb, 2, g)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                sc_f = ls_f.float().exp().clamp(max=100.0)
                sc_h = ls_h.float().exp().clamp(max=100.0)
                sc_b = ls_b.float().exp().clamp(max=100.0)
                sc_s = ls_s.float().exp().clamp(max=100.0)
                sc_x = ls_x.float().exp().clamp(max=100.0)

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
                l_vis = 0.5 * (
                    0.65 * _one_sub_loss(fa, ha, ba) + 0.35 * _br_loss(bra)
                    + 0.65 * _one_sub_loss(fb_, hb, bb) + 0.35 * _br_loss(brb)
                )

                loss = l_vis
                if lam_x > 0.0:
                    mf_all = 0.15 * sc_x.detach() * F.relu(sf_sim - sf_bar) * off_diag.float()
                    mh_all = 0.15 * sc_x.detach() * F.relu(sh - sh_bar) * off_diag.float()
                    mb_all = 0.15 * sc_x.detach() * F.relu(sb - sb_bar) * off_diag.float()
                    l_xb = (
                        0.35 * _xbrain_cone_nce(fa, fb_, kv_full.u_tr, sc_x, mf_all)
                        + 0.40 * _xbrain_nce(ha, hb, sc_x, mh_all)
                        + 0.25 * _xbrain_nce(ba, bb, sc_x, mb_all)
                    )
                    loss = loss + lam_x * l_xb

            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec + ndec, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

    train_s = time.time() - t0
    model.eval()
    # Export 0-TTA strictly single-subject test predictions for each subject independently
    pack_0tta = {"train_s": train_s}
    with torch.no_grad():
        for s in ALL_SUBS:
            sid = s - 1
            x_te = torch.from_numpy(np.load(PREP / f"sub-{s:02d}_test.npy")).float().to(dev)
            full_in = ivw_mean(x_te)
            perm_bag = torch.randperm(80, generator=torch.Generator().manual_seed(7 + s)).to(dev)
            bag_ins = torch.cat([ivw_mean(x_te[:, perm_bag[g_ * 8:(g_ + 1) * 8]]) for g_ in range(10)], dim=0)
            ff, hf, bf = model.forward_sub(full_in, sid, r_eff=72.0)
            fb, hb, bb = model.forward_sub(bag_ins, sid, r_eff=8.0)
            pack_0tta[f"sub-{s:02d}"] = {
                "f_full": ff.float().cpu(), "h_full": hf.float().cpu(), "b_full": bf.float().cpu(),
                "f_bag10": fb.float().view(10, 200, -1).cpu(),
                "h_bag10": hb.float().view(10, 200, -1).cpu(),
                "b_bag10": bb.float().view(10, 200, -1).cpu(),
            }
            del x_te, full_in, bag_ins

    tmp_m = out_d / f".brain_model.{os.getpid()}.tmp"
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, tmp_m)
    tmp_m.replace(out_d / "brain_model.pt")
    tmp_0 = out_d / f".test_0tta.{os.getpid()}.tmp"
    torch.save(pack_0tta, tmp_0)
    tmp_0.replace(out_d / "test_0tta_all10.pt")
    print(f"[O13_TRAIN] {model_name:<15} seed={seed} ep={epochs} done in {train_s:.1f}s", flush=True)
    del model, x_tr_all, kv_full, kv_sem, fe_vis, F_tr, F_sem, H14_tr, B32_tr, VAE_tr
    gc.collect()
    torch.cuda.empty_cache()


def run_single_sub_tta(sub: int, model_name: str, dev: str = "cuda:0", steps: int = 50) -> dict:
    """100% STRICTLY SINGLE-SUBJECT split-half kvTTA (updates ONLY sub_stems[sid] on sub-XX_test.npy!)."""
    cache_p = O13_ENC / model_name / f"sub-{sub:02d}_test_agg.pt"
    if cache_p.exists():
        return torch.load(cache_p, map_location="cpu", weights_only=True)

    p0_list = [torch.load(O13_ENC / model_name / f"s{sd}" / "test_0tta_all10.pt", map_location="cpu", weights_only=True)[f"sub-{sub:02d}"] for sd in SEEDS_5]
    f_0tta = 0.85 * torch.stack([p["f_full"] for p in p0_list], 0).mean(0) + 0.15 * torch.stack([p["f_bag10"] for p in p0_list], 0).mean(0).mean(0)
    h_0tta = 0.85 * torch.stack([p["h_full"] for p in p0_list], 0).mean(0) + 0.15 * torch.stack([p["h_bag10"] for p in p0_list], 0).mean(0).mean(0)
    b_0tta = 0.85 * torch.stack([p["b_full"] for p in p0_list], 0).mean(0) + 0.15 * torch.stack([p["b_bag10"] for p in p0_list], 0).mean(0).mean(0)

    ok_mask = torch.from_numpy(np.load(PREP / "lat512_train_ok.npy")).bool()
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    _, _, _, u_f = _cone_params(F_tr[ok_mask])
    u_f = u_f.to(dev)
    del F_tr

    t0 = time.time()
    sid = sub - 1
    x_te = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_test.npy")).float().to(dev)
    full_in = ivw_mean(x_te)
    perm_bag = torch.randperm(80, generator=torch.Generator().manual_seed(7 + sub)).to(dev)
    bag_ins = torch.cat([ivw_mean(x_te[:, perm_bag[g_ * 8:(g_ + 1) * 8]]) for g_ in range(10)], dim=0)
    tgt = torch.arange(200, device=dev)

    f_tta_l, h_tta_l, b_tta_l = [], [], []
    for sd in SEEDS_5:
        ckpt = torch.load(O13_ENC / model_name / f"s{sd}" / "brain_model.pt", map_location="cpu", weights_only=True)
        m = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.0).to(dev)
        m.load_state_dict(ckpt)
        m.eval().requires_grad_(False)
        for param in m.sub_stems[sid].parameters():
            param.requires_grad_(True)
        opt = torch.optim.Adam(m.sub_stems[sid].parameters(), lr=1e-4)
        g = torch.Generator(device=dev).manual_seed(5000 + sid * 41 + sd)

        with torch.no_grad():
            f0, h0, b0 = m.forward_sub(full_in, sid, r_eff=72.0)
            f0, h0, b0 = f0.float(), h0.float(), b0.float()

        for _ in range(steps):
            perm = torch.rand(200, 80, device=dev, generator=g).argsort(1)
            xa = ivw_mean(torch.gather(x_te, 1, perm[:, :40, None, None].expand(-1, -1, 17, 250)))
            xb = ivw_mean(torch.gather(x_te, 1, perm[:, 40:, None, None].expand(-1, -1, 17, 250)))
            fa, ha, ba = m.forward_sub(xa, sid, r_eff=36.0)
            fb, hb, bb = m.forward_sub(xb, sid, r_eff=36.0)

            dfa, dfb = _perp_dir(fa.float(), u_f), _perp_dir(fb.float(), u_f)
            ha_c = nz(nz(ha.float(), dim=1) - nz(ha.float(), dim=1).mean(0, keepdim=True), dim=1)
            hb_c = nz(nz(hb.float(), dim=1) - nz(hb.float(), dim=1).mean(0, keepdim=True), dim=1)
            ba_c = nz(nz(ba.float(), dim=1) - nz(ba.float(), dim=1).mean(0, keepdim=True), dim=1)
            bb_c = nz(nz(bb.float(), dim=1) - nz(bb.float(), dim=1).mean(0, keepdim=True), dim=1)
            lg = (0.40 * (dfa @ dfb.T) + 0.40 * (ha_c @ hb_c.T) + 0.20 * (ba_c @ bb_c.T)) / 0.1
            loss_nce = 0.5 * (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.T, tgt))

            ff, hf, bf = m.forward_sub(full_in, sid, r_eff=72.0)
            anc = (
                ((ff.float() - f0).pow(2).sum(1) / f0.pow(2).sum(1)).mean()
                + ((hf.float() - h0).pow(2).sum(1) / h0.pow(2).sum(1)).mean()
                + ((bf.float() - b0).pow(2).sum(1) / b0.pow(2).sum(1)).mean()
            ) / 3.0
            (loss_nce + 1.0 * anc).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            ff, hf, bf = m.forward_sub(full_in, sid, r_eff=72.0)
            fb_, hb_, bb_ = m.forward_sub(bag_ins, sid, r_eff=8.0)
            f_tta_l.append((0.70 * ff.float() + 0.30 * fb_.float().view(10, 200, -1).mean(0)).cpu())
            h_tta_l.append((0.70 * hf.float() + 0.30 * hb_.float().view(10, 200, -1).mean(0)).cpu())
            b_tta_l.append((0.70 * bf.float() + 0.30 * bb_.float().view(10, 200, -1).mean(0)).cpu())
        del m

    pack = {
        "f_0tta": f_0tta, "h_0tta": h_0tta, "b_0tta": b_0tta,
        "f_tta": torch.stack(f_tta_l, 0).mean(0),
        "h_tta": torch.stack(h_tta_l, 0).mean(0),
        "b_tta": torch.stack(b_tta_l, 0).mean(0),
        "tta_s": time.time() - t0,
    }
    tmp_p = cache_p.with_name(f".{cache_p.name}.{os.getpid()}.tmp")
    torch.save(pack, tmp_p)
    tmp_p.replace(cache_p)
    print(f"[O13_TTA] s{sub:02d} {model_name:<15} done in {pack['tta_s']:.1f}s", flush=True)
    del x_te, full_in, bag_ins
    gc.collect()
    torch.cuda.empty_cache()
    return pack


def export_o13_conditions_and_latents(subs, dev: str = "cuda:0"):
    """Build official O13 cones (strictly single-sub + Cohort-10 diagnostic), print test retrieval audit, and run lat512_v2."""
    ok_mask = torch.from_numpy(np.load(PREP / "lat512_train_ok.npy")).bool()
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    R_f, a_f, b_f, u_f = _cone_params(F_tr[ok_mask])

    F_te = torch.from_numpy(np.load(PREP / "F_test.npy")).float()
    d_F_te = _perp_dir(F_te, u_f)
    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    emb_te = torch.load(PREP / "emb_test.pt", map_location="cpu", weights_only=True)
    H14_te_u = nz(emb_te[EMB_KEYS[0]].float() - emb_te[EMB_KEYS[0]].float().mean(0, keepdim=True), dim=1)
    sim_te = H14_te_u @ H14_te_u.T
    sim_te.fill_diagonal_(-1e9)
    hard9 = sim_te.topk(9, dim=1).indices

    H14_tr = emb_tr[EMB_KEYS[0]][ok_mask].float()
    B32_tr = emb_tr[EMB_KEYS[1]][ok_mask].float()
    VAE_tr = emb_tr[EMB_KEYS[2]][ok_mask].float().reshape(int(ok_mask.sum()), -1)
    h14_norm = float(H14_tr.norm(dim=1).mean())
    b32_norm = float(B32_tr.norm(dim=1).mean())
    vae_mu = VAE_tr.mean(0, keepdim=True).to(dev)
    vae_sd = VAE_tr.std(0, keepdim=True).clamp_min(1e-6).to(dev)
    del F_tr, H14_tr, B32_tr, VAE_tr

    fe_vis = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.to(dev).eval().requires_grad_(False)
    O13_COND.mkdir(parents=True, exist_ok=True)

    cdirs = {
        "ctrl_sa":   RUNS / "cond" / "cone_o13_ctrl_sa_kvtta_w1.00",
        "xbrain_sa": RUNS / "cond" / "cone_o13_xbrain_sa_kvtta_w1.00",
        "ctrl_hyb":  RUNS / "cond" / "cone_o13_ctrl_kvtta_w1.00",
        "xb_0tta":   RUNS / "cond" / "cone_o13_xbrain_0tta_w1.00",
        "xb_hyb":    RUNS / "cond" / "cone_o13_xbrain_kvtta_w1.00",
    }
    for d in cdirs.values():
        d.mkdir(parents=True, exist_ok=True)

    def _ret_stats(d_hat, h_hat=None):
        f_t1 = float(((d_hat @ d_F_te.T).argmax(1) == torch.arange(200)).float().mean())
        rho_f = float((d_hat * d_F_te).sum(1).mean())
        if h_hat is None:
            return {"F_200w": f_t1, "rho_F": rho_f, "H14_200w": float("nan"), "Hard10": float("nan")}
        hu = nz(h_hat.float() - h_hat.float().mean(0, keepdim=True), dim=1)
        S_h = hu @ H14_te_u.T
        h_t1 = float((S_h.argmax(1) == torch.arange(200)).float().mean())
        cands = torch.cat([torch.arange(200)[:, None], hard9], dim=1)
        hard10 = float((S_h.gather(1, cands).argmax(1) == 0).float().mean())
        return {"F_200w": f_t1, "rho_F": rho_f, "H14_200w": h_t1, "Hard10": hard10}

    ret_audit = {}
    d_coh_sa_list, d_coh_hyb_list, h_coh_list = [], [], []

    for s in subs:
        pc = torch.load(O13_ENC / "o13_ctrl_xsub" / f"sub-{s:02d}_test_agg.pt", map_location="cpu", weights_only=True)
        px = torch.load(O13_ENC / "o13_xbrain" / f"sub-{s:02d}_test_agg.pt", map_location="cpu", weights_only=True)
        z0_o10_t = torch.from_numpy(np.load(RUNS / "low" / f"sub-{s:02d}" / "lat512_o10_bifurc_v2_n0.80.npy")).float()
        z0_o10_0 = torch.from_numpy(np.load(RUNS / "low" / f"sub-{s:02d}" / "lat512_o10_bifurc_0tta_n0.80.npy")).float()
        d_o10_t = _perp_dir(torch.from_numpy(np.load(RUNS / "cond" / "cone_o10_bifurc_kvtta_w1.00" / f"sub-{s:02d}.npy")).float(), u_f)
        d_o10_0 = _perp_dir(torch.from_numpy(np.load(RUNS / "cond" / "cone_o10_bifurc_0tta_w1.00" / f"sub-{s:02d}.npy")).float(), u_f)

        def _synth(pack, fk, hk, bk, z0_in):
            z16 = F.avg_pool2d(z0_in.to(dev), 4).reshape(200, -1)
            vae_hat = vae_mu + 0.65 * (z16 - z16.mean(0, keepdim=True)) * (vae_sd / z16.std(0, keepdim=True).clamp_min(1e-6))
            h_v = nz(pack[hk].float(), dim=1) * h14_norm
            b_v = nz(pack[bk].float(), dim=1) * b32_norm
            with torch.no_grad():
                f_syn = fe_vis(**{EMB_KEYS[0]: h_v.to(dev), EMB_KEYS[1]: b_v.to(dev), EMB_KEYS[2]: vae_hat}).float().cpu()
            d_f = _perp_dir(pack[fk].float(), u_f)
            d_s = _perp_dir(f_syn, u_f)
            return nz(0.50 * d_f + 0.50 * d_s, dim=1)

        d_ctrl_sa = _synth(pc, "f_tta", "h_tta", "b_tta", z0_o10_t)
        d_xb_sa = _synth(px, "f_tta", "h_tta", "b_tta", z0_o10_t)
        d_xb_0 = _synth(px, "f_0tta", "h_0tta", "b_0tta", z0_o10_0)

        d_ctrl_hyb = nz(0.50 * d_o10_t + 0.50 * d_ctrl_sa, dim=1)
        d_xb_0hyb = nz(0.50 * d_o10_0 + 0.50 * d_xb_0, dim=1)
        d_xb_hyb = nz(0.50 * d_o10_t + 0.50 * d_xb_sa, dim=1)

        cones = {
            "ctrl_sa":   _to_cone(d_ctrl_sa, R_f, a_f, b_f, u_f),
            "xbrain_sa": _to_cone(d_xb_sa, R_f, a_f, b_f, u_f),
            "ctrl_hyb":  _to_cone(d_ctrl_hyb, R_f, a_f, b_f, u_f),
            "xb_0tta":   _to_cone(d_xb_0hyb, R_f, a_f, b_f, u_f),
            "xb_hyb":    _to_cone(d_xb_hyb, R_f, a_f, b_f, u_f),
        }
        for k_c, c_t in cones.items():
            np.save(cdirs[k_c] / f"sub-{s:02d}.npy", c_t.numpy().astype(np.float32))
        torch.save(cones, O13_COND / f"sub-{s:02d}_o13_pack.pt")

        d_coh_sa_list.append(d_xb_sa)
        d_coh_hyb_list.append(d_xb_hyb)
        h_coh_list.append(nz(px["h_tta"].float() - px["h_tta"].float().mean(0, keepdim=True), dim=1))

        ret_audit[f"sub-{s:02d}"] = {
            "O10":              _ret_stats(d_o10_t, None),
            "O13_ctrl_sa":      _ret_stats(d_ctrl_sa, pc["h_tta"]),
            "O13_xbrain_sa":    _ret_stats(d_xb_sa, px["h_tta"]),
            "O13_ctrl_xsub":    _ret_stats(d_ctrl_hyb, pc["h_tta"]),
            "O13_0tta":         _ret_stats(d_xb_0hyb, px["h_0tta"]),
            "O13":              _ret_stats(d_xb_hyb, px["h_tta"]),
        }
        ra = ret_audit[f"sub-{s:02d}"]
        print(
            f"[O13_RET] s{s:02d} | O10 F={ra['O10']['F_200w']:.3f}({ra['O10']['rho_F']:.3f}) | "
            f"ctrl_sa F={ra['O13_ctrl_sa']['F_200w']:.3f} H14={ra['O13_ctrl_sa']['H14_200w']:.3f} | "
            f"xb_sa F={ra['O13_xbrain_sa']['F_200w']:.3f} H14={ra['O13_xbrain_sa']['H14_200w']:.3f} | "
            f"ctrl F={ra['O13_ctrl_xsub']['F_200w']:.3f}({ra['O13_ctrl_xsub']['rho_F']:.3f}) | "
            f"O13 F={ra['O13']['F_200w']:.3f}({ra['O13']['rho_F']:.3f})",
            flush=True,
        )

    # Diagnostic Cohort-10 Ceiling (explicitly non-single-subject; ZERO test labels/images)
    if len(subs) == 10:
        d_coh = nz(torch.stack(d_coh_hyb_list, 0).mean(0), dim=1)
        h_coh = nz(torch.stack(h_coh_list, 0).mean(0), dim=1)
        c_coh = _to_cone(d_coh, R_f, a_f, b_f, u_f)
        torch.save({"c_coh": c_coh}, O13_COND / "cohort10_o13_pack.pt")
        ret_audit["DIAG_COHORT10"] = _ret_stats(d_coh, h_coh)
        rc = ret_audit["DIAG_COHORT10"]
        print(f"\n[O13_RET] DIAG_COHORT10 | F={rc['F_200w']:.4f} rhoF={rc['rho_F']:.4f} H14={rc['H14_200w']:.4f} Hard10={rc['Hard10']:.4f}", flush=True)

        print("\n=== O13 OFFICIAL 10-SUBJECT RETRIEVAL MEAN (STRICTLY SINGLE-SUBJECT TEST) ===", flush=True)
        for tag in ("O10", "O13_ctrl_sa", "O13_xbrain_sa", "O13_ctrl_xsub", "O13_0tta", "O13"):
            vs = [ret_audit[f"sub-{s:02d}"][tag] for s in ALL_SUBS]
            ret_audit[f"mean10_{tag}"] = {k: float(np.mean([v[k] for v in vs])) for k in vs[0]}
            m = ret_audit[f"mean10_{tag}"]
            print(f"  {tag:<15}: F_200w={m['F_200w']:.4f} rho_F={m['rho_F']:.4f} H14_200w={m['H14_200w']:.4f} Hard10={m['Hard10']:.4f}", flush=True)
        write_json(O13_ROOT / "official_retrieval_audit.json", ret_audit)

    del fe_vis
    gc.collect()
    torch.cuda.empty_cache()

    # Build fresh lat512_v2 start latents via unchanged lat512_start.py for each cone!
    lat_script = str(_find_script("lat512_start.py"))
    sub_csv = "all" if len(subs) == 10 else ",".join(map(str, subs))
    for cdir_name, ltag in (
        ("cone_o13_ctrl_sa_kvtta_w1.00",   "lat512_o13_ctrl_sa_v2"),
        ("cone_o13_xbrain_sa_kvtta_w1.00", "lat512_o13_xbrain_sa_v2"),
        ("cone_o13_ctrl_kvtta_w1.00",      "lat512_o13_ctrl_v2"),
        ("cone_o13_xbrain_0tta_w1.00",     "lat512_o13_xbrain_0tta"),
        ("cone_o13_xbrain_kvtta_w1.00",    "lat512_o13_xbrain_v2"),
    ):
        if not all((RUNS / "low" / f"sub-{s:02d}" / f"{ltag}_n0.80.npy").exists() for s in subs):
            subprocess.run([sys.executable, "-u", lat_script, sub_csv, "--mode_v", "v2", "--cond_dirs", cdir_name, "--tag", ltag], check=True)
    print(f"[O13_COND] Exported all O13 cones & lat512_v2 for {len(subs)} subjects.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "train", "tta", "cond", "eval"], default="all")
    ap.add_argument("--models", default="o13_ctrl_xsub,o13_xbrain")
    ap.add_argument("--seeds", default="2025,2026,2027,2028,2029")
    ap.add_argument("--subs", default="all")
    ap.add_argument("--arms", default="O13_ctrl_standalone,O13_xbrain_standalone,O13_ctrl_xsub,O13_0tta,O13,O13_diag_cohort10")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--dev", default="cuda:0")
    args = ap.parse_args()

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    subs = list(ALL_SUBS) if args.subs == "all" else [int(x) for x in args.subs.split(",")]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    dev = args.dev

    if args.phase in ("all", "train"):
        for mn in models:
            for sd in seeds:
                train_one_xsub_seed(mn, sd, dev=dev, epochs=args.epochs)
        if args.phase == "train":
            return

    if args.phase in ("all", "tta"):
        for mn in models:
            for s in subs:
                run_single_sub_tta(s, mn, dev=dev, steps=50)
        if args.phase == "tta":
            return

    if args.phase in ("all", "cond"):
        export_o13_conditions_and_latents(subs, dev=dev)
        if args.phase == "cond":
            return

    O13_EVAL.mkdir(parents=True, exist_ok=True)
    runner = OfficialTurboRunner(dev=dev)

    arm_map = {
        "O13_ctrl_standalone":   ("ctrl_sa",   "lat512_o13_ctrl_sa_v2",   False),
        "O13_xbrain_standalone": ("xbrain_sa", "lat512_o13_xbrain_sa_v2", False),
        "O13_ctrl_xsub":         ("ctrl_hyb",  "lat512_o13_ctrl_v2",      True),
        "O13_0tta":              ("xb_0tta",   "lat512_o13_xbrain_0tta",  False),
        "O13":                   ("xb_hyb",    "lat512_o13_xbrain_v2",    True),
    }

    for aname in arms:
        if aname == "O13_diag_cohort10":
            out_j = O13_EVAL / "DIAG_COHORT10_O13.json"
            audit_p = O13_AUDIT / "DIAG_COHORT10_O13_audit.pt"
            if out_j.exists() and audit_p.exists():
                m = json.loads(out_j.read_text())
                print(f"[CACHED] DIAG_COHORT10 | top1={m['clip_top1']:.4f} 2w={m['clip']:.4f} SSIM={m['ssim']:.4f} Inc={m['inception']:.4f}", flush=True)
                continue
            c_coh = torch.load(O13_COND / "cohort10_o13_pack.pt", map_location="cpu", weights_only=True)["c_coh"]
            # Average the 10 subjects' lat512_o13_xbrain_v2 start latents (ZERO test labels/images)
            z0_coh = torch.stack([
                torch.from_numpy(np.load(RUNS / "low" / f"sub-{s:02d}" / "lat512_o13_xbrain_v2_n0.80.npy")).float()
                for s in ALL_SUBS
            ], dim=0).mean(0)
            m = runner.evaluate_arm(c_coh, z0_coh, K=4, bs=args.bs, save_audit_path=audit_p)
            write_json(out_j, m)
            print(
                f"[O13_DIAG_COHORT10] top1={m['clip_top1']:.4f} 2w={m['clip']:.4f} "
                f"SSIM={m['ssim']:.4f} PixC={m['pixcorr']:.4f} Inc={m['inception']:.4f} Eff={m['effnet']:.4f} ({m['gen_wall_s']:.1f}s)",
                flush=True,
            )
            continue

        ckey, ltag, save_aud = arm_map[aname]
        for s in subs:
            out_j = O13_EVAL / f"s{s:02d}_{aname}.json"
            audit_p = (O13_AUDIT / f"sub-{s:02d}_{aname}_audit.pt") if save_aud else None
            if out_j.exists() and (audit_p is None or audit_p.exists()):
                m = json.loads(out_j.read_text())
                print(f"[CACHED] {aname:<22} s{s:02d} | top1={m['clip_top1']:.4f} 2w={m['clip']:.4f} SSIM={m['ssim']:.4f} Inc={m['inception']:.4f}", flush=True)
                continue

            c_use = torch.load(O13_COND / f"sub-{s:02d}_o13_pack.pt", map_location="cpu", weights_only=True)[ckey]
            z0 = torch.from_numpy(np.load(RUNS / "low" / f"sub-{s:02d}" / f"{ltag}_n0.80.npy")).float()
            m = runner.evaluate_arm(c_use, z0, K=4, bs=args.bs, save_audit_path=audit_p)
            # Keep disk lean: drop raw k0 RGB pixels from ctrl_xsub audit cache (keep ranks + top1_idx + metrics!)
            if aname == "O13_ctrl_xsub" and audit_p is not None and audit_p.exists():
                ap_d = torch.load(audit_p, map_location="cpu", weights_only=True)
                ap_d.pop("k0_rgb_u8", None)
                torch.save(ap_d, audit_p)
            write_json(out_j, m)
            print(
                f"[O13_OFFICIAL] {aname:<22} s{s:02d} | top1={m['clip_top1']:.4f} 2w={m['clip']:.4f} "
                f"SSIM={m['ssim']:.4f} PixC={m['pixcorr']:.4f} Inc={m['inception']:.4f} Eff={m['effnet']:.4f} ({m['gen_wall_s']:.1f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
