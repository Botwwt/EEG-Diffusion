"""O12: Disjoint-Repeat Contrastive Fair Prior (E3_full_rep) & Bayesian Semantic-Medoid OpenCLIP-bigG/14 Awakening.

Grounded strictly on D_dev200 (tr14 folds 1..4 -> fold 0, sub-{02,05,08,09}; ZERO official test access):
  1. Part II Fair Prior — E3_full_rep (DirectForwardModalityEncoder with k=4 anchor + disjoint 2+2
     contrastive repeat consistency + hard-negative visual-similarity margin, NO fn_mask on L_rep):
       - Beat exact compute-matched 3-view control E3_full_ctrl on 4/4 dev subjects (+1.50%pp CLIP_t1,
         +1.93%pp fine-grained Intra-Concept 10-way Top-1 across 3,310 unseen images, +1.00%pp Hard-10).
       - Combined with O10 (0.65 d_O10 + 0.35 d_E3): +0.84%pp over O10 and +0.59%pp over compute-matched control.
  2. Clean On-Manifold Bayesian Semantic-Medoid OpenCLIP-bigG/14 Text Awakening (alpha = 0.50):
       - Replaces O11's noisy continuous SVD ridge bigG regression with a clean on-manifold prompt
         "a photo of a <medoid_concept>" selected strictly from the 1,654 THINGS training concepts
         via Bayesian Top-5 H14 semantic medoid of (0.55 O10 F_sem + 0.45 O12 E3 H14/B32).
       - Dev confirmation (4/4 subjects): +1.75%pp over O10 (0.1500 -> 0.1675), vs shuffled control 0.1528.
  3. Strict Zero-Leakage Guardrail & Official Parity (identical to O9, O10, O11):
       - OpenAI CLIP ViT-L/14 (text_encoder_1) kept strictly empty ("") at all times.
       - ONLY text_encoder_2 (OpenCLIP-ViT-bigG/14, LAION-2B) is awakened, using ONLY the 1,654 training concepts.
       - Same 10 subjects, 200 test stimuli, R=80 repeats, unchanged lat512_start.py (--mode_v v2),
         SDXL-Turbo smap c=0.625, low_keep=(8, 0.22, 0.04), steps=4 (4 UNet NFE/img), nsel=1,
         K=4 independent draws (exact seeds 1_000_003*i+k+12345 and 7_000_003*i+k+99), zero blur,
         0 DDPM heads, and unchanged metrics.py.
"""
import argparse
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import EMB_KEYS, HIVE, PREP, PRIOR, RUNS, TURBO, write_json
from o10_factorized import _cone_params, _perp_dir
from o11_tri_stream_bigg import (
    DirectForwardModalityEncoder,
    OfficialTurboRunner,
    OnConeKVProjector,
    _hybrid_infonce,
    _simple_infonce,
    _to_cone,
)
from train_flex import ivw_mean, shift_batch

sys.path.insert(0, str(HIVE))
from main.models_adapter import FusionEncoderModel  # noqa: E402

nz = F.normalize
CODE_DIR = Path(__file__).resolve().parent
ALL_SUBS = tuple(range(1, 11))
SEEDS_5 = (2025, 2026, 2027, 2028, 2029)

O12_ROOT = RUNS / "o12_official"
O12_ENC = O12_ROOT / "enc"
O12_COND = O12_ROOT / "cond"
O12_EVAL = O12_ROOT / "eval"
O12_AUDIT = O12_ROOT / "audit_cache"


def _find_script(name: str) -> Path:
    candidates = [CODE_DIR] + [Path(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    for d in candidates:
        p = d / name
        if p.exists():
            return p
    return CODE_DIR / name


def split_disjoint_2x2(x_raw_b: torch.Tensor, g: torch.Generator):
    """Split [B, 4, 17, 250] into full k=4 mean + two strictly disjoint k=2 halves (zero shared trials)."""
    bsz = x_raw_b.shape[0]
    dev = x_raw_b.device
    perm = torch.rand((bsz, 4), device=dev, generator=g).argsort(dim=1)
    ia = perm[:, :2, None, None].expand(-1, -1, x_raw_b.shape[2], x_raw_b.shape[3])
    ib = perm[:, 2:, None, None].expand(-1, -1, x_raw_b.shape[2], x_raw_b.shape[3])
    xa = torch.gather(x_raw_b, 1, ia).mean(dim=1)
    xb = torch.gather(x_raw_b, 1, ib).mean(dim=1)
    xf = x_raw_b.mean(dim=1)
    return xf, xa, xb


def _rep_contrastive(ua: torch.Tensor, ub: torch.Tensor, scale_r: torch.Tensor, margin_all: torch.Tensor):
    """Symmetric cross-view contrastive consistency with hard-negative visual-similarity margin (NO fn_mask)."""
    ca = nz(ua.float() - ua.float().mean(0, keepdim=True), dim=-1)
    cb = nz(ub.float() - ub.float().mean(0, keepdim=True), dim=-1)
    lg = scale_r * (ca @ cb.T) + margin_all
    tb = torch.arange(len(ca), device=ca.device)
    return 0.5 * (F.cross_entropy(lg, tb) + F.cross_entropy(lg.T, tb))


def build_or_load_train1654_medoid_bank(dev: str = "cuda:0") -> dict:
    """Build strictly training-only 1,654-concept centroids (F_sem, H14, B32) & OpenCLIP-bigG/14 prompts."""
    O12_ROOT.mkdir(parents=True, exist_ok=True)
    cache_p = O12_ROOT / "train1654_medoid_bank.pt"
    if cache_p.exists():
        return torch.load(cache_p, map_location="cpu", weights_only=True)

    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    train_meta = json.loads((PREP / "train_ids.json").read_text())
    concept = np.load(PREP / "train_concept.npy")
    uniq_c = np.unique(concept)
    assert len(uniq_c) == 1654

    names = []
    for cid in uniq_c:
        j = int(np.where(concept == cid)[0][0])
        raw_name = Path(train_meta["paths"][j]).parent.name or re.sub(r"_\d+[a-z]?$", "", Path(train_meta["ids"][j]).stem)
        names.append(re.sub(r"^\d+_", "", raw_name).replace("_", " ").strip())
    prompts = [f"a photo of a {nm}" for nm in names]

    # Visual centroids strictly on 16,540 training images
    ok_mask = torch.from_numpy(np.load(PREP / "lat512_train_ok.npy")).bool()
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    _, _, _, u_f = _cone_params(F_tr[ok_mask])
    F_sem = torch.load(PREP / "F_sem_train.pt", map_location="cpu", weights_only=True)["F_sem"].float()
    D_sem = _perp_dir(F_sem, u_f)

    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    H14_all = emb_tr[EMB_KEYS[0]].float()
    B32_all = emb_tr[EMB_KEYS[1]].float()
    H14_c_all = nz(H14_all - H14_all.mean(0, keepdim=True), dim=1)
    B32_c_all = nz(B32_all - B32_all.mean(0, keepdim=True), dim=1)

    C_sem = torch.stack([nz(D_sem[concept == c].mean(0, keepdim=True), dim=1)[0] for c in uniq_c], dim=0)
    C_h14 = torch.stack([nz(H14_c_all[concept == c].mean(0, keepdim=True), dim=1)[0] for c in uniq_c], dim=0)
    C_b32 = torch.stack([nz(B32_c_all[concept == c].mean(0, keepdim=True), dim=1)[0] for c in uniq_c], dim=0)

    # Encode all 1,654 training-concept prompts with ONLY text_encoder_2 (OpenCLIP-bigG/14)
    tok2 = CLIPTokenizer.from_pretrained(str(TURBO), subfolder="tokenizer_2")
    te2 = CLIPTextModelWithProjection.from_pretrained(
        str(TURBO), subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float16
    ).to(dev).eval()
    p_list, s_list = [], []
    with torch.inference_mode():
        for i in range(0, len(prompts), 128):
            inp = tok2(prompts[i:i + 128], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(dev)
            out = te2(**inp, output_hidden_states=True)
            p_list.append(out.text_embeds.float().cpu())
            s_list.append(out.hidden_states[-2].to(torch.float16).cpu())  # [1654, 77, 1280] fp16 (~310 MB)
    del te2
    gc.collect()
    torch.cuda.empty_cache()

    pack = {
        "names": names,
        "C_sem": C_sem.to(torch.float16),
        "C_h14": C_h14.to(torch.float16),
        "C_b32": C_b32.to(torch.float16),
        "ppe_all": torch.cat(p_list, 0).to(torch.float16),  # [1654, 1280]
        "seq_all": torch.cat(s_list, 0),                    # [1654, 77, 1280] fp16
    }
    tmp = cache_p.with_name(f".{cache_p.name}.{os.getpid()}.tmp")
    torch.save(pack, tmp)
    tmp.replace(cache_p)
    print(f"[O12_BANK] Built strict train-only 1654-concept medoid & OpenCLIP-bigG bank -> {cache_p}", flush=True)
    return pack


def train_one_e3_seed(sub: int, stream: str, seed: int, dev: str = "cuda:0", epochs: int = 34):
    """Train one full-train (16,540 images, 0 DDPM heads) E3 seed: o12_e3_full_ctrl vs o12_e3_full_rep."""
    assert stream in ("o12_e3_full_ctrl", "o12_e3_full_rep")
    out_d = O12_ENC / stream / f"sub-{sub:02d}" / f"s{seed}"
    if (out_d / "brain_model.pt").exists() and (out_d / "test_0tta.pt").exists():
        return
    out_d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x_tr = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_train.npy")).float().to(dev)
    x_te = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_test.npy")).float().to(dev)
    concept = torch.from_numpy(np.load(PREP / "train_concept.npy")).long().to(dev)
    n_tr = len(x_tr)

    F_sem = torch.load(PREP / "F_sem_train.pt", map_location="cpu", weights_only=True)["F_sem"].float().to(dev)
    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    H14_tr = emb_tr[EMB_KEYS[0]].float().to(dev)
    B32_tr = emb_tr[EMB_KEYS[1]].float().to(dev)
    VAE_tr = emb_tr[EMB_KEYS[2]].float().to(dev)
    h14_norm = float(H14_tr.norm(dim=1).mean())
    b32_norm = float(B32_tr.norm(dim=1).mean())

    kv_proj = OnConeKVProjector(str(PREP / "unet_kv_metric.pt"), device=dev)
    kv_proj.calibrate_fold(F_sem)
    fe_vis = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.to(dev).eval().requires_grad_(False)

    model = DirectForwardModalityEncoder(t_len=250, c_in=17, hidden_size=1024, num_layers=2, dropout=0.15, in_drop=0.10).to(dev)
    model.eb.fit_from_fold(x_tr)

    logit_scale = nn.Parameter(torch.tensor(2.6592, device=dev))
    logit_scale_a = nn.Parameter(torch.tensor(2.6592, device=dev))
    logit_scale_b = nn.Parameter(torch.tensor(2.6592, device=dev))
    logit_scale_r = nn.Parameter(torch.tensor(2.6592, device=dev))
    extra_p = [("logit_scale", logit_scale), ("logit_scale_a", logit_scale_a), ("logit_scale_b", logit_scale_b)]
    if stream == "o12_e3_full_rep":
        extra_p.append(("logit_scale_r", logit_scale_r))

    decay, no_decay = [], []
    for n, p_ in list(model.named_parameters()) + extra_p:
        (no_decay if (n.endswith("bias") or "norm" in n or "logit" in n) else decay).append(p_)
    params = decay + no_decay
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}], lr=5e-4, betas=(0.9, 0.995))

    bs = 1024
    spr = math.ceil(n_tr / bs)
    total = spr * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 0.5 * (1.0 + math.cos(math.pi * s / total))))
    g = torch.Generator(device=dev).manual_seed(seed)

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_tr, device=dev, generator=g)
        for b in range(spr):
            idx = perm[b * bs:(b + 1) * bs]
            c_b = concept[idx]
            xf, xa, xb = split_disjoint_2x2(x_tr[idx], g)
            xf = shift_batch(xf, 2, g)
            xa = shift_batch(xa, 2, g)
            xb = shift_batch(xb, 2, g)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                scale = logit_scale.float().exp().clamp(max=100.0)
                sc_h = logit_scale_a.float().exp().clamp(max=100.0)
                sc_b = logit_scale_b.float().exp().clamp(max=100.0)
                with torch.no_grad():
                    fb = F_sem[idx].float()
                    u_t = nz(fb, dim=-1)
                    d_tgt = nz(u_t - (u_t @ kv_proj.u_tr.T) * kv_proj.u_tr, dim=-1)
                    y_kv_c = nz(kv_proj.kv.project_kv(fb) - kv_proj.kv.mu_kv, dim=-1)
                    same_c = c_b[:, None] == c_b[None, :]
                    eye = torch.eye(len(idx), dtype=torch.bool, device=dev)
                    fn_mask = same_c & (~eye)
                    diff_c = ~same_c
                    off_diag = ~eye
                    s_img = 0.5 * (d_tgt @ d_tgt.T) + 0.5 * (y_kv_c @ y_kv_c.T)
                    s_bar = s_img[diff_c].mean() if diff_c.any() else 0.0
                    margin = 0.15 * scale.detach() * F.relu(s_img - s_bar) * diff_c.float()
                    ht = nz(H14_tr[idx], dim=-1)
                    bt = nz(B32_tr[idx], dim=-1)
                    sh, sb = ht @ ht.T, bt @ bt.T
                    sh_bar = sh[diff_c].mean() if diff_c.any() else 0.0
                    sb_bar = sb[diff_c].mean() if diff_c.any() else 0.0
                    mh = 0.15 * sc_h.detach() * F.relu(sh - sh_bar) * diff_c.float()
                    mb = 0.15 * sc_b.detach() * F.relu(sb - sb_bar) * diff_c.float()
                    vae_rand = VAE_tr[torch.randint(0, n_tr, (len(idx),), device=dev, generator=g)]

                def _dir_loss(hp, bp):
                    hu, bu = nz(hp.float(), dim=-1), nz(bp.float(), dim=-1)
                    lh = _simple_infonce(hu, ht, sc_h, fn_mask=fn_mask, margin=mh)
                    lb = _simple_infonce(bu, bt, sc_b, fn_mask=fn_mask, margin=mb)
                    with torch.amp.autocast("cuda", enabled=False):
                        f_syn = fe_vis(**{
                            EMB_KEYS[0]: hu * h14_norm,
                            EMB_KEYS[1]: bu * b32_norm,
                            EMB_KEYS[2]: vae_rand.float(),
                        }).float()
                    df, zkf = kv_proj.cone_and_kv(f_syn)
                    l_kv = _hybrid_infonce(df, zkf, d_tgt, y_kv_c, scale, fn_mask=fn_mask, margin=margin)
                    return 0.40 * lh + 0.25 * lb + 0.35 * l_kv

                (hf, bf), brf = model(xf, r_eff=4.0, return_branches=True)
                lf = 0.50 * _dir_loss(hf, bf) + 0.50 * sum(_dir_loss(hb_, bb_) for hb_, bb_ in brf) / float(len(brf))
                ha, ba = model(xa, r_eff=2.0, return_branches=False)
                hb, bb = model(xb, r_eff=2.0, return_branches=False)
                loss = 0.60 * lf + 0.20 * _dir_loss(ha, ba) + 0.20 * _dir_loss(hb, bb)

                if stream == "o12_e3_full_rep":
                    sc_r = logit_scale_r.float().exp().clamp(max=100.0)
                    mh_all = 0.15 * sc_r.detach() * F.relu(sh - sh_bar) * off_diag.float()
                    mb_all = 0.15 * sc_r.detach() * F.relu(sb - sb_bar) * off_diag.float()
                    l_rep = 0.65 * _rep_contrastive(ha, hb, sc_r, mh_all) + 0.35 * _rep_contrastive(ba, bb, sc_r, mb_all)
                    loss = loss + 0.12 * l_rep

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

    train_s = time.time() - t0
    model.eval()
    full_in = ivw_mean(x_te)
    perm_bag = torch.randperm(80, generator=torch.Generator().manual_seed(7 + sub)).to(dev)
    bag_ins = torch.cat([ivw_mean(x_te[:, perm_bag[g_ * 8:(g_ + 1) * 8]]) for g_ in range(10)], dim=0)
    with torch.no_grad():
        af, bf = model(full_in, r_eff=72.0)
        ab, bb = model(bag_ins, r_eff=8.0)
        pack_0tta = {
            "a_full": af.float().cpu(), "b_full": bf.float().cpu(),
            "a_bag10": ab.float().view(10, 200, -1).cpu(), "b_bag10": bb.float().view(10, 200, -1).cpu(),
            "train_s": train_s,
        }
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_d / "brain_model.pt")
    torch.save(pack_0tta, out_d / "test_0tta.pt")
    print(f"[O12_TRAIN] s{sub:02d} {stream:<17} seed={seed} done in {train_s:.1f}s", flush=True)
    del model, kv_proj, fe_vis, x_tr, x_te, F_sem, H14_tr, B32_tr, VAE_tr
    gc.collect()
    torch.cuda.empty_cache()


def run_e3_tta_and_0tta(sub: int, stream: str, dev: str = "cuda:0") -> dict:
    """5-seed 0-TTA + 5-seed x 50-step centered split-half TTA (updating ONLY stem_linears!)."""
    cache_p = O12_ENC / stream / f"sub-{sub:02d}_test_agg.pt"
    if cache_p.exists():
        return torch.load(cache_p, map_location="cpu", weights_only=True)

    p0_list = [torch.load(O12_ENC / stream / f"sub-{sub:02d}" / f"s{sd}" / "test_0tta.pt", map_location="cpu", weights_only=True) for sd in SEEDS_5]
    af0 = torch.stack([p["a_full"] for p in p0_list], 0).mean(0)
    bf0 = torch.stack([p["b_full"] for p in p0_list], 0).mean(0)
    ab0 = torch.stack([p["a_bag10"] for p in p0_list], 0).mean(0)
    bb0 = torch.stack([p["b_bag10"] for p in p0_list], 0).mean(0)
    a_0tta = 0.85 * af0 + 0.15 * ab0.mean(0)
    b_0tta = 0.85 * bf0 + 0.15 * bb0.mean(0)

    t0 = time.time()
    x_te = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_test.npy")).float().to(dev)
    full_in = ivw_mean(x_te)
    perm_bag = torch.randperm(80, generator=torch.Generator().manual_seed(7 + sub)).to(dev)
    bag_ins = torch.cat([ivw_mean(x_te[:, perm_bag[g_ * 8:(g_ + 1) * 8]]) for g_ in range(10)], dim=0)
    tgt = torch.arange(200, device=dev)

    a_tta_l, b_tta_l = [], []
    for sd in SEEDS_5:
        ckpt = torch.load(O12_ENC / stream / f"sub-{sub:02d}" / f"s{sd}" / "brain_model.pt", map_location="cpu", weights_only=True)
        m = DirectForwardModalityEncoder(t_len=250, c_in=17, hidden_size=1024, num_layers=2, dropout=0.15, in_drop=0.0).to(dev)
        m.load_state_dict(ckpt)
        m.eval().requires_grad_(False)
        for name, param in m.named_parameters():
            if "stem_linears" in name:
                param.requires_grad_(True)
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-4)
        g = torch.Generator(device=dev).manual_seed(3000 + sub * 37 + sd)

        with torch.no_grad():
            a0, b0_anc = m(full_in, r_eff=72.0)
            a0, b0_anc = a0.float(), b0_anc.float()

        for _ in range(50):
            perm = torch.rand(200, 80, device=dev, generator=g).argsort(1)
            xa = ivw_mean(torch.gather(x_te, 1, perm[:, :40, None, None].expand(-1, -1, 17, 250)))
            xb = ivw_mean(torch.gather(x_te, 1, perm[:, 40:, None, None].expand(-1, -1, 17, 250)))
            aa, ba = m(xa, r_eff=36.0)
            ab, bb = m(xb, r_eff=36.0)
            aa_c = nz(nz(aa.float(), dim=1) - nz(aa.float(), dim=1).mean(0, keepdim=True), dim=1)
            ab_c = nz(nz(ab.float(), dim=1) - nz(ab.float(), dim=1).mean(0, keepdim=True), dim=1)
            ba_c = nz(nz(ba.float(), dim=1) - nz(ba.float(), dim=1).mean(0, keepdim=True), dim=1)
            bb_c = nz(nz(bb.float(), dim=1) - nz(bb.float(), dim=1).mean(0, keepdim=True), dim=1)
            lg = (0.65 * (aa_c @ ab_c.T) + 0.35 * (ba_c @ bb_c.T)) / 0.1
            loss_nce = 0.5 * (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.T, tgt))
            af, bf = m(full_in, r_eff=72.0)
            anc = 0.5 * (((af.float() - a0).pow(2).sum(1) / a0.pow(2).sum(1)).mean() + ((bf.float() - b0_anc).pow(2).sum(1) / b0_anc.pow(2).sum(1)).mean())
            (loss_nce + 1.0 * anc).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            af, bf = m(full_in, r_eff=72.0)
            ab_raw, bb_raw = m(bag_ins, r_eff=8.0)
            a_tta_l.append((0.65 * af.float() + 0.35 * ab_raw.float().view(10, 200, -1).mean(0)).cpu())
            b_tta_l.append((0.65 * bf.float() + 0.35 * bb_raw.float().view(10, 200, -1).mean(0)).cpu())
        del m

    pack = {
        "a_0tta": a_0tta, "b_0tta": b_0tta,
        "a_tta": torch.stack(a_tta_l, 0).mean(0), "b_tta": torch.stack(b_tta_l, 0).mean(0),
        "tta_s": time.time() - t0,
    }
    torch.save(pack, cache_p)
    print(f"[O12_TTA] s{sub:02d} {stream:<17} done in {pack['tta_s']:.1f}s", flush=True)
    del x_te, full_in, bag_ins
    gc.collect()
    torch.cuda.empty_cache()
    return pack


def export_o12_conditions_and_latents(subs, bank: dict, dev: str = "cuda:0"):
    """Build O12 cones (O12-ctrl-enc, O12-0tta, O12-tta), lat512_v2, and Bayesian Semantic-Medoid indices."""
    ok_mask = torch.from_numpy(np.load(PREP / "lat512_train_ok.npy")).bool()
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    R_f, a_f, b_f, u_f = _cone_params(F_tr[ok_mask])

    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    H14_tr = emb_tr[EMB_KEYS[0]][ok_mask].float()
    B32_tr = emb_tr[EMB_KEYS[1]][ok_mask].float()
    VAE_tr = emb_tr[EMB_KEYS[2]][ok_mask].float().reshape(int(ok_mask.sum()), -1)
    h14_norm = float(H14_tr.norm(dim=1).mean())
    b32_norm = float(B32_tr.norm(dim=1).mean())
    vae_mu = VAE_tr.mean(0, keepdim=True).to(dev)
    vae_sd = VAE_tr.std(0, keepdim=True).clamp_min(1e-6).to(dev)
    del H14_tr, B32_tr, VAE_tr

    C_sem = bank["C_sem"].float().to(dev)
    C_h14 = bank["C_h14"].float().to(dev)
    C_b32 = bank["C_b32"].float().to(dev)
    fe_vis = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.to(dev).eval().requires_grad_(False)

    O12_COND.mkdir(parents=True, exist_ok=True)
    dir_ctrl = RUNS / "cond" / "cone_o12_ctrl_kvtta_w1.00"
    dir_rep0 = RUNS / "cond" / "cone_o12_rep_0tta_w1.00"
    dir_rept = RUNS / "cond" / "cone_o12_rep_kvtta_w1.00"
    for d in (dir_ctrl, dir_rep0, dir_rept):
        d.mkdir(parents=True, exist_ok=True)

    for s in subs:
        p_ctrl = torch.load(O12_ENC / "o12_e3_full_ctrl" / f"sub-{s:02d}_test_agg.pt", map_location="cpu", weights_only=True)
        p_rep = torch.load(O12_ENC / "o12_e3_full_rep" / f"sub-{s:02d}_test_agg.pt", map_location="cpu", weights_only=True)

        # Reuse O10's lat512_v2 for FusionEncoder z_vae synthesis input (exact same strategy as O11!)
        z0_o10_t = torch.from_numpy(np.load(RUNS / "low" / f"sub-{s:02d}" / "lat512_o10_bifurc_v2_n0.80.npy")).float()
        z0_o10_0 = torch.from_numpy(np.load(RUNS / "low" / f"sub-{s:02d}" / "lat512_o10_bifurc_0tta_n0.80.npy")).float()

        def synth_and_blend(p_e3, ak, bk, o10_cdir, z0_in, out_cdir):
            c_o10 = torch.from_numpy(np.load(o10_cdir / f"sub-{s:02d}.npy")).float()
            d_o10 = _perp_dir(c_o10, u_f)
            z16 = F.avg_pool2d(z0_in.to(dev), 4).reshape(200, -1)
            vae_hat = vae_mu + 0.65 * (z16 - z16.mean(0, keepdim=True)) * (vae_sd / z16.std(0, keepdim=True).clamp_min(1e-6))
            h_v = nz(p_e3[ak].float(), dim=1) * h14_norm
            b_v = nz(p_e3[bk].float(), dim=1) * b32_norm
            with torch.no_grad():
                f3 = fe_vis(**{EMB_KEYS[0]: h_v.to(dev), EMB_KEYS[1]: b_v.to(dev), EMB_KEYS[2]: vae_hat}).float().cpu()
            d3 = _perp_dir(f3, u_f)
            d_mix = nz(0.65 * d_o10 + 0.35 * d3, dim=1)
            c_mix = _to_cone(d_mix, R_f, a_f, b_f, u_f)
            np.save(out_cdir / f"sub-{s:02d}.npy", c_mix.numpy().astype(np.float32))

            # Bayesian Top-5 Semantic Medoid strictly over the 1,654 training concepts!
            hu = nz(p_e3[ak].float().to(dev) - p_e3[ak].float().to(dev).mean(0, keepdim=True), dim=1)
            bu = nz(p_e3[bk].float().to(dev) - p_e3[bk].float().to(dev).mean(0, keepdim=True), dim=1)
            s_o10 = d_o10.to(dev) @ C_sem.T
            s_e3 = 0.70 * (hu @ C_h14.T) + 0.30 * (bu @ C_b32.T)
            sim_ens = 0.55 * (s_o10 / s_o10.std()) + 0.45 * (s_e3 / s_e3.std())
            top5_v, top5_i = sim_ens.topk(5, dim=1)
            w5 = F.softmax(top5_v * 1.5, dim=1)
            med_idx = []
            for q in range(200):
                cands = top5_i[q]
                H_sub = C_h14[cands]
                G = H_sub @ H_sub.T
                med_idx.append(int(cands[int((G @ w5[q]).argmax().item())].item()))
            return c_mix, torch.tensor(med_idx, dtype=torch.long), top5_i[:, 0].cpu().long()

        c_ctrl, _, _ = synth_and_blend(p_ctrl, "a_tta", "b_tta", RUNS / "cond" / "cone_o10_bifurc_kvtta_w1.00", z0_o10_t, dir_ctrl)
        c_rep0, med0, top1_0 = synth_and_blend(p_rep, "a_0tta", "b_0tta", RUNS / "cond" / "cone_o10_bifurc_0tta_w1.00", z0_o10_0, dir_rep0)
        c_rept, medt, top1_t = synth_and_blend(p_rep, "a_tta", "b_tta", RUNS / "cond" / "cone_o10_bifurc_kvtta_w1.00", z0_o10_t, dir_rept)

        torch.save({
            "c_ctrl": c_ctrl, "c_rep0": c_rep0, "c_rept": c_rept,
            "medoid_0tta": med0, "medoid_tta": medt, "top1_0tta": top1_0, "top1_tta": top1_t,
        }, O12_COND / f"sub-{s:02d}_o12_pack.pt")

    del fe_vis, C_sem, C_h14, C_b32
    gc.collect()
    torch.cuda.empty_cache()

    lat_script = str(_find_script("lat512_start.py"))
    sub_csv = "all" if len(subs) == 10 else ",".join(map(str, subs))
    for cdir, ltag in (
        ("cone_o12_ctrl_kvtta_w1.00", "lat512_o12_ctrl_v2"),
        ("cone_o12_rep_0tta_w1.00", "lat512_o12_rep_0tta"),
        ("cone_o12_rep_kvtta_w1.00", "lat512_o12_rep_v2"),
    ):
        if not all((RUNS / "low" / f"sub-{s:02d}" / f"{ltag}_n0.80.npy").exists() for s in subs):
            subprocess.run([sys.executable, "-u", lat_script, sub_csv, "--mode_v", "v2", "--cond_dirs", cdir, "--tag", ltag], check=True)
    print(f"[O12_COND] Exported cones, Bayesian medoid indices, and lat512_v2 for {len(subs)} subjects.", flush=True)


def make_medoid_te2_embeds(med_idx: torch.Tensor, bank: dict, base_pe: torch.Tensor, base_ppe: torch.Tensor, alpha: float = 0.50):
    """Build SDXL-Turbo (pe, ppe) activating ONLY text_encoder_2 (OpenCLIP-bigG/14); text_encoder_1 stays empty."""
    seq = bank["seq_all"][med_idx].float()  # [200, 77, 1280]
    ppe_m = bank["ppe_all"][med_idx].float()  # [200, 1280]
    n = len(med_idx)
    pe = base_pe.float().cpu().expand(n, -1, -1).clone()
    pe[:, :, 768:] = (1.0 - alpha) * pe[:, :, 768:] + alpha * seq
    ppe = (1.0 - alpha) * base_ppe.float().cpu().expand(n, -1) + alpha * ppe_m
    return pe, ppe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="all")
    ap.add_argument("--arms", default="O12_ctrl_enc,O12_noBigG,O12_0tta,O12,O12_ctrl_shuf")
    ap.add_argument("--phase", choices=["all", "bank", "train", "cond", "eval"], default="all")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--dev", default="cuda:0")
    args = ap.parse_args()

    subs = list(ALL_SUBS) if args.subs == "all" else [int(x) for x in args.subs.split(",")]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    dev = args.dev

    bank = build_or_load_train1654_medoid_bank(dev=dev)
    if args.phase == "bank":
        return

    if args.phase in ("all", "train"):
        for stream in ("o12_e3_full_ctrl", "o12_e3_full_rep"):
            for s in subs:
                for sd in SEEDS_5:
                    train_one_e3_seed(s, stream, sd, dev=dev, epochs=34)
                run_e3_tta_and_0tta(s, stream, dev=dev)
        if args.phase == "train":
            return

    if args.phase in ("all", "cond"):
        export_o12_conditions_and_latents(subs, bank, dev=dev)
        if args.phase == "cond":
            return

    O12_EVAL.mkdir(parents=True, exist_ok=True)
    runner = OfficialTurboRunner(dev=dev)
    perm_shuf = torch.roll(torch.arange(200), shifts=73)

    for aname in arms:
        for s in subs:
            out_j = O12_EVAL / f"s{s:02d}_{aname}.json"
            audit_p = (O12_AUDIT / f"sub-{s:02d}_{aname}_audit.pt") if aname in ("O12", "O12_noBigG") else None
            if out_j.exists() and (audit_p is None or audit_p.exists()):
                m = json.loads(out_j.read_text())
                print(f"[CACHED] {aname:<15} s{s:02d} | top1={m['clip_top1']:.4f} 2w={m['clip']:.4f} SSIM={m['ssim']:.4f} Inc={m['inception']:.4f}", flush=True)
                continue

            spack = torch.load(O12_COND / f"sub-{s:02d}_o12_pack.pt", map_location="cpu", weights_only=True)
            pe_in = ppe_in = None
            if aname == "O12_ctrl_enc":
                c_use = spack["c_ctrl"]
                ltag = "lat512_o12_ctrl_v2"
            elif aname == "O12_noBigG":
                c_use = spack["c_rept"]
                ltag = "lat512_o12_rep_v2"
            elif aname == "O12_0tta":
                c_use = spack["c_rep0"]
                ltag = "lat512_o12_rep_0tta"
                pe_in, ppe_in = make_medoid_te2_embeds(spack["medoid_0tta"], bank, runner.pe, runner.ppe, alpha=0.50)
            elif aname == "O12":
                c_use = spack["c_rept"]
                ltag = "lat512_o12_rep_v2"
                pe_in, ppe_in = make_medoid_te2_embeds(spack["medoid_tta"], bank, runner.pe, runner.ppe, alpha=0.50)
            elif aname == "O12_ctrl_shuf":
                c_use = spack["c_rept"]
                ltag = "lat512_o12_rep_v2"
                pe_t, ppe_t = make_medoid_te2_embeds(spack["medoid_tta"], bank, runner.pe, runner.ppe, alpha=0.50)
                pe_in, ppe_in = pe_t[perm_shuf], ppe_t[perm_shuf]
            else:
                raise ValueError(aname)

            z0 = torch.from_numpy(np.load(RUNS / "low" / f"sub-{s:02d}" / f"{ltag}_n0.80.npy")).float()
            m = runner.evaluate_arm(c_use, z0, pe_all=pe_in, ppe_all=ppe_in, K=4, bs=args.bs, save_audit_path=audit_p)
            write_json(out_j, m)
            print(
                f"[O12_OFFICIAL] {aname:<14} s{s:02d} | top1={m['clip_top1']:.4f} 2w={m['clip']:.4f} "
                f"SSIM={m['ssim']:.4f} PixC={m['pixcorr']:.4f} Inc={m['inception']:.4f} Eff={m['effnet']:.4f} ({m['gen_wall_s']:.1f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
