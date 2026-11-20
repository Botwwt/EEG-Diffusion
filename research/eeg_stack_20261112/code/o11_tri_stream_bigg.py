"""O11: Tri-Stream Forward-Operator Synthesis & Reliability-Gated OpenCLIP-bigG Text Awakening.

Grounded strictly on D_dev200 (fold 0, sub-{02,05,08,09}; zero official test access):
  1. Upgraded 0-head Decoupled-Horizon Encoders (DecoupledHorizonKVEncoder, horizons 115/160/225,
     sub-repeat Empirical-Bayes conditioning, exact On-Cone (0,0) bidirectional hybrid InfoNCE,
     0 DDPM heads, 0 extra positive-pair weights):
       - Stream 1 (o11_e1_ffull): Direct EEG -> F_train (instance retinotopic + semantic fidelity).
       - Stream 2 (o11_e2_fsem): Direct EEG -> VAE-marginalized F_sem (same-concept FN-mask + margin).
       - Stream 3 (o11_e3_direct_fe): Compositional Multi-Modality Forward-Operator Inversion
         EEG -> (e_H14, e_B32) -> frozen FusionEncoder.vision_model -> On-Cone L_KV pullback
         (eliminates prototype/kNN collapse on unseen concepts).
       - Stream 4 (o11_e4_direct_bigg): Direct End-to-End EEG -> SDXL-Turbo native OpenCLIP-ViT-bigG/14
         text space (pooled_prompt_embeds [1280] + SVD_128 of hidden_states[-2][:, :16, :]), fitted
         strictly on the 1,654 THINGS training concepts.
  2. Strict Zero-Leakage Evaluation Guardrail:
       - SDXL-Turbo's text_encoder_1 (OpenAI CLIP ViT-L/14) remains strictly empty ("") at all times.
       - ONLY text_encoder_2 (OpenCLIP-ViT-bigG/14, LAION-2B) is awakened, via equal fusion of
         continuous visual-to-text ridge (Stream 3 -> bigG) and direct EEG-to-text (Stream 4).
  3. Per-Stimulus Spearman-Brown Repeat-Consistency Reliability Gating:
       - Estimates semantic reliability rho_i in [0, 1] directly from 10 disjoint R=80 EEG repeat bags
         (zero test image access) and sets per-stimulus text injection strength:
           alpha_i = clip(0.35 * (rho_i / mean(rho)), 0.10, 0.50).
  4. Official Evaluation Parity (identical to O9 and O10):
       - Same 10 subjects, 200 test stimuli, R=80 repeats, unchanged lat512_start.py (--mode_v v2),
         SDXL-Turbo smap c=0.625, low_keep=(8, 0.22, 0.04), steps=4 (4 UNet NFE/img), nsel=1,
         K=4 independent draws (exact seeds 1_000_003*i+k+12345 and 7_000_003*i+k+99), zero blur,
         and unchanged metrics.py (OpenAI CLIP ViT-L/14, InceptionV3, EffNet-B1, AlexNet2/5, SSIM, PixCorr).
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
from deep_enc import DecoupledHorizonKVEncoder, sample_sub_repeats
from o10_factorized import KVSubspaceMetric, _cone_params, _perp_dir
from train_flex import ivw_mean, shift_batch

sys.path.insert(0, str(HIVE))
from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel  # noqa: E402

nz = F.normalize
CODE_DIR = Path(__file__).resolve().parent
ALL_SUBS = tuple(range(1, 11))
SEEDS_5 = (2025, 2026, 2027, 2028, 2029)

O11_ROOT = RUNS / "o11_official"
O11_ENC = O11_ROOT / "enc"
O11_COND = O11_ROOT / "cond"
O11_EVAL = O11_ROOT / "eval"
O11_AUDIT = O11_ROOT / "audit_cache"


def _find_script(name: str) -> Path:
    candidates = [CODE_DIR] + [Path(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    for d in candidates:
        p = d / name
        if p.exists():
            return p
    return CODE_DIR / name


def _to_cone(d_perp: torch.Tensor, R_tr: float, a_tr: float, b_tr: float, u_tr: torch.Tensor, w: float = 1.0) -> torch.Tensor:
    d = nz(d_perp.float(), dim=1)
    u = u_tr.to(d.device, dtype=d.dtype)
    return R_tr * nz(a_tr * u + (w * b_tr) * d, dim=1)


def _lp_op(z: torch.Tensor, f: int) -> torch.Tensor:
    if f <= 1:
        return z
    return F.interpolate(F.avg_pool2d(z, f), size=z.shape[-1], mode="bilinear", align_corners=False)


def spearman_brown_reliability(bag_vecs: torch.Tensor) -> torch.Tensor:
    """Exact per-stimulus Spearman-Brown reliability rho_i in [0, 1] from B disjoint repeat bags [B, N, D]."""
    B = bag_vecs.shape[0]
    u = nz(bag_vecs.float() - bag_vecs.float().mean(dim=1, keepdim=True), dim=-1)
    mean_u = u.mean(dim=0)
    r_pair = ((B * mean_u.pow(2).sum(dim=-1) - 1.0) / max(1, B - 1)).clamp(0.0, 1.0)
    rho2 = (B * r_pair) / (1.0 + (B - 1) * r_pair).clamp_min(1e-6)
    return rho2.sqrt().clamp(0.0, 1.0)


class OnConeKVProjector(nn.Module):
    """Exact on-cone projection into frozen 70-layer SDXL-Turbo UNet Key-Value space (zero extra positive-pair terms)."""

    def __init__(self, kv_pack_path: str, device: str = "cuda:0"):
        super().__init__()
        self.kv = KVSubspaceMetric(kv_pack_path, device=device)
        self.register_buffer("u_tr", torch.zeros(1, 1024, device=device))
        self.alpha_tr = 0.616
        self.beta_tr = 0.788

    @torch.no_grad()
    def calibrate_fold(self, F_sub: torch.Tensor):
        self.kv.calibrate_fold(F_sub)
        f = F_sub.float()
        u = nz(f, dim=-1)
        mu = u.mean(dim=0, keepdim=True)
        self.alpha_tr = float(mu.norm().item())
        self.beta_tr = float(math.sqrt(max(1e-8, 1.0 - self.alpha_tr ** 2)))
        self.u_tr.copy_(mu / max(self.alpha_tr, 1e-8))
        kv_sum = torch.zeros((1, 8192), dtype=torch.float64, device=self.u_tr.device)
        for s in range(0, len(f), 2048):
            kv_sum += self.kv.project_kv(f[s:s + 2048]).double().sum(dim=0, keepdim=True)
        self.kv.mu_kv.copy_((kv_sum / float(len(f))).float())

    def cone_and_kv(self, c_raw: torch.Tensor):
        cu = nz(c_raw.float(), dim=-1)
        cu_c = cu - cu.mean(dim=0, keepdim=True)
        d_hat = nz(cu_c - (cu_c @ self.u_tr.T) * self.u_tr, dim=-1)
        c_cone = self.kv.nu_bar * nz(self.alpha_tr * self.u_tr + self.beta_tr * d_hat, dim=-1)
        z_kv_c = nz(self.kv.project_kv(c_cone) - self.kv.mu_kv, dim=-1)
        return d_hat, z_kv_c


class DirectForwardModalityEncoder(nn.Module):
    """DecoupledHorizon EEG trunk decoding compositional (e_H14 [1024], e_B32 [512]) for frozen FusionEncoder synthesis."""

    def __init__(self, t_len: int = 250, c_in: int = 17, hidden_size: int = 1024, num_layers: int = 2, dropout: float = 0.15, in_drop: float = 0.10):
        super().__init__()
        self.trunk = DecoupledHorizonKVEncoder(
            t_len=t_len, c_in=c_in, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, in_drop=in_drop, horizons=(115, 160, 225),
        )
        self.head_h14 = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1024))
        self.head_b32 = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 512))

    @property
    def eb(self):
        return self.trunk.eb

    def forward(self, x: torch.Tensor, r_eff=4.0, return_branches: bool = False):
        if return_branches:
            c_ens, cs = self.trunk(x, r_eff=r_eff, return_branches=True)
            return (self.head_h14(c_ens), self.head_b32(c_ens)), [(self.head_h14(cm), self.head_b32(cm)) for cm in cs]
        c_ens = self.trunk(x, r_eff=r_eff, return_branches=False)
        return self.head_h14(c_ens), self.head_b32(c_ens)


class DirectBigGTextEncoder(nn.Module):
    """DecoupledHorizon EEG trunk directly decoding SDXL-Turbo OpenCLIP-ViT-bigG/14 text space (ppe [1280], seq_z [128])."""

    def __init__(self, t_len: int = 250, c_in: int = 17, hidden_size: int = 1024, num_layers: int = 2, dropout: float = 0.15, in_drop: float = 0.10):
        super().__init__()
        self.trunk = DecoupledHorizonKVEncoder(
            t_len=t_len, c_in=c_in, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, in_drop=in_drop, horizons=(115, 160, 225),
        )
        self.head_ppe = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1280))
        self.head_seq = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 128))

    @property
    def eb(self):
        return self.trunk.eb

    def forward(self, x: torch.Tensor, r_eff=4.0, return_branches: bool = False):
        if return_branches:
            c_ens, cs = self.trunk(x, r_eff=r_eff, return_branches=True)
            return (self.head_ppe(c_ens), self.head_seq(c_ens)), [(self.head_ppe(cm), self.head_seq(cm)) for cm in cs]
        c_ens = self.trunk(x, r_eff=r_eff, return_branches=False)
        return self.head_ppe(c_ens), self.head_seq(c_ens)


def _hybrid_infonce(d_hat, z_kv_c, d_tgt, y_kv_c, scale, fn_mask=None, margin=None):
    lg = scale * (0.5 * (d_hat @ d_tgt.T) + 0.5 * (z_kv_c @ y_kv_c.T))
    if margin is not None:
        lg = lg + margin
    if fn_mask is not None:
        lg = lg.masked_fill(fn_mask, -1e4)
    tb = torch.arange(len(d_hat), device=d_hat.device)
    return 0.5 * (F.cross_entropy(lg, tb) + F.cross_entropy(lg.T, tb))


def _simple_infonce(u_hat, u_tgt, scale, fn_mask=None, margin=None):
    lg = scale * (u_hat @ u_tgt.T)
    if margin is not None:
        lg = lg + margin
    if fn_mask is not None:
        lg = lg.masked_fill(fn_mask, -1e4)
    tb = torch.arange(len(u_hat), device=u_hat.device)
    return 0.5 * (F.cross_entropy(lg, tb) + F.cross_entropy(lg.T, tb))


def build_or_load_bigG_train1654(dev: str = "cuda:0") -> dict:
    """Fit OpenCLIP-ViT-bigG/14 text targets & (H14, B32) -> bigG ridge strictly on the 1,654 training concepts."""
    O11_ROOT.mkdir(parents=True, exist_ok=True)
    cache_p = O11_ROOT / "bigG_train1654_pack.pt"
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
        clean = re.sub(r"^\d+_", "", raw_name).replace("_", " ").strip()
        names.append(clean)
    prompts = [f"a photo of a {nm}" for nm in names]

    tok2 = CLIPTokenizer.from_pretrained(str(TURBO), subfolder="tokenizer_2")
    te2 = CLIPTextModelWithProjection.from_pretrained(
        str(TURBO), subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float16
    ).to(dev).eval()

    p_list, s_list = [], []
    with torch.inference_mode():
        for i in range(0, len(prompts), 256):
            inp = tok2(prompts[i:i + 256], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(dev)
            out = te2(**inp, output_hidden_states=True)
            p_list.append(out.text_embeds.float().cpu())
            s_list.append(out.hidden_states[-2][:, :16, :].float().cpu())
    del te2
    gc.collect()
    torch.cuda.empty_cache()

    P_bigG = torch.cat(p_list, dim=0)                      # [1654, 1280]
    S_flat = torch.cat(s_list, dim=0).reshape(1654, -1)    # [1654, 20480]
    S_mu = S_flat.mean(0, keepdim=True)
    _, _, Vt_seq = torch.linalg.svd(S_flat - S_mu, full_matrices=False)
    V_seq = Vt_seq[:128]                                   # [128, 20480]
    Z_seq = (S_flat - S_mu) @ V_seq.T                      # [1654, 128]

    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    H14_all = emb_tr[EMB_KEYS[0]].float()
    B32_all = emb_tr[EMB_KEYS[1]].float()
    H14_c = torch.stack([H14_all[concept == c].mean(0) for c in uniq_c])
    B32_c = torch.stack([B32_all[concept == c].mean(0) for c in uniq_c])
    X_tr = torch.cat([nz(H14_c, dim=1), nz(B32_c, dim=1)], dim=1)  # [1654, 1536]

    A = X_tr.T @ X_tr + 1.0 * torch.eye(X_tr.shape[1])
    W_ppe = torch.linalg.solve(A, X_tr.T @ P_bigG)         # [1536, 1280]
    W_seq = torch.linalg.solve(A, X_tr.T @ Z_seq)          # [1536, 128]

    pack = {
        "P_bigG": P_bigG,
        "Z_seq": Z_seq,
        "S_mu": S_mu,
        "V_seq": V_seq,
        "W_ppe": W_ppe,
        "W_seq": W_seq,
        "ppe_rms": float(P_bigG.norm(dim=1).mean()),
        "seq_rms": float((S_flat - S_mu).norm(dim=1).mean()),
    }
    tmp = cache_p.with_name(f".{cache_p.name}.{os.getpid()}.tmp")
    torch.save(pack, tmp)
    tmp.replace(cache_p)
    print(f"[O11_BIGG] Built strict train-only 1654-concept OpenCLIP-bigG pack -> {cache_p}", flush=True)
    return pack


def build_bigG_embeds(ppe_hat: torch.Tensor, z_seq_hat: torch.Tensor, pack: dict, base_pe: torch.Tensor, base_ppe: torch.Tensor, alpha):
    """Construct SDXL-Turbo (pe, ppe) activating ONLY text_encoder_2 (OpenCLIP-bigG/14); text_encoder_1 stays empty."""
    ppe_hat = ppe_hat.float()
    ppe_hat = ppe_hat * (pack["ppe_rms"] / ppe_hat.norm(dim=1, keepdim=True).clamp_min(1e-6))

    seq_delta = z_seq_hat.float() @ pack["V_seq"]
    seq_delta = seq_delta * (pack["seq_rms"] / seq_delta.norm(dim=1, keepdim=True).clamp_min(1e-6))
    seq16 = (pack["S_mu"] + seq_delta).view(-1, 16, 1280)

    n = len(ppe_hat)
    if torch.is_tensor(alpha):
        a2 = alpha.float().cpu().view(n, 1)
        a3 = alpha.float().cpu().view(n, 1, 1)
    else:
        a2 = float(alpha)
        a3 = float(alpha)

    pe = base_pe.float().cpu().expand(n, -1, -1).clone()
    pe[:, :16, 768:] = (1.0 - a3) * pe[:, :16, 768:] + a3 * seq16
    ppe = (1.0 - a2) * base_ppe.float().cpu().expand(n, -1) + a2 * ppe_hat
    return pe, ppe


def train_one_stream_seed(sub: int, stream: str, seed: int, bigG_pack: dict, dev: str = "cuda:0", epochs: int = 34):
    """Train one full-train (16,540 images, 0 DDPM heads) encoder seed and export 0-TTA R=80 predictions."""
    out_d = O11_ENC / stream / f"sub-{sub:02d}" / f"s{seed}"
    if (out_d / "brain_model.pt").exists() and (out_d / "test_0tta.pt").exists():
        return
    out_d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x_tr = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_train.npy")).float().to(dev)
    x_te = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_test.npy")).float().to(dev)
    concept = torch.from_numpy(np.load(PREP / "train_concept.npy")).long().to(dev)
    n_tr = len(x_tr)

    is_direct = (stream == "o11_e3_direct_fe")
    is_bigG = (stream == "o11_e4_direct_bigg")
    kv_proj, fe_vis = None, None
    F_tr, F_sem, H14_tr, B32_tr, VAE_tr = None, None, None, None, None
    P_bigG_tr, Z_seq_tr, ppe_mu = None, None, None
    h14_norm = b32_norm = 1.0

    if is_bigG:
        P_bigG_tr = bigG_pack["P_bigG"].float().to(dev)
        Z_seq_tr = bigG_pack["Z_seq"].float().to(dev)
        ppe_mu = P_bigG_tr.mean(0, keepdim=True)
        model = DirectBigGTextEncoder(t_len=250, c_in=17, hidden_size=1024, num_layers=2, dropout=0.15, in_drop=0.10).to(dev)
    else:
        F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float().to(dev)
        F_sem = torch.load(PREP / "F_sem_train.pt", map_location="cpu", weights_only=True)["F_sem"].float().to(dev)
        T_mat = F_tr if stream == "o11_e1_ffull" else F_sem
        kv_proj = OnConeKVProjector(str(PREP / "unet_kv_metric.pt"), device=dev)
        kv_proj.calibrate_fold(T_mat)
        if is_direct:
            emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
            H14_tr = emb_tr[EMB_KEYS[0]].float().to(dev)
            B32_tr = emb_tr[EMB_KEYS[1]].float().to(dev)
            VAE_tr = emb_tr[EMB_KEYS[2]].float().to(dev)
            h14_norm = float(H14_tr.norm(dim=1).mean())
            b32_norm = float(B32_tr.norm(dim=1).mean())
            fe_vis = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.to(dev).eval().requires_grad_(False)
            model = DirectForwardModalityEncoder(t_len=250, c_in=17, hidden_size=1024, num_layers=2, dropout=0.15, in_drop=0.10).to(dev)
        else:
            model = DecoupledHorizonKVEncoder(t_len=250, c_in=17, hidden_size=1024, num_layers=2, dropout=0.15, in_drop=0.10, horizons=(115, 160, 225)).to(dev)

    model.eb.fit_from_fold(x_tr)

    logit_scale = nn.Parameter(torch.tensor(2.6592, device=dev))
    logit_scale_a = nn.Parameter(torch.tensor(2.6592, device=dev))
    logit_scale_b = nn.Parameter(torch.tensor(2.6592, device=dev))
    extra_p = [("logit_scale", logit_scale)] + ([("logit_scale_a", logit_scale_a), ("logit_scale_b", logit_scale_b)] if (is_direct or is_bigG) else [])
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
            xb, k_eff = sample_sub_repeats(x_tr[idx], g)
            xb = shift_batch(xb, 2, g)
            c_b = concept[idx]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                scale = logit_scale.float().exp().clamp(max=100.0)
                same_c = c_b[:, None] == c_b[None, :]
                eye = torch.eye(len(idx), dtype=torch.bool, device=dev)
                fn_mask_c = same_c & (~eye)
                diff_c = ~same_c

                if is_bigG:
                    (pe_e, sq_e), br_list = model(xb, r_eff=k_eff, return_branches=True)
                    sc_p = logit_scale_a.float().exp().clamp(max=100.0)
                    sc_s = logit_scale_b.float().exp().clamp(max=100.0)
                    with torch.no_grad():
                        pt_raw = P_bigG_tr[c_b]
                        pt = nz(pt_raw - ppe_mu, dim=-1)
                        st = nz(Z_seq_tr[c_b], dim=-1)
                        sp_sim = pt @ pt.T
                        ss_sim = st @ st.T
                        mp = 0.15 * sc_p.detach() * F.relu(sp_sim - sp_sim[diff_c].mean()) * diff_c.float()
                        ms = 0.15 * sc_s.detach() * F.relu(ss_sim - ss_sim[diff_c].mean()) * diff_c.float()

                    def _bg_loss(pp_out, sq_out):
                        pu = nz(pp_out.float() - ppe_mu, dim=-1)
                        su = nz(sq_out.float(), dim=-1)
                        lp = _simple_infonce(pu, pt, sc_p, fn_mask=fn_mask_c, margin=mp)
                        ls = _simple_infonce(su, st, sc_s, fn_mask=fn_mask_c, margin=ms)
                        l_cos = (1.0 - F.cosine_similarity(pp_out.float(), pt_raw, dim=-1)).mean()
                        return 0.55 * lp + 0.30 * ls + 0.15 * l_cos

                    loss = 0.50 * _bg_loss(pe_e, sq_e) + 0.50 * sum(_bg_loss(pb, sb) for pb, sb in br_list) / float(len(br_list))
                else:
                    with torch.no_grad():
                        fb = T_mat[idx].float()
                        u_t = nz(fb, dim=-1)
                        d_tgt = nz(u_t - (u_t @ kv_proj.u_tr.T) * kv_proj.u_tr, dim=-1)
                        y_kv_c = nz(kv_proj.kv.project_kv(fb) - kv_proj.kv.mu_kv, dim=-1)
                        if stream != "o11_e1_ffull":
                            fn_mask = fn_mask_c
                            s_img = 0.5 * (d_tgt @ d_tgt.T) + 0.5 * (y_kv_c @ y_kv_c.T)
                            s_bar = s_img[diff_c].mean() if diff_c.any() else 0.0
                            margin = 0.15 * scale.detach() * F.relu(s_img - s_bar) * diff_c.float()
                        else:
                            fn_mask, margin = None, None

                    if not is_direct:
                        c_ens, cs_br = model(xb, r_eff=k_eff, return_branches=True)
                        d_e, zk_e = kv_proj.cone_and_kv(c_ens)
                        l_ens = _hybrid_infonce(d_e, zk_e, d_tgt, y_kv_c, scale, fn_mask=fn_mask, margin=margin)
                        l_br = sum(
                            _hybrid_infonce(*kv_proj.cone_and_kv(cm), d_tgt, y_kv_c, scale, fn_mask=fn_mask, margin=margin)
                            for cm in cs_br
                        ) / float(len(cs_br))
                        loss = 0.50 * l_ens + 0.50 * l_br
                    else:
                        (h_e, b_e), br_list = model(xb, r_eff=k_eff, return_branches=True)
                        sc_h = logit_scale_a.float().exp().clamp(max=100.0)
                        sc_b = logit_scale_b.float().exp().clamp(max=100.0)
                        with torch.no_grad():
                            ht = nz(H14_tr[idx], dim=-1)
                            bt = nz(B32_tr[idx], dim=-1)
                            sh, sb = ht @ ht.T, bt @ bt.T
                            mh = 0.15 * sc_h.detach() * F.relu(sh - sh[diff_c].mean()) * diff_c.float()
                            mb = 0.15 * sc_b.detach() * F.relu(sb - sb[diff_c].mean()) * diff_c.float()
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

                        loss = 0.50 * _dir_loss(h_e, b_e) + 0.50 * sum(_dir_loss(hb, bb) for hb, bb in br_list) / float(len(br_list))

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
        if not (is_direct or is_bigG):
            cf = model(full_in, r_eff=72.0).float().cpu()
            cb = model(bag_ins, r_eff=8.0).float().view(10, 200, -1).mean(0).cpu()
            pack_0tta = {"full": cf, "bag": cb, "train_s": train_s}
        else:
            af, bf = model(full_in, r_eff=72.0)
            ab, bb = model(bag_ins, r_eff=8.0)
            ab10 = ab.float().view(10, 200, -1).cpu()
            bb10 = bb.float().view(10, 200, -1).cpu()
            pack_0tta = {"a_full": af.float().cpu(), "b_full": bf.float().cpu(), "a_bag10": ab10, "b_bag10": bb10, "train_s": train_s}

    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_d / "brain_model.pt")
    torch.save(pack_0tta, out_d / "test_0tta.pt")
    print(f"[O11_TRAIN] s{sub:02d} {stream:<18} seed={seed} done in {train_s:.1f}s", flush=True)

    del model, kv_proj, fe_vis, x_tr, x_te, F_tr, F_sem, H14_tr, B32_tr, VAE_tr
    gc.collect()
    torch.cuda.empty_cache()


def run_stream_tta_and_0tta(sub: int, stream: str, R_f: float, dev: str = "cuda:0") -> dict:
    """Export 5-seed 0-TTA and 5-seed x 50-step centered split-half TTA on R=80 official test EEG (0 DDPM heads)."""
    cache_p = O11_ENC / stream / f"sub-{sub:02d}_test_agg.pt"
    if cache_p.exists():
        return torch.load(cache_p, map_location="cpu", weights_only=True)

    is_direct = (stream == "o11_e3_direct_fe")
    is_bigG = (stream == "o11_e4_direct_bigg")
    is_pair = is_direct or is_bigG

    # Aggregate 0-TTA first
    p0_list = [torch.load(O11_ENC / stream / f"sub-{sub:02d}" / f"s{sd}" / "test_0tta.pt", map_location="cpu", weights_only=True) for sd in SEEDS_5]
    if not is_pair:
        f0 = torch.stack([p["full"] for p in p0_list], 0).mean(0)
        b0 = torch.stack([p["bag"] for p in p0_list], 0).mean(0)
        zero_tta = 0.85 * f0 + 0.15 * b0
    else:
        af0 = torch.stack([p["a_full"] for p in p0_list], 0).mean(0)
        bf0 = torch.stack([p["b_full"] for p in p0_list], 0).mean(0)
        ab0 = torch.stack([p["a_bag10"] for p in p0_list], 0).mean(0)  # [10, 200, D]
        bb0 = torch.stack([p["b_bag10"] for p in p0_list], 0).mean(0)
        a_0tta = 0.85 * af0 + 0.15 * ab0.mean(0)
        b_0tta = 0.85 * bf0 + 0.15 * bb0.mean(0)
        rho_0tta = spearman_brown_reliability(ab0)

    # Run 50-step x 5-seed centered split-half TTA (40 vs 40 repeats, updating ONLY stem_linears!)
    t0 = time.time()
    x_te = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_test.npy")).float().to(dev)
    full_in = ivw_mean(x_te)
    perm_bag = torch.randperm(80, generator=torch.Generator().manual_seed(7 + sub)).to(dev)
    bag_ins = torch.cat([ivw_mean(x_te[:, perm_bag[g_ * 8:(g_ + 1) * 8]]) for g_ in range(10)], dim=0)

    P, A_m, L_kv_t = None, None, None
    if not is_pair:
        kv_pack = torch.load(PREP / "unet_kv_metric.pt", map_location="cpu", weights_only=True)
        L_kv_t = kv_pack["L_kv"].float().T.contiguous().to(dev)
        P = IPProjectionModel(embed_dim=1024, cross_attention_dim=2048, sequence_length=4)
        P.load_state_dict(kv_pack["proj_state"])
        P = P.to(dev).eval().requires_grad_(False)
        A_m = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev).eval().requires_grad_(False)
    tgt = torch.arange(200, device=dev)

    out_tta, a_tta_l, b_tta_l, ab_tta_l = [], [], [], []
    for sd in SEEDS_5:
        ckpt = torch.load(O11_ENC / stream / f"sub-{sub:02d}" / f"s{sd}" / "brain_model.pt", map_location="cpu", weights_only=True)
        if is_direct:
            m = DirectForwardModalityEncoder(t_len=250, c_in=17, hidden_size=1024, num_layers=2, dropout=0.15, in_drop=0.0).to(dev)
        elif is_bigG:
            m = DirectBigGTextEncoder(t_len=250, c_in=17, hidden_size=1024, num_layers=2, dropout=0.15, in_drop=0.0).to(dev)
        else:
            m = DecoupledHorizonKVEncoder(t_len=250, c_in=17, hidden_size=1024, num_layers=2, dropout=0.15, in_drop=0.0, horizons=(115, 160, 225)).to(dev)
        m.load_state_dict(ckpt)
        m.eval().requires_grad_(False)
        for name, param in m.named_parameters():
            if "stem_linears" in name:
                param.requires_grad_(True)
        tparams = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.Adam(tparams, lr=1e-4)
        g = torch.Generator(device=dev).manual_seed(3000 + sub * 37 + sd)

        with torch.no_grad():
            c0 = m(full_in, r_eff=72.0)
            if is_pair:
                a0, b0_anc = c0[0].float(), c0[1].float()
            else:
                c0 = c0.float()

        for _ in range(50):
            perm = torch.rand(200, 80, device=dev, generator=g).argsort(1)
            xa = ivw_mean(torch.gather(x_te, 1, perm[:, :40, None, None].expand(-1, -1, 17, 250)))
            xb = ivw_mean(torch.gather(x_te, 1, perm[:, 40:, None, None].expand(-1, -1, 17, 250)))
            if not is_pair:
                ca = m(xa, r_eff=36.0).float()
                cb = m(xb, r_eff=36.0).float()
                ka = (P(A_m(nz(ca, dim=1) * R_f)) @ L_kv_t).flatten(1)
                kb = (P(A_m(nz(cb, dim=1) * R_f)) @ L_kv_t).flatten(1)
                ka = nz(ka - ka.mean(0, keepdim=True), dim=1)
                kb = nz(kb - kb.mean(0, keepdim=True), dim=1)
                lg = (ka @ kb.T) / 0.1
                loss_nce = 0.5 * (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.T, tgt))
                cf = m(full_in, r_eff=72.0).float()
                anc = ((cf - c0).pow(2).sum(1) / c0.pow(2).sum(1)).mean()
                (loss_nce + 1.0 * anc).backward()
            else:
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
            if not is_pair:
                ft = m(full_in, r_eff=72.0).float()
                bt = m(bag_ins, r_eff=8.0).float().view(10, 200, -1).mean(0)
                out_tta.append((0.65 * ft + 0.35 * bt).cpu())
            else:
                af, bf = m(full_in, r_eff=72.0)
                ab_raw, bb_raw = m(bag_ins, r_eff=8.0)
                ab10 = ab_raw.float().view(10, 200, -1)
                bb10 = bb_raw.float().view(10, 200, -1)
                a_tta_l.append((0.65 * af.float() + 0.35 * ab10.mean(0)).cpu())
                b_tta_l.append((0.65 * bf.float() + 0.35 * bb10.mean(0)).cpu())
                ab_tta_l.append(ab10.cpu())
        del m

    tta_s = time.time() - t0
    if not is_pair:
        pack = {"c_0tta": zero_tta, "c_tta": torch.stack(out_tta, 0).mean(0), "tta_s": tta_s}
    else:
        ab_tta_m = torch.stack(ab_tta_l, 0).mean(0)
        pack = {
            "a_0tta": a_0tta, "b_0tta": b_0tta, "rho_0tta": rho_0tta,
            "a_tta": torch.stack(a_tta_l, 0).mean(0),
            "b_tta": torch.stack(b_tta_l, 0).mean(0),
            "rho_tta": spearman_brown_reliability(ab_tta_m),
            "tta_s": tta_s,
        }
    torch.save(pack, cache_p)
    print(f"[O11_TTA] s{sub:02d} {stream:<18} done in {tta_s:.1f}s", flush=True)
    del x_te, full_in, bag_ins, P, A_m, L_kv_t
    gc.collect()
    torch.cuda.empty_cache()
    return pack


def export_tristream_conditions_and_latents(subs, bigG_pack: dict, dev: str = "cuda:0"):
    """Build Tri-Stream cone conditions (0-TTA & TTA), run unchanged lat512_start.py (--mode_v v2), and export OpenCLIP-bigG packs."""
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

    # Step A: Export initial E12 (0.5 E1 + 0.5 E2) cone to build initial lat512 for FusionEncoder z_vae input
    cdir_e12_0 = RUNS / "cond" / "cone_o11_e12_0tta_w1.00"
    cdir_e12_t = RUNS / "cond" / "cone_o11_e12_kvtta_w1.00"
    cdir_e12_0.mkdir(parents=True, exist_ok=True)
    cdir_e12_t.mkdir(parents=True, exist_ok=True)
    for s in subs:
        p1 = torch.load(O11_ENC / "o11_e1_ffull" / f"sub-{s:02d}_test_agg.pt", map_location="cpu", weights_only=True)
        p2 = torch.load(O11_ENC / "o11_e2_fsem" / f"sub-{s:02d}_test_agg.pt", map_location="cpu", weights_only=True)
        for tag, k, outd in (("0tta", "c_0tta", cdir_e12_0), ("tta", "c_tta", cdir_e12_t)):
            d12 = nz(0.50 * _perp_dir(p1[k], u_f) + 0.50 * _perp_dir(p2[k], u_f), dim=1)
            np.save(outd / f"sub-{s:02d}.npy", _to_cone(d12, R_f, a_f, b_f, u_f).numpy().astype(np.float32))

    lat_script = str(_find_script("lat512_start.py"))
    sub_csv = "all" if len(subs) == 10 else ",".join(map(str, subs))
    for cdir, ltag in (("cone_o11_e12_0tta_w1.00", "lat512_o11_e12_0tta"), ("cone_o11_e12_kvtta_w1.00", "lat512_o11_e12_v2")):
        if not all((RUNS / "low" / f"sub-{s:02d}" / f"{ltag}_n0.80.npy").exists() for s in subs):
            subprocess.run([sys.executable, "-u", lat_script, sub_csv, "--mode_v", "v2", "--cond_dirs", cdir, "--tag", ltag], check=True)

    # Step B: Synthesize Stream 3 via frozen FusionEncoder.vision_model -> Tri-Stream E123 cone + bigG text pack!
    fe_vis = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.to(dev).eval().requires_grad_(False)
    cdir_o11_0 = RUNS / "cond" / "cone_o11_tri_0tta_w1.00"
    cdir_o11_t = RUNS / "cond" / "cone_o11_tri_kvtta_w1.00"
    cdir_o11_0.mkdir(parents=True, exist_ok=True)
    cdir_o11_t.mkdir(parents=True, exist_ok=True)
    O11_COND.mkdir(parents=True, exist_ok=True)

    for s in subs:
        p1 = torch.load(O11_ENC / "o11_e1_ffull" / f"sub-{s:02d}_test_agg.pt", map_location="cpu", weights_only=True)
        p2 = torch.load(O11_ENC / "o11_e2_fsem" / f"sub-{s:02d}_test_agg.pt", map_location="cpu", weights_only=True)
        p3 = torch.load(O11_ENC / "o11_e3_direct_fe" / f"sub-{s:02d}_test_agg.pt", map_location="cpu", weights_only=True)
        p4 = torch.load(O11_ENC / "o11_e4_direct_bigg" / f"sub-{s:02d}_test_agg.pt", map_location="cpu", weights_only=True)

        sub_pack = {}
        for mode_tag, ck, ak, bk, rk, ltag, outd in (
            ("0tta", "c_0tta", "a_0tta", "b_0tta", "rho_0tta", "lat512_o11_e12_0tta", cdir_o11_0),
            ("tta", "c_tta", "a_tta", "b_tta", "rho_tta", "lat512_o11_e12_v2", cdir_o11_t),
        ):
            z0 = torch.from_numpy(np.load(RUNS / "low" / f"sub-{s:02d}" / f"{ltag}_n0.80.npy")).float()
            z16 = F.avg_pool2d(z0.to(dev), 4).reshape(200, -1)
            vae_hat = vae_mu + 0.65 * (z16 - z16.mean(0, keepdim=True)) * (vae_sd / z16.std(0, keepdim=True).clamp_min(1e-6))

            d1, d2 = _perp_dir(p1[ck], u_f), _perp_dir(p2[ck], u_f)
            h3 = nz(p3[ak], dim=1) * h14_norm
            b3 = nz(p3[bk], dim=1) * b32_norm
            with torch.no_grad():
                f3 = fe_vis(**{EMB_KEYS[0]: h3.to(dev), EMB_KEYS[1]: b3.to(dev), EMB_KEYS[2]: vae_hat}).float().cpu()
            d3 = _perp_dir(f3, u_f)
            d123 = nz(0.35 * d1 + 0.30 * d2 + 0.35 * d3, dim=1)
            c123 = _to_cone(d123, R_f, a_f, b_f, u_f)
            np.save(outd / f"sub-{s:02d}.npy", c123.numpy().astype(np.float32))

            x3 = torch.cat([nz(h3, dim=1), nz(b3, dim=1)], dim=1)
            ppe_e3, seq_e3 = x3 @ bigG_pack["W_ppe"], x3 @ bigG_pack["W_seq"]
            ppe_e4, seq_e4 = p4[ak].float(), p4[bk].float()
            ppe_34 = 0.5 * nz(ppe_e3, dim=1) + 0.5 * nz(ppe_e4, dim=1)
            seq_34 = 0.5 * nz(seq_e3, dim=1) + 0.5 * nz(seq_e4, dim=1)
            rho_c = 0.5 * (p3[rk].float() + p4[rk].float())
            alpha_rel = (0.35 * (rho_c / rho_c.mean().clamp_min(1e-6))).clamp(0.10, 0.50)

            sub_pack[mode_tag] = {
                "c123": c123,
                "ppe_34": ppe_34,
                "seq_34": seq_34,
                "rho_c": rho_c,
                "alpha_rel": alpha_rel,
            }
        torch.save(sub_pack, O11_COND / f"sub-{s:02d}_o11_pack.pt")

    del fe_vis
    gc.collect()
    torch.cuda.empty_cache()

    # Step C: Final native 512px start latents using the full Tri-Stream E123 cone!
    for cdir, ltag in (("cone_o11_tri_0tta_w1.00", "lat512_o11_tri_0tta"), ("cone_o11_tri_kvtta_w1.00", "lat512_o11_tri_v2")):
        if not all((RUNS / "low" / f"sub-{s:02d}" / f"{ltag}_n0.80.npy").exists() for s in subs):
            subprocess.run([sys.executable, "-u", lat_script, sub_csv, "--mode_v", "v2", "--cond_dirs", cdir, "--tag", ltag], check=True)
    print(f"[O11_COND] Exported Tri-Stream E123 cones, OpenCLIP-bigG packs, and lat512_v2 for {len(subs)} subjects.", flush=True)


class OfficialTurboRunner:
    """Official THINGS-EEG2 evaluator (exact parity with gen_exp.py & metrics.py, plus OpenCLIP-bigG text_encoder_2)."""

    def __init__(self, dev: str = "cuda:0"):
        from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
        from metrics import load_gt

        self.dev = dev
        self.dt = torch.float16
        self.pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=self.dt, variant="fp16")
        IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(self.pipe.unet)
        self.pipe.text_encoder.to(dev, self.dt)
        self.pipe.text_encoder_2.to(dev, self.dt)
        with torch.inference_mode():
            self.pe, _, self.ppe, _ = self.pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
        del self.pipe.text_encoder, self.pipe.text_encoder_2
        self.pipe.text_encoder = None
        self.pipe.text_encoder_2 = None
        torch.cuda.empty_cache()

        self.proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, self.dt).eval().requires_grad_(False)
        self.adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, self.dt).eval().requires_grad_(False)
        self.pipe.to(dev, self.dt)
        self.pipe.vae.enable_slicing()
        self.pipe.set_progress_bar_config(disable=True)
        self.p2 = StableDiffusionXLImg2ImgPipeline(**self.pipe.components)
        self.p2.set_progress_bar_config(disable=True)

        self.vae32 = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval().requires_grad_(False)
        self.vae32.enable_slicing()
        self.gt_ids, self.gt = load_gt()

    @torch.inference_mode()
    def evaluate_arm(self, c_all: torch.Tensor, z0_all: torch.Tensor, pe_all=None, ppe_all=None, K: int = 4, bs: int = 20, save_audit_path: Path = None) -> dict:
        from metrics import _nets, score as eval_metrics_score

        t0 = time.time()
        init = torch.cat([
            (self.vae32.decode(z0_all[j:j + 10].to(self.dev, torch.float32) / self.vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
            for j in range(0, 200, 10)
        ])

        steps = 4
        smap_c, s_rel, s_unrel = 0.625, 0.5, 1.0
        low_keep = (8, 0.22, 0.04)
        rho = torch.full((200, 1, 64, 64), smap_c)
        smap = s_unrel - (s_unrel - s_rel) * rho
        n_run = min(int(round(steps * s_unrel)), steps)
        hold = (n_run - steps * smap).clamp(min=0)

        rec = torch.empty(K, 200, 3, 512, 512)
        for s in range(0, 200, bs):
            q = list(range(s, min(200, s + bs)))
            Q = len(q)
            cc = c_all[q].to(self.dev, self.dt).repeat_interleave(K, dim=0)
            ips = self.proj(self.adapter(cc))
            pe_b = pe_all[q].to(self.dev, self.dt).repeat_interleave(K, dim=0) if pe_all is not None else self.pe.expand(Q * K, -1, -1)
            ppe_b = ppe_all[q].to(self.dev, self.dt).repeat_interleave(K, dim=0) if ppe_all is not None else self.ppe.expand(Q * K, -1)
            gens = [torch.Generator(self.dev).manual_seed(1_000_003 * i + k + 12345) for i in q for k in range(K)]

            i0 = init[q].repeat_interleave(K, dim=0).to(self.dev, self.dt)
            z0 = z0_all[q].to(self.dev).float().repeat_interleave(K, dim=0)
            eps = torch.stack([
                torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * i + k + 99))
                for i in q for k in range(K)
            ]).to(self.dev)
            h_ = hold[q].repeat_interleave(K, dim=0).to(self.dev)

            def cb(pp, i_step, t_cur, ck):
                lt = ck["latents"].float()
                si = int(pp.scheduler.step_index)
                sig = pp.scheduler.sigmas[si].to(self.dev)
                a = (h_ - i_step).clamp(0.0, 1.0)
                if a.max() > 0:
                    lt = a * (z0 + sig * eps) + (1.0 - a) * lt
                if i_step in (steps - 2, steps - 1):
                    f_lp, w2, w3 = low_keep
                    g_t = w2 if i_step == steps - 2 else w3
                    if g_t > 0:
                        lt = lt + g_t * _lp_op((z0 + sig * eps) - lt, f_lp)
                ck["latents"] = lt.to(self.dt)
                return ck

            lat_out = self.p2(
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
            out = (self.vae32.decode(lat_out.float() / self.vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0)
            img = (out.float().clamp(0.0, 1.0) * 255.0).round().div(255.0)
            rec[:, q] = img.cpu().view(Q, K, 3, 512, 512).transpose(0, 1)

        gen_s = float(time.time() - t0)
        ps = eval_metrics_score(rec, self.gt)
        mean_m = {k: float(np.mean([r[k] for r in ps])) for k in ps[0]}
        mean_m["per_draw"] = ps
        mean_m["nfe_per_img"] = 4
        mean_m["nsel"] = 1
        mean_m["gen_wall_s"] = gen_s
        mean_m["total_wall_s"] = float(time.time() - t0)

        if save_audit_path is not None:
            # Compute exact OpenAI CLIP ViT-L/14 [K, 200] rank & top-1 prediction matrix (exact metrics.py corrcoef formula)
            clip_fn, clip_pre, _ = _nets()["clip"]
            def _clip_np(x_rgb):
                feats = []
                for j in range(0, len(x_rgb), 50):
                    feats.append(clip_fn(clip_pre(x_rgb[j:j + 50].to(self.dev))).float().flatten(1).cpu())
                return torch.cat(feats, dim=0).numpy()
            gt_f = _clip_np(self.gt)
            N = len(gt_f)
            ranks = torch.empty(K, N, dtype=torch.int16)
            top1_idx = torch.empty(K, N, dtype=torch.int16)
            for k_i in range(K):
                re_f = _clip_np(rec[k_i])
                r_mat = np.corrcoef(gt_f, re_f)[:N, N:]  # [N_gt, N_pred], column j = pred j vs 200 gt candidates
                diag = np.diag(r_mat)[None, :]
                ranks[k_i] = torch.from_numpy((r_mat > diag).sum(axis=0).astype(np.int16) + 1)
                top1_idx[k_i] = torch.from_numpy(r_mat.argmax(axis=0).astype(np.int16))
            k0_u8 = (rec[0].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            save_audit_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "gt_ids": self.gt_ids,
                "ranks": ranks,
                "top1_idx": top1_idx,
                "k0_rgb_u8": k0_u8,
                "metrics": mean_m,
            }, save_audit_path)
            t1_audit = float((ranks == 1).float().mean().item())
            assert abs(t1_audit - mean_m["clip_top1"]) < 1e-5, (t1_audit, mean_m["clip_top1"])

        return mean_m


def export_hybrid_o10_conditions_and_latents(subs):
    """Blend O10's bifurcated KVEBPyrEncoder cone with O11's Tri-Stream cone (50/50 tangent) + lat512_v2."""
    ok_mask = torch.from_numpy(np.load(PREP / "lat512_train_ok.npy")).bool()
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    R_f, a_f, b_f, u_f = _cone_params(F_tr[ok_mask])

    cdir_hyb_0 = RUNS / "cond" / "cone_o11_on_o10_0tta_w1.00"
    cdir_hyb_t = RUNS / "cond" / "cone_o11_on_o10_kvtta_w1.00"
    cdir_hyb_0.mkdir(parents=True, exist_ok=True)
    cdir_hyb_t.mkdir(parents=True, exist_ok=True)

    for s in subs:
        spack = torch.load(O11_COND / f"sub-{s:02d}_o11_pack.pt", map_location="cpu", weights_only=True)
        for mk, o10_dir, out_dir in (
            ("0tta", RUNS / "cond" / "cone_o10_bifurc_0tta_w1.00", cdir_hyb_0),
            ("tta",  RUNS / "cond" / "cone_o10_bifurc_kvtta_w1.00", cdir_hyb_t),
        ):
            c_o10 = torch.from_numpy(np.load(o10_dir / f"sub-{s:02d}.npy")).float()
            c_o11 = spack[mk]["c123"].float()
            d_hyb = nz(0.50 * _perp_dir(c_o10, u_f) + 0.50 * _perp_dir(c_o11, u_f), dim=1)
            c_hyb = _to_cone(d_hyb, R_f, a_f, b_f, u_f)
            np.save(out_dir / f"sub-{s:02d}.npy", c_hyb.numpy().astype(np.float32))

    lat_script = str(_find_script("lat512_start.py"))
    sub_csv = "all" if len(subs) == 10 else ",".join(map(str, subs))
    for cdir, ltag in (
        ("cone_o11_on_o10_0tta_w1.00", "lat512_o11_on_o10_0tta"),
        ("cone_o11_on_o10_kvtta_w1.00", "lat512_o11_on_o10_v2"),
    ):
        if not all((RUNS / "low" / f"sub-{s:02d}" / f"{ltag}_n0.80.npy").exists() for s in subs):
            subprocess.run([sys.executable, "-u", lat_script, sub_csv, "--mode_v", "v2", "--cond_dirs", cdir, "--tag", ltag], check=True)
    print(f"[O11_ON_O10] Exported 50/50 O10+O11 hybrid cones + lat512_v2 for {len(subs)} subjects.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="all")
    ap.add_argument("--arms", default="O11_on_O10,O11_on_O10_noBigG,O11_on_O10_0tta,O11_on_O10_ctrl_shuf")
    ap.add_argument("--bs", type=int, default=20)
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    subs = list(ALL_SUBS) if args.subs == "all" else [int(x) for x in args.subs.split(",")]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    dev = args.dev

    bigG_pack = build_or_load_bigG_train1654(dev=dev)
    ok_mask = torch.from_numpy(np.load(PREP / "lat512_train_ok.npy")).bool()
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    R_f, _, _, _ = _cone_params(F_tr[ok_mask])

    # Phase 1: Train all 4 streams x 5 seeds across all subjects + TTA
    for stream in ("o11_e1_ffull", "o11_e2_fsem", "o11_e3_direct_fe", "o11_e4_direct_bigg"):
        for s in subs:
            for sd in SEEDS_5:
                train_one_stream_seed(s, stream, sd, bigG_pack, dev=dev, epochs=34)
            run_stream_tta_and_0tta(s, stream, R_f, dev=dev)

    # Phase 2: Export Tri-Stream conditions, lat512_v2 start latents, and OpenCLIP-bigG packs
    export_tristream_conditions_and_latents(subs, bigG_pack, dev=dev)
    if any(a.startswith("O11_on_O10") for a in arms):
        export_hybrid_o10_conditions_and_latents(subs)
    if args.skip_eval:
        return

    # Phase 3: Official 10-subject SDXL-Turbo generation & metrics.py evaluation (exact O9/O10 parity)
    O11_EVAL.mkdir(parents=True, exist_ok=True)
    runner = OfficialTurboRunner(dev=dev)
    perm_shuf = torch.roll(torch.arange(200), shifts=73)

    for aname in arms:
        print(f"\n=== [O11_EVAL] Starting official 10-subject arm: {aname} ===", flush=True)
        for s in subs:
            out_j = O11_EVAL / f"s{s:02d}_{aname}.json"
            audit_p = (O11_AUDIT / f"sub-{s:02d}_{aname}_audit.pt") if aname in ("O11", "O11_on_O10") else None
            if out_j.exists() and (audit_p is None or audit_p.exists()):
                m = json.loads(out_j.read_text())
                print(
                    f"[CACHED] {aname:<20} s{s:02d} | top1={m['clip_top1']:.4f} 2w={m['clip']:.4f} "
                    f"SSIM={m['ssim']:.4f} PixC={m['pixcorr']:.4f} Incep={m['inception']:.4f} Eff={m['effnet']:.4f}",
                    flush=True,
                )
                continue

            spack = torch.load(O11_COND / f"sub-{s:02d}_o11_pack.pt", map_location="cpu", weights_only=True)
            is_0tta = aname.endswith("_0tta")
            mk = "0tta" if is_0tta else "tta"
            if aname.startswith("O11_on_O10"):
                cdir = RUNS / "cond" / ("cone_o11_on_o10_0tta_w1.00" if is_0tta else "cone_o11_on_o10_kvtta_w1.00")
                ltag = "lat512_o11_on_o10_0tta" if is_0tta else "lat512_o11_on_o10_v2"
                c_use = torch.from_numpy(np.load(cdir / f"sub-{s:02d}.npy")).float()
            else:
                ltag = "lat512_o11_tri_0tta" if is_0tta else "lat512_o11_tri_v2"
                c_use = spack[mk]["c123"]
            z0 = torch.from_numpy(np.load(RUNS / "low" / f"sub-{s:02d}" / f"{ltag}_n0.80.npy")).float()

            pe_in = ppe_in = None
            if aname in ("O11", "O11_0tta"):
                pe_in, ppe_in = build_bigG_embeds(spack[mk]["ppe_34"], spack[mk]["seq_34"], bigG_pack, runner.pe, runner.ppe, alpha=spack[mk]["alpha_rel"])
            elif aname in ("O11_fixed35", "O11_on_O10", "O11_on_O10_0tta"):
                pe_in, ppe_in = build_bigG_embeds(spack[mk]["ppe_34"], spack[mk]["seq_34"], bigG_pack, runner.pe, runner.ppe, alpha=0.35)
            elif aname in ("O11_ctrl_shuf", "O11_on_O10_ctrl_shuf"):
                a_use = 0.35 if aname.startswith("O11_on_O10") else spack[mk]["alpha_rel"]
                pe_tmp, ppe_tmp = build_bigG_embeds(spack[mk]["ppe_34"], spack[mk]["seq_34"], bigG_pack, runner.pe, runner.ppe, alpha=a_use)
                pe_in, ppe_in = pe_tmp[perm_shuf], ppe_tmp[perm_shuf]

            m = runner.evaluate_arm(c_use, z0, pe_all=pe_in, ppe_all=ppe_in, K=4, bs=args.bs, save_audit_path=audit_p)
            write_json(out_j, m)
            print(
                f"[O11_OFFICIAL] {aname:<14} s{s:02d} | top1={m['clip_top1']:.4f} 2w={m['clip']:.4f} "
                f"SSIM={m['ssim']:.4f} PixC={m['pixcorr']:.4f} Incep={m['inception']:.4f} Eff={m['effnet']:.4f} "
                f"({m['gen_wall_s']:.1f}s gen / {m['total_wall_s']:.1f}s tot)",
                flush=True,
            )

        rows = [json.loads((O11_EVAL / f"s{s:02d}_{aname}.json").read_text()) for s in subs if (O11_EVAL / f"s{s:02d}_{aname}.json").exists()]
        if len(rows) == len(subs):
            print(
                f">>> MEAN {aname:<14} (n={len(subs)}): "
                f"CLIP_t1={np.mean([r['clip_top1'] for r in rows]):.4f}  "
                f"CLIP_2w={np.mean([r['clip'] for r in rows]):.4f}  "
                f"SSIM={np.mean([r['ssim'] for r in rows]):.4f}  "
                f"PixC={np.mean([r['pixcorr'] for r in rows]):.4f}  "
                f"Incep={np.mean([r['inception'] for r in rows]):.4f}  "
                f"EffNet={np.mean([r['effnet'] for r in rows]):.4f}  "
                f"NFE=4  gen_s/sub={np.mean([r['gen_wall_s'] for r in rows]):.1f}s",
                flush=True,
            )

    # Write final consolidated official summary table
    lines = ["=== O11 Official 10-Subject Benchmark (K=4 indep, nsel=1, 4 UNet NFE/img, 0 DDPM heads, zero blur) ==="]
    hdr = f"{'Arm':<15} | {'CLIP_t1':>8} {'CLIP_2w':>8} {'SSIM':>7} {'PixC':>7} {'Incep':>7} {'EffNet':>7} {'Alex2':>7} {'Alex5':>7} {'NFE':>4}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for aname in arms:
        rows = [json.loads((O11_EVAL / f"s{s:02d}_{aname}.json").read_text()) for s in subs if (O11_EVAL / f"s{s:02d}_{aname}.json").exists()]
        if len(rows) == len(subs):
            lines.append(
                f"{aname:<15} | {np.mean([r['clip_top1'] for r in rows]):>8.4f} {np.mean([r['clip'] for r in rows]):>8.4f} "
                f"{np.mean([r['ssim'] for r in rows]):>7.4f} {np.mean([r['pixcorr'] for r in rows]):>7.4f} "
                f"{np.mean([r['inception'] for r in rows]):>7.4f} {np.mean([r['effnet'] for r in rows]):>7.4f} "
                f"{np.mean([r['alex2'] for r in rows]):>7.4f} {np.mean([r['alex5'] for r in rows]):>7.4f}    4"
            )
    lines.append("\nPer-subject Headline O11 (CLIP_top1 / CLIP_2w / SSIM / PixCorr / Incep / EffNet):")
    for s in subs:
        p = O11_EVAL / f"s{s:02d}_O11.json"
        if p.exists():
            r = json.loads(p.read_text())
            lines.append(
                f"  sub-{s:02d}: top1={r['clip_top1']:.4f}  2w={r['clip']:.4f}  SSIM={r['ssim']:.4f}  "
                f"PixC={r['pixcorr']:.4f}  Incep={r['inception']:.4f}  EffNet={r['effnet']:.4f}"
            )
    txt = "\n".join(lines) + "\n"
    (O11_ROOT / "official_summary.txt").write_text(txt)
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    main()
