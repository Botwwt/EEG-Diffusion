"""Step 2 (exp6): End-to-End 4-Step Full-Chain Within-Concept Competition (DRaFT-Intra4).

Grounded directly in Step 1 (exp5) findings:
  - Weakening z0 anchoring from s625 -> s375 (hold=0.75, low_keep=(8,0.11,0.02)) immediately unlocked Arm3b's
    within-concept CLIP Intra10 from 18.45% -> 19.95% (+1.50%pp, dc10=21.15%, beating Arm1b by +1.15%pp/+2.90%pp)
    while keeping z0's valuable spatial scaffold (s000 dropped every arm to 17.4%).
  - Now we close BOTH remaining gaps of single-step SwapDiff:
      (1) Train-test trajectory mismatch: backpropagate directly through the actual 4-step SDXL-Turbo Euler-Ancestral
          rollout starting from genuine out-of-fold EEG-predicted z0_oof(E) (NOT noised GT latents!).
      (2) Objective mismatch: sample 4 same-concept photos from the SAME subject (R=4 EEG, shared noise seed),
          roll out 4 full images, score with frozen Brain-HIVE visual features (LAION OpenCLIP-H14 + OpenCLIP-B32
          + coarse 16x16 latent structure; ZERO OpenAI CLIP ViT-L/14 / Inception / AlexNet leakage!), and require
          EACH generated image to rank its own true photo #1 ahead of the other 3 same-concept siblings!

Strictly Controlled 4-Arm Comparison (all evaluated at s375 AND s625, 4-NFE nsel=1, R=4 FreeGen, sealed fold-0):
  - SwapDiff (Arm3b)         : Existing best single-step SwapDiff baseline.
  - D1_SimOnly_Cond          : 4-step DRaFT training ONLY cond_layer with positive similarity ONLY (NO competition).
  - D2_Compete4_Cond         : Exact same params/steps/LR/seed as D1, adding within-concept 4-way competition!
  - D3_Compete4_StepRouteZ0  : 4-step DRaFT within-concept 4-way competition jointly training:
                               (a) Neuro-Temporal Step-Specific Conditioning Routing (4 UNet steps x 3 EEG horizons),
                               (b) Zero-init PCA-manifold start-latent residual Delta_z0(E) on z0_ridge(E).

Independent Held-Out Evaluators (100% excluded from training):
  OpenAI CLIP ViT-L/14 (primary: R=4 FreeGen Intra10 Top-1 & g500), Inception-v3, AlexNet-5, AlexNet-2, PixCorr, SSIM.
"""
import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(os.environ["RAD_ROOT"])
for d in [ROOT / "code/exp_w7", ROOT / "code/exp_w6", ROOT / "code/rad", ROOT / "code/exp", ROOT / "code/Brain-HIVE"]:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

from common import EMB_KEYS, PREP, PRIOR, TURBO, concept_folds, write_json
from exp0_intra10_swap_gate import ALL_SUBS, DEV_SUBS, OUT_DIR, fit_tr14_ridge_z0, get_concept_splits, intra10_and_swap_from_sim
from exp1_swapdiff_3arm import ResidualIPCondLayer, build_tr14_resources, per_sample_cone
from exp5_z0_2x2_diag import extract_eval_feats, generate_batch_rollout, rollout_4step_diff, score_full500
from o11_tri_stream_bigg import _lp_op
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder
from train_flex import pyramid_feats, shift_batch

from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel

nz = F.normalize
DRAFT_DIR = OUT_DIR / "draft_intra4"
DRAFT_DIR.mkdir(parents=True, exist_ok=True)


def forward_branches_and_trunk(model: CrossSubjectHorizonEncoder, x: torch.Tensor, sid: int, r_eff: float = 4.0):
    """Exact CrossSubjectHorizonEncoder returning (fp, c_ens, cs_branches[3])."""
    xg = model.ebs[sid](x, r_eff=r_eff)
    hs = [
        model.stem_norms[i](model.sub_stems[sid][i](model.in_drop(pyramid_feats(xg[..., :t_h])))) + model.sub_emb.weight[sid]
        for i, t_h in enumerate(model.horizons)
    ]
    cs = model.block(torch.cat(hs, dim=0)).chunk(len(model.horizons), dim=0)  # 3 tensors [B, 1024] for (115, 160, 225)
    w_h = F.softmax(model.horizon_logits, dim=0)
    b_af = model.block.layers[-1].norm.bias.to(cs[0].dtype)
    d0 = cs[0] - b_af
    d_mix = sum(w_h[i] * (cs[i] - b_af) for i in range(len(cs)))
    scale = d0.norm(dim=-1, keepdim=True) / d_mix.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    c_ens = b_af + d_mix * scale
    return model.head_f(c_ens), c_ens, cs


class StepRoutedCondAndZ0(nn.Module):
    """Neuro-Temporal Step-Specific Conditioning Routing (4 UNet steps x 3 EEG horizons) + PCA-manifold Delta_z0."""

    def __init__(self, base_cond: ResidualIPCondLayer, v_pca_256: torch.Tensor, z0_rms_ref: float, rank: int = 96, max_rel_ips: float = 0.25, max_rel_z0: float = 0.35):
        super().__init__()
        self.base_cond = base_cond
        self.max_rel_ips = max_rel_ips
        self.max_rel_z0 = max_rel_z0
        self.z0_rms_ref = float(z0_rms_ref)
        self.register_buffer("V_pca", v_pca_256.float())  # [16384, 256]

        # Learned 4x3 neuro-temporal routing logits: step t in {0,1,2,3} attends over (short 0-460ms, mid 0-640ms, full 0-900ms)
        self.route_logits = nn.Parameter(torch.zeros(4, 3))
        self.step_ln = nn.LayerNorm(1024)
        self.step_down = nn.ModuleList([nn.Linear(1024, rank) for _ in range(4)])
        self.step_up = nn.ModuleList([nn.Linear(rank, 4 * 2048, bias=False) for _ in range(4)])
        for up in self.step_up:
            nn.init.zeros_(up.weight)

        # Zero-init PCA-manifold start-latent residual Delta_z0(h_eeg)
        self.z0_ln = nn.LayerNorm(1024)
        self.z0_down = nn.Linear(1024, 128)
        self.z0_up = nn.Linear(128, v_pca_256.shape[1], bias=False)
        nn.init.zeros_(self.z0_up.weight)

    def forward_ips_steps(self, c_ens: torch.Tensor, cs_branches: tuple, ips_proj: torch.Tensor) -> list:
        """Return list of 4 step-specific IP tensors [B, 4, 2048]."""
        ips0 = self.base_cond(c_ens, ips_proj)  # initialized from Arm3b's trained cond_layer!
        br_stack = torch.stack([b.float() for b in cs_branches], dim=1)  # [B, 3, 1024]
        w_route = F.softmax(self.route_logits, dim=-1)  # [4, 3]
        base_n = ips0.float().norm(dim=(1, 2), keepdim=True).clamp_min(1e-4)

        out_steps = []
        for t_i in range(4):
            h_t = torch.einsum("h,bhd->bd", w_route[t_i], br_stack)
            delta = self.step_up[t_i](F.silu(self.step_down[t_i](self.step_ln(h_t)))).view(-1, 4, 2048)
            del_n = delta.norm(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            sc = torch.minimum(torch.ones_like(del_n), (self.max_rel_ips * base_n) / del_n)
            out_steps.append(ips0 + (delta * sc).to(ips0.dtype))
        return out_steps

    def forward_z0(self, c_ens: torch.Tensor, cs_branches: tuple, z0_ridge: torch.Tensor) -> torch.Tensor:
        """Add bounded PCA-manifold residual Delta_z0 driven by early+ensemble EEG."""
        h_z = 0.5 * c_ens.float() + 0.5 * cs_branches[0].float()  # include early retinotopic branch!
        dz_coeff = self.z0_up(F.silu(self.z0_down(self.z0_ln(h_z))))  # [B, 256]
        dz_flat = dz_coeff @ self.V_pca.T  # [B, 16384]
        dz = dz_flat.view(-1, 4, 64, 64)
        dz_rms = dz.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
        sc = torch.minimum(torch.ones_like(dz_rms), (self.max_rel_z0 * self.z0_rms_ref) / dz_rms)
        return z0_ridge.float() + dz * sc


def rollout_4step_step_ips(
    unet,
    scheduler,
    ips_steps: list,
    z0: torch.Tensor,
    eps_init: torch.Tensor,
    step_noises: list,
    pe: torch.Tensor,
    ppe: torch.Tensor,
    atid: torch.Tensor,
    smap_c: float,
    low_keep,
    *,
    clean_update=None,
    remix_initial_layout=True,
    start_latent=None,
    start_step=0,
    stop_step=4,
) -> torch.Tensor:
    """Exact 4-step differentiable Euler-Ancestral rollout supporting per-step ips_steps[0..3]."""
    B = z0.shape[0]
    dt = unet.dtype
    steps = 4
    if (not 0 <= start_step < stop_step <= steps
            or (start_step > 0) != (start_latent is not None)
            or (start_latent is not None and start_latent.shape != z0.shape)):
        raise ValueError("The continuation requires the matching raw Euler state and step")
    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas
    hold_val = 2.0 * smap_c if remix_initial_layout else 0.0
    sig0 = float(sigmas[0])
    lt = (z0.float() + sig0 * eps_init.float()) if smap_c > 0.0 else (sig0 * eps_init.float())
    if start_latent is not None:
        lt = start_latent.float()

    pe_b = pe.expand(B, -1, -1).to(dt)
    ppe_b = ppe.expand(B, -1).to(dt)
    atid_b = atid.expand(B, -1).to(dt)

    for i_step in range(start_step, stop_step):
        sigma = float(sigmas[i_step])
        sigma_next = float(sigmas[i_step + 1])
        t_cur = timesteps[i_step].expand(B)
        ips_b = ips_steps[i_step].to(dt)

        model_in = (lt / math.sqrt(sigma * sigma + 1.0)).to(dt)
        eps_pred = unet(
            model_in,
            t_cur,
            encoder_hidden_states=pe_b,
            added_cond_kwargs={"text_embeds": ppe_b, "time_ids": atid_b},
            cross_attention_kwargs={"ip_hidden_states": ips_b},
        ).sample.float()
        if clean_update is not None:
            delta = clean_update(i_step, lt - sigma * eps_pred)
            if delta is not None:
                eps_pred = eps_pred - delta / sigma

        sigma_up = min(sigma_next, math.sqrt((sigma_next ** 2) * (sigma ** 2 - sigma_next ** 2) / (sigma ** 2))) if sigma_next > 0 else 0.0
        sigma_down = math.sqrt(max(0.0, sigma_next ** 2 - sigma_up ** 2))
        lt = lt + eps_pred * (sigma_down - sigma)
        if sigma_up > 0.0:
            lt = lt + sigma_up * step_noises[i_step].float()

        if smap_c > 0.0:
            a = max(0.0, min(1.0, hold_val - i_step))
            if a > 0.0:
                lt = a * (z0.float() + sigma_next * eps_init.float()) + (1.0 - a) * lt
            if low_keep is not None and i_step in (steps - 2, steps - 1):
                f_lp, w2, w3 = low_keep
                g_t = w2 if i_step == steps - 2 else w3
                if g_t > 0.0:
                    lt = lt + g_t * _lp_op((z0.float() + sigma_next * eps_init.float()) - lt, f_lp)
    return lt


def precompute_or_load_tr14_z0_oof(tr_idx_np: np.ndarray, concept_np: np.ndarray, dev: str = "cuda:0", subjects=ALL_SUBS) -> dict:
    """Fit a ridge prediction for each training row with its concept fold held out."""
    cache_p = DRAFT_DIR / "z0_oof_tr14_256.pt"
    if cache_p.exists():
        print(f"[CACHED] {cache_p}", flush=True)
        cached = torch.load(cache_p, map_location="cpu", weights_only=True)
        if cached["Z_oof"].shape[1] != len(tr_idx_np):
            raise ValueError("The cached layout predictions use a different training scope")
        return cached

    t0 = time.time()
    c_fold = concept_folds(1654, 5, seed=20261112)
    f_tr = c_fold[concept_np[tr_idx_np]]
    retained = np.unique(f_tr)
    print(f"[PRECOMPUTE] {len(retained)} concept-disjoint folds for {len(tr_idx_np)} layouts", flush=True)
    lat_all = np.load(PREP / "lat512_train.npy")[tr_idx_np]
    Y_tr = torch.from_numpy(lat_all).float().to(dev).flatten(1)
    del lat_all
    my = Y_tr.mean(0)
    Yc = Y_tr - my
    z0_rms_ref = float(Yc.pow(2).mean(dim=1).sqrt().mean())
    del Y_tr

    def lp_flat(y_flat, f):
        v = y_flat.view(-1, 4, 64, 64)
        return F.interpolate(F.avg_pool2d(v, f), size=64, mode="bilinear", align_corners=False).flatten(1)

    torch.manual_seed(20261115)
    n_pca = 256
    Yc_w = Yc + 1.5 * lp_flat(Yc, 4) + 0.75 * lp_flat(Yc, 2)
    _, _, V_p = torch.pca_lowrank(Yc_w, q=n_pca + 32, center=False, niter=4)
    del Yc_w
    V_pca, _ = torch.linalg.qr(V_p[:, :n_pca].double(), mode="reduced")
    Z_gt = Yc.double() @ V_pca  # [N_tr, 256]
    rms_real_lp4 = float(F.avg_pool2d(Yc.view(-1, 4, 64, 64), 4).flatten(1).pow(2).sum(1).mean().sqrt())
    del Yc

    WINDOWS = [(12, 62), (10, 75), (15, 90), (10, 125), (0, 250)]
    Z_oof = torch.zeros((10, len(tr_idx_np), n_pca), dtype=torch.float16)

    for sub in subjects:
        sid = sub - 1
        X_m4 = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_train.npy")[tr_idx_np].mean(1, dtype=np.float32))
        z_sub = torch.zeros((len(tr_idx_np), n_pca), dtype=torch.float32, device=dev)
        for fk in retained:
            val_m = torch.from_numpy(f_tr == fk)
            tr_m = ~val_m
            preds = []
            for t_a, t_b in WINDOWS:
                A_tr = X_m4[tr_m, :, t_a:t_b].flatten(1).to(dev)
                A_va = X_m4[val_m, :, t_a:t_b].flatten(1).to(dev)
                mu, sd = A_tr.mean(0, keepdim=True), A_tr.std(0, keepdim=True).clamp_min(1e-6)
                A_tr = ((A_tr - mu) / sd).double()
                A_va = ((A_va - mu) / sd).double()
                Cxx = (A_tr.T @ A_tr) / float(len(A_tr))
                Cxy = (A_tr.T @ Z_gt[tr_m.to(dev)]) / float(len(A_tr))
                lam = 3.0 * float(torch.trace(Cxx) / Cxx.shape[0])
                W = torch.linalg.solve(Cxx + lam * torch.eye(Cxx.shape[0], dtype=torch.float64, device=dev), Cxy)
                preds.append((A_va @ W).float())
            z_sub[val_m.to(dev)] = torch.stack(preds, 0).mean(0)
        # Match 16x16 RMS to 0.80 * real (same as fit_tr14_ridge_z0)
        y_sub = z_sub @ V_pca.float().T
        cur_rms = float(F.avg_pool2d(y_sub.view(-1, 4, 64, 64), 4).flatten(1).pow(2).sum(1).mean().sqrt())
        if cur_rms > 1e-6:
            z_sub = z_sub * (0.80 * rms_real_lp4 / cur_rms)
        Z_oof[sid] = z_sub.cpu().half()
        del X_m4, z_sub, y_sub

    pack = {"Z_oof": Z_oof, "V_pca": V_pca.float().cpu(), "my": my.float().cpu(), "z0_rms_ref": z0_rms_ref}
    torch.save(pack, cache_p)
    print(f"[SAVED] {cache_p} ({time.time()-t0:.1f}s)", flush=True)
    torch.cuda.empty_cache()
    return pack


def assignment_training_layout(layout, mode):
    if mode == "individual":
        return layout
    if mode == "group-mean":
        return layout.mean(dim=0, keepdim=True).expand_as(layout)
    raise ValueError(f"Unsupported assignment training layout: {mode}")


def layout_reference_scores(scores, reference):
    if reference is None:
        return scores
    interaction = reference - reference.mean(dim=1, keepdim=True)
    interaction = interaction - interaction.mean(dim=0, keepdim=True)
    return scores - interaction


def draft_similarity(latent, vae, vision_models, b32_alignment, normalization, targets):
    images = []
    for start in range(0, len(latent), 2):
        latents = (latent[start:start + 2] / vae.config.scaling_factor).to(torch.bfloat16)
        decoded = vae.decode(latents).sample.float()
        images.append((decoded / 2.0 + 0.5).clamp(0.0, 1.0))
    pixels = F.interpolate(torch.cat(images, dim=0), size=224, mode="bilinear", align_corners=False)
    mean, std = normalization
    pixels = ((pixels - mean) / std).to(torch.bfloat16)
    h14, b32 = vision_models
    feat_h14 = nz(h14(pixel_values=pixels).image_embeds.float(), dim=-1)
    feat_b32 = nz(b32(pixel_values=pixels).image_embeds.float() @ b32_alignment, dim=-1)
    feat_layout = nz(F.avg_pool2d(latent.float(), 4).flatten(1), dim=-1)
    target_h14, target_b32, target_layout = targets
    return (0.60 * (feat_h14 @ target_h14.T) + 0.25 * (feat_b32 @ target_b32.T)
            + 0.15 * (feat_layout @ target_layout.T))


def train_draft_arm(
    mode: str,
    init_ckpt: Path,
    res: dict,
    z0_oof_pack: dict,
    pipe_unet,
    scheduler,
    vae_bf,
    adapter_bf,
    proj_bf,
    h14_vis,
    b32_vis,
    b32_fix,
    pe_bf,
    ppe_bf,
    atid_bf,
    steps: int = 200,
    seed: int = 2025,
    dev: str = "cuda:0",
    *,
    training_layout: str = "individual",
    assignment_score: str = "absolute",
) -> Path:
    """Train one DRaFT-Intra4 arm: d1_simonly_cond | d2_compete4_cond | d3_compete4_steproute_z0."""
    assert mode in ("d1_simonly_cond", "d2_positive_half_cond", "d2_compete4_cond", "d3_compete4_steproute_z0")
    if training_layout not in ("individual", "group-mean"):
        raise ValueError(f"Unsupported assignment training layout: {training_layout}")
    if training_layout != "individual" and mode != "d2_compete4_cond":
        raise ValueError("The group-mean layout experiment requires within-concept assignment")
    if assignment_score not in ("absolute", "layout-reference"):
        raise ValueError(f"Unsupported assignment score: {assignment_score}")
    if assignment_score == "layout-reference" and (mode != "d2_compete4_cond" or training_layout != "individual"):
        raise ValueError("Layout references require the paired loss and individual layouts")
    out_p = DRAFT_DIR / f"{mode}_st{steps}_s{seed}.pt"
    if out_p.exists():
        saved = torch.load(out_p, map_location="meta", weights_only=True)
        if (saved.get("training_layout", "individual") != training_layout
                or saved.get("assignment_score", "absolute") != assignment_score):
            raise ValueError("The cached fit uses a different assignment configuration. Use a separate run directory.")
        print(f"[CACHED] {out_p}", flush=True)
        return out_p

    t0 = time.time()
    pack = torch.load(init_ckpt, map_location="cpu", weights_only=True)
    # Freeze EEG backbone (train ONLY conditioning layer / step-router / Delta_z0 as specified!)
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
    model.eval().requires_grad_(False)

    base_cond = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev)
    base_cond.load_state_dict({k: v.float() for k, v in pack["cond_layer"].items()})

    # Keep frozen reference copy for gentle trust-region regularization (prevents adversarial scorer hacking!)
    cond_ref = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev)
    cond_ref.load_state_dict(base_cond.state_dict())
    cond_ref.eval().requires_grad_(False)

    V_pca = z0_oof_pack["V_pca"].to(dev)   # [16384, 256]
    my_z0 = z0_oof_pack["my"].to(dev)      # [16384]
    Z_oof = z0_oof_pack["Z_oof"]           # [10, N_tr, 256] fp16 on CPU

    step_router = None
    if mode == "d3_compete4_steproute_z0":
        step_router = StepRoutedCondAndZ0(base_cond, V_pca, z0_oof_pack["z0_rms_ref"]).to(dev)
        train_params = list(step_router.parameters())
        lr = 2.0e-4
    else:
        train_params = list(base_cond.parameters())
        lr = 1.5e-4

    opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=0.05, betas=(0.9, 0.995))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    R_tr, a_tr, b_tr = pack["cone"]["R"], pack["cone"]["alpha"], pack["cone"]["beta"]
    u_tr = pack["cone"]["u_tr"].to(dev).float()

    reference_steps = None
    if assignment_score == "layout-reference":
        with torch.no_grad():
            midpoint = (R_tr * u_tr).reshape(1, -1).to(torch.bfloat16)
            tokens = proj_bf(adapter_bf(midpoint)).expand(4, -1, -1)
            reference_steps = [tokens] * 4

    # Exact s375 regime locked in by exp5!
    smap_c = 0.375
    low_keep = (8, 0.11, 0.02)

    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)

    c_groups = res["tr14_c_groups"]
    n_c = len(c_groups)
    g = torch.Generator(device=dev).manual_seed(seed * 101 + 7)
    rng = np.random.default_rng(seed * 101 + 7)

    # 4 concept groups x 4 photos = 16 full 4-step rollouts per optimizer step
    n_micro = 4
    tau = 15.0
    use_compete = mode in ("d2_compete4_cond", "d3_compete4_steproute_z0")

    print(f"[TRAIN {mode}] steps={steps}, 4 groups x 4 same-concept photos/step, s375, compete={use_compete}, lr={lr}", flush=True)
    if step_router is not None:
        step_router.train()
    else:
        base_cond.train()

    run_top1, run_sim, run_ce = [], [], []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        st_top1, st_sim, st_ce = 0.0, 0.0, 0.0

        for mb in range(n_micro):
            # Sample 1 subject & 1 concept, pick 4 distinct photos from its 10 photos (exact R=4 real EEG!)
            sid = int(rng.choice(res["subject_ids"]))
            ci = int(rng.integers(0, n_c))
            pick4 = rng.choice(10, size=4, replace=False)
            locs = torch.tensor(c_groups[ci, pick4], dtype=torch.long)
            locs_dev = locs.to(dev)

            x4 = res["x_tr_all"][sid][locs].float().mean(1).to(dev)  # [4, 17, 250]
            x4 = shift_batch(x4, 2, g)

            # Genuine out-of-fold EEG-predicted z0_oof for these 4 photos!
            with torch.no_grad():
                z0_oof = (Z_oof[sid, locs].float().to(dev) @ V_pca.T + my_z0).view(4, 4, 64, 64)
                z0_gt = res["lat_tr_cpu"][locs].float().to(dev)
                h14_gt = nz(res["H14_tr"][locs_dev].float(), dim=-1)  # [4, 1024]
                b32_gt = nz(res["B32_tr"][locs_dev].float(), dim=-1)  # [4, 512]
                z0_gt_lp = nz(F.avg_pool2d(z0_gt, 4).flatten(1), dim=-1)  # [4, 1024] coarse 16x16 structure!

                # Frozen EEG backbone forward
                fp, c_ens, cs_br = forward_branches_and_trunk(model, x4, sid, r_eff=4.0)
                c_cone = per_sample_cone(fp, R_tr, a_tr, b_tr, u_tr).to(torch.bfloat16)
                ips_proj = proj_bf(adapter_bf(c_cone))
                ips_ref = cond_ref(c_ens, ips_proj).detach().float()

            # Fresh random noise seed SHARED across all 4 same-concept photos in this group!
            eps_sh = torch.randn(1, 4, 64, 64, device=dev, generator=g).expand(4, -1, -1, -1)
            sn_sh = [torch.randn(1, 4, 64, 64, device=dev, generator=g).expand(4, -1, -1, -1) for _ in range(4)]

            # Trainable conditioning (+ optional step routing & Delta_z0)
            if step_router is not None:
                ips_steps = step_router.forward_ips_steps(c_ens, cs_br, ips_proj)
                z0_use = step_router.forward_z0(c_ens, cs_br, z0_oof)
                l_reg = 0.05 * sum((ip.float() - ips_ref).pow(2).mean() for ip in ips_steps) / 4.0 + 0.05 * (z0_use - z0_oof).pow(2).mean()
            else:
                ips_cur = base_cond(c_ens, ips_proj)
                ips_steps = [ips_cur, ips_cur, ips_cur, ips_cur]
                z0_use = assignment_training_layout(z0_oof, training_layout)
                l_reg = 0.05 * (ips_cur.float() - ips_ref).pow(2).mean()

            targets = (h14_gt, b32_gt, z0_gt_lp)
            models = (h14_vis, b32_vis)
            normalization = (clip_mean, clip_std)
            reference = None
            if reference_steps is not None:
                with torch.no_grad():
                    reference_latent = rollout_4step_step_ips(
                        pipe_unet, scheduler, reference_steps, z0_use, eps_sh, sn_sh,
                        pe_bf, ppe_bf, atid_bf, smap_c, low_keep,
                    )
                    reference = draft_similarity(
                        reference_latent, vae_bf, models, b32_fix, normalization, targets,
                    )

            lat_out = rollout_4step_step_ips(
                pipe_unet, scheduler, ips_steps, z0_use, eps_sh, sn_sh,
                pe_bf, ppe_bf, atid_bf, smap_c, low_keep,
            )
            S_4 = draft_similarity(lat_out, vae_bf, models, b32_fix, normalization, targets)
            diag_sim = S_4.diag().mean()
            l_sim = -diag_sim
            tgt = torch.arange(4, device=dev)
            assignment = layout_reference_scores(S_4, reference)
            l_ce = 0.5 * (F.cross_entropy(tau * assignment, tgt) + F.cross_entropy(tau * assignment.T, tgt))

            if use_compete:
                loss_mb = (1.0 * l_ce + 0.50 * l_sim + l_reg) / float(n_micro)
            elif mode == "d2_positive_half_cond":
                # Leave the positive and trust coefficients at their D2 values. Only the paired term is removed.
                loss_mb = (0.50 * l_sim + l_reg) / float(n_micro)
            else:
                # SimOnly: positive similarity to own GT photo ONLY (NO within-concept competition!)
                # Weight 12.0 matches d(l_ce)/d(S_ii) = tau*(1-0.25) = 11.25 + 0.50 = 11.75 for exact gradient-scale parity!
                loss_mb = (12.0 * l_sim + l_reg) / float(n_micro)

            loss_mb.backward()
            st_top1 += float((S_4.detach().argmax(dim=1) == tgt).float().mean()) / n_micro
            st_sim += float(diag_sim.detach()) / n_micro
            st_ce += float(l_ce.detach()) / n_micro

        if step == 1:
            gn = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in train_params if p.grad is not None))
            print(f"  [{mode} STEP-1 DIAG] grad_norm={gn:.4f} mem_MB={torch.cuda.max_memory_allocated()//(1024*1024)}", flush=True)

        torch.nn.utils.clip_grad_norm_(train_params, 2.0)
        opt.step()
        sched.step()

        run_top1.append(st_top1)
        run_sim.append(st_sim)
        run_ce.append(st_ce)
        if step in (1, 10, 30, 60, 90, 120, 150, 200) or step == steps:
            el = time.time() - t0
            eta = (el / step) * (steps - step)
            extra_s = ""
            if step_router is not None:
                wr = F.softmax(step_router.route_logits.detach(), dim=-1).cpu().numpy()
                extra_s = f" | route(t0/t3)={wr[0].round(2).tolist()}/{wr[3].round(2).tolist()}"
            print(
                f"  [{mode}] step {step:3d}/{steps} 4w_t1={np.mean(run_top1[-20:])*100:.1f}% "
                f"sim={np.mean(run_sim[-20:]):.4f} ce={np.mean(run_ce[-20:]):.3f} ({el:.0f}s, ETA {eta:.0f}s){extra_s}",
                flush=True,
            )

    save_obj = {
        "backbone_ckpt": str(init_ckpt),
        "cond_layer": {k: v.half().cpu() for k, v in base_cond.state_dict().items()},
        "step_router": {k: v.half().cpu() for k, v in step_router.state_dict().items()} if step_router is not None else None,
        "cone": pack["cone"],
        "mode": mode,
        "steps": steps,
        "training_layout": training_layout,
        "assignment_score": assignment_score,
    }
    torch.save(save_obj, out_p)
    print(f"[SAVED] {out_p} ({time.time()-t0:.1f}s)", flush=True)
    return out_p


@torch.inference_mode()
def evaluate_draft_arm(
    arm_name: str,
    ckpt_p: Path,
    subs,
    sel50_idx,
    z0_by_sub,
    z0_oof_pack,
    pipe_unet,
    scheduler,
    vae32,
    adapter,
    proj,
    pe,
    ppe,
    atid,
    gt_rgb50,
    gt_feats,
    nets,
    h14_vis,
    H14_gt50,
    dev="cuda:0",
):
    """Evaluate one DRaFT / SwapDiff arm on R=4 FreeGen at BOTH s375 and s625 across independent evaluators + H14."""
    pack = torch.load(ckpt_p, map_location="cpu", weights_only=True)
    if "state_dict" in pack:
        sd_backbone = pack["state_dict"]
    else:
        sd_backbone = torch.load(pack["backbone_ckpt"], map_location="cpu", weights_only=True)["state_dict"]
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in sd_backbone.items()})
    model.eval()

    base_cond = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev)
    base_cond.load_state_dict({k: v.float() for k, v in pack["cond_layer"].items()})
    base_cond.eval()

    step_router = None
    if pack.get("step_router") is not None:
        step_router = StepRoutedCondAndZ0(base_cond, z0_oof_pack["V_pca"].to(dev), z0_oof_pack["z0_rms_ref"]).to(dev)
        step_router.load_state_dict({k: v.float() for k, v in pack["step_router"].items()})
        step_router.eval()

    R_tr, a_tr, b_tr = pack["cone"]["R"], pack["cone"]["alpha"], pack["cone"]["beta"]
    u_tr = pack["cone"]["u_tr"].to(dev).float()

    cids_500 = [i // 10 for i in range(500)]
    eps_map = {ci: torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * ci + 99)) for ci in range(50)}
    sn_map = {
        ci: [torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(1_000_003 * ci + 12345 + 17 * st)) for st in range(4)]
        for ci in range(50)
    }
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)

    res_by_reg = {}
    for rtag, smap_c, lk in [("s375", 0.375, (8, 0.11, 0.02)), ("s625", 0.625, (8, 0.22, 0.04))]:
        per_sub = {}
        for sub in subs:
            sid = sub - 1
            X_raw = np.load(PREP / f"sub-{sub:02d}_train.npy")
            xm4 = torch.from_numpy(X_raw[sel50_idx].mean(1)).float()
            del X_raw
            z0_e = z0_by_sub[sub]

            # Build per-step ips & z0 for all 500 images
            ips_steps_all = [[], [], [], []]
            z0_use_l = []
            for i in range(0, 500, 100):
                fp, c_ens, cs_br = forward_branches_and_trunk(model, xm4[i:i + 100].to(dev), sid, r_eff=4.0)
                c_cone = per_sample_cone(fp, R_tr, a_tr, b_tr, u_tr).to(torch.float16)
                ips_p = proj(adapter(c_cone))
                if step_router is not None:
                    st_l = step_router.forward_ips_steps(c_ens, cs_br, ips_p)
                    for k in range(4):
                        ips_steps_all[k].append(st_l[k].cpu())
                    z0_use_l.append(step_router.forward_z0(c_ens, cs_br, z0_e[i:i + 100].to(dev)).cpu())
                else:
                    ip = base_cond(c_ens, ips_p).cpu()
                    for k in range(4):
                        ips_steps_all[k].append(ip)
                    z0_use_l.append(z0_e[i:i + 100])
            ips_steps_cat = [torch.cat(lst, 0) for lst in ips_steps_all]
            z0_final = torch.cat(z0_use_l, 0)

            # Generate 500 images via exact 4-step rollout
            rec = torch.empty(500, 3, 512, 512, dtype=torch.float32)
            for s0 in range(0, 500, 50):
                q = list(range(s0, min(500, s0 + 50)))
                cids_q = [cids_500[i] for i in q]
                eps_q = torch.stack([eps_map[ci] for ci in cids_q]).to(dev)
                sn_q = [torch.stack([sn_map[ci][st] for ci in cids_q]).to(dev) for st in range(4)]
                ips_q = [ips_steps_cat[st][q].to(dev) for st in range(4)]
                lat_out = rollout_4step_step_ips(pipe_unet, scheduler, ips_q, z0_final[q].to(dev), eps_q, sn_q, pe, ppe, atid, smap_c, lk)
                for j0 in range(0, len(q), 10):
                    img_b = (vae32.decode(lat_out[j0:j0 + 10].float() / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
                    rec[s0 + j0:s0 + j0 + len(img_b)] = img_b

            # Score on ALL strictly independent evaluators (OpenAI CLIP ViT-L/14, Alex5, Inception, PixCorr, SSIM)
            m = score_full500(rec, gt_rgb50, gt_feats, nets, dev=dev)
            # Also report training-scorer H14 separately for transparency (train-scorer vs independent-scorer check!)
            h_l = []
            for i in range(0, 500, 50):
                x224 = ((F.interpolate(rec[i:i + 50].to(dev), size=224, mode="bilinear", align_corners=False) - clip_mean) / clip_std).to(torch.float16)
                h_l.append(h14_vis(pixel_values=x224).image_embeds.float().cpu())
            H_rec = torch.cat(h_l, 0)
            Hr = nz(H_rec - H_rec.mean(0, keepdim=True), dim=1).view(50, 10, -1)
            Hg = nz(H14_gt50 - H14_gt50.mean(0, keepdim=True), dim=1).view(50, 10, -1)
            m["h14_train_scorer"] = intra10_and_swap_from_sim(torch.bmm(Hr, Hg.transpose(1, 2)).numpy())
            per_sub[f"sub-{sub:02d}"] = m

        m_avg = {
            "clip_t10": float(np.mean([v["clip"]["top1"] for v in per_sub.values()])),
            "clip_dc10": float(np.mean([v["clip"]["top1_dc"] for v in per_sub.values()])),
            "clip_sw": float(np.mean([v["clip"]["swap2x2"] for v in per_sub.values()])),
            "clip_g500": float(np.mean([v["clip"]["global500_top1"] for v in per_sub.values()])),
            "h14_t10": float(np.mean([v["h14_train_scorer"]["top1"] for v in per_sub.values()])),
            "h14_sw": float(np.mean([v["h14_train_scorer"]["swap2x2"] for v in per_sub.values()])),
            "incep_sw": float(np.mean([v["inception"]["swap2x2"] for v in per_sub.values()])),
            "alex5_t10": float(np.mean([v["alex5"]["top1"] for v in per_sub.values()])),
            "alex5_sw": float(np.mean([v["alex5"]["swap2x2"] for v in per_sub.values()])),
            "pix_sw": float(np.mean([v["pixcorr"]["swap2x2"] for v in per_sub.values()])),
            "ssim": float(np.mean([v["ssim"] for v in per_sub.values()])),
        }
        res_by_reg[rtag] = {"mean": m_avg, "per_sub": per_sub}
        print(
            f">>> [{arm_name:24s} {rtag}] "
            f"INDEP CLIP-L14: t10={m_avg['clip_t10']*100:5.2f}% dc10={m_avg['clip_dc10']*100:5.2f}% sw={m_avg['clip_sw']*100:5.2f}% g500={m_avg['clip_g500']*100:4.2f}% | "
            f"Inc sw={m_avg['incep_sw']*100:5.2f}% | A5: t10={m_avg['alex5_t10']*100:5.2f}% sw={m_avg['alex5_sw']*100:5.2f}% | "
            f"Px sw={m_avg['pix_sw']*100:5.2f}% | SSIM={m_avg['ssim']:.4f} || (train-H14: t10={m_avg['h14_t10']*100:5.2f}% sw={m_avg['h14_sw']*100:5.2f}%)",
            flush=True,
        )
    del model, base_cond, step_router
    torch.cuda.empty_cache()
    return res_by_reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--modes", default="d1_simonly_cond,d2_compete4_cond,d3_compete4_steproute_z0")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    dev = "cuda:0"
    t_all = time.time()
    arm3b_ckpt = OUT_DIR / "arm3b_rep_rawswap_condlyr_st600_s2025.pt"
    assert arm3b_ckpt.exists(), f"Missing {arm3b_ckpt}"

    res = build_tr14_resources(dev=dev)
    z0_oof_pack = precompute_or_load_tr14_z0_oof(res["tr_idx_np"], res["concept_np"], dev=dev)
    z0_by_sub = {sub: fit_tr14_ridge_z0(sub, res["tr_idx_np"], res["sel50_idx"], dev=dev) for sub in DEV_SUBS}

    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    from transformers import CLIPVisionModelWithProjection
    from metrics import _nets

    # 1. Load frozen bf16 SDXL-Turbo UNet + VAE + Brain-HIVE LAION OpenCLIP-H14 & B32 for DRaFT training
    pipe_tr = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=torch.bfloat16, variant="fp16")
    IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(pipe_tr.unet)
    pipe_tr.to(dev, torch.bfloat16)
    pipe_tr.scheduler.set_timesteps(4, device=dev)
    with torch.no_grad():
        pe_bf, _, ppe_bf, _ = pipe_tr.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
        atid_bf = pipe_tr._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=torch.bfloat16, text_encoder_projection_dim=pipe_tr.text_encoder_2.config.projection_dim).to(dev)
        pe_bf, ppe_bf, atid_bf = pe_bf.detach().clone(), ppe_bf.detach().clone(), atid_bf.detach().clone()
    del pipe_tr.text_encoder, pipe_tr.text_encoder_2
    pipe_tr.text_encoder = None
    pipe_tr.text_encoder_2 = None
    pipe_tr.unet.eval().requires_grad_(False)
    pipe_tr.unet.enable_gradient_checkpointing()
    vae_bf = pipe_tr.vae.to(dev, torch.bfloat16).eval().requires_grad_(False)
    vae_bf.enable_gradient_checkpointing()

    adapter_bf = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, torch.bfloat16).eval().requires_grad_(False)
    proj_bf = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, torch.bfloat16).eval().requires_grad_(False)

    h14_vis = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / "models" / EMB_KEYS[0]), torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    h14_vis.gradient_checkpointing_enable()
    b32_vis = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / "models" / EMB_KEYS[1]), torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    b32_vis.gradient_checkpointing_enable()
    b32_fix = torch.load(PREP / "b32_fix.pt", weights_only=True).float().to(dev)

    ckpts = {"SwapDiff_Arm3b": arm3b_ckpt}
    for m in modes:
        ckpts[m] = train_draft_arm(
            m, arm3b_ckpt, res, z0_oof_pack, pipe_tr.unet, pipe_tr.scheduler, vae_bf,
            adapter_bf, proj_bf, h14_vis, b32_vis, b32_fix, pe_bf, ppe_bf, atid_bf,
            steps=args.steps, seed=2025, dev=dev,
        )

    del pipe_tr, vae_bf, adapter_bf, proj_bf, h14_vis, b32_vis, pe_bf, ppe_bf, atid_bf
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Full Independent Held-Out Evaluation on Dev-4 (s375 AND s625)
    dt = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=dt, variant="fp16")
    IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(pipe.unet)
    pipe.to(dev, dt)
    pipe.unet.eval().requires_grad_(False)
    pipe.scheduler.set_timesteps(4, device=dev)
    with torch.inference_mode():
        pe, _, ppe, _ = pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
        atid = pipe._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=dt, text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).to(dev)
    del pipe.text_encoder, pipe.text_encoder_2, pipe.vae
    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, dt).eval().requires_grad_(False)
    proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, dt).eval().requires_grad_(False)
    vae32 = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval().requires_grad_(False)
    vae32.enable_slicing()
    h14_eval = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / "models" / EMB_KEYS[0]), torch_dtype=dt).to(dev).eval().requires_grad_(False)

    lat50 = torch.from_numpy(np.load(PREP / "lat512_train.npy")[res["sel50_idx"]]).float()
    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    H14_gt50 = emb_tr[EMB_KEYS[0]][res["sel50_idx"]].float()
    with torch.inference_mode():
        gt_rgb50 = torch.cat([
            (vae32.decode(lat50[j:j + 20].to(dev, torch.float32) / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
            for j in range(0, 500, 20)
        ])
        nets = _nets()
        gt_feats = extract_eval_feats(gt_rgb50, nets, dev=dev)

    res_p = DRAFT_DIR / "exp6_draft_dev4_results.json"
    out_res = json.loads(res_p.read_text()) if res_p.exists() else {"arms": {}}
    for arm_name, cp in ckpts.items():
        if arm_name in out_res["arms"]:
            print(f"[CACHED EVAL] {arm_name}", flush=True)
            continue
        out_res["arms"][arm_name] = evaluate_draft_arm(
            arm_name, cp, DEV_SUBS, res["sel50_idx"], z0_by_sub, z0_oof_pack,
            pipe.unet, pipe.scheduler, vae32, adapter, proj, pe, ppe, atid,
            gt_rgb50, gt_feats, nets, h14_eval, H14_gt50, dev=dev,
        )
        out_res["wall_s"] = round(time.time() - t_all, 1)
        write_json(res_p, out_res)

    print(f"=== EXP 6 (DRaFT-INTRA4 DEV-4 COMPARISON) DONE in {round(time.time()-t_all, 1)}s -> {res_p} ===", flush=True)


if __name__ == "__main__":
    main()
