"""Phase 3: Complete 2x2 Factorial + Conditioning-Layer (4x2048) Proof of Raw-eps 2x2 EEG-Swap Denoising Contrast.

Grounded directly in Phase 2's breakthrough discovery:
  - Pure Raw-eps 2x2 Swap (Arm2v2_raw_noblur, ZERO L_rep_intra, ZERO extra params) beat Arm1_RepIntra on
    L2 UNet Swap (+7.3%pp -> 81.67%), L2 dc10 (+4.3%pp -> 31.10%), FixZ0 CLIP Swap (64.34%), FixZ0 Alex5
    (22.70% / 74.30%), FreeGen CLIP Swap (66.40%), FreeGen Inception Swap (63.20%, +2.23%pp), and FreeGen
    Alex5 (28.00% / 79.26%), whereas Teacher-F_gt RB Swap (rb_cone) stalled at 75.13% UNet swap — proving
    real-image denoising supervision extracts photo-specific geometry BEYOND F_gt!

Phase 3 completes the strictly controlled factorial & architectural table (all on tr14 -> sealed fold-0, R=4 EEG):
  1. Arm1_RepIntra_1400       : Step-matched representation-only control (1,400 total steps of L_base + L_rep_intra,
                                0 extra params; takes ~70s) to rule out step-count confounds!
  2. Arm2v3_RawSwap_CondLyr   : From Arm0, ZERO L_rep_intra, pure Raw-eps 2x2 Swap + zero-init 4x2048 ResidualIPCondLayer
                                (isolates conditioning layer with pure denoising swap!).
  3. Arm3a_RepRawSwap_Cone    : From Arm1 (800->1400 steps, EXACT 0 extra params, step-matched to Arm1_1400!), adding
                                Raw-eps 2x2 Swap (w_rec=1.5, w_swap=0.45) + 0.5 L_rep_intra + 1.0 L_base.
  4. Arm3b_RepRawSwap_CondLyr : Same as Arm3a + zero-init 4x2048 ResidualIPCondLayer (the complete bottom-up design:
                                clean 1024-D semantic cone + 4x2048 beyond-F_gt denoising-swap IP tokens!).
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

from common import PREP, PRIOR, TURBO, write_json
from exp0_intra10_swap_gate import DEV_SUBS, OUT_DIR, fit_tr14_ridge_z0
from exp1_swapdiff_3arm import (
    ResidualIPCondLayer,
    build_tr14_resources,
    compute_base_anchor_loss,
    evaluate_arm_complete,
    forward_with_trunk,
    per_sample_cone,
    train_arm1_rep_intra,
)
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder
from train_flex import shift_batch

from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel

nz = F.normalize


def compute_rep_intra_loss(model, res, sid_c, c_sel, g, dev, pack_cone, ls_in):
    """Same exact within-concept 10-way InfoNCE + 2x2 swap in representation space as Arm1_RepIntra."""
    kv_full = res["kv_full"]
    u_tr = kv_full.u_tr
    c_groups = res["tr14_c_groups"]
    g_locs = c_groups[c_sel].reshape(-1)  # [8*10 = 80]
    g_locs_t = torch.from_numpy(g_locs).long()
    x_c = res["x_tr_all"][sid_c][g_locs_t].float().mean(1).to(dev)
    x_c = shift_batch(x_c, 2, g)
    fp, hp, bp, _ = forward_with_trunk(model, x_c, sid_c, r_eff=4.0)

    idx_c_dev = g_locs_t.to(dev)
    with torch.no_grad():
        ff = res["F_tr"][idx_c_dev].float()
        uf = nz(ff, dim=-1)
        df_t = nz(uf - (uf @ u_tr.T) * u_tr, dim=-1)
        ykv_t = nz(kv_full.kv.project_kv(ff) - kv_full.kv.mu_kv, dim=-1)
        ht = nz(res["H14_tr"][idx_c_dev].float(), dim=-1)
        bt = nz(res["B32_tr"][idx_c_dev].float(), dim=-1)

    c_cone = per_sample_cone(fp, pack_cone["R"], pack_cone["alpha"], pack_cone["beta"], u_tr)
    df_p = nz(nz(fp.float(), dim=-1) - (nz(fp.float(), dim=-1) @ u_tr.T) * u_tr, dim=-1)
    zk_p = nz(kv_full.kv.project_kv(c_cone) - kv_full.kv.mu_kv, dim=-1)
    hp_n = nz(hp.float(), dim=-1)
    bp_n = nz(bp.float(), dim=-1)
    sc_in = ls_in.float().exp().clamp(5.0, 50.0)
    nc = len(c_sel)

    def _intra(P, T):
        Pb, Tb = P.view(nc, 10, -1), T.view(nc, 10, -1)
        S = torch.bmm(Pb, Tb.transpose(1, 2))
        tgt = torch.arange(10, device=dev).unsqueeze(0).expand(nc, 10)
        ce = 0.5 * (
            F.cross_entropy((sc_in * S).reshape(nc * 10, 10), tgt.reshape(nc * 10))
            + F.cross_entropy((sc_in * S.transpose(1, 2)).reshape(nc * 10, 10), tgt.reshape(nc * 10))
        )
        iu, ju = torch.triu_indices(10, 10, offset=1, device=dev)
        delta = (S[:, iu, ju] + S[:, ju, iu]) - (S[:, iu, iu] + S[:, ju, ju])
        return ce + 0.5 * F.softplus(8.0 * delta).mean()

    return 0.35 * _intra(df_p, df_t) + 0.35 * _intra(zk_p, ykv_t) + 0.20 * _intra(hp_n, ht) + 0.10 * _intra(bp_n, bt)


def train_raw_swap_v3(
    tag: str,
    init_ckpt: Path,
    res: dict,
    pipe_unet,
    alphas_cp: torch.Tensor,
    adapter,
    proj,
    pe,
    ppe,
    atid,
    use_cond_layer: bool,
    w_rep_intra: float,
    w_rec: float = 1.5,
    w_swap: float = 0.45,
    steps: int = 600,
    seed: int = 2025,
    dev: str = "cuda:0",
) -> Path:
    out_p = OUT_DIR / f"{tag}_st{steps}_s{seed}.pt"
    if out_p.exists():
        print(f"[CACHED] {out_p}", flush=True)
        return out_p

    t0 = time.time()
    pack = torch.load(init_ckpt, map_location="cpu", weights_only=True)
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=0.20, in_drop=0.15).to(dev)
    model.load_state_dict({k: v.float() for k, v in pack["state_dict"].items()})
    cond_layer = ResidualIPCondLayer(d_in=1024, rank=128, n_tok=4, d_tok=2048, max_rel_norm=0.35).to(dev) if use_cond_layer else None

    R_tr, a_tr, b_tr = pack["cone"]["R"], pack["cone"]["alpha"], pack["cone"]["beta"]
    u_tr = pack["cone"]["u_tr"].to(dev).float()

    ls_f = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_h = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_b = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_x = nn.Parameter(torch.tensor(2.8, device=dev))
    ls_in = nn.Parameter(torch.tensor(2.8, device=dev))

    params = [{"params": list(model.parameters()) + [ls_f, ls_h, ls_b, ls_x, ls_in], "lr": 1.0e-4, "weight_decay": 0.03}]
    if cond_layer is not None:
        params.append({"params": list(cond_layer.parameters()), "lr": 3.0e-4, "weight_decay": 0.01})
    opt = torch.optim.AdamW(params, betas=(0.9, 0.995))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    pipe_unet.eval().requires_grad_(False)
    pipe_unet.enable_gradient_checkpointing()
    adapter.eval().requires_grad_(False)
    proj.eval().requires_grad_(False)

    g = torch.Generator(device=dev).manual_seed(seed * 89 + 23)
    rng = np.random.default_rng(seed * 89 + 23)
    c_groups = res["tr14_c_groups"]
    n_c = len(c_groups)
    n_tr = len(res["tr_idx_np"])

    # Active 4-step timesteps + small t=999 weight for global semantic-layout consistency
    T_CHOICES = [999, 749, 499, 249]
    T_PROBS = [0.10, 0.35, 0.35, 0.20]
    BETA_RAW = {999: 500.0, 749: 180.0, 499: 120.0, 249: 160.0}

    n_micro = 2
    pairs_per_mb = 8  # 16 same-subject same-concept pairs/step

    print(
        f"[TRAIN {tag}] init={init_ckpt.name} steps={steps} use_cl={use_cond_layer} "
        f"w_rep_intra={w_rep_intra} w_rec={w_rec} w_swap={w_swap}",
        flush=True,
    )
    model.train()
    if cond_layer is not None:
        cond_layer.train()

    run_sw, run_delta, run_rec = [], [], []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        # 1. Sharp inter-concept anchor (+ optional L_rep_intra)
        idx_dev = torch.randint(0, n_tr, (512,), device=dev, generator=g)
        idx_cpu = idx_dev.cpu()
        s_pair = rng.choice(res["subject_ids"], size=2, replace=len(res["subject_ids"]) == 1)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            l_anchor = compute_base_anchor_loss(
                model, res, int(s_pair[0]), int(s_pair[1]), idx_cpu, idx_dev, g, dev,
                ls_f, ls_h, ls_b, ls_x, remove_fn_mask=True,
            )
            if w_rep_intra > 0.0:
                sid_c = int(rng.choice(res["subject_ids"]))
                c_sel = rng.choice(n_c, size=8, replace=False)
                l_in = compute_rep_intra_loss(model, res, sid_c, c_sel, g, dev, pack["cone"], ls_in)
                l_anchor = l_anchor + w_rep_intra * l_in
        l_anchor.backward()
        if step == 1:
            g_anc = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in model.parameters() if p.grad is not None))

        # 2. Doubly-balanced 2x2 Raw-eps EEG-Swap Denoising Contrast through FROZEN SDXL-Turbo UNet
        step_rec_val, step_sw_val, step_d_val, step_win = 0.0, 0.0, 0.0, 0.0
        for mb in range(n_micro):
            sid = int(rng.choice(res["subject_ids"]))
            c_sel = rng.choice(n_c, size=pairs_per_mb, replace=False)
            loc_a, loc_b = [], []
            for ci in c_sel:
                ab_idx = rng.choice(10, size=2, replace=False)
                loc_a.append(c_groups[ci, ab_idx[0]])
                loc_b.append(c_groups[ci, ab_idx[1]])
            loc_a_t = torch.tensor(loc_a, dtype=torch.long)
            loc_b_t = torch.tensor(loc_b, dtype=torch.long)

            ea = res["x_tr_all"][sid][loc_a_t].float().mean(1).to(dev)
            eb = res["x_tr_all"][sid][loc_b_t].float().mean(1).to(dev)
            ea, eb = shift_batch(ea, 2, g), shift_batch(eb, 2, g)

            z0_a = res["lat_tr_cpu"][loc_a_t].float().to(dev)
            z0_b = res["lat_tr_cpu"][loc_b_t].float().to(dev)

            t_int = int(rng.choice(T_CHOICES, p=T_PROBS))
            beta_t = BETA_RAW[t_int]
            ab_val = float(alphas_cp[t_int])
            sq_ab, sq_1mab = math.sqrt(ab_val), math.sqrt(1.0 - ab_val)

            eps_a = torch.randn(z0_a.shape, device=dev, generator=g, dtype=torch.float32)
            eps_b = torch.randn(z0_b.shape, device=dev, generator=g, dtype=torch.float32)
            zt_a = (sq_ab * z0_a + sq_1mab * eps_a).to(torch.bfloat16)
            zt_b = (sq_ab * z0_b + sq_1mab * eps_b).to(torch.bfloat16)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                e_both = torch.cat([ea, eb], dim=0)
                fp_both, _, _, h_both = forward_with_trunk(model, e_both, sid, r_eff=4.0)
                c_both = per_sample_cone(fp_both, R_tr, a_tr, b_tr, u_tr).to(torch.bfloat16)
                ips_both = proj(adapter(c_both))
                if cond_layer is not None:
                    ips_both = cond_layer(h_both, ips_both)
                ips_a, ips_b = ips_both[:pairs_per_mb], ips_both[pairs_per_mb:]

                zt_4 = torch.cat([zt_a, zt_b, zt_a, zt_b], dim=0)
                ips_4 = torch.cat([ips_a, ips_b, ips_b, ips_a], dim=0)
                B4 = zt_4.shape[0]
                t_ten = torch.full((B4,), t_int, device=dev, dtype=torch.long)

                pred_4 = pipe_unet(
                    zt_4, t_ten,
                    encoder_hidden_states=pe.expand(B4, -1, -1),
                    added_cond_kwargs={"text_embeds": ppe.expand(B4, -1), "time_ids": atid.expand(B4, -1)},
                    cross_attention_kwargs={"ip_hidden_states": ips_4},
                ).sample.float()

                p_aa, p_bb, p_ab, p_ba = pred_4.chunk(4, dim=0)
                d_aa = (p_aa - eps_a).pow(2).mean(dim=(1, 2, 3))
                d_bb = (p_bb - eps_b).pow(2).mean(dim=(1, 2, 3))
                d_ab = (p_ab - eps_a).pow(2).mean(dim=(1, 2, 3))
                d_ba = (p_ba - eps_b).pow(2).mean(dim=(1, 2, 3))

                l_rec = 0.5 * (d_aa + d_bb).mean()
                delta_sw = (d_aa + d_bb) - (d_ab + d_ba)
                l_sw = F.softplus(beta_t * delta_sw).mean()
                l_diff = (w_rec * l_rec + w_swap * l_sw) / float(n_micro)

            l_diff.backward()
            step_rec_val += float(l_rec.detach()) / n_micro
            step_sw_val += float(l_sw.detach()) / n_micro
            step_d_val += float(delta_sw.detach().mean()) / n_micro
            step_win += float((delta_sw.detach() < 0.0).float().mean()) / n_micro

        if step == 1:
            g_tot = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in model.parameters() if p.grad is not None))
            g_cl = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in cond_layer.parameters() if p.grad is not None)) if cond_layer is not None else 0.0
            print(f"  [{tag} STEP-1 DIAG] grad_anc={g_anc:.3f} grad_tot={g_tot:.3f} grad_cl={g_cl:.3f} mem_MB={torch.cuda.max_memory_allocated()//(1024*1024)}", flush=True)

        all_p = list(model.parameters()) + [ls_f, ls_h, ls_b, ls_x, ls_in] + (list(cond_layer.parameters()) if cond_layer is not None else [])
        torch.nn.utils.clip_grad_norm_(all_p, 2.0)
        opt.step()
        sched.step()

        run_sw.append(step_win)
        run_delta.append(step_d_val)
        run_rec.append(step_rec_val)
        if step in (1, 10, 100, 200, 400, 600):
            elapsed = time.time() - t0
            eta = (elapsed / step) * (steps - step)
            print(
                f"  [{tag}] step {step:3d}/{steps} anc={float(l_anchor.detach()):.3f} rec={np.mean(run_rec[-30:]):.4f} "
                f"swL={step_sw_val:.3f} tr_swWin={np.mean(run_sw[-30:])*100:.1f}% delta={np.mean(run_delta[-30:]):.2e} ({elapsed:.0f}s, ETA {eta:.0f}s)",
                flush=True,
            )

    model.eval()
    if cond_layer is not None:
        cond_layer.eval()
    torch.save(
        {
            "state_dict": {k: v.half().cpu() for k, v in model.state_dict().items()},
            "cond_layer": {k: v.half().cpu() for k, v in cond_layer.state_dict().items()} if cond_layer is not None else None,
            "cone": pack["cone"],
            "arm": tag,
            "steps": steps,
        },
        out_p,
    )
    print(f"[SAVED] {out_p} ({time.time()-t0:.1f}s)", flush=True)
    return out_p


def main():
    dev = "cuda:0"
    t_all = time.time()
    arm0_ckpt = OUT_DIR / "arm0_tr14_xbrain_s2025.pt"
    arm1_ckpt = OUT_DIR / "arm1_rep_intra_st800_s2025.pt"
    res = build_tr14_resources(dev=dev)
    z0_by_sub = {sub: fit_tr14_ridge_z0(sub, res["tr_idx_np"], res["sel50_idx"], dev=dev) for sub in DEV_SUBS}

    # 1. Step-matched Arm1_RepIntra_1400 (1,400 steps total -> exact step-count control for Arm3a!)
    arm1_1400_ckpt = train_arm1_rep_intra(arm0_ckpt, res, steps=1400, seed=2025, dev=dev)

    # 2. Load frozen bf16 SDXL-Turbo UNet
    from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
    from metrics import _nets

    pipe_tr = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=torch.bfloat16, variant="fp16")
    IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(pipe_tr.unet)
    pipe_tr.to(dev, torch.bfloat16)
    with torch.no_grad():
        pe_bf, _, ppe_bf, _ = pipe_tr.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
        atid_bf = pipe_tr._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=torch.bfloat16, text_encoder_projection_dim=pipe_tr.text_encoder_2.config.projection_dim).to(dev)
        pe_bf, ppe_bf, atid_bf = pe_bf.detach().clone(), ppe_bf.detach().clone(), atid_bf.detach().clone()
    del pipe_tr.text_encoder, pipe_tr.text_encoder_2, pipe_tr.vae
    pipe_tr.text_encoder = None
    pipe_tr.text_encoder_2 = None
    pipe_tr.vae = None
    alphas_cp = pipe_tr.scheduler.alphas_cumprod.clone()
    adapter_bf = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, torch.bfloat16).eval().requires_grad_(False)
    proj_bf = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, torch.bfloat16).eval().requires_grad_(False)
    torch.cuda.empty_cache()

    # Train 3 Phase-3 models:
    arm2v3_cl = train_raw_swap_v3(
        "arm2v3_rawswap_condlyr", arm0_ckpt, res, pipe_tr.unet, alphas_cp, adapter_bf, proj_bf, pe_bf, ppe_bf, atid_bf,
        use_cond_layer=True, w_rep_intra=0.0, w_rec=1.5, w_swap=0.45, steps=600, seed=2025, dev=dev,
    )
    arm3a_cone = train_raw_swap_v3(
        "arm3a_rep_rawswap_cone", arm1_ckpt, res, pipe_tr.unet, alphas_cp, adapter_bf, proj_bf, pe_bf, ppe_bf, atid_bf,
        use_cond_layer=False, w_rep_intra=0.5, w_rec=1.5, w_swap=0.45, steps=600, seed=2025, dev=dev,
    )
    arm3b_cl = train_raw_swap_v3(
        "arm3b_rep_rawswap_condlyr", arm1_ckpt, res, pipe_tr.unet, alphas_cp, adapter_bf, proj_bf, pe_bf, ppe_bf, atid_bf,
        use_cond_layer=True, w_rep_intra=0.5, w_rec=1.5, w_swap=0.45, steps=600, seed=2025, dev=dev,
    )

    del pipe_tr, adapter_bf, proj_bf, pe_bf, ppe_bf, atid_bf
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Full Controlled Evaluation in identical fp16 OfficialTurboRunner pipeline
    dt = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=dt, variant="fp16")
    IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(pipe.unet)
    pipe.to(dev, dt)
    pipe.unet.eval().requires_grad_(False)
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        pe, _, ppe, _ = pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
        atid = pipe._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=dt, text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).to(dev)
    del pipe.text_encoder, pipe.text_encoder_2
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    p2 = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    p2.set_progress_bar_config(disable=True)
    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, dt).eval().requires_grad_(False)
    proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, dt).eval().requires_grad_(False)
    vae32 = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval().requires_grad_(False)
    vae32.enable_slicing()

    lat50 = torch.from_numpy(np.load(PREP / "lat512_train.npy")[res["sel50_idx"]]).float()
    with torch.inference_mode():
        gt_rgb50 = torch.cat([
            (vae32.decode(lat50[j:j + 20].to(dev, torch.float32) / vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
            for j in range(0, 500, 20)
        ])
        nets = _nets()
        gt_feats = {}
        for name, (mod, pre, layer) in nets.items():
            if name == "effnet":
                continue
            fl = []
            for i in range(0, 500, 50):
                y = mod(pre(gt_rgb50[i:i + 50].to(dev)))
                fl.append((y if layer is None else y[layer]).float().flatten(1).cpu())
            gt_feats[name] = torch.cat(fl, 0)

    res_p = OUT_DIR / "exp1_4arm_results.json"
    results = json.loads(res_p.read_text()) if res_p.exists() else {"arms": {}}
    for arm_name, ckpt in [
        ("Arm1_RepIntra_1400", arm1_1400_ckpt),
        ("Arm2v3_RawSwap_CondLyr", arm2v3_cl),
        ("Arm3a_RepRawSwap_Cone", arm3a_cone),
        ("Arm3b_RepRawSwap_CondLyr", arm3b_cl),
    ]:
        results["arms"][arm_name] = evaluate_arm_complete(
            arm_name, ckpt, res, pipe, p2, vae32, adapter, proj, pe, ppe, atid,
            lat50, gt_rgb50, gt_feats, z0_by_sub, dev=dev,
        )
        results["wall_s"] = round(time.time() - t_all, 1)
        write_json(res_p, results)

    print(f"=== PHASE 3 COMPLETE in {round(time.time()-t_all, 1)}s -> {res_p} ===", flush=True)


if __name__ == "__main__":
    main()
