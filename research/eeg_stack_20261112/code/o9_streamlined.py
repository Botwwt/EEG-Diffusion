"""Streamlined 8-checkpoint EEG-to-Image Pipeline O9 / O9-0tta / O9b (research/eeg_stack_20261112/code/o9_streamlined.py).

Compresses O8's 92 checkpoints and 3,900 TTA steps down to 1 unified encoder architecture
(7 seeds KVEBPyrEncoder) + 1 unchanged BoundedConditionAdapter (8 checkpoints/subject):
  1. kv_metric : Closed-form 70-layer SDXL-Turbo UNet IPAttnProcessor2_0 Key-Value pullback
                 operator L_kv (eff_rank = 237.1 / 2048).
  2. enc       : KVEBPyrEncoder (within-fold Empirical-Bayes TargetSNRGate + nested horizons
                 0-1000 / 0-700 / 0-480 ms) trained strictly with 2-term CenteredKVLoss.
  3. head      : 1 unchanged BoundedConditionAdapter trained on 7-seed 5-fold OOF via DDPM.
  4. cond      : O9-0tta (0 TTA steps) and O9 (350 centered L_kv split-half TTA steps + bag10
                 at r_eff=8.0), restored to exact training-cone marginal (w=1.00).
  5. eval      : Native 512px v2 start latents + gen_exp.py (K=4, nsel=1, no blur; --tmf for O9b).
"""
import argparse
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import HIVE, PREP, PRIOR, ROOT, RUNS, TURBO, concept_folds, write_json
from deep_enc import CenteredKVLoss, KVEBPyrEncoder
from train_flex import ivw_mean, shift_batch

CODE_DIR = Path(__file__).resolve().parent
_eeg_dir = CODE_DIR.parents[2] / "eeg"
if not _eeg_dir.exists():
    _eeg_dir = CODE_DIR.parents[1] / "eeg"
sys.path.insert(0, str(_eeg_dir))
from brainhive_bounded_component_v1 import BoundedConditionAdapter  # noqa: E402

sys.path.insert(0, str(HIVE))
from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel  # noqa: E402

nz = F.normalize
ALL_SUBS = tuple(range(1, 11))
SEEDS_7 = (2025, 2026, 2027, 2028, 2029, 2030, 2031)
ENC_TAG = "o9_uni"


def last_norm_from_deep(sd):
    if "out_block.norm.weight" in sd:
        return sd["out_block.norm.weight"].float(), sd["out_block.norm.bias"].float()
    idxs_b = [int(k.split(".")[2]) for k in sd if k.startswith("block.layers.") and k.endswith(".norm.weight")]
    if idxs_b:
        li = max(idxs_b)
        return sd[f"block.layers.{li}.norm.weight"].float(), sd[f"block.layers.{li}.norm.bias"].float()
    idxs = [int(k.split(".")[3]) for k in sd if k.startswith("backbone.block.layers.") and k.endswith(".norm.weight")]
    li = max(idxs)
    return sd[f"backbone.block.layers.{li}.norm.weight"].float(), sd[f"backbone.block.layers.{li}.norm.bias"].float()


def detect_gpus(gpus_arg="", min_free_mb=18000):
    if gpus_arg:
        return [int(g.strip()) for g in gpus_arg.split(",") if g.strip()]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        gpus = [int(ln.split(",")[0]) for ln in out.strip().splitlines() if int(ln.split(",")[1]) >= min_free_mb]
        if gpus:
            return gpus
    except Exception:
        pass
    return list(range(max(1, torch.cuda.device_count())))


def run_buckets_subproc(task_list, gpus, phase_tag, bs_head=32):
    if not task_list:
        return
    with tempfile.TemporaryDirectory(prefix=f"o9_{phase_tag}_") as tmp_s:
        wdir = Path(tmp_s)
        buckets = [task_list[i::len(gpus)] for i in range(len(gpus))]
        procs = []
        for gi, gpu in enumerate(gpus):
            if not buckets[gi]:
                continue
            in_f = wdir / f"{phase_tag}_gpu{gpu}.json"
            in_f.write_text(json.dumps({"phase": phase_tag, "tasks": buckets[gi], "bs": bs_head}))
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            procs.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker_in", str(in_f)], env=env))
        for pr in procs:
            rc = pr.wait()
            if rc != 0:
                raise RuntimeError(f"Worker failed with rc={rc} in phase={phase_tag}")


def _oof_200way(pred_u, tgt_u, ho_idx):
    t200 = torch.arange(200, device=pred_u.device)
    accs = []
    for ch in range(0, len(ho_idx) - 199, 200):
        sl = ho_idx[ch:ch + 200]
        accs.append(float(((pred_u[sl] @ tgt_u[sl].T).argmax(1) == t200).float().mean()))
    return float(np.mean(accs))


def _bag10_forward(model, x_te, sub, dev):
    perm_bag = torch.randperm(80, generator=torch.Generator().manual_seed(7 + sub)).to(dev)
    bag_ins = torch.cat([ivw_mean(x_te[:, perm_bag[g_ * 8:(g_ + 1) * 8]]) for g_ in range(10)], dim=0)
    return model(bag_ins, r_eff=8.0).float().view(10, 200, 1024).mean(0)


def _cone_params(F_tr_cpu):
    R_tr = float(F_tr_cpu.norm(dim=1).mean())
    mu_tr = nz(F_tr_cpu.float(), dim=1).mean(0, keepdim=True)
    alpha_tr = float(mu_tr.norm())
    beta_tr = float((1.0 - alpha_tr ** 2) ** 0.5)
    u_tr = mu_tr / alpha_tr
    return R_tr, alpha_tr, beta_tr, u_tr


def _comb_cone(o_main, o_bag, F_tr_cpu):
    R_tr, alpha_tr, beta_tr, u_tr = _cone_params(F_tr_cpu)

    def _perp_u(v):
        cu = nz(v.float(), dim=1)
        dc = nz(cu - cu.mean(0, keepdim=True), dim=1)
        return nz(dc - (dc @ u_tr.T) * u_tr, dim=1)

    d_comb = nz(0.85 * _perp_u(o_main) + 0.15 * _perp_u(o_bag), dim=1)
    return (R_tr * nz(alpha_tr * u_tr + beta_tr * d_comb, dim=1)).numpy().astype(np.float32)


def _norm_match(t_list, ref_n):
    m = torch.stack(t_list, dim=0).mean(0)
    return m * (ref_n / m.norm(dim=1, keepdim=True).clamp_min(1e-8))


def train_one_enc(sub, fold, seed, epochs=32, lr=5e-4, dev="cuda:0"):
    out = RUNS / "enc" / ENC_TAG / f"sub-{sub:02d}" / f"fold{fold}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    done_basic = (out / "metrics.json").exists() and (out / "brain_model.pt").exists()
    if fold >= 0 and done_basic and (out / "cond_oof_r4.npy").exists() and (out / "oof_index.npy").exists():
        return
    if fold < 0 and done_basic and (out / "cond_test_r80.npy").exists() and (out / "cond_test_bag10.npy").exists():
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x_raw = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_train.npy")).float().to(dev)
    x_mean = x_raw.mean(1).contiguous()
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float().to(dev)
    F_u = nz(F_tr, dim=1)
    concept = torch.from_numpy(np.load(PREP / "train_concept.npy")).long().to(dev)
    img_fold = torch.from_numpy(concept_folds(1654, 5))[concept.cpu()].to(dev)

    if fold < 0:
        tr_idx = torch.arange(len(concept), device=dev)
        ho_idx = torch.empty(0, dtype=torch.long, device=dev)
    else:
        tr_idx = torch.nonzero(img_fold != fold).squeeze(1)
        ho_idx = torch.nonzero(img_fold == fold).squeeze(1)
    n_tr = len(tr_idx)

    kv_mod = CenteredKVLoss(str(PREP / "unet_kv_metric.pt"), device=dev)
    kv_mod.calibrate_fold(F_tr, tr_idx)

    model = KVEBPyrEncoder(t_len=250, c_in=17, hidden_size=1024, num_layers=2, dropout=0.15, multi_horizon=True).to(dev)
    model.eb.fit_from_fold(x_raw[tr_idx])

    bs = 1024
    logit_scale = nn.Parameter(torch.tensor(2.6592, device=dev))
    decay, no_decay = [], []
    for n, p_ in list(model.named_parameters()) + [("logit_scale", logit_scale)]:
        (no_decay if (n.endswith("bias") or "norm" in n or n in ("logit_scale", "horizon_logits")) else decay).append(p_)
    params = decay + no_decay
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.995), eps=1e-8,
    )
    spr = math.ceil(n_tr / bs)
    total = spr * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 0.5 * (1.0 + math.cos(math.pi * s / total))))
    g = torch.Generator(device=dev).manual_seed(seed)

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_tr, device=dev, generator=g)
        for b in range(spr):
            idx = tr_idx[perm[b * bs:(b + 1) * bs]]
            xb = shift_batch(x_mean[idx], 2, g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c_raw = model(xb, r_eff=4.0)
                loss = kv_mod(c_raw, F_tr[idx], logit_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

    train_s = time.time() - t0
    model.eval()
    rec = {
        "sub": sub,
        "arm": "A3_kveb_msh_kv",
        "fold": fold,
        "seed": seed,
        "params": sum(p_.numel() for p_ in model.parameters()),
        "train_s": train_s,
        "args": {"arch": "KVEBPyrEncoder", "loss": "CenteredKVLoss", "epochs": epochs, "lr": lr, "t0": 0, "t1": 250},
    }

    if len(ho_idx) > 0:
        with torch.no_grad():
            outs, kvs_p, kvs_t = [], [], []
            for s in range(0, len(ho_idx), 1024):
                sl = ho_idx[s:s + 1024]
                co = model(x_mean[sl], r_eff=4.0).float()
                outs.append(co)
                kvs_p.append(kv_mod.project_kv(nz(co, dim=-1) * kv_mod.nu_bar))
                kvs_t.append(kv_mod.project_kv(F_u[sl] * kv_mod.nu_bar))
            c_oof = torch.cat(outs, dim=0)
            kv_p = torch.cat(kvs_p, dim=0)
            kv_t = torch.cat(kvs_t, dim=0)

        cu = nz(c_oof, dim=1)
        fu = F_u[ho_idx]
        idx_rel = torch.arange(len(ho_idx), device=dev)
        u_tr_axis = nz(kv_mod.mu_clip, dim=1)
        cu_perp = nz((cu - cu.mean(0, keepdim=True)) - ((cu - cu.mean(0, keepdim=True)) @ u_tr_axis.T) * u_tr_axis, dim=1)
        fu_perp = nz(fu - (fu @ u_tr_axis.T) * u_tr_axis, dim=1)

        rec["oof_cos"] = float((cu * fu).sum(1).mean())
        rec["oof_200w"] = _oof_200way(cu, fu, idx_rel)
        rec["oof_perp_200w"] = _oof_200way(cu_perp, fu_perp, idx_rel)
        rec["kv_mse"] = float(F.mse_loss(kv_p, kv_t).item())
        rec["kv_c_200w"] = _oof_200way(nz(kv_p - kv_mod.mu_kv, dim=1), nz(kv_t - kv_mod.mu_kv, dim=1), idx_rel)
        np.save(out / "cond_oof_r4.npy", c_oof.cpu().numpy())
        np.save(out / "oof_index.npy", ho_idx.cpu().numpy())
    else:
        x_te = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_test.npy")).float().to(dev)
        with torch.no_grad():
            c_ivw = model(ivw_mean(x_te), r_eff=72.0).float()
            c_bag10 = _bag10_forward(model, x_te, sub, dev)
        np.save(out / "cond_test_r80.npy", c_ivw.cpu().numpy())
        np.save(out / "cond_test_bag10.npy", c_bag10.cpu().numpy())

    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "brain_model.pt")
    write_json(out / "metrics.json", rec)
    del model, kv_mod, x_raw, x_mean, F_tr, F_u
    gc.collect()
    torch.cuda.empty_cache()


def ensure_test_exports(sub, dev="cuda:0"):
    missing = []
    for sd in SEEDS_7:
        d = RUNS / "enc" / ENC_TAG / f"sub-{sub:02d}" / f"fold-1_seed{sd}"
        assert (d / "brain_model.pt").exists(), f"Missing brain_model.pt at {d}"
        if not (d / "cond_test_r80.npy").exists() or not (d / "cond_test_bag10.npy").exists():
            missing.append((sd, d))
    if not missing:
        return
    x_te = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_test.npy")).float().to(dev)
    ivw_in = ivw_mean(x_te)
    for sd, d in missing:
        model = KVEBPyrEncoder(t_len=250, c_in=17, hidden_size=1024, num_layers=2, dropout=0.15, multi_horizon=True).to(dev)
        model.load_state_dict(torch.load(d / "brain_model.pt", map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            if not (d / "cond_test_r80.npy").exists():
                np.save(d / "cond_test_r80.npy", model(ivw_in, r_eff=72.0).float().cpu().numpy())
            if not (d / "cond_test_bag10.npy").exists():
                np.save(d / "cond_test_bag10.npy", _bag10_forward(model, x_te, sub, dev).cpu().numpy())
        del model
    del x_te, ivw_in
    gc.collect()
    torch.cuda.empty_cache()


def load_7seed_oof_and_affines(sub):
    concept = torch.from_numpy(np.load(PREP / "train_concept.npy"))
    fold = torch.from_numpy(concept_folds(1654, 5))[concept]
    C = torch.zeros(16540, 1024, dtype=torch.float32)
    W = torch.zeros(5, 1024, dtype=torch.float32)
    B = torch.zeros(5, 1024, dtype=torch.float32)
    for k in range(5):
        ref_d = RUNS / "enc" / ENC_TAG / f"sub-{sub:02d}" / f"fold{k}_seed{SEEDS_7[0]}"
        idx = torch.from_numpy(np.load(ref_d / "oof_index.npy"))
        c0 = torch.from_numpy(np.load(ref_d / "cond_oof_r4.npy")).float()
        cs = [c0] + [
            torch.from_numpy(np.load(RUNS / "enc" / ENC_TAG / f"sub-{sub:02d}" / f"fold{k}_seed{sd}" / "cond_oof_r4.npy")).float()
            for sd in SEEDS_7[1:]
        ]
        m = torch.stack(cs, dim=0).mean(0)
        C[idx] = m * (c0.norm(dim=1, keepdim=True) / m.norm(dim=1, keepdim=True).clamp_min(1e-8))
        sd_pt = torch.load(ref_d / "brain_model.pt", map_location="cpu", weights_only=True)
        W[k], B[k] = last_norm_from_deep(sd_pt)
    return C, W, B, fold


def _run_ddpm_head_loop(sub, bs, passes, dev):
    from diffusers import DDPMScheduler, StableDiffusionXLPipeline

    SEED = 1729
    torch.manual_seed(SEED)
    out = RUNS / "ord_o9_uni7_ddpm" / f"sub-{sub:02d}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "head.pt").exists() and (out / "log.json").exists():
        return

    C, W, B, fold = load_7seed_oof_and_affines(sub)
    valid = torch.nonzero(torch.from_numpy(np.load(PREP / "lat512_train_ok.npy"))).squeeze(1)

    C, W, B, fold = C.to(dev), W.to(dev), B.to(dev), fold.to(dev)
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).to(dev)
    lat_tr = torch.from_numpy(np.load(PREP / "lat512_train.npy")).to(dev)

    pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=torch.float16, variant="fp16")
    IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(pipe.unet)
    pipe.to(dev, torch.float16)
    P = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, torch.float16).eval().requires_grad_(False)
    A = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, torch.float16).eval().requires_grad_(False)
    for m in pipe.components.values():
        if isinstance(m, torch.nn.Module):
            m.eval().requires_grad_(False)
    pipe.unet.train()
    pipe.unet.enable_gradient_checkpointing()
    with torch.no_grad():
        prompt, _, pooled, _ = pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False)
    prompt = prompt.detach().clone().to(torch.bfloat16)
    pooled = pooled.detach().clone().to(torch.bfloat16)
    time_ids = pipe._get_add_time_ids(
        (512, 512), (0, 0), (512, 512), dtype=torch.bfloat16, text_encoder_projection_dim=pooled.shape[-1]
    ).to(dev)

    ddpm = DDPMScheduler.from_config(pipe.scheduler.config)
    del pipe.vae, pipe.text_encoder, pipe.text_encoder_2
    torch.cuda.empty_cache()

    head = BoundedConditionAdapter(predict_risk=False).to(dev)
    lr_eff = 1e-4 * math.sqrt(bs / 16.0)
    opt = torch.optim.AdamW(head.parameters(), lr=lr_eff, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    total = int(len(valid) * passes)
    order_rng = torch.Generator().manual_seed(SEED)
    noise_rng = torch.Generator(dev).manual_seed(SEED)
    order = torch.cat([valid[torch.randperm(len(valid), generator=order_rng)] for _ in range(int(np.ceil(passes)))])[:total]
    steps_n = (total + bs - 1) // bs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_n, eta_min=lr_eff * 0.1)

    log, t0 = [], time.time()
    acc = dict(main=0.0, aux=0.0, n=0)
    for step, s in enumerate(range(0, total, bs)):
        i = order[s:s + bs].to(dev)
        fk = fold[i]
        c, w, b = C[i], W[fk], B[fk]
        latent = lat_tr[i].float()
        eps = torch.randn(latent.shape, device=dev, generator=noise_rng)
        t = torch.randint(0, 1000, (len(i),), device=dev, generator=noise_rng)
        z_in = ddpm.add_noise(latent, eps, t)

        opt.zero_grad(set_to_none=True)
        changed, _ = head(c, w, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                u_tgt = P(A(F_tr[i]))
            u_stu = P(A(changed))
            n_cur = len(z_in)
            pred_stu = pipe.unet(
                z_in.to(pipe.unet.dtype),
                t,
                encoder_hidden_states=prompt.expand(n_cur, -1, -1),
                added_cond_kwargs=dict(text_embeds=pooled.expand(n_cur, -1), time_ids=time_ids.expand(n_cur, -1)),
                cross_attention_kwargs={"ip_hidden_states": u_stu},
                return_dict=False,
            )[0].float()
            main_loss = F.mse_loss(pred_stu, eps)
            aux_cos = (1.0 - F.cosine_similarity(u_stu.float(), u_tgt.float(), dim=-1)).mean()
            loss = main_loss + 0.1 * aux_cos

        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        sched.step()
        acc["main"] += float(main_loss)
        acc["aux"] += float(aux_cos)
        acc["n"] += 1
        if (step + 1) % 200 == 0 or s + bs >= total:
            rec = dict(step=step + 1, main=acc["main"] / acc["n"], aux_a=acc["aux"] / acc["n"], seconds=time.time() - t0)
            log.append(rec)
            acc = dict(main=0.0, aux=0.0, n=0)

    torch.save({k: v.cpu() for k, v in head.state_dict().items()}, out / "head.pt")
    write_json(out / "log.json", dict(sub=sub, mode="ddpm", enc_tag=ENC_TAG, seeds=list(SEEDS_7), passes=passes, bs=bs, updates=steps_n, seed=SEED, log=log))
    del pipe, P, A, F_tr, lat_tr, C, W, B
    gc.collect()
    torch.cuda.empty_cache()


def train_head_ddpm(sub, bs=32, dev="cuda:0"):
    try:
        _run_ddpm_head_loop(sub, bs=bs, passes=2.5, dev=dev)
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        _run_ddpm_head_loop(sub, bs=16, passes=2.5, dev=dev)


def _load_7seed_test_means(sub):
    ref_d = RUNS / "enc" / ENC_TAG / f"sub-{sub:02d}" / f"fold-1_seed{SEEDS_7[0]}"
    ivw0 = torch.from_numpy(np.load(ref_d / "cond_test_r80.npy")).float()
    ref_n = ivw0.norm(dim=1, keepdim=True)
    ivws = [ivw0] + [
        torch.from_numpy(np.load(RUNS / "enc" / ENC_TAG / f"sub-{sub:02d}" / f"fold-1_seed{sd}" / "cond_test_r80.npy")).float()
        for sd in SEEDS_7[1:]
    ]
    bags = [
        torch.from_numpy(np.load(RUNS / "enc" / ENC_TAG / f"sub-{sub:02d}" / f"fold-1_seed{sd}" / "cond_test_bag10.npy")).float()
        for sd in SEEDS_7
    ]
    sd_pt = torch.load(ref_d / "brain_model.pt", map_location="cpu", weights_only=True)
    w, b = last_norm_from_deep(sd_pt)
    return _norm_match(ivws, ref_n), _norm_match(bags, ref_n), ref_n, w, b


def export_sub_conditions(sub, dev="cuda:0"):
    F_tr_cpu = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    R_tr = float(F_tr_cpu.norm(dim=1).mean())
    m_ivw, m_bag, ref_n, ref_w, ref_b = _load_7seed_test_means(sub)

    head = BoundedConditionAdapter(predict_risk=False).eval()
    head.load_state_dict(torch.load(RUNS / "ord_o9_uni7_ddpm" / f"sub-{sub:02d}" / "head.pt", map_location="cpu", weights_only=True))

    out_0tta = RUNS / "cond" / "cone_o9_uni7_0tta_ddpm_w1.00" / f"sub-{sub:02d}.npy"
    if not out_0tta.exists():
        with torch.no_grad():
            o_ivw, _ = head(m_ivw, ref_w.expand_as(m_ivw), ref_b.expand_as(m_ivw))
            o_bag, _ = head(m_bag, ref_w.expand_as(m_bag), ref_b.expand_as(m_bag))
        out_0tta.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_0tta, _comb_cone(o_ivw, o_bag, F_tr_cpu))

    out_tta = RUNS / "cond" / "cone_o9_uni7_kvtta_ddpm_w1.00" / f"sub-{sub:02d}.npy"
    if out_tta.exists():
        return

    raw_cache = RUNS / "cond" / "raw_o9_uni7_kvtta" / f"sub-{sub:02d}.pt"
    if raw_cache.exists():
        m_tta = torch.load(raw_cache, map_location="cpu", weights_only=True)["m_tta"]
    else:
        x_te = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_test.npy")).float().to(dev)
        full_in = ivw_mean(x_te)
        perm_bag = torch.randperm(80, generator=torch.Generator().manual_seed(7 + sub)).to(dev)
        bag_ins = torch.cat([ivw_mean(x_te[:, perm_bag[g_ * 8:(g_ + 1) * 8]]) for g_ in range(10)], dim=0)

        kv_pack = torch.load(PREP / "unet_kv_metric.pt", map_location="cpu", weights_only=True)
        L_kv_t = kv_pack["L_kv"].float().T.contiguous().to(dev)
        P = IPProjectionModel(embed_dim=1024, cross_attention_dim=2048, sequence_length=4)
        P.load_state_dict(kv_pack["proj_state"])
        P = P.to(dev).eval().requires_grad_(False)
        A_m = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev).eval().requires_grad_(False)

        tgt = torch.arange(200, device=dev)
        tta_preds = []
        for sd_seed in SEEDS_7:
            d = RUNS / "enc" / ENC_TAG / f"sub-{sub:02d}" / f"fold-1_seed{sd_seed}"
            m = KVEBPyrEncoder(t_len=250, c_in=17, hidden_size=1024, num_layers=2, dropout=0.15, multi_horizon=True).to(dev)
            m.load_state_dict(torch.load(d / "brain_model.pt", map_location="cpu", weights_only=True))
            m.eval().requires_grad_(False)
            for name, param in m.named_parameters():
                parts = name.split(".")
                if len(parts) == 4 and parts[0] == "stems" and parts[2] == "0":
                    param.requires_grad_(True)
            train_params = [p for p in m.parameters() if p.requires_grad]
            with torch.no_grad():
                c0 = m(full_in, r_eff=72.0).float()

            opt = torch.optim.Adam(train_params, lr=1e-4)
            g = torch.Generator(device=dev).manual_seed(3000 + sub * 37 + sd_seed)
            for _ in range(50):
                perm = torch.rand(200, 80, device=dev, generator=g).argsort(1)
                xa = ivw_mean(torch.gather(x_te, 1, perm[:, :40, None, None].expand(-1, -1, 17, 250)))
                xb = ivw_mean(torch.gather(x_te, 1, perm[:, 40:, None, None].expand(-1, -1, 17, 250)))
                ca = m(xa, r_eff=36.0).float()
                cb = m(xb, r_eff=36.0).float()
                ka = (P(A_m(nz(ca, dim=1) * R_tr)) @ L_kv_t).flatten(1)
                kb = (P(A_m(nz(cb, dim=1) * R_tr)) @ L_kv_t).flatten(1)
                ka = nz(ka - ka.mean(0, keepdim=True), dim=1)
                kb = nz(kb - kb.mean(0, keepdim=True), dim=1)
                lg = (ka @ kb.T) / 0.1
                loss_nce = 0.5 * (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.T, tgt))
                cf = m(full_in, r_eff=72.0).float()
                anc = ((cf - c0).pow(2).sum(1) / c0.pow(2).sum(1)).mean()
                (loss_nce + 1.0 * anc).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                full_tta = m(full_in, r_eff=72.0).float()
                bag_tta = m(bag_ins, r_eff=8.0).float().view(10, 200, 1024).mean(0)
                tta_preds.append((0.65 * full_tta + 0.35 * bag_tta).cpu())
            del m

        m_tta = _norm_match(tta_preds, ref_n)
        raw_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"m_tta": m_tta}, raw_cache)
        del x_te, full_in, bag_ins, P, A_m, L_kv_t
        gc.collect()
        torch.cuda.empty_cache()

    with torch.no_grad():
        o_tta, _ = head(m_tta, ref_w.expand_as(m_tta), ref_b.expand_as(m_tta))
        o_bag, _ = head(m_bag, ref_w.expand_as(m_bag), ref_b.expand_as(m_bag))
    out_tta.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_tta, _comb_cone(o_tta, o_bag, F_tr_cpu))


def run_gen_exp_item(task):
    sub, arm_name, cond_rel, lat_rel, use_tmf = task
    out = RUNS / "img" / f"s{sub:02d}_{arm_name}"
    if (out / "metrics.json").exists():
        return
    cmd = [
        sys.executable,
        str(CODE_DIR / "gen_exp.py"),
        "--run", f"t0_r4/sub-{sub:02d}/fold-1_seed2025",
        "--cond_file", f"{cond_rel}/sub-{sub:02d}.npy",
        "--lat_init", f"sub-{sub:02d}/{lat_rel}",
        "--smap", "c0.625,0.5,1.0",
        "--low_keep", "8,0.22,0.04",
        "--K", "4",
        "--save", "0",
        "--name", f"s{sub:02d}_{arm_name}",
    ]
    if use_tmf:
        cmd.extend(["--tmf", "1,0.10,0.10,0", "--tmf_cr", "1"])
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subs", default="all", help="Comma-separated subject ints or 'all'")
    p.add_argument("--mode", choices=["all", "kv", "enc", "head", "cond", "eval"], default="all")
    p.add_argument("--gpus", default="", help="Comma-separated GPU indices (default: auto-detect free GPUs)")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--worker_in", default="")
    args = p.parse_args()

    if args.worker_in:
        payload = json.loads(Path(args.worker_in).read_text())
        ph = payload["phase"]
        bs_head = int(payload.get("bs", 32))
        for t in payload["tasks"]:
            if ph == "enc":
                train_one_enc(t[0], t[1], t[2], epochs=32, lr=5e-4, dev="cuda:0")
            elif ph == "bag10":
                ensure_test_exports(t, dev="cuda:0")
            elif ph == "head":
                train_head_ddpm(t, bs=bs_head, dev="cuda:0")
            elif ph == "cond":
                export_sub_conditions(t, dev="cuda:0")
            elif ph == "gen":
                run_gen_exp_item(t)
        return

    subs = list(ALL_SUBS) if args.subs == "all" else [int(x) for x in args.subs.split(",")]
    gpus = detect_gpus(args.gpus)

    if args.mode in ("all", "kv") and not (PREP / "unet_kv_metric.pt").exists():
        subprocess.run([sys.executable, str(CODE_DIR / "kv_metric.py")], check=True)

    if args.mode in ("all", "enc"):
        enc_tasks = [(s, f, sd) for s in subs for sd in SEEDS_7 for f in (0, 1, 2, 3, 4, -1)]
        run_buckets_subproc(enc_tasks, gpus, "enc")
        run_buckets_subproc(subs, gpus, "bag10")

    if args.mode in ("all", "head"):
        run_buckets_subproc(subs, gpus, "head", bs_head=args.bs)

    if args.mode in ("all", "cond"):
        run_buckets_subproc(subs, gpus, "cond")

    if args.mode in ("all", "eval"):
        sub_csv = "all" if len(subs) == 10 else ",".join(map(str, subs))
        subprocess.run(
            [sys.executable, str(CODE_DIR / "lat512_start.py"), sub_csv, "--mode_v", "v2", "--cond_dirs", "cone_o9_uni7_0tta_ddpm_w1.00", "--tag", "lat512_o9_0tta_ddpm"],
            check=True,
        )
        subprocess.run(
            [sys.executable, str(CODE_DIR / "lat512_start.py"), sub_csv, "--mode_v", "v2", "--cond_dirs", "cone_o9_uni7_kvtta_ddpm_w1.00", "--tag", "lat512_o9_v2_ddpm"],
            check=True,
        )
        arms = [
            ("o9_0tta_w1_notmf", "cone_o9_uni7_0tta_ddpm_w1.00", "lat512_o9_0tta_ddpm_n0.80.npy", False),
            ("o9_w1_notmf", "cone_o9_uni7_kvtta_ddpm_w1.00", "lat512_o9_v2_ddpm_n0.80.npy", False),
            ("o9_w1_tmf", "cone_o9_uni7_kvtta_ddpm_w1.00", "lat512_o9_v2_ddpm_n0.80.npy", True),
        ]
        gen_tasks = [(s, a, c_r, l_r, tmf) for (a, c_r, l_r, tmf) in arms for s in subs]
        run_buckets_subproc(gen_tasks, gpus, "gen")


if __name__ == "__main__":
    main()
