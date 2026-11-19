"""O10: Bifurcated Factorized Forward-Operator EEG-to-Image Pipeline (research/eeg_stack_20261112/code/o10_factorized.py).

Low-level improvements grounded in Bottleneck Localization & Forward-Operator Identifiability:
  1. Drops both redundant positive-pair alignment weights (0.35, 0.25 -> 0.0, 0.0) in CenteredKVLoss,
     leaving single-term bidirectional hybrid InfoNCE in (target, UNet 70-layer Key-Value L_kv).
  2. Eliminates the 525,568-param BoundedConditionAdapter DDPM head completely (0 adapter heads per stream),
     running 350-step centered L_kv 0-head split-half TTA.
  3. Stream 1 (o10_ffull_zero00, O9-zero00-0head): (0,0) bidirectional hybrid InfoNCE directly on F_train,
     retaining instance-level retinotopic/VAE alignment & SSIM.
  4. Stream 2 (o10_fsem_fnm, O10-sem-only): (0,0) hybrid InfoNCE on VAE-marginalized F_sem with same-concept
     false-negative masking and visual-confuser margin (beta_conf = 0.15), plus FusionEncoder PhysReSyn
     combining (e_H14, e_B32) posterior expectations with ridge z_vae on the F_train[ok] cone (w = 1.00).
  5. Bifurcated Forward-Operator Synthesis (Headline O10, eta = 0.50): blends unit orthogonal tangent
     directions d_full (eta = 0.0) and d_sem_phys (eta = 1.0) on the native F_train[ok] cone (w = 1.00),
     evaluated via unchanged lat512_start.py (--mode_v v2) and gen_exp.py (K = 4 indep, nsel = 1, 2, 4,
     zero blur), alongside O10-0tta and single-stream ablations.
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

from common import EMB_KEYS, HIVE, PREP, PRIOR, RUNS, concept_folds, write_json
from deep_enc import KVEBPyrEncoder
from train_flex import ivw_mean, shift_batch

sys.path.insert(0, str(HIVE))
from main.layers import ResNet1DLayer  # noqa: E402
from main.models_adapter import FusionEncoderModel, IPProjectionModel  # noqa: E402

nz = F.normalize
CODE_DIR = Path(__file__).resolve().parent
TMP_ROOT = RUNS / "tmp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TMP_ROOT)

ALL_SUBS = tuple(range(1, 11))
SEEDS_7 = (2025, 2026, 2027, 2028, 2029, 2030, 2031)
ENC_TAG = "o10_fsem_fnm"
ENC_TAG_FULL = "o10_ffull_zero00"


def _find_script(name: str) -> Path:
    candidates = [CODE_DIR] + [Path(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    for d in candidates:
        p = d / name
        if p.exists():
            return p
    return CODE_DIR / name


def detect_gpus(gpus_arg: str = "", min_free_mb: int = 18000):
    if gpus_arg:
        return [int(g.strip()) for g in gpus_arg.split(",") if g.strip()]
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    gpus = [int(ln.split(",")[0]) for ln in out.strip().splitlines() if int(ln.split(",")[1]) >= min_free_mb]
    if not gpus:
        raise RuntimeError(f"No GPUs satisfy min_free_mb={min_free_mb}; nvidia-smi:\n{out}")
    return gpus


def run_buckets_subproc(task_list, gpus, phase_tag: str):
    if not task_list:
        return
    with tempfile.TemporaryDirectory(prefix=f"o10_{phase_tag}_", dir=str(TMP_ROOT)) as tmp_s:
        wdir = Path(tmp_s)
        buckets = [task_list[i::len(gpus)] for i in range(len(gpus))]
        procs = []
        for gi, gpu in enumerate(gpus):
            if not buckets[gi]:
                continue
            in_f = wdir / f"{phase_tag}_gpu{gpu}.json"
            in_f.write_text(json.dumps({"phase": phase_tag, "tasks": buckets[gi]}))
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["TMPDIR"] = str(TMP_ROOT)
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker_in", str(in_f)]
            procs.append((gpu, subprocess.Popen(cmd, env=env)))
        for gpu, pr in procs:
            rc = pr.wait()
            if rc != 0:
                raise RuntimeError(f"Worker failed on GPU {gpu} with rc={rc} in phase={phase_tag}")


def _cone_params(F_mat: torch.Tensor):
    f = F_mat.float()
    R_tr = float(f.norm(dim=1).mean())
    mu_tr = nz(f, dim=1).mean(0, keepdim=True)
    alpha_tr = float(mu_tr.norm())
    beta_tr = float(max(1e-8, 1.0 - alpha_tr ** 2) ** 0.5)
    u_tr = mu_tr / max(alpha_tr, 1e-8)
    return R_tr, alpha_tr, beta_tr, u_tr


def _perp_dir(v: torch.Tensor, u_axis: torch.Tensor) -> torch.Tensor:
    cu = nz(v.float(), dim=1)
    u = u_axis.to(cu.device, dtype=cu.dtype)
    dc = nz(cu - cu.mean(0, keepdim=True), dim=1)
    return nz(dc - (dc @ u.T) * u, dim=1)


def _norm_match(t_list, ref_n: torch.Tensor) -> torch.Tensor:
    m = torch.stack(t_list, dim=0).mean(0)
    return m * (ref_n / m.norm(dim=1, keepdim=True).clamp_min(1e-8))


def _bag10_forward(model, x_te: torch.Tensor, sub: int, dev: str) -> torch.Tensor:
    perm_bag = torch.randperm(80, generator=torch.Generator().manual_seed(7 + sub)).to(dev)
    bag_ins = torch.cat([ivw_mean(x_te[:, perm_bag[g_ * 8:(g_ + 1) * 8]]) for g_ in range(10)], dim=0)
    return model(bag_ins, r_eff=8.0).float().view(10, 200, 1024).mean(0)


def _oof_200way(pred_u: torch.Tensor, tgt_u: torch.Tensor, ho_idx: torch.Tensor) -> float:
    t200 = torch.arange(200, device=pred_u.device)
    accs = []
    for ch in range(0, len(ho_idx) - 199, 200):
        sl = ho_idx[ch:ch + 200]
        accs.append(float(((pred_u[sl] @ tgt_u[sl].T).argmax(1) == t200).float().mean()))
    return float(np.mean(accs))


def build_fsem_train(dev: str = "cuda:0") -> Path:
    """Construct VAE-marginalized semantic target F_sem strictly on the 16,540 training images."""
    out_p = PREP / "F_sem_train.pt"
    if out_p.exists():
        return out_p

    ok_idx = torch.nonzero(torch.from_numpy(np.load(PREP / "lat512_train_ok.npy"))).squeeze(1).long()
    emb_tr = {k: v.float() for k, v in torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True).items()}
    n_all = len(emb_tr[EMB_KEYS[0]])

    fe = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.to(dev).eval().requires_grad_(False)
    gen = torch.Generator().manual_seed(20261118)
    f_sem = torch.zeros((n_all, 1024), dtype=torch.float32)
    m_draws = 16
    bs = 1024

    with torch.no_grad():
        for _ in range(m_draws):
            draw_idx = ok_idx[torch.randint(0, len(ok_idx), (n_all,), generator=gen)]
            for s in range(0, n_all, bs):
                e = min(n_all, s + bs)
                out_s = fe(**{
                    EMB_KEYS[0]: emb_tr[EMB_KEYS[0]][s:e].to(dev),
                    EMB_KEYS[1]: emb_tr[EMB_KEYS[1]][s:e].to(dev),
                    EMB_KEYS[2]: emb_tr[EMB_KEYS[2]][draw_idx[s:e]].to(dev),
                }).float().cpu()
                f_sem[s:e] += out_s / float(m_draws)

    tmp = out_p.with_name(f".{out_p.name}.{os.getpid()}.tmp")
    torch.save({"F_sem": f_sem.to(torch.float16), "m_draws": m_draws}, tmp)
    tmp.replace(out_p)
    del fe, emb_tr
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[F_sem] Saved {out_p} (N={n_all}, m_draws={m_draws})", flush=True)
    return out_p


class KVSubspaceMetric(nn.Module):
    """Frozen 70-layer SDXL-Turbo UNet Key-Value pullback subspace without positive-pair loss terms."""

    def __init__(self, kv_pack_path: str, device: str = "cuda:0"):
        super().__init__()
        pack = torch.load(kv_pack_path, map_location="cpu", weights_only=True)
        self.adapter = ResNet1DLayer(hidden_size=1024, intermediate_size=4096, norm=False)
        self.adapter.load_state_dict(pack["adapter_state"])
        self.adapter.to(device).eval().requires_grad_(False)

        self.proj = IPProjectionModel(embed_dim=1024, cross_attention_dim=2048, sequence_length=4)
        self.proj.load_state_dict(pack["proj_state"])
        self.proj.to(device).eval().requires_grad_(False)

        self.register_buffer("L_kv_t", pack["L_kv"].float().T.contiguous().to(device))
        self.register_buffer("mu_kv", torch.zeros(1, 8192, device=device))
        self.register_buffer("mu_clip", torch.zeros(1, 1024, device=device))
        self.nu_bar = 32.0

    def project_kv(self, c_cone: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            return (self.proj(self.adapter(c_cone.float())).float() @ self.L_kv_t).flatten(1)

    @torch.no_grad()
    def calibrate_fold(self, F_sub: torch.Tensor):
        f = F_sub.float()
        self.nu_bar = float(f.norm(dim=1).mean().item())
        u = nz(f, dim=-1)
        self.mu_clip.copy_(u.mean(dim=0, keepdim=True))
        kv_sum = torch.zeros((1, 8192), dtype=torch.float64, device=self.mu_kv.device)
        for s in range(0, len(u), 2048):
            kv_sum += self.project_kv(u[s:s + 2048] * self.nu_bar).double().sum(dim=0, keepdim=True)
        self.mu_kv.copy_((kv_sum / float(len(u))).float())


def train_one_enc(sub: int, fold: int, seed: int, stream: str = "fsem", epochs: int = 32, lr: float = 5e-4, beta_conf: float = 0.15, dev: str = "cuda:0"):
    enc_tag = ENC_TAG_FULL if stream == "ffull" else ENC_TAG
    out = RUNS / "enc" / enc_tag / f"sub-{sub:02d}" / f"fold{fold}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if fold >= 0 and (out / "metrics.json").exists():
        return
    if fold < 0 and (out / "metrics.json").exists() and (out / "brain_model.pt").exists() and (out / "cond_test_r80.npy").exists() and (out / "cond_test_bag10.npy").exists():
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x_raw = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_train.npy")).float().to(dev)
    x_mean = x_raw.mean(1).contiguous()
    F_sem = torch.load(PREP / "F_sem_train.pt", map_location="cpu", weights_only=True)["F_sem"].float().to(dev) if stream == "fsem" else None
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float().to(dev) if (fold >= 0 or stream == "ffull") else None
    T_mat = F_tr if stream == "ffull" else F_sem
    T_u = nz(T_mat, dim=1)

    concept = torch.from_numpy(np.load(PREP / "train_concept.npy")).long().to(dev)
    img_fold = torch.from_numpy(concept_folds(1654, 5))[concept.cpu()].to(dev)

    if fold < 0:
        tr_idx = torch.arange(len(concept), device=dev)
        ho_idx = torch.empty(0, dtype=torch.long, device=dev)
    else:
        tr_idx = torch.nonzero(img_fold != fold).squeeze(1)
        ho_idx = torch.nonzero(img_fold == fold).squeeze(1)
    n_tr = len(tr_idx)

    kv_sub = KVSubspaceMetric(str(PREP / "unet_kv_metric.pt"), device=dev)
    kv_sub.calibrate_fold(T_mat[tr_idx])

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
            c_b = concept[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c_raw = model(xb, r_eff=4.0)
                u_hat = nz(c_raw.float(), dim=-1)
                with torch.no_grad():
                    u_tgt = T_u[idx]
                    y_kv_c = nz(kv_sub.project_kv(u_tgt * kv_sub.nu_bar) - kv_sub.mu_kv, dim=-1)
                    if stream == "fsem":
                        same_c = c_b[:, None] == c_b[None, :]
                        eye = torch.eye(len(idx), dtype=torch.bool, device=dev)
                        fn_mask = same_c & (~eye)
                        diff_c = ~same_c
                        s_img = 0.5 * (u_tgt @ u_tgt.T) + 0.5 * (y_kv_c @ y_kv_c.T)
                        s_bar = s_img[diff_c].mean() if diff_c.any() else 0.0

                z_kv_c = nz(kv_sub.project_kv(u_hat * kv_sub.nu_bar) - kv_sub.mu_kv, dim=-1)
                scale = logit_scale.exp().clamp(max=100.0)
                lg = scale * (0.5 * (u_hat @ u_tgt.T) + 0.5 * (z_kv_c @ y_kv_c.T))
                if stream == "fsem":
                    margin = beta_conf * scale.detach() * F.relu(s_img - s_bar) * diff_c.float()
                    lg = (lg + margin).masked_fill(fn_mask, -1e4)

                tb = torch.arange(len(idx), device=dev)
                loss = 0.5 * (F.cross_entropy(lg, tb) + F.cross_entropy(lg.T, tb))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

    train_s = time.time() - t0
    model.eval()
    rec = {
        "sub": sub,
        "arm": "O9_zero00_0head_ffull" if stream == "ffull" else "O10_kveb_fsem_fnm_zero00",
        "fold": fold,
        "seed": seed,
        "params": sum(p_.numel() for p_ in model.parameters()),
        "train_s": train_s,
        "args": {
            "arch": "KVEBPyrEncoder",
            "loss": "HybridKVInfoNCE_00" if stream == "ffull" else "ConfusionAwareKVInfoNCE",
            "target": "F_train" if stream == "ffull" else "F_sem_train",
            "w_kv": 0.0,
            "w_u": 0.0,
            "beta_conf": 0.0 if stream == "ffull" else beta_conf,
            "epochs": epochs,
            "lr": lr,
            "t0": 0,
            "t1": 250,
            "oof_protocol": "distinct_200_concepts_ho[::10][:200]" if stream == "ffull" else "ho_idx_chunks",
        },
    }

    if len(ho_idx) > 0:
        with torch.no_grad():
            outs, kvs_p, kvs_t = [], [], []
            for s in range(0, len(ho_idx), 1024):
                sl = ho_idx[s:s + 1024]
                co = model(x_mean[sl], r_eff=4.0).float()
                outs.append(co)
                kvs_p.append(kv_sub.project_kv(nz(co, dim=-1) * kv_sub.nu_bar))
                kvs_t.append(kv_sub.project_kv(T_u[sl] * kv_sub.nu_bar))
            c_oof = torch.cat(outs, dim=0)
            kv_p = torch.cat(kvs_p, dim=0)
            kv_t = torch.cat(kvs_t, dim=0)

        cu = nz(c_oof, dim=1)
        tu = T_u[ho_idx]
        idx_rel = torch.arange(len(ho_idx), device=dev)
        idx_dist = idx_rel[::10][:200] if len(idx_rel[::10]) >= 200 else idx_rel
        u_tr_axis = nz(kv_sub.mu_clip, dim=1)
        cu_perp = nz((cu - cu.mean(0, keepdim=True)) - ((cu - cu.mean(0, keepdim=True)) @ u_tr_axis.T) * u_tr_axis, dim=1)
        tu_perp = nz(tu - (tu @ u_tr_axis.T) * u_tr_axis, dim=1)
        kv_pc = nz(kv_p - kv_sub.mu_kv, dim=1)
        kv_tc = nz(kv_t - kv_sub.mu_kv, dim=1)

        rec["oof_cos"] = float((cu * tu).sum(1).mean())
        rec["kv_mse"] = float(F.mse_loss(kv_p, kv_t).item())
        ftu = nz(F_tr[ho_idx], dim=1)
        ftu_perp = nz(ftu - (ftu @ u_tr_axis.T) * u_tr_axis, dim=1)

        eval_idx = idx_dist if stream == "ffull" else idx_rel
        rec["oof_200w"] = _oof_200way(cu, tu, eval_idx)
        rec["oof_perp_200w"] = _oof_200way(cu_perp, tu_perp, eval_idx)
        rec["kv_c_200w"] = _oof_200way(kv_pc, kv_tc, eval_idx)
        rec["oof_ftrain_200w"] = _oof_200way(cu, ftu, eval_idx)
        rec["oof_ftrain_perp_200w"] = _oof_200way(cu_perp, ftu_perp, eval_idx)
        rec["oof_distinct_200w"] = _oof_200way(cu, tu, idx_dist)
        rec["oof_distinct_perp_200w"] = _oof_200way(cu_perp, tu_perp, idx_dist)
    else:
        x_te = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_test.npy")).float().to(dev)
        with torch.no_grad():
            c_ivw = model(ivw_mean(x_te), r_eff=72.0).float()
            c_bag10 = _bag10_forward(model, x_te, sub, dev)
        np.save(out / "cond_test_r80.npy", c_ivw.cpu().numpy())
        np.save(out / "cond_test_bag10.npy", c_bag10.cpu().numpy())
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "brain_model.pt")
        del x_te

    write_json(out / "metrics.json", rec)
    del model, kv_sub, x_raw, x_mean, F_sem, F_tr, T_mat, T_u
    gc.collect()
    torch.cuda.empty_cache()


def train_one_ffull_enc(sub: int, fold: int, seed: int, dev: str = "cuda:0"):
    train_one_enc(sub, fold, seed, stream="ffull", dev=dev)


def _export_stream_sub(sub: int, enc_tag: str, out_0tta: Path, out_tta: Path, raw_cache: Path, dev: str = "cuda:0"):
    """Export 0-head conditions (0-TTA and 350-step centered L_kv split-half TTA) onto the F_train[ok] cone."""
    if out_0tta.exists() and out_tta.exists():
        return

    ok_mask = torch.from_numpy(np.load(PREP / "lat512_train_ok.npy")).bool()
    F_tr_cpu = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    R_f, a_f, b_f, u_f = _cone_params(F_tr_cpu[ok_mask])

    ref_d = RUNS / "enc" / enc_tag / f"sub-{sub:02d}" / f"fold-1_seed{SEEDS_7[0]}"
    ivw0 = torch.from_numpy(np.load(ref_d / "cond_test_r80.npy")).float()
    ref_n = ivw0.norm(dim=1, keepdim=True)
    ivws = [torch.from_numpy(np.load(RUNS / "enc" / enc_tag / f"sub-{sub:02d}" / f"fold-1_seed{sd}" / "cond_test_r80.npy")).float() for sd in SEEDS_7]
    bags = [torch.from_numpy(np.load(RUNS / "enc" / enc_tag / f"sub-{sub:02d}" / f"fold-1_seed{sd}" / "cond_test_bag10.npy")).float() for sd in SEEDS_7]
    m_ivw = _norm_match(ivws, ref_n)
    m_bag = _norm_match(bags, ref_n)

    def _make_cone(m_main: torch.Tensor) -> np.ndarray:
        d_comb = nz(0.85 * _perp_dir(m_main, u_f) + 0.15 * _perp_dir(m_bag, u_f), dim=1)
        return (R_f * nz(a_f * u_f + b_f * d_comb, dim=1)).numpy().astype(np.float32)

    if not out_0tta.exists():
        out_0tta.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_0tta, _make_cone(m_ivw))

    if raw_cache.exists():
        m_tta = torch.load(raw_cache, map_location="cpu", weights_only=True)["m_tta"]
    else:
        t0 = time.time()
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
            d = RUNS / "enc" / enc_tag / f"sub-{sub:02d}" / f"fold-1_seed{sd_seed}"
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
                ka = (P(A_m(nz(ca, dim=1) * R_f)) @ L_kv_t).flatten(1)
                kb = (P(A_m(nz(cb, dim=1) * R_f)) @ L_kv_t).flatten(1)
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
        torch.save({"m_tta": m_tta, "tta_s": time.time() - t0}, raw_cache)
        del x_te, full_in, bag_ins, P, A_m, L_kv_t
        gc.collect()
        torch.cuda.empty_cache()

    out_tta.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_tta, _make_cone(m_tta))


def export_nophys_sub(sub: int, dev: str = "cuda:0"):
    """Export Stream-2 (F_sem + FN-mask) 0-head conditions before PhysReSyn."""
    _export_stream_sub(
        sub,
        ENC_TAG,
        RUNS / "cond" / "cone_o10_nophys_0tta_w1.00" / f"sub-{sub:02d}.npy",
        RUNS / "cond" / "cone_o10_nophys_kvtta_w1.00" / f"sub-{sub:02d}.npy",
        RUNS / "cond" / "raw_o10_kvtta" / f"sub-{sub:02d}.pt",
        dev=dev,
    )


def export_ffull_sub(sub: int, dev: str = "cuda:0"):
    """Export Stream-1 (F_train, (0,0), 0-head) conditions (0-TTA and 350-step centered L_kv TTA)."""
    _export_stream_sub(
        sub,
        ENC_TAG_FULL,
        RUNS / "cond" / "cone_o10_ffull_0tta_w1.00" / f"sub-{sub:02d}.npy",
        RUNS / "cond" / "cone_o10_ffull_kvtta_w1.00" / f"sub-{sub:02d}.npy",
        RUNS / "cond" / "raw_o10_ffull_kvtta" / f"sub-{sub:02d}.pt",
        dev=dev,
    )


def export_physresyn_all(subs, lam_phys: float = 0.40, k_prior: int = 24, tau_prior: float = 0.04, gamma_vae: float = 0.65, dev: str = "cuda:0"):
    """Apply Factorized Forward-Operator Re-Synthesis (PhysReSyn) using frozen FusionEncoder.vision_model."""
    t0 = time.time()
    ok_mask = torch.from_numpy(np.load(PREP / "lat512_train_ok.npy")).bool()
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    F_sem = torch.load(PREP / "F_sem_train.pt", map_location="cpu", weights_only=True)["F_sem"].float()
    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    R_f, a_f, b_f, u_f = _cone_params(F_tr[ok_mask])

    H14_tr = emb_tr[EMB_KEYS[0]][ok_mask].float().to(dev)
    B32_tr = emb_tr[EMB_KEYS[1]][ok_mask].float().to(dev)
    VAE_tr = emb_tr[EMB_KEYS[2]][ok_mask].float().reshape(int(ok_mask.sum()), -1)
    h14_norm = float(H14_tr.norm(dim=1).mean())
    b32_norm = float(B32_tr.norm(dim=1).mean())
    vae_mu = VAE_tr.mean(0, keepdim=True).to(dev)
    vae_sd = VAE_tr.std(0, keepdim=True).clamp_min(1e-6).to(dev)
    del emb_tr, VAE_tr

    Fs_p = _perp_dir(F_sem[ok_mask], u_f).to(dev)
    fe = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.to(dev).eval().requires_grad_(False)

    specs = [
        ("cone_o10_nophys_0tta_w1.00", "lat512_o10_nophys_0tta_n0.80.npy", "cone_o10_0tta_w1.00"),
        ("cone_o10_nophys_kvtta_w1.00", "lat512_o10_nophys_v2_n0.80.npy", "cone_o10_kvtta_w1.00"),
    ]
    for sub in subs:
        for src_cond, lat_name, dst_cond in specs:
            dst_p = RUNS / "cond" / dst_cond / f"sub-{sub:02d}.npy"
            if dst_p.exists():
                continue
            c_sem = torch.from_numpy(np.load(RUNS / "cond" / src_cond / f"sub-{sub:02d}.npy")).float()
            d_sem = _perp_dir(c_sem, u_f).to(dev)
            z0 = torch.from_numpy(np.load(RUNS / "low" / f"sub-{sub:02d}" / lat_name)).float().to(dev)

            sim = d_sem @ Fs_p.T
            vv, topi = sim.topk(k_prior, dim=1)
            wk = torch.softmax(vv / tau_prior, dim=1)
            h14_hat = (wk[:, :, None] * H14_tr[topi]).sum(1)
            h14_hat = h14_hat * (h14_norm / h14_hat.norm(dim=1, keepdim=True).clamp_min(1e-8))
            b32_hat = (wk[:, :, None] * B32_tr[topi]).sum(1)
            b32_hat = b32_hat * (b32_norm / b32_hat.norm(dim=1, keepdim=True).clamp_min(1e-8))

            z16 = F.avg_pool2d(z0, 4).reshape(200, -1)
            vae_hat = vae_mu + gamma_vae * (z16 - z16.mean(0, keepdim=True)) * (vae_sd / z16.std(0, keepdim=True).clamp_min(1e-6))

            with torch.no_grad():
                f_phys = fe(**{EMB_KEYS[0]: h14_hat, EMB_KEYS[1]: b32_hat, EMB_KEYS[2]: vae_hat}).float().cpu()
            d_phys = _perp_dir(f_phys, u_f)
            d_blend = nz((1.0 - lam_phys) * d_sem.cpu() + lam_phys * d_phys, dim=1)
            c_out = (R_f * nz(a_f * u_f + b_f * d_blend, dim=1)).numpy().astype(np.float32)
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst_p, c_out)

    del fe, H14_tr, B32_tr, Fs_p
    gc.collect()
    torch.cuda.empty_cache()
    dt_phys = time.time() - t0
    write_json(RUNS / "cond" / "o10_physresyn_timing.json", {"physresyn_total_s": dt_phys, "n_subs": len(subs), "per_sub_s": dt_phys / max(1, len(subs))})
    print(f"[PhysReSyn] Finished {len(subs)} subjects in {dt_phys:.1f}s", flush=True)


def export_bifurcated_all(subs, eta: float = 0.50):
    """Combine Stream-1 (F_train, (0,0)) and Stream-2 (F_sem + PhysReSyn) unit tangent directions onto F_train[ok]."""
    t0 = time.time()
    ok_mask = torch.from_numpy(np.load(PREP / "lat512_train_ok.npy")).bool()
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    R_f, a_f, b_f, u_f = _cone_params(F_tr[ok_mask])

    specs = [
        ("cone_o10_ffull_0tta_w1.00", "cone_o10_0tta_w1.00", "cone_o10_bifurc_0tta_w1.00"),
        ("cone_o10_ffull_kvtta_w1.00", "cone_o10_kvtta_w1.00", "cone_o10_bifurc_kvtta_w1.00"),
    ]
    for sub in subs:
        for src_full, src_sem_phys, dst_dir in specs:
            dst_p = RUNS / "cond" / dst_dir / f"sub-{sub:02d}.npy"
            if dst_p.exists():
                continue
            p_full = RUNS / "cond" / src_full / f"sub-{sub:02d}.npy"
            p_sem = RUNS / "cond" / src_sem_phys / f"sub-{sub:02d}.npy"
            if not p_full.exists() or not p_sem.exists():
                raise FileNotFoundError(f"Missing bifurcation inputs for sub-{sub:02d}: {p_full} / {p_sem}")
            c_full = torch.from_numpy(np.load(p_full)).float()
            c_sem = torch.from_numpy(np.load(p_sem)).float()
            d_full = _perp_dir(c_full, u_f)
            d_sem = _perp_dir(c_sem, u_f)
            d_blend = nz((1.0 - eta) * d_full + eta * d_sem, dim=1)
            c_out = (R_f * nz(a_f * u_f + b_f * d_blend, dim=1)).numpy().astype(np.float32)
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst_p, c_out)

    dt_b = time.time() - t0
    write_json(RUNS / "cond" / "o10_bifurc_timing.json", {"eta": eta, "bifurc_total_s": dt_b, "n_subs": len(subs), "per_sub_s": dt_b / max(1, len(subs))})
    print(f"[BIFURC] Exported eta={eta:.2f} conditions for {len(subs)} subjects in {dt_b:.2f}s", flush=True)


def run_gen_exp_item(task):
    if len(task) == 7:
        sub, arm_name, cond_rel, lat_rel, nsel, selvar_arg, enc_tag = task
    else:
        sub, arm_name, cond_rel, lat_rel, nsel, selvar_arg = task
        enc_tag = ENC_TAG
    selvar_spec = "0.24,0.22@4;0.24,0.22@8" if selvar_arg is True else (selvar_arg or "")

    out = RUNS / "img" / f"s{sub:02d}_{arm_name}"
    metrics_p = out / "metrics.json"
    timing_p = out / "timing.json"
    if metrics_p.exists():
        if not timing_p.exists():
            write_json(timing_p, {"nsel": nsel, "unet_nfe_per_img": 4 * nsel, "wall_clock_s": None})
        return

    cmd = [
        sys.executable,
        "-u",
        str(_find_script("gen_exp.py")),
        "--run", f"{enc_tag}/sub-{sub:02d}/fold-1_seed2025",
        "--cond_file", f"{cond_rel}/sub-{sub:02d}.npy",
        "--lat_init", f"sub-{sub:02d}/{lat_rel}",
        "--smap", "c0.625,0.5,1.0",
        "--low_keep", "8,0.22,0.04",
        "--K", "4",
        "--bs", "3",
        "--nsel", str(nsel),
        "--save", "0",
        "--name", f"s{sub:02d}_{arm_name}",
    ]
    if nsel > 1:
        cmd.extend(["--sel_vae", "0.24", "--sel_grad", "0.22"])
        if selvar_spec:
            cmd.extend(["--selvar", selvar_spec])
    t0 = time.time()
    subprocess.run(cmd, check=True)
    dt_s = time.time() - t0
    write_json(timing_p, {"nsel": nsel, "unet_nfe_per_img": 4 * nsel, "wall_clock_s": float(dt_s)})


def _run_lat512_jobs(lat_script: str, sub_csv: str, subs, specs, gpus):
    pending = [
        (cdir, tag) for (cdir, tag) in specs
        if not all((RUNS / "low" / f"sub-{s:02d}" / f"{tag}_n0.80.npy").exists() for s in subs)
    ]
    for i in range(0, len(pending), len(gpus)):
        batch = pending[i:i + len(gpus)]
        procs = []
        for gi, (cdir, tag) in enumerate(batch):
            sub_env = os.environ.copy()
            sub_env["CUDA_VISIBLE_DEVICES"] = str(gpus[gi])
            sub_env["TMPDIR"] = str(TMP_ROOT)
            cmd = [sys.executable, "-u", lat_script, sub_csv, "--mode_v", "v2", "--cond_dirs", cdir, "--tag", tag]
            procs.append((tag, subprocess.Popen(cmd, env=sub_env)))
        for tag, pr in procs:
            rc = pr.wait()
            if rc != 0:
                raise RuntimeError(f"lat512_start.py failed for tag={tag} with rc={rc}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subs", default="all", help="Comma-separated subject ints or 'all'")
    p.add_argument(
        "--mode",
        choices=[
            "all", "n1", "fsem", "enc", "cond", "eval_n1", "eval_sel16", "eval",
            "bifurc", "enc_ffull", "cond_bifurc", "eval_bifurc_n1", "eval_bifurc_sel4",
        ],
        default="all",
    )
    p.add_argument("--gpus", default="", help="Comma-separated GPU indices (default: auto-detect free GPUs)")
    p.add_argument("--worker_in", default="")
    args = p.parse_args()

    if args.worker_in:
        payload = json.loads(Path(args.worker_in).read_text())
        ph = payload["phase"]
        for t in payload["tasks"]:
            if ph == "fsem":
                build_fsem_train(dev="cuda:0")
            elif ph == "enc":
                train_one_enc(t[0], t[1], t[2], stream="fsem", dev="cuda:0")
            elif ph == "enc_ffull":
                train_one_ffull_enc(t[0], t[1], t[2], dev="cuda:0")
            elif ph == "nophys":
                export_nophys_sub(t, dev="cuda:0")
            elif ph == "ffull_cond":
                export_ffull_sub(t, dev="cuda:0")
            elif ph == "physresyn":
                export_physresyn_all(t, dev="cuda:0")
            elif ph == "gen":
                run_gen_exp_item(t)
        return

    subs = list(ALL_SUBS) if args.subs == "all" else [int(x) for x in args.subs.split(",")]
    sub_csv = "all" if len(subs) == 10 else ",".join(map(str, subs))
    gpus = detect_gpus(args.gpus)
    lat_script = str(_find_script("lat512_start.py"))

    if args.mode in ("all", "n1", "fsem") and not (PREP / "F_sem_train.pt").exists():
        run_buckets_subproc(["build_fsem"], [gpus[0]], "fsem")

    if args.mode in ("all", "n1", "enc"):
        # 5-fold OOF diagnostics on seed 2025 (5 folds) + 7-seed full training (-1 fold)
        enc_tasks = [(s, f, 2025) for s in subs for f in (0, 1, 2, 3, 4)] + [(s, -1, sd) for s in subs for sd in SEEDS_7]
        run_buckets_subproc(enc_tasks, gpus, "enc")

    if args.mode in ("all", "n1", "cond"):
        run_buckets_subproc(subs, gpus, "nophys")
        _run_lat512_jobs(
            lat_script, sub_csv, subs,
            [("cone_o10_nophys_0tta_w1.00", "lat512_o10_nophys_0tta"), ("cone_o10_nophys_kvtta_w1.00", "lat512_o10_nophys_v2")],
            gpus,
        )
        run_buckets_subproc([subs], [gpus[0]], "physresyn")
        _run_lat512_jobs(
            lat_script, sub_csv, subs,
            [("cone_o10_0tta_w1.00", "lat512_o10_0tta"), ("cone_o10_kvtta_w1.00", "lat512_o10_v2")],
            gpus,
        )

    if args.mode in ("all", "n1", "eval_n1", "eval"):
        arms_n1 = [
            ("o10_0tta_w1_notmf", "cone_o10_0tta_w1.00", "lat512_o10_0tta_n0.80.npy", 1, "", ENC_TAG),
            ("o10_nophys_w1_notmf", "cone_o10_nophys_kvtta_w1.00", "lat512_o10_nophys_v2_n0.80.npy", 1, "", ENC_TAG),
            ("o10_w1_notmf", "cone_o10_kvtta_w1.00", "lat512_o10_v2_n0.80.npy", 1, "", ENC_TAG),
        ]
        gen_tasks_n1 = [(s, a, c_r, l_r, nsel, wsv, et) for (a, c_r, l_r, nsel, wsv, et) in arms_n1 for s in subs]
        run_buckets_subproc(gen_tasks_n1, gpus, "gen")

    if args.mode in ("all", "eval_sel16", "eval"):
        gen_tasks_s16 = [(s, "o10_sel16_notmf", "cone_o10_kvtta_w1.00", "lat512_o10_v2_n0.80.npy", 16, True, ENC_TAG) for s in subs]
        run_buckets_subproc(gen_tasks_s16, gpus, "gen")

    if args.mode in ("all", "bifurc", "enc_ffull"):
        enc_tasks_full = [(s, f, 2025) for s in subs for f in (0, 1, 2, 3, 4)] + [(s, -1, sd) for s in subs for sd in SEEDS_7]
        run_buckets_subproc(enc_tasks_full, gpus, "enc_ffull")

    if args.mode in ("all", "bifurc", "cond_bifurc"):
        run_buckets_subproc(subs, gpus, "ffull_cond")
        export_bifurcated_all(subs, eta=0.50)
        _run_lat512_jobs(
            lat_script, sub_csv, subs,
            [
                ("cone_o10_ffull_kvtta_w1.00", "lat512_o10_ffull_v2"),
                ("cone_o10_bifurc_0tta_w1.00", "lat512_o10_bifurc_0tta"),
                ("cone_o10_bifurc_kvtta_w1.00", "lat512_o10_bifurc_v2"),
            ],
            gpus,
        )

    if args.mode in ("all", "bifurc", "eval_bifurc_n1"):
        arms_bifurc_n1 = [
            ("o10_ffull00_w1_notmf", "cone_o10_ffull_kvtta_w1.00", "lat512_o10_ffull_v2_n0.80.npy", 1, "", ENC_TAG_FULL),
            ("o10_bifurc_0tta_w1_notmf", "cone_o10_bifurc_0tta_w1.00", "lat512_o10_bifurc_0tta_n0.80.npy", 1, "", ENC_TAG),
            ("o10_bifurc_w1_notmf", "cone_o10_bifurc_kvtta_w1.00", "lat512_o10_bifurc_v2_n0.80.npy", 1, "", ENC_TAG),
        ]
        gen_tasks_bn1 = [(s, a, c_r, l_r, nsel, wsv, et) for (a, c_r, l_r, nsel, wsv, et) in arms_bifurc_n1 for s in subs]
        run_buckets_subproc(gen_tasks_bn1, gpus, "gen")

    if args.mode in ("all", "bifurc", "eval_bifurc_sel4"):
        gen_tasks_bs4 = [(s, "o10_bifurc_sel4_notmf", "cone_o10_bifurc_kvtta_w1.00", "lat512_o10_bifurc_v2_n0.80.npy", 4, "0.24,0.22@2", ENC_TAG) for s in subs]
        run_buckets_subproc(gen_tasks_bs4, gpus, "gen")


if __name__ == "__main__":
    main()
