"""Faithful Top-Conference Baselines on Official THINGS-EEG2 (10 subs x 200 unseen concepts, R=80, 4-NFE SDXL-Turbo)."""
import os, sys, gc, math, time, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RAD = Path(os.environ.get("RAD_ROOT", "/path/to/rad"))
for sub in ("code/exp_obj", "code/exp_w7", "code/exp_w6", "code/rad", "code/exp", "code/Brain-HIVE"):
    p = str(RAD / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from main.models_brain import BrainEncoder, BrainMLP, NICE, ATM
from main.models_adapter import IPProjectionModel, IPAttnAdapterModel, FusionEncoderModel
from eval_obj_suite import ObjReconEvaluator

PREP = RAD / "data/prep"
TURBO = RAD / "models/sdxl-turbo"
PRIOR = RAD / "models/Fusion-Prior-H14_B32_VAE"
OUT_ROOT = RAD / "runs/obj_bench/baselines"
ALL_SUBS = list(range(1, 11))

def nz(x, dim=-1):
    return F.normalize(x.float(), dim=dim)

def clip_infonce(pred, tgt, logit_scale):
    p = nz(pred, dim=-1)
    t = nz(tgt, dim=-1)
    s = logit_scale.float().exp().clamp(max=100.0)
    lg = (p @ t.T) * s
    y = torch.arange(len(p), device=p.device)
    return 0.5 * (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y))

class CogCapMM(nn.Module):
    """CognitionCapturer-MM (AAAI 2025 adapted): dual-modality EEG -> Image(F) + Text(bigG) alignment."""
    def __init__(self, c_num=17, seq_len=250, hidden=1024):
        super().__init__()
        self.enc_vis = BrainMLP(c_num * seq_len, hidden, num_layers=2, dropout=0.25)
        self.enc_txt = BrainMLP(c_num * seq_len, hidden, num_layers=2, dropout=0.25)
        self.head_vis = nn.Linear(hidden, 1024)
        self.head_txt = nn.Linear(hidden, 1280)

    def forward(self, x):
        return self.head_vis(self.enc_vis(x)), self.head_txt(self.enc_txt(x))

def train_baseline_sub(method: str, sub: int, F_tr: torch.Tensor, F_te: torch.Tensor,
                       T_bg_tr: torch.Tensor, mu_bg: torch.Tensor, dev: str = "cuda:0"):
    torch.manual_seed(2025 + sub); torch.cuda.manual_seed_all(2025 + sub)
    x_tr = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_train.npy")).float()  # [16540, 4, 17, 250]
    x_te = torch.from_numpy(np.load(PREP / f"sub-{sub:02d}_test.npy")).float()   # [200, 80, 17, 250]
    x_te_mean = x_te.mean(dim=1).to(dev)  # Standard R=80 trial average
    n_tr = len(x_tr)

    if method == "nice":
        # Official NICE (Song et al., ICLR 2024)
        model = BrainEncoder("nice", brain_channels=17, brain_sequence_length=250, hidden_size=1024, embedding_dim=1440).to(dev)
        epochs, lr, bs = 45, 5e-4, 512
    elif method == "atm_s":
        # Official ATM-S (Li et al., NeurIPS 2024: iTransformer + TSConv)
        model = BrainEncoder("atm", brain_channels=17, brain_sequence_length=250, hidden_size=1024, dropout=0.25).to(dev)
        epochs, lr, bs = 45, 5e-4, 512
    elif method == "brain_hive":
        # Official Brain-HIVE (BrainMLP + Fusion-Prior)
        model = BrainEncoder("brain_mlp", brain_channels=17, brain_sequence_length=250, hidden_size=1024, dropout=0.1).to(dev)
        epochs, lr, bs = 35, 5e-4, 1024
    elif method == "cogcap_mm":
        # CognitionCapturer-MM (AAAI 2025 dual Image+Text modality alignment adapted)
        model = CogCapMM(c_num=17, seq_len=250, hidden=1024).to(dev)
        epochs, lr, bs = 45, 5e-4, 1024
    else:
        raise ValueError(method)

    ls = nn.Parameter(torch.tensor(2.6592, device=dev))
    ls_t = nn.Parameter(torch.tensor(2.6592, device=dev))
    opt = torch.optim.AdamW(list(model.parameters()) + [ls, ls_t], lr=lr, weight_decay=0.05)
    spr = math.ceil(n_tr / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=spr * epochs)
    g = torch.Generator().manual_seed(2025 + sub)
    sub_id_t = torch.full((bs,), sub - 1, dtype=torch.long, device=dev)

    # Target norm calibration (Fusion-Prior expects norm ~32.0, matching F_train norm!)
    f_norm_ref = F_tr.norm(dim=-1).mean().item()
    F_tr_dev = F_tr.to(dev)
    T_bg_dev = T_bg_tr.to(dev) if method == "cogcap_mm" else None

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n_tr, generator=g)
        for b in range(spr):
            idx = perm[b * bs:(b + 1) * bs]
            # Standard random-repeat averaging (1..4 repeats) matching Brain-HIVE/ATM-S training
            xb_raw = x_tr[idx]  # [B, 4, 17, 250]
            k_rep = int(torch.randint(1, 5, (1,), generator=g).item())
            if k_rep == 4:
                xb = xb_raw.mean(dim=1).to(dev)
            else:
                ridx = torch.randperm(4, generator=g)[:k_rep]
                xb = xb_raw[:, ridx].mean(dim=1).to(dev)
            yb = F_tr_dev[idx]
            sid = sub_id_t[:len(idx)]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                if method == "cogcap_mm":
                    pv, pt = model(xb)
                    l_v = clip_infonce(pv, yb, ls) + 0.5 * F.mse_loss(nz(pv), nz(yb))
                    tb = T_bg_dev[idx]
                    pt_c = pt.float() - pt.float().mean(0, keepdim=True)
                    l_t = clip_infonce(pt_c, tb, ls_t) + 0.5 * (1.0 - (nz(pt_c) * tb).sum(-1)).mean()
                    loss = 0.55 * l_v + 0.45 * l_t
                else:
                    pv = model(xb, sid)
                    # Standard Brain-HIVE / ATM-S / NICE hybrid InfoNCE + MSE in F-space
                    loss = clip_infonce(pv, yb, ls) + 0.5 * F.mse_loss(nz(pv), nz(yb))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

    model.eval()
    with torch.no_grad():
        sid_te = torch.full((200,), sub - 1, dtype=torch.long, device=dev)
        if method == "cogcap_mm":
            c_raw, ppe_raw = model(x_te_mean)
            c_pred = nz(c_raw) * f_norm_ref
            ppe_pred = nz(ppe_raw.float() - ppe_raw.float().mean(0, keepdim=True))
        else:
            c_raw = model(x_te_mean, sid_te)
            c_pred = nz(c_raw) * f_norm_ref
            ppe_pred = None

        # Retrieval diagnostic on F_test
        sim = nz(c_pred).cpu() @ nz(F_te).T
        diag = torch.arange(200)
        f_t1 = float((sim.argmax(-1) == diag).float().mean().item() * 100.0)
        f_t5 = float((sim.topk(5, -1).indices == diag[:, None]).any(-1).float().mean().item() * 100.0)

    del model, x_tr, x_te, x_te_mean, F_tr_dev, T_bg_dev
    torch.cuda.empty_cache()
    return c_pred.cpu(), (ppe_pred.cpu() if ppe_pred is not None else None), f_t1, f_t5

class TurboGen:
    def __init__(self, dev="cuda:0"):
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline
        self.dev = dev
        self.dt = torch.float16
        self.pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=self.dt, variant="fp16")
        IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(self.pipe.unet)
        self.pipe.to(dev, self.dt)
        with torch.inference_mode():
            self.pe_empty, _, self.ppe_empty, _ = self.pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False)
        self.proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, self.dt).eval().requires_grad_(False)
        self.adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, self.dt).eval().requires_grad_(False)
        self.pipe.vae.enable_slicing()
        self.pipe.set_progress_bar_config(disable=True)
        self.vae32 = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval().requires_grad_(False)
        self.vae32.enable_slicing()

    @torch.inference_mode()
    def generate(self, c_all: torch.Tensor, ppe_delta: torch.Tensor = None, alpha_ppe: float = 0.0, bs: int = 20):
        out_imgs = torch.empty(200, 3, 512, 512, dtype=torch.uint8)
        ppe_norm = float(self.ppe_empty.float().norm().item())
        for s in range(0, 200, bs):
            q = list(range(s, min(200, s + bs)))
            Q = len(q)
            cc = c_all[q].to(self.dev, self.dt)
            ips = self.proj(self.adapter(cc))
            pe_b = self.pe_empty.expand(Q, -1, -1)
            if ppe_delta is not None and alpha_ppe > 0:
                pd = ppe_delta[q].to(self.dev, torch.float32)
                ppe_b = (self.ppe_empty.float().expand(Q, -1) + alpha_ppe * ppe_norm * pd).to(self.dt)
            else:
                ppe_b = self.ppe_empty.expand(Q, -1)
            gens = [torch.Generator(self.dev).manual_seed(1_000_003 * i + 12345) for i in q]
            lat = self.pipe(
                prompt_embeds=pe_b,
                pooled_prompt_embeds=ppe_b,
                num_inference_steps=4,
                guidance_scale=0.0,
                output_type="latent",
                generator=gens,
                cross_attention_kwargs={"ip_hidden_states": ips},
            ).images
            rgb = (self.vae32.decode(lat.float() / self.vae32.config.scaling_factor).sample / 2.0 + 0.5).clamp(0, 1)
            out_imgs[q] = (rgb * 255.0).round().to(torch.uint8).cpu()
        return out_imgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", type=str, required=True)
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    F_te = torch.from_numpy(np.load(PREP / "F_test.npy")).float()
    tb = torch.load(RAD / "runs/obj_bench/text_banks/text_banks_laion.pt", map_location="cpu", weights_only=False)
    tr_cid = tb["tr_img2cid"].long()
    bg_raw = tb["bigg_train"][tr_cid].float()
    mu_bg = bg_raw.mean(0, keepdim=True)
    T_bg_tr = nz(bg_raw - mu_bg, dim=-1)

    gen = TurboGen("cuda:0")
    ev = ObjReconEvaluator("cuda:0")

    for method in methods:
        t0 = time.time()
        mdir = OUT_ROOT / method
        mdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for s in ALL_SUBS:
            cache_img = mdir / f"sub-{s:02d}_imgs.pt"
            cache_m = mdir / f"sub-{s:02d}_metrics.json"
            if cache_img.exists() and cache_m.exists():
                m = json.load(open(cache_m))
                rows.append(m)
                print(f"  [CACHED] {method} sub-{s:02d} ZS200={m['zs_obj200_t1']:5.2f} ZS1823={m['zs_obj1823_t1']:4.2f} Img200={m['clip_img200_t1']:5.2f}", flush=True)
                continue
            c_pred, ppe_pred, f_t1, f_t5 = train_baseline_sub(method, s, F_tr, F_te, T_bg_tr, mu_bg, dev="cuda:0")
            imgs = gen.generate(c_pred, ppe_delta=ppe_pred, alpha_ppe=(0.35 if method == "cogcap_mm" else 0.0))
            m = ev.evaluate(imgs, compute_lowlevel=True)
            m["eeg_f_t1"] = f_t1; m["eeg_f_t5"] = f_t5
            torch.save({"k0_rgb_u8": imgs, "c_pred": c_pred, "ppe_pred": ppe_pred}, cache_img)
            json.dump(m, open(cache_m, "w"), indent=2)
            rows.append(m)
            print(f"  [{method:10s}] sub-{s:02d} | EEG_F={f_t1:5.1f}% | ZS200={m['zs_obj200_t1']:5.2f}/{m['zs_obj200_t5']:5.2f} | ZS1823={m['zs_obj1823_t1']:4.2f}/{m['zs_obj1823_t5']:5.2f} | Cat27={m['coarse27_t1']:4.1f}% | IN1K={m['in1k_inc_t1']:4.1f}/{m['in1k_inc_t5']:4.1f} | Img200={m['clip_img200_t1']:5.2f}% | 2wC={m['clip_2way']:5.2f}%", flush=True)
        mean_m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        json.dump({"per_sub": rows, "mean": mean_m}, open(RAD / f"runs/obj_bench/evals/{method}_all10.json", "w"), indent=2)
        print(f"\n=== [{method}] 10-SUB MEAN ({time.time()-t0:.1f}s) ===", flush=True)
        for k in ["eeg_f_t1", "zs_obj200_t1", "zs_obj200_t5", "zs_obj1823_t1", "zs_obj1823_t5", "coarse27_t1", "in1k_inc_t1", "in1k_inc_t5", "in1k_eff_t1", "obj_crisp", "clip_img200_t1", "clip_2way", "incep_2way", "alex2", "alex5", "eff_dist", "pixcorr", "ssim"]:
            print(f"  {k:16s}: {mean_m[k]:.4f}", flush=True)

if __name__ == "__main__":
    main()
