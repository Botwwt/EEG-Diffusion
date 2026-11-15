"""Generator extending gen.py for O7 and B4 (gen_exp.py):
  1. --lat_init <file>: direct [200, 4, 64, 64] native 512px VAE latent start
  2. --tmf <spec> & --tmf_cr 1: step-1/2 Tweedie z0_hat coupling across the K=4 parallel trajectories of each query
     (unchanged 4-trajectory NFE per query, but the K=4 output samples of a query are coupled rather than independent draws;
     omit --tmf for strictly independent K=4 draws) with per-channel 64x64 spatial contrast restoration
  3. --low_keep <f,w2[,w3]>: gentle low-band tether to the noised native start latent at step i=2 (and optionally i=3)
  4. --nsel <N> with --sel_cond, --spat_cond, --sel_spat, --sel_vae, --sel_grad (plus --selvar):
     scores candidates using Brain-HIVE's own FEmbedder (F + 4x16x16 VAE latent), the 4-orientation spatial-contour
     projection S_cand (from spat1024.pt, no evaluation-network weights), and 32x32 finite-difference RGB spatial gradients.
Random seeds and metrics.py evaluation match gen.py identically.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from common import EMB_KEYS, HIVE, IMG, PREP, PRIOR, RUNS, TURBO, write_json
from metrics import score

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--name", required=True)
p.add_argument("--use_A", type=int, default=1)
p.add_argument("--init_file", default="", help="RGB start [200,3,h,h] relative to runs/low")
p.add_argument("--lat_init", default="", help="Native latent start [200,4,64,64] relative to runs/low")
p.add_argument("--cond_file", required=True, help="Condition [200,1024] relative to runs/cond")
p.add_argument("--sel_cond", default="", help="Semantic condition [200,1024] for --nsel scoring (defaults to --cond_file)")
p.add_argument("--spat_cond", default="", help="Optional steerable-contour query [200,1024] relative to runs/cond")
p.add_argument("--smap", default="c0.611,0.5,1.0")
p.add_argument("--smap_lp", type=int, default=0)
p.add_argument("--low_keep", default="", help="'f,w2[,w3]': gentle low-band anchor at step i=2 (and optionally i=3)")
p.add_argument("--mf", default="")
p.add_argument("--tmf", default="", help="'M,g1,g2,g3[,lp3,mode]': Tweedie-Rao-Blackwellized coupling on z0_hat")
p.add_argument("--tmf_cr", type=int, default=0, help="1: restore per-channel 64x64 spatial contrast std after Tweedie coupling")
p.add_argument("--nsel", type=int, default=1)
p.add_argument("--sel_spat", type=float, default=0.0, help="weight of steerable-contour similarity to --spat_cond")
p.add_argument("--sel_vae", type=float, default=0.0, help="weight of 4x16x16 VAE latent correlation with start latent")
p.add_argument("--sel_grad", type=float, default=0.0, help="weight of 32x32 RGB + finite-diff gradient correlation with start image")
p.add_argument("--selvar", default="", help="optional ';'-separated '[wspat,]wvae,wgrad@n' extra variants scored in same nsel pass")
p.add_argument("--post", default="")
p.add_argument("--amp", type=float, default=1.0)
p.add_argument("--steps", type=int, default=4)
p.add_argument("--K", type=int, default=4)
p.add_argument("--bs", type=int, default=3)
p.add_argument("--save", type=int, default=0)
p.add_argument("--dump", type=int, default=0)
args = p.parse_args()

dev, dt = "cuda", torch.float16
run = RUNS / "enc" / args.run
out = RUNS / "img" / args.name
out.mkdir(parents=True, exist_ok=True)
if (out / "metrics.json").exists():
    print("already done:", out)
    sys.exit(0)
ea = json.loads((run / "metrics.json").read_text())["args"]

sys.path.insert(0, str(HIVE))
from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel  # noqa: E402

nz = F.normalize
f_cond = Path(args.cond_file)
c = torch.from_numpy(np.load(f_cond if f_cond.is_absolute() else RUNS / "cond" / f_cond)).float()
assert c.shape == (200, 1024)
if args.amp != 1.0:
    cn = nz(c, dim=1)
    m = cn.mean(0, keepdim=True)
    c = c.norm(dim=1, keepdim=True) * nz(m + args.amp * (cn - m), dim=1)

f_sc = Path(args.sel_cond) if args.sel_cond else f_cond
c_sel = torch.from_numpy(np.load(f_sc if f_sc.is_absolute() else RUNS / "cond" / f_sc)).float()
qv = nz(nz(c_sel, dim=1) - nz(c_sel, dim=1).mean(0, keepdim=True), dim=1)

q_spat = None
sp_pack = None
mu_sp = None
if args.spat_cond:
    f_sp = Path(args.spat_cond)
    sp_arr = torch.from_numpy(np.load(f_sp if f_sp.is_absolute() else RUNS / "cond" / f_sp)).float()
    q_spat = nz(sp_arr - sp_arr.mean(0, keepdim=True), dim=1)
    sp_pack = torch.load(PREP / "spat1024.pt", map_location="cpu", weights_only=True)
    mu_sp = nz(sp_pack["S_train"].float(), dim=1).mean(0, keepdim=True)


def cand_spat_emb(lat16_cpu):
    v16 = lat16_cpu.view(-1, 4, 16, 16)
    blks = [
        v16.flatten(1),
        (v16[:, :, :, 1:] - v16[:, :, :, :-1]).flatten(1),
        (v16[:, :, 1:, :] - v16[:, :, :-1, :]).flatten(1),
        (v16[:, :, 1:, 1:] - v16[:, :, :-1, :-1]).flatten(1),
        (v16[:, :, 1:, :-1] - v16[:, :, :-1, 1:]).flatten(1),
    ]
    parts = [w * (b - m) / r for b, m, r, w in zip(blks, sp_pack["mu"], sp_pack["rms"], sp_pack["weights"])]
    s_raw = torch.cat(parts, dim=1) @ sp_pack["P_spat"]
    return nz(nz(s_raw, dim=1) - mu_sp, dim=1)


info = dict(args=vars(args), encoder_args=ea, cond_norm=float(c.norm(dim=1).mean()))

from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline  # noqa: E402

pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=dt, variant="fp16")
attn = IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter")
attn.bind_unet(pipe.unet)
proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, dt).eval()
A = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, dt).eval()
pipe.to(dev, dt)
pipe.vae.enable_slicing()
pipe.set_progress_bar_config(disable=True)
vae32 = copy.deepcopy(pipe.vae).float()

with torch.inference_mode():
    pe, _, ppe, _ = pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)

ids = json.loads((PREP / "test_ids.json").read_text())
gt_tf = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(512), T.ToTensor()])
files = {f.stem: f for f in (IMG / "test_images").glob("*/*.jpg")}
gt = torch.stack([gt_tf(Image.open(files[i]).convert("RGB")) for i in ids["ids"]])

K, N, NS = args.K, len(c), args.nsel
assert bool(args.init_file) ^ bool(args.lat_init), "Specify exactly one of --init_file or --lat_init"
p2 = StableDiffusionXLImg2ImgPipeline(**pipe.components)
p2.set_progress_bar_config(disable=True)

if args.init_file:
    f_in = Path(args.init_file)
    init = torch.from_numpy(np.load(f_in if f_in.is_absolute() else RUNS / "low" / f_in)).float()
    init = F.interpolate(init, size=512, mode="bicubic").clamp(0, 1)
    with torch.no_grad():
        z0_all = torch.cat([vae32.encode(init[j:j + 20].to(dev) * 2 - 1).latent_dist.mean.cpu() * vae32.config.scaling_factor
                            for j in range(0, 200, 20)])
else:
    f_lat = Path(args.lat_init)
    z0_all = torch.from_numpy(np.load(f_lat if f_lat.is_absolute() else RUNS / "low" / f_lat)).float()
    assert z0_all.shape == (200, 4, 64, 64)
    with torch.no_grad():
        init = torch.cat([(vae32.decode(z0_all[j:j + 20].to(dev) / vae32.config.scaling_factor).sample / 2 + 0.5).clamp(0, 1).cpu()
                          for j in range(0, 200, 20)])

z16_ref = F.avg_pool2d(z0_all, 4).flatten(1)


def rgb_grad32(im_cpu):
    s32 = F.interpolate(im_cpu.float(), size=32, mode="area")
    dx = s32[:, :, :, 1:] - s32[:, :, :, :-1]
    dy = s32[:, :, 1:, :] - s32[:, :, :-1, :]
    return torch.cat([
        (s32 - s32.mean(dim=(2, 3), keepdim=True)).flatten(1) * 0.5,
        dx.flatten(1),
        dy.flatten(1),
    ], dim=1)


g32_ref = rgb_grad32(init)


def row_corr(a, b):
    ac = a - a.mean(1, keepdim=True)
    bc = b - b.mean(1, keepdim=True)
    return (ac * bc).sum(1) / (ac.norm(dim=1) * bc.norm(dim=1) + 1e-8)


kind, s_rel, s_unrel = args.smap.split(",")
s_rel, s_unrel = float(s_rel), float(s_unrel)
assert kind.startswith("c") and kind != "const"
rho = torch.full((200, 1, 64, 64), float(kind[1:]))
smap = s_unrel - (s_unrel - s_rel) * rho
strength = s_unrel
n_run = min(int(args.steps * s_unrel), args.steps)
hold = (n_run - args.steps * smap).clamp(min=0)
info.update(smap_mean_strength=float(smap.mean()), smap_mean_hold=float(hold.mean()))


def lp_op(z, f):
    if f <= 1:
        return z
    return F.interpolate(F.avg_pool2d(z, f), size=z.shape[-1], mode="bilinear", align_corners=False)


mf_cfg = None
tmf_cfg = None
if args.mf:
    parts = args.mf.split(",")
    M_part = int(parts[0])
    g_steps = {1: float(parts[1]), 2: float(parts[2]), 3: float(parts[3])}
    mf_lp = int(parts[4]) if len(parts) > 4 else 0
    mf_cfg = (M_part, g_steps, mf_lp)
elif args.tmf:
    parts = args.tmf.split(",")
    M_part = int(parts[0])
    g_steps = {1: float(parts[1]), 2: float(parts[2]), 3: float(parts[3])}
    lp3 = int(parts[4]) if len(parts) > 4 else 0
    tmf_all = (parts[5] == "all") if len(parts) > 5 else False
    tmf_cfg = (M_part, g_steps, lp3, tmf_all)
else:
    M_part = 1

last_z0 = {"val": None}
hook_handle = None
if tmf_cfg is not None:
    def _unet_hook(mod, inp, out_u):
        t_scalar = float(inp[1].flatten()[0].item())
        si = int((p2.scheduler.timesteps.float().to(inp[0].device) - t_scalar).abs().argmin().item())
        sig_cur = float(p2.scheduler.sigmas[si].item())
        z_t = inp[0].float() * ((sig_cur ** 2 + 1.0) ** 0.5)
        eps_p = (out_u[0] if isinstance(out_u, tuple) else out_u.sample).float()
        last_z0["val"] = z_t - sig_cur * eps_p
    hook_handle = p2.unet.register_forward_hook(_unet_hook)

lk_cfg = None
if args.low_keep:
    parts = [float(x) for x in args.low_keep.split(",")]
    lk_cfg = (int(parts[0]), parts[1], parts[2] if len(parts) > 2 else 0.0)

femb = None
mu_fn = None
selvars = []
if NS > 1:
    from fenc import FEmbedder  # noqa: E402
    femb = FEmbedder(dev)
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy")).float()
    mu_fn = nz(F_tr, dim=1).mean(0, keepdim=True)
    if args.selvar:
        for spec in args.selvar.split(";"):
            ws_str, n_s = spec.split("@")
            ws_parts = [float(x) for x in ws_str.split(",")]
            if len(ws_parts) == 2:
                wsp_, wv_, wg_ = 0.0, ws_parts[0], ws_parts[1]
            else:
                wsp_, wv_, wg_ = ws_parts[0], ws_parts[1], ws_parts[2]
            assert int(n_s) <= NS
            selvars.append((spec, wsp_, wv_, wg_, int(n_s)))


def gen_candidate(q, n_cand):
    Q = len(q)
    KM = K * M_part
    with torch.inference_mode():
        cc = c[q].to(dev, dt)
        ips = proj(A(cc) if args.use_A else cc).repeat_interleave(KM, dim=0)
        pe_q = pe.expand(Q * KM, -1, -1)
        ppe_q = ppe.expand(Q * KM, -1)
        kw = dict(prompt_embeds=pe_q, pooled_prompt_embeds=ppe_q,
                  num_inference_steps=args.steps, guidance_scale=0.0, output_type="pt",
                  cross_attention_kwargs={"ip_hidden_states": ips})

        gens = [torch.Generator(dev).manual_seed(1_000_003 * i + k + 12345 + 100_003 * (n_cand * M_part + m))
                for i in q for k in range(K) for m in range(M_part)]
        i0 = init[q].repeat_interleave(KM, dim=0).to(dev, dt)
        z0 = z0_all[q].to(dev).float().repeat_interleave(KM, dim=0)
        eps = torch.stack([torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * i + k + 100_003 * (n_cand * M_part + m) + 99))
                           for i in q for k in range(K) for m in range(M_part)]).to(dev)
        h_ = hold[q].repeat_interleave(KM, dim=0).to(dev)

        def cb(pp, i, t, ck):
            lt = ck["latents"].float()
            if tmf_cfg is not None and i in tmf_cfg[1] and tmf_cfg[1][i] > 0 and last_z0["val"] is not None:
                M_p, g_map, lp3, tmf_all = tmf_cfg
                g_val = g_map[i]
                z0_h = last_z0["val"]
                zv = z0_h.view(Q, K, 4, 64, 64) if M_p == 1 else (z0_h.view(Q, K * M_p, 4, 64, 64) if tmf_all else z0_h.view(Q * K, M_p, 4, 64, 64))
                zmean = zv.mean(dim=1, keepdim=True).expand_as(zv).reshape_as(z0_h)
                delta = zmean - z0_h
                if i == 3 and lp3 > 1:
                    delta = lp_op(delta, lp3)
                z0_new = z0_h + g_val * delta
                if args.tmf_cr:
                    mu_h = z0_h.mean(dim=(2, 3), keepdim=True)
                    sd_h = z0_h.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                    mu_n = z0_new.mean(dim=(2, 3), keepdim=True)
                    sd_n = z0_new.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                    z0_new = mu_n + (z0_new - mu_n) * (sd_h / sd_n)
                lt = lt + (z0_new - z0_h)
            m_ = (h_ - i).clamp(0, 1)
            if float(m_.max()) > 0:
                sig = pp.scheduler.sigmas[pp.scheduler.step_index].to(dev)
                tgt = z0 + sig * eps
                if args.smap_lp > 0:
                    full_mask = (m_ >= 0.999).float()
                    part_mask = (m_ < 0.999).float() * m_
                    lt = full_mask * tgt + (1.0 - full_mask) * (lt + part_mask * (lp_op(tgt, args.smap_lp) - lp_op(lt, args.smap_lp)))
                else:
                    lt = m_ * tgt + (1.0 - m_) * lt
            if lk_cfg is not None and i in (2, 3):
                f_lk, w2, w3 = lk_cfg
                w_lk = w2 if i == 2 else w3
                if w_lk > 0:
                    sig = pp.scheduler.sigmas[pp.scheduler.step_index].to(dev)
                    tgt = z0 + sig * eps
                    lt = lt + w_lk * (lp_op(tgt, f_lk) - lp_op(lt, f_lk))
            if mf_cfg is not None and i in mf_cfg[1] and mf_cfg[1][i] > 0:
                M_p, g_map, mf_lp_f = mf_cfg
                g_val = g_map[i]
                lv = lt.view(Q, K, 4, 64, 64) if M_p == 1 else lt.view(Q * K, M_p, 4, 64, 64)
                lmean = lv.mean(dim=1, keepdim=True).expand_as(lv).reshape_as(lt)
                if mf_lp_f > 0:
                    lt = lt + g_val * (lp_op(lmean, mf_lp_f) - lp_op(lt, mf_lp_f))
                else:
                    lt = (1.0 - g_val) * lt + g_val * lmean
            ck["latents"] = lt.to(ck["latents"].dtype)
            return ck

        kw.update(callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
        img = p2(image=i0, strength=strength, generator=gens, **kw).images
        if M_part > 1:
            img = img.view(Q, K, M_part, 3, 512, 512)[:, :, 0].reshape(Q * K, 3, 512, 512)
        img = (img.float().clamp(0, 1) * 255).round().div(255)
        return img


rec = torch.empty(K, N, 3, 512, 512)
sv_rec = {spec: torch.empty(K, N, 3, 512, 512, dtype=torch.uint8) for spec, _, _, _, _ in selvars}
eff_bs = max(1, args.bs // max(1, M_part // 2))

for s in range(0, N, eff_bs):
    q = list(range(s, min(N, s + eff_bs)))
    Q = len(q)
    if NS == 1:
        img = gen_candidate(q, 0)
        rec[:, q] = img.cpu().view(Q, K, 3, 512, 512).transpose(0, 1)
    else:
        best_sc = torch.full((Q * K,), -1e9)
        best_img = torch.empty(Q * K, 3, 512, 512)
        sv_best = {spec: torch.full((Q * K,), -1e9) for spec, _, _, _, _ in selvars}
        sv_cur = {spec: torch.empty(Q * K, 3, 512, 512, dtype=torch.uint8) for spec, _, _, _, _ in selvars}

        qv_qk = qv[q].repeat_interleave(K, dim=0)
        z16_qk = z16_ref[q].repeat_interleave(K, dim=0)
        g32_qk = g32_ref[q].repeat_interleave(K, dim=0)
        qsp_qk = q_spat[q].repeat_interleave(K, dim=0) if q_spat is not None else None

        for n_cand in range(NS):
            img = gen_candidate(q, n_cand)
            img_cpu = img.cpu()
            with torch.no_grad():
                pts = femb.parts(img)
                f_emb = femb.fuse(**pts).float().cpu()
                lat16 = pts[EMB_KEYS[2]].float().cpu()
            sc_sem = (nz(nz(f_emb, dim=1) - mu_fn, dim=1) * qv_qk).sum(1)
            sc_spat = (cand_spat_emb(lat16) * qsp_qk).sum(1) if qsp_qk is not None else torch.zeros_like(sc_sem)
            sc_vae = row_corr(lat16, z16_qk)
            sc_grd = row_corr(rgb_grad32(img_cpu), g32_qk)

            sc = sc_sem + args.sel_spat * sc_spat + args.sel_vae * sc_vae + args.sel_grad * sc_grd
            upd = sc > best_sc
            best_img[upd], best_sc[upd] = img_cpu[upd], sc[upd]

            if selvars:
                img8 = (img_cpu * 255).round().to(torch.uint8)
                for spec, wsp_, wv_, wg_, n_lim in selvars:
                    if n_cand < n_lim:
                        sv_s = sc_sem + wsp_ * sc_spat + wv_ * sc_vae + wg_ * sc_grd
                        u_ = sv_s > sv_best[spec]
                        sv_cur[spec][u_], sv_best[spec][u_] = img8[u_], sv_s[u_]

        rec[:, q] = best_img.view(Q, K, 3, 512, 512).transpose(0, 1)
        for spec in sv_cur:
            sv_rec[spec][:, q] = sv_cur[spec].view(Q, K, 3, 512, 512).transpose(0, 1)

if hook_handle is not None:
    hook_handle.remove()
del p2, pipe, attn, proj, A, vae32, femb
torch.cuda.empty_cache()
assert torch.isfinite(rec).all()
if args.dump:
    np.save(out / "rec_u8.npy", (rec * 255).round().to(torch.uint8).numpy())

per_sample = score(rec, gt)
info["per_sample"] = per_sample
info["mean"] = {k: float(np.mean([s_[k] for s_ in per_sample])) for k in per_sample[0]}
if selvars:
    info["selvar"] = {}
    for spec in sv_rec:
        ps = score(sv_rec[spec].float() / 255, gt)
        pm = {k: float(np.mean([s_[k] for s_ in ps])) for k in ps[0]}
        info["selvar"][spec] = pm
        print(f"{args.name} selvar {spec} | pixcorr {pm['pixcorr']:.4f} ssim {pm['ssim']:.4f} alex2 {pm['alex2']:.4f} "
              f"alex5 {pm['alex5']:.4f} incep {pm['inception']:.4f} clip {pm['clip']:.4f} effnet {pm['effnet']:.4f} clip_top1 {pm['clip_top1']:.4f}", flush=True)
if args.post:
    from torchvision.transforms.functional import gaussian_blur  # noqa: E402

    def lowpass(x, sg):
        ks = int(6 * sg) | 1
        return torch.cat([gaussian_blur(x[i:i + 50].to(dev), [ks, ks], [sg, sg]).cpu() for i in range(0, len(x), 50)])
    info["post"] = {}
    for spec in args.post.split(";"):
        w, sg, beta = (float(v) for v in spec.split(","))
        ls = lowpass(init, sg)
        out_ = torch.empty_like(rec)
        for k in range(K):
            lg = lowpass(rec[k], sg)
            out_[k] = ((1 - w) * lg + w * ls + beta * (rec[k] - lg)).clamp(0, 1)
        ps = score(out_, gt)
        pm = {k: float(np.mean([s_[k] for s_ in ps])) for k in ps[0]}
        info["post"][spec] = pm
        print(f"{args.name} post {spec} | pixcorr {pm['pixcorr']:.4f} ssim {pm['ssim']:.4f} alex2 {pm['alex2']:.4f} "
              f"alex5 {pm['alex5']:.4f} incep {pm['inception']:.4f} clip {pm['clip']:.4f} effnet {pm['effnet']:.4f} clip_top1 {pm['clip_top1']:.4f}", flush=True)

write_json(out / "metrics.json", info)
mm = info["mean"]
print(f"{args.name} | pixcorr {mm['pixcorr']:.4f} ssim {mm['ssim']:.4f} alex2 {mm['alex2']:.4f} alex5 {mm['alex5']:.4f} "
      f"incep {mm['inception']:.4f} clip {mm['clip']:.4f} effnet {mm['effnet']:.4f} clip_top1 {mm['clip_top1']:.4f}", flush=True)
