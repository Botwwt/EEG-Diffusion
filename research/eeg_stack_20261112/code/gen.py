"""Generate official-test reconstructions from one trained encoder and score them.

Chain: EEG (mean of the chosen repeats) -> encoder c -> [calibration] -> [A] -> IP projection P -> IP attention
-> SDXL-Turbo (4 steps, guidance 0, 512x512, empty prompt), as in Brain-HIVE build_reconstruction.py.
  --use_A 0  official script at commit f06db69: P(c), the Fusion Prior's residual adapter A is skipped
  --use_A 1  paper Eq. 6 / prior training: P(A(c))
Calibration (--alpha > 0): c' = ||c|| * normalize(c/||c|| + alpha * g), g = mean(F/||F||) - mean(c_oof/||c_oof||),
both means over training images, c_oof from encoders that never saw the image's concept (same tag/seed, folds 0-4).

Noise is fixed per (query, sample) and per batch, so every arm sees identical noise. Metrics follow
main/utils_eval.py (MindEye-style); SwAV is omitted (its weights cannot be downloaded here).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from common import CH17, HIVE, IMG, PREP, PRIOR, RUNS, TURBO, write_json

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="encoder run dir relative to runs/enc, e.g. t0_r4/sub-08/fold-1_seed2025")
p.add_argument("--name", required=True)
p.add_argument("--use_A", type=int, default=1)
p.add_argument("--alpha", type=float, default=0.0)
p.add_argument("--nsel", type=int, default=1, help="candidates per output image; the one whose embedding fits the EEG condition best is kept")
p.add_argument("--init_file", default="", help="low-level start images [200, 3, h, h] in [0, 1] (.npy, relative to runs/low); img2img instead of pure noise")
p.add_argument("--strength", type=float, default=0.5, help="img2img strength with --init_file (4 steps * strength are run)")
p.add_argument("--steps", type=int, default=4, help="sampler steps (official: 4)")
p.add_argument("--blend", type=float, default=0.0, help="final image = (1 - blend) * generated + blend * start image")
p.add_argument("--cond_file", default="", help="precomputed conditions [200, 1024] .npy, relative to runs/cond")
p.add_argument("--amp", type=float, default=1.0, help="c' = |c| * normalize(m + amp * (c/|c| - m)), m = mean direction of the 200 test conditions (label-free)")
p.add_argument("--cfg", type=float, default=1.0, help="classifier-free guidance scale on the IP condition; the negative is the mean test condition")
p.add_argument("--ip_scale", type=float, default=1.0, help="weight of the IP attention branch in every block (official 1)")
p.add_argument("--prompt_file", default="", help="200 text prompts (.json list, relative to runs/prompt); default: empty prompt")
p.add_argument("--loop", type=float, default=0.0, help="closed-loop correction strength: generate, embed the images back to F, push each condition along the part of its "
               "centred direction the images miss, regenerate with the same noise")
p.add_argument("--loop_iters", type=int, default=1)
p.add_argument("--sel_low", type=float, default=0.0, help="best-of-N score = semantic fit + sel_low * pixel correlation (64 px grey) with the low-level start image")
p.add_argument("--proj_ckpt", default="", help="fine-tuned IP projection (ipft.py), path relative to runs/")
p.add_argument("--post", default="", help="extra scored variants (';'-separated 'w,sigma,beta'): out = (1-w)*LF(gen) + w*LF(start) + beta*(gen - LF(gen)), "
               "LF = Gaussian blur with sigma px at 512; or 'm,sigma,beta': high band shrunk towards the mean of the K samples; the main metrics stay those of the unmodified images")
p.add_argument("--selvar", default="", help="extra best-of-N variants scored in the same run (';'-separated 'target@n'): target = '+'-joined weighted terms "
               "enc (raw EEG direction, the default target), head (corrected condition, centred on the test mean), knn (kNN prototype of training-image "
               "embeddings), e.g. 'enc+knn0.5@8'; n = candidates used (<= nsel)")
p.add_argument("--dump", type=int, default=0, help="save the final images as uint8 rec.npy [K, 200, 3, 512, 512] (for offline post-processing)")
p.add_argument("--amp_mix", default="", help="comma-separated amplification factors cycled over the best-of-N candidates (candidate n uses factor n mod L); overrides --amp")
p.add_argument("--smap", default="", help="reliability-driven differential img2img: '<kind>,s_rel,s_unrel', kind map / img (controls: const, shuf). The strength at each latent "
               "position is s_unrel - (s_unrel - s_rel) * rho, rho = split-half reliability of the start-image latent (rel.py: map per position, or "
               "one value per image); positions are held at the noised start latent until their own number of steps remains (--strength is ignored)")
p.add_argument("--ilvr", default="", help="low-band anchoring (ILVR, Choi et al. 2021, transferred): 'f,w,k'. After each of the first k sampler steps "
               "(k = 0: every step but the last) the latent's low band LP_f (average-pool by f, bilinear back) is moved by w towards that of the "
               "start latent noised to the current level: x += w * (LP_f(z0 + sigma * eps) - LP_f(x)). f = 4 keeps exactly the 16 x 16 band the "
               "EEG decoder predicts; uses --strength as the img2img start")
p.add_argument("--ip_sched", default="", help="IP-attention weight per sampler step, comma-separated over the steps that are run (e.g. '1,1.5,1,1' with "
               "4 steps); set before each step through the step callback")
p.add_argument("--reps", default="0:80", help="test repeats averaged into the input, start:stop")
p.add_argument("--K", type=int, default=4, help="images per query")
p.add_argument("--bs", type=int, default=0, help="queries per batch (0: about 20 images per batch)")
p.add_argument("--save", type=int, default=1, help="save sample 0 of every query as JPEG")
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
from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPAttnProcessor2_0, IPProjectionModel  # noqa: E402
from main.models_brain import BrainEncoder  # noqa: E402

nz = torch.nn.functional.normalize

# ------------------------------------------------------------ conditions
x = torch.from_numpy(np.load(PREP / f"sub-{ea['sub']:02d}_test.npy"))
if ea["chan"] == "typo":
    x[:, :, CH17.index("POz")] = x[:, :, CH17.index("Oz")].clone()
a, b = (int(v) for v in args.reps.split(":"))
x = x[:, a:b].mean(1)[..., ea["t0"]:ea["t1"]]
if ea["inorm"]:
    x = x / x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
enc = BrainEncoder(backbone="brain_mlp", brain_channels=17, brain_sequence_length=ea["t1"] - ea["t0"],
                   hidden_size=1024, embedding_dim=1440, dropout=0.1)
enc.load_state_dict(torch.load(run / "brain_model.pt", weights_only=True))
enc.eval()
with torch.no_grad():
    c = enc.backbone(x).float()                                           # [200, 1024]
c_enc = c.clone()
if args.cond_file:  # precomputed conditions (prior.py); --run only supplies the metadata
    f = Path(args.cond_file)
    c = torch.from_numpy(np.load(f if f.is_absolute() else RUNS / "cond" / f)).float()
    assert c.shape == (200, 1024) and args.alpha == 0
info = dict(args=vars(args), encoder_args=ea, cond_norm=float(c.norm(dim=1).mean()))
if args.alpha > 0 or args.nsel > 1:
    F_tr = torch.from_numpy(np.load(PREP / "F_train.npy"))
    oof = torch.zeros_like(F_tr)
    seen = torch.zeros(len(F_tr), dtype=torch.bool)
    for k in range(5):
        d = run.parent / f"fold{k}_seed{ea['seed']}"
        idx = torch.from_numpy(np.load(d / "oof_index.npy"))
        oof[idx] = torch.from_numpy(np.load(d / "cond_oof_r4.npy"))
        seen[idx] = True
    assert seen.all()
    mu_fn, mu_x = nz(F_tr, dim=1).mean(0), nz(oof, dim=1).mean(0)
    g = mu_fn - mu_x
    qv = nz(nz(c_enc, dim=1) - mu_x, dim=1)                                # centred EEG direction, used to score candidates
c_head = c.clone()
if args.selvar:
    assert args.nsel > 1
    Fc = nz(nz(F_tr.float(), dim=1) - mu_fn, dim=1)                     # centred training-image embeddings
    sim = qv @ Fc.T                                                         # [200, 16540]
    tv, ti = sim.topk(20, dim=1)
    knn = nz((torch.softmax(tv / 0.05, dim=1).unsqueeze(-1) * Fc[ti]).sum(1), dim=1)
    hd = nz(nz(c_head, dim=1) - nz(c_head, dim=1).mean(0), dim=1)
    terms = dict(enc=qv, head=hd, knn=knn)
    selvars = []
    for spec in args.selvar.split(";"):
        tgt, n_ = spec.split("@")
        v = torch.zeros_like(qv)
        for t_ in tgt.split("+"):
            name = t_.rstrip("0123456789.")
            v = v + float(t_[len(name):] or 1) * terms[name]
        assert int(n_) <= args.nsel
        selvars.append((spec, nz(v, dim=1), int(n_)))
if args.alpha > 0:
    info.update(gap_norm=float(g.norm()), gap_cos_before=float((nz(c, dim=1) @ nz(g, dim=0)).mean()))
    c = c.norm(dim=1, keepdim=True) * nz(nz(c, dim=1) + args.alpha * g, dim=1)
def amplify(c, a):
    cn = nz(c, dim=1)
    m = cn.mean(0, keepdim=True)
    return c.norm(dim=1, keepdim=True) * nz(m + a * (cn - m), dim=1)


c_mix = [amplify(c, float(a)) for a in args.amp_mix.split(",")] if args.amp_mix else None
if args.amp != 1.0:
    cn = nz(c, dim=1)
    m = cn.mean(0, keepdim=True)
    c = c.norm(dim=1, keepdim=True) * nz(m + args.amp * (cn - m), dim=1)
c_neg = nz(c.mean(0, keepdim=True), dim=1) * c.norm(dim=1).mean()       # generic condition, the CFG negative

# ------------------------------------------------------------ generator
from diffusers import StableDiffusionXLPipeline  # noqa: E402

pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=dt, variant="fp16")
attn = IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter")
attn.bind_unet(pipe.unet)
proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj")
if args.proj_ckpt:
    proj.load_state_dict(torch.load(RUNS / args.proj_ckpt, weights_only=True))
proj = proj.to(dev, dt).eval()
A = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, dt).eval()
pipe.to(dev, dt)
if args.ip_scale != 1.0:
    n_ip = 0
    for proc in pipe.unet.attn_processors.values():
        if isinstance(proc, IPAttnProcessor2_0):
            proc.scale, n_ip = args.ip_scale, n_ip + 1
    assert n_ip > 0
    info["n_ip_processors"] = n_ip
IP_PROCS = [proc for proc in pipe.unet.attn_processors.values() if isinstance(proc, IPAttnProcessor2_0)]
ip_sched = [float(v) for v in args.ip_sched.split(",")] if args.ip_sched else []


def set_ip(v):
    for proc in IP_PROCS:
        proc.scale = v


if ip_sched:
    assert args.init_file and IP_PROCS
pipe.vae.enable_slicing()
pipe.set_progress_bar_config(disable=True)
args.bs = args.bs or max(1, 20 // args.K)
info["scheduler"] = type(pipe.scheduler).__name__
with torch.inference_mode():
    pe, _, ppe, _ = pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
    pe_all = ppe_all = None
    if args.prompt_file:
        prompts = json.loads((RUNS / "prompt" / args.prompt_file).read_text())
        assert len(prompts) == 200
        enc_ = [pipe.encode_prompt(prompts[i:i + 50], device=dev, do_classifier_free_guidance=False)[::2] for i in range(0, 200, 50)]
        pe_all, ppe_all = torch.cat([e[0] for e in enc_]), torch.cat([e[1] for e in enc_])
        info["prompts_first"] = prompts[:5]

ids = json.loads((PREP / "test_ids.json").read_text())
gt_tf = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(512), T.ToTensor()])
files = {f.stem: f for f in (IMG / "test_images").glob("*/*.jpg")}
gt = torch.stack([gt_tf(Image.open(files[i]).convert("RGB")) for i in ids["ids"]])   # [200, 3, 512, 512]

K, N, NS = args.K, len(c), args.nsel
init = None
if args.init_file:
    from diffusers import StableDiffusionXLImg2ImgPipeline  # noqa: E402
    f = Path(args.init_file)
    init = torch.from_numpy(np.load(f if f.is_absolute() else RUNS / "low" / f)).float()
    init = torch.nn.functional.interpolate(init, size=512, mode="bicubic").clamp(0, 1)
    p2 = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    p2.set_progress_bar_config(disable=True)
smap = None
ilvr = None
if args.ilvr:
    assert init is not None and not args.smap
    f_, w_, k_ = args.ilvr.split(",")
    ilvr = (int(f_), float(w_), int(k_))
    n_run = min(int(args.steps * args.strength), args.steps)
    vae32 = __import__("copy").deepcopy(pipe.vae).float()
if args.smap:
    assert init is not None
    kind, s_rel, s_unrel = args.smap.split(",")
    s_rel, s_unrel = float(s_rel), float(s_unrel)
    assert 0.25 <= s_rel <= s_unrel <= 1
    if kind.startswith("c") and kind != "const":  # fixed value: no reliability file needed
        rz, lm = None, torch.zeros(200, 1, 16, 16)
    else:
        rz = np.load(RUNS / "rel" / f"sub-{ea['sub']:02d}.npz")
        lm = torch.from_numpy(rz["low_map"]).float()[:, None]             # [200, 1, 16, 16]
    if kind == "map":
        rho = lm
    elif kind == "img":
        rho = torch.from_numpy(rz["low"]).float().view(-1, 1, 1, 1).expand(-1, 1, 16, 16)
    elif kind == "const":    # control: same mean reliability everywhere (no spatial or per-image information)
        rho = torch.full_like(lm, float(lm.clamp(0, 1).mean()))
    elif kind == "shuf":     # control: each image gets the map of another image (fixed permutation)
        rho = lm[torch.randperm(200, generator=torch.Generator().manual_seed(5))]
    elif kind.startswith("c"):  # control: one fixed reliability for every subject and position, e.g. 'c0.611'
        rho = torch.full_like(lm, float(kind[1:]))
    else:
        raise ValueError(kind)
    rho = torch.nn.functional.interpolate(rho, size=64, mode="bilinear", align_corners=False).clamp(0, 1)
    smap = s_unrel - (s_unrel - s_rel) * rho                              # [200, 1, 64, 64] strength per latent position
    args.strength = s_unrel
    n_run = min(int(args.steps * s_unrel), args.steps)
    hold = (n_run - args.steps * smap).clamp(min=0)                       # steps each position is held at the noised start latent
    vae32 = __import__("copy").deepcopy(pipe.vae).float()
    info.update(smap_mean_strength=float(smap.mean()), smap_mean_hold=float(hold.mean()))
def generate(cv, q, s, n):
    lat = torch.stack([torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(1_000_003 * i + k + 12345 + 100_003 * n))
                       for i in q for k in range(K)]).to(dev, dt)
    with torch.inference_mode():
        cc = cv[q].to(dev, dt)
        ips = proj(A(cc) if args.use_A else cc).repeat_interleave(K, dim=0)
        m_ = len(ips)
        pe_q = pe.expand(m_, -1, -1) if pe_all is None else pe_all[q].repeat_interleave(K, dim=0)
        ppe_q = ppe.expand(m_, -1) if ppe_all is None else ppe_all[q].repeat_interleave(K, dim=0)
        kw = dict(prompt_embeds=pe_q, pooled_prompt_embeds=ppe_q,
                  num_inference_steps=args.steps, guidance_scale=0.0, output_type="pt")
        if args.cfg > 1:  # the pipeline batches [negative, positive]; the IP tokens must follow the same order
            cn_ = c_neg.to(dev, dt)
            neg = proj(A(cn_) if args.use_A else cn_).expand(m_, -1, -1)
            ips = torch.cat([neg, ips])
            kw.update(guidance_scale=args.cfg, negative_prompt_embeds=pe.expand(m_, -1, -1), negative_pooled_prompt_embeds=ppe.expand(m_, -1))
        kw["cross_attention_kwargs"] = {"ip_hidden_states": ips}
        if init is None:
            img = pipe(height=512, width=512, latents=lat, generator=torch.Generator(dev).manual_seed(777 + s + 100_000 * n), **kw).images
        else:  # start from the noised low-level image; one generator per image keeps the noise tied to (query, sample)
            gens = [torch.Generator(dev).manual_seed(1_000_003 * i + k + 12345 + 100_003 * n) for i in q for k in range(K)]
            i0 = init[q].repeat_interleave(K, dim=0).to(dev, dt)
            cbs = []
            if smap is not None or ilvr is not None:
                with torch.no_grad():
                    z0 = vae32.encode(i0.float() * 2 - 1).latent_dist.mean * vae32.config.scaling_factor
                eps = torch.stack([torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * i + k + 100_003 * n + 99))
                                   for i in q for k in range(K)]).to(dev)
            if ilvr is not None:
                f_, w_, k_ = ilvr

                def lp(z):
                    return torch.nn.functional.interpolate(torch.nn.functional.avg_pool2d(z, f_), size=z.shape[-1], mode="bilinear", align_corners=False)

                def cb(pp, i, t, ck):
                    if i < n_run - 1 and (k_ == 0 or i < k_):
                        sig = pp.scheduler.sigmas[pp.scheduler.step_index].to(dev)
                        lt = ck["latents"].float()
                        ck["latents"] = (lt + w_ * (lp(z0 + sig * eps) - lp(lt))).to(ck["latents"].dtype)
                    return ck
                cbs.append(cb)
            if smap is not None:
                h_ = hold[q].repeat_interleave(K, dim=0).to(dev)

                def cb(pp, i, t, ck):
                    m_ = (h_ - i).clamp(0, 1)
                    if float(m_.max()) > 0:
                        sig = pp.scheduler.sigmas[pp.scheduler.step_index].to(dev)
                        lt = ck["latents"]
                        ck["latents"] = (m_ * (z0 + sig * eps) + (1 - m_) * lt.float()).to(lt.dtype)
                    return ck
                cbs.append(cb)
            if ip_sched:
                set_ip(ip_sched[0])

                def cb_ip(pp, i, t, ck):
                    set_ip(ip_sched[min(i + 1, len(ip_sched) - 1)])
                    return ck
                cbs.append(cb_ip)
            if cbs:
                def cb_all(pp, i, t, ck):
                    for f in cbs:
                        ck = f(pp, i, t, ck)
                    return ck
                kw.update(callback_on_step_end=cb_all, callback_on_step_end_tensor_inputs=["latents"])
            img = p2(image=i0, strength=args.strength, generator=gens, **kw).images
            if args.blend > 0:
                img = (1 - args.blend) * img.float() + args.blend * i0.float()
    img = (img.float().clamp(0, 1) * 255).round().div(255)            # same 8-bit round trip as saving PNGs
    return img


rec = torch.empty(K, N, 3, 512, 512)
if NS > 1 or args.loop > 0:
    from fenc import FEmbedder  # noqa: E402
    femb = FEmbedder(dev)
for it in range(args.loop_iters if args.loop > 0 else 0):  # closed loop: first pass with candidate-0 noise
    fimg = torch.zeros(N, c.shape[1])
    for s in range(0, N, args.bs):
        q = list(range(s, min(N, s + args.bs)))
        fimg[q] = nz(femb(generate(c, q, s, 0)).cpu().float(), dim=1).view(len(q), K, -1).mean(1)
    u = nz(c, dim=1)
    dc, df = u - u.mean(0), fimg - fimg.mean(0)                                # centred, label-free (test-batch means)
    before = float((nz(dc, dim=1) * nz(df, dim=1)).sum(1).mean())
    u2 = u + args.loop * dc.norm(dim=1, keepdim=True) * (nz(dc, dim=1) - nz(df, dim=1))
    c = c.norm(dim=1, keepdim=True) * nz(u2, dim=1)
    info.setdefault("loop_cos_image_vs_condition", []).append(before)
    print(f"loop {it}: centred cos(image, condition) {before:.4f}", flush=True)
if NS > 1:
    n_changed, sc0_sum, scb_sum = 0, 0.0, 0.0
    sv_rec = {spec: torch.empty(K, N, 3, 512, 512, dtype=torch.uint8) for spec, _, _ in (selvars if args.selvar else [])}
for s in range(0, N, args.bs):
    q = list(range(s, min(N, s + args.bs)))
    best = torch.full((len(q) * K,), -1e9)
    cur = torch.empty(len(q) * K, 3, 512, 512)
    if args.selvar:
        sv_best = {spec: torch.full((len(q) * K,), -1e9) for spec, _, _ in selvars}
        sv_cur = {spec: torch.empty(len(q) * K, 3, 512, 512, dtype=torch.uint8) for spec, _, _ in selvars}
    for n in range(NS):  # candidate 0 is exactly the image a run without selection produces
        img = generate(c if c_mix is None else c_mix[n % len(c_mix)], q, s, n)
        if NS == 1:
            cur = img.cpu()
            break
        f = femb(img).cpu()
        sc = (nz(nz(f, dim=1) - mu_fn, dim=1) * qv[q].repeat_interleave(K, dim=0)).sum(1)
        if args.sel_low > 0:
            assert init is not None
            small = lambda z: torch.nn.functional.interpolate(z.float().mean(1, keepdim=True), size=64, mode="area").flatten(1)  # noqa: E731
            a_, b_ = small(img.cpu()), small(init[q].repeat_interleave(K, dim=0))
            a_, b_ = a_ - a_.mean(1, keepdim=True), b_ - b_.mean(1, keepdim=True)
            sc = sc + args.sel_low * (a_ * b_).sum(1) / (a_.norm(dim=1) * b_.norm(dim=1) + 1e-8)
        if args.selvar:
            fn_ = nz(nz(f, dim=1) - mu_fn, dim=1)
            img8 = (img.cpu() * 255).round().to(torch.uint8)
            for spec, v, n_ in selvars:
                if n < n_:
                    sv_sc = (fn_ * v[q].repeat_interleave(K, dim=0)).sum(1)
                    u_ = sv_sc > sv_best[spec]
                    sv_cur[spec][u_], sv_best[spec][u_] = img8[u_], sv_sc[u_]
        if n == 0:
            sc0_sum += float(sc.sum())
        upd = sc > best
        cur[upd], best[upd] = img.cpu()[upd], sc[upd]
        if n > 0:
            n_changed += int(upd.sum())
    if NS > 1:
        scb_sum += float(best.sum())
    rec[:, q] = cur.view(len(q), K, 3, 512, 512).transpose(0, 1)
    if args.selvar:
        for spec in sv_cur:
            sv_rec[spec][:, q] = sv_cur[spec].view(len(q), K, 3, 512, 512).transpose(0, 1)
if NS > 1:
    info.update(sel_score_first=sc0_sum / (N * K), sel_score_kept=scb_sum / (N * K), sel_updates_per_image=n_changed / (N * K))
p2 = None
femb = None
del pipe, attn, proj, A
torch.cuda.empty_cache()
assert torch.isfinite(rec).all()
if args.save:
    (out / "images").mkdir(exist_ok=True)
    for i, name in enumerate(ids["ids"]):
        Image.fromarray((rec[0, i].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
            out / "images" / f"{name}.jpg", quality=92)
    grid = torch.cat([torch.cat([gt[i], rec[0, i], rec[min(1, K - 1), i]], dim=2) for i in range(0, 200, 25)], dim=1)
    Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).resize((768, 2048)).save(
        out / "grid_target_s0_s1.jpg", quality=90)

if args.dump:
    np.save(out / "rec_u8.npy", (rec * 255).round().to(torch.uint8).numpy())
from metrics import score  # noqa: E402

per_sample = score(rec, gt)
info["per_sample"] = per_sample
info["mean"] = {k: float(np.mean([s[k] for s in per_sample])) for k in per_sample[0]}
write_json(out / "metrics.json", info)
mm = info["mean"]
if args.selvar:
    info["selvar"] = {}
    for spec in sv_rec:
        ps = score(sv_rec[spec].float() / 255, gt)
        pm = {k: float(np.mean([s_[k] for s_ in ps])) for k in ps[0]}
        info["selvar"][spec] = pm
        print(f"{args.name} selvar {spec} | " + " ".join(f"{k} {v:.4f}" for k, v in pm.items()), flush=True)
    write_json(out / "metrics.json", info)
if args.post:
    assert init is not None
    from torchvision.transforms.functional import gaussian_blur  # noqa: E402

    def lowpass(x, sg):
        ks = int(6 * sg) | 1
        return torch.cat([gaussian_blur(x[i:i + 50].to(dev), [ks, ks], [sg, sg]).cpu() for i in range(0, len(x), 50)])
    info["post"] = {}
    for spec in args.post.split(";"):
        out_ = torch.empty_like(rec)
        if spec.startswith("m"):  # 'm,sigma,beta': shrink the high band (sigma 0: all of it) towards the mean of the K samples of a query
            sg, beta = (float(v) for v in spec.split(",")[1:])
            hf = torch.stack([rec[k] - lowpass(rec[k], sg) for k in range(K)]) if sg > 0 else rec
            hm = hf.mean(0)
            for k in range(K):
                out_[k] = (rec[k] - (1 - beta) * (hf[k] - hm)).clamp(0, 1)
        else:
            w, sg, beta = (float(v) for v in spec.split(","))
            ls = lowpass(init, sg)
            for k in range(K):
                lg = lowpass(rec[k], sg)
                out_[k] = ((1 - w) * lg + w * ls + beta * (rec[k] - lg)).clamp(0, 1)
        ps = score(out_, gt)
        pm = {k: float(np.mean([s_[k] for s_ in ps])) for k in ps[0]}
        info["post"][spec] = pm
        print(f"{args.name} post {spec} | " + " ".join(f"{k} {v:.4f}" for k, v in pm.items()), flush=True)
    write_json(out / "metrics.json", info)
print(f"{args.name} | pixcorr {mm['pixcorr']:.4f} ssim {mm['ssim']:.4f} alex2 {mm['alex2']:.4f} alex5 {mm['alex5']:.4f} "
      f"incep {mm['inception']:.4f} clip {mm['clip']:.4f} effnet {mm['effnet']:.4f} clip_top1 {mm['clip_top1']:.4f}", flush=True)
