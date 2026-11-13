"""The 'ordinary' bounded condition correction of Botwwt/EEG-Diffusion, re-run on this pipeline's encoders.

Network: research/eeg/brainhive_bounded_component_v1.py (imported unchanged, 525,568 parameters, radius 0.25).
Recipe:  research/eeg/brainhive_full1654_ordinary_engine_20261101_v1.py -- frozen SDXL-Turbo UNet / A / P, loss =
         noise-prediction MSE at a random DDPM step + 0.1 * (1 - cos) between P(A(c')) and P(A(F)) tokens, AdamW lr 1e-4,
         wd 0.01, grad clip 1, batch 16, 3 passes, last checkpoint, seed 1729, strength 1. Inputs are out-of-fold
         conditions (4-repeat mean), each with the LayerNorm affine of the encoder that produced it.
Differences from theirs: one head per subject (theirs pooled the ten subjects into one head) and five concept folds.

  python ordh.py latents                 -> data/prep/lat512_train.npy (VAE mode latents of the training images)
  python ordh.py train <sub> [passes]    -> runs/ord/sub-XX/head.pt + log.json
  python ordh.py apply <sub>             -> runs/cond/ord/sub-XX.npy (corrected official-test conditions, 80 repeats)
  python ordh.py train all [passes]     -> one head shared by the ten subjects (the co-author's pooled recipe) -> runs/ord*/pooled
  ORD_HEAD=<name> python ordh.py train all / apply <sub>  -> shared head saved as / read from runs/ord*/<name>; conditions -> runs/cond/ord*_<name>/
  ORD_TAG=<encoder tag>                  -> use runs/enc/<tag> encoders; outputs go to runs/ord_<tag>, runs/cond/ord_<tag>
  ORD_ENS=mw7 python ordh.py train <sub> -> train on the mean of the out-of-fold conditions of the O6 ensemble (windows
                                            0-250 / 0-150 / 0-100 x seeds 2025, 2028-2033, per fold), rescaled per sample to
                                            the norm of the seed-2025 full-window fold encoder, with that encoder's affine,
                                            so that the head sees at training the kind of input it gets at test (O6);
                                            -> runs/ord_mw7/sub-XX/head.pt (recipe otherwise unchanged)
"""
import json
import os
import sys
import time

from pathlib import Path

import numpy as np
import torch

from common import HIVE, IMG, PREP, PRIOR, ROOT, RUNS, TURBO, concept_folds, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eeg"))  # research/eeg of this repository
from brainhive_bounded_component_v1 import BoundedConditionAdapter  # noqa: E402

dev = "cuda"
SEED = 1729
TAG = os.environ.get("ORD_TAG", "")   # encoder tag; '' = the original per-subject encoders (t0_r4 / t0_r4f)
SFX = f"_{TAG}" if TAG else ""
ENS = os.environ.get("ORD_ENS", "")
ENS_SETS = {"mw7": (["t0_r4", "t0_r4_w150", "t0_r4_w100"], [2025, 2028, 2029, 2030, 2031, 2032, 2033])}
if ENS:
    SFX += f"_{ENS}"
GK, BK = "backbone.block.layers.0.norm.weight", "backbone.block.layers.0.norm.bias"


def enc_dir(sub, fold):
    for tag in ((TAG,) if TAG else ("t0_r4", "t0_r4f")):  # t0_r4f: same recipe, re-run only to have the fold encoders saved
        d = RUNS / "enc" / tag / f"sub-{sub:02d}" / f"fold{fold}_seed2025"
        if (d / "brain_model.pt").exists() and (d / "metrics.json").exists():
            return d
    raise FileNotFoundError(f"no saved encoder for sub {sub} fold {fold}")


def affine(d):
    sd = torch.load(d / "brain_model.pt", weights_only=True)
    return sd[GK].float(), sd[BK].float()


cmd = sys.argv[1]
if cmd == "latents":
    from diffusers import AutoencoderKL
    from PIL import Image
    from torchvision import transforms as T
    vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval()
    vae.enable_slicing()
    tf = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(512), T.ToTensor(), T.Normalize([.5] * 3, [.5] * 3)])
    ids = json.loads((PREP / "train_ids.json").read_text())["ids"]
    files = {f.stem: f for f in (IMG / "training_images").glob("*/*.jpg")}
    lat = np.zeros((len(ids), 4, 64, 64), np.float16)
    ok = np.ones(len(ids), bool)
    for s in range(0, len(ids), 32):
        idx, ims = [], []
        for j in range(s, min(len(ids), s + 32)):
            try:
                ims.append(tf(Image.open(files[ids[j]]).convert("RGB")))
                idx.append(j)
            except OSError:
                ok[j] = False
        with torch.no_grad():
            z = vae.encode(torch.stack(ims).to(dev)).latent_dist.mode().float() * vae.config.scaling_factor
        lat[idx] = z.cpu().numpy().astype(np.float16)
        if s % 3200 == 0:
            print("latents", s, flush=True)
    np.save(PREP / "lat512_train.npy", lat)
    np.save(PREP / "lat512_train_ok.npy", ok)
    print("LATENTS_DONE", int(ok.sum()), "of", len(ids))
    sys.exit(0)

POOLED = sys.argv[2] == "all"
subs = list(range(1, 11)) if POOLED else [int(sys.argv[2])]
sub = subs[0]
HEAD = os.environ.get("ORD_HEAD", "")      # name of a shared head (train all: where to save it; apply: which one to use)
out = RUNS / f"ord{SFX}" / ((HEAD or "pooled") if POOLED else f"sub-{sub:02d}")
out.mkdir(parents=True, exist_ok=True)

if cmd == "apply":
    d = enc_dir(sub, -1)
    w, b = affine(d)
    c = torch.from_numpy(np.load(d / "cond_test_r80.npy")).float()
    head = BoundedConditionAdapter(predict_risk=False).eval()
    head.load_state_dict(torch.load((RUNS / f"ord{SFX}" / HEAD if HEAD else out) / "head.pt", weights_only=True))
    with torch.no_grad():
        cc, info = head(c, w.expand_as(c), b.expand_as(c))
    nz = torch.nn.functional.normalize
    Fn = nz(torch.from_numpy(np.load(PREP / "F_test.npy")), dim=1)
    t = torch.arange(200)
    acc = lambda x: float(((nz(x, dim=1) @ Fn.T).argmax(1) == t).float().mean())  # noqa: E731
    cdir = RUNS / "cond" / (f"ord{SFX}" + (f"_{HEAD}" if HEAD else ""))
    cdir.mkdir(parents=True, exist_ok=True)
    np.save(cdir / f"sub-{sub:02d}.npy", cc.numpy())
    ang = torch.rad2deg(torch.acos((nz(c, dim=1) * nz(cc, dim=1)).sum(1).clamp(-1, 1)))
    print(f"sub-{sub:02d} apply: angle moved {float(ang.mean()):.2f} deg | cos to F {float((nz(c, dim=1) * Fn).sum(1).mean()):.3f} -> "
          f"{float((nz(cc, dim=1) * Fn).sum(1).mean()):.3f} | top1 {acc(c):.3f} -> {acc(cc):.3f}", flush=True)
    sys.exit(0)

assert cmd == "train"
passes = float(sys.argv[3]) if len(sys.argv) > 3 else 3
torch.manual_seed(SEED)
concept = torch.from_numpy(np.load(PREP / "train_concept.npy"))
fold = torch.from_numpy(concept_folds(1654, 5))[concept]
NI, S = 16540, len(subs)
C = torch.zeros(S, NI, 1024)
W, B = torch.zeros(S * 5, 1024), torch.zeros(S * 5, 1024)    # row j*5+k: affine of subject j's fold-k encoder
for j, s_ in enumerate(subs):
    for k in range(5):
        d = enc_dir(s_, k)
        idx = torch.from_numpy(np.load(d / "oof_index.npy"))
        assert (fold[idx] == k).all()
        C[j, idx] = torch.from_numpy(np.load(d / "cond_oof_r4.npy")).float()
        if ENS:
            tags, seeds = ENS_SETS[ENS]
            cs = []
            for tg in tags:
                for sd_ in seeds:
                    e = d if (tg, sd_) == ("t0_r4", 2025) else RUNS / "enc" / tg / f"sub-{s_:02d}" / f"fold{k}_seed{sd_}"
                    assert (torch.from_numpy(np.load(e / "oof_index.npy")) == idx).all(), e
                    cs.append(torch.from_numpy(np.load(e / "cond_oof_r4.npy")).float())
            m_ = torch.stack(cs).mean(0)
            C[j, idx] = m_ * C[j, idx].norm(dim=1, keepdim=True) / m_.norm(dim=1, keepdim=True)
        W[j * 5 + k], B[j * 5 + k] = affine(d)
pair_img, pair_sub = torch.arange(S * NI) % NI, torch.arange(S * NI) // NI   # sample n = (subject n // NI, image n % NI)
data = dict(c=C.view(-1, 1024).to(dev), w=W.to(dev), b=B.to(dev), fold=fold.to(dev), img=pair_img.to(dev), sid=pair_sub.to(dev),
            target=torch.from_numpy(np.load(PREP / "F_train.npy")).to(dev), latent=torch.from_numpy(np.load(PREP / "lat512_train.npy")).to(dev))
valid = torch.nonzero(torch.from_numpy(np.load(PREP / "lat512_train_ok.npy"))[pair_img]).squeeze(1)

sys.path.insert(0, str(HIVE))
from diffusers import DDPMScheduler, StableDiffusionXLPipeline  # noqa: E402
from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel  # noqa: E402

pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=torch.float16, variant="fp16")
attn = IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter")
attn.bind_unet(pipe.unet)
pipe.to(dev, torch.float16)
P = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, torch.float16).eval().requires_grad_(False)
A = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, torch.float16).eval().requires_grad_(False)
for m in pipe.components.values():
    if isinstance(m, torch.nn.Module):
        m.eval().requires_grad_(False)
pipe.unet.train()
if os.environ.get("ORD_CKPT", "1") == "1":  # checkpointing only trades speed for memory; the numbers are the same
    pipe.unet.enable_gradient_checkpointing()
with torch.no_grad():
    prompt, _, pooled, _ = pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False)
prompt, pooled = prompt.detach().clone().to(torch.bfloat16), pooled.detach().clone().to(torch.bfloat16)
time_ids = pipe._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=torch.bfloat16, text_encoder_projection_dim=pooled.shape[-1]).to(dev)
ddpm = DDPMScheduler.from_config(pipe.scheduler.config)
assert ddpm.config.prediction_type == "epsilon" and ddpm.config.num_train_timesteps == 1000
del pipe.vae, pipe.text_encoder, pipe.text_encoder_2
torch.cuda.empty_cache()


def denoise(noisy, t, tokens):
    n = len(noisy)
    return pipe.unet(noisy.to(pipe.unet.dtype), t, encoder_hidden_states=prompt.expand(n, -1, -1),
                     added_cond_kwargs=dict(text_embeds=pooled.expand(n, -1), time_ids=time_ids.expand(n, -1)),
                     cross_attention_kwargs={"ip_hidden_states": tokens}, return_dict=False)[0]


head = BoundedConditionAdapter(predict_risk=False).to(dev)
assert sum(p.numel() for p in head.parameters()) == 525568
opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01, betas=(.9, .999), eps=1e-8)
noise_rng = torch.Generator(dev).manual_seed(SEED)
order_rng = torch.Generator().manual_seed(SEED)
total = int(len(valid) * passes)
order = torch.cat([valid[torch.randperm(len(valid), generator=order_rng)] for _ in range(int(np.ceil(passes)))])[:total]
log, t0, acc = [], time.time(), dict(eps=0.0, tok=0.0, n=0)
Fnn = torch.nn.functional
for step, s in enumerate(range(0, total, 16)):
    n_ = order[s:s + 16].to(dev)
    i = data["img"][n_]
    r_ = data["sid"][n_] * 5 + data["fold"][i]
    c, w, b = data["c"][n_], data["w"][r_], data["b"][r_]
    latent = data["latent"][i].float()
    noise = torch.randn(latent.shape, device=dev, generator=noise_rng)
    t = torch.randint(0, 1000, (len(i),), device=dev, generator=noise_rng)
    noisy = ddpm.add_noise(latent, noise, t)
    opt.zero_grad(set_to_none=True)
    changed, info = head(c, w, b)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            teacher = P(A(data["target"][i]))
        tokens = P(A(changed))
        pred = denoise(noisy, t, tokens)
        eps_loss = Fnn.mse_loss(pred.float(), noise)
        tok_loss = (1 - Fnn.cosine_similarity(tokens.float(), teacher.float(), dim=-1)).mean()
        loss = eps_loss + .1 * tok_loss
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    acc["eps"] += float(eps_loss); acc["tok"] += float(tok_loss); acc["n"] += 1
    if (step + 1) % 100 == 0 or s + 16 >= total:
        rec = dict(step=step + 1, eps=acc["eps"] / acc["n"], tok=acc["tok"] / acc["n"], seconds=time.time() - t0)
        log.append(rec)
        print(f"{'pooled' if POOLED else f'sub-{sub:02d}'} step {step + 1}/{(total + 15) // 16} eps {rec['eps']:.4f} tok {rec['tok']:.4f} {rec['seconds']:.0f}s "
              f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        acc = dict(eps=0.0, tok=0.0, n=0)
torch.save({k: v.cpu() for k, v in head.state_dict().items()}, out / "head.pt")
write_json(out / "log.json", dict(subs=subs, passes=passes, updates=step + 1, seed=SEED, log=log))
print(f"ORD_TRAIN_DONE {'pooled' if POOLED else f'sub-{sub:02d}'}", flush=True)
