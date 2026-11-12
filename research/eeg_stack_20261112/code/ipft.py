"""Fine-tune the generator's IP projection P on EEG conditions (the interface the Fusion Prior only saw with clean F).

The shared ordinary head (frozen) corrects the out-of-fold EEG conditions of all ten subjects; P is trained so that
SDXL-Turbo denoises the training image from P(A(head(c))). Loss = noise-prediction MSE at a random DDPM step
+ tok_w * (1 - cos(P(A(c')), P0(A(F))))     (EEG tokens towards the original tokens of the true image)
+ anchor_w * (1 - cos(P(A(F)), P0(A(F))))   (P keeps its behaviour on clean image embeddings)
P0 = the original projection (frozen). Everything else (UNet, IP attention, A, head, encoders) stays frozen.

python ipft.py <encoder tag> <head name> <out name> [passes=1] [lr=1e-5]
  -> runs/ipft/<out name>/proj.pt + log.json      (gen.py --proj_ckpt runs/ipft/<out name>/proj.pt)
"""
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import HIVE, PREP, PRIOR, RUNS, TURBO, concept_folds, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eeg"))  # research/eeg of this repository
from brainhive_bounded_component_v1 import BoundedConditionAdapter  # noqa: E402

dev, SEED = "cuda", 1729
GK, BK = "backbone.block.layers.0.norm.weight", "backbone.block.layers.0.norm.bias"
tag, hname, oname = sys.argv[1], sys.argv[2], sys.argv[3]
passes = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
lr = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-5
tok_w, anchor_w = float(os.environ.get("IPFT_TOK", "0.1")), float(os.environ.get("IPFT_ANCHOR", "0.1"))
out = RUNS / "ipft" / oname
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)

concept = torch.from_numpy(np.load(PREP / "train_concept.npy"))
fold = torch.from_numpy(concept_folds(1654, 5))[concept]
subs, NI = list(range(1, 11)), 16540
S = len(subs)
C = torch.zeros(S, NI, 1024)
W, B = torch.zeros(S * 5, 1024), torch.zeros(S * 5, 1024)
for j, s_ in enumerate(subs):
    for k in range(5):
        d = RUNS / "enc" / tag / f"sub-{s_:02d}" / f"fold{k}_seed2025"
        idx = torch.from_numpy(np.load(d / "oof_index.npy"))
        C[j, idx] = torch.from_numpy(np.load(d / "cond_oof_r4.npy")).float()
        sd = torch.load(d / "brain_model.pt", weights_only=True)
        W[j * 5 + k], B[j * 5 + k] = sd[GK].float(), sd[BK].float()
head = BoundedConditionAdapter(predict_risk=False).eval()
head.load_state_dict(torch.load(RUNS / f"ord_{tag}" / hname / "head.pt", weights_only=True))
pair_img, pair_sub = torch.arange(S * NI) % NI, torch.arange(S * NI) // NI
r_all = pair_sub * 5 + fold[pair_img]
with torch.no_grad():   # the frozen head, applied once to every (subject, image) condition
    Cc = torch.cat([head(C.view(-1, 1024)[k:k + 8192], W[r_all[k:k + 8192]], B[r_all[k:k + 8192]])[0] for k in range(0, S * NI, 8192)])
data = dict(c=Cc.to(dev), img=pair_img.to(dev), target=torch.from_numpy(np.load(PREP / "F_train.npy")).to(dev),
            latent=torch.from_numpy(np.load(PREP / "lat512_train.npy")).to(dev))
valid = torch.nonzero(torch.from_numpy(np.load(PREP / "lat512_train_ok.npy"))[pair_img]).squeeze(1)

sys.path.insert(0, str(HIVE))
from diffusers import DDPMScheduler, StableDiffusionXLPipeline  # noqa: E402
from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel  # noqa: E402

pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=torch.float16, variant="fp16")
attn = IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter")
attn.bind_unet(pipe.unet)
pipe.to(dev, torch.float16)
P0 = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, torch.float16).eval().requires_grad_(False)
P = copy.deepcopy(P0).float().train().requires_grad_(True)
A = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, torch.float16).eval().requires_grad_(False)
for m in pipe.components.values():
    if isinstance(m, torch.nn.Module):
        m.eval().requires_grad_(False)
pipe.unet.train()
pipe.unet.enable_gradient_checkpointing()
with torch.no_grad():
    prompt, _, pooled, _ = pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False)
prompt, pooled = prompt.detach().clone().to(torch.bfloat16), pooled.detach().clone().to(torch.bfloat16)
time_ids = pipe._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=torch.bfloat16, text_encoder_projection_dim=pooled.shape[-1]).to(dev)
ddpm = DDPMScheduler.from_config(pipe.scheduler.config)
del pipe.vae, pipe.text_encoder, pipe.text_encoder_2
torch.cuda.empty_cache()


def denoise(noisy, t, tokens):
    n = len(noisy)
    return pipe.unet(noisy.to(pipe.unet.dtype), t, encoder_hidden_states=prompt.expand(n, -1, -1),
                     added_cond_kwargs=dict(text_embeds=pooled.expand(n, -1), time_ids=time_ids.expand(n, -1)),
                     cross_attention_kwargs={"ip_hidden_states": tokens}, return_dict=False)[0]


opt = torch.optim.AdamW(P.parameters(), lr=lr, weight_decay=.01)
noise_rng = torch.Generator(dev).manual_seed(SEED)
order_rng = torch.Generator().manual_seed(SEED)
total = int(len(valid) * passes)
order = torch.cat([valid[torch.randperm(len(valid), generator=order_rng)] for _ in range(int(np.ceil(passes)))])[:total]
F_ = torch.nn.functional
log, t0, acc = [], time.time(), dict(eps=0.0, tok=0.0, anc=0.0, n=0)
print(f"ipft {oname}: {sum(p.numel() for p in P.parameters())} trainable parameters, {(total + 15) // 16} steps, lr {lr}", flush=True)
for step, s in enumerate(range(0, total, 16)):
    n_ = order[s:s + 16].to(dev)
    i = data["img"][n_]
    latent = data["latent"][i].float()
    noise = torch.randn(latent.shape, device=dev, generator=noise_rng)
    t = torch.randint(0, 1000, (len(i),), device=dev, generator=noise_rng)
    noisy = ddpm.add_noise(latent, noise, t)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            aF = A(data["target"][i].half())
            teacher = P0(aF)
            ac = A(data["c"][n_].half())
        tokens = P(ac.float())
        pred = denoise(noisy, t, tokens)
        eps_loss = F_.mse_loss(pred.float(), noise)
        tok_loss = (1 - F_.cosine_similarity(tokens.float(), teacher.float(), dim=-1)).mean()
        anc_loss = (1 - F_.cosine_similarity(P(aF.float()).float(), teacher.float(), dim=-1)).mean()
        loss = eps_loss + tok_w * tok_loss + anchor_w * anc_loss
    assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(P.parameters(), 1.0)
    opt.step()
    acc["eps"] += float(eps_loss); acc["tok"] += float(tok_loss); acc["anc"] += float(anc_loss); acc["n"] += 1
    if (step + 1) % 100 == 0 or s + 16 >= total:
        rec = {k: acc[k] / acc["n"] for k in ("eps", "tok", "anc")} | dict(step=step + 1, seconds=time.time() - t0)
        log.append(rec)
        print(f"ipft step {step + 1}/{(total + 15) // 16} eps {rec['eps']:.4f} tok {rec['tok']:.4f} anchor {rec['anc']:.4f} {rec['seconds']:.0f}s "
              f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        acc = dict(eps=0.0, tok=0.0, anc=0.0, n=0)
    if (step + 1) % 3000 == 0:
        torch.save({k: v.detach().cpu() for k, v in P.state_dict().items()}, out / f"proj_step{step + 1}.pt")
torch.save({k: v.detach().cpu() for k, v in P.state_dict().items()}, out / "proj.pt")
write_json(out / "log.json", dict(tag=tag, head=hname, passes=passes, lr=lr, tok_w=tok_w, anchor_w=anchor_w, updates=step + 1, log=log))
print(f"IPFT_DONE {oname}", flush=True)
