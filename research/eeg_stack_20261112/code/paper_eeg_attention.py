"""Train image attention on real EEG conditions from concept-disjoint encoders."""

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_oof_conditions import HELD_OUT, SEED, SOURCES, fitted_native, native_features, registered_rows, require, study_root


RANK, STEPS = 8, 1024
MATCHED_MODES = ("seen", "out-of-fold", "out-of-fold-wrong")
FIT_MODES = ("seen", "out-of-fold", "out-of-fold-rollout")
RENDER_MODES = ("native", *MATCHED_MODES, "out-of-fold-rollout", "out-of-fold-rollout-wrong")
TIMESTEPS = (999, 749, 499, 249)


class ImageProjectionUpdate(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.down = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.enabled = True

    def forward(self, x):
        original = self.base(x)
        if not self.enabled:
            return original
        return original + F.linear(F.linear(x.float(), self.down), self.up).to(original.dtype)

    def export(self):
        return {"up": self.up.detach().cpu().clone(), "down": self.down.detach().cpu().clone()}


@contextmanager
def without_updates(updates):
    prior = [module.enabled for module in updates.values()]
    try:
        for module in updates.values():
            module.enabled = False
        yield
    finally:
        for module, enabled in zip(updates.values(), prior, strict=True):
            module.enabled = enabled


def image_layers(unet):
    layers = {}
    for name, process in sorted(unet.attn_processors.items()):
        if hasattr(process, "to_k_ip"):
            require(isinstance(process.to_k_ip, nn.Linear) and isinstance(process.to_v_ip, nn.Linear),
                    "Unexpected image-attention projections")
            for attr in ("to_k_ip", "to_v_ip"):
                layers[f"{name}.{attr}"] = process, attr
    require(len(layers) >= 2, "The U-Net does not have the pinned image attention mechanism")
    return layers


def install_updates(unet, rank):
    updates = {}
    for name, (process, attr) in image_layers(unet).items():
        update = ImageProjectionUpdate(getattr(process, attr), rank)
        setattr(process, attr, update)
        updates[name] = update
    return updates


@contextmanager
def merge_image_updates(layers, learned):
    require(set(layers) == set(learned), "Checkpoint image projections differ from the target U-Net")
    original = {}
    try:
        with torch.no_grad():
            for name, layer in layers.items():
                state = learned[name]
                up, down = (state[key].to(device=layer.weight.device, dtype=torch.float32) for key in ("up", "down"))
                require(up.ndim == down.ndim == 2 and (up.shape[0], down.shape[1]) == layer.weight.shape
                        and up.shape[1] == down.shape[0], "Image-attention weight shape differs")
                require(torch.isfinite(up).all() and torch.isfinite(down).all(), "Nonfinite image-attention update")
                original[name] = layer.weight.detach().clone()
                layer.weight.copy_((layer.weight.float() + up @ down).to(layer.weight.dtype))
        yield
    finally:
        with torch.no_grad():
            for name, weight in original.items():
                layers[name].weight.copy_(weight)


def response_loss(predicted, teacher, neutral):
    # Scale error by the effect of supplying the true photo to the frozen U-Net.
    reference_power = (teacher - neutral).float().square().flatten(1).mean(1)
    floor = (.25 * reference_power.median()).clamp_min(1e-6)
    error = (predicted - teacher).float().square().flatten(1).mean(1)
    return (error / reference_power.clamp_min(floor)).mean()


def sampling_plan(n):
    rng = np.random.default_rng(SEED)
    # Every person supplies equally many distinct observed photographs.
    photos = np.stack([rng.permutation(n)[:STEPS] for _ in SOURCES], 1)
    steps = np.tile(TIMESTEPS, STEPS // len(TIMESTEPS))
    rng.shuffle(steps)
    return photos, steps


def predict_noise(unet, noisy, t, image_tokens, prompt):
    text, pooled, dims = prompt
    length = len(noisy)
    steps = torch.full((length,), t, dtype=torch.long, device=noisy.device)
    return unet(noisy, steps, encoder_hidden_states=text.expand(length, -1, -1),
                added_cond_kwargs={"text_embeds": pooled.expand(length, -1), "time_ids": dims.expand(length, -1)},
                cross_attention_kwargs={"ip_hidden_states": image_tokens}).sample.float()


@torch.no_grad()
def rollout_conditioning_input(scheduler, epsilon, timestep, predict, rng):
    """Return the frozen generator's denoiser input at one step of a free rollout."""
    scheduler.set_timesteps(4, device=epsilon.device)
    latent = epsilon * float(scheduler.init_noise_sigma)
    for t in scheduler.timesteps:
        model_input = scheduler.scale_model_input(latent, t)
        time = int(t)
        if time == timestep:
            return model_input
        predicted = predict(model_input, time)
        latent = scheduler.step(predicted, t, latent, generator=rng, return_dict=False)[0]
    raise ValueError("Requested timestep is absent from the four-step rollout")


def train(mode):
    from common import PREP
    from paper_render import generator_inputs
    from paper_render_cache import PinnedFiles, atomic_file, save_json
    from paper_training import generator

    root = study_root()
    free_rollout = mode == "out-of-fold-rollout"
    rows, _, folds = registered_rows(PREP)
    plan, timetable = sampling_plan(len(rows))
    latent_path = PREP / "lat512_train.npy"
    real_path = PREP / "F_train.npy"
    source_paths = [root / f"conditions-sub-{subject:02d}.pt" for subject in SOURCES]
    pins = PinnedFiles()
    split_hash = pins.pin(PREP / "paper_split.json")
    generator_hashes = generator_inputs(pins)
    proof = dict(train_rows_sha256=hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
                 split=split_hash, generator=generator_hashes, latents=pins.pin(latent_path), real=pins.pin(real_path),
                 conditions=[pins.pin(path) for path in source_paths], code=pins.pin(Path(__file__)),
                 schedule=hashlib.sha256(plan.astype("<i8").tobytes() + timetable.astype("<i8").tobytes()).hexdigest())
    dest = root / f"attention-{mode}.pt"
    if dest.is_file():
        cache = torch.load(dest, map_location="cpu", weights_only=True)
        require(cache["proof"] == proof and cache["mode"] == mode, "Existing attention weights have different sources")
        return
    bank = []
    summaries = {}
    for subject, path in zip(SOURCES, source_paths, strict=True):
        record = torch.load(path, map_location="cpu", weights_only=True)
        require(record["sources"]["subject"] == subject and record["sources"]["split"] == split_hash
                and record["sources"]["images"] == proof["real"]
                and np.array_equal(record["rows"].numpy(), rows) and np.array_equal(record["folds"].numpy(), folds)
                and np.array_equal(record["seen_source_fold"].numpy(), folds % 4 + 1),
                "Condition cache disagrees with the registered EEG and photograph rows")
        condition_key = "out_of_fold" if free_rollout else mode.replace("-", "_")
        bank.append(record[condition_key])
        summaries[str(subject)] = record["stats"]
    condition = torch.stack(bank, dim=1).cuda().float()
    require(tuple(condition.shape) == (len(rows), len(SOURCES), 1024) and torch.isfinite(condition).all(),
            "Invalid condition bank")
    del bank, record
    latents = np.load(latent_path, mmap_mode="r", allow_pickle=False)
    require(latents.shape == (16540, 4, 64, 64), "Incorrect clean VAE latent axes")
    real = torch.from_numpy(np.load(real_path, mmap_mode="r", allow_pickle=False)[rows].copy()).cuda().float()
    dtype = torch.bfloat16
    pipe, adapter, projection, prompt, pooled, dims = generator(dtype, training=True)
    pipe.vae = None
    require(tuple(pipe.scheduler.timesteps.tolist()) == TIMESTEPS
            and pipe.scheduler.config.prediction_type == "epsilon", "Expected the pinned four denoising steps")
    torch.manual_seed(SEED)
    updates = install_updates(pipe.unet, RANK)
    params = [p for update in updates.values() for p in (update.up, update.down)]
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=.05)
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-5)
    rng = torch.Generator(device="cuda").manual_seed(SEED + 1)
    ancestral_rng = torch.Generator(device="cuda").manual_seed(SEED + 2)
    photo_count = len(SOURCES)
    axes = torch.arange(photo_count, device="cuda")
    text_inputs = prompt, pooled, dims
    with torch.no_grad():
        mean_token = projection(adapter(real.mean(0, keepdim=True).to(dtype))).expand(photo_count, -1, -1)
    metrics = []
    for step, (chosen, time) in enumerate(zip(plan, timetable, strict=True), start=1):
        time = int(time)
        indices = torch.from_numpy(chosen).to("cuda")
        noise = torch.randn((photo_count, *latents.shape[1:]), device="cuda", generator=rng)
        with torch.no_grad():
            eeg = projection(adapter(condition[indices, axes].to(dtype)))
            photo = projection(adapter(real[indices].to(dtype)))
            with without_updates(updates):
                if free_rollout:
                    def denoise(value, t):
                        return predict_noise(pipe.unet, value.to(dtype), t, eeg, text_inputs)

                    z_t = rollout_conditioning_input(pipe.scheduler, noise, time, denoise, ancestral_rng).to(dtype)
                else:
                    clean = torch.from_numpy(latents[rows[chosen]].copy()).cuda().float()
                    alpha = float(pipe.scheduler.alphas_cumprod[time])
                    z_t = (math.sqrt(alpha) * clean + math.sqrt(1 - alpha) * noise).to(dtype)
                teacher, neutral = predict_noise(pipe.unet, torch.cat((z_t, z_t)), time,
                                                 torch.cat((photo, mean_token)), text_inputs).chunk(2)
        pred = predict_noise(pipe.unet, z_t, time, eeg, text_inputs)
        loss = response_loss(pred, teacher, neutral)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        decay.step()
        metrics.append(float(loss.detach()))
        if step == 1 or step % 32 == 0:
            print("attention", mode, step, f"{np.mean(metrics[-32:]):.6f}", flush=True)
    pins.verify()
    pack = dict(mode=mode, proof=proof, training_subjects=list(SOURCES), rank=RANK, steps=STEPS,
                image_projection_updates={name: layer.export() for name, layer in updates.items()},
                timesteps=list(TIMESTEPS), seed=SEED, training_samples=STEPS * len(SOURCES),
                distinct_photos=int(np.unique(plan).size), training_loss=metrics,
                state_source="native free rollout" if free_rollout else "forward-noised original-photo latent")
    atomic_file(dest, lambda stream: torch.save(pack, stream))
    save_json(dest.with_suffix(".json"), dict(mode=mode, steps=STEPS, training_samples=pack["training_samples"], state_source=pack["state_source"],
              distinct_photos=pack["distinct_photos"], condition_statistics=summaries,
              trainable_count=sum(param.numel() for param in params), source_hashes=proof))


@torch.inference_mode()
def render(subject, panel, modes):
    from common import PREP, ROOT, TURBO
    from diffusers import AutoencoderKL
    from paper_render import draw_images, generator_inputs, physical_eeg, query_seed, runtime
    from paper_render_cache import PinnedFiles, render_transaction
    from paper_score import score
    from paper_training import generator

    root = study_root()
    files = PinnedFiles()
    split_path = PREP / "paper_split.json"
    split_sha = files.pin(split_path)
    split = json.loads(split_path.read_bytes())
    rows = split[f"{panel}_rows"]
    phase = "test" if panel == "official" else "train"
    raw, physical = physical_eeg(subject, phase, files)
    require(len(rows) == len(set(rows)) == (200 if panel == "official" else 500) and min(rows) >= 0 and max(rows) < len(raw),
            "Unexpected query rows")
    require(panel != "official" or rows == list(range(len(raw))), "Expected the official stimulus order")
    if panel == "fresh":
        require(not np.intersect1d(rows, split["train_rows"]).size, "Fresh photos cannot have trained the model")
    repeats = [0, 1, 2, 3]
    model, native = fitted_native(subject, torch.device("cuda"))
    native_sha = files.pin(native)
    means = torch.from_numpy(raw[rows][:, repeats].copy()).cuda().float().mean(1)
    features = native_features(model, means).cuda()
    del model, means
    base_assets = generator_inputs(files)
    pipe, adapter, project, prompt, pooled, dims = generator(torch.float16)
    pipe.vae = None
    vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).cuda().eval()
    vae.enable_slicing()
    tokens = torch.cat([project(adapter(x.half())).cpu() for x in features.split(20)])
    layers = {name: getattr(process, attr) for name, (process, attr) in image_layers(pipe.unet).items()}
    seeds = [query_seed(panel, i) for i in range(len(rows))]
    folder_list = []
    for arm in modes:
        mode = arm.removesuffix("-wrong")
        checkpoint_sha, pack = None, None
        if mode != "native":
            path = root / f"attention-{mode}.pt"
            checkpoint_sha = files.pin(path)
            pack = torch.load(path, map_location="cpu", weights_only=True)
            require(pack["proof"]["generator"] == base_assets and pack["proof"]["split"] == split_sha and pack["mode"] == mode
                    and set(pack["training_subjects"]).isdisjoint(HELD_OUT), "Adapter and evaluation asset registrations differ")
        output = root / "images" / arm / f"{panel}-r4" / f"sub-{subject:02d}"
        donors = np.arange(len(rows))
        if arm.endswith("-wrong"):
            # Official rows are distinct concepts. Fresh rows are sorted in blocks of ten per concept.
            donors = np.roll(donors, len(rows) // 2)
        request = dict(render_cache_version=2, subject=subject, arm=arm, training="four-source-shared-adapter",
                       n=len(rows), budget=f"{panel}-r4", row=rows, repeats=[repeats], seeds=seeds,
                       eeg_donor=donors.tolist(), prompt="", draws=1, unet_calls_per_draw=4,
                       physical_eeg=physical, split_sha256=split_sha, native_checkpoint_sha256=native_sha,
                       attention_checkpoint_sha256=checkpoint_sha, render_source_sha256=files.pin(Path(__file__)),
                       generator_files_sha256=base_assets, updates_merged=pack is not None, runtime=runtime())

        def generate(stage, starts, complete):
            updates = nullcontext() if pack is None else merge_image_updates(layers, pack["image_projection_updates"])
            with updates:
                draw_images(stage, pipe, vae, (prompt, pooled, dims), tokens[donors], seeds,
                            batch_starts=starts, batch_complete=complete)

        render_transaction(output, request, generate, files.verify)
        folder_list.append(output)
        print("render complete", subject, panel, arm, flush=True)
    del pipe, adapter, project, vae, layers
    torch.cuda.empty_cache()
    for folder in folder_list:
        if not (folder / "scores.json").exists():
            score(folder, panel, Path(os.environ["ARCHIVE"]), folder)
    return folder_list


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("mode", choices=FIT_MODES)
    draw = commands.add_parser("render")
    draw.add_argument("--subject", type=int, choices=HELD_OUT, required=True)
    draw.add_argument("--panel", choices=("official", "fresh"), default="official")
    draw.add_argument("--arms", choices=RENDER_MODES, nargs="+", default=MATCHED_MODES)
    args = cli.parse_args()
    if args.command == "fit":
        train(args.mode)
    else:
        render(args.subject, args.panel, args.arms)
