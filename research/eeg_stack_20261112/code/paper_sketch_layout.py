"""Learn a low-resolution visual start from EEG, with supervision on decoded RGB."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, atomic_file, render_transaction, require, save_json


EPOCHS = 12
FIT_BATCH = 384
FINETUNE_STEPS = 384
FINETUNE_BATCH = 8
SEED = 20261126
IDENTITY_WEIGHT = .15
ARMS = ("d2", "sketch-mse", "sketch-rgb", "true-sketch")
MODES = ARMS[1:3]
PROJECTED_ARMS = ("sketch-rgb-projected", "true-sketch-projected", "sketch-rgb-no-remix", "true-sketch-no-remix")
PROJECTION = (0., .35, .65, 1.)


class SketchReadout(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_scale", scale)
        self.stems = nn.ModuleList(nn.Linear(mean.shape[1], 512) for _ in range(len(mean)))
        self.block = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 512), nn.GELU(), nn.Dropout(.25),
                                   nn.Linear(512, 512))
        self.head = nn.Sequential(nn.LayerNorm(512), nn.Dropout(.25), nn.Linear(512, 4 * 16 * 16))

    def forward(self, eeg, subject):
        values = (eeg.float() - self.input_mean[subject]) / self.input_scale[subject]
        hidden = self.stems[subject](values)
        return self.head(hidden + self.block(hidden)).float().view(-1, 4, 16, 16)


def eeg_bins(raw, rows, repeat_groups):
    result = []
    for start in range(0, len(rows), 256):
        x = torch.from_numpy(np.array(raw[rows[start:start + 256]], copy=True)).cuda().float()
        group_mean = torch.stack([x[:, group].mean(1) for group in repeat_groups]).mean(0)
        binned = group_mean.unflatten(-1, (50, 5)).mean(-1).flatten(1)
        result.append(binned.half())
    return torch.cat(result)


def decode(vae, sketch):
    latent = sketch.float() / vae.config.scaling_factor
    return (vae.decode(latent).sample.float() * .5 + .5).clamp(0., 1.)


def projection_delta(clean, sketch, gain):
    # Repeating each correction within its cell leaves the finer residual intact.
    coarse = F.avg_pool2d(clean, 4)
    return gain * F.interpolate(sketch - coarse, scale_factor=4, mode="nearest")


def decoder():
    from common import TURBO
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16",
                                         torch_dtype=torch.float32).cuda().eval().requires_grad_(False)


def load_latents(path, rows):
    from common import EMB_KEYS

    saved = torch.load(path, map_location="cpu", weights_only=True)[EMB_KEYS[2]]
    require(len(saved) == 16540 and saved[0].numel() == 1024, "Expected registered 128-pixel VAE latents")
    return saved[rows].reshape(len(rows), 4, 16, 16).cuda().float()


def training_data(files):
    from common import PREP
    from paper_oof_conditions import registered_rows
    from paper_render import physical_eeg

    rows, classes, _ = registered_rows(PREP)
    split = files.pin(PREP / "paper_split.json")
    source = PREP / "emb_train.pt"
    latent_sha = files.pin(source)
    raw_latents = load_latents(source, rows)
    center = raw_latents.mean(0)
    scale = raw_latents.std((0, 2, 3), keepdim=True).clamp_min(1e-4)
    waves, receipts = [], []
    for person in range(1, 11):
        eeg, proof = physical_eeg(person, "train", files)
        waves.append(eeg_bins(eeg, rows, [[0, 1, 2, 3]]))
        receipts.append(proof)
        print("Sketch physical training EEG", person, len(rows), flush=True)
    return rows, classes, torch.stack(waves), raw_latents, center, scale, dict(
        split_sha256=split, latent_sha256=latent_sha, training_eeg=receipts, observations=len(rows) * len(waves),
    )


def fit_base(model, eeg, target):
    optim = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=.04)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, EPOCHS, eta_min=5e-5)
    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(EPOCHS):
        batch_loss = []
        for person in rng.permutation(len(eeg)).tolist():
            rows = torch.from_numpy(rng.permutation(len(target))).cuda()
            for batch in rows.split(FIT_BATCH):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(eeg[person, batch], person)
                    loss = F.smooth_l1_loss(pred, target[batch])
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                batch_loss.append(loss.detach())
        history.append(float(torch.stack(batch_loss).mean()))
        require(np.isfinite(history[-1]), "Nonfinite direct sketch fit")
        schedule.step()
        print("Sketch regression epoch", epoch + 1, "/", EPOCHS, history[-1], flush=True)
    return history


def finetune(model, eeg, target, rgb_loss=None):
    torch.manual_seed(SEED + 1)
    rng = np.random.default_rng(SEED + 1)
    # Both continuations change only the same last layer.
    model.eval().requires_grad_(False)
    model.head[-1].requires_grad_(True)
    optim = torch.optim.AdamW(model.head[-1].parameters(), lr=2e-4, weight_decay=0.)
    history = []
    for step in range(FINETUNE_STEPS):
        person = step % len(eeg)
        batch = torch.from_numpy(rng.choice(len(target), FINETUNE_BATCH, replace=False)).cuda()
        pred = model(eeg[person, batch], person)
        latent_loss = F.smooth_l1_loss(pred, target[batch])
        semantic_loss = pred.new_zeros(()) if rgb_loss is None else rgb_loss(pred, batch)
        loss = latent_loss + IDENTITY_WEIGHT * semantic_loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head[-1].parameters(), 2., error_if_nonfinite=True)
        optim.step()
        history.append([float(latent_loss.detach()), float(semantic_loss.detach())])
        if (step + 1) % 48 == 0:
            print("Sketch refinement", "mse" if rgb_loss is None else "rgb", step + 1,
                  np.mean(history[-48:], axis=0).tolist(), flush=True)
    return history


def visual_loss(rows, classes, center, scale, proof, bank_path, files):
    from common import ROOT
    from paper_dino import MODEL_FILES
    from paper_identity_bank import object_logits, visual_model
    from paper_rgb_object import dino_cls
    from paper_render import generator_inputs

    model_dir = ROOT / "models/paper-dinov2-small"
    assets = {name: files.pin(model_dir / name) for name in MODEL_FILES}
    require(assets == MODEL_FILES, "Unexpected sketch teacher weights")
    bank_sha = files.pin(bank_path)
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    require(bank["proof"]["dino"] == assets and bank["proof"]["split"] == proof["split_sha256"]
            and np.array_equal(bank["rows"].numpy(), rows)
            and np.array_equal(bank["concepts"][bank["labels"]].numpy(), classes),
            "DINO bank and EEG training rows differ")
    features, labels, sums = (bank[k].cuda() for k in ("image_features", "labels", "sums"))
    gen = generator_inputs(files)
    vae, dino = decoder(), visual_model(model_dir)

    def loss(pred, indices):
        rgb = decode(vae, pred * scale + center)
        image = dino_cls(dino, rgb)
        own = features[indices]
        scores = object_logits(image, own, labels[indices], sums, "identity")
        identity = F.cross_entropy(scores / .10, labels[indices])
        photo = 1. - (image * own).sum(-1).mean()
        return identity + .5 * photo

    return loss, {"dino_bank_sha256": bank_sha, "dino_model": assets, "generator_files_sha256": gen}


def fit(output, bank_path):
    require(not any((output / f"{mode}.pt").exists() for mode in MODES), "Each planned fit uses a fresh output directory")
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    files = PinnedFiles()
    rows, classes, eeg, latent, center, scale, proof = training_data(files)
    norm = (latent - center) / scale
    x = eeg.float()
    model = SketchReadout(x.mean(1), x.std(1).clamp_min(1e-5)).cuda()
    del x
    history = fit_base(model, eeg, norm)
    base = {key: value.detach().clone() for key, value in model.state_dict().items()}
    visual, assets = visual_loss(rows, classes, center, scale, proof, bank_path, files)
    input_files = ("paper_sketch_layout.py", "paper_rgb_object.py", "paper_identity_bank.py", "paper_oof_conditions.py")
    proof.update(assets, code={name: files.pin(Path(__file__).with_name(name)) for name in input_files}, seed=SEED,
                 epochs=EPOCHS, finetune_steps=FINETUNE_STEPS, finetune_batch=FINETUNE_BATCH, identity_weight=IDENTITY_WEIGHT)
    output.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        model.load_state_dict(base, strict=True)
        fine = finetune(model, eeg, norm, visual if mode == "sketch-rgb" else None)
        files.verify()
        path = output / f"{mode}.pt"
        data = {"model": model.state_dict(), "latent_center": center, "latent_scale": scale,
                "mode": mode, "proof": proof, "base_history": history, "finetune_history": fine}
        atomic_file(path, lambda stream: torch.save(data, stream))
        report = {k: data[k] for k in ("mode", "proof", "base_history", "finetune_history")}
        save_json(path.with_suffix(".json"), dict(report, checkpoint_sha256=files.pin(path)))


def load_readout(path, mode, subject, train_proof, split_sha, trainer_source, files):
    digest = files.pin(path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    proof = data["proof"]
    require(data["mode"] == mode and proof["split_sha256"] == split_sha
            and proof["training_eeg"][subject - 1] == train_proof
            and proof["epochs"] == EPOCHS and proof["finetune_steps"] == FINETUNE_STEPS
            and proof["code"][Path(__file__).name] == files.pin(trainer_source),
            "The loaded sketch fit differs from the planned protocol")
    weight = data["model"]
    model = SketchReadout(weight["input_mean"], weight["input_scale"])
    model.load_state_dict(weight, strict=True)
    return model.cuda().eval(), data["latent_center"].cuda(), data["latent_scale"].cuda(), digest


@torch.inference_mode()
def render(subject, panel, arms, output, fits, trainer_source):
    from common import PREP, RUNS
    from paper_d2_object_swap import d2_models
    from paper_photo_top1 import ordinary_readout, project_condition, render_inputs
    from paper_render import (draw_images, generator, generator_inputs, layout, physical_eeg, query_seed, render_code,
                              runtime, tensor_fingerprint)
    from paper_uniform_baselines import reference_layout

    require(RUNS.name == "shared", "Use the registered shared D2 model")
    require(not any(arm.startswith("true-sketch") for arm in arms) or panel == "dev",
            "Reference sketch is a development-only diagnostic")
    files = PinnedFiles()
    raw, rows, _, budget, groups, _, split_sha, dev_sha, eeg = render_inputs(subject, panel, files)
    train_proof = eeg if panel != "official" else physical_eeg(subject, "train", files)[1]
    ridge, layout_source = reference_layout(subject, split_sha, files)
    starts = layout(ridge, raw, groups)
    assets = generator_inputs(files)
    pipe, adapter, projection, prompt, pooled, dims = generator(torch.float16)
    pipe.vae = None
    vae = decoder()
    vae.enable_slicing()
    enc, residual, cone, d2 = d2_models(files)
    tokens = []
    for begin in range(0, len(raw), 20):
        data = torch.from_numpy(np.array(raw[begin:begin + 20], copy=True)).cuda().float()
        features, hidden = ordinary_readout(enc, data, subject, groups)
        tokens.append(project_condition(features, hidden, residual, cone, adapter, projection))
    tokens = torch.cat(tokens)
    prompts = prompt, pooled, dims
    seeds = [query_seed(panel, j) for j in range(len(rows))]
    code = dict(render_code(files, "d2"), sketch=files.pin(Path(__file__)))
    measured = eeg_bins(raw, np.arange(len(raw)), groups)
    proof = dict(render_cache_version=2, n=len(rows), row=rows, subject=subject, training="shared",
                 budget=budget, repeats=groups, prompt="", draws=1, unet_calls_per_draw=4, seeds=seeds,
                 eeg_donor=list(range(len(rows))), split_sha256=split_sha, development_split_sha256=dev_sha,
                 physical_eeg=eeg, d2_sources=d2, generator_files_sha256=assets, render_code_sha256=code,
                 token_sha256=tensor_fingerprint(tokens), trainer_source_sha256=files.pin(trainer_source), runtime=runtime())
    for arm in arms:
        sketch_hash = None
        latent = starts
        remix = not arm.endswith("-no-remix")
        base_arm = arm.removesuffix("-projected").removesuffix("-no-remix")
        projected = arm in PROJECTED_ARMS
        if base_arm in MODES:
            reader, mean, scale, sketch_hash = load_readout(fits / f"{base_arm}.pt", base_arm, subject,
                                                           train_proof, split_sha, trainer_source, files)
            sketch = reader(measured, subject - 1) * scale + mean
        elif base_arm == "true-sketch":
            truth = PREP / "emb_train.pt"
            sketch_hash = files.pin(truth)
            sketch = load_latents(truth, np.asarray(rows))
        if arm != "d2":
            require(bool(torch.isfinite(sketch).all()), "Sketch has nonfinite latent values")
            latent = F.interpolate(sketch, (64, 64), mode="bilinear", align_corners=False).cpu()
        request = dict(proof, arm=arm, layout_source=layout_source if arm == "d2" else None,
                       layout_sha256=tensor_fingerprint(latent),
                       sketch_sha256=sketch_hash, diagnostic_only=base_arm == "true-sketch", layout_strength=.375,
                       clean_projection=list(PROJECTION) if projected else None)
        if not remix:
            request["remix_initial_layout"] = False

        def update(batch, step, clean):
            gain = PROJECTION[step]
            return projection_delta(clean, sketch[batch], gain) if gain else None

        folder = output / "images" / "shared" / arm / budget / f"sub-{subject:02d}"

        def generate(stage, batches, done):
            draw_images(stage, pipe, vae, prompts, tokens, seeds, latent, batch_starts=batches, batch_complete=done,
                        clean_update=update if projected else None,
                        layout_low_keep=None if projected else (8, .11, .02), remix_initial_layout=remix)

        render_transaction(folder, request, generate, files.verify)
        print("Sketch render", subject, arm, len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    training = sub.add_parser("fit")
    training.add_argument("--dino-bank", type=Path, required=True)
    sampling = sub.add_parser("render")
    sampling.add_argument("--subject", type=int, choices=range(1, 11), required=True)
    sampling.add_argument("--panel", choices=("dev", "fresh", "official"), default="dev")
    sampling.add_argument("--arms", nargs="+", choices=ARMS + PROJECTED_ARMS, required=True)
    sampling.add_argument("--fits", type=Path, required=True)
    sampling.add_argument("--trainer-source", type=Path, default=Path(__file__),
                          help="Exact archived training source when replaying a fit after render-only edits")
    for command in (training, sampling):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(bool(os.environ.get("SLURM_JOB_ID")), "Use the Slurm scheduler for EEG and model workloads")
    if args.action == "fit":
        fit(args.output, args.dino_bank)
    else:
        render(args.subject, args.panel, args.arms, args.output, args.fits, args.trainer_source)
