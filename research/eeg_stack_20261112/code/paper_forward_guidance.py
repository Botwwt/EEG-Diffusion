"""Guide one native four-step draw by how well its predicted EEG matches measured EEG."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from paper_render_cache import PinnedFiles, render_transaction, require


ARMS = ("forward-noise", "forward-unweighted", "forward-wrong")
GUIDANCE_STEP = 1
UPDATE_RMS = .15
REPEATS = [0, 1, 2, 3]
POSTERIOR_PENALTY = .1


def forward_weights(visual, eeg, half_difference):
    visual_center = visual.mean(0)
    visual_scale = visual.std(0).clamp_min(1e-5)
    eeg_center = eeg.mean(0)
    eeg_scale = eeg.std(0).clamp_min(1e-5)
    predictors = (visual - visual_center) / visual_scale
    response = (eeg - eeg_center) / eeg_scale
    gram = predictors.T @ predictors / len(predictors)
    gram.diagonal().add_(1.)
    plain = torch.linalg.solve(gram, predictors.T @ response / len(predictors))
    noise = half_difference / eeg_scale
    covariance = noise.T @ noise / len(noise)
    covariance.diagonal().add_(.3 * covariance.diagonal().mean().clamp_min(1e-5))
    factor = torch.linalg.cholesky(covariance)
    weighted = torch.linalg.solve_triangular(factor, plain.T, upper=False).T
    return dict(plain=plain, weighted=weighted, eeg_center=eeg_center, eeg_scale=eeg_scale,
                visual_center=visual_center, visual_scale=visual_scale, noise_factor=factor)


def bounded_descent(clean, loss, magnitude):
    with torch.enable_grad():
        preview = clean.detach().requires_grad_(True)
        (gradient,) = torch.autograd.grad(loss(preview), preview)
    rms = gradient.flatten(1).square().mean(1).sqrt().view(-1, 1, 1, 1)
    require(torch.isfinite(rms).all(), "Non-finite EEG likelihood gradient")
    return -magnitude * gradient / rms.clamp_min(1e-6)


def update_condition(condition, center, bridge, model, eeg):
    original = condition - center
    prior = original @ bridge
    gram = model @ model.T
    gram.diagonal().add_(POSTERIOR_PENALTY)
    gain = torch.linalg.solve(gram, model)
    correction = (eeg - prior @ model) @ gain.T
    bridge_gram = bridge.T @ bridge
    bridge_gram.diagonal().add_(.1 * bridge_gram.diagonal().mean().clamp_min(1e-8))
    lifted = correction @ torch.linalg.solve(bridge_gram, bridge.T)
    limit = (.5 * original.norm(dim=1, keepdim=True)).clamp_min(1e-6)
    return condition + lifted / (lifted.norm(dim=1, keepdim=True) / limit).clamp_min(1.)


def binned_eeg(raw, rows, repeats):
    views = []
    for selected in np.array_split(np.asarray(rows), (len(rows) + 127) // 128):
        data = torch.from_numpy(raw[selected][:, repeats, :, :200].copy()).to(device="cuda", dtype=torch.float32)
        views.append(data.reshape(-1, len(repeats), 17, 40, 5).mean(-1).flatten(2))
    return torch.cat(views)


@torch.no_grad()
def run(subject, panel, method, repeat_budget=4, variants=ARMS):
    from common import PREP, ROOT
    from paper_dino import MODEL_FILES
    from paper_identity_bank import bank_root, visual_model
    from paper_identity_render import load_generator
    from paper_oof_conditions import registered_rows, study_root
    from paper_render import condition, donor_rows, draw_images, generator_inputs, groups, physical_eeg, query_seed
    from paper_rgb_object import dino_cls
    from paper_score import score

    require(repeat_budget == 4 or (repeat_budget == 80 and panel == "official" and method == "condition"),
            "The R80 comparison uses the official 80-repeat acquisition and the conditioning update")
    budget = f"{panel}-r{repeat_budget}"
    groups_for_query = groups(budget)
    query_repeats = np.asarray(groups_for_query).ravel().tolist()
    pins = PinnedFiles()
    split_path = PREP / "paper_split.json"
    split_sha = pins.pin(split_path)
    split = json.loads(split_path.read_bytes())
    development_sha = None
    if panel == "dev":
        design = PREP / "paper_development.json"
        development_sha = pins.pin(design)
        plan = json.loads(design.read_bytes())
        require(subject in plan["subjects"], "Not a registered development participant")
        rows = plan["dev_rows"]
    else:
        rows = split[f"{panel}_rows"]
    train_rows, training_classes, _ = registered_rows(PREP)
    phase = "test" if panel == "official" else "train"
    require(phase == "test" or not np.intersect1d(train_rows, rows).size, "Query stimuli entered the forward fit")
    train, physical_train = physical_eeg(subject, "train", pins)
    raw, physical = (physical_eeg(subject, "test", pins) if phase == "test" else (train, physical_train))
    require(len(rows) == len(set(rows)) == (500 if panel == "fresh" else 200), "Unexpected guidance panel")

    photo_path = bank_root() / "training-photo-dino.pt"
    photo_sha = pins.pin(photo_path)
    bank = torch.load(photo_path, map_location="cpu", weights_only=True)
    require(bank["proof"]["split"] == split_sha and bank["proof"]["dino"] == MODEL_FILES
            and np.array_equal(bank["rows"].numpy(), train_rows)
            and np.array_equal(bank["concepts"].numpy()[bank["labels"].numpy()], training_classes),
            "Visual EEG encoder training pairs disagree")
    trials = binned_eeg(train, train_rows, REPEATS)
    mean = trials.mean(1)
    gap = (trials[:, :2].mean(1) - trials[:, 2:].mean(1)) * .5
    weights = forward_weights(bank["image_features"].cuda(), mean, gap)
    del trials, mean, gap
    observed = (binned_eeg(raw, rows, query_repeats).mean(1) - weights["eeg_center"]) / weights["eeg_scale"]
    observed_white = torch.linalg.solve_triangular(weights["noise_factor"], observed.T, upper=False).T

    vision_path = ROOT / "models/paper-dinov2-small"
    vision_hashes = {key: pins.pin(vision_path / key) for key in MODEL_FILES}
    require(vision_hashes == MODEL_FILES, "Unexpected visual encoding model weights")
    dino = visual_model(vision_path) if method == "latent" else None
    generator_assets = generator_inputs(pins)
    pipe, adapter, projection, prompts, vae = load_generator(torch.float16, False)
    if method == "latent":
        tokens, checkpoint = condition("native-paper", subject, raw[rows], groups_for_query, adapter, projection, files=pins)
        fused_sha = None
    else:
        from paper_object_margin_encoder import visual_transfer

        fused_path = PREP / "F_train.npy"
        fused_sha = pins.pin(fused_path)
        fused = torch.from_numpy(np.load(fused_path, mmap_mode="r", allow_pickle=False)[train_rows].copy()).cuda().float()
        center, _, visual_bridge = visual_transfer(fused, bank["image_features"].cuda())
        bridge = visual_bridge / weights["visual_scale"]
        prior, checkpoint = condition("native-paper", subject, raw[rows], groups_for_query, features_only=True, files=pins)
        prior = prior.cuda().float()
    seeds = [query_seed(panel, i) for i in range(len(rows))]
    if panel == "official":
        donors = np.roll(np.arange(len(rows)), len(rows) // 2)
    else:
        concepts = PREP / "train_concept.npy"
        pins.pin(concepts)
        donors = donor_rows("token-reliable-wrong", panel, np.load(concepts, allow_pickle=False)[rows])
    root = study_root() / ("forward-guidance" if method == "latent" else "forward-condition")
    inputs = dict(physical_eeg=physical, forward_training_eeg=physical_train, split_sha256=split_sha,
                  development_split_sha256=development_sha, vision_sha256=vision_hashes, photo_bank_sha256=photo_sha,
                  generator_files_sha256=generator_assets, native_checkpoint_sha256=checkpoint, fused_training_sha256=fused_sha,
                  code_sha256={file: pins.pin(Path(__file__).with_name(file)) for file in
                               (Path(__file__).name, "paper_render.py", "exp6_draft_intra4.py")})
    folders = []
    for variant in variants:
        weighted = "-unweighted" not in variant
        source = donors if variant.endswith("-wrong") else np.arange(len(rows))
        eeg = (observed_white if weighted else observed)[source]
        model = weights["weighted" if weighted else "plain"]
        arm = variant if method == "latent" else variant.replace("forward-", "condition-")
        if method == "condition":
            adjusted = update_condition(prior, center, bridge, model, eeg)
            tokens = torch.cat([projection(adapter(batch.half())).cpu() for batch in adjusted.split(20)])
        dest = root / arm / budget / f"sub-{subject:02d}"
        request = dict(render_cache_version=2, arm=arm, subject=subject, n=len(rows), row=rows, seeds=seeds,
                       prompt="", repeats=groups_for_query, draws=1, unet_calls_per_draw=4, guidance_eeg_donor=source.tolist(),
                       native_eeg_donor=list(range(len(rows))), method=method,
                       guidance_step=GUIDANCE_STEP if method == "latent" else None,
                       update_rms=UPDATE_RMS if method == "latent" else None,
                       posterior_penalty=POSTERIOR_PENALTY if method == "condition" else None,
                       gradient_microbatch=1 if method == "latent" else None, **inputs)

        def change(batch, step, current):
            if step != GUIDANCE_STEP:
                return None
            predicted_eeg = eeg[batch]
            updates = []
            for clean, target in zip(current.split(1), predicted_eeg.split(1), strict=True):
                def loss(latent):
                    rgb = (vae.decode(latent / vae.config.scaling_factor).sample.float() * .5 + .5).clamp(0., 1.)
                    visual = (dino_cls(dino, rgb) - weights["visual_center"]) / weights["visual_scale"]
                    return (visual @ model - target).square().mean()

                updates.append(bounded_descent(clean, loss, UPDATE_RMS))
            return torch.cat(updates)

        def generate(stage, starts, complete):
            draw_images(stage, pipe, vae, prompts, tokens, seeds, clean_update=change if method == "latent" else None,
                        batch_starts=starts, batch_complete=complete)

        render_transaction(dest, request, generate, pins.verify)
        print("forward guided image render", subject, panel, arm, len(rows), flush=True)
        folders.append(dest)
    del dino, vae, pipe, adapter, projection
    torch.cuda.empty_cache()
    for folder in folders:
        if not (folder / "scores.json").is_file():
            score(folder, panel, Path(os.environ["ARCHIVE"]), folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, required=True, choices=range(1, 11))
    parser.add_argument("--panel", choices=("dev", "official", "fresh"), default="dev")
    parser.add_argument("--method", choices=("latent", "condition"), default="latent")
    parser.add_argument("--repeat-budget", choices=(4, 80), type=int, default=4)
    parser.add_argument("--arms", choices=(*ARMS, "forward-unweighted-wrong"), nargs="+", default=ARMS)
    args = parser.parse_args()
    run(args.subject, args.panel, args.method, args.repeat_budget, args.arms)
