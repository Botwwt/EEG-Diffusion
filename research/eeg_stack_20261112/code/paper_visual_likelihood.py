"""Guide a single four-step R80 reconstruction with the frozen EEG H/14 readout."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_middle_teacher import CONFIG_SHA, MODEL, MODEL_SHAS, image_processor
from paper_render_cache import PinnedFiles, atomic_file, render_transaction, require, save_json


ARMS = ("h14-image-guide", "h14-image-guide-wrong")
UPDATE_STEP = 1
STRENGTH = .15
BLOCK = 16
MICROBATCH = 2


def band_limited_step(gradient, block=BLOCK, strength=STRENGTH):
    flow = gradient.float()
    coarse = F.interpolate(F.avg_pool2d(flow, block), scale_factor=block, mode="nearest")
    change = flow - coarse
    rms = change.flatten(1).square().mean(1).sqrt()[:, None, None, None]
    require(torch.isfinite(rms).all(), "The image alignment gradient is not finite")
    step = change * (strength / rms.clamp_min(1e-6))
    return torch.where(rms > 1e-6, step, 0.)


def image_guidance(clean, target, vae, image_tower, mean, std, *, block=BLOCK, strength=STRENGTH):
    require(len(clean) == len(target), "Each preview needs a matched EEG direction")

    def similarity(image, start):
        return (image * target[start:start + len(image)]).sum()

    return image_alignment_step(clean, vae, image_tower, mean, std, similarity, block=block, strength=strength)


def image_alignment_step(clean, vae, image_tower, mean, std, similarity, *, block=BLOCK, strength=STRENGTH):
    updates = []
    for start in range(0, len(clean), MICROBATCH):
        initial = clean[start:start + MICROBATCH]
        with torch.enable_grad():
            latent = initial.detach().float().requires_grad_(True)
            rgb = (vae.decode(latent / vae.config.scaling_factor).sample * .5 + .5).clamp(0., 1.)
            resized = F.interpolate(rgb, size=(224, 224), mode="bicubic", align_corners=False, antialias=True).clamp(0., 1.)
            pixels = ((resized - mean) / std).half()
            image = F.normalize(image_tower(pixel_values=pixels).image_embeds.float())
            gradient, = torch.autograd.grad(similarity(image, start), latent)
        updates.append(band_limited_step(gradient, block, strength))
    return torch.cat(updates)


@torch.no_grad()
def run(subject, output, arms):
    from common import ROOT, RUNS
    from transformers import CLIPVisionModelWithProjection
    from paper_d2_object_swap import d2_models
    from paper_horizon_readout import repeat_views
    from paper_identity_render import load_generator
    from paper_photo_top1 import DEFAULT_LAYOUT, render_inputs
    from paper_render import condition, draw_images, generator_inputs, layout, query_seed, runtime, tensor_fingerprint
    from paper_token_evidence import score_outputs
    from paper_uniform_baselines import reference_layout

    require(RUNS.name == "shared", "The alignment experiment uses the reference D2 backbone")
    files = PinnedFiles()
    raw, rows, _, budget, repeats, _, split, development, physical = render_inputs(subject, "official", files)
    require(budget == "official-r80" and rows == list(range(200)) and repeats == [list(range(r,r+4)) for r in range(0,80,4)],
            "Use the complete official physical R80 queries")
    encoder, _, _, source = d2_models(files)
    readouts = []
    for start in range(0, len(raw), 20):
        trials = torch.from_numpy(raw[start:start + 20].copy()).cuda().float()
        values = [repeat_views(encoder, trials[:, view], subject)[1][:, 0] for view in repeats]
        readouts.append(torch.stack(values).mean(0))
    target = F.normalize(torch.cat(readouts))
    del encoder
    require(target.shape == (len(rows), 1024) and torch.isfinite(target).all(), "Missing frozen native H/14 evidence")
    output.mkdir(parents=True, exist_ok=True)
    target_path = output / "h14-target.npz"
    atomic_file(target_path, lambda stream: np.savez_compressed(stream, person=np.asarray(subject), row=rows, embedding=target.cpu().numpy()))
    target_sha = files.pin(target_path)
    model_dir = ROOT / "models" / MODEL
    vision = files.pin(model_dir / "model.safetensors")
    require(vision in MODEL_SHAS and files.pin(model_dir / "config.json") == CONFIG_SHA, "Unexpected frozen H/14 critic")
    image_tower = CLIPVisionModelWithProjection.from_pretrained(model_dir, local_files_only=True, use_safetensors=True,
                                                                torch_dtype=torch.float16).cuda().eval().requires_grad_(False)
    processor = image_processor()
    mean = torch.tensor(processor.image_mean, device="cuda")[None, :, None, None]
    std = torch.tensor(processor.image_std, device="cuda")[None, :, None, None]
    assets = generator_inputs(files)
    fitted, fitted_layout = reference_layout(subject, split, files)
    starts = layout(fitted, raw, repeats)
    pipe, adapter, projector, prompt, vae = load_generator(torch.float16, False)
    tokens, checkpoint, base = condition("d2", subject, raw, repeats, adapter, projector, files=files, return_source=True)
    require(base == source and pipe.scheduler.timesteps.tolist() == [999, 749, 499, 249], "Frozen D2 generator inputs disagree")
    seeds = [query_seed("official", i) for i in range(len(rows))]
    strength, keep = DEFAULT_LAYOUT
    record = dict(render_cache_version=2, subject=subject, budget=budget, n=len(rows), row=rows, split_sha256=split,
                  development_split_sha256=development, repeats=repeats, generator=assets, layout_source=fitted_layout,
                  layout_sha256=tensor_fingerprint(starts), base_token_sha256=tensor_fingerprint(tokens),
                  d2_source=base, checkpoint_sha256=checkpoint, physical_eeg=physical, seeds=seeds, prompt="", training="shared",
                  target_sha256=target_sha, h14_weights=vision, guidance_step=UPDATE_STEP, max_update_rms=STRENGTH,
                  preserved_latent_block=BLOCK, gradient_microbatch=MICROBATCH, previews=1, draws=1, unet_calls_per_draw=4,
                  reference_pixels_available_to_arm=False, runtime=runtime(), code_sha256=files.pin(Path(__file__)))
    folders = []
    for arm in arms:
        donors = np.roll(np.arange(len(rows)), 1) if arm.endswith("-wrong") else np.arange(len(rows))
        query = target[donors]
        destination = output / "images/shared" / arm / budget / f"sub-{subject:02d}"

        def update(batch, step, clean):
            if step == UPDATE_STEP:
                return image_guidance(clean, query[batch], vae, image_tower, mean, std)
            return None

        def draw(stage, batches, finish):
            draw_images(stage, pipe, vae, prompt, tokens, seeds, starts, batch_starts=batches, batch_complete=finish,
                        layout_strength=strength, layout_low_keep=keep, clean_update=update)

        render_transaction(destination, dict(record, arm=arm, guidance_donors=donors.tolist()), draw, files.verify)
        folders.append(destination)
        print("Visual guidance renders verified", subject, arm, len(rows), flush=True)
    del image_tower, vae, pipe, adapter, projector
    torch.cuda.empty_cache()
    score_outputs(folders, "official")
    save_json(output / "h14-guidance.json", dict(record=record, arms=list(arms)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, required=True, choices=range(1, 11))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    args = parser.parse_args()
    run(args.subject, args.output, args.arms)
