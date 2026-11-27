"""Test whether early visual feedback can change object structure on held-out development classes."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, render_transaction, require
from paper_visual_likelihood import CONFIG_SHA, MODEL, MODEL_SHAS, image_guidance, image_processor


ARMS = ("d2", "d2-no-remix", "early-truth", "early-truth-wrong", "early-eeg", "early-eeg-wrong")
STEPS = (0, 1)
RMS = .6
BLOCK = 64


@torch.no_grad()
def run(subject, output):
    from common import EMB_KEYS, PREP, ROOT, RUNS
    from transformers import CLIPVisionModelWithProjection
    from paper_d2_object_swap import d2_models
    from paper_horizon_readout import registration, repeat_views
    from paper_identity_render import load_generator
    from paper_photo_top1 import DEFAULT_LAYOUT, render_inputs
    from paper_render import condition, donor_rows, draw_images, generator_inputs, layout, query_seed, runtime, tensor_fingerprint
    from paper_token_evidence import score_outputs
    from paper_uniform_baselines import reference_layout

    require(RUNS.name == "shared", "Use the frozen 1,323-concept reference D2")
    files = PinnedFiles()
    (_, dev, _), labels, registered = registration(PREP, files)
    raw, rows, _, budget, repeats, _, split, development, physical = render_inputs(subject, "dev", files)
    require(budget == "dev-r4" and np.array_equal(dev, rows) and repeats == [list(range(4))],
            "Expected the complete registered R4 development panel")
    encoder, _, _, source = d2_models(files)
    estimates = []
    for start in range(0, len(raw), 20):
        trials = torch.from_numpy(raw[start:start + 20].copy()).cuda().float()
        estimates.append(repeat_views(encoder, trials[:, repeats[0]], subject)[1][:, 0])
    predicted = F.normalize(torch.cat(estimates))
    del encoder

    ref = PREP / "emb_train.pt"
    reference_sha = files.pin(ref)
    truth = F.normalize(torch.load(ref, map_location="cpu", weights_only=True)[EMB_KEYS[0]][rows].cuda().float())
    wrong = donor_rows("token-reliable-wrong", "dev", labels[dev])
    matched = np.arange(len(rows))
    require(predicted.shape == truth.shape == (len(rows), 1024)
            and torch.isfinite(predicted).all() and torch.isfinite(truth).all(), "Incomplete visual conditions")
    critics = ROOT / "models" / MODEL
    vision = files.pin(critics / "model.safetensors")
    require(vision in MODEL_SHAS and files.pin(critics / "config.json") == CONFIG_SHA, "Unregistered frozen image tower")
    critic = CLIPVisionModelWithProjection.from_pretrained(
        critics, local_files_only=True, use_safetensors=True, torch_dtype=torch.float16,
    ).cuda().eval().requires_grad_(False)
    processor = image_processor()
    mean = torch.tensor(processor.image_mean, device="cuda")[None, :, None, None]
    std = torch.tensor(processor.image_std, device="cuda")[None, :, None, None]
    assets = generator_inputs(files)
    fitted_layout, ridge = reference_layout(subject, split, files)
    starts = layout(fitted_layout, raw, repeats)
    pipe, adapter, projector, prompt, vae = load_generator(torch.float16, False)
    tokens, checkpoint, base = condition("d2", subject, raw, repeats, adapter, projector, files=files, return_source=True)
    require(base == source and pipe.scheduler.timesteps.tolist() == [999, 749, 499, 249], "Frozen D2 generator inputs disagree")
    seeds = [query_seed("dev", i) for i in range(len(rows))]
    strength, keep = DEFAULT_LAYOUT
    receipt = dict(
        render_cache_version=2, training="shared", subject=subject, budget=budget, n=len(rows), row=rows,
        repeats=repeats, split_sha256=split, development_split_sha256=development, physical_eeg=physical,
        d2_source=base, checkpoint_sha256=checkpoint, layout_source=ridge, layout_sha256=tensor_fingerprint(starts),
        base_token_sha256=tensor_fingerprint(tokens), generator=assets, registration=registered,
        code_sha256=files.pin(Path(__file__)), guidance_code_sha256=files.pin(Path(image_guidance.__code__.co_filename)),
        runtime=runtime(), prompt="", draws=1, unet_calls_per_draw=4, seeds=seeds,
    )
    configs = (
        (None, matched, True, False), (None, matched, False, False),
        (truth, matched, False, True), (truth, wrong, False, True),
        (predicted, matched, False, False), (predicted, wrong, False, False),
    )
    destinations = []
    for arm, (target, donors, remix, oracle) in zip(ARMS, configs, strict=True):
        destination = output / "images/shared" / arm / budget / f"sub-{subject:02d}"
        target = None if target is None else target[donors]

        def update(batch, step, clean):
            if step in STEPS:
                return image_guidance(clean, target[batch], vae, critic, mean, std, block=BLOCK, strength=RMS)
            return None

        def draw(stage, batches, finish):
            draw_images(stage, pipe, vae, prompt, tokens, seeds, starts, batch_starts=batches, batch_complete=finish,
                        layout_strength=strength, layout_low_keep=keep, remix_initial_layout=remix,
                        clean_update=update if target is not None else None)

        proof = dict(receipt, arm=arm, remix_initial_layout=remix, layout_strength=strength, layout_low_keep=keep,
                     guidance_steps=STEPS if target is not None else (), guidance_rms=RMS if target is not None else 0.,
                     preserved_latent_block=BLOCK, guiding_vector_sha256=None if target is None else tensor_fingerprint(target),
                     guidance_donors=donors.tolist(), guided_by_true_image=oracle,
                     h14_weights=vision if target is not None else None, truth_embedding_sha256=reference_sha if oracle else None)
        render_transaction(destination, proof, draw, files.verify)
        destinations.append(destination)
        print("Early visual guidance renders verified", subject, arm, len(rows), flush=True)
    del pipe, vae, critic, adapter, projector
    torch.cuda.empty_cache()
    score_outputs(destinations, "dev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, type=int, choices=range(1, 11))
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    run(arguments.subject, arguments.output)
