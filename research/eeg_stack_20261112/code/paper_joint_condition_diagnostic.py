"""Cross true layout and relative tokens on whole, held-out object concepts."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, render_transaction, require, save_json
from paper_token_evidence import project_tokens, score_outputs, token_parts


def layout_donors(layouts, labels):
    """Select the nearest layout within and outside each query's own concept."""
    summaries = F.avg_pool2d(layouts.float(), 4).flatten(1)
    distances = torch.cdist(summaries, summaries).square()
    group = torch.from_numpy(labels)
    same = group[:, None] == group[None, :]
    self_match = torch.eye(len(group), dtype=torch.bool)
    same_distance = distances.masked_fill(~same | self_match, torch.inf)
    different_distance = distances.masked_fill(same, torch.inf)
    require(torch.isfinite(same_distance).any(1).all() and torch.isfinite(different_distance).any(1).all(),
            "Every held stimulus needs both donor types")
    return same_distance.argmin(1), different_distance.argmin(1)


def project_reference_layout(reference, fitted):
    """Limit photo-derived geometry to the very same layout basis used by D2."""
    basis, mean = fitted["basis"], fitted["mean"]
    centered = reference.float().flatten(1) - mean
    return ((centered @ basis) @ basis.T + mean).reshape_as(reference)


class ReplacedImageProjection(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.reference = None

    def forward(self, tokens):
        if self.reference is None:
            return self.base(tokens)
        require(self.reference.shape == tokens.shape, "The oracle must preserve the native image token axes")
        return self.base(self.reference)


class RoutedUNet:
    def __init__(self, unet, projections, reference, steps):
        self.unet, self.dtype = unet, unet.dtype
        self.projections, self.reference, self.steps = projections, reference, steps

    def __call__(self, samples, timestep, **kwargs):
        if int(timestep[0]) not in self.steps:
            return self.unet(samples, timestep, **kwargs)
        try:
            for projection in self.projections:
                projection.reference = self.reference
            return self.unet(samples, timestep, **kwargs)
        finally:
            for projection in self.projections:
                projection.reference = None


def render_routing(output, common, pipe, vae, prompt, eeg_tokens, oracle, layout, pins):
    from paper_eeg_attention import image_layers
    from paper_render import draw_images, tensor_fingerprint

    wrapped = {}
    for name, (processor, attr) in image_layers(pipe.unet).items():
        wrapped[name] = ReplacedImageProjection(getattr(processor, attr))
        setattr(processor, attr, wrapped[name])
    keys = [name for name in wrapped if name.endswith(".to_k_ip")]
    values = [name for name in wrapped if name.endswith(".to_v_ip")]
    terminal_values = [name for name in values if name.startswith("up_blocks.0.")]
    require(len(keys) == len(values) > 0 and terminal_values, "Unexpected image-attention layer registry")
    steps = pipe.scheduler.timesteps.tolist()
    require(steps == [999, 749, 499, 249], "Expected the registered four-step generator")
    modes = {
        "d2": ([], [], False),
        "reference-relative": ([], steps, True),
        "reference-keys": (keys, steps, False),
        "reference-values": (values, steps, False),
        "reference-keys-early": (keys, steps[:2], False),
        "reference-terminal-values": (terminal_values, steps[-1:], False),
    }
    subject, budget = common["subject"], common["budget"]
    paths = []
    for arm, (names, timesteps, full) in modes.items():
        tokens = oracle if full else eeg_tokens
        reference_used = bool(timesteps)
        manifest = dict(common, arm=arm, training="relative-routing", eeg_donor=list(range(common["n"])),
                        reference_pixels_available_to_arm=reference_used, all_image_projections=full,
                        reference_projections=names, reference_timesteps=timesteps,
                        base_token_sha256=tensor_fingerprint(eeg_tokens), layout_sha256=tensor_fingerprint(layout),
                        input_token_sha256=tensor_fingerprint(tokens),
                        relative_oracle_sha256=tensor_fingerprint(oracle) if reference_used else None)
        folder = output / "images" / "relative-routing" / arm / budget / f"sub-{subject:02d}"

        def render(stage, starts, complete):
            for start in starts:
                unet = RoutedUNet(pipe.unet, [wrapped[name] for name in names], oracle[start:start + 8].cuda(), timesteps)
                runner = SimpleNamespace(unet=unet, scheduler=pipe.scheduler)
                draw_images(stage, runner, vae, prompt, tokens, common["seeds"], layout, batch_starts=[start],
                            batch_complete=complete, layout_strength=common["layout_strength"],
                            layout_low_keep=common["layout_low_keep"])

        render_transaction(folder, manifest, render, pins.verify)
        paths.append(folder)
        print(arm, subject, common["n"], flush=True)
    return modes, paths


@torch.inference_mode()
def run(subject, output, routing=False):
    from common import PREP, RUNS
    from paper_identity_render import load_generator
    from paper_photo_top1 import DEFAULT_LAYOUT, render_inputs
    from paper_render import condition, draw_images, generator_inputs, layout, query_seed, runtime, tensor_fingerprint
    from paper_uniform_baselines import reference_layout

    require(RUNS.name == "shared", "Expected the registered shared D2 checkpoint")
    pins = PinnedFiles()
    raw, rows, fit_rows, budget, groups, _, split_sha, dev_sha, eeg = render_inputs(subject, "dev", pins)
    concepts_path = PREP / "train_concept.npy"
    labels = np.load(concepts_path, allow_pickle=False)[rows]
    require(len(np.unique(labels)) == 20 and (np.unique(labels, return_counts=True)[1] == 10).all(),
            "Expected the held 20-concept, ten-photo development panel")
    checkpoint, layout_source = reference_layout(subject, split_sha, pins)
    eeg_layout = layout(checkpoint, raw, groups)

    visuals = PREP / "F_train.npy"
    latents = PREP / "lat512_train.npy"
    oracle_proof = {path.name: pins.pin(path) for path in (visuals, latents, concepts_path)}
    features = np.load(visuals, mmap_mode="r", allow_pickle=False)
    native_latents = np.load(latents, mmap_mode="r", allow_pickle=False)
    require(features.shape == (16540, 1024) and native_latents.shape == (16540, 4, 64, 64),
            "Photo cache differs from the registered VAE and fused vision sources")
    require(np.isfinite(features[rows]).all() and np.isfinite(native_latents[rows]).all(), "Nonfinite oracle input")
    reference_layouts = project_reference_layout(torch.from_numpy(native_latents[rows].copy()), checkpoint)
    sibling, different = layout_donors(reference_layouts, labels)
    matched = torch.arange(len(rows))
    require((sibling != matched).all() and np.all(labels[sibling.numpy()] == labels)
            and np.all(labels[different.numpy()] != labels), "Reference donor labels changed")

    pipe, adapter, projector, prompt, vae = load_generator(torch.float16, False)
    eeg_tokens, d2_hash, d2_proof = condition(
        "d2", subject, raw, groups, adapter, projector, files=pins, return_source=True
    )
    _, reference_rel = token_parts(project_tokens(torch.from_numpy(features[rows].copy()), adapter, projector).cpu().float())
    eeg_mean, _ = token_parts(eeg_tokens.float())
    oracle = (eeg_mean + reference_rel).half()
    oracle_sibling = (eeg_mean + reference_rel[sibling]).half()
    oracle_different = (eeg_mean + reference_rel[different]).half()
    del adapter, projector

    output.mkdir(parents=True, exist_ok=True)
    seeds = [query_seed("dev", row) for row in matched.tolist()]
    strength, keep = DEFAULT_LAYOUT
    common = dict(render_cache_version=2, subject=subject, training="joint-condition", budget=budget,
                  n=len(rows), row=rows, repeats=groups, prompt="", draws=1, unet_calls_per_draw=4, seeds=seeds,
                  split_sha256=split_sha, development_split_sha256=dev_sha, physical_eeg=eeg,
                  checkpoint_sha256=d2_hash, d2_model_source=d2_proof, layout_fit=layout_source,
                  layout_strength=strength, layout_low_keep=list(keep), training_rows=len(fit_rows),
                  generator_files_sha256=generator_inputs(pins), code_sha256=pins.pin(Path(__file__)), runtime=runtime())
    if routing:
        modes, paths = render_routing(output, common, pipe, vae, prompt, eeg_tokens, oracle, eeg_layout, pins)
        del pipe, vae
        torch.cuda.empty_cache()
        score_outputs(paths, "dev")
        save_json(output / "relative-routing.json", dict(subject=subject, arms=list(modes),
                                                        reference_eeg_condition=common, source_proof=oracle_proof))
        return
    arms = {
        "d2": (False, None, True),
        "reference-layout": (True, None, True),
        "reference-relative": (False, matched, True),
        "reference-both": (True, matched, True),
        "sibling-relative": (True, sibling, True),
        "different-relative": (True, different, True),
        "d2-no-remix": (False, None, False),
        "reference-both-no-remix": (True, matched, False),
    }
    calibrated_tokens = {"reference-relative": oracle, "reference-both": oracle,
                         "reference-both-no-remix": oracle, "sibling-relative": oracle_sibling,
                         "different-relative": oracle_different}
    paths = []
    for name, (true_layout, donor, remix) in arms.items():
        tokens = calibrated_tokens.get(name, eeg_tokens)
        initial = reference_layouts if true_layout else eeg_layout
        folder = output / "images" / "joint-condition" / name / budget / f"sub-{subject:02d}"
        manifest = dict(common, arm=name, eeg_donor=matched.tolist(),
                        reference_pixels_available_to_arm=true_layout or donor is not None,
                        reference_layout=true_layout, reference_relative_rows=None if donor is None else donor.tolist(),
                        oracle_sources=oracle_proof if true_layout or donor is not None else None,
                        layout_sha256=tensor_fingerprint(initial), token_sha256=tensor_fingerprint(tokens),
                        remix_initial_layout=remix)

        def render(stage, starts, complete):
            draw_images(stage, pipe, vae, prompt, tokens, seeds, initial, batch_starts=starts,
                        batch_complete=complete, layout_strength=strength, layout_low_keep=keep,
                        remix_initial_layout=remix)

        render_transaction(folder, manifest, render, pins.verify)
        paths.append(folder)
        print(name, subject, len(rows), flush=True)
    del pipe, vae
    torch.cuda.empty_cache()
    score_outputs(paths, "dev")
    save_json(output / "joint-condition.json", dict(subject=subject, arms=list(arms),
                                                   reference_eeg_condition=common, source_proof=oracle_proof))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routing", action="store_true", help="Replace only selected key/value projections with true relative tokens")
    args = parser.parse_args()
    run(args.subject, args.output, args.routing)
