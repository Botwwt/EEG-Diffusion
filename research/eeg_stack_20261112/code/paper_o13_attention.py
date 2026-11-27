"""Compare late D2/O13 image-token mixing with a mixture of separately normalized attention responses."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, render_transaction, require
from paper_spatial_value import SpatialImageValue


TIMESTEPS = (999, 749, 499, 249)
MATCHED = "o13-attention-late"
WRONG = "o13-attention-late-wrong"
TOKEN_MIX = "o13-token-mix-late"
IDENTITY = "o13-attention-identity"
DEFAULT_ARMS = ("d2", TOKEN_MIX, MATCHED, WRONG)
ARMS = (*DEFAULT_ARMS, IDENTITY)


def token_midpoint(base, companion):
    dtype = torch.promote_types(base.dtype, torch.float32)
    return torch.lerp(base.to(dtype), companion.to(dtype), .5).to(base.dtype)


def compose_features(vision, keys, h14, b32, layouts, donors):
    """Exchange semantic views at their source and retain each recipient's third input."""
    donors = donors.to(h14.device)
    result = []
    for start in range(0, len(layouts), 20):
        batch = slice(start, start + 20)
        inputs = (h14[donors[batch]], b32[donors[batch]], layouts[batch])
        result.append(vision(**dict(zip(keys, inputs, strict=True))).float())
    return torch.cat(result)


class O13ImageAttention(SpatialImageValue):
    """Add the difference of two independent image attention responses before the output projection."""

    def __init__(self, attention):
        processor = attention.processor
        super().__init__(attention, processor.to_v_ip)
        self.key_layer = processor.to_k_ip
        self.weight = 0.

    def _add_image_residual(self, module, inputs):
        if self.reference is None:
            return None
        query, key, value = (self.activations.pop(name) for name in ("query", "key", "value"))
        other_key = F.linear(self.reference, self.key_layer.weight, self.key_layer.bias)
        other_value = F.linear(self.reference, self.value_layer.weight, self.value_layer.bias)
        require(other_key.shape == key.shape and other_value.shape == value.shape,
                "Replacement must preserve the native image attention axes")
        query = self._heads(query)
        if self.query_norm is not None:
            query = self.query_norm(query)
        native = F.scaled_dot_product_attention(query, self._heads(key), self._heads(value))
        alternate = F.scaled_dot_product_attention(query, self._heads(other_key), self._heads(other_value))
        output = inputs[0]
        dtype = torch.promote_types(output.dtype, torch.float32)
        difference = alternate.to(dtype) - native.to(dtype)
        correction = self.weight * self.scale * difference.transpose(1, 2).flatten(2)
        return (output + correction.to(output.dtype),)


class LateAttentionUNet:
    def __init__(self, unet, routes, reference, weight):
        self.unet, self.dtype = unet, unet.dtype
        self.routes, self.reference, self.weight = routes, reference, weight

    def __call__(self, samples, timestep, **kwargs):
        if self.weight == 0. or int(timestep[0]) not in TIMESTEPS[2:]:
            return self.unet(samples, timestep, **kwargs)
        native = kwargs["cross_attention_kwargs"]["ip_hidden_states"]
        require(self.reference.shape == native.shape and self.reference.dtype == native.dtype,
                "Reference must use the native image token shape and dtype")
        if torch.equal(native, self.reference):
            return self.unet(samples, timestep, **kwargs)
        try:
            for route in self.routes:
                route.reference, route.weight = self.reference, self.weight
            return self.unet(samples, timestep, **kwargs)
        finally:
            for route in self.routes:
                route.reference = None
                route.activations.clear()


@torch.inference_mode()
def run(subject, panel, arms, output, score=False):
    from common import EMB_KEYS, PREP, RUNS
    from paper_d2_object_swap import d2_models
    from paper_identity_render import load_generator
    from paper_photo_top1 import (DEFAULT_LAYOUT, balanced_groups, consensus_hidden, direction_blend,
                                  keep_token_mean, load_composition, ordinary_readout, project_condition, render_inputs)
    from paper_render import (donor_rows, draw_images, generator_inputs, layout, query_seed, render_code,
                              runtime, tensor_fingerprint)
    from paper_uniform_baselines import reference_layout

    require(RUNS.name == "shared", "Use the registered shared D2 fit root")
    files = PinnedFiles()
    raw, rows, training_rows, budget, groups, repeats, split, dev, physical = render_inputs(subject, panel, files)
    ridge, layout_source = reference_layout(subject, split, files)
    starts = layout(ridge, raw, groups)
    generator_files = generator_inputs(files)
    pipe, adapter, projection, prompts, vae = load_generator(torch.float16, False)
    require(tuple(pipe.scheduler.timesteps.tolist()) == TIMESTEPS, "Expected the registered four-step scheduler")
    d2, residual, cone, d2_sources = d2_models(files)
    composition_needed = bool(set(arms) & {TOKEN_MIX, MATCHED, WRONG})
    composition_sources = None
    if composition_needed:
        brain, vision, layout_features, h_scale, b_scale, composition_sources = load_composition(
            subject, training_rows, starts, files)
    f_base, h_base, h14, b32 = [], [], [], []
    for start in range(0, len(rows), 20):
        trials = torch.from_numpy(np.array(raw[start:start + 20], copy=True)).cuda().float()
        feature, hidden = ordinary_readout(d2, trials, subject, groups)
        f_base.append(feature)
        h_base.append(hidden)
        if composition_needed:
            pooled = consensus_hidden(brain, trials[:, repeats], subject)
            h14.append(F.normalize(brain.head_h14(pooled)) * h_scale)
            b32.append(F.normalize(brain.head_b32(pooled)) * b_scale)
    f0, h0 = torch.cat(f_base), torch.cat(h_base)
    base = project_condition(f0, h0, residual, cone, adapter, projection)
    require(base.shape == (len(rows), 4, 2048) and torch.isfinite(base).all(), "Invalid D2 image tokens")
    own = np.arange(len(rows))
    donors = own
    if WRONG in arms:
        if panel == "official":
            labels = own  # The official panel presents exactly one photograph per concept.
        else:
            label_file = PREP / "train_concept.npy"
            files.pin(label_file)
            labels = np.load(label_file, mmap_mode="r", allow_pickle=False)[rows]
        donors = donor_rows("token-reliable-wrong", panel, labels)
    candidates = {}
    if composition_needed:
        h_view, b_view = torch.cat(h14), torch.cat(b32)
        for arm, order in ((MATCHED, own), (WRONG, donors)):
            if arm in arms or (arm == MATCHED and TOKEN_MIX in arms):
                visual = compose_features(vision, EMB_KEYS, h_view, b_view, layout_features, torch.from_numpy(order))
                tangent = direction_blend(cone, (visual,), (1.,))
                projected = project_condition(tangent, h0, residual, cone, adapter, projection)
                candidates[arm] = keep_token_mean(base, projected)
                require(torch.isfinite(candidates[arm]).all(), "Nonfinite O13 image tokens")
        del brain, vision, h_view, b_view, layout_features
    del d2, residual, adapter, projection, f0, h0, f_base, h_base, h14, b32

    routes = {name: O13ImageAttention(pipe.unet.get_submodule(name.removesuffix(".processor")))
              for name, processor in pipe.unet.attn_processors.items()
              if hasattr(processor, "to_k_ip") and hasattr(processor, "to_v_ip")}
    require(len(routes) == 70, "Expected the 70 pinned image-attention locations")
    source = render_code(files, "d2")
    for name in ("paper_o13_attention.py", "paper_photo_top1.py", "paper_o13_composition.py",
                 "paper_spatial_value.py", "paper_d2_object_swap.py", "paper_identity_render.py",
                 "paper_uniform_baselines.py"):
        source[name] = files.pin(Path(__file__).with_name(name))
    seeds = [query_seed(panel, i) for i in range(len(rows))]
    strength, keep = DEFAULT_LAYOUT
    base_sha = tensor_fingerprint(base)
    common = dict(render_cache_version=2, training="shared", subject=subject, budget=budget,
                  n=len(rows), row=rows, repeats=groups, eeg_donor=own.tolist(), seeds=seeds,
                  prompt="", draws=1, unet_calls_per_draw=4, split_sha256=split,
                  development_split_sha256=dev, physical_eeg=physical, d2_sources=d2_sources,
                  checkpoint_sha256=d2_sources["condition_checkpoint"]["sha256"], layout_mode="ridge-shell",
                  layout_source=layout_source, layout_sha256=tensor_fingerprint(starts), layout_strength=strength,
                  layout_low_keep=list(keep), generator_files_sha256=generator_files, render_code_sha256=source,
                  runtime=runtime(), token_sha256=[base_sha] * 4)
    outputs = {}
    for arm in arms:
        companion = candidates.get(MATCHED if arm == TOKEN_MIX else arm, base)
        reference = token_midpoint(base, companion) if arm == TOKEN_MIX else companion
        weight = {"d2": 0., TOKEN_MIX: 1.}.get(arm, .5)
        uses_o13 = arm in (TOKEN_MIX, MATCHED, WRONG)
        order = donors if arm == WRONG else own
        request = dict(common, arm=arm, semantic_condition=arm, late_timesteps=list(TIMESTEPS[2:]) if weight else [],
                       attention_layers=sorted(routes) if weight else [], attention_response_weight=weight,
                       companion_token_sha256=tensor_fingerprint(companion),
                       late_reference_token_sha256=tensor_fingerprint(reference),
                       token_midpoint_before_projection=arm == TOKEN_MIX, composition_sources=composition_sources if uses_o13 else None,
                       composition_semantic_donor=order.tolist() if uses_o13 else None,
                       recipient_composition_layout_preserved=True if uses_o13 else None,
                       consensus={"whole_weight": .85, "bag_weight": .15,
                                  "balanced_r80": balanced_groups(80) if panel == "official" else None} if uses_o13 else None)
        folder = output / "images" / "shared" / arm / budget / f"sub-{subject:02d}"

        def generate(stage, batch_starts, done):
            for start in batch_starts:
                late = LateAttentionUNet(pipe.unet, routes.values(), reference[start:start + 8].cuda(), weight)
                renderer = SimpleNamespace(unet=late, scheduler=pipe.scheduler)
                draw_images(stage, renderer, vae, prompts, base, seeds, starts, batch_starts=[start],
                            batch_complete=done, layout_strength=strength, layout_low_keep=keep)

        render_transaction(folder, request, generate, files.verify)
        outputs[arm] = folder
        print("rendered", subject, panel, arm, len(rows), flush=True)

    if {"d2", IDENTITY}.issubset(outputs):
        def pixels(arm):
            return json.loads((outputs[arm] / "render.json").read_bytes())["pixels_sha256"]

        require(pixels("d2") == pixels(IDENTITY), "Identity attention did not reproduce the native D2 PNGs")
        print("verified identical D2 and identity PNGs", subject, panel, len(rows), flush=True)
    del pipe, vae, routes
    torch.cuda.empty_cache()
    if score:
        from paper_token_evidence import score_outputs

        score_outputs(outputs.values(), panel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, choices=range(1, 11), required=True)
    parser.add_argument("--panel", choices=("dev", "fresh", "official"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=DEFAULT_ARMS)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID"), "Use a Slurm compute node for EEG and model workloads")
    require(len(args.arms) == len(set(args.arms)), "Duplicate method in run")
    run(args.subject, args.panel, args.arms, args.output, args.score)


if __name__ == "__main__":
    main()
