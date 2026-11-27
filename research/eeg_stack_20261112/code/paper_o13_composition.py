"""Test the O13 visual composition with a single query and no test-batch adaptation."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, render_transaction, require


ARMS = ("shared-d2", "o13-direct", "o13-compose", "d2-o13-compose", "d2-o13-compose-wrong")
R4 = [[0, 1, 2, 3]]
LAYOUT_GAIN = .65
MIX = .5


def binned_layout(coefficients, basis, mean):
    # Average pooling is linear, so only the small basis needs a full-resolution decode.
    pooled_basis = F.avg_pool2d(basis.T.reshape(-1, 4, 64, 64), 4).flatten(1)
    pooled_mean = F.avg_pool2d(mean.reshape(1, 4, 64, 64), 4).flatten(1)
    return coefficients @ pooled_basis + pooled_mean


def match_training_layout(predicted, held_out_training, visual_training):
    standardized = (predicted - held_out_training.mean(0)) / held_out_training.std(0).clamp_min(1e-6)
    return visual_training.mean(0) + LAYOUT_GAIN * standardized * visual_training.std(0)


def mix_directions(first, second, axis):
    first, second = first.float(), second.float()
    axis = F.normalize(axis.to(first).reshape(1, -1))
    first = F.normalize(first - (first @ axis.T) * axis)
    second = F.normalize(second - (second @ axis.T) * axis)
    return F.normalize((1. - MIX) * first + MIX * second)


def pooled_head(head, hidden):
    grouped, individual = hidden
    return .72 * head(grouped) + .28 * head(individual).reshape(len(grouped), 4, -1).mean(1)


@torch.inference_mode()
def run(subject, output):
    from common import EMB_KEYS, PREP, PRIOR, RUNS
    from exp0_intra10_swap_gate import OUT_DIR, get_concept_splits
    from exp6_draft_intra4 import DRAFT_DIR
    from main.models_adapter import FusionEncoderModel
    from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder
    from paper_d2_object_swap import d2_models, image_tokens, repeat_hidden
    from paper_identity_render import load_generator
    from paper_render import (draw_images, fit_layout, generator_inputs, layout, per_sample_cone, physical_eeg,
                              query_seed, render_code, runtime, tensor_fingerprint)
    from paper_token_evidence import score_outputs

    require(RUNS.name == "shared", "Use the archived shared O13-style and D2 checkpoints")
    files = PinnedFiles()
    split_path = PREP / "paper_split.json"
    split_sha = files.pin(split_path)
    split = json.loads(split_path.read_bytes())
    rows = split["official_rows"]
    train_rows = split["train_rows"]
    require(rows == list(range(200)) and np.array_equal(train_rows, get_concept_splits()[2]),
            "Use the registered official image order and only the registered training concepts")
    train, train_proof = physical_eeg(subject, "train", files)
    eeg, physical = physical_eeg(subject, "test", files)
    generator = generator_inputs(files)
    pipe, adapter, project, prompts, vae = load_generator(torch.float16, False)
    ridge, layout_source = fit_layout(subject, split, train, files=files, train_proof=train_proof, with_source=True)
    starts = layout(ridge, eeg, R4)
    seeds = [query_seed("official", row) for row in rows]
    donors = np.roll(rows, len(rows) // 2).tolist()

    base_path = OUT_DIR / "arm0_tr14_xbrain_s2025.pt"
    fold_layout_path = DRAFT_DIR / "z0_oof_tr14_256.pt"
    visual_path = PREP / "emb_train.pt"
    training_assets = {path.name: files.pin(path) for path in (base_path, fold_layout_path, visual_path)}
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    require(base["seed"] == 2025 and base["epochs"] == 150, "Unexpected historical O13-style fit")
    encoder = CrossSubjectHorizonEncoder().cuda().float().eval().requires_grad_(False)
    encoder.load_state_dict(base["state_dict"], strict=True)
    d2, residual, d2_cone, d2_sources = d2_models(files)
    visual_encoder = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.cuda().float().eval()
    fold_layout = torch.load(fold_layout_path, map_location="cpu", weights_only=True)
    require(fold_layout["Z_oof"].shape == (10, len(train_rows), 256), "Unexpected per-person fold-out layout rows")
    train_low = binned_layout(fold_layout["Z_oof"][subject - 1].cuda().float(),
                              fold_layout["V_pca"].cuda().float(), fold_layout["my"].cuda().float())
    visual_low = torch.load(visual_path, map_location="cpu", weights_only=True)[EMB_KEYS[2]][train_rows]
    visual_low = visual_low.reshape(len(train_rows), -1).cuda().float()
    layout_features = match_training_layout(F.avg_pool2d(starts.cuda().float(), 4).flatten(1), train_low, visual_low)
    require(torch.isfinite(layout_features).all() and visual_low.shape == train_low.shape,
            "Visual composition requires aligned finite layout coordinates")

    original, clip_h, clip_b, fitted, fitted_hidden, archived = [], [], [], [], [], []
    for start in range(0, len(rows), 20):
        raw = torch.from_numpy(eeg[start:start + 20, :4].copy()).cuda().float()
        h_base, h_d2 = repeat_hidden(encoder, raw, subject), repeat_hidden(d2, raw, subject)
        original.append(pooled_head(encoder.head_f, h_base))
        clip_h.append(F.normalize(pooled_head(encoder.head_h14, h_base)) * base["h14_norm"])
        clip_b.append(F.normalize(pooled_head(encoder.head_b32, h_base)) * base["b32_norm"])
        fitted.append(pooled_head(d2.head_f, h_d2))
        fitted_hidden.append(.72 * h_d2[0] + .28 * h_d2[1].reshape(len(raw), 4, -1).mean(1))
        archived.append(image_tokens(d2.head_f, residual, h_d2, d2_cone, adapter, project).cpu())
    o13_f, clip_h, clip_b, fitted_f, hidden = map(torch.cat, (original, clip_h, clip_b, fitted, fitted_hidden))

    def synthesize(order):
        return torch.cat([visual_encoder(**dict(zip(EMB_KEYS, batch, strict=True))).float()
                          for batch in zip(clip_h[order].split(20), clip_b[order].split(20),
                                           layout_features.split(20), strict=True)])

    synth, synth_wrong = synthesize(rows), synthesize(donors)

    def condition(features, cone, residual_input=None):
        c = per_sample_cone(features, cone["R"], cone["alpha"], cone["beta"], cone["u_tr"])
        projected = torch.cat([project(adapter(block.half())) for block in c.split(20)])
        return (projected if residual_input is None else residual(residual_input, projected)).cpu()

    o13_cone = base["cone"]
    variants = {
        "shared-d2": torch.cat(archived),
        "o13-direct": condition(o13_f, o13_cone),
        "o13-compose": condition(mix_directions(o13_f, synth, o13_cone["u_tr"]), o13_cone),
        "d2-o13-compose": condition(mix_directions(fitted_f, synth, d2_cone["u_tr"]), d2_cone, hidden),
        "d2-o13-compose-wrong": condition(mix_directions(fitted_f, synth_wrong, d2_cone["u_tr"]), d2_cone, hidden),
    }
    render_sources = render_code(files, "d2")
    for function in (run, load_generator, image_tokens):
        path = Path(inspect.getfile(function))
        render_sources[path.name] = files.pin(path)
    shared = dict(render_cache_version=2, subject=subject, budget="official-r4", n=len(rows), row=rows,
                  repeats=R4, seeds=seeds, draws=1, unet_calls_per_draw=4, prompt="", diagnostic_only=False,
                  split_sha256=split_sha, physical_eeg=physical, layout_source=layout_source, layout_donor=rows,
                  generator_files_sha256=generator, training_assets=training_assets, d2_sources=d2_sources,
                  mix=MIX, layout_gain=LAYOUT_GAIN, render_sources=render_sources, runtime=runtime())
    folders = []
    for arm in ARMS:
        tokens = variants[arm]
        destination = output / "images" / f"sub-{subject:02d}" / arm
        uses_composition = arm in ("o13-compose", "d2-o13-compose", "d2-o13-compose-wrong")
        donor = (donors if arm.endswith("-wrong") else rows) if uses_composition else None
        request = dict(shared, arm=arm, eeg_donor=rows, composition_donor=donor,
                       token_map=tensor_fingerprint(tokens))

        def generate(folder, batches, complete):
            draw_images(folder, pipe, vae, prompts, tokens, seeds, starts, batch_starts=batches, batch_complete=complete)

        render_transaction(destination, request, generate, files.verify)
        folders.append(destination)
        print("O13 composition render", subject, arm, len(rows), flush=True)
    del pipe, adapter, project, vae
    torch.cuda.empty_cache()
    score_outputs(folders, "official")
    files.verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=int, choices=range(1, 11))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.subject, args.output)
