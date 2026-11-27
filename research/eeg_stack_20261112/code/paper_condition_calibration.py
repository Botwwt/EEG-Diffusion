"""Test training-only token calibration and an equal-energy correction to shared D2."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from paper_condition_probe import reference
from paper_render_cache import PinnedFiles, render_transaction, require, save_json
from paper_token_evidence import project_tokens, score_outputs, token_parts


ARMS = ("shared-d2", "center", "spread", "spread-wrong", "d2-relative", "d2-relative-wrong")
RELATIVE_STRENGTH = .5


def fit_moments(training_tokens, photo_tokens):
    require(training_tokens.shape == photo_tokens.shape and training_tokens.ndim == 3,
            "The aligned training token banks must share the image-condition shape")
    center, radius = reference(training_tokens)
    photo_center, photo_radius = reference(photo_tokens)
    require(radius > 0 and photo_radius > 0 and torch.isfinite(radius + photo_radius), "Invalid training spread")
    return dict(center=center, photo_center=photo_center, gain=(photo_radius / radius))


def adjust(query, calibration, scale=True):
    gain = calibration["gain"] if scale else 1.
    return calibration["photo_center"] + gain * (query.float() - calibration["center"])


def relative_shape(tokens):
    return token_parts(tokens)[1]


def length(tokens):
    return tokens.flatten(1).norm(dim=1)[:, None, None]


def with_norm(tokens, target_norm):
    norm = length(tokens)
    require(torch.all((norm > 1e-8) | (target_norm <= 1e-8)), "Cannot rescale a missing EEG direction")
    return tokens * (target_norm / norm.clamp_min(1e-8))


def orthogonal_update(candidate, d2_direction, requested_norm=None):
    content = relative_shape(candidate)
    power = d2_direction.square().sum((1, 2), keepdim=True).clamp_min(1e-16)
    parallel = (content * d2_direction).sum((1, 2), keepdim=True) / power
    residual = content - parallel * d2_direction
    return residual if requested_norm is None else with_norm(residual, requested_norm)


def token_conditions(packet, adapter, projection):
    models = packet["models"]
    truth = project_tokens(packet["visual"], adapter, projection)
    train = project_tokens(models["joint"]["fit_fused"], adapter, projection)
    joint = models["joint"]["query_ip"].cuda().float()
    d2 = models["d2"]["query_ip"].cuda().float()
    d2_train = models["d2"]["fit_ip"].cuda().float()
    full = fit_moments(train, truth)
    relative = fit_moments(relative_shape(train), relative_shape(truth))
    d2_train_mean = d2_train.mean(0)
    d2_direction = relative_shape(d2 - d2_train_mean)
    amplified = adjust(joint, full)
    donor = torch.arange(len(joint), device=joint.device).roll(len(joint) // 2)
    offset = amplified - full["photo_center"]
    spread_wrong = full["photo_center"] + with_norm(offset[donor], length(offset))
    complementary = (relative_shape(joint) - relative["center"]) * (RELATIVE_STRENGTH * relative["gain"])
    update = orthogonal_update(complementary, d2_direction)
    wrong_update = orthogonal_update(complementary[donor], d2_direction, length(update))
    require(torch.allclose(length(update), length(wrong_update), atol=1e-5, rtol=1e-5),
            "Matched and exchanged D2 updates must have the same amplitude")
    proof = dict(full_gain=float(full["gain"]), relative_gain=float(relative["gain"]), relative_strength=RELATIVE_STRENGTH,
                 center_translation_l2=float((full["center"] - full["photo_center"]).norm()),
                 joint_offset_median=float(length(offset).median()), d2_delta_median=float(length(update).median()),
                 d2_delta_over_own_residual_median=float((length(update) / length(d2_direction).clamp_min(1e-8)).median()),
                 equal_norm_delta_max_abs=float((length(update) - length(wrong_update)).abs().max()))
    values = (d2, adjust(joint, full, scale=False), amplified, spread_wrong, d2 + update, d2 + wrong_update)
    return {arm: value.cpu().half() for arm, value in zip(ARMS, values, strict=True)}, donor.tolist(), proof


@torch.inference_mode()
def run(subject, probe, output):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_oof_conditions import registered_rows
    from paper_render import draw_images, fit_layout, generator_inputs, groups, layout, physical_eeg, query_seed

    files = PinnedFiles()
    probe_sha, receipt_sha = files.pin(probe), files.pin(probe.with_suffix(".json"))
    receipt = json.loads(probe.with_suffix(".json").read_bytes())
    require(probe_sha == receipt["output_sha256"], "The condition packet changed after measurement")
    packet = torch.load(probe, map_location="cpu", weights_only=True)
    source = packet["proof"]
    train_rows, _, _ = registered_rows(PREP)
    split = PREP / "paper_split.json"
    split_plan = json.loads(split.read_bytes())
    rows = split_plan["official_rows"]
    repeat = groups("official-r4")
    train, train_proof = physical_eeg(subject, "train", files)
    eeg, test_proof = physical_eeg(subject, "test", files)
    require(source == receipt["proof"] and source["subject"] == subject and source["training"] == train_proof
            and source["official"] == test_proof and source["repeats"] == repeat
            and source["split_sha256"] == files.pin(split) and source["visual_sha256"] == files.pin(PREP / "F_train.npy")
            and np.array_equal(train_rows, packet["rows"].numpy()) and rows == list(range(200)),
            "Training calibration and render input receipts disagree")
    ridged, layout_proof = fit_layout(subject, split_plan, train, files=files, train_proof=train_proof, with_source=True)
    initial = layout(ridged, eeg, repeat)
    generator = generator_inputs(files)
    require(generator == source["generators"], "Token calibration needs the same frozen generator")
    pipe, adapter, project, prompt, vae = load_generator(torch.float16, False)
    conditions, wrong, calibration = token_conditions(packet, adapter, project)
    del packet, adapter, project
    path = output / f"sub-{subject:02d}"
    path.mkdir(parents=True, exist_ok=True)
    save_json(path / "calibration.json", dict(input_sha256=probe_sha, input_receipt_sha256=receipt_sha, **calibration))
    seeds = [query_seed("official", row) for row in rows]
    proof = dict(render_cache_version=2, subject=subject, row=rows, n=len(rows), budget="official-r4",
                 repeats=repeat, physical_eeg=test_proof, split_sha256=source["split_sha256"], generators=generator,
                 probe_sha256=probe_sha, source_models=source["sources"],
                 code_sha256=files.pin(Path(__file__)), calibration=calibration, seeds=seeds, draws=1,
                 unet_calls_per_draw=4, prompt="", diagnostic_only=False)
    outputs = []
    for arm, tokens in conditions.items():
        has_layout = arm == "shared-d2" or arm.startswith("d2-relative")
        donor = wrong if arm.endswith("-wrong") else rows
        request = dict(proof, arm=arm, layout=layout_proof if has_layout else None,
                       d2_eeg_donor=rows if has_layout else None, eeg_donor=donor)
        folder = path / arm

        def generate(stage, starts, complete):
            draw_images(stage, pipe, vae, prompt, tokens, seeds, initial if has_layout else None,
                        batch_starts=starts, batch_complete=complete)

        render_transaction(folder, request, generate, files.verify)
        outputs.append(folder)
        print("Calibrated render", subject, arm, len(rows), flush=True)
    del pipe, vae
    torch.cuda.empty_cache()
    score_outputs(outputs, "official")
    files.verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=int, choices=range(1, 11))
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.subject, args.probe, args.output)
