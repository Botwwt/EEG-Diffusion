"""Render the exact verified R80 H/14 readout through the image Value projections."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from paper_h14_value_probe import blend_relative
from paper_joint_condition_diagnostic import ReplacedImageProjection, RoutedUNet
from paper_render_cache import PinnedFiles, render_transaction, require
from paper_token_evidence import project_tokens, score_outputs


ARMS = ("r80-d2-replay", "r80-h14-value", "r80-h14-value-wrong")
TIMING_ARMS = ("r80-d2-replay", "r80-value-early", "r80-value-early-wrong", "r80-value-late", "r80-value-late-wrong")


def routes(timesteps, rows, timing):
    matched = np.arange(len(rows))
    wrong = np.roll(matched, 1)
    if timing:
        pre, post = timesteps[:2], timesteps[2:]
        return zip(TIMING_ARMS, (([], matched), (pre, matched), (pre, wrong), (post, matched), (post, wrong)), strict=True)
    return zip(ARMS, (([], matched), (timesteps, matched), (timesteps, wrong)), strict=True)


def guidance_target(archive, subject, binding, files):
    """Read the registered physical R80 vector already used in the pixel-feedback study."""
    observed = files.pin(archive)
    seal = archive.with_suffix(".tar.sha256")
    files.pin(seal)
    require(observed == seal.read_text(encoding="ascii").split()[0], "R80 guidance score archive changed")
    with tarfile.open(archive, "r") as source:
        raw = source.extractfile("h14-target.npz").read()
        proof = json.load(source.extractfile("h14-guidance.json"))["record"]
    require(proof["subject"] == subject and proof["budget"] == "official-r80" and sha256(raw).hexdigest() == proof["target_sha256"]
            and all(proof[field] == reference for field, reference in binding.items()), "Different physical R80 inputs")
    with np.load(io.BytesIO(raw), allow_pickle=False) as target:
        require(int(target["person"]) == subject and target["row"].tolist() == binding["row"],
                "The R80 H/14 target uses a different query order")
        values = torch.from_numpy(target["embedding"].copy()).float()
    require(values.shape == (len(binding["row"]), 1024) and torch.isfinite(values).all()
            and torch.allclose(values.norm(dim=-1), torch.ones(len(values)), atol=1e-5, rtol=0.),
            "Incomplete normalized R80 H/14 embedding")
    return values, {"archive_sha256": observed, "target_sha256": proof["target_sha256"]}


@torch.inference_mode()
def run(subject, archive, output, timing=False):
    from common import EMB_KEYS, PREP, PRIOR, RUNS
    from main.models_adapter import FusionEncoderModel
    from paper_eeg_attention import image_layers
    from paper_identity_render import load_generator
    from paper_photo_top1 import DEFAULT_LAYOUT, render_inputs
    from paper_render import condition, draw_images, generator_inputs, layout, query_seed, runtime, tensor_fingerprint
    from paper_uniform_baselines import reference_layout

    require(RUNS.name == "shared", "Use the same D2 backbone as the original ten-person guidance run")
    files = PinnedFiles()
    raw, rows, train, budget, repeats, _, split, development, physical = render_inputs(subject, "official", files)
    require(budget == "official-r80" and rows == list(range(200)) and repeats == [list(range(r, r+4)) for r in range(0,80,4)],
            "Expected all official queries and their original 80 physical repetitions")
    weights = generator_inputs(files)
    fixed_layout, ridge = reference_layout(subject, split, files)
    starts = layout(fixed_layout, raw, repeats)
    pipe, adapter, projector, prompt, vae = load_generator(torch.float16, False)
    tokens, checkpoint, source = condition("d2", subject, raw, repeats, adapter, projector, files=files, return_source=True)
    seeds = [query_seed("official", i) for i in range(len(rows))]
    binding = dict(row=rows, physical_eeg=physical, repeats=repeats, seeds=seeds, split_sha256=split,
                   development_split_sha256=development, d2_source=source, checkpoint_sha256=checkpoint)
    estimate, target_proof = guidance_target(archive, subject, binding, files)

    path = PREP / "emb_train.pt"
    refs_sha = files.pin(path)
    refs = torch.load(path, map_location="cpu", weights_only=True)
    h14, *rest = EMB_KEYS
    style = {key: refs[key][train].float().mean(0, keepdim=True).cuda() for key in rest}
    norm = refs[h14][train].float().norm(dim=1).median().cuda()
    del refs
    fusion = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.cuda().float().eval()
    fused = []
    for batch in estimate.cuda().split(64):
        other = {k: x.expand(len(batch), *x.shape[1:]) for k, x in style.items()}
        fused.append(fusion(**{h14: F.normalize(batch) * norm, **other}))
    projected = project_tokens(torch.cat(fused), adapter, projector).cpu()
    del fusion, adapter, projector
    timesteps = pipe.scheduler.timesteps.tolist()
    require(timesteps == [999, 749, 499, 249], "Unexpected four-step sampler")
    projections = []
    for proc, attr in image_layers(pipe.unet).values():
        if attr == "to_v_ip":
            replacement = ReplacedImageProjection(getattr(proc, attr))
            setattr(proc, attr, replacement)
            projections.append(replacement)
    require(len(projections) == 70, "Image Value registry differs from the development experiment")
    strength, keep = DEFAULT_LAYOUT
    common = dict(
        **binding, render_cache_version=2, training="shared", subject=subject, budget=budget, n=len(rows), prompt="",
        draws=1, unet_calls_per_draw=4, layout_source=ridge, layout_sha256=tensor_fingerprint(starts),
        base_token_sha256=tensor_fingerprint(tokens), generator=weights, runtime=runtime(),
        target_source=target_proof, train_visual_sha256=refs_sha, train_h14_norm=float(norm),
        code_sha256=files.pin(Path(__file__)), reference_pixels_available_to_arm=False,
    )
    output_folders = []
    for arm, (steps, donors) in routes(timesteps, rows, timing):
        values = blend_relative(tokens, projected[donors], 1.) if steps else None
        path = output / "images/shared" / arm / budget / f"sub-{subject:02d}"

        def draw(stage, batches, finish):
            for first in batches:
                visual = None if values is None else values[first:first + 8].cuda()
                unet = RoutedUNet(pipe.unet, projections, visual, steps)
                runner = SimpleNamespace(unet=unet, scheduler=pipe.scheduler)
                draw_images(stage, runner, vae, prompt, tokens, seeds, starts, batch_starts=[first], batch_complete=finish,
                            layout_strength=strength, layout_low_keep=keep)

        receipt = dict(common, arm=arm, value_timesteps=steps, value_donors=donors.tolist(),
                       value_token_sha256=None if values is None else tensor_fingerprint(values))
        render_transaction(path, receipt, draw, files.verify)
        output_folders.append(path)
        print("Verified official R80 RGB Value render", subject, arm, len(rows), flush=True)
    del pipe, vae
    torch.cuda.empty_cache()
    score_outputs(output_folders, "official")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, required=True, choices=range(1, 11))
    parser.add_argument("--target-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-study", action="store_true", help="Compare early and late Value routing with paired wrong-EEG controls")
    a = parser.parse_args()
    run(a.subject, a.target_archive, a.output, a.timing_study)
