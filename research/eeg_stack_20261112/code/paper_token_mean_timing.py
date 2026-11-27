"""Test when the query mean of the native image tokens is useful for object generation."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from paper_render_cache import PinnedFiles, require
from paper_token_evidence import development, project_tokens, render_arms, score_outputs, token_parts

ARMS = ("native-paper", "fixed-mean-early", "fixed-mean-late", "fixed-mean-all", "fixed-mean-early-wrong")


def step_conditions(native, fixed, arm, donors):
    if arm == "native-paper":
        return (native,) * 4
    mean, relative = token_parts(native)
    if arm == "fixed-mean-early-wrong":
        relative = relative[list(donors)]
        native = (mean + relative).to(native.dtype)
    neutral = (fixed + relative).to(native.dtype)
    if arm == "fixed-mean-all":
        return (neutral,) * 4
    if arm == "fixed-mean-late":
        return native, native, neutral, neutral
    require(arm in ("fixed-mean-early", "fixed-mean-early-wrong"), "Unknown token mean experiment")
    return neutral, neutral, native, native


@torch.inference_mode()
def render(subject, panel, arms):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_oof_conditions import registered_rows, study_root
    from paper_render import condition, donor_rows, draw_images, generator_inputs, groups, physical_eeg, query_seed

    root, files = study_root(), PinnedFiles()
    split = PREP / "paper_split.json"
    split_sha = files.pin(split)
    development_sha = None
    if panel == "dev":
        plan, rows = development(PREP, files)
        require(subject in plan["subjects"], "Use the development registration")
        development_sha = files.pin(PREP / "paper_development.json")
    else:
        rows = json.loads(split.read_bytes())["official_rows"]
        require(rows == list(range(200)), "Official benchmark registration differs")
    photo_file = PREP / "F_train.npy"
    labels = PREP / "train_concept.npy"
    preparation = {path.name: files.pin(path)
                   for path in (photo_file, labels, PREP / "lat512_train_ok.npy", Path(inspect.getfile(registered_rows)))}
    train, _, _ = registered_rows(PREP)
    train_visual = torch.from_numpy(np.load(photo_file, mmap_mode="r", allow_pickle=False)[train].copy())
    repeats = groups(f"{panel}-r4")
    raw, physical = physical_eeg(subject, "train" if panel == "dev" else "test", files)
    native, checkpoint = condition("native-paper", subject, raw[rows], repeats, features_only=True, files=files)
    assets = generator_inputs(files)
    pipeline = load_generator(torch.float16, False)
    _, adapter, projection, _, _ = pipeline
    fixed = project_tokens(train_visual, adapter, projection).mean((0, 1), keepdim=True)
    true_token = project_tokens(native, adapter, projection).half()
    require(torch.isfinite(fixed).all() and torch.isfinite(true_token).all(), "Nonfinite visual or query tokens")
    order = list(range(len(rows)))
    swapped = np.roll(order, len(order) // 2).tolist()
    if panel == "dev":
        swapped = donor_rows("token-reliable-wrong", panel, np.load(labels, allow_pickle=False)[rows]).tolist()
    variants = []
    for arm in arms:
        donor = swapped if arm.endswith("-wrong") else order
        step_tokens = step_conditions(true_token, fixed, arm, donor)
        variants.append((arm, tuple(t.cpu() for t in step_tokens), dict(relative_eeg_row=donor)))
    seeds = [query_seed(panel, k) for k in order]
    shared = dict(subject=subject, budget=f"{panel}-r4", repeats=repeats, physical_eeg=physical,
                  split_sha256=split_sha, development_split_sha256=development_sha, training_mean_sources=preparation,
                  native_checkpoint_sha256=checkpoint, generator_files_sha256=assets,
                  code_sha256=files.pin(Path(__file__)), render_source_sha256=files.pin(Path(inspect.getfile(draw_images))))
    output = render_arms(root / "mean-timing" / panel / f"sub-{subject:02d}", variants, rows, seeds,
                         shared, files, pipeline)
    del pipeline, adapter, projection
    torch.cuda.empty_cache()
    score_outputs(output, panel)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("subject", type=int, choices=range(1, 11))
    cli.add_argument("--panel", choices=("dev", "official"), default="dev")
    cli.add_argument("--arms", choices=ARMS, nargs="+", default=ARMS)
    args = cli.parse_args()
    render(args.subject, args.panel, args.arms)
