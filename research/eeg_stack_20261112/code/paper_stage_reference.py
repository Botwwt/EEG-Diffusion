"""Locate where the four-step reconstruction responds to better object information."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from paper_render_cache import PinnedFiles, render_transaction, require


TIMESTEPS = (999, 749, 499, 249)
BASE = "shared-d2"
REFERENCE_STEPS = tuple(f"photo-{time}" for time in TIMESTEPS)
ARMS = (BASE, "photo-all", "other-photo-all", *REFERENCE_STEPS, "eeg76-all")
R4 = [[0, 1, 2, 3]]
R76 = np.arange(4, 80).reshape(-1, 4).tolist()


def condition_plan(arm):
    if arm in REFERENCE_STEPS:
        index = REFERENCE_STEPS.index(arm)
        return tuple("photo" if step == index else "eeg4" for step in range(len(TIMESTEPS)))
    source = {BASE: "eeg4", "photo-all": "photo", "other-photo-all": "other-photo", "eeg76-all": "eeg76"}[arm]
    return (source,) * len(TIMESTEPS)


def step_tokens(plan, sources, donors):
    available = dict(sources)
    if "other-photo" in plan:
        available["other-photo"] = sources["photo"][donors]
    return tuple(available[source] for source in plan)


@torch.inference_mode()
def run(subject, output, arms):
    from common import PREP, RUNS
    from paper_identity_render import load_generator
    from paper_render import (condition, draw_images, fit_layout, generator_inputs, layout,
                              physical_eeg, query_seed, render_code, runtime, tensor_fingerprint)
    from paper_token_evidence import project_tokens, score_outputs
    from paper_visual_ceiling import official_features

    require(RUNS.name == "shared" and len(set(arms)) == len(arms), "Select distinct arms of the shared D2 comparison")
    files = PinnedFiles()
    split_path = PREP / "paper_split.json"
    split_sha = files.pin(split_path)
    split = json.loads(split_path.read_bytes())
    rows = split["official_rows"]
    require(rows == list(range(200)), "Unexpected official stimulus order")
    train, train_proof = physical_eeg(subject, "train", files)
    eeg, eeg_proof = physical_eeg(subject, "test", files)
    generators = generator_inputs(files)
    pipe, adapter, projection, prompts, vae = load_generator(torch.float16, False)
    require(tuple(pipe.scheduler.timesteps.tolist()) == TIMESTEPS, "Changed sampler schedule")
    ridge, start_proof = fit_layout(subject, split, train, files=files, train_proof=train_proof, with_source=True)
    starts = layout(ridge, eeg, R4)
    seeds = [query_seed("official", row) for row in rows]
    donors = np.roll(rows, len(rows) // 2).tolist()
    code = render_code(files, "d2")
    for function in (run, load_generator, project_tokens, official_features):
        source = Path(inspect.getfile(function))
        code[source.name] = files.pin(source)
    shared = dict(render_cache_version=2, subject=subject, budget="official-r4", n=len(rows), row=rows,
                  repeats=R4, seeds=seeds, draws=1, unet_calls_per_draw=4, prompt="", layout_donor=rows,
                  split_sha256=split_sha, physical_eeg=eeg_proof, layout_source=start_proof,
                  layout_sha256=tensor_fingerprint(starts), timesteps=list(TIMESTEPS),
                  generator_files_sha256=generators, render_sources=code, runtime=runtime())

    requested = {value for arm in arms for value in condition_plan(arm)}
    tokens, proof = {}, {}
    for source, groups in (("eeg4", R4), ("eeg76", R76)):
        if source in requested:
            values, checkpoint, origin = condition("d2", subject, eeg, groups, adapter, projection,
                                                   files=files, return_source=True)
            tokens[source] = values
            proof[source] = dict(physical_repeats=groups, checkpoint_sha256=checkpoint, model=origin,
                                 eeg_donor=rows, tokens=tensor_fingerprint(values))
    if requested.intersection(("photo", "other-photo")):
        features, photo_rows, sources = official_features(PREP, files)
        require(photo_rows == rows, "The reference condition covers different stimuli")
        values = project_tokens(features, adapter, projection).cpu().half()
        tokens["photo"] = values
        proof["photo"] = dict(visual_donor=rows, sources=sources, tokens=tensor_fingerprint(values))
        proof["other-photo"] = dict(visual_donor=donors, sources=sources, tokens=tensor_fingerprint(values[donors]))
    require(all(value.shape == (len(rows), 4, 2048) and torch.isfinite(value).all() for value in tokens.values()),
            "Invalid condition arrays")

    folders = []
    for arm in arms:
        plan = condition_plan(arm)
        step_condition = step_tokens(plan, tokens, donors)
        destination = output / "images" / f"sub-{subject:02d}" / arm
        request = dict(shared, arm=arm, diagnostic_only=arm != BASE, step_condition=list(plan),
                       condition_sources={name: proof[name] for name in sorted(set(plan))})

        def generate(folder, batches, complete):
            draw_images(folder, pipe, vae, prompts, step_condition, seeds, starts,
                        batch_starts=batches, batch_complete=complete)

        render_transaction(destination, request, generate, files.verify)
        folders.append(destination)
        print("Step reference render", subject, arm, len(rows), flush=True)
    del pipe, adapter, projection, vae
    torch.cuda.empty_cache()
    score_outputs(folders, "official")
    files.verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=int, choices=range(1, 11))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=ARMS)
    args = parser.parse_args()
    run(args.subject, args.output, args.arms)
