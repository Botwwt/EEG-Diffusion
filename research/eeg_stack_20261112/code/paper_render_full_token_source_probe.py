"""Generate paired development images from direct full-token source conditions."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from paper_render_cache import PinnedFiles, render_transaction, require
from paper_token_evidence import project_tokens, score_outputs


ARMS = ("d2", "full-token-gaussian", "full-token-flow", "full-token-flow-wrong")


@torch.inference_mode()
def run(subject, conditions, output):
    from common import PREP, RUNS
    from paper_d2_object_swap import d2_models
    from paper_identity_render import load_generator
    from paper_photo_top1 import DEFAULT_LAYOUT, ordinary_readout, render_inputs
    from paper_render import draw_images, generator_inputs, layout, per_sample_cone, query_seed, runtime, tensor_fingerprint
    from paper_uniform_baselines import reference_layout

    require(os.environ.get("SLURM_JOB_ID") and RUNS.name == "shared", "Use a compute node and the frozen shared fit")
    files = PinnedFiles()
    meta = conditions / "full-token-source.json"
    archive = conditions / "full-token-conditions.npz"
    receipt_sha, input_sha = files.pin(meta), files.pin(archive)
    fit = json.loads(meta.read_bytes())
    raw, rows, _, budget, groups, _, split, development, physical = render_inputs(subject, "dev", files)
    labels_path = PREP / "train_concept.npy"
    require(fit["conditions_sha256"] == input_sha and fit["photos"] == 13230 and fit["n"] == 2000
            and fit["token_shape"] == [4, 2048] and fit["proof"]["split"] == split
            and fit["proof"]["development"] == development and fit["proof"]["labels"] == files.pin(labels_path)
            and fit["proof"]["truth"] == files.pin(PREP / "F_train.npy"), "The token prior uses a different development registration")
    labels = np.load(labels_path, allow_pickle=False)[rows]
    with np.load(archive, allow_pickle=False) as data:
        index, donors = data["people"].tolist().index(subject), data["donors"]
        require(np.array_equal(rows, data["rows"]) and np.array_equal(np.sort(donors), np.arange(len(rows)))
                and np.all(labels != labels[donors]), "Full-token rows or donor control changed")
        sets = {arm: torch.from_numpy(data[arm][index]).cuda() for arm in ARMS[1:]}
    require(all(t.shape == (len(rows), 4, 2048) and t.dtype == torch.float16 and bool(t.isfinite().all())
                for t in sets.values()), "Invalid full-token condition tensor")
    require(torch.equal(sets["full-token-flow-wrong"], sets["full-token-flow"][donors]), "Wrong token assignment changed")
    model_files = generator_inputs(files)
    require(all(model_files[k] == v for k, v in fit["generator"].items()), "The token projector changed")
    pipe, adapter, projection, prompts, vae = load_generator(torch.float16, False)
    encoder, residual, cone, base_fit = d2_models(files)
    require(base_fit["condition_checkpoint"]["sha256"] == fit["eeg"][str(subject)]["checkpoint"],
            "The original EEG score used a different D2 checkpoint")
    original = []
    for i in range(0, len(raw), 20):
        trials = torch.from_numpy(np.array(raw[i:i + 20], copy=True)).cuda().float()
        original.append(ordinary_readout(encoder, trials, subject, groups))
    features, hidden = (torch.cat(t) for t in zip(*original, strict=True))
    fused = per_sample_cone(features, cone["R"], cone["alpha"], cone["beta"], cone["u_tr"])
    direct = project_tokens(fused, adapter, projection).half()
    d2 = residual(hidden, direct)
    delta = d2.float() - direct.float()
    conditions_after_residual = {"d2": d2.cpu(), **{name: (t.float() + delta).half().cpu() for name, t in sets.items()}}
    ridge, layout_source = reference_layout(subject, split, files)
    z0 = layout(ridge, raw, groups)
    strength, keep = DEFAULT_LAYOUT
    seeds = [query_seed("dev", i) for i in range(len(rows))]
    current = list(range(len(rows)))
    base = dict(render_cache_version=2, training="full-token-source", budget=budget, subject=subject, n=len(rows),
                row=rows, repeats=groups, seeds=seeds, eeg_donor=current, split_sha256=split,
                development_split_sha256=development, prompt="", draws=1, unet_calls_per_draw=4,
                physical_eeg=physical, checkpoint_sha256=fit["eeg"][str(subject)]["checkpoint"],
                d2_sources=base_fit, generator_files_sha256=model_files, source_conditions_sha256=input_sha,
                source_receipt_sha256=receipt_sha, layout_source=layout_source, layout_sha256=tensor_fingerprint(z0),
                layout_strength=strength, layout_low_keep=list(keep), fixed_d2_residual_sha256=tensor_fingerprint(delta),
                code_sha256=files.pin(Path(__file__)), runtime=runtime())
    folders = []
    for arm, tokens in conditions_after_residual.items():
        folder = output / f"images/full-token-source/{arm}/{budget}/sub-{subject:02d}"
        proof = dict(base, arm=arm, token_sha256=tensor_fingerprint(tokens),
                     source_donors=donors.tolist() if arm.endswith("-wrong") else current)

        def generate(stage, batches, done):
            draw_images(stage, pipe, vae, prompts, tokens, seeds, z0, batch_starts=batches, batch_complete=done,
                        layout_strength=strength, layout_low_keep=keep)

        render_transaction(folder, proof, generate, files.verify)
        folders.append(folder)
        print("Rendered direct full-token condition", subject, arm, len(rows), flush=True)
    del pipe, vae
    torch.cuda.empty_cache()
    score_outputs(folders, "dev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, choices=(1, 3), required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    run(opts.subject, opts.conditions, opts.output)
