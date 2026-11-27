"""Measure how much object identity the fixed relative-token basis can transmit."""

import inspect
import os
from pathlib import Path

import numpy as np
import torch

from paper_direct_tokens import restore_relative
from paper_render_cache import PinnedFiles, require
from paper_token_evidence import development, project_tokens, render_arms, score_outputs, token_parts


@torch.inference_mode()
def oracle():
    from common import PREP
    from paper_identity_render import load_generator
    from paper_oof_conditions import study_root
    from paper_render import condition, donor_rows, draw_images, generator_inputs, groups, physical_eeg, query_seed

    subject = 1
    root, files = study_root(), PinnedFiles()
    _, rows = development(PREP, files)
    target_file = root / "training-targets.pt"
    target_digest = files.pin(target_file)
    bank = torch.load(target_file, map_location="cuda", weights_only=True)
    split = files.pin(PREP / "paper_split.json")
    photo_file = PREP / "F_train.npy"
    labels = PREP / "train_concept.npy"
    require(files.pin(photo_file) == bank["sources"]["photos"] and split == bank["sources"]["split"]
            and files.pin(labels) == bank["sources"]["labels"], "The diagnostic and target basis must use the same registration")
    basis = bank["basis"]
    repeats = groups("dev-r4")
    raw, proof = physical_eeg(subject, "train", files)
    native, checkpoint = condition("native-paper", subject, raw[rows], repeats, features_only=True, files=files)
    assets = generator_inputs(files)
    require(all(assets[f"prior/{key}"] == digest for key, digest in bank["sources"]["projection"].items()),
            "The token projection must match the fixed training basis")
    pipeline = load_generator(torch.float16, False)
    _, adapter, projection, _, _ = pipeline
    visual = np.load(photo_file, mmap_mode="r", allow_pickle=False)
    original = torch.from_numpy(visual[rows].copy())
    _, relative = token_parts(project_tokens(original, adapter, projection))
    coefficients = (relative - basis["center"]).flatten(1) @ basis["axes"] / basis["scale"]
    reconstructed = restore_relative(coefficients, basis, calibrate=True)
    mean, _ = token_parts(project_tokens(native, adapter, projection))
    own = list(range(len(rows)))
    swapped = donor_rows("token-reliable-wrong", "dev", np.load(labels, allow_pickle=False)[rows]).tolist()
    variants = [(name, (mean + reconstructed[donor]).half().cpu(), dict(relative_photo_donor=donor))
                for name, donor in (("own-relative-oracle", own), ("wrong-relative-oracle", swapped))]
    seeds = [query_seed("dev", row) for row in own]
    shared = dict(subject=subject, diagnostic_only=True, budget="dev-r4", repeats=repeats,
                  training_targets_sha256=target_digest, split_sha256=split,
                  development_split_sha256=files.pin(PREP / "paper_development.json"),
                  visual_sha256=files.pin(photo_file), physical_eeg=proof, native_checkpoint_sha256=checkpoint,
                  generator_files_sha256=assets, code_sha256=files.pin(Path(__file__)),
                  render_source_sha256=files.pin(Path(inspect.getfile(draw_images))))
    outputs = render_arms(root / "ceiling" / f"sub-{subject:02d}", variants, rows, seeds, shared, files, pipeline)
    del pipeline, adapter, projection
    torch.cuda.empty_cache()
    score_outputs(outputs, "dev")


if __name__ == "__main__":
    oracle()
