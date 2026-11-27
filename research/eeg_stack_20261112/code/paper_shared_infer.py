"""Render one image with the registered shared M1a+D2 from one participant's EEG."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from paper_object_infer import load_trial_mean


def study_budget(trials, subject):
    if not 1 <= subject <= 10:
        raise ValueError('Use the participant index of the registered shared fit')
    if trials.ndim != 3 or trials.shape[1:] != (17, 250) or trials.dtype.kind != 'f' or not np.isfinite(trials).all():
        raise ValueError('Expected finite float EEG in the 17 posterior electrode by 250 sample order')
    if len(trials) not in (4, 80):
        raise ValueError('Use four or eighty physically measured presentations')
    return f'official-r{len(trials)}'


def reconstruct(trials, subject, seed, output):
    import torch

    from common import PREP, RUNS
    from paper_identity_render import load_generator
    from paper_render import condition, draw_images, groups, layout
    from paper_render_cache import PinnedFiles
    from paper_uniform_baselines import reference_layout

    budget = study_budget(trials, subject)
    if not 0 <= seed < 2**63 or output.suffix.lower() != '.png':
        raise ValueError('Use a nonnegative 63-bit seed and a .png output file')
    if RUNS.name != 'shared' or not torch.cuda.is_available():
        raise RuntimeError('Use the shared M1a+D2 fit directory and a CUDA device')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    files = PinnedFiles()
    split = files.pin(PREP / 'paper_split.json')
    fitted, _ = reference_layout(subject, split, files)
    eeg = trials.astype(np.float32, copy=False)[None]
    repeats = groups(budget)
    with torch.inference_mode(), TemporaryDirectory(dir=output.parent) as workspace:
        pipe, adapter, projection, prompts, vae = load_generator(torch.float16, False)
        z0 = layout(fitted, eeg, repeats)
        tokens, _ = condition('d2', subject, eeg, repeats, adapter, projection, files=files)
        draw_images(Path(workspace), pipe, vae, prompts, tokens, [seed], z0)
        files.verify()
        os.link(Path(workspace) / '00000.png', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eeg', type=Path, required=True, help='Preprocessed float .npy with four or eighty physical presentations')
    parser.add_argument('--subject', type=int, required=True, choices=range(1, 11))
    parser.add_argument('--seed', type=int, default=20261125)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reconstruct(load_trial_mean(args.eeg), args.subject, args.seed, args.output)
