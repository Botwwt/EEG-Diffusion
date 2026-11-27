"""Render matched Brain-HiVE comparisons for alternative four-repeat EEG pools."""

import argparse
import inspect
import os
from pathlib import Path

import numpy as np
import torch

from paper_repeat_latency import aggregate
from paper_render_cache import PinnedFiles, require, save_json


ARMS = ("native-paper", "channel-precision", "symmetric-offset", "aligned-offset", "aligned-wrong")
BUDGETS = ("dev-r4", "official-r4")


def image_root(root, budget, subject):
    return root / "images" / budget / f"sub-{subject:02d}"


@torch.inference_mode()
def infer(root, budget, subject):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_object_foreground_render import split_input
    from paper_oof_conditions import fitted_native
    from paper_render import generator_inputs, groups, physical_eeg, query_seed, tensor_fingerprint
    from paper_soft_text import wrong_donors
    from paper_token_evidence import project_tokens, render_arms

    files = PinnedFiles()
    path = root / f"sub-{subject:02d}.pt"
    fit_hash = files.pin(path)
    state = torch.load(path, weights_only=True, map_location="cuda")
    raw, measurement, rows = split_input(subject, budget, files)
    training_measurement = measurement
    if budget != "dev-r4":
        _, training_measurement = physical_eeg(subject, "train", files)
    training = state["proof"]
    alignment_code = files.pin(Path(inspect.getfile(aggregate)))
    require(training["subject"] == subject and training["split"] == files.pin(PREP / "paper_split.json")
            and training["labels"] == files.pin(PREP / "train_concept.npy")
            and training["train_eeg"]["eeg"] == training_measurement["array_sha256"]
            and training["code"] == alignment_code, "Alignment must use this person's registered train EEG")
    native, checkpoint = fitted_native(subject, "cuda")
    repeats = groups(budget)
    require(repeats == [[0, 1, 2, 3]], "Only registered four-repeat queries are supported")
    features = {arm: [] for arm in ARMS[:-1]}
    changes = []
    for first in range(0, len(rows), 20):
        block = torch.from_numpy(np.array(raw[first:first + 20, repeats[0]], copy=True)).cuda().float()
        result = aggregate(block, state["center"], state["axes"], state["strength"])
        for arm, label in zip(ARMS[:-1], ("mean", "precision", "symmetric", "aligned"), strict=True):
            features[arm].append(native.backbone(result[label]))
        changes.append(torch.stack(((result["aligned"] - result["symmetric"]).square().flatten(1).mean(1),
                                    (result["symmetric"] - result["precision"]).square().flatten(1).mean(1))))
    features = {arm: torch.cat(parts) for arm, parts in features.items()}
    donors = wrong_donors(len(rows)).tolist()
    features["aligned-wrong"] = features["aligned-offset"][donors]
    panel = budget.split("-")[0]
    seeds = [query_seed(panel, i) for i in range(len(rows))]
    assets = generator_inputs(files)
    pipeline = load_generator(torch.float16, False)
    _, adapter, projection, _, _ = pipeline
    proof = dict(subject=subject, budget=budget, repeats=repeats, physical_eeg=measurement,
                 alignment_sha256=fit_hash, fit=training, native_checkpoint_sha256=files.pin(checkpoint),
                 generator_inputs=assets, code_sha256=files.pin(__file__))
    positions = list(range(len(rows)))
    variants = []
    for arm, value in features.items():
        tokens = project_tokens(value, adapter, projection).half().cpu()
        variants.append((arm, tokens, dict(image_token_sha256=tensor_fingerprint(tokens),
                                          image_eeg_donor=donors if arm == "aligned-wrong" else positions)))
    out = image_root(root, budget, subject)
    out.mkdir(parents=True, exist_ok=True)
    signed_only, unsigned = torch.cat(changes, dim=1).mean(1).sqrt().tolist()
    files.verify()
    save_json(out / "waveform-changes.json", dict(signed_only_rms=signed_only, unsigned_rms=unsigned, proof=proof))
    render_arms(out, variants, rows, seeds, proof, files, pipeline)


def score(root, budget, subject):
    from paper_token_evidence import score_outputs

    parent = image_root(root, budget, subject)
    score_outputs([parent / arm for arm in ARMS], budget.split("-")[0])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("infer", "score"))
    cli.add_argument("subject", type=int, choices=range(1, 11))
    cli.add_argument("--budget", choices=BUDGETS, required=True)
    args = cli.parse_args()
    root = Path(os.environ["EEG_CROSSFIT_ROOT"])
    (infer if args.phase == "infer" else score)(root, args.budget, args.subject)
