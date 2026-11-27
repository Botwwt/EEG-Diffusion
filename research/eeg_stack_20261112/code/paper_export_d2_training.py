"""Export frozen four-trial D2 representations for matched semantic-head experiments."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from paper_render_cache import PinnedFiles, require, save_json


@torch.inference_mode()
def run(subject, output):
    from common import PREP, RUNS
    from paper_render import condition, groups, physical_eeg

    require(RUNS.name == "shared", "Use the fixed shared D2 checkpoint")
    files = PinnedFiles()
    split_path = PREP / "paper_split.json"
    dev_path = PREP / "paper_development.json"
    labels_path = PREP / "train_concept.npy"
    split_sha, dev_sha, labels_sha = (files.pin(path) for path in (split_path, dev_path, labels_path))
    plan = json.loads(split_path.read_bytes())
    training = np.asarray(plan["train_rows"])
    development = np.asarray(json.loads(dev_path.read_bytes())["dev_rows"])
    labels = np.load(labels_path, allow_pickle=False)
    train_objects = labels[training]
    dev_objects = labels[development]
    require(len(training) == 13230 and len(development) == 200 and len(np.unique(train_objects)) == 1323
            and set(train_objects).isdisjoint(dev_objects), "Changed whole-concept development split")
    raw, eeg_proof = physical_eeg(subject, "train", files)
    ordered = np.concatenate([training, development])
    repeated = groups("dev-r4")
    embeddings, checkpoint, condition_proof = condition("d2", subject, raw[ordered], repeated,
                                                       features_only=True, files=files, return_source=True)
    require(embeddings.shape == (len(ordered), 1024) and bool(torch.isfinite(embeddings).all()), "Invalid D2 output")
    files.verify()

    output.mkdir(parents=True, exist_ok=True)
    data = output / f"d2-train-dev-sub-{subject:02d}.npz"
    np.savez_compressed(data, train_rows=training, train_concept=train_objects, train_fused=embeddings[:len(training)].numpy(),
                        dev_rows=development, dev_concept=dev_objects, dev_fused=embeddings[len(training):].numpy())
    proof = dict(subject=subject, n_train=len(training), n_dev=len(development), repeats=repeated, checkpoint=checkpoint,
                 condition=condition_proof, eeg=eeg_proof, split=split_sha, development=dev_sha, labels=labels_sha,
                 data_sha256=files.pin(data), code_sha256=files.pin(Path(__file__)), weight_updates=0)
    save_json(output / f"d2-train-dev-sub-{subject:02d}.json", proof)
    print("Verified frozen D2 features", subject, len(training), len(development), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    run(opts.subject, opts.output)
