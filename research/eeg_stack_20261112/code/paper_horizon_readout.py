"""Expose the three archived D2 temporal representations under the M1a repeat rule."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from paper_render_cache import PinnedFiles, atomic_file, require, save_json


VIEWS = ("native", "460-ms", "640-ms", "900-ms")


def registration(prep, pins):
    paths = dict(split=prep / "paper_split.json", development=prep / "paper_development.json",
                 labels=prep / "train_concept.npy", identities=prep / "train_ids.json")
    sources = {key: pins.pin(path) for key, path in paths.items()}
    split = json.loads(paths["split"].read_bytes())
    rows = tuple(np.asarray(part, dtype=np.int64) for part in (
        split["train_rows"], json.loads(paths["development"].read_bytes())["dev_rows"], split["fresh_rows"],
    ))
    labels = np.load(paths["labels"], allow_pickle=False)
    require(labels.shape == (16540,) and labels.dtype.kind in "iu", "Unexpected concept index")
    concepts = [np.unique(labels[panel], return_counts=True) for panel in rows]
    for panel, (names, counts), expected in zip(rows, concepts, (1323, 20, 50), strict=True):
        require(panel.shape == (expected * 10,) and len(set(panel)) == len(panel)
                and len(names) == expected and np.all(counts == 10), "The registered object panels changed")
    require(len(set(np.concatenate([group[0] for group in concepts]))) == 1393, "The three panels must hold out whole concepts")
    return rows, labels, sources


def repeat_views(encoder, trials, subject):
    from paper_render import forward_branches_and_trunk, prec_erp_analytic

    _, aggregate, views = forward_branches_and_trunk(encoder, prec_erp_analytic(trials), subject - 1, 4.)
    _, independent, evidence = forward_branches_and_trunk(encoder, trials.flatten(0, 1), subject - 1, 1.)
    n = len(trials)

    def aggregate_views(values):
        return torch.stack(values, dim=1)

    pooled = aggregate_views((aggregate, *views))
    individual = aggregate_views((independent, *evidence))
    hidden = .72 * pooled + .28 * individual.reshape(n, 4, len(VIEWS), -1).mean(1)
    h14 = .72 * encoder.head_h14(pooled) + .28 * encoder.head_h14(individual).reshape(n, 4, len(VIEWS), -1).mean(1)
    return hidden, h14


@torch.inference_mode()
def export(subject, output):
    from common import PREP, RUNS
    from paper_d2_object_swap import d2_models
    from paper_render import physical_eeg, render_code

    require(RUNS.name == "shared", "Use the archived shared D2 model")
    pins = PinnedFiles()
    (training, development, fresh), _, registration_proof = registration(PREP, pins)
    raw, eeg_proof = physical_eeg(subject, "train", pins)
    encoder, _, _, source = d2_models(pins)
    require(tuple(encoder.horizons) == (115, 160, 225), "Frozen temporal bounds differ from the study")
    proof = dict(registration=registration_proof, source=source, eeg=eeg_proof, code=pins.pin(Path(__file__)),
                 encoder_code=render_code(pins, "d2"), subject=subject, views=list(VIEWS), weights=[.72, .28],
                 repeats=list(range(4)), eeg_layout="recorded physical trials", train_size=len(training),
                 development_size=len(development), fresh_size=len(fresh))
    output.mkdir(parents=True, exist_ok=True)
    features = output / f"sub-{subject:02d}.npy"
    auxiliary = features.with_suffix(".h14.npz")
    receipt = features.with_suffix(".json")
    if receipt.is_file():
        previous = json.loads(receipt.read_bytes())
        require(previous == dict(proof=proof, features_sha256=pins.pin(features), h14_sha256=pins.pin(auxiliary)),
                "An incompatible horizon export already exists")
        return
    all_rows = np.concatenate((training, development, fresh))
    all_hidden = np.empty((len(all_rows), len(VIEWS), 1024), dtype=np.float16)
    held_h14 = []
    for start in range(0, len(all_rows), 80):
        indices = all_rows[start:start + 80]
        trials = torch.from_numpy(raw[indices].copy()).cuda().float()
        hidden, h14 = repeat_views(encoder, trials, subject)
        require(hidden.shape == h14.shape == (len(indices), len(VIEWS), 1024), "Unexpected temporal readout dimensions")
        all_hidden[start:start + len(indices)] = hidden.half().cpu().numpy()
        first_held = max(0, len(training) - start)
        if first_held < len(indices):
            held_h14.append(h14[first_held:].cpu().numpy())
        if start % 3200 == 0:
            print(f"Exporting D2 horizons for sub-{subject:02d} at row {start}", flush=True)
    held_h14 = np.concatenate(held_h14)
    require(held_h14.shape == (700, len(VIEWS), 1024) and np.isfinite(held_h14).all() and np.isfinite(all_hidden).all(),
            "Invalid native temporal representations")
    pins.verify()
    atomic_file(features, lambda stream: np.save(stream, all_hidden, allow_pickle=False))
    atomic_file(auxiliary, lambda stream: np.savez_compressed(stream, row=all_rows[len(training):], embedding=held_h14))
    save_json(receipt, dict(proof=proof, features_sha256=pins.pin(features), h14_sha256=pins.pin(auxiliary)))
    print(f"Verified {len(all_rows)} D2 temporal representations for sub-{subject:02d}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, choices=range(1, 11), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.subject, args.output)
