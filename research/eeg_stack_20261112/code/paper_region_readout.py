"""Train one shared EEG readout per area-controlled H/14 semantic target."""

import argparse
from pathlib import Path

import numpy as np
import torch

from paper_horizon_readout import registration
from paper_raw_semantic_readout import leave_photo_out, normalize, read_json
from paper_region_value_probe import REFERENCE_VIEWS, load_views
from paper_render_cache import PinnedFiles, atomic_file, require, save_json
from paper_semantic_contrast import data_registry, matched_montages
from paper_temporal_readout import BATCH, SAMPLES_PER_BIN, SEED, TemporalReadout, held_concepts, score, train


EPOCHS = 12
SEGMENTS = 4
ARMS = ("whole", "coarse", "foreground", "background")


def run(benchmark, view_path, output, device):
    pins = PinnedFiles()
    prep = benchmark / "verified/data/prep"
    (registered_train, _, _), _, registered = registration(prep, pins)
    views, view_receipt = load_views(view_path, "train", registered_train, registered, pins)
    raw, train_rows, dev_rows, labels, category, image, text, layout, data_proof = data_registry(benchmark)
    require(np.array_equal(train_rows, registered_train), "Supervision must match the EEG training rows")
    held_rows = held_concepts(benchmark, train_rows, dev_rows, labels)
    full = benchmark / "native_baseline_candidate/enigma/prep"
    data, queries, _, recording, norm = matched_montages(raw, full, train_rows, held_rows, data_proof["photo_registry"],
                                                         bin_samples=SAMPLES_PER_BIN, dtype=np.float16, with_normalization=True)
    eeg, queries = torch.from_numpy(data).to(device), torch.from_numpy(queries).to(device)
    del data
    previous = benchmark / "semantic-energy-check/temporal-segments-20261127/visual-targets.pt"
    initializer = read_json(previous.with_suffix(".json"))
    normalizer = previous.with_name("normalization.npy")
    prior = initializer["sources"]
    require(prior["data"] == data_proof and prior["eeg"] == recording and prior["segments"] == SEGMENTS
            and pins.pin(previous) == initializer["weights_sha256"]
            and pins.pin(normalizer) == prior["normalization_sha256"]
            and np.array_equal(norm, np.load(normalizer, allow_pickle=False)),
            "Initialization must use the same training classes, EEG, and normalization")
    initial = torch.load(previous, map_location="cpu", weights_only=True)

    def fresh_model():
        model = TemporalReadout(segments=SEGMENTS)
        model.load_state_dict(initial, strict=True)
        return model

    group = np.unique(labels[train_rows], return_inverse=True)[1]
    require(np.array_equal(np.bincount(group), np.full(1323, 10)), "Training targets require ten photos per concept")
    groups = np.argsort(group, kind="stable").reshape(1323, 10)
    photos = image[train_rows]
    target_photo = torch.from_numpy(photos).to(device)
    target_text = torch.from_numpy(text[labels[train_rows]]).to(device)
    visual = {"whole": photos, **{name: values.numpy() for name, values in views.items()}}
    require(set(visual) == set(REFERENCE_VIEWS), "Readout control or region target is missing")
    proof = dict(data=data_proof, eeg=recording, registration=registered, region_view_receipt=view_receipt,
                 region_view_sha256=pins.pin(view_path), initial_sha256=pins.pin(previous), normalization_sha256=pins.pin(normalizer),
                 epochs=EPOCHS, batch_size=BATCH, seed=SEED, segments=SEGMENTS, samples_per_bin=SAMPLES_PER_BIN,
                 repeats=[0, 1, 2, 3], physical_subjects=10, train_count=len(train_rows), view="other-nine",
                 code={name: pins.pin(Path(__file__).with_name(name)) for name in (
                     Path(__file__).name, "paper_region_value_probe.py", "paper_temporal_readout.py", "paper_semantic_contrast.py")})
    output.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        predictions = output / f"{arm}.npz"
        model_file, receipt_file = predictions.with_suffix(".pt"), predictions.with_suffix(".json")
        if receipt_file.exists():
            existing = read_json(receipt_file)
            require(existing["arm"] == arm and existing["sources"] == proof
                    and pins.pin(model_file) == existing["weights_sha256"] and pins.pin(predictions) == existing["predictions_sha256"],
                    "Existing region readout uses different training inputs")
            continue
        prototype = normalize(leave_photo_out(visual[arm], group))
        targets = target_photo, torch.from_numpy(prototype).to(device=device, dtype=torch.float32), target_text
        print("Training semantic target", arm, flush=True)
        photo, semantic, audit = train("visual-targets", eeg, queries, *targets, groups, EPOCHS, model_file, model_factory=fresh_model)
        atomic_file(predictions, lambda stream: np.savez_compressed(stream, embedding=semantic, photo_embedding=photo,
                                                                   row=held_rows, concept=category[labels[held_rows]]))
        pins.verify()
        save_json(receipt_file, dict(arm=arm, sources=proof, weights_sha256=pins.pin(model_file),
                                     predictions_sha256=pins.pin(predictions), **audit))
    comparisons = (("foreground", "background"), ("foreground", "whole"), ("foreground", "coarse"))
    score(output, benchmark, held_rows, labels, category, image, layout, proof, ARMS,
          comparisons=comparisons, report_name="region-readout-verified.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("view_bank", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(args.benchmark, args.view_bank, args.output, args.device)
