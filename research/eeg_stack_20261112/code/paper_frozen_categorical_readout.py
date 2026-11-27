"""Retest the same categorical head from frozen shared-D2 fusion features."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_categorical_readout import concepts, decode_eeg, evaluate, evaluation_sources, quantize, scores, train
from paper_fusion_siglip_probe import official_conditions, train_coordinates
from paper_layout_mixture_composition import native_conditions
from paper_raw_semantic_readout import leave_photo_out, normalize
from paper_render_cache import PinnedFiles, atomic_file, require, save_json
from paper_semantic_contrast import data_registry
from paper_tangent_calibration import read_inputs
from paper_temporal_readout import held_concepts


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 384), nn.GELU(), nn.LayerNorm(384))
        self.photo = nn.Linear(384, 1024)
        self.identity = nn.Linear(384, 1024)

    def forward(self, estimate):
        hidden = self.trunk(estimate)
        return tuple(F.normalize(head(hidden).float(), dim=-1) for head in (self.photo, self.identity))


def scalar_bins(target):
    probabilities = torch.tensor([.25,.5,.75], device=target.device)
    boundaries = torch.quantile(target, probabilities, dim=0).T
    ids = (target[..., None] > boundaries).sum(-1)
    mask = F.one_hot(ids, 4).to(target)
    counts = mask.sum(0)
    require(bool((counts > 0).all()), "All four bins per dimension must be represented in training")
    levels = (mask * target[..., None]).sum(0) / counts
    return ids, boundaries, levels


@torch.inference_mode()
def decode_scalar(model, values, levels):
    output = [model(chunk) for chunk in values.flatten(0,1).split(256)]
    _, continuous, logits = (torch.cat(parts) for parts in zip(*output, strict=True))
    directions = logits.softmax(-1)
    mean = (directions * levels).sum(-1)
    mode = levels.gather(1, directions.argmax(-1).T).T
    return {key: F.normalize(vector.reshape(*values.shape[:2], -1), dim=-1) for key, vector in
            (("continuous", continuous), ("posterior-mean", mean), ("posterior-mode", mode))}


def run(benchmark, root, epochs, quantile=False):
    device = "cuda"
    pins = PinnedFiles()
    _, rows, dev, labels, category, photo, names, _, vision = data_registry(benchmark)
    held = held_concepts(benchmark, rows, dev, labels)
    source, train_rows, development_rows, _, _, _, _, _, common, audit = read_inputs(benchmark, pins)
    require(np.array_equal(train_rows, rows) and np.array_equal(development_rows, dev), "The frozen training split differs")
    reference = root / "readout.json"
    raw_model = root / "categorical-readout.pt"
    registered = json.loads(reference.read_bytes())
    require(pins.pin(raw_model) == registered["weights"] and registered["fit"]["visual"] == vision,
            "The reference codebook or training split changed")
    fixed = torch.load(raw_model, map_location=device, weights_only=True)
    center, books = fixed["center"], fixed["codebooks"]
    del fixed
    train_x, dev_x, _, proof = train_coordinates(source, rows, dev, common, audit, pins, device)
    path = benchmark / "token-rank-check/fused-r80/official-r80-fused.npz"
    seal = json.loads(path.with_suffix(".json").read_bytes())
    require(seal["output_sha256"] == pins.pin(path) and seal["split"] == common["split"]
            and seal["repeats"] == [list(range(i,i+4)) for i in range(0,80,4)], "Invalid frozen R80 archive")
    official_x, _, _ = official_conditions(path, device)
    for subject in range(1, 11):
        key = str(subject)
        info = json.loads((source / f"d2-train-dev-sub-{subject:02d}.json").read_bytes())
        r80 = seal["per_subject"][key]
        require(info["condition"] == r80["condition"] and info["checkpoint"] == proof[key]["checkpoint"] == r80["checkpoint"],
                "The train and test checkpoints differ")
        for condition, actual in (("train", info["eeg"]), ("test", r80["eeg"])):
            expected = registered["fit"]["physical"][key][condition]
            require(all(actual[k] == value for k, value in expected.items()), "Physical trials differ from the raw readout")
    groups = np.unique(labels[rows], return_inverse=True)[1]
    ordered = np.argsort(groups, kind="stable").reshape(-1, 10)
    image = torch.from_numpy(photo[rows]).float().to(device)
    prototype = torch.from_numpy(normalize(leave_photo_out(photo[rows], groups))).to(image)
    text = torch.from_numpy(names[labels[rows]]).to(image)
    if quantile:
        ids, cuts, levels = scalar_bins(prototype)
        settings = dict(code_shape=(1024,4))
    else:
        ids = quantize(prototype-center, books)
        settings = {}
    model, schedule = train(train_x, ordered, image, prototype, text, ids, epochs, encoder_factory=Adapter, **settings)
    output = root / ("quantile-frozen-d2" if quantile else "frozen-d2")
    output.mkdir(parents=True, exist_ok=True)
    weights = output / "weights.pt"
    atomic_file(weights, lambda f: torch.save({key: v.cpu() for key,v in model.state_dict().items()}, f))
    native, native_proof = native_conditions(source / "horizon-readout-20261127/native-h14",
        source / "background-invariant-20261127/guide", held, registered["fit"], pins)
    visual = source / "alignvis-20261127/alignvis-image-teachers.npz"
    authority = json.loads(visual.with_suffix(".json").read_bytes())
    test_csv = benchmark / "inputs/image_set/test_photos.csv"
    require(pins.pin(visual) == authority["array_sha256"] and pins.pin(test_csv) == authority["photo_registries"]["test"],
            "Changed scoring reference")
    with np.load(visual, allow_pickle=False) as values:
        official_names = values["test_word"]
        require(np.array_equal(official_names, concepts(test_csv)) and set(official_names).isdisjoint(category[labels[rows]]),
                "Unexpected official stimuli")
        official_photo = torch.from_numpy(normalize(values["test/h14"])).to(device)
    dev_photo = torch.from_numpy(photo[dev]).to(device)
    evaluator = evaluation_sources(benchmark, rows, image, np.r_[category[labels[dev]], official_names], pins, device)
    panels = (("dev", "held", slice(200), category[labels[dev]], dev_photo),
              ("official", "official", slice(200), official_names, official_photo))
    result = {}
    for key, estimate in (("held", dev_x), ("official", official_x)):
        if quantile:
            result[key] = decode_scalar(model, estimate, levels)
        else:
            _, result[key], _ = decode_eeg(model, estimate[:, :, None], center, books)
    summary, results = evaluate(output, native, result, panels, evaluator, pins)
    oracle = {}
    if quantile:
        for panel, _, _, label, target in panels:
            test_ids = (target[..., None] > cuts).sum(-1)
            decoded = F.normalize(levels.gather(1, test_ids.T).T, dim=-1)
            h = scores(decoded[None], target, label, evaluator).sum((0,1)).tolist()
            expected = scores(target[None], target, label, evaluator).sum((0,1)).tolist()
            oracle[panel] = dict(quantized=h, full=expected)
        fields = output / "ordinal-visual-levels.pt"
        atomic_file(fields, lambda f: torch.save(dict(cuts=cuts.cpu(), levels=levels.cpu()), f))
        oracle["bin_parameters"] = pins.pin(fields)
        print("True photograph ordinal controls", json.dumps(oracle), flush=True)
    save_json(output / "readout.json", dict(raw_readout=pins.pin(reference), frozen_train=proof,
        frozen_test=pins.pin(path), sources=native_proof, weights=pins.pin(weights), code=pins.pin(Path(__file__)),
        shared_code=pins.pin(Path(__import__("paper_categorical_readout").__file__)),
        training=schedule, data=results, results=summary, oracle=oracle))
    pins.verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--quantile", action="store_true")
    args = parser.parse_args()
    run(args.benchmark, args.root, args.epochs, args.quantile)
