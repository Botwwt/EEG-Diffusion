"""Measure open-vocabulary readout from frozen D2 before image generation."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from paper_aggregate import bootstrap_ci, units
from paper_independent_naming import MODEL, WEIGHTS_SHA, vocabulary
from paper_render_cache import PinnedFiles, atomic_file, require, save_json
from paper_tangent_calibration import read_inputs


REGULARIZATION = (.001, .01, .1, 1.)


def train_coordinates(source, train, dev, proof, audit, pins, device):
    train_eeg, held_eeg, centers, receipts = [], [], [], {}
    for subject in range(1, 11):
        path = source / f"d2-train-dev-sub-{subject:02d}.npz"
        meta = path.with_suffix(".json")
        receipt = json.loads(meta.read_bytes())
        digest = pins.pin(path)
        require(receipt["data_sha256"] == digest == audit["conditions"][str(subject)]
                and receipt["subject"] == subject and receipt["weight_updates"] == 0
                and all(receipt[key] == proof[key] for key in ("split", "development", "labels")),
                "Frozen training conditions do not match the registered split")
        with np.load(path, allow_pickle=False) as values:
            require(np.array_equal(values["train_rows"], train) and np.array_equal(values["dev_rows"], dev),
                    "Training vectors use a different photograph order")
            x = F.normalize(torch.from_numpy(values["train_fused"]).to(device).float(), dim=-1)
            held = F.normalize(torch.from_numpy(values["dev_fused"]).to(device).float(), dim=-1)
        centers.append(x.mean(0))
        train_eeg.append(x)
        held_eeg.append(held)
        receipts[str(subject)] = {"array": digest, "metadata": pins.pin(meta), "checkpoint": receipt["checkpoint"]}
    return torch.stack(train_eeg), torch.stack(held_eeg), torch.stack(centers), receipts


def official_conditions(archive, device):
    with np.load(archive, allow_pickle=False) as data:
        require(np.array_equal(data["subjects"], np.arange(1, 11)) and np.array_equal(data["row"], np.arange(200)),
                "Official row order changed")
        test_x = F.normalize(torch.from_numpy(data["embedding"]).to(device).float(), dim=-1)
        test_photo = F.normalize(torch.from_numpy(data["target"]).to(device).float(), dim=-1)
        hits = data["feature_top1"] == 1
        require(hits.shape == (10, 200) and hits.sum() == 937, "Hit mask changed")
    return test_x, test_photo, hits


def ridge_system(x, y):
    people, n, dimension = x.shape
    mean_x, mean_y = x.mean(1, keepdim=True), y.mean(0)
    centered = x - mean_x
    design = centered.reshape(people * n, dimension)
    cov = (design.T @ design) / (people * n)
    target = centered.mean(0).T @ (y - mean_y) / n
    scale = cov.diagonal().mean()
    return cov, target, scale, mean_x, mean_y


def solve_ridge(system, strength):
    cov, target, scale, mean_x, mean_y = system
    weights = torch.linalg.solve(cov + strength * scale * torch.eye(len(cov), device=cov.device), target)
    return weights, mean_x, mean_y


def predict(x, fit):
    weights, mean_x, mean_y = fit
    return F.normalize((x - mean_x) @ weights + mean_y, dim=-1)


def select_ridge(x, y, concepts):
    # The development and official test folds play no role in selecting regularization.
    validation = torch.as_tensor(concepts % 10 == 0, device=x.device)
    require(bool(validation.any()) and bool((~validation).any()), "Empty training-only selection fold")
    system = ridge_system(x[:, ~validation], y[~validation])
    scores = [(predict(x[:, validation], solve_ridge(system, strength)) * y[validation]).sum(-1).mean().item()
              for strength in REGULARIZATION]
    strength = REGULARIZATION[int(np.argmax(scores))]
    return solve_ridge(ridge_system(x, y), strength), dict(strength=strength, inner_cosine=scores)


def rank_credit(logits, target, k):
    expected = np.take_along_axis(logits, np.broadcast_to(target, logits.shape[:-1])[..., None], axis=-1)
    better = (logits > expected).sum(-1)
    tied = (logits == expected).sum(-1)
    return ((k - better) / tied).clip(0, 1)


def measure(value, bank, inverse, target):
    logits = (value @ bank.T).cpu().numpy()[..., inverse]
    require(bool(np.isfinite(logits).all()), "Nonfinite independent naming scores")
    return np.stack((rank_credit(logits, target, 1), rank_credit(logits, target, 5)), axis=-1)


def encode_photographs(model, processor, archive, records, device, batch_size):
    features = []
    with zipfile.ZipFile(archive) as pictures:
        for start in range(0, len(records), batch_size):
            batch = []
            for record in records[start:start + batch_size]:
                raw = pictures.read(record["archive_member"])
                require(hashlib.sha256(raw).hexdigest() == record["image_sha256"],
                        "Photograph does not match the registered row")
                with Image.open(io.BytesIO(raw)) as image:
                    batch.append(image.convert("RGB"))
            pixels = processor(images=batch, return_tensors="pt").to(device)
            features.append(F.normalize(model.get_image_features(**pixels).float()))
            if start % (batch_size * 25) == 0:
                print("Encoded reference photographs", start + len(batch), len(records), flush=True)
    return torch.cat(features)


def frozen_visual_features(model_dir, assets, train, dev, plan, files, device):
    from transformers import AutoModel, AutoProcessor, __version__

    model_files = {p.name: files.pin(p) for p in sorted(model_dir.iterdir()) if p.is_file() and not p.name.startswith(".")}
    require(model_files.get("model.safetensors") == WEIGHTS_SHA, "Wrong frozen SigLIP2 model")
    proc = AutoProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, use_safetensors=True).to(device).eval()
    words, text, inverse, target, test_registry, test_proof = vocabulary(assets, files)
    _, _, dev_inverse, dev_target, dev_registry, dev_proof = vocabulary(assets, files, plan)
    require(np.array_equal(inverse, dev_inverse), "Development and official vocabulary order differs")
    with (assets / "train_photos.csv").open(encoding="utf-8-sig", newline="") as handle:
        registry = list(csv.DictReader(handle))
    test_words = {r["unique_id"] for r in test_registry}
    require(len(test_words) == 200 and all(registry[i]["unique_id"] not in test_words for i in train),
            "Official concepts overlap visual-only fitting")
    require([int(r["row0"]) for r in dev_registry] == dev.tolist(), "Development rows changed")
    train_archive, test_archive = assets / "training_images.zip", assets / "test_images.zip"
    proof = dict(model_files=model_files, transformers=__version__, test=test_proof, development=dev_proof,
                 training_images=files.pin(train_archive), test_images=files.pin(test_archive))
    with torch.inference_mode():
        prompts = [f"This is a photo of {item}." for item in text]
        length = model.config.text_config.max_position_embeddings
        bank = torch.cat([F.normalize(model.get_text_features(**proc(text=prompts[i:i + 32], padding="max_length",
                              max_length=length, truncation=True, return_tensors="pt").to(device)).float())
                          for i in range(0, len(prompts), 32)])
        train_y = encode_photographs(model, proc, train_archive, [registry[i] for i in train], device, 48)
        dev_y = encode_photographs(model, proc, train_archive, dev_registry, device, 48)
        test_y = encode_photographs(model, proc, test_archive, test_registry, device, 48)
    require(len(words) == 1854 and train_y.shape[0] == len(train) and test_y.shape[0] == 200,
            "Incomplete visual targets")
    return bank, inverse, (train_y, dev_y, test_y), (dev_target, target), proof


@torch.inference_mode()
def run(benchmark, model, output, device):
    files = PinnedFiles()
    source, train, dev, photo, concepts, dev_labels, _, _, inputs, audit = read_inputs(benchmark, files)
    x, dev_x, eeg_centers, receipts = train_coordinates(source, train, dev, inputs, audit, files, device)
    archive = benchmark / "token-rank-check/fused-r80/official-r80-fused.npz"
    meta = archive.with_suffix(".json")
    official = json.loads(meta.read_bytes())
    require(files.pin(archive) == official["output_sha256"] and official["split"] == inputs["split"]
            and official["reproduced_top1"] == 937 and sorted(official["per_subject"], key=int) == list(receipts)
            and all(r["checkpoint"] == official["per_subject"][p]["checkpoint"] for p, r in receipts.items()),
            "Official conditions differ from the archived 937-hit R80 cohort")
    test_x, test_photo, hits = official_conditions(archive, device)
    plan = benchmark / "verified/data/prep/paper_development.json"
    bank, inverse, (train_y, dev_y, test_y), (dev_target, test_target), model_proof = frozen_visual_features(
        model, benchmark / "inputs/image_set", train, dev, plan, files, device)
    photo_x = F.normalize(torch.tensor(photo[train], device=device), dim=-1)
    pfit, photo_hyper = select_ridge(photo_x[None], train_y, concepts)
    efit, eeg_hyper = select_ridge(x, train_y, concepts)
    require(torch.allclose(eeg_centers[:, None], efit[1]), "Changed per-participant training center")
    dev_photo = F.normalize(torch.tensor(photo[dev], device=device), dim=-1)

    # Rolling within each participant creates a no-retraining, wrong-stimulus control.
    valid_dev_shift = next(k for k in range(1, len(dev)) if not np.any(np.roll(dev_labels, k) == dev_labels))
    held = {
        "photo_reference": measure(dev_y, bank, inverse, dev_target),
        "photo_fused_positive": measure(predict(dev_photo[None], pfit)[0], bank, inverse, dev_target),
        "image_bridge": measure(predict(dev_x, pfit), bank, inverse, dev_target),
        "eeg_bridge": measure(predict(dev_x, efit), bank, inverse, dev_target),
        "eeg_bridge_wrong": measure(predict(dev_x.roll(valid_dev_shift, dims=1), efit), bank, inverse, dev_target),
    }
    official_names = {
        "photo_reference": measure(test_y, bank, inverse, test_target),
        "photo_fused_positive": measure(predict(test_photo[None], pfit)[0], bank, inverse, test_target),
        "image_bridge": measure(predict(test_x, pfit), bank, inverse, test_target),
        "eeg_bridge": measure(predict(test_x, efit), bank, inverse, test_target),
        "eeg_bridge_wrong": measure(predict(test_x.roll(1, dims=1), efit), bank, inverse, test_target),
    }
    results = {}
    for panel, observed, labels in (("dev", held, dev_labels), ("official", official_names, test_target)):
        summary = {name: value.reshape(-1, 2).sum(0).tolist() for name, value in observed.items()}
        comparisons = {}
        for a, b in (("eeg_bridge", "image_bridge"), ("eeg_bridge", "eeg_bridge_wrong")):
            delta = observed[a] - observed[b]
            lo, hi = bootstrap_ci(units(delta, labels, panel))
            comparisons[f"{a} minus {b}"] = np.stack((delta.mean((0, 1)), lo, hi), axis=1).tolist()
        results[panel] = {"credits": summary, "paired": comparisons}
    results["official"]["photo_hits"] = {
        name: (value[hits] if value.ndim == 3 else value[np.nonzero(hits)[1]]).sum(0).tolist()
        for name, value in official_names.items()
    }
    output.mkdir(parents=True, exist_ok=True)
    observed = output / "siglip-fused-observations.npz"
    atomic_file(observed, lambda stream: np.savez_compressed(stream, hit_mask=hits, official_row=np.arange(200),
                dev_row=dev, **{f"official_{k}": v for k, v in official_names.items()},
                **{f"dev_{k}": v for k, v in held.items()}))
    bridge = output / "siglip-fused-evaluation-bridge.pt"
    atomic_file(bridge, lambda stream: torch.save(
        dict(readout=tuple(t.cpu() for t in pfit), text_bank=bank.cpu(), names=torch.as_tensor(inverse),
             official_target=torch.as_tensor(test_target)), stream))
    inputs.update({"official_archive": files.pin(archive), "official_receipt": files.pin(meta)})
    report = dict(model=MODEL, vision_and_text=model_proof, input=inputs, evidence=receipts, observations=files.pin(observed),
                  evaluation_bridge_sha256=files.pin(bridge),
                  code=files.pin(Path(__file__)), selection=dict(photo=photo_hyper, eeg=eeg_hyper), n_train=len(train),
                  metrics=("name_top1", "name_top5"), official_hit_rows=int(hits.sum()), dev_wrong_shift=valid_dev_shift,
                  results=results)
    files.verify()
    save_json(output / "siglip-fused-probe.json", report)
    print(json.dumps(results, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    arguments = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark", "model", "output"):
        arguments.add_argument(name, type=Path)
    arguments.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    options = arguments.parse_args()
    torch.set_num_threads(4)
    run(options.benchmark, options.model, options.output, options.device)
