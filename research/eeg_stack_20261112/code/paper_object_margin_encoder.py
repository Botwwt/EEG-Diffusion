"""Train the native EEG encoder to separate visually similar objects without changing inference."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_render_cache import require


MODES = ("object-margin-uniform", "object-margin-hard")
SEED = 20261125
TEMPERATURE = .07


def visual_transfer(fused, dino):
    """Estimate the object-sensitive directions of the same feature consumed by the generator."""
    center, visual_center = fused.mean(0), dino.mean(0)
    centered = fused - center
    gram = centered.T @ centered / len(centered)
    gram.diagonal().add_(.1 * gram.diagonal().mean())
    weight = torch.linalg.solve(gram, centered.T @ (dino - visual_center) / len(centered))
    return center, visual_center, weight


def confuser_log_prior(photos, prototypes, labels, count, multiplier):
    require(0 < count < len(prototypes) and multiplier >= 1., "Invalid negative weighting")
    resemblance = (photos @ prototypes.T).scatter(1, labels[:, None], float("-inf"))
    neighbors = resemblance.topk(count, dim=1).indices
    prior = torch.full_like(resemblance, -math.log1p(count * (multiplier - 1) / (len(prototypes) - 1)))
    prior.scatter_add_(1, neighbors, torch.full_like(neighbors, math.log(multiplier), dtype=prior.dtype))
    return prior.scatter(1, labels[:, None], float("-inf"))


def object_loss(predicted, photos, prototypes, labels, log_prior):
    condition = F.normalize(predicted)
    positive = (condition * photos).sum(1, keepdim=True) / TEMPERATURE
    logits = (condition @ prototypes.T / TEMPERATURE + log_prior).scatter(1, labels[:, None], positive)
    return F.cross_entropy(logits, labels)


def fit(subject):
    from common import PREP, RUNS
    from paper_identity_bank import bank_root
    from paper_instance_residual import make_encoder
    from paper_object_tokens import BATCH, EPOCHS, epoch_batches, group_index
    from paper_oof_conditions import registered_rows
    from paper_render import physical_eeg
    from paper_render_cache import PinnedFiles, atomic_file, save_json, sha

    require(RUNS.name == f"personal-{subject:02d}", "Use the registered personal encoder directory")
    plan = PREP / "paper_development.json"
    require(subject in json.loads(plan.read_bytes())["subjects"], "Use only the preregistered development participants")
    native = RUNS / f"enc/native/sub-{subject:02d}/fold0_seed2025/brain_model.pt"
    root = RUNS / f"enc/object_tokens/sub-{subject:02d}"
    root.mkdir(parents=True, exist_ok=True)
    pins = PinnedFiles()
    bank_path = bank_root() / "training-photo-dino.pt"
    image_path = PREP / "F_train.npy"
    raw, eeg_proof = physical_eeg(subject, "train", pins)
    proof = dict(eeg=eeg_proof, native=pins.pin(native), photographs=pins.pin(image_path),
                 dino=pins.pin(bank_path), split=pins.pin(PREP / "paper_split.json"),
                 development=pins.pin(plan), code=pins.pin(Path(__file__)),
                 batch_code=pins.pin(Path(__file__).with_name("paper_object_tokens.py")))
    pending = []
    for mode in MODES:
        file = root / f"{mode}.pt"
        receipt = file.with_suffix(".json")
        if file.is_file() and receipt.is_file():
            previous = json.loads(receipt.read_bytes())
            require(previous["proof"] == proof and previous["checkpoint_sha256"] == sha(file)
                    and previous["subject"] == subject and previous["mode"] == mode,
                    "Existing encoder used different supervision")
        else:
            pending.append(mode)
    if not pending:
        return

    rows, classes, _ = registered_rows(PREP)
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    require(bank["proof"]["split"] == proof["split"] and np.array_equal(bank["rows"].numpy(), rows)
            and np.array_equal(bank["concepts"].numpy()[bank["labels"].numpy()], classes),
            "The original EEG, image and object teachers disagree about the training rows")
    device = torch.device("cuda")
    eeg = torch.cat([torch.from_numpy(raw[ix].copy()).to(device=device, dtype=torch.float32).mean(1)
                     for ix in np.array_split(rows, math.ceil(len(rows) / 128))])
    photo = torch.from_numpy(np.load(image_path, mmap_mode="r", allow_pickle=False)[rows].copy()).float().to(device)
    dino, labels, sums = (bank[key].to(device) for key in ("image_features", "labels", "sums"))
    require(tuple(photo.shape) == (len(rows), 1024) and tuple(dino.shape) == (len(rows), 384)
            and tuple(sums.shape) == (1323, 384) and torch.isfinite(photo).all() and torch.isfinite(dino).all(),
            "Unexpected original photograph representations")
    ids, ordered = group_index(labels)
    require(torch.equal(ids, labels), "Image category indexes are not contiguous")
    center, visual_center, transfer = visual_transfer(photo, dino)
    exact = F.normalize(dino - visual_center)
    prototypes = F.normalize(sums / ordered.shape[1] - visual_center)
    spread = (photo - center).square().mean().clamp_min(1e-6)
    original = torch.load(native, map_location="cpu", weights_only=True)
    steps = EPOCHS * ordered.shape[1] * math.ceil(len(ordered) / BATCH)

    for mode in pending:
        negative_prior = confuser_log_prior(exact, prototypes, labels, 32,
                                           9. if mode == "object-margin-hard" else 1.)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        rng = torch.Generator(device=device).manual_seed(SEED)
        brain = make_encoder(device).float()
        brain.load_state_dict(original)
        brain.train()
        optimizer = torch.optim.AdamW(brain.parameters(), lr=6e-5, betas=(.9, .995), weight_decay=.05)
        decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=5e-6)
        history = []
        for epoch in range(EPOCHS):
            totals = np.zeros(2)
            for batch in epoch_batches(ordered, rng):
                predicted = brain.backbone(eeg[batch])
                target = photo[batch]
                logits = F.normalize(predicted - center) @ F.normalize(target - center).T / TEMPERATURE
                diagonal = torch.arange(len(batch), device=device)
                core = .5 * (F.cross_entropy(logits, diagonal) + F.cross_entropy(logits.T, diagonal))
                core = core + (predicted - target).square().mean() / spread
                auxiliary = object_loss((predicted - center) @ transfer, exact[batch], prototypes,
                                        labels[batch], negative_prior[batch])
                optimizer.zero_grad(set_to_none=True)
                (core + .25 * auxiliary).backward()
                torch.nn.utils.clip_grad_norm_(brain.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                decay.step()
                totals += len(batch) * np.array([float(core.detach()), float(auxiliary.detach())])
            history.append((totals / len(eeg)).tolist())
            print("object margin", subject, mode, epoch + 1, history[-1], flush=True)
        pins.verify()
        file = root / f"{mode}.pt"
        atomic_file(file, lambda stream: torch.save({k: v.cpu() for k, v in brain.state_dict().items()}, stream))
        save_json(file.with_suffix(".json"), dict(mode=mode, subject=subject, proof=proof, checkpoint_sha256=sha(file),
                  photos=len(rows), steps=steps, seed=SEED, loss=history, confusers=32,
                  confuser_weight=9. if mode == "object-margin-hard" else 1.))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, required=True)
    fit(parser.parse_args().subject)
