"""Fit one EEG encoder to generator image tokens shared by different photos of an object."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from paper_instance_residual import make_encoder

MODES = ("token-photo", "token-object", "token-reliable")
SEED = 20261125
EPOCHS, BATCH = 6, 256


def sha(path):
    with Path(path).open("rb") as reader:
        return hashlib.file_digest(reader, "sha256").hexdigest()


def group_index(labels):
    # Do not rely on the acquisition order of the training photographs.
    _, inverse, counts = torch.unique(labels, sorted=True, return_inverse=True, return_counts=True)
    if len(counts) < 3 or not torch.all(counts == 10):
        raise ValueError("Expected complete ten-photograph training concepts")
    return inverse, torch.argsort(inverse, stable=True).reshape(-1, 10)


def cross_photo_targets(tokens, inverse, ordered, gain):
    sums = tokens[ordered].sum(1)
    others = (sums[inverse] - tokens) / (ordered.shape[1] - 1)
    return others + gain.reshape(1, -1, 1) * (tokens - others)


def reliability(eeg, tokens, ordered):
    # Each held-out fold contains complete training concepts. Predict compressed visual
    # residuals from raw EEG, independently of the pretrained neural encoder.
    device = tokens.device
    n_concepts = len(ordered)
    n = len(tokens)
    x = eeg[:, :, :200].reshape(n, 17, 40, 5).mean(-1).flatten(1)[ordered]
    x = (x - x.mean(1, keepdim=True)).double()
    g = torch.Generator(device="cpu").manual_seed(SEED)
    # Uniform coordinates avoid a dense 8192 x 8192 covariance and treat the four IP tokens separately.
    dims = torch.randperm(tokens.shape[-1], generator=g)[:128].to(device)
    y = tokens[:, :, dims][ordered]
    y = (y - y.mean(1, keepdim=True)).double()
    folds = torch.empty(n_concepts, dtype=torch.long)
    folds[torch.randperm(n_concepts, generator=g)] = torch.arange(n_concepts) % 3
    folds = folds.to(device)
    err, energy = (torch.zeros(tokens.shape[1], device=device, dtype=torch.float64) for _ in range(2))
    for held in range(3):
        tr, va = folds != held, folds == held
        train_x, val_x = x[tr].flatten(0, 1), x[va].flatten(0, 1)
        train_y, val_y = y[tr].flatten(0, 1), y[va].flatten(0, 1)
        scale = train_x.std(0).clamp_min(1e-6)
        train_x, val_x = train_x / scale, val_x / scale
        gram = train_x.T @ train_x / len(train_x)
        gram.diagonal().add_(1.)
        weight = torch.linalg.solve(gram, train_x.T @ train_y.flatten(1) / len(train_x))
        predict = (val_x @ weight).view_as(val_y)
        err += (predict - val_y).square().sum((0, 2))
        energy += val_y.square().sum((0, 2))
    if not torch.all(energy > 0):
        raise ValueError("Degenerate within-object visual token")
    r2 = 1 - err / energy
    return r2.clamp(0., 1.).sqrt().float(), r2.tolist()


def epoch_batches(ordered, g, batch_size=BATCH):
    # Every photograph appears once. Minibatches contain one photograph per distinct concept.
    device = ordered.device
    order = torch.rand(ordered.shape, generator=g, device=device).argsort(dim=1)
    photos = ordered.gather(1, order)
    for col in photos.T:
        yield from col[torch.randperm(len(col), generator=g, device=device)].split(batch_size)


def fit(subject, modes):
    from common import PREP, PRIOR, RUNS
    from main.models_adapter import FusionEncoderModel, IPProjectionModel
    from paper_training import check_eeg

    if RUNS.name != f"personal-{subject:02d}":
        raise ValueError("Expected the personal fitting directory for the EEG participant")
    native = RUNS / f"enc/native/sub-{subject:02d}/fold0_seed2025/brain_model.pt"
    dest = RUNS / f"enc/object_tokens/sub-{subject:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    split = PREP / "paper_split.json"
    rows = np.array(json.loads(split.read_bytes())["train_rows"], dtype=np.int64)
    labels = np.load(PREP / "train_concept.npy", allow_pickle=False)[rows]
    from common import concept_folds
    if len(rows) != 13230 or not np.all(concept_folds(1654)[labels] != 0) or len(np.unique(labels)) != 1323:
        raise ValueError("Training is limited to the registered 1,323 concepts")
    eeg_path = PREP / f"sub-{subject:02d}_train.npy"
    proof = dict(native=sha(native), code=sha(__file__), split=sha(split), concept=sha(PREP/"train_concept.npy"),
                 eeg=sha(eeg_path), F=sha(PREP/"F_train.npy"),
                 adapter=sha(PRIOR/"fusion_encoder/diffusion_pytorch_model.safetensors"),
                 projection=sha(PRIOR/"proj/diffusion_pytorch_model.safetensors"))
    pending = []
    for mode in modes:
        file = dest / f"{mode}.pt"
        receipt = file.with_suffix(".json")
        if file.is_file() and receipt.is_file():
            previous = json.loads(receipt.read_bytes())
            if previous["sources"] != proof or previous["checkpoint_sha256"] != sha(file):
                raise ValueError(f"Existing fit has different sources for {mode}")
        else:
            pending.append(mode)
    if not pending:
        return
    check_eeg((subject,), "train")
    device = torch.device("cuda")
    ids, ordered = group_index(torch.as_tensor(labels, device=device))
    raw = np.load(eeg_path, mmap_mode="r")
    # Take the average in float32, matching the registered native renderer.
    eeg = torch.cat([torch.from_numpy(raw[r].copy()).to(device=device, dtype=torch.float32).mean(1)
                     for r in np.array_split(rows, math.ceil(len(rows)/128))])
    image = torch.from_numpy(np.load(PREP/"F_train.npy", mmap_mode="r")[rows].copy()).to(device)
    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.cuda().float().eval().requires_grad_(False)
    proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").cuda().float().eval().requires_grad_(False)
    with torch.no_grad():
        tokens = torch.cat([proj(adapter(f.float())) for f in image.split(128)])
        if tokens.shape != (len(rows), 4, 2048) or not torch.isfinite(tokens).all():
            raise ValueError("Unexpected visual teacher output")
        gain, cv = reliability(eeg, tokens, ordered)
        center = tokens.mean(0)
        variance = (tokens - center).square().mean((0, 2)).clamp_min(1e-6)
    native_state = torch.load(native, map_location="cpu", weights_only=True)
    for mode in pending:
        selected = {"token-photo": torch.ones_like(gain), "token-object": torch.zeros_like(gain),
                    "token-reliable": gain}[mode]
        target = cross_photo_targets(tokens, ids, ordered, selected)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        enc = make_encoder(device)
        enc.load_state_dict(native_state)
        enc.train()
        opt = torch.optim.AdamW(enc.parameters(), lr=8e-5, betas=(.9, .995), weight_decay=.05)
        g = torch.Generator(device=device).manual_seed(SEED)
        steps = EPOCHS * ordered.shape[1] * math.ceil(len(ordered) / BATCH)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-6)
        log = []
        for epoch in range(EPOCHS):
            loss_sum = 0.
            for ix in epoch_batches(ordered, g):
                # The frozen, differentiable projector puts the loss in the exact space consumed by SDXL.
                pred = proj(adapter(enc.backbone(eeg[ix]).float()))
                tgt = target[ix]
                score = F.normalize((pred-center).flatten(1)) @ F.normalize((tgt-center).flatten(1)).T / .07
                diag = torch.arange(len(ix), device=device)
                contrast = .5 * (F.cross_entropy(score, diag) + F.cross_entropy(score.T, diag))
                calibration = ((pred-tgt).square().mean(-1) / variance).mean()
                loss = contrast + calibration
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                sched.step()
                loss_sum += float(loss.detach()) * len(ix)
            log.append(loss_sum / len(rows))
            print(mode, subject, epoch+1, log[-1], flush=True)
        file = dest / f"{mode}.pt"
        with tempfile.NamedTemporaryFile(dir=dest, suffix=".pt", delete=False) as tmp:
            tmp_name = Path(tmp.name)
        try:
            torch.save({k: v.cpu() for k, v in enc.state_dict().items()}, tmp_name)
            tmp_name.replace(file)
        finally:
            tmp_name.unlink(missing_ok=True)
        meta = dict(mode=mode, subject=subject, sources=proof, checkpoint_sha256=sha(file), seed=SEED, steps=steps,
                    photos=len(rows), concepts=len(ordered), epochs=EPOCHS, loss=log, raw_cv_r2=cv,
                    token_gains=selected.tolist(), frozen_generator=True, training_supervision="training concept groups and visual features")
        receipt = file.with_suffix(".json")
        tmp_receipt = receipt.with_suffix(f".{os.getpid()}.tmp")
        tmp_receipt.write_text(json.dumps(meta, indent=1, allow_nan=False))
        tmp_receipt.replace(receipt)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", type=int, required=True, choices=range(1, 11))
    p.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    a = p.parse_args()
    fit(a.subject, a.modes)
