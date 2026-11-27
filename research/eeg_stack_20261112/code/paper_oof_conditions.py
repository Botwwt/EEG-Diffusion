"""Construct generator training conditions without letting the EEG encoders see their object classes."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


SOURCES = (1, 2, 3, 4)
HELD_OUT = (5, 6, 7, 8, 9, 10)
SEED = 20261125


def require(condition, message):
    if not condition:
        raise ValueError(message)


def concept_splits(labels, folds):
    require(labels.ndim == 1 and folds.shape == labels.shape and np.array_equal(np.unique(folds), [1, 2, 3, 4]),
            "Expected four complete training folds")
    splits = []
    for held in (1, 2, 3, 4):
        fit = np.flatnonzero(folds != held)
        predict = np.flatnonzero(folds == held)
        require(not np.intersect1d(labels[fit], labels[predict]).size, "A concept spans a fitting and predicted fold")
        splits.append((held, fit, predict))
    return splits


def condition_stats(pred, target):
    require(pred.shape == target.shape and pred.ndim == 2 and torch.isfinite(pred).all() and torch.isfinite(target).all(),
            "Expected finite aligned visual conditions")
    center = target.mean(0)
    radius = (target - center).square().sum(1).mean().sqrt()
    require(radius > 0, "Degenerate visual target radius")
    centered = pred - center
    return {
        "mean_target_cos": float(F.cosine_similarity(pred, target).mean()),
        "mean_centered_target_cos": float(F.cosine_similarity(centered, target - center).mean()),
        "radius_over_target_rms": float(centered.square().sum(1).mean().sqrt() / radius),
        "rmse": float((pred - target).square().mean().sqrt()),
    }


def study_root():
    return Path(os.environ["EEG_CROSSFIT_ROOT"])


def registered_rows(prep):
    from common import concept_folds

    split = json.loads((prep / "paper_split.json").read_bytes())
    rows = np.asarray(split["train_rows"], dtype=np.int64)
    labels = np.load(prep / "train_concept.npy", allow_pickle=False)
    ok = np.load(prep / "lat512_train_ok.npy", allow_pickle=False)
    require(labels.shape == ok.shape == (16540,), "Unexpected photograph support")
    folds = concept_folds(1654)[labels]
    require(len(rows) == 13230 and np.array_equal(rows, np.flatnonzero((folds != 0) & ok)),
            "Only the registered 1,323 training concepts may supply EEG and target images")
    require(not np.intersect1d(rows, split["fresh_rows"]).size, "A fresh photograph appears in training")
    return rows, labels[rows], folds[rows]


def fitted_native(subject, device):
    from common import RUNS
    from paper_instance_residual import make_encoder

    require(RUNS.name == f"personal-{subject:02d}", "Select this participant's native training directory")
    checkpoint = RUNS / f"enc/native/sub-{subject:02d}/fold0_seed2025/brain_model.pt"
    model = make_encoder(device).float().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.requires_grad_(False), checkpoint


@torch.no_grad()
def native_features(model, means):
    return torch.cat([model.backbone(x).float().cpu() for x in means.split(128)])


def prepare(subject):
    from common import EMB_KEYS, PREP, PRIOR
    from main.models_adapter import FusionEncoderModel
    from paper_instance_residual import digest_array, train_shadow
    from paper_render import physical_eeg
    from paper_render_cache import PinnedFiles, atomic_file, save_json

    rows, labels, folds = registered_rows(PREP)
    splits = concept_splits(labels, folds)
    device = torch.device("cuda")
    files = PinnedFiles()
    raw, receipt = physical_eeg(subject, "train", files)
    model, native = fitted_native(subject, device)
    metadata = dict(subject=subject, eeg=receipt, native=files.pin(native), code=files.pin(Path(__file__)),
                    shadow_code=files.pin(Path(__file__).with_name("paper_instance_residual.py")),
                    split=files.pin(PREP / "paper_split.json"), images=files.pin(PREP / "F_train.npy"),
                    visual_keys=files.pin(PREP / "emb_train.pt"),
                    fusion=files.pin(PRIOR / "fusion_encoder/diffusion_pytorch_model.safetensors"))
    root = study_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"conditions-sub-{subject:02d}.pt"
    if path.is_file():
        old = torch.load(path, map_location="cpu", weights_only=True)
        require(old["sources"] == metadata and np.array_equal(old["rows"].numpy(), rows)
                and np.array_equal(old["folds"].numpy(), folds), "Existing condition cache uses different data")
        print("Verified existing", path, flush=True)
        return
    means = torch.empty(len(rows), 17, 250, device=device)
    for start in range(0, len(rows), 128):
        sl = slice(start, start + 128)
        means[sl] = torch.from_numpy(raw[rows[sl]].copy()).to(device=device, dtype=torch.float32).mean(1)
    native_seen = native_features(model, means)
    del model
    ordered = torch.from_numpy(rows)
    embeddings = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    require(set(embeddings) == set(EMB_KEYS), "Missing pinned H/14, B/32 or VAE training supervision")
    embeddings = {key: value[ordered].float().to(device) for key, value in embeddings.items()}
    teacher = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.to(device)
    teacher.train().requires_grad_(False)
    for layer in teacher.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = .1
    out = torch.empty(len(rows), 1024)
    seen = torch.empty_like(out)
    log = []
    for held, fit, predict in splits:
        shadow = train_shadow(means, embeddings, torch.as_tensor(fit, device=device), teacher, 2025 + held, device)
        out[predict] = native_features(shadow, means[predict])
        # Cycle which shadow supplies each seen-concept prediction. Both banks have
        # the same four encoders, fitting sizes, initialization seeds and loss.
        seen_fold = (held - 2) % 4 + 1
        seen_rows = np.flatnonzero(folds == seen_fold)
        require(np.isin(seen_rows, fit).all(), "Seen control must have trained on the predicted rows")
        seen[seen_rows] = native_features(shadow, means[seen_rows])
        log.append({"fold": held, "trained": len(fit), "predicted": len(predict),
                    "control_seen_fold": seen_fold, "control_rows": len(seen_rows),
                    "training_rows_sha256": digest_array(rows[fit]), "predicted_rows_sha256": digest_array(rows[predict])})
        print("cross-fit", subject, held, log[-1], flush=True)
        del shadow
    real = torch.from_numpy(np.load(PREP / "F_train.npy", mmap_mode="r")[rows].copy()).float()
    require(real.shape == seen.shape == out.shape == (len(rows), 1024), "Unexpected visual feature axes")
    stats = {"seen": condition_stats(seen, real), "out_of_fold": condition_stats(out, real),
             "native_seen": condition_stats(native_seen, real)}
    files.verify()
    packet = dict(rows=ordered, folds=torch.from_numpy(folds.copy()), seen_source_fold=torch.from_numpy(folds % 4 + 1),
                  seen=seen, out_of_fold=out,
                  stats=stats, sources=metadata, split_audits=log)
    atomic_file(path, lambda stream: torch.save(packet, stream))
    save_json(path.with_suffix(".json"), dict(stats=stats, sources=metadata, split_audits=log))
    print("condition distribution", subject, stats, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", choices=SOURCES, type=int, required=True)
    prepare(p.parse_args().subject)
