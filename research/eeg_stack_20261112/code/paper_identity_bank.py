"""Fit photograph-independent object prototypes using only registered training images."""

import json
import os
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from paper_oof_conditions import require, study_root


def class_sums(features, labels):
    populations = torch.bincount(labels)
    require(len(populations) >= 2 and int(populations.min()) >= 2, "Each training object requires another photo")
    return features.new_zeros(len(populations), features.shape[1]).index_add_(0, labels, features)


def object_logits(generated, own_photo, labels, sums, mode):
    require(mode in ("identity", "photo"), "Use the whole object or exact photo as the positive")
    positives = F.normalize(sums[labels] - own_photo) if mode == "identity" else own_photo
    matched = (generated * positives).sum(1, keepdim=True)
    return (generated @ F.normalize(sums).T).scatter(1, labels[:, None], matched)


def bank_root():
    return study_root() / "object-identity"


def visual_model(path):
    from transformers import Dinov2Model

    model = Dinov2Model.from_pretrained(str(path), local_files_only=True, use_safetensors=True).cuda().float()
    require((model.config.patch_size, model.config.hidden_size, model.config.num_hidden_layers) == (14, 384, 12),
            "Unexpected DINOv2 training teacher")
    return model.eval().requires_grad_(False)


@torch.no_grad()
def prepare():
    from common import PREP, ROOT
    from paper_dino import MODEL_FILES, TRAIN_ZIPS
    from paper_render_cache import PinnedFiles, atomic_file, save_json
    from paper_rgb_object import dino_cls, read_photo, registered_photos

    root = bank_root()
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "training-photo-dino.pt"
    archive = Path(os.environ["ARCHIVE"]) / "assets/things_images_v254/training_images.zip"
    dino_path = ROOT / "models/paper-dinov2-small"
    pins = PinnedFiles()
    proof = dict(archive=pins.pin(archive), split=pins.pin(PREP / "paper_split.json"),
                 dino={key: pins.pin(dino_path / key) for key in MODEL_FILES},
                 photo_index=pins.pin(PREP / "train_ids.json"), eeg_index=pins.pin(PREP / "eeg_train_ids.json"),
                 code=pins.pin(Path(__file__)), preprocess=pins.pin(Path(__file__).with_name("paper_rgb_object.py")))
    require(proof["archive"] in TRAIN_ZIPS and proof["dino"] == MODEL_FILES, "Unexpected pinned imagery or DINO weights")
    require(json.loads((PREP / "train_ids.json").read_bytes())["ids"] ==
            json.loads((PREP / "eeg_train_ids.json").read_bytes())["ids"], "Training photo and EEG order differ")
    rows, labels, members = registered_photos(PREP)
    if dest.is_file():
        existing = torch.load(dest, weights_only=True, map_location="cpu")
        require(existing["proof"] == proof and np.array_equal(existing["rows"].numpy(), rows), "Training bank input changed")
        return
    model = visual_model(dino_path)
    objects, inverse = np.unique(labels, return_inverse=True)
    inverse = torch.from_numpy(inverse).cuda()
    all_features = []
    with zipfile.ZipFile(archive) as photos:
        for start in range(0, len(rows), 24):
            read = [read_photo(photos, members[int(r)], 512) for r in rows[start:start+24]]
            rgb = torch.from_numpy(np.stack(read)).permute(0, 3, 1, 2).cuda().float().div_(255)
            all_features.append(dino_cls(model, rgb))
            if start % 1200 == 0:
                print("identity training bank", min(start + 24, len(rows)), len(rows), flush=True)
    features = torch.cat(all_features)
    sums = class_sums(features, inverse)
    # On each original training photo, estimate recognition using only the other
    # nine original pictures of its class for the positive prototype.
    logits = object_logits(features, features, inverse, sums, "identity")
    correct = int((logits.argmax(1) == inverse).sum())
    info = {"photos": len(rows), "concepts": len(sums), "leave_one_photo_correct": correct}
    pins.verify()
    packet = dict(rows=torch.from_numpy(rows), labels=inverse.cpu(), concepts=torch.from_numpy(objects),
                  image_features=features.cpu(), sums=sums.cpu(), proof=proof, recognition=info)
    atomic_file(dest, lambda stream: torch.save(packet, stream))
    save_json(dest.with_suffix(".json"), dict(info, proof=proof))
    print("identity teacher cross-photo check", info, flush=True)


if __name__ == "__main__":
    prepare()
