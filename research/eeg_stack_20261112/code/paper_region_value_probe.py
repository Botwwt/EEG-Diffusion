"""Hold coarse appearance fixed while restoring details to complementary image regions."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from paper_horizon_readout import registration
from paper_outline_routing import MASK_RECEIPTS
from paper_render_cache import PinnedFiles, atomic_file, require, save_json


VIEWS = ("coarse", "foreground", "background")
REFERENCE_VIEWS = (*VIEWS, "whole")
COARSE_SIZE = 8


def appearance_views(pixels, mask):
    coarse = F.interpolate(F.interpolate(pixels, size=(COARSE_SIZE, COARSE_SIZE), mode="area"),
                           size=pixels.shape[-2:], mode="bilinear", align_corners=False)
    background = 1 - mask
    foreground_mass = mask.mean(dim=(2, 3), keepdim=True)
    background_mass = background.mean(dim=(2, 3), keepdim=True)
    budget = torch.minimum(foreground_mass, background_mass)
    front = mask * (budget / foreground_mass.clamp_min(1e-8))
    back = background * (budget / background_mass.clamp_min(1e-8))
    detail = pixels - coarse
    return coarse, coarse + front * detail, coarse + back * detail, (front, back)


def load_views(path, phase, expected_rows, registry, pins):
    receipt = path.with_suffix(".json")
    proof = json.loads(receipt.read_bytes())
    require(proof["phase"] == phase and proof["registration"] == registry
            and proof["mask"] == MASK_RECEIPTS[phase] and proof["archive_sha256"] == pins.pin(path)
            and proof["scope"] == ("reference-only" if phase == "dev" else "training-only"),
            "Region conditions come from a different photo panel or preprocessing run")
    with np.load(path, allow_pickle=False) as bank:
        require(np.array_equal(bank["row"], expected_rows), "Region conditions use a different photo ordering")
        names = REFERENCE_VIEWS if phase == "dev" else VIEWS
        features = {name: torch.from_numpy(bank[name].copy()).float() for name in names}
    require(all(x.shape == (len(expected_rows), 1024) and torch.isfinite(x).all() for x in features.values()),
            "Region condition bank is incomplete")
    return features, pins.pin(receipt)


@torch.inference_mode()
def prepare(benchmark, model_dir, mask_path, phase, output, batch_size):
    from PIL import Image
    from transformers import CLIPVisionModelWithProjection
    from paper_middle_teacher import CONFIG_SHA, MODEL_SHAS, image_processor

    require(batch_size > 0, "Use a positive visual inference batch size")
    pins = PinnedFiles()
    prep = benchmark / "verified/data/prep"
    (train, dev, _), _, registered = registration(prep, pins)
    selection = train if phase == "train" else dev
    mask_digest = pins.pin(mask_path)
    require(mask_digest == MASK_RECEIPTS[phase], "Unexpected archived foreground masks")
    photos = benchmark / "inputs/image_set"
    archive_path = photos / "training_images.zip"
    photo_path = photos / "train_photos.csv"
    archive_receipt = photos / "training_archive_verification.json"
    inputs = dict(photo_registry=pins.pin(photo_path), source_images=pins.pin(archive_path),
                  archive_receipt=pins.pin(archive_receipt),
                  weights=pins.pin(model_dir / "model.safetensors"), config=pins.pin(model_dir / "config.json"),
                  code=pins.pin(__file__), view_preprocess=pins.pin(Path(__file__).with_name("paper_middle_teacher.py")))
    require(inputs["weights"] in MODEL_SHAS and inputs["config"] == CONFIG_SHA, "Unexpected CLIP H/14 image tower")
    with photo_path.open(encoding="utf-8-sig", newline="") as stream:
        registry = list(csv.DictReader(stream))
    ids = json.loads((prep / "train_ids.json").read_bytes())["ids"]
    require([int(entry["row0"]) for entry in registry] == list(range(16540))
            and [entry["image_id"] for entry in registry] == ids, "Photo registration differs from the EEG rows")
    with np.load(mask_path, allow_pickle=False) as packet:
        mask = packet["masks"]
        proof = json.loads(packet["source"].item())
        verification = json.loads(archive_receipt.read_bytes())
        same_images = (proof["archive"] == verification["local_archive_sha256"]
                       and inputs["source_images"] == verification["reconstructed_sha256"] == verification["osf_reported_archive_sha256"]
                       and verification["matching_compressed_sizes_crc32_names_and_lengths"] is True)
        require(np.array_equal(packet["rows"], selection) and packet["valid"].all()
                and mask.shape == (len(selection), 160, 160) and mask.dtype == np.uint8
                and (proof["archive"] == inputs["source_images"] or same_images) and proof["split"] == registered["split"]
                and proof["development"] == registered["development"] and proof["panel"] == phase,
                "Archived foreground proof does not match this split")

    model = CLIPVisionModelWithProjection.from_pretrained(model_dir, local_files_only=True, use_safetensors=True,
                                                          torch_dtype=torch.float16).cuda().eval()
    processor = image_processor()
    names = REFERENCE_VIEWS if phase == "dev" else VIEWS
    embeddings = {name: np.empty((len(selection), 1024), dtype=np.float32) for name in names}
    diagnostics = np.empty((len(selection), 3), dtype=np.float32)
    with zipfile.ZipFile(archive_path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "Ambiguous archive members")
        for first in range(0, len(selection), batch_size):
            block = slice(first, first + batch_size)
            images = []
            for row in selection[block]:
                photo = registry[row]
                data = archive.read(photo["archive_member"])
                require(hashlib.sha256(data).hexdigest() == photo["image_sha256"], "Photo bytes differ from the manifest")
                with Image.open(io.BytesIO(data)) as image:
                    require(image.size == (500, 500), "Expected uncropped square input for mask alignment")
                    images.append(image.convert("RGB"))
            pixels = processor(images=images, return_tensors="pt").pixel_values.float().cuda()
            seg = torch.from_numpy(mask[block]).cuda().float().unsqueeze(1) / 255
            seg = F.interpolate(seg, size=pixels.shape[-2:], mode="bilinear", align_corners=False)
            coarse, fg, bg, (front, back) = appearance_views(pixels, seg)
            require(torch.allclose(front.mean((1, 2, 3)), back.mean((1, 2, 3)), atol=1e-6, rtol=0),
                    "The foreground and background controls must use the same total opacity")
            stats = torch.stack((front.mean((1, 2, 3)), (fg - coarse).square().mean((1, 2, 3)).sqrt(),
                                 (bg - coarse).square().mean((1, 2, 3)).sqrt()), dim=-1)
            diagnostics[block] = stats.cpu().numpy()
            values = (coarse, fg, bg, pixels) if phase == "dev" else (coarse, fg, bg)
            for name, image in zip(names, values, strict=True):
                embedding = model(pixel_values=image.half()).image_embeds.float()
                embeddings[name][block] = F.normalize(embedding).cpu().numpy()
            if first % (batch_size * 25) == 0 or first + len(images) == len(selection):
                print(phase, first + len(images), "/", len(selection), flush=True)

    if phase == "dev":
        native = benchmark / "native_baseline_candidate/enigma/targets/train_h14.npz"
        inputs["native_h14"] = pins.pin(native)
        with np.load(native, allow_pickle=False) as packet:
            require(np.array_equal(packet["photo_ids"], ids), "Native H/14 uses different photo IDs")
            target = F.normalize(torch.from_numpy(packet["images"][selection]).float())
        agreement = torch.from_numpy(embeddings["whole"]) @ target.T
        require(agreement.diag().min() > .995 and torch.equal(agreement.argmax(1), torch.arange(len(selection))),
                "The transformed views use a different H/14 space or photo order than the original photographs")
        inputs["whole_cosine_quantiles"] = torch.quantile(agreement.diag(), torch.tensor([0., .5, 1.])).tolist()
        inputs["whole_registered_matches"] = len(selection)
    require(all(np.isfinite(out).all() for out in embeddings.values()), "Non-finite region features")
    pins.verify()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_file(output, lambda stream: np.savez_compressed(stream, row=selection, details=diagnostics, **embeddings))
    save_json(output.with_suffix(".json"), dict(phase=phase, n=len(selection), registration=registered, mask=mask_digest,
                                               sources=inputs, archive_sha256=pins.pin(output), coarse_size=COARSE_SIZE,
                                               scope="reference-only" if phase == "dev" else "training-only",
                                               diagnostics_mean=diagnostics.mean(0).tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "dev"))
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("masks", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    prepare(args.benchmark, args.model_dir, args.masks, args.phase, args.output, args.batch_size)
