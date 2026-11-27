"""Score already-rendered RGB images with a second frozen image-language model."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, require, save_json


WEIGHTS_SHA = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
MODEL = "google/siglip2-base-patch16-224@75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


def vocabulary(assets, files, development=None, fresh=None):
    require(development is None or fresh is None, "Use one registered held-out panel")
    plan = fresh or development
    word_path = assets / "concepts1854.csv"
    photo_path = assets / ("test_photos.csv" if plan is None else "train_photos.csv")
    words_sha, photos_sha = files.pin(word_path), files.pin(photo_path)
    with word_path.open(encoding="utf-8-sig", newline="") as stream:
        names = {row["unique_id"]: row["display_name"].strip().lower() for row in csv.DictReader(stream)}
    with photo_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    size = 200 if plan is None else 16540
    require(len(names) == 1854 and [int(r["row0"]) for r in rows] == list(range(size)), "Wrong evaluation registry")
    sources = dict(concepts=words_sha, photographs=photos_sha)
    if plan is not None:
        panel, key, count = ("fresh", "fresh_rows", 50) if fresh is not None else ("development", "dev_rows", 20)
        sources[panel] = files.pin(plan)
        proof = json.loads(plan.read_bytes())
        order = proof[key]
        require(len(order) == len(set(order)) == count * 10 and all(type(i) is int and 0 <= i < size for i in order),
                "Invalid held-out stimulus indices")
        if fresh is not None:
            require(set(order).isdisjoint(proof["train_rows"]) and proof["trial_indices_R4"] == list(range(4)),
                    "The fresh panel must use the registered four trials and hold out training stimuli")
        rows = [rows[i] for i in order]
        _, counts = np.unique([row["unique_id"] for row in rows], return_counts=True)
        require(len(counts) == count and np.all(counts == 10), "Expected ten photographs for each held-out concept")
    normalized, inverse = np.unique(list(names.values()), return_inverse=True)
    lookup = {key: i for i, key in enumerate(names)}
    target = np.asarray([lookup[row["unique_id"]] for row in rows])
    return np.asarray(list(names)), normalized, inverse, target, rows, sources


def accuracy(logits, wanted):
    ceiling = logits.max(1, keepdims=True)
    score = logits[np.arange(len(wanted)), wanted] == ceiling[:, 0]
    return score / (logits == ceiling).sum(1), logits.argmax(1)


@torch.inference_mode()
def run(model_dir, assets, archive, output, people, arms, development=None, device="cuda", official_budget="r4", fresh=None):
    from PIL import Image
    from transformers import AutoModel, AutoProcessor, __version__

    files = PinnedFiles()
    inputs = {path.name: files.pin(path) for path in model_dir.iterdir() if path.is_file() and not path.name.startswith(".")}
    require(inputs.get("model.safetensors") == WEIGHTS_SHA, "Wrong SigLIP2 release weights")
    archive_sha = files.pin(archive)
    keys, text, inverse, target, rows, sources = vocabulary(assets, files, development, fresh)
    indices = [int(row["row0"]) for row in rows]
    held_out = development is not None or fresh is not None
    require(not held_out or all("/" not in arm for arm in arms), "Use named arms from the registered held-out render")
    require(official_budget in ("r4", "r80") and (not held_out or official_budget == "r4"), "Invalid panel and repeat budget")
    repeats = np.arange(80 if official_budget == "r80" else 4).reshape(-1, 4).tolist()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, use_safetensors=True).to(device).eval()
    text_length = model.config.text_config.max_position_embeddings
    phrases = [f"This is a photo of {item}." for item in text]
    encoded = []
    for start in range(0, len(phrases), 32):
        group = phrases[start:start + 32]
        tokens = processor(text=group, padding="max_length", max_length=text_length,
                           truncation=True, return_tensors="pt").to(device)
        encoded.append(F.normalize(model.get_text_features(**tokens).float()))
    bank = torch.cat(encoded)

    def evaluate(payloads):
        all_logits = []
        for offset in range(0, len(payloads), 24):
            images = []
            for data in payloads[offset:offset + 24]:
                with Image.open(io.BytesIO(data)) as image:
                    images.append(image.convert("RGB"))
            pixels = processor(images=images, return_tensors="pt").to(device)
            features = F.normalize(model.get_image_features(**pixels).float())
            all_logits.append((features @ bank.T).cpu().numpy()[:, inverse])
        values = np.concatenate(all_logits)
        require(np.isfinite(values).all(), "Independent image scorer returned a nonfinite logit")
        hit, prediction = accuracy(values, target)
        return dict(credits=float(hit.sum()), per_row=hit.tolist(), top=keys[prediction].tolist())

    photos = assets / ("training_images.zip" if held_out else "test_images.zip")
    sources["photo_zip"] = files.pin(photos)
    with zipfile.ZipFile(photos) as raw:
        originals = [raw.read(row["archive_member"]) for row in rows]
    require([hashlib.sha256(data).hexdigest() for data in originals] == [row["image_sha256"] for row in rows],
            "Reference image bytes do not match the photograph manifest")
    reference = evaluate(originals)
    summary = {}
    with tarfile.open(archive) as packet:
        for person in people:
            summary[person] = {}
            for arm in arms:
                track, separator, saved_arm = arm.partition("/")
                if fresh is not None:
                    prefix, saved_arm = f"images/shared/{arm}/fresh-r4/sub-{person:02d}", arm
                elif development is not None:
                    prefix, saved_arm = f"images/dev/sub-{person:02d}/{arm}", arm
                elif separator:
                    prefix = f"historical/{track}/{saved_arm}/official-{official_budget}/sub-{person:02d}"
                else:
                    prefix, saved_arm = f"images/sub-{person:02d}/{arm}", arm
                stored = packet.extractfile(f"{prefix}/render.json").read()
                receipt = json.loads(stored)
                require(receipt["row"] == indices and receipt["repeats"] == repeats
                        and receipt["arm"] == saved_arm and receipt["subject"] == person
                        and (not separator or receipt["training"] == track)
                        and (official_budget != "r80" or receipt["budget"] == "official-r80")
                        and (development is None or (receipt["budget"] == "dev-r4"
                                                     and receipt["development_split_sha256"] == sources["development"])),
                        "RGB archive does not describe the selected reconstruction")
                if fresh is not None:
                    require(receipt["budget"] == "fresh-r4" and receipt["split_sha256"] == sources["fresh"],
                            "RGB archive does not use the fresh split")
                images = [packet.extractfile(f"{prefix}/{i:05d}.png").read() for i in range(len(rows))]
                require([hashlib.sha256(data).hexdigest() for data in images] == receipt["pixels_sha256"],
                        "Generated PNG bytes differ from the committed render")
                result = evaluate(images)
                summary[person][arm] = dict(result, render_sha256=hashlib.sha256(stored).hexdigest())
                print("SigLIP2", person, arm, result["credits"], flush=True)
    files.verify()
    save_json(output, dict(model=MODEL, model_files_sha256=inputs, transformers_version=__version__,
                           images_sha256=archive_sha, sources=sources, reference=reference, people=people,
                           subjects=summary, totals={a: sum(summary[p][a]["credits"] for p in people) for a in arms},
                           code_sha256=files.pin(Path(__file__)), n_per_arm=len(rows) * len(people)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "assets", "archive", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", type=int, required=True)
    parser.add_argument("--arms", nargs="+", required=True, help="New arm names or historical training-track/arm")
    panel = parser.add_mutually_exclusive_group()
    panel.add_argument("--development-plan", type=Path, help="Score a registered development panel")
    panel.add_argument("--fresh-split", type=Path, help="Score the complete 50-concept R4 holdout")
    parser.add_argument("--official-budget", choices=("r4", "r80"), default="r4")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    run(args.model, args.assets, args.archive, args.output, args.subjects, args.arms, args.development_plan, args.device,
        official_budget=args.official_budget, fresh=args.fresh_split)
