"""Measure how rendered subject scores change when only the EEG condition is replaced."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from paper_aggregate import bootstrap, panel_seeds
from paper_independent_naming import MODEL, WEIGHTS_SHA, vocabulary
from paper_render_cache import PinnedFiles, require, save_json


ROWS = np.arange(200)
DONOR = np.roll(ROWS, 100)
FIELDS = ("target_name", "donor_name", "name_margin", "target_photo", "donor_photo", "photo_margin")


def check_pair(correct, exchanged, subject, arms):
    shared = ("subject", "budget", "row", "seeds", "repeats", "prompt", "draws", "unet_calls_per_draw")
    require(all(correct[key] == exchanged[key] for key in shared), "Paired render requests have different budgets or rows")
    require(correct["subject"] == subject and correct["budget"] == "official-r4" and correct["row"] == ROWS.tolist()
            and correct["seeds"] == panel_seeds("official", len(ROWS)).tolist() and correct["repeats"] == [[0, 1, 2, 3]]
            and correct["prompt"] == "" and correct["draws"] == 1 and correct["unet_calls_per_draw"] == 4,
            "Unexpected official stimulus or acquisition budget")
    require((correct["arm"], exchanged["arm"]) == arms and correct["eeg_donor"] == ROWS.tolist()
            and exchanged["eeg_donor"] == DONOR.tolist(), "EEG donor order differs")
    fixed_layout = correct.get("layout_source", correct.get("layout"))
    fixed_layout_order = correct.get("layout_donor", correct.get("d2_eeg_donor"))
    require(fixed_layout and fixed_layout == exchanged.get("layout_source", exchanged.get("layout"))
            and fixed_layout_order == ROWS.tolist()
            and fixed_layout_order == exchanged.get("layout_donor", exchanged.get("d2_eeg_donor")),
            "A layout change would confound the effect of the EEG condition")
    generators = correct.get("generator_files_sha256", correct.get("generators"))
    require(generators and generators == exchanged.get("generator_files_sha256", exchanged.get("generators"))
            and correct.get("fit") == exchanged.get("fit") and correct["physical_eeg"] == exchanged["physical_eeg"],
            "The two PNG sets were produced by different models or measurements")


def measure(score):
    intervals, _ = bootstrap(score, 10000, 20261128)
    return {key: dict(mean=float(score[:, :, i].mean()), ci95=intervals[:, i].tolist(),
                      by_subject=score[:, :, i].mean(1).tolist()) for i, key in enumerate(FIELDS)}


def embeddings(processor, model, data, device):
    rows = []
    for start in range(0, len(data), 24):
        batch = []
        for payload in data[start:start + 24]:
            with Image.open(io.BytesIO(payload)) as image:
                batch.append(image.convert("RGB"))
        features = model.get_image_features(**processor(images=batch, return_tensors="pt").to(device))
        rows.append(F.normalize(features.float()))
    return torch.cat(rows)


def row_scores(pictures, words, originals):
    name_scores, photo_scores = (pictures @ target.T for target in (words, originals))
    values = []
    for matrix in (name_scores.cpu().numpy(), photo_scores.cpu().numpy()):
        own, other = matrix[ROWS, ROWS], matrix[ROWS, DONOR]
        values.extend((own, other, own - other))
    return np.column_stack(values)


@torch.inference_mode()
def run(model_dir, assets, images, output, people, pairs, device):
    from transformers import AutoModel, AutoProcessor, __version__

    requested = [arm for pair in pairs for arm in pair]
    require(people and len(set(people)) == len(people) and len(set(requested)) == len(requested),
            "Choose distinct participants and nonoverlapping render pairs")
    files = PinnedFiles()
    model_files = {p.name: files.pin(p) for p in model_dir.iterdir() if p.is_file() and not p.name.startswith(".")}
    require(model_files.get("model.safetensors") == WEIGHTS_SHA, "Wrong SigLIP2 revision")
    _, names, inverse, targets, photo_index, sources = vocabulary(assets, files)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, use_safetensors=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    phrases = [f"This is a photo of {name}." for name in names[inverse[targets]]]
    text_length = model.config.text_config.max_position_embeddings
    words = []
    for start in range(0, len(phrases), 32):
        tokens = processor(text=phrases[start:start + 32], padding="max_length", max_length=text_length,
                           truncation=True, return_tensors="pt").to(device)
        words.append(F.normalize(model.get_text_features(**tokens).float()))
    words = torch.cat(words)
    zip_path = assets / "test_images.zip"
    sources["photo_zip"] = files.pin(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        photo_bytes = [archive.read(item["archive_member"]) for item in photo_index]
    require([hashlib.sha256(data).hexdigest() for data in photo_bytes] == [r["image_sha256"] for r in photo_index],
            "Reference image bytes differ from the registry")
    originals = embeddings(processor, model, photo_bytes, device)
    source, scores = {}, {}
    for correct, wrong in pairs:
        source[correct], source[wrong] = {}, {}
        scores[correct], scores[wrong] = [], []
        for person in people:
            folders = [images / f"sub-{person:02d}" / arm for arm in (correct, wrong)]
            bodies = [(folder / "render.json").read_bytes() for folder in folders]
            records = [json.loads(body) for body in bodies]
            check_pair(*records, person, (correct, wrong))
            for arm, folder, body, receipt in zip((correct, wrong), folders, bodies, records, strict=True):
                originals_hash = receipt["pixels_sha256"]
                payloads = [(folder / f"{i:05d}.png").read_bytes() for i in ROWS]
                require([hashlib.sha256(data).hexdigest() for data in payloads] == originals_hash,
                        f"PNG bytes differ from the render manifest at {folder}")
                features = embeddings(processor, model, payloads, device)
                values = row_scores(features, words, originals)
                require(np.isfinite(values).all(), "Nonfinite score for the exchanged EEG experiment")
                scores[arm].append(values)
                source[arm][person] = hashlib.sha256(body).hexdigest()
    arrays = {arm: np.stack(items) for arm, items in scores.items()}
    differences = {a: arrays[a] - arrays[b] for a, b in pairs}
    pair_report = {a: measure(delta) for a, delta in differences.items()}
    for a, b in zip(tuple(differences)[:-1], tuple(differences)[1:], strict=True):
        pair_report[f"{b}_minus_{a}"] = measure(differences[b] - differences[a])
    files.verify()
    save_json(output, dict(people=people, fields=FIELDS, donors=DONOR.tolist(), model=MODEL,
                           model_files_sha256=model_files, transformers=__version__, reference_inputs=sources,
                           code_sha256=files.pin(Path(__file__)), render_sha256=source,
                           scores={a: measure(v) for a, v in arrays.items()}, differences=pair_report,
                           per_row={k: v.tolist() for k, v in arrays.items()}))
    for arm, delta in pair_report.items():
        print(arm, "name margin", delta["name_margin"], "photo margin", delta["photo_margin"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("model", "assets", "images", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", type=int, required=True)
    parser.add_argument("--pair", metavar=("CORRECT", "EXCHANGED"), nargs=2, action="append", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.model, args.assets, args.images, args.output, args.subjects, args.pair, args.device)
