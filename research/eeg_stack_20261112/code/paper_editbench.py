"""Score confirmed official R80 queries on the complete public EEG-EditBench edits."""

import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from paper_aggregate import bootstrap_ci
from paper_horizon_readout import VIEWS
from paper_middle_teacher import CONFIG_SHA, MODEL_SHAS, image_processor
from paper_raw_semantic_readout import normalize, read_json
from paper_render_cache import PinnedFiles, atomic_file, require, save_json


FAMILIES = ("identity", "attribute", "background", "removal")
TOTALS = (780, 979, 199, 179)
EDITBENCH_REVISION = "737a6709e58d243cd8d404770bf82983e3a5ae6e"


def edit_outcomes(scores, n_original, source_row):
    """Test the original gallery and the source-specific edit set separately."""
    original = scores[..., :n_original]
    same = np.eye(n_original, dtype=bool)
    true = np.diagonal(original, axis1=-2, axis2=-1)
    other = np.where(same, -np.inf, original).max(-1)
    delta = true[..., source_row] - scores[..., source_row, n_original + np.arange(len(source_row))]
    wins = delta > 0
    with_edits = np.ones_like(true, dtype=bool)
    for i, origin in enumerate(source_row):
        with_edits[..., origin] &= wins[..., i]
    return true > other, with_edits, wins


def registry(assets, manifest_path, official):
    edits = [json.loads(s) for s in manifest_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    with official.open(encoding="utf-8-sig", newline="") as stream:
        table = list(csv.DictReader(stream))
    require([int(row["row0"]) for row in table] == list(range(200)), "Unexpected official photo list")
    query_ids = [f"{int(row['split_class_id']):05d}" for row in table]
    rows = {photo: i for i, photo in enumerate(query_ids)}
    originals = [None] * len(rows)
    edited, source_row, family = [], [], []
    require(len({edit["record_id"] for edit in edits}) == len(edits) == 2137, "Edited stimuli are repeated or missing")
    for edit in edits:
        i = rows[edit["concept_id"]]
        source, changed = (PurePosixPath(edit[key]) for key in ("source_image_path", "edited_image_path"))
        require(not source.is_absolute() and not changed.is_absolute()
                and ".." not in source.parts and ".." not in changed.parts and source.name == table[i]["filename"]
                and edit["raw_concept"] == table[i]["unique_id"] and source.parent == changed.parent.parent,
                "An EditBench intervention refers to another official photo")
        require(originals[i] is None or originals[i] == str(source), "One stimulus has more than one origin")
        originals[i] = str(source)
        edited.append(str(changed))
        source_row.append(i)
        family.append(FAMILIES.index("identity" if edit["edit_type"] == "replacement" else edit["edit_type"]))
    family = np.asarray(family, dtype=np.int64)
    require(all(originals) and tuple(np.bincount(family)) == TOTALS and len(set(edited)) == len(edited)
            and not set(originals).intersection(edited), "Did not receive the complete four-family public benchmark")
    image_names = originals + edited
    require(all((assets / name).is_file() for name in image_names), "The pinned public image release is incomplete")
    return table, query_ids, image_names, np.asarray(source_row, dtype=np.int64), family


@torch.inference_mode()
def prepare(benchmark, assets, manifest, model_dir, output):
    from transformers import CLIPVisionModelWithProjection

    files = PinnedFiles()
    official = benchmark / "inputs/image_set/test_photos.csv"
    stimuli = official.with_name("test_images.zip")
    hf = assets / "manifests/eeg_editbench.jsonl"
    require(files.pin(hf) == files.pin(manifest), "Pinned GitHub and data release manifests differ")
    sources = dict(manifest=files.pin(manifest), registry=files.pin(official), stimuli=files.pin(stimuli), config=files.pin(model_dir / "config.json"),
                   weights=files.pin(model_dir / "model.safetensors"), code=files.pin(Path(__file__)),
                   visual_preprocess=files.pin(Path(__file__).with_name("paper_middle_teacher.py")))
    require(sources["weights"] in MODEL_SHAS and sources["config"] == CONFIG_SHA, "Different H/14 image tower")
    table, query_ids, names, rows, family = registry(assets, manifest, official)
    model = CLIPVisionModelWithProjection.from_pretrained(model_dir, local_files_only=True, use_safetensors=True,
                                                          torch_dtype=torch.float16).cuda().eval()
    processor = image_processor()
    trial = sha256()
    values = np.empty((len(names), 1024), dtype=np.float32)
    hashes = []
    canonical = []
    with zipfile.ZipFile(stimuli) as archive:
        for row in table:
            raw = archive.read(row["archive_member"])
            require(sha256(raw).hexdigest() == row["image_sha256"], "Official stimulus pixels changed")
            with Image.open(io.BytesIO(raw)) as picture:
                canonical.append(picture.convert("RGB"))
    pixels_mae, file_matches = [], 0
    for start in range(0, len(names), 32):
        images = []
        for i, name in enumerate(names[start:start + 32], start=start):
            raw = (assets / name).read_bytes()
            digest = sha256(raw).hexdigest()
            trial.update(f"{name}\t{digest}\n".encode())
            hashes.append(digest)
            with Image.open(io.BytesIO(raw)) as picture:
                picture = picture.convert("RGB")
            if i < len(table):
                file_matches += digest == table[i]["image_sha256"]
                original = np.asarray(canonical[i].resize(picture.size, Image.Resampling.BICUBIC), dtype=np.int16)
                pixels_mae.append(float(np.abs(np.asarray(picture, dtype=np.int16) - original).mean() / 255.))
            images.append(picture)
        z = model(pixel_values=processor(images=images, return_tensors="pt").pixel_values.cuda().half()).image_embeds.float()
        values[start:start + len(images)] = F.normalize(z).cpu().numpy()
        if (start // 32) % 20 == 0 or start + len(images) == len(names):
            print("Independent EditBench visual references", start + len(images), "/", len(names), flush=True)
    original_tower = []
    for first in range(0, len(canonical), 32):
        x = processor(images=canonical[first:first + 32], return_tensors="pt").pixel_values.cuda().half()
        original_tower.append(F.normalize(model(pixel_values=x).image_embeds.float()).cpu().numpy())
    native = np.concatenate(original_tower)
    correspondence = native @ values[:len(table)].T
    require(np.array_equal(correspondence.argmax(1), np.arange(len(table))) and np.isfinite(values).all()
            and np.isfinite(native).all(), "Resized public originals do not match the official stimuli")
    files.verify()
    atomic_file(output, lambda stream: np.savez_compressed(stream, image_id=names, query_id=query_ids, embedding=values,
                                                           canonical=native, family=family, origin=rows, hashes=hashes))
    save_json(output.with_suffix(".json"), dict(sources=sources, release=EDITBENCH_REVISION, photos=len(names),
                                               families=dict(zip(FAMILIES, TOTALS, strict=True)),
                                               scope="evaluation-only", reference_sha256=trial.hexdigest(), array_sha256=files.pin(output),
                                               original_file_matches=file_matches, h14_correspondences=len(table),
                                               source_pixel_mae=np.quantile(pixels_mae, [0, .5, 1]).tolist(),
                                               source_h14_cosine=np.quantile(correspondence.diagonal(), [0, .5, 1]).tolist()))


def point_metrics(outcomes, family):
    original, context, edit = outcomes
    def item(x):
        return dict(n=x.size, hits=int(x.sum()), accuracy=float(x.mean()))

    return {"200way_original": item(original), "ep_top1": item(context),
            **{name: item(edit[..., family == j]) for j, name in enumerate(FAMILIES)}, "overall": item(edit)}


def cluster_edits(edit, family, origin, n_original):
    result = np.zeros((*edit.shape[:-1], n_original, len(FAMILIES)), dtype=np.float64)
    for j in range(len(FAMILIES)):
        chosen = np.flatnonzero(family == j)
        for i in chosen:
            result[..., origin[i], j] += edit[..., i]
        result[..., j] *= n_original / len(chosen)
    return result


def score(visual, query_bank, output):
    files = PinnedFiles()
    reference_receipt = read_json(visual.with_suffix(".json"))
    query_receipt = read_json(query_bank.with_suffix(".json"))
    require(reference_receipt["scope"] == "evaluation-only" and reference_receipt["array_sha256"] == files.pin(visual)
            and query_receipt["array_sha256"] == files.pin(query_bank) and query_receipt["n"] == 2000
            and query_receipt["repeats"] == [list(range(r, r + 4)) for r in range(0, 80, 4)]
            and query_receipt["concept_file"] == reference_receipt["sources"]["registry"],
            "Image edits and R80 queries have different scope or source identities")
    with np.load(query_bank, allow_pickle=False) as q:
        require(np.array_equal(q["row"], np.arange(200)) and np.array_equal(q["person"], np.arange(1, 11)),
                "Official EEG queries must use all ten people in registered stimulus order")
        features = normalize(q["embedding"])
    require(features.shape == (10, 200, 4, 1024) and np.isfinite(features).all(), "Incorrect official H/14 code")
    with np.load(visual, allow_pickle=False) as z:
        images, canonical, origin, family, names, query_ids = (z[k] for k in ("embedding", "canonical", "origin", "family", "image_id", "query_id"))
    require(images.shape == (2337, 1024) and len(query_ids) == 200 and tuple(np.bincount(family)) == TOTALS,
            "The public benchmark has an incomplete bank")
    values, detail, summaries, comparisons = {}, {}, {}, {}
    for index, tag in enumerate(VIEWS):
        values[tag] = features[:, :, index] @ normalize(images).T
    values["native-wrong-eeg"] = np.roll(values["native"], 1, axis=1)
    values["native-canonical-source"] = values["native"].copy()
    values["native-canonical-source"][:, :, :len(query_ids)] = features[:, :, 0] @ normalize(canonical).T
    proof = {"native": query_receipt["checkpoint"], "queries": files.pin(query_bank), "references": files.pin(visual),
             "schema": files.pin(Path(__file__)), "physical": query_receipt["physical"], "subjects": 10, "repeats": 80}
    for arm, matrix in values.items():
        results = edit_outcomes(matrix, len(query_ids), origin)
        summaries[arm] = point_metrics(results, family)
        prefix = np.stack(results[:2], axis=-1)
        detail[arm] = np.concatenate((prefix, cluster_edits(results[-1], family, origin, len(query_ids))), axis=-1)
    names_of_metrics = ("200way_original", "ep_top1", *FAMILIES)
    for arm in values:
        if arm == "native":
            continue
        delta = detail["native"] - detail[arm]
        lo, hi = bootstrap_ci(delta)
        comparisons[f"native minus {arm}"] = dict(zip(names_of_metrics, np.stack((delta.mean((0, 1)), lo, hi), axis=1).tolist(), strict=True))
    files.verify()
    archive = output.with_suffix(".npz")
    atomic_file(archive, lambda s: np.savez_compressed(s, query_id=query_ids, image_id=names, family=family, origin=origin, **values))
    save_json(output, dict(proof=proof, summary=summaries, paired=comparisons, arrays=files.pin(archive)))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="phase", required=True)
    image = sub.add_parser("prepare")
    for name in ("benchmark", "assets", "manifest", "model_dir", "output"):
        image.add_argument(name, type=Path)
    measured = sub.add_parser("score")
    for name in ("visual", "query_bank", "output"):
        measured.add_argument(name, type=Path)
    args = cli.parse_args()
    torch.set_num_threads(4)
    if args.phase == "prepare":
        prepare(args.benchmark, args.assets, args.manifest, args.model_dir, args.output)
    else:
        score(args.visual, args.query_bank, args.output)
