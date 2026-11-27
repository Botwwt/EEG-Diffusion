"""Measure the public edits against the exact nine archived R80 image systems."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from paper_aggregate import bootstrap_ci
from paper_editbench import FAMILIES, TOTALS, cluster_edits, edit_outcomes, point_metrics
from paper_independent_naming import WEIGHTS_SHA
from paper_middle_teacher import CONFIG_SHA, MODEL_SHAS, image_processor
from paper_raw_semantic_readout import read_json
from paper_render_cache import PinnedFiles, atomic_file, require, save_json


ROWS = {
    "atm": "atm-author", "cogcap": "cogcap-author", "hive": "brain-hive-native", "enigma": "enigma-native",
    "d2": "m1a-d2", "o13-late": "o13-late", "native-full-late": "native-full-late",
    "full-d2": "m1a-d2-full-training", "full-hive-late": "m1a-d2-full-training-hive-late",
}


def signed_images(benchmark, person, main, last_fit, identity):
    system = benchmark / "native-r80-siglip2"
    bundles = {
        f"atm-sub{person:02d}": ("atm",), f"cogcap-sub{person:02d}": ("cogcap",), f"enigma-{person:02d}": ("enigma",),
        f"shared-{person:02d}": ("hive", "d2"), f"candidate-{person:02d}": ("o13-late", "native-full-late"),
    }
    name = f"sub-{person:02d}"
    for bundle, arms in bundles.items():
        with tarfile.open(system / f"{bundle}.tar") as archive:
            pack = json.load(archive.extractfile("pack.json"))
            require(pack["format"] == "siglip2-r80-rgb224-v1" and pack["row"] == list(range(200))
                    and pack["subject"] == person and pack["test_photos_sha256"] == identity,
                    "Published image packet did not follow the official EEG rows")
            for arm in arms:
                spec = pack["methods"][arm]
                expected = main[ROWS[arm]]["sources"][name]["ordered_image_hashes_sha256"]
                require(spec["raw_order_sha256"] == expected == sha256("\n".join(spec["raw_sha256"]).encode()).hexdigest(),
                        "Image packet differs from the main table")
                data = [archive.extractfile(f"staged/{arm}/{i:05d}.png").read() for i in range(200)]
                require([sha256(x).hexdigest() for x in data] == spec["staged_sha256"], "Committed 224×224 pixels changed")
                yield arm, data, expected
    with tarfile.open(benchmark / "full-shared-official" / f"{name}.rgb.tar") as archive:
        for arm, saved in (("full-d2", "d2"), ("full-hive-late", "native-full-late")):
            folder = f"images/shared-full/{saved}/official-r80/{name}"
            source = last_fit["receipts"][str(person)][saved]
            receipt = archive.extractfile(f"{folder}/render.json").read()
            result = archive.extractfile(f"{folder}/scores.json").read()
            r, score = json.loads(receipt), json.loads(result)
            require(sha256(receipt).hexdigest() == source["render_sha256"] and sha256(result).hexdigest() == source["scores_sha256"]
                    and score["ordered_image_hashes_sha256"] == main[ROWS[arm]]["sources"][name]["ordered_image_hashes_sha256"]
                    and r["row"] == list(range(200)) and r["budget"] == "official-r80" and r["subject"] == person,
                    "Full-training packet differs from the main table")
            data = [archive.extractfile(f"{folder}/{i:05d}.png").read() for i in range(200)]
            require([sha256(x).hexdigest() for x in data] == r["pixels_sha256"], "Full-training PNG payload changed")
            yield arm, data, score["ordered_image_hashes_sha256"]


@torch.inference_mode()
def run(benchmark, assets, reference, h14_dir, siglip_dir, output):
    from transformers import AutoModel, AutoProcessor, CLIPVisionModelWithProjection

    files = PinnedFiles()
    table = benchmark / "full-shared-official/full-shared-main-table.json"
    full_meta = table.with_name("full-shared-verified.json")
    main, expanded = read_json(table)["r80_native"]["rows"], read_json(full_meta)
    refs = reference.with_suffix(".json")
    referee = read_json(refs)
    require(referee["scope"] == "evaluation-only" and files.pin(reference) == referee["array_sha256"], "Invalid EditBench bank")
    with np.load(reference, allow_pickle=False) as z:
        names, queries, origins, family, h14_bank, hashes = (z[k] for k in ("image_id", "query_id", "origin", "family", "embedding", "hashes"))
    require(len(queries) == 200 and len(names) == 2337 and tuple(np.bincount(family)) == TOTALS, "Incomplete public benchmark")
    proof = dict(reference=files.pin(reference), table=files.pin(table), full_training=files.pin(full_meta),
                 code=files.pin(Path(__file__)), h14=files.pin(h14_dir / "model.safetensors"),
                 siglip2=files.pin(siglip_dir / "model.safetensors"), siglip2_pixel=files.pin(siglip_dir / "preprocessor_config.json"))
    require(proof["h14"] in MODEL_SHAS and files.pin(h14_dir / "config.json") == CONFIG_SHA and proof["siglip2"] == WEIGHTS_SHA,
            "Image evaluators must match the frozen releases")
    clip_proc = image_processor()
    clip = CLIPVisionModelWithProjection.from_pretrained(h14_dir, local_files_only=True, use_safetensors=True,
                                                         torch_dtype=torch.float16).cuda().eval()
    sig_proc = AutoProcessor.from_pretrained(siglip_dir, local_files_only=True, use_fast=False)
    siglip = AutoModel.from_pretrained(siglip_dir, local_files_only=True, use_safetensors=True).cuda().eval()

    def features(contents, include_clip=True, evaluation_size=None):
        h14, sig = [], []
        for start in range(0, len(contents), 32):
            pixels = []
            for payload in contents[start:start + 32]:
                with Image.open(io.BytesIO(payload)) as image:
                    rgb = image.convert("RGB")
                if evaluation_size is not None and rgb.size != evaluation_size:
                    rgb = rgb.resize(evaluation_size, Image.Resampling.BILINEAR)
                pixels.append(rgb)
            if include_clip:
                x = clip_proc(images=pixels, return_tensors="pt").pixel_values.cuda().half()
                h14.append(F.normalize(clip(pixel_values=x).image_embeds.float()))
            y = sig_proc(images=pixels, return_tensors="pt").to("cuda")
            sig.append(F.normalize(siglip.get_image_features(**y).float()))
        return torch.cat(h14) if h14 else None, torch.cat(sig)

    edits = [(assets / name).read_bytes() for name in names]
    require(all(sha256(data).hexdigest() == digest for data, digest in zip(edits, hashes, strict=True)), "Public source image changed")
    _, siglip_bank = features(edits, include_clip=False)
    del edits
    banks = F.normalize(torch.from_numpy(h14_bank).cuda()), siglip_bank
    embeddings = {name: {arm: [] for arm in ROWS} for name in ("h14", "siglip2")}
    receipts = {}
    for p in range(1, 11):
        receipts[str(p)] = {}
        for arm, pictures, expected in signed_images(benchmark, p, main, expanded, referee["sources"]["registry"]):
            for name, x, visual in zip(embeddings, features(pictures, evaluation_size=(224, 224)), banks, strict=True):
                embeddings[name][arm].append((x @ visual.T).cpu().numpy())
            receipts[str(p)][arm] = expected
            print("Verified actual RGB and scored both image towers", p, arm, flush=True)
    summaries, comparisons, values = {}, {}, {}
    metrics = ("200way_original", "ep_top1", *FAMILIES)
    for name, by_arm in embeddings.items():
        summaries[name], comparisons[name] = {}, {}
        per_query = {}
        for arm, score in by_arm.items():
            v = np.stack(score)
            require(v.shape == (10, 200, 2337) and np.isfinite(v).all(), "Incomplete actual RGB score matrix")
            seen = edit_outcomes(v, 200, origins)
            summaries[name][arm] = point_metrics(seen, family)
            per_query[arm] = np.concatenate((np.stack(seen[:2], axis=-1), cluster_edits(seen[-1], family, origins, 200)), axis=-1)
            values[f"{name}/{arm}"] = v
        for arm, score in per_query.items():
            for opponent in ("atm", "cogcap", "hive", "enigma", "d2"):
                if arm == opponent:
                    continue
                delta = score - per_query[opponent]
                lo, hi = bootstrap_ci(delta)
                comparisons[name][f"{arm} minus {opponent}"] = dict(zip(metrics, np.stack((delta.mean((0, 1)), lo, hi), axis=1).tolist(), strict=True))
    files.verify()
    record = output.with_suffix(".npz")
    atomic_file(record, lambda f: np.savez_compressed(f, query_id=queries, image_id=names, origin=origins, family=family, **values))
    save_json(output, dict(proof=proof, sources=receipts, count=18000, per_arm=2000, scope="archived-generated-rgb224",
                           summary=summaries, paired=comparisons, arrays=files.pin(record)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark", "assets", "reference", "h14_dir", "siglip_dir", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    torch.set_num_threads(4)
    run(args.benchmark, args.assets, args.reference, args.h14_dir, args.siglip_dir, args.output)
