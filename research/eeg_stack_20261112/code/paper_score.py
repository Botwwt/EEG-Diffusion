"""Score a complete folder of blind reconstructions against original THINGS photographs."""
import argparse
import csv
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torchvision import models, transforms as T
from torchvision.models.feature_extraction import create_feature_extractor

from common import PREP, write_json
from eval_instance_starts import load_visual_scorer
from score_official_objects import load_dictionary, load_tokenizer, text_bank, topk_credit, unit


def normal(x, pearson=False):
    x = x.astype(np.float64)
    return unit(x - x.mean(1, keepdims=True) if pearson else x)


def prep(x):
    return T.ToTensor()(T.CenterCrop(512)(T.Resize(512, interpolation=T.InterpolationMode.BILINEAR)(x.convert("RGB"))))


def features(x, net, metric):
    size = {"alex2": 256, "alex5": 256, "inception": 342, "clip": 224, "swav": 224, "effnet": 255}[metric]
    mean, sd = ([.48145466, .4578275, .40821073], [.26862954, .26130258, .27577711]) if metric == "clip" else ([.485, .456, .406], [.229, .224, .225])
    x = T.Normalize(mean, sd)(T.Resize(size, interpolation=T.InterpolationMode.BILINEAR)(x.cuda()))
    if metric == "clip":
        return net.encode_image(x).float().cpu().numpy()
    return next(iter(net(x).values())).flatten(1).float().cpu().numpy()


def load_metric(archive, name):
    if name == "clip":
        return load_visual_scorer(archive)[0]
    constructors = {"alex2": models.alexnet, "alex5": models.alexnet, "inception": models.inception_v3,
                    "effnet": models.efficientnet_b1, "swav": models.resnet50}
    weights = {"alex2": "alexnet-owt-7be5be79.pth", "alex5": "alexnet-owt-7be5be79.pth",
               "inception": "inception_v3_google-0cc3c7bd.pth", "effnet": "efficientnet_b1-c27df63c.pth",
               "swav": "swav_800ep_pretrain.pth.tar"}
    kw = {"init_weights": False} if name == "inception" else {}
    net = constructors[name](weights=None, **kw)
    state = torch.load(archive / "assets/evaluation_v272" / weights[name], map_location="cpu", weights_only=True)
    if name == "swav":
        diff = net.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=False)
        assert set(diff.missing_keys) == {"fc.weight", "fc.bias"} and all(k.startswith(("projection_head.", "prototypes.")) for k in diff.unexpected_keys)
    else:
        net.load_state_dict(state)
    layer = {"alex2": "features.4", "alex5": "features.11"}.get(name, "avgpool")
    return create_feature_extractor(net, [layer]).cuda().eval().requires_grad_(False)


@torch.inference_mode()
def score(folder, panel, archive, output):
    off = archive / "runs/visual_fidelity_20261125_000818/official"
    if panel == "official":
        with (off / "originals_manifest.csv").open() as f:
            records = list(csv.DictReader(f))
        assert [int(r["eeg_row"]) for r in records] == list(range(200))
        rows = np.arange(200)
        members = [r["archive_member"] for r in records]
        labels = [r["unique_id"] for r in records]
        source = off / "test_images.zip"
    else:
        if panel == "fresh":
            rows = np.array(json.loads((PREP / "paper_split.json").read_text())["fresh_rows"])
        else:
            rows = np.array(json.loads((PREP / "paper_development.json").read_text())["dev_rows"])
        assert len(set(rows)) == len(rows) == (500 if panel == "fresh" else 200)
        paths = json.loads((PREP / "train_ids.json").read_text())["paths"]
        members = [paths[i] for i in rows]
        labels = [Path(m).parent.name.split("_", 1)[1] for m in members]
        source = archive / "assets/things_images_v254/training_images.zip"
    patterns = [p for p in ("{:05d}.png", "{:03d}.jpg", "{:03d}.png") if (folder / p.format(0)).is_file()]
    assert len(patterns) == 1, f"Expected one unambiguous row naming scheme in {folder}"
    generated = [folder / patterns[0].format(i) for i in range(len(rows))]
    assert all(p.is_file() for p in generated), f"Incomplete image coverage in {folder}"
    image_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in generated]
    provenance = {"ordered_image_hashes_sha256": hashlib.sha256("\n".join(image_hashes).encode()).hexdigest()}
    if (folder / "render.json").is_file():
        source_bytes = (folder / "render.json").read_bytes()
        meta = json.loads(source_bytes)
        assert meta["pixels_sha256"] == image_hashes and meta["row"] == rows.tolist()
        provenance["render_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    if (folder / "manifest.csv").is_file():
        source_bytes = (folder / "manifest.csv").read_bytes()
        external = list(csv.DictReader(io.StringIO(source_bytes.decode("utf-8-sig"))))
        assert [int(r["eeg_row"]) for r in external] == rows.tolist()
        assert [r["unique_id"] for r in external] == labels
        assert [r["reconstruction_sha256"] for r in external] == image_hashes
        assert [r["reference_sha256"] for r in external] == [r["image_sha256"] for r in records]
        provenance["source_manifest_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    ids, words, inverse, _, _ = load_dictionary(off / "dictionary.csv")
    lookup = {word: i for i, word in enumerate(ids)}
    true = np.array([lookup[s] for s in labels])
    result = {"row": rows, "concept": np.array(labels)}
    with zipfile.ZipFile(source) as z:
        rgb, drawing = [], []
        for i, (m, path) in enumerate(zip(members, generated)):
            data = z.read(m)
            if panel == "official":
                assert hashlib.sha256(data).hexdigest() == records[i]["image_sha256"]
            with Image.open(io.BytesIO(data)) as orig, Image.open(path) as pred:
                rgb.append(prep(orig).mul_(255).round_().byte())
                drawing.append(prep(pred).mul_(255).round_().byte())
        # Keep byte tensors on the host, and stream float batches to the GPU.
    result["pixcorr"], result["ssim"] = [], []
    for a, b in zip(rgb, drawing):
        a, b = [T.Resize(425, interpolation=T.InterpolationMode.BILINEAR)(x.float().div(255)).permute(1, 2, 0).numpy() for x in (a, b)]
        result["pixcorr"].append(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        result["ssim"].append(structural_similarity(rgb2gray(a), rgb2gray(b), data_range=1., gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
    for name in ("clip", "alex2", "alex5", "inception", "effnet", "swav"):
        net = load_metric(archive, name)
        def encode(images):
            return np.concatenate([features(torch.stack(images[i:i+12]).float().div_(255), net, name) for i in range(0, len(images), 12)])
        real, pred = encode(rgb), encode(drawing)
        if name in ("effnet", "swav"):
            result[name + "_distance"] = 1 - (normal(pred, True) * normal(real, True)).sum(1)
        else:
            sim = normal(pred, True) @ normal(real, True).T
            result[name + "_2way"] = (sim < sim.diagonal()[:, None]).sum(1) / (len(rows) - 1)
            if name == "clip":
                result["clip_image_top1"] = topk_credit(sim, 1).diagonal()
                tok = load_tokenizer(archive / "assets/native_generator_v243/turbo/tokenizer")
                bank = text_bank(net, tok, words, 64)
                for prefix, x in (("generated", pred), ("original", real)):
                    logits = (unit(x) @ bank.T)[:, inverse]
                    result[prefix + "_name1854"] = topk_credit(logits, 1)[np.arange(len(rows)), true]
                    result[prefix + "_top_word"] = ids[logits.argmax(1)]
                if panel != "official":
                    within, swaps, raw_within = (np.zeros(len(rows)) for _ in range(3))
                    cosine = normal(pred - pred.mean(0)) @ normal(real - real.mean(0)).T
                    raw = normal(pred) @ normal(real).T
                    result[f"clip_global{len(rows)}_centered"] = topk_credit(cosine, 1).diagonal()
                    for label in set(labels):
                        idx = np.flatnonzero(np.array(labels) == label)
                        assert len(idx) == 10
                        m = cosine[np.ix_(idx, idx)]
                        within[idx] = topk_credit(m, 1).diagonal()
                        raw_within[idx] = topk_credit(raw[np.ix_(idx, idx)], 1).diagonal()
                        diff = m.diagonal()[:, None] + m.diagonal()[None, :] - m - m.T
                        np.fill_diagonal(diff, 0.)
                        swaps[idx] = ((diff > 0).sum(1) + .5 * ((diff == 0).sum(1) - 1)) / 9
                    result["clip_within10"] = within
                    result["clip_within10_uncentered"] = raw_within
                    result["clip_within_swap"] = swaps
        del net, real, pred
        torch.cuda.empty_cache()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "observations.npz"
    np.savez_compressed(destination.with_suffix(".partial.npz"), **result)
    destination.with_suffix(".partial.npz").replace(destination)
    means = {k: float(np.mean(v)) for k, v in result.items() if k != "row" and np.asarray(v).dtype.kind in "biuf"}
    assert all(np.isfinite(list(means.values())))
    write_json(output / "scores.json", {"panel": panel, "n": len(rows), "raw_RGB": True, "mean": means, **provenance,
              "observations_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    print(folder, means, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("panel", choices=("official", "fresh", "dev"))
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    score(a.folder, a.panel, Path(os.environ["ARCHIVE"]), a.output or a.folder)
