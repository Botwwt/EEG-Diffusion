"""Measure foreground mask moments on registered THINGS crops with a fixed U2NetP."""

import argparse
import csv
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_SHA = "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"
ZIP_SHA = "b91edf1a316142406a2ae28bdbb673712f1d5c38f57c0aec9dcfb5da446cabf1"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def measure(session, image):
    from render_official_words import crop512

    crop = crop512(image.convert("RGB")).resize((320, 320), Image.Resampling.LANCZOS)
    pixels = np.asarray(crop, dtype=np.float32)
    norm = (
        pixels / np.maximum(pixels.max(), 1) - np.array([0.485, 0.456, 0.406], np.float32)
    ) / np.array([0.229, 0.224, 0.225], np.float32)
    tensor = np.ascontiguousarray(norm.transpose(2, 0, 1)[None])
    pred = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: tensor})[0][
        0, 0
    ]
    mass = np.maximum(pred - pred.min(), 0)
    if not np.isfinite(mass).all() or mass.max() <= 1e-8:
        raise ValueError("Saliency model produced a degenerate mask")
    mass /= mass.max()
    xy = (np.arange(320, dtype=np.float64) + 0.5) / 320
    axes = np.stack((mass.mean(0), mass.mean(1))).astype(np.float64)
    weights = axes / axes.sum(1, keepdims=True)
    center = weights @ xy
    spread = np.sqrt(np.sum(weights * (xy - center[:, None]) ** 2, axis=1))
    return np.concatenate((center, np.log(np.maximum(spread, 1 / 320))))


def encode(args):
    import onnxruntime as ort
    from prepare_registered_subjects import ROWS_SHA

    if (
        digest(args.model) != MODEL_SHA
        or digest(args.images) != ZIP_SHA
        or digest(args.manifest) != ROWS_SHA
    ):
        raise ValueError("Unexpected stimulus index, images or foreground model")
    with np.load(args.generic, allow_pickle=False) as d:
        rows = d["rows"]
    with np.load(args.target, allow_pickle=False) as d:
        target = d["rows"]
    if len(rows) != 1280 or len(target) != 240 or np.intersect1d(rows // 10, target // 10).size:
        raise ValueError("Expected disjoint calibration and target classes")
    rows = np.unique(np.r_[rows, target])
    with args.manifest.open(encoding="utf8", newline="") as f:
        records = list(csv.DictReader(f))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = False
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=[args.provider])
    if session.get_providers()[0] != args.provider:
        raise RuntimeError("Requested inference provider unavailable")
    out, hashes = [], []
    with zipfile.ZipFile(args.images) as z:
        for i, row in enumerate(rows):
            m = records[row]
            if int(m["train_row"]) != row:
                raise ValueError("Stimulus row index mismatch")
            blob = z.read(f"training_images/{m['official_folder']}/{m['stimulus_stem']}.jpg")
            hashes.append(hashlib.sha256(blob).hexdigest())
            with Image.open(io.BytesIO(blob)) as im:
                out.append(measure(session, im))
            if i % 100 == 0:
                print(i, "of", len(rows), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        rows=rows,
        geometry=np.stack(out),
        jpeg_sha256=hashes,
        keys=["center_x", "center_y", "log_sd_x", "log_sd_y"],
        model_sha256=MODEL_SHA,
        images_sha256=ZIP_SHA,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "images", "manifest", "generic", "target", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--provider",
        choices=("DmlExecutionProvider", "CPUExecutionProvider", "CUDAExecutionProvider"),
        default="CPUExecutionProvider",
    )
    encode(parser.parse_args())
