"""Measure a preregistered clean-light versus textured background on generic photos."""

import argparse
import csv
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from measure_foreground_geometry import digest
from measure_personal_pose import mask_input
from PIL import Image
from register_fresh_photo_panel import read, sha
from render_official_words import crop512
from run_prospective_readout import frozen_npz
from scipy.ndimage import minimum_filter

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
DESIGN = PUBLIC / "background_attribute_design.json"
LOCK = "b2c55b8b6a92e35f019bb3268daa59f9edb283c52bdd82ac2995a9cc7406d06c"
LUMA = np.array([0.2126, 0.7152, 0.0722])


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def attributes(rgb, raw):
    difference = raw - raw.min()
    check(np.isfinite(difference).all() and difference.max() > 1e-8, "Invalid saliency map")
    outside = minimum_filter(difference <= 0.10 * difference.max(), size=9, mode="nearest")
    good = outside[::2, ::2]
    rgb = rgb[::2, ::2].astype(float) / 255
    luma = rgb @ LUMA
    appearance = np.r_[rgb.mean(axis=(0, 1)), luma.std()]
    if good.mean() < 0.15:
        return 0.5, [good.mean(), 0, 0, 0], appearance
    neutral = (luma >= 0.75) & (np.ptp(rgb, axis=2) <= 0.20)
    light = neutral[good].mean()
    std = luma[good].std()
    pairs = [good[1:] & good[:-1], good[:, 1:] & good[:, :-1]]
    edge = sum(
        ((np.abs(np.diff(luma, axis=i)) >= 0.06) & valid).sum() for i, valid in enumerate(pairs)
    ) / sum(valid.sum() for valid in pairs)
    if light >= 0.75 and std <= 0.12 and edge <= 0.12:
        value = 1.0
    elif light <= 0.40 and (std >= 0.16 or edge >= 0.15):
        value = 0.0
    else:
        value = 0.5
    return value, [good.mean(), light, std, edge], appearance


def measure(a):
    import onnxruntime as ort

    check(sha(DESIGN) == LOCK, "Changed background protocol")
    d = read(DESIGN)
    path = a.data / "photo-structure/training_images.zip"
    model = a.data / "photo-structure/u2netp.onnx"
    manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    for file, key in ((manifest, "manifest"), (model, "u2netp"), (path, "photos")):
        check(digest(file) == d["inputs"][key], "Changed " + key)
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.enable_mem_pattern = False
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model), sess_options=opts, providers=[a.provider])
    check(session.get_providers()[0] == a.provider, "Requested mask backend unavailable")
    inp, out = session.get_inputs()[0].name, session.get_outputs()[0].name
    ids = d["fit_classes"] + d["calibration_classes"] + d["evaluation_classes"]
    check(len(set(ids)) == len(ids) == 512, "Generic split overlap")
    with ZipFile(path) as z:
        for n, c in enumerate(ids):
            dest = a.out / "annotations" / f"{c:04d}.npz"
            if dest.is_file():
                with np.load(dest, allow_pickle=False) as cached:
                    check(
                        str(cached["design"]) == LOCK
                        and str(cached["source_sha256"]) == sha(Path(__file__)),
                        "Stale annotation cache",
                    )
                continue
            rows = np.arange(c * 10, c * 10 + 10)
            labels, stats, rgb, hashes = [], [], [], []
            for row in rows:
                r = records[row]
                check(int(r["train_row"]) == row, "Photo index changed")
                blob = z.read(f"training_images/{r['official_folder']}/{r['stimulus_stem']}.jpg")
                hashes.append(hashlib.sha256(blob).hexdigest())
                with Image.open(io.BytesIO(blob)) as im:
                    crop = crop512(im.convert("RGB"))
                    x = mask_input(crop)
                raw = session.run([out], {inp: x})[0][0, 0]
                pixels = np.asarray(crop.resize((320, 320), Image.Resampling.LANCZOS))
                value, stat, app = attributes(pixels, raw)
                labels.append(value)
                stats.append(stat)
                rgb.append(app)
            frozen_npz(
                dest,
                dict(
                    rows=rows,
                    labels=labels,
                    background=stats,
                    appearance=rgb,
                    jpeg_sha256=hashes,
                    design=LOCK,
                    source_sha256=sha(Path(__file__)),
                ),
            )
            if n % 50 == 0 or n == 511:
                print("Generic objects measured", n + 1, "of", len(ids), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/eeg-background-attribute"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--provider", default="DmlExecutionProvider")
    measure(p.parse_args())
