"""Measure U2NetP soft-mask pose on generic training photos and allowed personal12 support photos."""

import argparse
import csv
import hashlib
import io
import time
import zipfile
from pathlib import Path

import numpy as np
from measure_foreground_geometry import MODEL_SHA, ZIP_SHA, digest
from PIL import Image
from prepare_trial_object_inputs import check
from register_fresh_photo_panel import read, sha
from render_official_words import crop512
from run_prospective_readout import frozen_npz

LOCK = "d77e8c16357a4e779d9f8fdbad096144c53f8a9fb134713a4bc6c16aa309fc4c"
KEYS = ("center_x", "center_y", "log_radius", "anisotropy_xx_yy", "anisotropy_xy")


def mask_input(im):
    crop = crop512(im.convert("RGB")).resize((320, 320), Image.Resampling.LANCZOS)
    pix = np.asarray(crop, dtype=np.float32)
    pix = (pix / np.maximum(pix.max(), 1) - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.ascontiguousarray(pix.transpose(2, 0, 1)[None])


def moments(mask):
    mass = np.maximum(mask - mask.min(), 0)
    if not np.isfinite(mass).all() or mass.max() <= 1e-8:
        raise ValueError("Degenerate saliency")
    mass = (mass / mass.max()).astype(float)
    mass /= mass.sum()
    xy = (np.arange(mass.shape[0], dtype=float) + 0.5) / mass.shape[0]
    x, y = mass.sum(0) @ xy, mass.sum(1) @ xy
    u, v = xy - x, xy - y
    xx, yy = mass.sum(0) @ (u * u), mass.sum(1) @ (v * v)
    xy_cov = v @ mass @ u
    t = xx + yy
    if t <= 0 or xx * yy + 1e-13 < xy_cov**2:
        raise ValueError("Invalid covariance")
    return np.array([x, y, 0.5 * np.log(t), (xx - yy) / t, 2 * xy_cov / t]), np.array(
        [x, y, 0.5 * np.log(xx), 0.5 * np.log(yy)]
    )


def run(a):
    import onnxruntime as ort

    d = read(a.out / "design.json")
    if sha(a.out / "design.json") != LOCK:
        raise ValueError("Wrong pose protocol")
    cache = a.data / "photo-structure"
    model = cache / "u2netp.onnx"
    archive = cache / "training_images.zip"
    manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    check(
        sha(model) == MODEL_SHA
        and digest(archive) == ZIP_SHA
        and sha(manifest) == d["inputs"]["image_row_manifest"],
        "Wrong foreground model, image archive or photo manifest",
    )
    train_manifest = cache / "expanded/features/sub-04.json"
    check(sha(train_manifest) == d["people"]["sub-04"]["generic_source"], "Wrong generic objects")
    gen = read(train_manifest)["training_concepts"]
    generic = np.ravel(np.array(gen)[:, None] * 10 + np.arange(10))
    allowed = set()
    for s in d["subjects"]:
        b = a.data / (
            "trial-object-encoder/personal12" if s == "sub-04" else f"trial-object-replication/{s}"
        )
        f = b / "render_inputs/conditions" / s / "personal12/conditions_512.npz"
        check(sha(f) == d["people"][s]["mixed_conditions"], "Unknown word/support packet")
        with np.load(f) as z:
            for fold in range(5):
                mask = z["folds"] == fold
                r = z["support_rows"][mask].ravel()
                check(not np.isin(r, z["rows"][mask]).any(), "Query included in fold support")
                allowed.update(r.tolist())
    rows = np.unique([*generic, *allowed])
    check(
        len(generic) == len(set(generic)) == 5120 and not set(generic) & allowed,
        "Generic and personal support photos overlap or generic photos are missing",
    )
    with manifest.open(encoding="utf8", newline="") as f:
        records = list(csv.DictReader(f))
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 4
    opt.inter_op_num_threads = 1
    opt.enable_mem_pattern = False
    opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model), sess_options=opt, providers=[a.provider])
    check(session.get_providers()[0] == a.provider, "Requested ONNX provider is unavailable")
    inp = session.get_inputs()[0].name
    out = session.get_outputs()[0].name
    legacy_file = cache / "foreground_geometry.npz"
    check(sha(legacy_file) == d["inputs"]["legacy_geometry"], "Changed prior geometry cache")
    with np.load(legacy_file) as z:
        old = dict(zip(z["rows"], z["geometry"], strict=True))
    output = []
    checks = []
    errors = []
    since = time.monotonic()
    with zipfile.ZipFile(archive) as z:
        for n, row in enumerate(rows):
            m = records[row]
            check(int(m["train_row"]) == row, "Stimulus ordering differs")
            data = z.read(f"training_images/{m['official_folder']}/{m['stimulus_stem']}.jpg")
            checks.append(hashlib.sha256(data).hexdigest())
            with Image.open(io.BytesIO(data)) as im:
                x = mask_input(im)
            pred = session.run([out], {inp: x})[0][0, 0]
            value, prior = moments(pred)
            if row in old:
                errors.append(float(np.abs(prior - old[row]).max()))
                if errors[-1] > 0.002:
                    raise ValueError(f"Prior saliency cache disagreement on {row}: {errors[-1]}")
            output.append(value)
            if n + 1 in (10, 100) or (n + 1) % 500 == 0:
                print(n + 1, len(rows), round(time.monotonic() - since, 2), flush=True)
    frozen_npz(
        a.out / "sources.npz",
        dict(
            rows=rows,
            pose=np.stack(output),
            keys=np.array(KEYS),
            jpeg_sha256=np.array(checks),
            model_sha256=MODEL_SHA,
            archive_sha256=ZIP_SHA,
            registration_sha256=LOCK,
            generic_rows=generic,
            allowed_support_rows=np.array(sorted(allowed), dtype=int),
            prior_mask_overlap=len(errors),
            max_prior_moment_error=max(errors),
            provider=a.provider,
        ),
    )
    print(
        "Measured", len(rows), "matched historical", len(errors), "max_abs", max(errors), flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("outputs/personal-pose"))
    p.add_argument("--provider", default="DmlExecutionProvider")
    run(p.parse_args())
