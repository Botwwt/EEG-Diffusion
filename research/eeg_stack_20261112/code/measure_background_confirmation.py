"""Reuse the frozen original measurement on the registered extra objects or legal image supports."""

import argparse
import csv
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from measure_background_attribute import DESIGN as FIRST
from measure_background_attribute import PUBLIC, attributes, check, digest, mask_input
from PIL import Image
from register_fresh_photo_panel import read, sha
from render_official_words import crop512
from run_prospective_readout import frozen_npz

DESIGN = PUBLIC / "background_attribute_confirmation_design.json"
LOCK = "dcb8a753301787d41fa5b82bfc4e66c93eebe00e88715cd91760686f512b09c4"
PANELS = ("mix13", "mix14", "mix23", "mix24")


def selected_rows(a, d):
    if a.stage == "generic":
        return np.ravel(np.array(d["objects"])[:, None] * 10 + np.arange(10))
    check(read(a.out / "summary.json")["gate_passed"], "Generic confirmation gate closed")
    root = a.data / "fresh-mix/render_inputs/learned"
    check(
        sha(root / "design.json") == d["sources"]["target_render_design"], "Target design changed"
    )
    cfg = read(root / "design.json")
    allowed = set()
    for panel in PANELS:
        path = root / "conditions/sub-04" / panel / "conditions_512.npz"
        check(sha(path) == cfg["packets_sha256"]["sub-04/" + panel], "Wrong target packet")
        with np.load(path, allow_pickle=False) as z:
            for f in range(5):
                active = z["folds"] == f
                support = z["support_rows"][active]
                check(not np.isin(support, z["rows"][active]).any(), "Query in active supports")
                allowed.update(support.ravel().tolist())
    return np.array(sorted(allowed))


def run(a):
    import onnxruntime as ort

    check(sha(DESIGN) == LOCK, "Confirmation registration changed")
    d = read(DESIGN)
    first = read(FIRST)
    check(
        sha(FIRST) == d["sources"]["base_design"]
        and sha(Path(__file__).with_name("measure_background_attribute.py"))
        == d["sources"]["measure_source"],
        "Historical rule changed",
    )
    rows = selected_rows(a, d)
    model = a.data / "photo-structure/u2netp.onnx"
    archive = a.data / "photo-structure/training_images.zip"
    manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    for path, key in ((model, "u2netp"), (archive, "photos"), (manifest, "manifest")):
        check(digest(path) == first["inputs"][key], "Wrong " + key)
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        tab = list(csv.DictReader(f))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.enable_mem_pattern = False
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model), sess_options=opts, providers=[a.provider])
    check(session.get_providers()[0] == a.provider, "Unavailable mask provider")
    inp = session.get_inputs()[0].name
    out = session.get_outputs()[0].name
    with ZipFile(archive) as z:
        for n, obj in enumerate(np.unique(rows // 10)):
            rr = rows[rows // 10 == obj]
            dest = a.out / "annotations" / a.stage / f"{obj:04d}.npz"
            if dest.exists():
                with np.load(dest, allow_pickle=False) as v:
                    check(
                        str(v["design"]) == LOCK
                        and str(v["measure_source"]) == sha(Path(__file__))
                        and np.array_equal(v["rows"], rr),
                        "Stale annotation rows",
                    )
                continue
            code = []
            stats = []
            sums = []
            for row in rr:
                r = tab[row]
                check(int(r["train_row"]) == row, "Registration row order changed")
                blob = z.read(f"training_images/{r['official_folder']}/{r['stimulus_stem']}.jpg")
                sums.append(hashlib.sha256(blob).hexdigest())
                with Image.open(io.BytesIO(blob)) as im:
                    crop = crop512(im.convert("RGB"))
                raw = session.run([out], {inp: mask_input(crop)})[0][0, 0]
                y, st, _ = attributes(
                    np.asarray(crop.resize((320, 320), Image.Resampling.LANCZOS)), raw
                )
                code.append(y)
                stats.append(st)
            frozen_npz(
                dest,
                dict(
                    rows=rr,
                    labels=code,
                    background=stats,
                    jpeg_sha256=sums,
                    design=LOCK,
                    measure_source=sha(Path(__file__)),
                ),
            )
            if n % 100 == 0 or n == 511:
                print(a.stage, n + 1, "objects", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["generic", "supports"])
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("outputs/eeg-background-confirmation"))
    p.add_argument("--provider", default="DmlExecutionProvider")
    run(p.parse_args())
