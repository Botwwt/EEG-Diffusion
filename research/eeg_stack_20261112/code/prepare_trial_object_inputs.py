"""Prepare sub-04's separate THINGS-EEG2 trials for 512 training and 24 old test objects.

Each <concept:04d>.npy is a contiguous, mmap-compatible float32 array with shape
[10 photos, 4 recorded trials, 63 original channels, 125 times at 250 Hz].
The time window is [0, 500) milliseconds. No interpolation or averaging is used.

Use the existing prepare_expanded_personal, prepare_double_calibration and
test_cohort_session_matched caches for sub-04. Their pinned per-object digests
verify every source range, including newly downloaded ranges. The header and
footer come from prepare_registered_subjects and the official training image
row manifest comes from fit_official_eeg_anchors. Run with --group generic128
--offline for the original calibration cache or with --group all for the
complete set. The all-group receipt keeps the training and evaluation IDs separate.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from prepare_registered_subjects import (
    OFFSET,
    REPO,
    REV,
    ROW_BYTES,
    ROWS_SHA,
    SHAPE,
    SIZE,
    axes,
    http,
)

CODE = Path(__file__).resolve().parent
PUBLIC = CODE.parent / "results/object_fidelity"
DATA = CODE.parents[2] / "outputs"
SUB = "sub-04"
DIR = "photo-structure/expanded/features"
BLOCK_BYTES = 10 * ROW_BYTES
OUTPUT_SHAPE = (10, 4, 63, 125)
SOURCE_SHA = "a658b1a82504f6f426a53a4a3ac28601b9988ec109b676d85b211a57a13d3e27"
RECEIPTS = {
    DIR: "5814b21bf3fe3a0b6fe46c3db7cb11025029057a8044deb5fd25e64b47390bd0",
    "double-calibration/features": "3c092f32df53e329d35c8c7c1666a397d66db50bb8978168f725eb51f506a43c",
    "session-matched-cohort/features": "32f971485143b1f84ac492e6434068be926e88be56368c5a73034e0e4558ed18",
}
DESIGNS = {
    "double_calibration_design.json": "f30831aab67d1b735985876710c51eddc3bd60729c0ccc72e803c02f7f34088d",
    "cohort_session_matched_design.json": "ebe8c6ec8710ec8a4886e095b89a87ce479bc42b38ad484d5c829aa0137b6599",
}


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verified_json(path, expected):
    data = path.read_bytes()
    check(sha(data) == expected, f"Different registration at {path}")
    return json.loads(data)


def atomic(path, data):
    with NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            f.write(data)
            f.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def write_json(path, data):
    payload = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    if path.exists():
        check(path.read_bytes() == payload, f"Existing receipt differs at {path}")
    else:
        atomic(path, payload)


def inputs(root):
    double, cohort = [verified_json(PUBLIC / name, checksum) for name, checksum in DESIGNS.items()]
    sources = [verified_json(root / d / (SUB + ".json"), h) for d, h in RECEIPTS.items()]
    old, extra, strict = sources
    groups = {
        "generic128": old["original_concepts"],
        "generic512": double["old_objects"],
        "stress12": cohort["panels"]["stress12"]["selected_concepts"],
        "new_three4": cohort["panels"]["new_three4"]["selected_concepts"],
    }
    classes = groups["generic512"] + groups["stress12"] + groups["new_three4"]
    check(
        len(set(classes)) == len(classes) == 536
        and groups["generic512"] == old["training_concepts"]
        and len(set(groups["generic128"])) == 128
        and set(groups["generic128"]) <= set(groups["generic512"])
        and not set(classes).intersection(double["fifth_objects"]),
        "Invalid training/panel partition",
    )
    source = f"{REPO}/resolve/{REV}/Preprocessed_data_250Hz/{SUB}/preprocessed_eeg_training.npy"
    check(
        all(x["subject"] == SUB and x["source"] == source for x in sources)
        and old["source_full_lfs_sha256"]
        == extra["source_full_lfs_sha256"]
        == strict["official_full_source_sha256"]
        == SOURCE_SHA,
        "Different official release",
    )
    expected = {c: old["raw_block_sha256"][str(c)] for c in groups["generic512"]}
    for k, r in (("stress12", extra), ("new_three4", strict)):
        expected.update({c: r["raw_sha256"][str(c)] for c in groups[k]})
    cached = root / "object-transfer/ranges" / REV / SUB
    tail = OFFSET + SHAPE[0] * ROW_BYTES
    channels, t = axes(
        (cached / f"0-{OFFSET - 1}.bin").read_bytes(),
        (cached / f"{tail}-{SIZE - 1}.bin").read_bytes(),
    )
    check(np.allclose(t[50:175], np.arange(125) / 250, atol=1e-7, rtol=0), "Time axes changed")
    buf = (root / "official-object-readout/sub-03/atm_train_rows.csv").read_bytes()
    check(sha(buf) == ROWS_SHA, "Image row ordering changed")
    table = list(csv.DictReader(io.StringIO(buf.decode("utf-8-sig"))))
    check([int(row["train_row"]) for row in table] == list(range(16540)), "Missing training rows")
    for c in classes:
        rows = table[c * 10 : (c + 1) * 10]
        check(
            len({r["unique_id"] for r in rows}) == 1
            and len({r["stimulus_stem"] for r in rows}) == 10,
            "Ten distinct photographs per concept required",
        )
    return groups, expected, cached, source, channels.tolist(), table


def convert(concept, expected, cached, source, args):
    start = OFFSET + concept * BLOCK_BYTES
    interval = f"{start}-{start + BLOCK_BYTES - 1}"
    files = [cached / (interval + ".bin")]
    if args.raw_cache:
        files.append(args.raw_cache / f"{concept:04d}.bin")
    available = next((f for f in files if f.is_file()), None)
    if available:
        raw = available.read_bytes()
    else:
        if args.offline:
            raise FileNotFoundError(
                f"Missing verified local 63-channel block for concept {concept}"
            )
        raw = http(
            source + "?range=" + interval,
            {"Range": "bytes=" + interval},
            BLOCK_BYTES,
            f"bytes {interval}/{SIZE}",
        )
    check(len(raw) == BLOCK_BYTES and sha(raw) == expected, f"Wrong source block for {concept}")
    eeg = np.frombuffer(raw, dtype="<f8").reshape(10, *SHAPE[1:])[:, :, :, :125]
    cropped = np.ascontiguousarray(eeg, dtype="<f4")
    check(np.isfinite(cropped).all(), f"Nonfinite EEG for {concept}")
    buffer = io.BytesIO()
    np.save(buffer, cropped, allow_pickle=False)
    payload = buffer.getvalue()
    file = args.out / f"{concept:04d}.npy"
    if file.exists():
        check(file.read_bytes() == payload, f"Existing trial packet differs for {concept}")
    else:
        atomic(file, payload)
    return dict(source_raw_sha256=expected, output_sha256=sha(payload))


def run(args):
    groups, expected, cached, source, channels, table = inputs(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.group == "all":
        names = ["generic128", "generic512", "stress12", "new_three4"]
    else:
        names = [args.group]
    meta = dict(
        subject=SUB,
        source=source,
        source_full_lfs_sha256=SOURCE_SHA,
        revision=REV,
        training_manifest_sha256=ROWS_SHA,
        time_source="Original 250 post-onset bins, first 125 retained",
        times_seconds=(np.arange(125) / 250).tolist(),
        channel_names=channels,
        shape=list(OUTPUT_SHAPE),
        dtype="<f4",
        order=["photo", "physical_trial", "channel", "time"],
        source_axes=list(SHAPE),
        source_dtype="<f8",
        source_tensor_offset=OFFSET,
        acquisition="Four distinct upstream MVNN-preprocessed trials in stored order. No trial averaging.",
        source_receipts_sha256=RECEIPTS,
        design_sha256=DESIGNS,
    )
    completed = {}
    for name in names:
        ids = groups[name]
        with ThreadPoolExecutor(args.workers) as pool:
            futures = {
                pool.submit(convert, c, expected[c], cached, source, args): c
                for c in ids
                if c not in completed
            }
            for n, f in enumerate(as_completed(futures), 1):
                completed[futures[f]] = f.result()
                if n % 32 == 0:
                    print(name, n, "/", len(futures), flush=True)
        cases = {
            str(c): dict(
                **completed[c], official_row_start=c * 10, unique_id=table[c * 10]["unique_id"]
            )
            for c in ids
        }
        write_json(
            args.out / (name + ".json"),
            dict(
                **meta,
                split=name,
                concepts=ids,
                n_photos=len(ids) * 10,
                n_presentations=len(ids) * 40,
                blocks=cases,
            ),
        )
        print("Ready", name, len(ids), flush=True)
    if args.group == "all":
        write_json(
            args.out / "receipt.json",
            dict(
                **meta,
                n_unique_concepts=len(completed),
                n_unique_photos=len(completed) * 10,
                n_presentations=len(completed) * 40,
                split_receipts_sha256={
                    name: sha((args.out / (name + ".json")).read_bytes()) for name in names
                },
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=["all", "generic128", "generic512", "stress12", "new_three4"],
        default="all",
    )
    parser.add_argument("--workers", type=int, choices=range(1, 6), default=5)
    parser.add_argument(
        "--offline", action="store_true", help="Only read locally available full trial blocks"
    )
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument(
        "--raw-cache", type=Path, help="Optional predownloaded <concept:04d>.bin ranges"
    )
    parser.add_argument("--out", type=Path, default=DATA / "trial-object-inputs/sub-04")
    run(parser.parse_args())
