"""Read the two registered R4 panels from each participant's pinned public training EEG.

HTTP ranges cover the serialized header, metadata tail and 24 ten-photo blocks.
The EEG container is a protocol-4 pickle despite its .npy filename. Inspect its
opcodes as data and decode only fixed-width numeric byte strings. Never unpickle.
"""

import argparse
import csv
import hashlib
import json
import pickletools
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from decode_eeg_layout import CHANNELS
from fit_registered_objects import WORDS

REV = "dbe34bb2407164f70883c661c76664f0d596e522"
REPO = "https://huggingface.co/datasets/LidongYang/EEG_Image_decode"
API = "https://huggingface.co/api/datasets/LidongYang/EEG_Image_decode"
TIME_SOURCE = "https://github.com/dongyangli-del/EEG_Image_decode/blob/49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d/Retrieval/eegdatasets_leaveone.py#L161"
HEADER_SHA = "ddc7610cd64c87ba999d6e7fbc80307053844be1c71373b9ea3e370e34f983bf"
TAIL_SHA = "7c81b4d1402994dd1c01155248a6a7969130bd64bc06e611d77a639dd8b0afae"
ROWS_SHA = "dafc55af7366974feae0e11418680990ff8d74707ec5b89e398fd5a26567c473"
SHAPE = (16540, 4, 63, 250)
ROW_BYTES = 4 * 63 * 250 * 8
SIZE = 8_336_163_007
OFFSET = 184
ORDER_REV = "101997d4f14c41b615bcaf8c0907740106d260c5"
ORDER_SOURCE = "https://github.com/dongyangli-del/EEG_Image_decode/blob/49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d/EEG-preprocessing/"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_bytes(data)
    tmp.replace(path)


def store_json(path, data):
    write(path, (json.dumps(data, indent=2, allow_nan=False) + "\n").encode())


def registered_rows(args):
    if digest(args.train_rows.read_bytes()) != ROWS_SHA:
        raise ValueError("Expected the pinned training-row export of fit_official_eeg_anchors.py")
    with args.train_rows.open(encoding="utf8", newline="") as f:
        ordered = list(csv.DictReader(f))
    labels = np.array([r["unique_id"] for r in ordered])
    if [int(r["train_row"]) for r in ordered] != list(range(SHAPE[0])):
        raise ValueError("Invalid public training row index")
    objects = labels[::10]
    if len(set(objects)) != 1654 or not np.array_equal(np.repeat(objects, 10), labels):
        raise ValueError("Expected ten photos per original training concept")
    eligible = np.flatnonzero(~np.isin(objects, WORDS))
    replica = objects[np.random.default_rng(260426).choice(eligible, len(WORDS), replace=False)]
    for path, panel in ((args.registered, list(WORDS)), (args.replication, replica.tolist())):
        if path is None:
            continue
        data = json.loads(path.read_text())
        words = np.array(data["words"])
        rows = np.array(data["rows"])
        if (
            data["words"] != panel
            or not np.array_equal(np.sort(rows), np.flatnonzero(np.isin(labels, words)))
            or not np.array_equal(labels[rows], words[data["truth"]])
        ):
            raise ValueError("Registered labels and original public training rows disagree")
    rows = np.flatnonzero(np.isin(labels, [*WORDS, *replica]))
    if rows.shape != (240,) or not np.all(np.diff(rows.reshape(24, 10), axis=1) == 1):
        raise ValueError("Expected 24 complete ten-photo blocks")
    return rows, labels[rows], np.array([ordered[r]["stimulus_stem"] for r in rows])


def http(url, headers, expected, content_range=None):
    for attempt in range(5):
        try:
            req = Request(url, headers={**headers, "Accept-Encoding": "identity"})
            with urlopen(req, timeout=65) as r:
                if content_range and (
                    r.status != 206 or r.headers.get("Content-Range") != content_range
                ):
                    raise ValueError(f"Expected exact HTTP 206 {content_range}")
                raw = r.read(expected + 1)
            if (content_range and len(raw) != expected) or len(raw) > expected:
                raise ValueError(
                    f"Incorrect response length for {content_range or 'file metadata'}"
                )
            return raw
        except (TimeoutError, OSError, ValueError) as exc:
            if attempt == 4:
                raise
            print(f"Retry {attempt + 1} for {content_range or 'HF metadata'}: {exc}", flush=True)
            time.sleep(1 + attempt)


def ranged(source, first, last, cache):
    key = f"{first}-{last}"
    path = cache / (key + ".bin")
    receipt = cache / (key + ".sha256")
    if path.is_file() and receipt.is_file():
        raw = path.read_bytes()
        if len(raw) == last - first + 1 and digest(raw) == receipt.read_text().strip():
            return raw
    raw = http(
        source + "?range=" + key, {"Range": "bytes=" + key}, last - first + 1, f"bytes {key}/{SIZE}"
    )
    write(path, raw)
    write(receipt, digest(raw).encode())
    return raw


def axes(head, tail):
    # These pinned bytes encode the shape, little-endian f8 dtype, False
    # Fortran-order flag and one BINBYTES8 argument. Its body follows OFFSET.
    length = int(np.prod(SHAPE)) * 8
    if (
        digest(head) != HEADER_SHA
        or digest(tail) != TAIL_SHA
        or head[-9:] != b"\x8e" + struct.pack("<Q", length)
    ):
        raise ValueError("Unexpected pickle metadata, tensor dtype/order or byte span")
    sanitized = head[:-8] + bytes(8) + tail
    ops = [(o.name, arg) for o, arg, _ in pickletools.genops(sanitized)]
    names = [v for k, v in ops if k == "SHORT_BINUNICODE"]
    channels = names[names.index("ch_names") + 1 : names.index("times")]
    (raw_times,) = [v for k, v in ops if k == "BINBYTES"]
    original_times = np.frombuffer(raw_times, dtype="<f8").copy()
    if len(set(channels)) != 63 or not np.allclose(original_times, np.arange(-50, 250) / 250):
        raise ValueError("Unexpected channel names or sample times")
    return np.array(channels), original_times


def subject_packet(number, rows, labels, stems, args):
    sub = f"sub-{number:02}"
    dest = args.out / f"{sub}.npz"
    report = args.out / f"{sub}.json"
    if dest.is_file() and report.is_file():
        saved = json.loads(report.read_text())
        if (
            saved.get("subject") == sub
            and saved.get("revision") == REV
            and saved.get("train_rows_sha256") == ROWS_SHA
            and saved.get("packet_sha256") == digest(dest.read_bytes())
            and saved.get("rows") == rows.tolist()
        ):
            print(sub, "verified existing packet", flush=True)
            return
    folder = f"Preprocessed_data_250Hz/{sub}"
    key = folder + "/preprocessed_eeg_training.npy"
    source = f"{REPO}/resolve/{REV}/{key}"
    cache = args.out / "ranges" / REV / sub
    cache.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(http(f"{API}/tree/{REV}/{folder}?expand=true", {}, 100_000))
    (entry,) = [m for m in metadata if m["path"] == key]
    if entry["size"] != SIZE or entry["lfs"]["size"] != SIZE or len(entry["lfs"]["oid"]) != 64:
        raise ValueError("Unexpected pinned HF object size or LFS identifier")
    end = OFFSET + SHAPE[0] * ROW_BYTES
    head = ranged(source, 0, OFFSET - 1, cache)
    tail = ranged(source, end, SIZE - 1, cache)
    channels, source_times = axes(head, tail)
    means = np.empty((len(rows), 63, 250), np.float32)
    blocks = {}

    def fetch(i):
        start, stop = int(rows[i]), int(rows[i + 9]) + 1
        first, last = OFFSET + start * ROW_BYTES, OFFSET + stop * ROW_BYTES - 1
        raw = ranged(source, first, last, cache)
        eeg = np.frombuffer(raw, dtype="<f8").reshape(10, *SHAPE[1:])
        if not np.isfinite(eeg).all():
            raise ValueError(f"Invalid EEG for {sub} rows {start} to {stop - 1}")
        avg = eeg.astype(np.float32).mean(axis=1, dtype=np.float32)
        return i, avg, dict(rows=[start, stop - 1], bytes=[first, last], sha256=digest(raw))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(fetch, i) for i in range(0, len(rows), 10)]
        for task in as_completed(tasks):
            i, avg, rec = task.result()
            means[i : i + 10] = avg
            blocks[i] = rec
            if len(blocks) % 6 == 0:
                print(sub, f"{len(blocks)}/24 checked blocks", flush=True)
    tmp = dest.with_suffix(".partial.npz")
    np.savez(
        tmp,
        rows=rows,
        mean_eeg=means,
        channels=channels,
        times=source_times[50:],
        source_times=source_times,
        subject=sub,
        repeats=4,
        unique_ids=labels,
        stimulus_stems=stems,
    )
    tmp.replace(dest)
    store_json(
        report,
        dict(
            subject=sub,
            revision=REV,
            source=source,
            source_size=SIZE,
            source_full_lfs_sha256=entry["lfs"]["oid"],
            source_tree_blob=entry["oid"],
            full_file_hashed_locally=False,
            range_validation="HTTP 206 with exact Content-Range and byte count; per-block SHA-256",
            header_sha256=HEADER_SHA,
            tail_sha256=TAIL_SHA,
            tensor_offset=OFFSET,
            serialized_shape=SHAPE,
            dtype="<f8",
            array_order="C",
            axes=["training_image", "trial", "channel", "time"],
            time_source=TIME_SOURCE,
            times="250 samples at 250 Hz, use source_times[50:]",
            preprocessing="Released preprocessed values, float32 arithmetic mean of four trials; source units retained",
            train_rows_sha256=ROWS_SHA,
            rows=rows.tolist(),
            row_ranges=[blocks[k] for k in sorted(blocks)],
            data_bytes=len(rows) * ROW_BYTES,
            packet_sha256=digest(dest.read_bytes()),
        ),
    )
    print(sub, "READY", dest, flush=True)


def check_packets(numbers, rows, labels, stems, root):
    """Rebuild all means from saved numeric ranges, without network access."""
    results = {}
    for number in numbers:
        sub = f"sub-{number:02}"
        path = root / f"{sub}.npz"
        receipt = json.loads(path.with_suffix(".json").read_text())
        cache = root / "ranges" / REV / sub
        end = OFFSET + SHAPE[0] * ROW_BYTES
        channels, times = axes(
            (cache / f"0-{OFFSET - 1}.bin").read_bytes(),
            (cache / f"{end}-{SIZE - 1}.bin").read_bytes(),
        )
        with np.load(path, allow_pickle=False) as data:
            if (
                receipt["packet_sha256"] != digest(path.read_bytes())
                or receipt["subject"] != sub
                or receipt["revision"] != REV
                or receipt["train_rows_sha256"] != ROWS_SHA
                or str(data["subject"]) != sub
                or int(data["repeats"]) != 4
                or not np.array_equal(data["rows"], rows)
                or not np.array_equal(data["stimulus_stems"], stems)
                or not np.array_equal(data["unique_ids"], labels)
                or not np.array_equal(data["channels"], channels)
                or not np.array_equal(data["times"], times[50:])
            ):
                raise ValueError(f"{sub} packet registration differs from its source")
            eeg = data["mean_eeg"]
        if eeg.shape != (240, 63, 250) or eeg.dtype != np.float32 or not np.isfinite(eeg).all():
            raise ValueError(f"{sub} invalid mean EEG")
        for group, block in zip(rows.reshape(-1, 10), receipt["row_ranges"], strict=True):
            first, last = int(group[0]), int(group[-1])
            begin, end = OFFSET + first * ROW_BYTES, OFFSET + (last + 1) * ROW_BYTES - 1
            blob = (cache / f"{begin}-{end}.bin").read_bytes()
            if block != dict(rows=[first, last], bytes=[begin, end], sha256=digest(blob)):
                raise ValueError(f"{sub} range identity mismatch at {first}")
            trials = np.frombuffer(blob, dtype="<f8").reshape(10, 4, 63, 250).astype(np.float32)
            fixed_mean = (trials[:, 0] + trials[:, 1] + trials[:, 2] + trials[:, 3]) / 4
            if not np.array_equal(fixed_mean, eeg[np.searchsorted(rows, group)]):
                raise ValueError(f"{sub} mean differs from its four source trials at {first}")
        results[sub] = dict(
            packet_sha256=receipt["packet_sha256"],
            source_full_lfs_sha256=receipt["source_full_lfs_sha256"],
            waveform_sha256=digest(eeg.tobytes()),
            rows=len(rows),
            raw_ranges=len(receipt["row_ranges"]),
        )
    if len({v["waveform_sha256"] for v in results.values()}) != len(results):
        raise ValueError("Two participant packets contain the same waveform tensor")
    return dict(
        revision=REV,
        check="Exact float32 recomputation from four trials for every source row and channel",
        subjects=results,
    )


def session_halves(number, rows, root, schedule_path):
    """Recover the two-trial mean from each of a photo's two original sessions."""
    sub = f"sub-{number:02}"
    receipt = json.loads((root / f"{sub}.json").read_text())
    source = root / f"{sub}.npz"
    if receipt["packet_sha256"] != digest(source.read_bytes()) or receipt["subject"] != sub:
        raise ValueError(f"{sub} invalid four-repeat source")
    with np.load(source, allow_pickle=False) as data:
        if not np.array_equal(rows, data["rows"]) or str(data["subject"]) != sub:
            raise ValueError(f"{sub} stimulus rows or subject changed")
        posterior = np.array([data["channels"].tolist().index(ch) for ch in CHANNELS])
        four = np.take(data["mean_eeg"], posterior, axis=1)
        times = data["times"]
    cache = root / "ranges" / REV / sub
    end = OFFSET + SHAPE[0] * ROW_BYTES
    channels, all_times = axes(
        (cache / f"0-{OFFSET - 1}.bin").read_bytes(), (cache / f"{end}-{SIZE - 1}.bin").read_bytes()
    )
    if not np.array_equal(times, all_times[50:]) or not np.array_equal(
        channels[posterior], CHANNELS
    ):
        raise ValueError("Packet time or channel axis differs from the public record")
    schedule_bytes = schedule_path.read_bytes()
    schedule = json.loads(schedule_bytes)
    ids = np.asarray(schedule["subjects"][sub]["session_pairs"], dtype=np.int64)[rows // 10].T
    if (
        schedule["revision"] != ORDER_REV
        or ids.shape != (2, len(rows))
        or np.any(ids[0] >= ids[1])
        or np.min(ids) < 1
        or np.max(ids) > 4
    ):
        raise ValueError(f"{sub} invalid chronological session pair")
    # save_prepr concatenates the two trials from each session in chronological
    # order, then applies a shared sklearn shuffle to the four-slot axis.
    order = np.random.RandomState(20200220).permutation(4)
    early, late = np.flatnonzero(order < 2), np.flatnonzero(order >= 2)
    result = np.empty((2, len(rows), len(posterior), len(times)), dtype=np.float32)
    diffs = []
    for n, (group, rec) in enumerate(zip(rows.reshape(-1, 10), receipt["row_ranges"], strict=True)):
        first, last = int(group[0]), int(group[-1])
        begin, end = OFFSET + first * ROW_BYTES, OFFSET + (last + 1) * ROW_BYTES - 1
        raw = (cache / f"{begin}-{end}.bin").read_bytes()
        if rec != dict(rows=[first, last], bytes=[begin, end], sha256=digest(raw)):
            raise ValueError(f"{sub} raw range changed")
        x = np.take(
            np.frombuffer(raw, dtype="<f8").reshape(10, *SHAPE[1:]), posterior, axis=2
        ).astype(np.float32)
        ix = slice(10 * n, 10 * n + 10)
        result[0, ix] = x[:, early].mean(axis=1)
        result[1, ix] = x[:, late].mean(axis=1)
        difference = np.abs(result[:, ix].mean(axis=0) - four[ix])
        bound = 4 * np.finfo(np.float32).eps * np.abs(x).max(axis=1)
        if np.any(difference > bound) or not np.isfinite(difference).all():
            raise ValueError(f"Early/late means disagree with {sub}'s true four-trial mean")
        diffs.append(float(difference.max()))
    dest = root / "halves" / f"{sub}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".partial.npz")
    np.savez(
        tmp,
        rows=rows,
        half_eeg=result,
        channels=np.array(CHANNELS),
        times=times,
        half_names=["early", "late"],
        session_ids=ids,
        stored_slots=np.stack([early, late]),
        subject=sub,
        repeats_per_half=2,
    )
    tmp.replace(dest)
    store_json(
        dest.with_suffix(".json"),
        dict(
            subject=sub,
            parent_packet_sha256=receipt["packet_sha256"],
            packet_sha256=digest(dest.read_bytes()),
            source_revision=REV,
            schedule_sha256=digest(schedule_bytes),
            order_source=ORDER_SOURCE + "preprocessing_utils.py#L262-L299",
            order_source_sha256="08a99fc9a049752ccf7c142e3c7a1753e4c8e1547df4af82b700260dbeb49163",
            seed_source=ORDER_SOURCE + "preprocessing.py#L52",
            seed_source_sha256="69ce0b307ee6a9012c421a07ca4cb64b3052413b3237a247d2340a7bc7dbd9ab",
            interpretation="Two planned sessions per photo, early and late relative to that photo. Slot assignment follows the released ATM pickle writer; session IDs follow the published stimulus presentation schedules.",
            original_slot_at_stored_slot=order.tolist(),
            stored_slots=[early.tolist(), late.tolist()],
            maximum_absolute_recombination_error=max(diffs),
            max_error_by_block=diffs,
            n_checked_blocks=len(diffs),
            tolerance="4 float32 eps times per-point largest absolute physical trial value",
        ),
    )
    print(
        sub,
        "HALVES",
        dest,
        f"all {len(diffs)} blocks validated; max |difference|={max(diffs):.3g}",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-rows",
        type=Path,
        default=Path("outputs/official-object-readout/sub-03/atm_train_rows.csv"),
    )
    p.add_argument(
        "--registered", type=Path, help="Also cross-check the first panel's fitted evaluation.json"
    )
    p.add_argument(
        "--replication",
        type=Path,
        help="Also cross-check the second panel's fitted evaluation.json",
    )
    p.add_argument("--out", type=Path, default=Path("outputs/object-transfer"))
    p.add_argument("--subjects", type=int, nargs="+", default=[3, 4], choices=range(1, 11))
    p.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--session-halves",
        action="store_true",
        help="Offline: split cached physical trials by the two planned EEG sessions per photo",
    )
    p.add_argument(
        "--sessions",
        type=Path,
        help="Verified audit_registered_sessions.py sessions.json (defaults to --out/sessions.json)",
    )
    group.add_argument(
        "--check-only",
        type=Path,
        help="Offline: verify existing packets against every cached source range and write a JSON report here",
    )
    args = p.parse_args()
    rows, labels, stems = registered_rows(args)
    if args.check_only:
        store_json(args.check_only, check_packets(args.subjects, rows, labels, stems, args.out))
        print("Verified", len(args.subjects), "participants")
    elif args.session_halves:
        for number in args.subjects:
            session_halves(number, rows, args.out, args.sessions or args.out / "sessions.json")
    else:
        for number in args.subjects:
            subject_packet(number, rows, labels, stems, args)
