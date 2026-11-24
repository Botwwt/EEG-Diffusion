"""Fetch a fixed disjoint training panel for sub-04 and sub-05 from public ATM EEG.

Requires the subject's audited registered-panel packet, frozen presentation
schedule and common public training-row export. Writes only to generic/ and
the pinned HTTP byte-range cache. The 128 classes are fixed before reading EEG.
"""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from decode_eeg_layout import CHANNELS
from prepare_registered_subjects import (
    OFFSET,
    ORDER_REV,
    ORDER_SOURCE,
    REPO,
    REV,
    ROW_BYTES,
    ROWS_SHA,
    SHAPE,
    SIZE,
    axes,
    digest,
    ranged,
    registered_rows,
    store_json,
)

SEED = 260431


def select(args):
    excluded_rows, _, _ = registered_rows(args)
    excluded = np.unique(excluded_rows // 10)
    candidates = np.setdiff1d(np.arange(1654), excluded)
    objects = np.sort(np.random.default_rng(SEED).choice(candidates, 128, replace=False))
    rows = (10 * objects[:, None] + np.arange(10)).ravel()
    with args.train_rows.open(encoding="utf8", newline="") as f:
        manifest = list(csv.DictReader(f))
    names = np.array([manifest[r]["unique_id"] for r in rows])
    stems = np.array([manifest[r]["stimulus_stem"] for r in rows])
    if (
        len(excluded) != 24
        or len(set(names)) != 128
        or len(set(stems)) != 1280
        or np.intersect1d(rows, excluded_rows).size
    ):
        raise ValueError("Generic row registration disagrees with the 24 excluded objects")
    return rows, objects, excluded, names, stems


def prepare(subject, selection, args):
    rows, objects, excluded, labels, stems = selection
    sub = f"sub-{subject:02}"
    root = args.root
    out = root / "generic"
    out.mkdir(exist_ok=True, parents=True)
    dest = out / f"{sub}.npz"
    receipt = dest.with_suffix(".json")
    schedule_raw = (root / "sessions.json").read_bytes()
    schedule = json.loads(schedule_raw)
    src_receipt = json.loads((root / f"{sub}.json").read_bytes())
    url = f"{REPO}/resolve/{REV}/Preprocessed_data_250Hz/{sub}/preprocessed_eeg_training.npy"
    if (
        src_receipt["source"] != url
        or src_receipt["source_size"] != SIZE
        or src_receipt["revision"] != REV
        or src_receipt["subject"] != sub
        or schedule["revision"] != ORDER_REV
    ):
        raise ValueError("Unexpected public EEG or schedule registration")
    session_ids = np.asarray(schedule["subjects"][sub]["session_pairs"], dtype=np.int64)[
        rows // 10
    ].T
    if (
        session_ids.shape != (2, len(rows))
        or np.any(session_ids[0] >= session_ids[1])
        or np.min(session_ids) < 1
        or np.max(session_ids) > 4
    ):
        raise ValueError("Invalid chronological session numbers")
    if receipt.exists() and dest.exists():
        old = json.loads(receipt.read_text())
        if (
            old["subject"] == sub
            and old["selection_seed"] == SEED
            and old["concepts"] == objects.tolist()
            and old["schedule_sha256"] == digest(schedule_raw)
            and old["revision"] == REV
            and old["packet_sha256"] == digest(dest.read_bytes())
        ):
            print(sub, "verified existing generic training packet", flush=True)
            return
    cache = root / "ranges" / REV / sub
    cache.mkdir(exist_ok=True, parents=True)
    tail = OFFSET + SHAPE[0] * ROW_BYTES
    channels, source_times = axes(
        ranged(url, 0, OFFSET - 1, cache), ranged(url, tail, SIZE - 1, cache)
    )
    post = [channels.tolist().index(c) for c in CHANNELS]
    slots = np.random.RandomState(20200220).permutation(4)
    early, late = np.flatnonzero(slots < 2), np.flatnonzero(slots >= 2)
    mean = np.empty((len(rows), len(post), 250), np.float32)
    halves = np.empty((2, *mean.shape), np.float32)
    records = {}

    def fetch(k):
        low = int(rows[10 * k])
        high = low + 10
        first, last = OFFSET + low * ROW_BYTES, OFFSET + high * ROW_BYTES - 1
        raw = ranged(url, first, last, cache)
        x = np.take(np.frombuffer(raw, dtype="<f8").reshape(10, *SHAPE[1:]), post, axis=2).astype(
            np.float32
        )
        if not np.isfinite(x).all():
            raise ValueError(f"{sub} contains nonfinite trial values at {low}")
        four = x.mean(axis=1)
        two = np.stack([x[:, early].mean(axis=1), x[:, late].mean(axis=1)])
        absolute = np.abs(two.mean(axis=0) - four)
        if np.any(absolute > 4 * np.finfo(np.float32).eps * np.abs(x).max(axis=1)):
            raise ValueError("Half means do not recombine to the four-trial mean")
        block = dict(
            rows=[low, high - 1],
            bytes=[first, last],
            sha256=digest(raw),
            recombination_max_abs=float(absolute.max()),
        )
        return k, four, two, block

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(fetch, k) for k in range(len(objects))]
        for job in as_completed(jobs):
            k, four, two, block = job.result()
            idx = slice(10 * k, 10 * k + 10)
            mean[idx], halves[:, idx], records[k] = four, two, block
            if len(records) % 16 == 0:
                print(sub, f"{len(records)}/128 physical training blocks verified", flush=True)
    tmp = dest.with_suffix(".partial.npz")
    np.savez(
        tmp,
        rows=rows,
        mean_eeg=mean,
        half_eeg=halves,
        channels=CHANNELS,
        times=source_times[50:],
        half_names=["early", "late"],
        session_ids=session_ids,
        stored_slots=np.stack([early, late]),
        subject=sub,
        repeats=4,
        repeats_per_half=2,
        unique_ids=labels,
        stimulus_stems=stems,
        concepts=objects,
        excluded_concepts=excluded,
        selection_seed=SEED,
    )
    tmp.replace(dest)
    store_json(
        receipt,
        dict(
            subject=sub,
            revision=REV,
            source=url,
            source_size=SIZE,
            source_full_lfs_sha256=src_receipt["source_full_lfs_sha256"],
            full_file_hashed_locally=False,
            data_bytes=len(rows) * ROW_BYTES,
            selection_seed=SEED,
            selection="Sorted simple random sample of 128 of 1,630 training concepts after excluding both 12-class panels; all ten photos per concept.",
            concepts=objects.tolist(),
            excluded_concepts=excluded.tolist(),
            train_rows_sha256=ROWS_SHA,
            schedule_sha256=digest(schedule_raw),
            session_writer=ORDER_SOURCE + "preprocessing_utils.py#L262-L299",
            stored_slots=[early.tolist(), late.tolist()],
            channels=CHANNELS,
            time_seconds=[float(source_times[50]), float(source_times[-1])],
            output_dtype="float32 (cast before taking the 4-trial and 2-trial means)",
            row_ranges=[records[k] for k in sorted(records)],
            maximum_absolute_recombination_error=max(
                b["recombination_max_abs"] for b in records.values()
            ),
            packet_sha256=digest(dest.read_bytes()),
        ),
    )
    print(sub, "GENERIC READY", dest, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/object-transfer"))
    p.add_argument(
        "--train-rows",
        type=Path,
        default=Path("outputs/official-object-readout/sub-03/atm_train_rows.csv"),
    )
    p.add_argument("--registered", type=Path)
    p.add_argument("--replication", type=Path)
    p.add_argument("--subjects", type=int, nargs="+", choices=(4, 5), default=[4, 5])
    p.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = p.parse_args()
    chosen = select(args)
    print("Generic class IDs (sorted):", chosen[1].tolist(), flush=True)
    for subject in args.subjects:
        prepare(subject, chosen, args)
