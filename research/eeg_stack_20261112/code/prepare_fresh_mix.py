"""Fetch the 48 registered sub-04 objects, keep the verified full signal, and derive inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from decode_eeg_layout import CHANNELS
from prepare_expanded_personal import design_block
from prepare_registered_subjects import OFFSET, REV, ROW_BYTES, SHAPE, SIZE, axes, digest, http
from prepare_trial_object_inputs import SOURCE_SHA, atomic, check
from register_fresh_mix import LOCK, SUB
from register_fresh_photo_panel import freeze, read, sha
from run_prospective_readout import frozen_npz


def acquire(a):
    design = read(LOCK)
    check(sha(a.out / "design.json") == sha(LOCK), "Registration changed")
    legacy = read(a.data / "photo-structure/expanded/features/sub-04.json")
    check(legacy["source_full_lfs_sha256"] == SOURCE_SHA, "Different original EEG release")
    ids = sorted(c for v in design["panels"].values() for c in v["object_ids"])
    check(len(set(ids)) == len(ids) == 48, "Candidate class reused")
    cache = a.data / "object-transfer/ranges" / REV / SUB
    tail = OFFSET + SHAPE[0] * ROW_BYTES
    channels, times = axes(
        (cache / f"0-{OFFSET - 1}.bin").read_bytes(),
        (cache / f"{tail}-{SIZE - 1}.bin").read_bytes(),
    )
    post = [channels.tolist().index(c) for c in CHANNELS]
    rawdir = a.out / "eeg/raw"
    rawdir.mkdir(parents=True, exist_ok=True)
    source = legacy["source"]

    def one(c):
        lo, hi = OFFSET + 10 * c * ROW_BYTES, OFFSET + 10 * (c + 1) * ROW_BYTES - 1
        interval = f"{lo}-{hi}"
        path = rawdir / f"{c:04d}.bin"
        receipt = path.with_suffix(".json")
        if receipt.exists():
            raw = path.read_bytes()
            check(
                read(receipt)
                == dict(concept=c, source=source, interval=interval, sha256=digest(raw)),
                "Changed cached original range",
            )
        else:
            check(not a.offline, "A verified target range is missing")
            content = f"bytes {interval}/{SIZE}"
            first = http(
                source + "?range=" + interval,
                {"Range": "bytes=" + interval},
                10 * ROW_BYTES,
                content,
            )
            raw = http(
                source + "?range=" + interval + "&verify=fresh-mix",
                {"Range": "bytes=" + interval},
                10 * ROW_BYTES,
                content,
            )
            check(first == raw, "Independent HTTP range reads differ")
            atomic(path, raw)
            freeze(receipt, dict(concept=c, source=source, interval=interval, sha256=digest(raw)))
        x = np.frombuffer(raw, "<f8").reshape(10, *SHAPE[1:])
        check(np.isfinite(x).all(), "Nonfinite EEG")
        return (
            c,
            np.ascontiguousarray(x[:, :, :, :125], dtype=np.float32),
            design_block(raw, post),
            digest(raw),
        )

    blocks = {}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(one, c) for c in ids]
        for i, f in enumerate(as_completed(futures), 1):
            c, trials, features, raw_sha = f.result()
            blocks[c] = (trials, features, raw_sha)
            if i % 12 == 0:
                print(i, "/48 full blocks independently verified", flush=True)
    dest = a.out / "eeg/sub-04.npz"
    frozen_npz(
        dest,
        dict(
            subject=SUB,
            rows=(np.array(ids)[:, None] * 10 + np.arange(10)).ravel(),
            trials=np.concatenate([blocks[c][0] for c in ids]),
            features=np.concatenate([blocks[c][1] for c in ids]),
            channels=channels,
            posterior_channels=np.array(CHANNELS),
            source_times=times[50:],
            trial_times=times[50:175],
            design_sha256=sha(LOCK),
        ),
    )
    freeze(
        dest.with_suffix(".json"),
        dict(
            source=source,
            source_full_lfs_sha256=SOURCE_SHA,
            revision=REV,
            design_sha256=sha(LOCK),
            packet_sha256=sha(dest),
            raw_sha256={str(c): blocks[c][2] for c in ids},
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/fresh-mix"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--workers", type=int, default=6, choices=range(1, 13))
    p.add_argument("--offline", action="store_true")
    acquire(p.parse_args())
