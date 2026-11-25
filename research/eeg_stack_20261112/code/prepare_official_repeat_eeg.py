"""Recover five fixed nested repeat budgets from the pinned sub-04 official test voltage file."""

import argparse
import hashlib
import pickletools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from decode_eeg_layout import CHANNELS
from prepare_expanded_personal import extract
from prepare_registered_subjects import REPO, REV, TAIL_SHA, digest, http

SIZE = 2016003002
OFFSET = 179
BODY = 2016000000
FULL_SHA = "cba83bc49213f8783d29e1717e63c0b429967601ea417adf6671dbfd2336c03e"
HEADER = "a4acb6d6c41eb2125ca958b8547a1f9b970f83d06efb994f8f380c3a3b0caef2"
URL = f"{REPO}/resolve/{REV}/Preprocessed_data_250Hz/sub-04/preprocessed_eeg_test.npy"
BUDGETS = (4, 8, 16, 32, 80)
SEED = 260541


def portion(dst, lo, hi):
    if not dst.exists() or dst.stat().st_size != hi - lo + 1:
        data = http(
            URL + f"?repeat-objects={lo}-{hi}",
            {"Range": f"bytes={lo}-{hi}"},
            hi - lo + 1,
            f"bytes {lo}-{hi}/{SIZE}",
        )
        tmp = dst.with_suffix(".partial")
        tmp.write_bytes(data)
        tmp.replace(dst)
    return dst.read_bytes()


def main(a):
    base = a.out / "sub-04"
    cache = base / "ranges"
    cache.mkdir(parents=True, exist_ok=True)
    head = portion(cache / "header.bin", 0, OFFSET - 1)
    tail = portion(cache / "tail.bin", OFFSET + BODY, SIZE - 1)
    if digest(head) != HEADER or digest(tail) != TAIL_SHA:
        raise ValueError("Official test-array header, dimensions or time/channel axes changed")
    # The file is a protocol-4 pickle. Inspect metadata opcodes without executing the pickle.
    seq = list(pickletools.genops(head[:-4] + bytes(4) + tail))
    words = [arg for op, arg, _ in seq if op.name == "SHORT_BINUNICODE"]
    channels = words[words.index("ch_names") + 1 : words.index("times")]
    ix = [channels.index(k) for k in CHANNELS]
    permutation = np.random.default_rng(SEED).permutation(80)
    n_per_block = 2
    byte_per_photo = BODY // 200

    def block(start):
        raw = portion(
            cache / f"{start:03}.bin",
            OFFSET + start * byte_per_photo,
            OFFSET + (start + n_per_block) * byte_per_photo - 1,
        )
        values = np.take(
            np.frombuffer(raw, "<f8").reshape(n_per_block, 80, 63, 250), ix, axis=2
        ).astype(np.float32)
        y = np.stack(
            [extract(values[:, permutation[:b]].mean(1).astype(np.float32)) for b in BUDGETS]
        )
        chunks = (
            values[:, permutation].reshape(n_per_block, 20, 4, 17, 250).mean(2).astype(np.float32)
        )
        f = extract(chunks.reshape(-1, 17, 250)).reshape(n_per_block, 20, 1003)
        return start, y, np.stack([f[:, : b // 4].mean(1) for b in BUDGETS])

    features = np.empty((5, 200, 1003), np.float32)
    mean_four = np.empty_like(features)
    with ThreadPoolExecutor(a.workers) as ex:
        for start, y, f in ex.map(block, range(0, 200, n_per_block)):
            features[:, start : start + n_per_block] = y
            mean_four[:, start : start + n_per_block] = f
            if start % 20 == 0:
                print("sub-04 photos complete through", start + n_per_block, flush=True)
    h = hashlib.sha256(head)
    for start in range(0, 200, n_per_block):
        h.update((cache / f"{start:03}.bin").read_bytes())
    h.update(tail)
    if h.hexdigest() != FULL_SHA or not np.isfinite(features).all():
        raise ValueError("Full original checksum or extracted values disagree")
    np.savez(
        base / "test_budgets.npz",
        rows=np.arange(200),
        budgets=BUDGETS,
        permutation=permutation,
        features=features,
        mean_four_features=mean_four,
        source_sha256=FULL_SHA,
        source_url=URL,
        channels=CHANNELS,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/official-repeat-object"))
    p.add_argument("--workers", type=int, default=6)
    main(p.parse_args())
