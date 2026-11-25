"""Convert the SHA-pinned author archives into physical-repeat, native-f16 posterior EEG."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from stage_public_inputs import MANIFEST, ROOT, digest

PREP = ROOT / "prep17"
os.environ.setdefault("RAD_ROOT", str(ROOT))
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from common import CH17, safe_load_pt


def write_json(path, obj):
    content = json.dumps(obj, indent=1, ensure_ascii=False).encode("utf-8")
    if path.exists() and path.read_bytes() == content:
        return
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(content)
    tmp.replace(path)


def official_splits():
    meta_path = ROOT / "image_set" / "image_metadata.npy"
    if digest(meta_path) != "681a75633c59a356a66ca19a277429c1d48a00a98405ff96e53ee0b9f00b397c":
        raise ValueError("Official OSF image metadata checksum differs")
    meta = np.load(meta_path, allow_pickle=True).item()
    splits = {}
    PREP.mkdir(exist_ok=True)
    # These are the original EEG path prefixes. The OSF training ZIP uses
    # training_images/ for the same 16,540 class-folder/filename pairs.
    for split, folder, expected in (("train", "train_images", 16540), ("test", "test_images", 200)):
        cats = meta[f"{split}_img_concepts"]
        files = meta[f"{split}_img_files"]
        paths = [f"{folder}/{c}/{f}" for c, f in zip(cats, files, strict=True)]
        ids = [Path(p).stem for p in paths]
        if len(ids) != expected or len(set(ids)) != expected:
            raise ValueError(f"Nonunique official {split} photo ID")
        ids_file = PREP / f"{split}_ids.json"
        write_json(ids_file, {"ids": ids, "paths": paths})
        splits[split] = dict(paths=paths, ids=ids,
                             classes=np.array([int(c.split("_", 1)[0]) - 1 for c in cats]),
                             names=np.array([c.split("_", 1)[1] for c in cats]), ids_sha256=digest(ids_file))
    return splits


def channel_order():
    path = ROOT / "Brain-HIVE" / "configs" / "THINGS_EEG_CHANNELS.jsonl"
    entries = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]
    entries.sort(key=lambda v: v["index"])
    if [r["index"] for r in entries] != list(range(63)) or len(set(CH17)) != 17:
        raise ValueError("Unexpected source montage")
    return [r["name"] for r in entries]


def prepare(asset, split, official, montage):
    source = ROOT / asset["path"]
    subject = source.parent.name
    name = f"{subject}_{split}"
    dest = PREP / f"{name}.npy"
    receipt = PREP / f"{name}.json"
    sessions = PREP / "sessions" / f"{name}.npy"
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if (digest(dest) != previous["output_sha256"] or previous["source_sha256"] != asset["sha256"]
                or digest(sessions) != previous["session_sha256"] or official["ids_sha256"] != previous["ids_sha256"]):
            raise ValueError(f"Existing receipt no longer matches {name}")
        return
    if source.stat().st_size != asset["bytes"] or digest(source) != asset["sha256"]:
        raise ValueError(f"Source file differs from public LFS SHA-256: {name}")
    d = safe_load_pt(source)
    x = np.asarray(d["eeg"])
    n, r = (16540, 4) if split == "train" else (200, 80)
    if x.shape != (n, r, 63, 250) or x.dtype != np.dtype("<f2") or list(d["ch_names"]) != montage:
        raise ValueError(f"Unexpected raw EEG axes, type or montage: {name}")
    times = np.asarray(d["times"])
    if times.shape != (300,) or not np.allclose(times, np.arange(-50, 250) / 250, rtol=0, atol=1e-9):
        raise ValueError(f"Unexpected source time metadata: {name}")
    paths = np.asarray(d["img"])
    if paths.shape != (n, r) or not np.all(paths == paths[:, :1]):
        raise ValueError(f"Mixed stimulus identities across repeats: {name}")
    source_paths = paths[:, 0].tolist()
    key = {Path(p).stem: index for index, p in enumerate(source_paths)}
    if set(key) != set(official["ids"]) or len(key) != n:
        raise ValueError(f"EEG photo IDs differ from official OSF image set: {name}")
    ordered = np.array([key[image_id] for image_id in official["ids"]], dtype=np.intp)
    if [source_paths[index] for index in ordered] != official["paths"]:
        raise ValueError(f"EEG split class-folder/filename differs from official metadata: {name}")
    if not np.all(np.asarray(d["label"])[ordered] == official["classes"][:, None]):
        raise ValueError(f"EEG split-class IDs do not match stimulus metadata: {name}")
    if not np.all(np.asarray(d["text"])[ordered] == official["names"][:, None]):
        raise ValueError(f"Source concept keys do not match photo folders: {name}")
    native_session = np.asarray(d["session"])[ordered]
    if native_session.shape != (n, r) or not np.isin(native_session, [0, 1, 2, 3]).all():
        raise ValueError(f"Unexpected session identifiers: {name}")
    sessions.parent.mkdir(exist_ok=True)
    session_part = sessions.with_suffix(".partial")
    with session_part.open("wb") as f:
        np.save(f, native_session.astype(np.uint8), allow_pickle=False)
    session_part.replace(sessions)
    selected = [montage.index(c) for c in CH17]
    temp = dest.with_suffix(".partial")
    output = np.lib.format.open_memmap(temp, mode="w+", dtype="<f2", shape=(n, r, 17, 250))
    row_order_identity = np.array_equal(ordered, np.arange(n))
    body_hash = hashlib.sha256()
    for first in range(0, n, 48):
        rows = slice(first, min(first + 48, n))
        batch = x[rows] if row_order_identity else x[ordered[rows]]
        block = np.take(batch, selected, axis=2)
        if not np.isfinite(block).all():
            raise ValueError(f"Nonfinite posterior EEG in {name}, row {first}")
        output[rows] = block
        body_hash.update(block)
    output.flush()
    del output
    stored = np.load(temp, mmap_mode="r", allow_pickle=False)
    for channel, source_channel in enumerate(selected):
        probe = [0, n // 2, n - 1]
        if not np.array_equal(stored[probe, :, channel], x[ordered[probe], :, source_channel]):
            raise ValueError(f"Incorrect channel write for {name} {CH17[channel]}")
    del stored
    out_hash = digest(temp)
    temp.replace(dest)
    session_counts = np.stack([(native_session == i).sum(axis=1) for i in range(4)], axis=1)
    result = dict(subject=subject, split=split, path=dest.name, sha256=out_hash, output_sha256=out_hash,
                  bytes=dest.stat().st_size, output_body_sha256=body_hash.hexdigest(),
                  source_path=asset["path"], source_url=asset["url"], source_sha256=asset["sha256"],
                  source_bytes=asset["bytes"], source_shape=list(x.shape), shape=[n, r, 17, 250],
                  dtype="float16", axes=["stimulus", "physical_repeat", "channel", "sample"],
                  channel_names=CH17, source_channel_indices=selected, sample_rate_hz=250,
                  sample_times_seconds="0.000 to 0.996 in steps of 0.004; retained all 250 stored samples",
                  included_source_time_metadata="The release includes 300 time labels from -0.200 to 0.996 for 250 stored samples.",
                  processing="Selected and reordered channels only; no repeat averaging, amplitude transformation, or normalization.",
                  ids_file=f"{split}_ids.json", ids_sha256=official["ids_sha256"],
                  source_row_order_matches_official=row_order_identity, checked_physical_photo_ids=n * r,
                  session_file=f"sessions/{name}.npy", session_sha256=digest(sessions),
                  session_source_key="session", session_ids="0-based source IDs",
                  min_session_counts_per_stimulus=session_counts.min(axis=0).tolist(),
                  max_session_counts_per_stimulus=session_counts.max(axis=0).tolist())
    write_json(receipt, result)
    print("READY", name, result["shape"], out_hash, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true", help="consume source files as the concurrent downloader verifies them")
    parser.add_argument("--subjects", nargs="+", type=int, choices=range(1, 11))
    args = parser.parse_args()
    splits = official_splits()
    montage = channel_order()
    archive = json.loads(MANIFEST.read_bytes())["assets"]
    pending = [a for a in archive if a["group"] == "eeg" and
               (not args.subjects or int(Path(a["path"]).parent.name[4:]) in args.subjects)]
    while pending:
        for asset in pending.copy():
            path = ROOT / asset["path"]
            if path.exists():
                prepare(asset, path.stem, splits[path.stem], montage)
                pending.remove(asset)
        if not pending or not args.wait:
            break
        time.sleep(4)
    if pending:
        raise SystemExit("Missing requested source files: " + ", ".join(a["path"] for a in pending))


if __name__ == "__main__":
    main()
