"""Join released THINGS-EEG1 markers to physical EEG for registered training photos."""

import argparse
from bisect import bisect_left, bisect_right
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

from paper_render_cache import PinnedFiles, atomic_file, require, save_json

PAST, FUTURE = .30, .24
SOURCE = "https://data.nemar.org/on003825/v1.0.2/"


def eligible_epochs(timestamps, photographs, allowed, folds):
    require(len(timestamps) == len(photographs), "Missing stimulus timing")
    result = []
    used = set()
    for n, (onset, photo) in enumerate(zip(timestamps, photographs, strict=True)):
        if photo not in allowed or photo in used:
            continue
        start = bisect_left(timestamps, onset - PAST)
        stop = bisect_right(timestamps, onset + FUTURE)
        context = {label for label, _ in photographs[start:stop]}
        if not context.issubset(folds):
            continue
        used_folds = sum(1 << k for k in {folds[label] for label in context})
        result.append((n, allowed[photo], used_folds))
        used.add(photo)
    return result


def verify_file(path, entry, pins):
    fingerprint = pins.pin(path)
    require(path.stat().st_size == entry["size"], "The public derivative size differs")
    algorithm = entry["checksum_algorithm"]
    with path.open("rb") as source:
        if algorithm == "git":
            m = hashlib.sha1(f"blob {entry['size']}\x00".encode())
            for chunk in iter(lambda: source.read(1 << 20), b""):
                m.update(chunk)
            digest = m.hexdigest()
        else:
            require(algorithm in ("md5", "sha256"), "Unexpected public file checksum")
            digest = fingerprint if algorithm == "sha256" else hashlib.file_digest(source, algorithm).hexdigest()
    require(digest == entry["checksum"], f"Public release bytes differ at {path.name}")
    return fingerprint


def path_key(path, prefixed):
    p = PurePosixPath(path.replace("\\", "/"))
    return p.parent.name.split("_", 1)[1] if prefixed else p.parent.name, p.name.lower()


def registered_stimulus(row, manifest):
    path = PurePosixPath(row["stim"])
    if row["stim"] not in manifest or path.parts[0] != "stimuli":
        return False
    if "stimname" in row:
        return row["stimname"].lower() == path.name.lower()
    return row.get("object") == path.parent.name


def run(args):
    from common import CH17, concept_folds
    from scipy.io import loadmat

    files = PinnedFiles()
    subject = f"sub-{args.subject:02d}"
    manifest_file = args.public / "manifest.json"
    manifest = {x["path"]: x for x in json.loads(manifest_file.read_bytes())}
    file_stem = f"{subject}_task-rsvp_continuous"
    set_path, samples_path = (args.public / (file_stem + extension) for extension in (".set", ".fdt"))
    events = args.events
    split = args.registration / "paper_split.json"
    index = args.registration / "train_ids.json"
    label_path = args.registration / "train_concept.npy"
    proof = dict(source=SOURCE, manifest=files.pin(manifest_file), split=files.pin(split),
                 photo_index=files.pin(index), concept_ids=files.pin(label_path), code=files.pin(Path(__file__)))
    proof["files"] = {
        path.name: verify_file(path, manifest[name], files)
        for path, name in ((set_path, f"derivatives/eeglab/{set_path.name}"),
                           (samples_path, f"derivatives/eeglab/{samples_path.name}"),
                           (events, f"{subject}/eeg/{subject}_task-rsvp_events.tsv"))
    }
    rows = json.loads(split.read_bytes())["train_rows"]
    paths = json.loads(index.read_bytes())["paths"]
    labels = np.load(label_path, allow_pickle=False)
    expected_folds = concept_folds(1654)[labels]
    known = {path_key(paths[i], True): i for i in rows}
    fold_by_name = {photo[0]: int(expected_folds[row]) for photo, row in known.items()}
    require(len(known) == len(rows) == 13230 and len(fold_by_name) == 1323
            and set(fold_by_name.values()) == {1, 2, 3, 4}, "Only the registered 1,323 objects may train the model")
    with events.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    time = [float(row["onset"]) for row in records]
    require(all(x < y for x, y in zip(time, time[1:])), "BIDS events must be in physical presentation order")
    require(all(registered_stimulus(r, manifest) for r in records),
            "Stimulus paths are missing from the public release")
    stimulus = [path_key(r["stim"], False) for r in records]
    allowed = eligible_epochs(time, stimulus, known, fold_by_name)
    require(allowed, "No registered photos have usable neighboring stimuli")
    chosen = np.array(allowed, dtype=np.int64)
    eeg = loadmat(set_path, variable_names=("EEG",), simplify_cells=True)["EEG"]
    require(eeg["srate"] == 250 and eeg["trials"] == 1 and eeg["data"] == samples_path.name,
            "Unexpected time series in the continuous EEGLAB file")
    latency = np.array([event["latency"] - 1 for event in eeg["event"]
                        if str(event["type"]).split() == ["E", "1"]], dtype=np.float64)
    require(latency.shape == (len(time),) and np.all(np.diff(latency) > 0)
            and np.max(np.abs(latency / 250 - np.asarray(time))) <= .004,
            "The continuous presentation order differs from the BIDS stimulus events")
    channel_names = [c["labels"] for c in eeg["chanlocs"]]
    require(len(set(channel_names)) == eeg["nbchan"] and set(CH17).issubset(channel_names),
            "Posterior channels differ from the benchmark montage")
    indexes = [channel_names.index(name) for name in CH17]
    positions = np.rint(latency[chosen[:, 0]]).astype(np.int64)
    length = int(FUTURE * 250)
    require(samples_path.stat().st_size == 4 * eeg["nbchan"] * eeg["pnts"]
            and positions.min() >= 0 and positions.max() + length <= eeg["pnts"],
            "Registered epoch outside the physical EEG samples")
    raw = np.memmap(samples_path, dtype="<f4", mode="r", shape=(eeg["pnts"], eeg["nbchan"]))
    values = np.empty((len(chosen), len(CH17), length), dtype=np.float32)
    for start in range(0, len(chosen), 128):
        part = positions[start:start+128, None] + np.arange(length)[None, :]
        values[start:start+128] = raw[part[:, :, None], indexes].transpose(0, 2, 1)
    require(np.isfinite(values).all(), "Nonfinite samples in the registered epoch")
    files.verify()
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / f"{subject}-registered-context.npz"
    atomic_file(dest, lambda f: np.savez_compressed(f, eeg=values, source_event=chosen[:, 0],
                train_row=chosen[:, 1], context_fold_mask=chosen[:, 2], latency=latency[chosen[:, 0]]))
    metadata = dict(subject=subject, rate=250, shape=list(values.shape), channels=CH17,
                    event_count=len(time), matched_training_photos=len(set(stimulus).intersection(known)),
                    matched_training_events=sum(photo in known for photo in stimulus),
                    retained_photos=len(set(chosen[:, 1])), retained_objects=len(set(labels[chosen[:, 1]])),
                    context_seconds=[-PAST, FUTURE], context_excludes_all_unregistered_objects=True,
                    training_rows_outside_context_fold={str(k): int(((chosen[:, 2] & (1 << k)) == 0).sum())
                                                        for k in range(1, 5)},
                    max_marker_discrepancy_ms=float(np.max(np.abs(latency / 250 - time)) * 1e3),
                    files=proof, sha256=files.pin(dest))
    save_json(dest.with_suffix(".json"), metadata)
    print(json.dumps({k:v for k,v in metadata.items() if k not in ("files","channels")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, choices=range(1, 51), required=True)
    for item in ("public", "events", "registration", "output"):
        parser.add_argument(f"--{item}", type=Path, required=True)
    run(parser.parse_args())
