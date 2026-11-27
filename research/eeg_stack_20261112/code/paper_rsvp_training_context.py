"""Verify the correspondence between released training epochs and successive RSVP events."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SEED = 20200220
REVISION = "cf0540ffd250d8681ef31a9a9e345f5053e4c90b"
PHOTO_COUNT = 16540


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(4 << 20), b""):
            value.update(data)
    return value.hexdigest()


def occurrence_slots(events, *, merged_shuffle=True):
    grouped = defaultdict(list)
    for e in sorted(events):
        grouped[e[0]].append(e)
    if len(grouped) != 2 or any(len(by_session) != 2 for by_session in grouped.values()):
        raise ValueError("Each training photograph must occur twice in each of two sessions")
    first = np.random.RandomState(SEED).permutation(2)
    ordered = [items[i] for items in grouped.values() for i in first]
    return [ordered[i] for i in np.random.RandomState(SEED).permutation(4)] if merged_shuffle else ordered


def select_neighbors(events):
    pairs = {}
    for a, b in zip(events, events[1:]):
        block, sequence, position, sample, photo = a
        next_block, next_seq, next_position, next_sample, following_photo = b
        if ((block, sequence) == (next_block, next_seq) and position + 1 == next_position
                and 195 <= next_sample - sample <= 205 and min(photo, following_photo) >= 0):
            pairs[sample] = next_sample
    return pairs


def prepare(events, subject, split, output):
    recordings = json.loads((events / "receipt.json").read_text(encoding="utf-8"))
    if recordings["revision"] != REVISION:
        raise ValueError("Unexpected BIDS event registration")
    slot_map = events / f"sub-{subject:02d}_train.npy"
    sessions = np.load(slot_map, allow_pickle=False)
    allowed_rows = np.array(json.loads(split.read_text(encoding="utf-8"))["train_rows"])
    if len(allowed_rows) != 13230 or len(set(allowed_rows)) != 13230:
        raise ValueError("Expected registered training images only")
    allowed = np.zeros(PHOTO_COUNT, dtype=bool)
    allowed[allowed_rows] = True
    history = [[] for _ in range(PHOTO_COUNT)]
    neighbor, sources = {}, {}
    for session in range(1, 5):
        for run in range(1, 6):
            filename = f"sub-{subject:02d}_ses-{session:02d}_task-train_run-{run:02d}_events.tsv"
            event_file = events / filename
            evidence = recordings["files"][filename]
            if sha(event_file) != evidence["sha256"]:
                raise ValueError(f"BIDS event file differs from receipt: {filename}")
            sources[filename] = evidence
            with event_file.open(encoding="utf-8", newline="") as f:
                table = list(csv.DictReader(f, delimiter="\t"))
            elapsed = [int(record["sample"]) for record in table]
            if len(table) != evidence["rows"] or any(a >= b for a, b in zip(elapsed, elapsed[1:])):
                raise ValueError("Training events must be in increasing acquisition order")
            entries = []
            for record in table:
                code = int(record["tot_img_number"]) - 1 if record["trial_type"] == "image" else -1
                entry = tuple(int(record[key]) for key in ("block", "sequence", "img_in_sequence", "sample"))
                entries.append((*entry, code))
                if code >= 0:
                    history[code].append((session - 1, run, entry[-1]))
            for a, b in select_neighbors(entries).items():
                neighbor[session - 1, run, a] = (session - 1, run, b)
    slots = {}
    for image, items in enumerate(history):
        # Extra events at interrupted sequences do not identify which epochs were retained.
        if len(items) != 4 or len({item[0] for item in items}) != 2:
            continue
        # The Brain-HiVE export stores the session groups before the final ATM permutation.
        ordered = occurrence_slots(items, merged_shuffle=False)
        if [event[0] for event in ordered] != sessions[image].tolist():
            raise ValueError("BIDS events reproduce neither the image nor the repeat-session order")
        slots[image] = ordered
    order = {event: (image, repeat) for image, records in slots.items() for repeat, event in enumerate(records)}
    edges = []
    for reference in allowed_rows:
        for repeat, location in enumerate(slots.get(reference, ())):
            following = neighbor.get(location)
            if following not in order:
                continue
            image, exact = order[following]
            if not allowed[image]:
                continue
            difference = following[-1] - location[-1]
            same = np.flatnonzero(sessions[image] == location[0])
            other = np.flatnonzero(sessions[image] != location[0])
            alternate = same[same != exact].item()
            edges.append([reference, repeat, image, exact, alternate, *other, round(difference / 4)])
    output.mkdir(parents=True, exist_ok=True)
    pairs = output / f"sub-{subject:02d}-training-context.npy"
    np.save(pairs, np.asarray(edges, dtype=np.int32), allow_pickle=False)
    info = dict(subject=subject, next_pairs=len(edges), photo_count=PHOTO_COUNT, unambiguous_photos=len(slots),
                exact_session_slots=len(slots) * 4,
                repeat_order="Sessions in recorded order, source shuffle within each two-epoch group",
                revision=REVISION, sources=sources, session_sha256=sha(slot_map), split_sha256=sha(split),
                context_array_sha256=sha(pairs), prepare_code_sha256=sha(Path(__file__)))
    pairs.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print("RSVP training order", subject, len(edges), flush=True)


def correlation(a, b):
    a = (a - a.mean(axis=-1, keepdims=True)).reshape(len(a), -1)
    b = (b - b.mean(axis=-1, keepdims=True)).reshape(len(b), -1)
    magnitude = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return (a * b).sum(axis=-1) / np.maximum(magnitude, 1e-9)


def audit(packet, eeg_root):
    registration = json.loads(packet.with_suffix(".json").read_text(encoding="utf-8"))
    subject = registration["subject"]
    if sha(packet) != registration["context_array_sha256"]:
        raise ValueError("Neighbor receipt differs from actual mapping")
    source = eeg_root / f"sub-{subject:02d}_train.npy"
    eeg = np.load(source, mmap_mode="r", allow_pickle=False)
    if eeg.shape != (PHOTO_COUNT, 4, 17, 250):
        raise ValueError("Unexpected training EEG shape")
    raw_hash = sha(source)
    metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    if raw_hash != metadata["output_sha256"] or registration["session_sha256"] != metadata["session_sha256"]:
        raise ValueError("Training EEG and repeat-order provenance disagree")
    edges = np.load(packet, allow_pickle=False)
    session_path = eeg_root / metadata["session_file"]
    if sha(session_path) != registration["session_sha256"]:
        raise ValueError("Session index changed after event mapping")
    sessions = np.load(session_path, allow_pickle=False)
    head, used, context, _, _, first_other, _, _ = edges.T
    group = sessions[head, used] * 4 + sessions[context, first_other]
    donor = np.empty(len(edges), dtype=int)
    for condition in np.unique(group):
        indices = np.flatnonzero(group == condition)
        indices = indices[np.argsort(context[indices], kind="stable")]
        donors = np.roll(indices, len(indices) // 2)
        if np.any(context[indices] == context[donors]):
            raise ValueError("Matched-session donor reused the true neighboring photograph")
        donor[indices] = donors
    rows = []
    for start in range(0, len(edges), 768):
        head, used, neighbor, exact, alternate, other1, other2, lag = edges[start:start + 768].T
        a = eeg[head, used].astype(np.float32)
        window = np.arange(12, 44)[None, None, :]
        shifted = np.take_along_axis(a, np.broadcast_to(window + lag[:, None, None], (len(a), 17, 32)), axis=2)
        overlap = eeg[neighbor, exact, :, 12:44].astype(np.float32)
        duplicate = eeg[neighbor, alternate, :, 12:44].astype(np.float32)
        cross_session = (eeg[neighbor, other1, :, 12:44].astype(np.float32)
                         + eeg[neighbor, other2, :, 12:44].astype(np.float32)) * .5
        wrong = edges[donor[start:start + len(a)]]
        unrelated = (eeg[wrong[:, 2], wrong[:, 5], :, 12:44].astype(np.float32)
                     + eeg[wrong[:, 2], wrong[:, 6], :, 12:44].astype(np.float32)) * .5
        rows.append(np.column_stack([correlation(shifted, overlap), correlation(shifted, duplicate),
                                     correlation(shifted, cross_session), correlation(shifted, unrelated),
                                     correlation(a[:, :, 12:44], cross_session)]))
    points = np.concatenate(rows)
    summary = dict(subject=subject, n=len(points), input=registration, verified_eeg_sha256=raw_hash,
                   comparisons=("matched_same_acquisition", "other_repeat_same_session",
                                "next_photo_other_session", "different_photo_same_other_session",
                                "next_photo_other_session_before_its_onset"),
                   means=points.mean(0).tolist(), medians=np.median(points, axis=0).tolist(),
                   exact_order_better=float(np.mean(points[:, 0] > points[:, 1])),
                   true_vs_donor=float(np.mean(points[:, 2] > points[:, 3])))
    output = packet.with_name(packet.stem + "-audit.json")
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("RSVP voltage verification", subject, len(points), summary["means"], summary["exact_order_better"], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="action", required=True)
    make = sub.add_parser("prepare")
    make.add_argument("subject", type=int)
    make.add_argument("events", type=Path)
    make.add_argument("split", type=Path)
    make.add_argument("output", type=Path)
    check = sub.add_parser("audit")
    check.add_argument("packet", type=Path)
    check.add_argument("eeg_root", type=Path)
    args = cli.parse_args()
    if args.action == "prepare":
        prepare(args.events, args.subject, args.split, args.output)
    else:
        audit(args.packet, args.eeg_root)
