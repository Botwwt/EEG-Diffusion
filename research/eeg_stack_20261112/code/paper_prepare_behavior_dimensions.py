"""Align the published THINGS visual dimension predictions to registered training photos."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np


REVISION = "b1843e7eb92858c9c4a9ef88b4a5faea24752b49"
BASE_URL = f"https://raw.githubusercontent.com/ViCCo-Group/dimension_encoding/{REVISION}/data/66d"
NAMES = "file_names_THINGS.txt"
DIMENSIONS = "labels_short.txt"
PREDICTIONS = "predictions_66d_elastic_clip-ViT_visual_THINGS.txt"
BLOBS = {
    NAMES: "5d95ca43b07813407e80b9cbcb1659f995218559",
    DIMENSIONS: "1cd4bdb8d7861f81c4d2f4d66622d321db438e08",
    PREDICTIONS: "8a6421952a65390661dd95f465c52123549d1b5f",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aligned_attributes(names, values, registered):
    if values.ndim != 2 or len(values) != len(names) or len(set(names)) != len(names):
        raise ValueError("The authors' dimension matrix must have one row for each unique filename")
    if len(set(registered)) != len(registered):
        raise ValueError("Registered photographs must have unique filenames")
    lookup = {name: index for index, name in enumerate(names)}
    try:
        rows = np.asarray([lookup[name] for name in registered], dtype=np.int64)
    except KeyError as missing:
        raise ValueError(f"Registered photograph {missing.args[0]} not found in the dimension archive") from missing
    return values[rows], rows


def prepare(source, prep, output):
    sources = {}
    for name, expected in BLOBS.items():
        raw = (source / name).read_bytes()
        git_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if git_blob != expected:
            raise ValueError(f"{name} differs from the pinned data release")
        sources[name] = dict(git_blob=expected, sha256=hashlib.sha256(raw).hexdigest(),
                             bytes=len(raw), source=f"{BASE_URL}/{name}")
    names = (source / NAMES).read_text(encoding="utf-8").splitlines()
    labels = (source / DIMENSIONS).read_text(encoding="utf-8").splitlines()
    attributes = np.loadtxt(source / PREDICTIONS, dtype=np.float32)
    if len(names) != 26107 or len(labels) != 66 or attributes.shape != (len(names), 66) or not np.isfinite(attributes).all():
        raise ValueError("Expected the complete published THINGS feature matrix")
    split_path, photo_path, dev_path = (prep / name for name in ("paper_split.json", "train_ids.json", "paper_development.json"))
    split = json.loads(split_path.read_bytes())
    development = json.loads(dev_path.read_bytes())
    index = json.loads(photo_path.read_bytes())
    rows = np.asarray(split["train_rows"], dtype=np.int64)
    if rows.shape != (13230,) or np.any(np.diff(rows) <= 0) or np.intersect1d(rows, split["fresh_rows"]).size \
            or np.intersect1d(rows, development["dev_rows"]).size:
        raise ValueError("Expected only sorted, distinct registered training rows")
    registered_names = [PurePosixPath(index["paths"][i].replace("\\", "/")).name for i in rows]
    if any(PurePosixPath(name).stem != index["ids"][i] for name, i in zip(registered_names, rows, strict=True)):
        raise ValueError("Registered paths and photograph identities disagree")
    matched, source_rows = aligned_attributes(names, attributes, registered_names)
    if (matched.std(axis=0) <= 1e-5).any():
        raise ValueError("Every attribute needs variation on registered photographs")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "registered-training-dimensions.npz"
    temporary = path.with_suffix(".part.npz")
    np.savez_compressed(temporary, rows=rows, names=np.asarray(registered_names), source_rows=source_rows, attributes=matched)
    temporary.replace(path)
    record = dict(revision=REVISION, source_files=sources, columns=labels, training_photo_index=sha(photo_path),
                  restricted_split_sha256=sha(split_path), development_sha256=sha(dev_path),
                  photo_count=len(rows), attributes=len(labels), source_photo_count=len(names), pack_sha256=sha(path),
                  source_definition="Published CLIP-ViT regression predictions of 66 behavior-derived THINGS object dimensions. Registered photographs only.")
    path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(path.name, len(rows), len(labels), record["pack_sha256"], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("prep", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    prepare(args.source, args.prep, args.output)
