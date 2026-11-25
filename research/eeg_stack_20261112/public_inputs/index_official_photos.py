"""Build the evaluator's stable train/test photo and 1,854-concept lookup tables."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

from PIL import Image
import numpy as np

from stage_public_inputs import MANIFEST, ROOT, digest


def save_csv(path, rows, lineterminator="\n"):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, list(rows[0]), lineterminator=lineterminator)
        writer.writeheader()
        writer.writerows(rows)


def main():
    base = ROOT / "image_set"
    meta_file = base / "image_metadata.npy"
    concepts_file = base / "things_concepts.tsv"
    sources = json.loads(MANIFEST.read_bytes())["assets"]
    verified = {}
    for asset in sources:
        if asset["group"] == "images":
            path = ROOT / asset["path"]
            actual = digest(path)
            if actual != asset["sha256"]:
                raise ValueError(f"Source checksum mismatch for {path}")
            verified[path.name] = actual
    meta = np.load(meta_file, allow_pickle=True).item()
    with concepts_file.open(encoding="utf-8-sig", newline="") as src:
        vocabulary = {r["uniqueID"]: (i, r) for i, r in enumerate(csv.DictReader(src, delimiter="\t"))}
    if len(vocabulary) != 1854:
        raise ValueError("Expected 1,854 unique THINGS concept keys")
    concepts = {}
    photo_hashes = {}
    modes = {}
    for split, folder, count, shots in (("train", "training_images", 1654, 10), ("test", "test_images", 200, 1)):
        records = []
        image_counter = Counter()
        mode_counter = Counter()
        with zipfile.ZipFile(base / f"{folder}.zip") as archive:
            members = {name for name in archive.namelist() if not name.endswith("/")}
            arrays = (meta[f"{split}_img_{part}"] for part in ("concepts", "concepts_THINGS", "files"))
            for i, (folder_name, global_name, fname) in enumerate(zip(*arrays, strict=True)):
                split_num, cid = folder_name.split("_", 1)
                global_num, cname = global_name.split("_", 1)
                if cname != cid:
                    raise ValueError(f"Concept mismatch at {split} row {i}")
                tsv_row, cdata = vocabulary[cid]
                ident = dict(things_id=int(global_num), unique_id=cid, display_name=cdata["Word"],
                             things_tsv_row0=tsv_row, wordnet_id=cdata["WordNet ID"], split=split,
                             split_class_id=int(split_num), split_class_index0=int(split_num) - 1)
                if cid in concepts and concepts[cid] != ident:
                    raise ValueError(f"Conflicting class IDs for {cid}")
                concepts[cid] = ident
                image_counter[cid] += 1
                member = f"{folder}/{folder_name}/{fname}"
                if member not in members:
                    raise ValueError(f"Missing {member}")
                members.remove(member)
                payload = archive.read(member)
                with Image.open(io.BytesIO(payload)) as img:
                    mode_counter[img.mode] += 1
                    width, height = img.size
                    img.load()
                records.append(dict(row0=i, image_id=Path(fname).stem, filename=fname, archive_member=member,
                                    things_id=int(global_num), unique_id=cid, display_name=cdata["Word"],
                                    split_class_id=int(split_num), split_class_index0=int(split_num) - 1,
                                    image_sha256=hashlib.sha256(payload).hexdigest(), width=width, height=height))
        if (members or len(records) != count * shots or set(image_counter.values()) != {shots}
                or len(image_counter) != count or len({r["image_id"] for r in records}) != len(records)):
            raise ValueError(f"Inconsistent stimulus inventory for {split}")
        path = base / f"{split}_photos.csv"
        save_csv(path, records)
        photo_hashes[path.name] = digest(path)
        if split == "test":
            official = [dict(eeg_row=r["row0"], unique_id=r["unique_id"], archive_member=r["archive_member"],
                             image_sha256=r["image_sha256"]) for r in records]
            path = base / "originals_manifest.csv"
            save_csv(path, official, lineterminator="\r\n")
            photo_hashes[path.name] = digest(path)
        modes[split] = dict(mode_counter)
    if set(concepts) != set(vocabulary) or {a["things_id"] for a in concepts.values()} != set(range(1, 1855)):
        raise ValueError("Official 1–1,854 class ID series is incomplete")
    concept_list = sorted(concepts.values(), key=lambda a: a["things_id"])
    save_csv(base / "concepts1854.csv", concept_list)
    naming = [dict(things_id=concepts[k]["things_id"], unique_id=k, word=row["Word"],
                   manual_categories=row["Top-down Category (manual selection)"])
              for k, (_, row) in vocabulary.items()]
    save_csv(base / "dictionary.csv", naming, lineterminator="\r\n")
    photo_hashes["dictionary.csv"] = digest(base / "dictionary.csv")
    report = dict(source_sha256=verified, index_sha256=photo_hashes | {"concepts1854.csv": digest(base / "concepts1854.csv")},
                  train_photos=16540, test_photos=200, train_concepts=1654, test_concepts=200, concepts=1854,
                  distinct_display_names=len({r["display_name"] for r in concept_list}), jpeg_modes=modes,
                  unique_key="unique_id, the uniqueID in the THINGS table",
                  global_id="1-based numeric prefix in *_img_concepts_THINGS",
                  split_class_id="1-based numeric prefix in *_img_concepts",
                  eeg_class_label="split_class_index0",
                  tsv_order_exceptions=[r["unique_id"] for r in concept_list if r["things_tsv_row0"]+1 != r["things_id"]])
    (base / "official_photo_index.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
