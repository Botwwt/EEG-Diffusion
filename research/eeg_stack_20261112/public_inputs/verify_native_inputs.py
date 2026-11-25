"""Re-read the 20 pinned archives and check every prepared sample and trial identity."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from prepare_native_eeg import safe_load_pt
from stage_public_inputs import AUTHOR_SOURCES, MANIFEST, ROOT, digest


PREP = ROOT / "prep17"
PHOTOS = ROOT / "image_set"
POSTERIOR = "P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2".split()
EEG_REVISION = "944f0d50e37e125eb6b7b98d14a41571c9ed0ef7"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_hash(path, expected):
    actual = digest(path)
    require(actual == expected, f"SHA-256 mismatch at {path}: {actual}")
    return actual


def write_atomic(path, data):
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(data)
    tmp.replace(path)


def official_rows(split, metadata, prep_hashes):
    images = "training_images" if split == "train" else "test_images"
    eeg_prefix = "train_images" if split == "train" else "test_images"
    folders = metadata[f"{split}_img_concepts"]
    files = metadata[f"{split}_img_files"]
    ids = [Path(name).stem for name in files]
    paths = [f"{eeg_prefix}/{c}/{f}" for c, f in zip(folders, files, strict=True)]
    names = [c.split("_", 1)[1] for c in folders]
    labels = [int(c.split("_", 1)[0]) - 1 for c in folders]
    global_folders = metadata[f"{split}_img_concepts_THINGS"]
    count = 16540 if split == "train" else 200
    require(len(ids) == len(set(ids)) == count, f"{split} stimulus inventory")
    ids_file = PREP / f"{split}_ids.json"
    require(read_json(ids_file) == {"ids": ids, "paths": paths}, f"{split} shared row IDs")
    prep_hashes[ids_file.name] = digest(ids_file)
    photo_table = PHOTOS / f"{split}_photos.csv"
    with photo_table.open(encoding="utf-8", newline="") as table, zipfile.ZipFile(PHOTOS / f"{images}.zip") as archive:
        for i, (folder, filename, row) in enumerate(zip(folders, files, csv.DictReader(table), strict=True)):
            member = f"{images}/{folder}/{filename}"
            global_id, key = global_folders[i].split("_", 1)
            require(
                int(row["row0"]) == i and row["image_id"] == ids[i] and row["filename"] == filename
                and row["archive_member"] == member and row["unique_id"] == names[i] == key
                and int(row["split_class_index0"]) == labels[i] and int(row["things_id"]) == int(global_id),
                f"{split} photo index identity at row {i}",
            )
            require(hashlib.sha256(archive.read(member)).hexdigest() == row["image_sha256"], f"JPEG at {member}")
    return np.asarray(paths), np.asarray(labels), np.asarray(names)


def verify_eeg(asset, reference, montage, prep_hashes):
    archive = ROOT / asset["path"]
    subject, split = archive.parent.name, archive.stem
    tag = f"{subject}_{split}"
    rec_file = PREP / f"{tag}.json"
    rec = read_json(rec_file)
    nrows, repeats = (16540, 4) if split == "train" else (200, 80)
    expected_shape = (nrows, repeats, 17, 250)
    expected_sourceshape = (nrows, repeats, 63, 250)
    channels = [montage.index(ch) for ch in POSTERIOR]
    ids_file = f"{split}_ids.json"
    session_file = f"sessions/{tag}.npy"
    require(archive.stat().st_size == rec["source_bytes"] == asset["bytes"], f"{tag} source length")
    source_hash = checked_hash(archive, asset["sha256"])
    require(
        rec["subject"] == subject and rec["split"] == split and rec["path"] == f"{tag}.npy"
        and rec["session_file"] == session_file and rec["ids_file"] == ids_file
        and rec["source_sha256"] == source_hash and rec["source_url"] == asset["url"]
        and rec["source_path"] == asset["path"] and f"/{EEG_REVISION}/" in asset["url"]
        and rec["ids_sha256"] == prep_hashes[ids_file] and rec["channel_names"] == POSTERIOR
        and rec["source_channel_indices"] == channels and rec["sample_rate_hz"] == 250
        and rec["axes"] == ["stimulus", "physical_repeat", "channel", "sample"]
        and rec["dtype"] == "float16" and rec["checked_physical_photo_ids"] == nrows * repeats,
        f"{tag} source or axis receipt",
    )
    raw = safe_load_pt(archive)
    source_eeg = np.asarray(raw["eeg"])
    require(
        source_eeg.shape == expected_sourceshape == tuple(rec["source_shape"])
        and source_eeg.dtype == np.dtype("<f2") and list(raw["ch_names"]) == montage,
        f"{tag} source EEG",
    )
    times = np.asarray(raw["times"])
    require(times.shape == (300,) and np.allclose(times, np.arange(-50, 250) / 250, rtol=0, atol=1e-9), f"{tag} times")
    paths, labels, names = reference
    raw_paths = np.asarray(raw["img"])
    row_lookup = {p: i for i, p in enumerate(raw_paths[:, 0])}
    require(len(row_lookup) == nrows and set(row_lookup) == set(paths), f"{tag} row identity")
    order = np.asarray([row_lookup[p] for p in paths], dtype=np.intp)
    row_order_matches = bool(np.array_equal(order, np.arange(nrows)))
    require(rec["source_row_order_matches_official"] == row_order_matches, f"{tag} row order")
    for key, column in (("img", paths), ("label", labels), ("text", names)):
        values = np.asarray(raw[key])
        require(values.shape == (nrows, repeats) and np.all(values[order] == column[:, None]), f"{tag} {key}")
    session_path = PREP / session_file
    sessions = np.load(session_path, allow_pickle=False)
    require(
        sessions.shape == (nrows, repeats) and sessions.dtype == np.uint8 and np.isin(sessions, range(4)).all()
        and np.array_equal(sessions, np.asarray(raw["session"])[order]),
        f"{tag} physical session assignments",
    )
    session_counts = np.stack([(sessions == i).sum(axis=1) for i in range(4)], axis=1)
    require(
        rec["min_session_counts_per_stimulus"] == session_counts.min(axis=0).tolist()
        and rec["max_session_counts_per_stimulus"] == session_counts.max(axis=0).tolist(), f"{tag} session counts",
    )
    if split == "test":
        require(np.all(sessions == np.repeat(np.arange(4), 20)), f"{tag} ordering within the four test sessions")
    eeg_file = PREP / rec["path"]
    eeg = np.load(eeg_file, mmap_mode="r", allow_pickle=False)
    require(
        eeg.shape == expected_shape == tuple(rec["shape"]) and eeg.dtype == np.dtype("<f2") and eeg.flags.c_contiguous
        and eeg.offset + eeg.nbytes == eeg_file.stat().st_size == rec["bytes"], f"{tag} saved EEG header",
    )
    body = hashlib.sha256()
    block_rows = max(1, 320 // repeats)
    for first in range(0, nrows, block_rows):
        rows = slice(first, min(first + block_rows, nrows))
        block = eeg[rows]
        native_rows = source_eeg[rows] if row_order_matches else source_eeg[order[rows]]
        expected = np.take(native_rows, channels, axis=2)
        require(np.isfinite(block).all() and np.array_equal(block.view("<u2"), expected.view("<u2")), f"{tag} EEG row {first}")
        body.update(block)
    require(body.hexdigest() == rec["output_body_sha256"] and rec["sha256"] == rec["output_sha256"], f"{tag} EEG body digest")
    prep_hashes[eeg_file.name] = checked_hash(eeg_file, rec["output_sha256"])
    prep_hashes[session_file] = checked_hash(session_path, rec["session_sha256"])
    prep_hashes[rec_file.name] = digest(rec_file)
    return {
        "subject": subject, "split": split, "source_sha256": source_hash,
        "eeg_sha256": prep_hashes[eeg_file.name], "receipt_sha256": prep_hashes[rec_file.name],
        "session_sha256": prep_hashes[session_file], "ids_sha256": prep_hashes[ids_file],
        "stimulus_rows": nrows, "physical_trials": nrows * repeats, "compared_float16_samples": int(eeg.size),
        "source_row_order_matches_official": row_order_matches, "session_totals": session_counts.sum(axis=0).tolist(),
        "session_per_stimulus_minimum": session_counts.min(axis=0).tolist(),
        "session_per_stimulus_maximum": session_counts.max(axis=0).tolist(),
    }


def main():
    manifest = read_json(MANIFEST)
    require(manifest["eeg_dataset_revision"] == EEG_REVISION, "Unexpected EEG revision")
    other_assets = {}
    for asset in manifest["assets"]:
        if asset["group"] in ("images", "visual"):
            path = ROOT / asset["path"]
            require(path.stat().st_size == asset["bytes"], f"{path} length")
            other_assets[asset["path"]] = checked_hash(path, asset["sha256"])
    code = {}
    for dirname, receipt in read_json(AUTHOR_SOURCES).items():
        for name, expected in receipt["files"].items():
            checked_hash(ROOT / dirname / name, expected)
        code[dirname] = {"revision": receipt["revision"], "verified_files": len(receipt["files"])}
    official = read_json(PHOTOS / "official_photo_index.json")
    for name, expected in official["index_sha256"].items():
        checked_hash(PHOTOS / name, expected)
    metadata = np.load(PHOTOS / "image_metadata.npy", allow_pickle=True).item()
    prepared_hashes = {}
    reference = {s: official_rows(s, metadata, prepared_hashes) for s in ("train", "test")}
    require(set(reference["train"][2]).isdisjoint(reference["test"][2]), "Train and test concepts overlap")
    montage_path = ROOT / "Brain-HIVE/configs/THINGS_EEG_CHANNELS.jsonl"
    entries = sorted((json.loads(line) for line in montage_path.read_text().splitlines()), key=lambda entry: entry["index"])
    require([v["index"] for v in entries] == list(range(63)), "Source montage ordering")
    montage = [v["name"] for v in entries]
    eeg_assets = [a for a in manifest["assets"] if a["group"] == "eeg"]
    expected = {(f"sub-{i:02}", s) for i in range(1, 11) for s in ("train", "test")}
    require(len(eeg_assets) == 20 and {(Path(a["path"]).parent.name, Path(a["path"]).stem) for a in eeg_assets} == expected, "EEG archive inventory")
    checks = []
    for asset in eeg_assets:
        result = verify_eeg(asset, reference[Path(asset["path"]).stem], montage, prepared_hashes)
        checks.append(result)
        print(result["subject"], result["split"], result["physical_trials"], result["eeg_sha256"], flush=True)
    sums = "".join(f"{hashed}  {name}\n" for name, hashed in sorted(prepared_hashes.items()))
    write_atomic(PREP / "SHA256SUMS", sums.encode("ascii"))
    report = {
        "eeg_revision": EEG_REVISION, "public_manifest_sha256": digest(MANIFEST),
        "verified_public_eeg_archives": len(checks), "prepared_sha256sums_sha256": digest(PREP / "SHA256SUMS"),
        "verified_prepared_files": len(prepared_hashes), "verified_photographs": {s: len(ref[0]) for s, ref in reference.items()},
        "total_physical_trials": sum(r["physical_trials"] for r in checks),
        "total_compared_float16_samples": sum(r["compared_float16_samples"] for r in checks),
        "other_verified_public_assets": other_assets, "verified_author_code": code, "eeg": checks,
    }
    write_atomic(ROOT / "native_input_verification.json", (json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print("VERIFIED", report["total_physical_trials"], "physical trials;", report["total_compared_float16_samples"], "samples")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
