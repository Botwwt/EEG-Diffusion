"""Install verified physical EEG and visual galleries into the benchmark's data/prep directory."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile

from stage_public_inputs import AUTHOR_SOURCES, MANIFEST, ROOT, digest


def install(source, destination, expected):
    """Preserve existing files with the expected bytes, including on a retry."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if digest(destination) != expected:
            raise ValueError(f"Existing file differs: {destination}")
        return
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        try:
            os.link(source, staged)
        except OSError:
            shutil.copyfile(source, staged)
        if digest(staged) != expected:
            raise ValueError(f"Source checksum differs: {source}")
        staged.replace(destination)


def handoff(prep, training_targets=None):
    native = ROOT / "prep17"
    gallery = ROOT / "official-retrieval-reference"
    verification = ROOT / "native_input_verification.json"
    report = json.loads(verification.read_bytes())
    metadata = json.loads(MANIFEST.read_bytes())
    eeg_assets = {a["path"]: a for a in metadata["assets"] if a["group"] == "eeg"}
    cases = {(case["subject"], case["split"]) for case in report["eeg"]}
    expected = {(f"sub-{s:02d}", split) for s in range(1, 11) for split in ("train", "test")}
    if (cases != expected or len(report["eeg"]) != 20 or report["eeg_revision"] != metadata["eeg_dataset_revision"]
            or report["total_physical_trials"] != 821600 or report["total_compared_float16_samples"] != 3491800000):
        raise ValueError("Expected the complete verified ten-person data")
    sums = native / "SHA256SUMS"
    if digest(sums) != report["prepared_sha256sums_sha256"]:
        raise ValueError("Prepared EEG manifest differs from the verification report")
    checks = {name: checksum for checksum, name in (line.split("  ", 1) for line in sums.read_text().splitlines())}
    prep.mkdir(parents=True, exist_ok=True)
    for case in report["eeg"]:
        subject, split = case["subject"], case["split"]
        name = f"{subject}_{split}"
        raw = f"data/things-eeg/Preprocessed_data_250Hz_whiten/{subject}/{split}.pt"
        if eeg_assets[raw]["sha256"] != case["source_sha256"]:
            raise ValueError(f"EEG source revision differs at {name}")
        for filename, field in ((name + ".npy", "eeg_sha256"), (name + ".json", "receipt_sha256"),
                                ("sessions/" + name + ".npy", "session_sha256")):
            if checks[filename] != case[field]:
                raise ValueError(f"Verification receipts disagree at {filename}")
            install(native / filename, prep / filename, checks[filename])
    with tempfile.TemporaryDirectory(dir=prep) as temp:
        for split in ("train", "test"):
            ids_name = f"{split}_ids.json"
            eeg_ids = native / ids_name
            install(eeg_ids, prep / f"eeg_{ids_name}", checks[ids_name])
            # EEG uses train_images/ and the original JPEG archive uses training_images/.
            with (ROOT / "image_set" / f"{split}_photos.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            ids = [r["image_id"] for r in rows]
            source_map = json.loads(eeg_ids.read_bytes())
            members = [r["archive_member"] for r in rows]
            expected_members = [p.replace("train_images/", "training_images/", 1) for p in source_map["paths"]]
            if (ids != source_map["ids"] or members != expected_members
                    or [int(r["row0"]) for r in rows] != list(range(len(rows)))):
                raise ValueError(f"Original photograph and EEG row orders disagree for {split}")
            photo_file = Path(temp) / ids_name
            value = {"ids": ids, "paths": members}
            photo_file.write_bytes(json.dumps(value, indent=1, ensure_ascii=False).encode("utf-8"))
            install(photo_file, prep / ids_name, digest(photo_file))
    ref_file = gallery / "paper_test_F_receipt.json"
    ref = json.loads(ref_file.read_bytes())
    true_ids = json.loads((prep / "test_ids.json").read_bytes())["ids"]
    fusion = next(a for a in metadata["assets"] if a["path"] == "fusion-prior-source/fusion_encoder/diffusion_pytorch_model.safetensors")
    if (ref["ordered_ids"] != true_ids or ref["id_file_sha256"] != checks["test_ids.json"]
            or ref["F_test_shape"] != [200, 1024] or ref["F_test_dtype"] != "float32"
            or ref["visual_embeddings_revision"] != metadata["visual_embeddings_revision"]
            or ref["fusion_prior_revision"] != metadata["fusion_prior_revision"]
            or ref["fusion_encoder_sha256"] != fusion["sha256"]):
        raise ValueError("Official visual gallery provenance differs from the published inputs")
    install(gallery / "F_test.npy", prep / "F_test.npy", ref["F_test_sha256"])
    install(ref_file, prep / ref_file.name, digest(ref_file))
    for source in (sums, verification, MANIFEST, AUTHOR_SOURCES):
        install(source, prep / "paper_input_provenance" / source.name, digest(source))
    if training_targets:
        for name, field in (("train_image_ids.json", "original_training_order_sha256"),
                            ("train_visual_embeddings.pt", "rebuilt_training_aggregate_sha256")):
            install(gallery / name, training_targets / name, ref[field])
    print(f"Installed 20 verified EEG arrays, their receipts and sessions, both row maps, and the official gallery in {prep}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", type=Path, required=True, help="Benchmark RAD_ROOT/data/prep directory")
    parser.add_argument("--training-targets", type=Path, help="Also stage the public train ID and tensor pack for the native training recipe")
    args = parser.parse_args()
    handoff(args.prep, args.training_targets)
