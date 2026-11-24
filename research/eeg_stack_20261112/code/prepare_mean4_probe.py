"""Expose aligned archived sub-03 mean-of-four EEG and VAE modes to the CPU probe."""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def prepare(archive: Path, output: Path) -> None:
    import torch

    data = archive / "data"
    eeg = data / "recovery_20261108_v236"
    visual = data / "native_T_latents_v256"
    manifest = data / "full_T_recovery_20261108_v238/image_manifest.csv"
    registration = json.loads((visual / "registration.json").read_text())
    layout = json.loads((eeg / "sub-03_layout.json").read_text())
    rows = np.asarray(registration["array_rows"], dtype=np.int64)
    t_rows = json.loads((eeg / "current_split_manifest.json").read_text())["image_rows"]["T"]
    if rows.tolist() != t_rows or len(rows) != len(np.unique(rows)):
        raise ValueError("VAE rows disagree with the archived training partition")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        stimuli = sorted(
            (item for item in csv.DictReader(stream) if item["dataset_split"] == "train"),
            key=lambda item: int(item["array_row"]),
        )
    image_ids = json.loads((data / "native_training_v239/train_image_ids.json").read_text())
    if ([int(item["array_row"]) for item in stimuli] != list(range(16540))
            or [Path(item["stimulus_id"]).stem for item in stimuli] != image_ids
            or layout["selected_image_indices"] != list(range(len(stimuli)))):
        raise ValueError("Original EEG and visual stimulus row manifests disagree")
    labels = [item["concept_id"] for item in stimuli]
    unique = list(dict.fromkeys(labels))
    label_ids = {label: i for i, label in enumerate(unique)}
    concepts = np.array([label_ids[label] for label in labels])
    if len(unique) != 1654 or not np.array_equal(concepts, np.arange(16540) // 10):
        raise ValueError("Expected ten ordered photos per original training concept")
    ch17 = "P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2".split()
    channels = [layout["channels"].index(name) for name in ch17]
    raw = torch.load(eeg / "sub-03_train_mean4.pt", weights_only=True, map_location="cpu", mmap=True)
    latent_path = (visual / "native_VAE_T_modes.npy").resolve()
    if tuple(raw.shape) != (16540, 63, 250) or np.load(latent_path, mmap_mode="r").shape != (len(rows), 4, 64, 64):
        raise ValueError("Unexpected archive array shapes")
    # Each invocation gets a new directory. Publish the four-file view only when complete.
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent) as staging:
        temp = Path(staging) / "prep"
        temp.mkdir()
        signal = np.lib.format.open_memmap(
            temp / "sub-03_train.npy", mode="w+", dtype=np.float32, shape=(len(rows), 17, 250)
        )
        for start in range(0, len(rows), 128):
            batch = rows[start:start + 128]
            signal[start:start + len(batch)] = raw[batch][:, channels, :].numpy()
        del signal
        np.save(temp / "train_concept.npy", concepts[rows])
        np.save(temp / "lat512_train_ok.npy", np.ones(len(rows), dtype=bool))
        target = temp / "lat512_train.npy"
        try:
            target.symlink_to(latent_path)
        except OSError:
            try:
                os.link(latent_path, target)
            except OSError:
                shutil.copyfile(latent_path, target)
        temp.rename(output)
    print(f"Aligned {len(rows)} four-repeat means with VAE modes across {len(set(concepts[rows]))} concepts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.archive, args.out)
