"""Bind the published AlignVis image embeddings to the registered THINGS-EEG2 photographs."""

import argparse
import csv
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np

from paper_render_cache import PinnedFiles, atomic_file, require, save_json


REVISION = "7955a8e5bd77535cc5178e0fcacb5f53edcf41b2"
TOWERS = {
    "synclr-base": ("_dreamsim_synclr_vitb16_noalign.npy", 768),
    "synclr-aligned": ("_dreamsim_synclr_vitb16.npy", 768),
    "clip-base": ("_CLIP_ViT-B32_noalign.npy", 512),
    "clip-aligned": ("_dreamsim_clip_vitb32.npy", 512),
    "dinov2-base": ("_DINOv2_ViT-B14_noalign.npy", 768),
    "dinov2-aligned": ("_dreamsim_dinov2_vitb14.npy", 768),
    "h14": ("_OpenCLIP_ViT-H14_laion2b_noalign.npy", 1024),
}


def photo_registry(assets, files):
    data, proof = {}, {}
    for phase, expected in (("train", 16540), ("test", 200)):
        path = assets / f"{phase}_photos.csv"
        proof[phase] = files.pin(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
        require([int(r["row0"]) for r in records] == list(range(expected)), "Unexpected photograph registry order")
        stems = [PurePosixPath(r["archive_member"]).with_suffix("").as_posix() for r in records]
        require(len(set(stems)) == expected, "Photo registry contains duplicate archive members")
        data[phase] = (records, stems)
    return data, proof


def read_shards(release, assets, output):
    files = PinnedFiles()
    manifest_path = release / "release.json"
    manifest_sha = files.pin(manifest_path)
    metadata = json.loads(manifest_path.read_bytes())
    entries = metadata["entries"]
    require(metadata["revision"] == REVISION and len(entries) == 66, "Wrong AlignVis release")
    registry, proof = photo_registry(assets, files)
    index = {stem: (phase, i) for phase, (_, stems) in registry.items() for i, stem in enumerate(stems)}
    arrays = {
        phase: {key: np.full((len(stems), dim), np.nan, np.float32) for key, (_, dim) in TOWERS.items()}
        for phase, (_, stems) in registry.items()
    }
    count = {f"{phase}/{key}": 0 for phase in arrays for key in TOWERS}
    payload_proof = {}
    suffixes = tuple(v[0] for v in TOWERS.values())
    at_suffix = dict(zip(suffixes, TOWERS, strict=True))
    for entry in entries:
        shard = release / "shards" / PurePosixPath(entry["rfilename"]).name
        digest = files.pin(shard)
        require(shard.stat().st_size == entry["size"] and digest == entry["lfs"]["sha256"], f"Unexpected {shard.name}")
        payload_proof[shard.name] = digest
        with tarfile.open(shard, "r:") as package:
            for member in package:
                if not member.name.endswith(suffixes):
                    continue
                suffix = next(s for s in suffixes if member.name.endswith(s))
                stem = member.name[:-len(suffix)]
                require(member.isfile() and stem in index and 1000 <= member.size <= 10000,
                        f"Unregistered embedding {member.name}")
                phase, idx = index[stem]
                key = at_suffix[suffix]
                slot = arrays[phase][key][idx]
                with package.extractfile(member) as handle:
                    value = np.load(io.BytesIO(handle.read()), allow_pickle=False).ravel()
                require(np.isnan(slot).all() and value.shape == slot.shape and value.dtype.kind == "f"
                        and np.isfinite(value).all() and np.linalg.norm(value) > 0, f"Unexpected embedding {member.name}")
                slot[:] = value
                count[f"{phase}/{key}"] += 1
    require(all(np.isfinite(v).all() for a in arrays.values() for v in a.values()), "The selected releases contain missing features")
    data = {f"{phase}/{key}": val for phase, model in arrays.items() for key, val in model.items()}
    for phase, (rows, _) in registry.items():
        data[f"{phase}_image_id"] = np.array([r["image_id"] for r in rows])
        data[f"{phase}_word"] = np.array([r["unique_id"] for r in rows])
    files.verify()
    atomic_file(output, lambda stream: np.savez(stream, **data))
    save_json(output.with_suffix(".json"), dict(
        revision=REVISION, manifest_sha256=manifest_sha, shards=payload_proof, photo_registries=proof,
        array_sha256=files.pin(output), features=count, towers=TOWERS,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("release", type=Path)
    cli.add_argument("assets", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    read_shards(args.release, args.assets, args.output)
