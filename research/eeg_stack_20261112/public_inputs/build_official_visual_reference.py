"""Assemble the official 200-photo evaluation gallery from the Brain-HiVE release on CPU."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from stage_public_inputs import AUTHOR_SOURCES, MANIFEST, ROOT


TEACHER = ROOT / "fusion-prior-source/fusion_encoder"
OUT = ROOT / "official-retrieval-reference"
WEIGHT_SHA = "b345fca423c5fe43178999ae38255bc2297011ce85c1ed664f5d54a9843e6e96"
CONFIG_SHA = "4f72dcc0d21583817047b5987a5589213abecbb706a8c261431e298650a422b3"
TRAIN_PACK_SHA = "85e47d03976789ee0b29e3b005c146ac69caee7eb30d8bd5dfb2ce7bf93562b2"
TRAIN_ID_SHA = "cdfcc38d19517977bee9d1985691a1bec3ea322ce825c280d3227202f94500b9"
CLUSTER_TRAIN_TENSORS = {
    "CLIP-ViT-B-32-laion2B-s34B-b79K": "eb1072737cd6856d436a69d5c7ec69c5f5e0104c1f4d140675901589689bccd0",
    "CLIP-ViT-H-14-laion2B-s32B-b79K": "379670964cce6cd82c3a6ffd4fa839e478921b1987b757c34ea74e6356d89be8",
    "vae": "a139ca1206edd21917d4c687a520104db8a5f81e9ca8a1155a67d1f09e4632af",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_hashes(features):
    return {k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k, v in features.items()}


def release_embeddings(split, dims, manifest):
    import pyarrow.parquet as pq
    import torch

    ids = json.loads((ROOT / f"prep17/{split}_ids.json").read_text())["ids"]
    count = {"train": 16540, "test": 200}[split]
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError(f"Invalid official {split} IDs")
    data, shards = {}, []
    for key, dim in dims.items():
        tensor = torch.empty(count, dim, dtype=torch.float32)
        offset = 0
        for part in range(2):
            file = ROOT / "data/things-eeg/embeddings" / f"things_{split}_{key}-part-{part:05d}-of-00001.parquet"
            record = manifest[file.relative_to(ROOT).as_posix()]
            if file.stat().st_size != record["bytes"] or sha(file) != record["sha256"]:
                raise ValueError(f"Publisher checksum mismatch at {file}")
            first = offset
            for batch in pq.ParquetFile(file).iter_batches(batch_size=256, columns=["image_id", "emb"]):
                names = batch.column(0).to_pylist()
                if names != ids[offset:offset + len(names)]:
                    raise ValueError(f"Publisher row order differs from the EEG at {file}, row {offset}")
                array = batch.column(1).values.to_numpy(zero_copy_only=False).reshape(len(names), dim)
                if array.dtype != np.float32 or not np.isfinite(array).all():
                    raise ValueError(f"Invalid float32 features at {file}")
                tensor[offset:offset + len(names)].copy_(torch.from_numpy(array.copy()))
                offset += len(names)
            shards.append(dict(path=record["path"], url=record["url"], sha256=record["sha256"],
                               rows=offset-first, dimension=dim, key=key))
        if offset != count:
            raise ValueError(f"Incomplete {split} gallery for {key}")
        data[key] = tensor
    return ids, data, shards


def build(args):
    import pyarrow
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    manifest = json.loads(MANIFEST.read_bytes())
    assets = {a["path"]: a for a in manifest["assets"] if a["group"] == "visual"}
    config_file, weight_file = TEACHER / "config.json", TEACHER / "diffusion_pytorch_model.safetensors"
    if sha(config_file) != CONFIG_SHA or sha(weight_file) != WEIGHT_SHA:
        raise ValueError("Fusion Prior differs from the pinned configuration or v239 training weights")
    config = json.loads(config_file.read_text())
    dims = config["proj_meta"]
    train_ids, train, train_shards = release_embeddings("train", dims, assets)
    train_id_bytes = (json.dumps(train_ids, indent=2, allow_nan=False) + "\n").encode()
    if hashlib.sha256(train_id_bytes).hexdigest() != TRAIN_ID_SHA:
        raise ValueError("Training ID order differs from the v239 archive")
    train_hashes = tensor_hashes(train)
    if train_hashes != CLUSTER_TRAIN_TENSORS:
        raise ValueError("Public training tensors differ from the three SHA-256 values measured on the cluster")
    if args.reference_training_pack:
        if sha(args.reference_training_pack) != TRAIN_PACK_SHA:
            raise ValueError("The supplied v239 training pack does not match the archived hash")
        archived = torch.load(args.reference_training_pack, map_location="cpu", weights_only=True)
        if train_hashes != tensor_hashes(archived):
            raise ValueError("One or more train branch tensors differ from the v239 archive")
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT, prefix=".write-train-") as tmp:
        path = Path(tmp) / "train_visual_embeddings.pt"
        torch.save(train, path)
        reconstructed_train = sha(path)
        if tensor_hashes(torch.load(path, map_location="cpu", weights_only=True)) != train_hashes:
            raise ValueError("Saved training embeddings differ from their published tensor hashes")
        id_file = Path(tmp) / "train_image_ids.json"
        id_file.write_bytes(train_id_bytes)
        path.replace(OUT / path.name)
        id_file.replace(OUT / id_file.name)
    del train

    ids, data, test_shards = release_embeddings("test", dims, assets)
    code_file = ROOT / "Brain-HIVE/main/layers.py"
    code_record = json.loads(AUTHOR_SOURCES.read_bytes())["Brain-HIVE"]
    if sha(code_file) != code_record["files"]["main/layers.py"]:
        raise ValueError("Unexpected FusionEncoder code")
    sys.path.insert(0, str(ROOT / "Brain-HIVE"))
    from main.layers import FusionEncoder

    encoder = FusionEncoder(**{k: v for k, v in config.items() if not k.startswith("_")}).float().eval().requires_grad_(False)
    state = {k.removeprefix("vision_model."): v for k, v in load_file(str(weight_file), device="cpu").items()
             if k.startswith("vision_model.")}
    encoder.load_state_dict(state, strict=True)
    if sum(v.numel() for v in encoder.parameters()) != 11015168:
        raise ValueError("Unexpected fusion vision architecture")
    torch.use_deterministic_algorithms(True)
    with torch.inference_mode():
        fused = encoder(**data)
        replay = torch.cat([encoder(**{k: v[i:i + 31] for k, v in data.items()}) for i in range(0, len(ids), 31)])
        abs_err = float((fused - replay).abs().max())
        cosine = float(torch.nn.functional.cosine_similarity(fused, replay).min())
    if fused.shape != (200, 1024) or fused.dtype != torch.float32 or not torch.isfinite(fused).all() or cosine < .99999:
        raise ValueError("Unexpected fused reference output or inconsistent minibatch replay")
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT, prefix=".write-") as tmp:
        temp = Path(tmp)
        pack = temp / "test_visual_embeddings.pt"
        array = temp / "F_test.npy"
        torch.save(data, pack)
        np.save(array, fused.numpy(), allow_pickle=False)
        if tensor_hashes(torch.load(pack, weights_only=True, map_location="cpu")) != tensor_hashes(data):
            raise ValueError("Saved test pack changed a feature tensor")
        if not np.array_equal(np.load(array, allow_pickle=False), fused.numpy()):
            raise ValueError("Saved fused test array changed a value")
        result = dict(
            purpose="Fixed visual gallery for offline scoring of EEG feature predictions.",
            split="official THINGS-EEG2 test", ordered_ids=ids, count=len(ids),
            F_test_sha256=sha(array), F_test_shape=list(fused.shape), F_test_dtype="float32",
            test_visual_embeddings_sha256=sha(pack), test_visual_tensor_sha256=tensor_hashes(data),
            id_file="prep17/test_ids.json", id_file_sha256=sha(ROOT / "prep17/test_ids.json"),
            dimensions=dims, visual_embeddings_revision=manifest["visual_embeddings_revision"], test_shards=test_shards,
            fusion_prior_revision=manifest["fusion_prior_revision"], fusion_encoder_sha256=WEIGHT_SHA,
            fusion_config_sha256=CONFIG_SHA,
            cluster_training_fusion_sha256=WEIGHT_SHA, cluster_training_fusion_hash_confirmed=True,
            fused_space="The frozen vision_model output before the separate adapter, token projection or normalization.",
            author_code_revision=code_record["revision"], author_layer_source_sha256=sha(code_file),
            train_shards=train_shards, original_training_aggregate_sha256=TRAIN_PACK_SHA,
            rebuilt_training_aggregate_sha256=reconstructed_train,
            training_serialized_container_matches_original=reconstructed_train == TRAIN_PACK_SHA,
            original_training_tensors_verified=True,
            original_training_order_sha256=TRAIN_ID_SHA, rebuilt_training_tensor_sha256=train_hashes,
            cluster_training_tensor_sha256=CLUSTER_TRAIN_TENSORS,
            cluster_training_tensor_verification="All 16,540 rows in each branch. SHA-256 over contiguous float32 tensor bytes, measured independently by the cluster CPU check on 2026-11-25.",
            max_abs_error_across_minibatch_sizes=abs_err, min_cosine_across_minibatch_sizes=cosine,
            cpu_torch_version=str(torch.__version__), cpu_threads=4, pyarrow_version=pyarrow.__version__,
            input_representation="Unmodified public H14, B32 and VAE float32 embeddings matched by filename stem.",
        )
        meta = temp / "paper_test_F_receipt.json"
        meta.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for file in (pack, array, meta):
            file.replace(OUT / file.name)
    print("F_test.npy", result["F_test_sha256"], "test_visual_embeddings.pt", result["test_visual_embeddings_sha256"])
    print("Verified v239 training tensor bytes for all 16,540 rows in all three branches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-training-pack", type=Path, help="check all tensors against the original v239 pack")
    build(parser.parse_args())
