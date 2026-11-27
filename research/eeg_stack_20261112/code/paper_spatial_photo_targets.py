"""Extract registered visual codes for overlapping spatial crops."""
import argparse
import hashlib
import inspect
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from paper_middle_teacher import CONFIG_SHA, MODEL, MODEL_SHAS, image_processor, registry, selected_zip_digest
from paper_render_cache import PinnedFiles, atomic_file, require, save_json


B32_WEIGHT = "f2d60f0f89c8d7f52e66ae8e1321b3f2fd3b491daba23fbde0a73bbba94828fb"
B32_CONFIG = "1284cbff35169abb23a1c5525a8b0f543c7bd191d4b9aed63880c1571bc4191c"
B32_MAP = "0a96a5915b59d0d46f1b99427ca394aaf08fceb1ecfc0f311f60594c4f3fc344"


def crop_boxes(size):
    width, height = size
    require(width == height and width >= 10, "Expected a square image with room for four crops")
    side = round(.7 * width)
    offset = width - side
    return [(x, y, x + side, y + side) for y in (0, offset) for x in (0, offset)]


def registered_oracle_members(photos, eeg, rows):
    ids, paths = photos["ids"], photos["paths"]
    require(ids == eeg["ids"] and len(ids) == len(set(ids)) == len(paths),
            "Official photograph and EEG identity tables differ")
    members = [paths[i] for i in rows]
    for row, member in zip(rows, members, strict=True):
        path = PurePosixPath(member)
        require(len(path.parts) == 3 and path.parts[0] == "test_images"
                and path.suffix.lower() == ".jpg" and path.stem == ids[row],
                "Unexpected official photograph identity")
    return members


def read_targets(path, rows, phase):
    data = torch.load(path, map_location="cpu", weights_only=True)
    require(data["metadata"]["phase"] == phase and np.array_equal(data["rows"].numpy(), rows),
            "Spatial cache must match the requested row registration")
    features = data["crops"]
    require(features.shape == (len(rows), 4, 1024) and features.dtype == torch.float16
            and torch.isfinite(features).all(), "Malformed cached spatial visual features")
    require(hashlib.sha256(features.numpy().tobytes()).hexdigest() == data["metadata"]["crops_sha256"],
            "Cached spatial features changed")
    return data


@torch.inference_mode()
def prepare(root, phase, archive_path, batch_size):
    from PIL import Image
    from transformers import CLIPVisionModelWithProjection
    from common import EMB_KEYS, PREP, PRIOR, ROOT
    from main.models_adapter import FusionEncoderModel

    files = PinnedFiles()
    official = phase == "oracle"
    split_file = PREP / "paper_split.json"
    split = files.pin(split_file)
    registration = json.loads(split_file.read_bytes())
    id_paths = [PREP / name for name in (("test_ids.json", "eeg_test_ids.json") if official
                                       else ("train_ids.json", "eeg_train_ids.json", "train_concept.npy"))]
    identity = {p.name: files.pin(p) for p in id_paths}
    if official:
        rows = np.asarray(registration["official_rows"], dtype=np.int64)
        require(np.array_equal(rows, np.arange(200)), "Unexpected official registration")
        photos, eeg = (json.loads(path.read_bytes()) for path in id_paths)
        members = registered_oracle_members(photos, eeg, rows)
    else:
        rows, _, members = registry(PREP)

    model_dirs = [ROOT / "models" / key for key in EMB_KEYS[:2]]
    model_shas = [files.pin(p / "model.safetensors") for p in model_dirs]
    config_shas = [files.pin(p / "config.json") for p in model_dirs]
    align_path = PREP / "b32_fix.pt"
    b32_map_sha = files.pin(align_path)
    require(EMB_KEYS[0] == MODEL and model_shas[0] in MODEL_SHAS and model_shas[1] == B32_WEIGHT
            and config_shas == [CONFIG_SHA, B32_CONFIG] and b32_map_sha == B32_MAP,
            "Unexpected frozen CLIP branches or training-derived B32 alignment")
    fusion_dir = PRIOR / "fusion_encoder"
    fusion_proof = {p.name: files.pin(p) for p in
                    (fusion_dir / "config.json", fusion_dir / "diffusion_pytorch_model.safetensors")}
    reference = PREP / ("F_test.npy" if official else "F_train.npy")
    reference_sha = files.pin(reference)
    branches_path = root / "test_visual_embeddings.pt" if official else PREP / "emb_train.pt"
    branches_sha = files.pin(branches_path)
    if official:
        receipt = PREP / "paper_test_F_receipt.json"
        identity[receipt.name] = files.pin(receipt)
        record = json.loads(receipt.read_bytes())
        require(record["ordered_ids"] == photos["ids"] and record["F_test_sha256"] == reference_sha
                and record["test_visual_embeddings_sha256"] == branches_sha,
                "Official visual feature receipt disagrees with the registered photo order")
    reference_full = np.load(reference, mmap_mode="r", allow_pickle=False)
    proof = dict(split_sha256=split, row_files=identity, teacher_weights_sha256=model_shas,
                 teacher_config_sha256=config_shas, full_image_features_sha256=reference_sha,
                 reference_branches_sha256=branches_sha, b32_alignment_sha256=b32_map_sha, fusion_files=fusion_proof,
                 code_sha256=files.pin(Path(__file__)),
                 registry_and_transforms_sha256=files.pin(Path(inspect.getfile(image_processor))),
                 crop_boxes_xyxy=crop_boxes((500, 500)), targets="crop H14/B32 offsets; preserve whole-image VAE")
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"visual-{phase}.pt"
    with zipfile.ZipFile(archive_path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "Ambiguous photograph ZIP")
        proof["ordered_entries_sha256"] = selected_zip_digest(archive, members)
        if output.exists():
            old = read_targets(output, rows, phase)
            require(old["metadata"]["sources"] == proof, "Spatial visual cache was built with other inputs")
            return
        models = [CLIPVisionModelWithProjection.from_pretrained(
            str(p), local_files_only=True, use_safetensors=True, torch_dtype=torch.float16
        ).cuda().eval().requires_grad_(False) for p in model_dirs]
        require([m.config.projection_dim for m in models] == [1024, 512], "Unexpected CLIP architecture")
        original = torch.load(branches_path, map_location="cpu", weights_only=True)
        require(set(original) == set(EMB_KEYS) and [original[k].shape for k in EMB_KEYS]
                == [(len(reference_full), width) for width in (1024, 512, 1024)], "Invalid released visual inputs")
        b32_map = torch.load(align_path, map_location="cuda", weights_only=True).float()
        fusion = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder",
                                                    local_files_only=True).vision_model.cuda().float().eval()
        process = image_processor()

        def load_photo(member):
            raw = archive.read(member)
            with Image.open(io.BytesIO(raw)) as image:
                require(image.size == (500, 500), "Expected a registered 500px photograph")
                return image.convert("RGB"), hashlib.sha256(raw).digest()

        crops = torch.empty(len(rows), 4, 1024, dtype=torch.float16)
        image_hash = hashlib.sha256()
        branch_cosines, fused_cosines = {k: [] for k in EMB_KEYS[:2]}, []
        for start in range(0, len(rows), batch_size):
            images = []
            for member in members[start:start + batch_size]:
                photo, digest = load_photo(member)
                image_hash.update(digest)
                images.append(photo)
                images.extend(photo.crop(box) for box in crop_boxes(photo.size))
            batch = len(images) // 5
            indices = rows[start:start + batch]
            released = {key: original[key][indices].cuda().float() for key in EMB_KEYS}
            pixels = process(images=images, return_tensors="pt").pixel_values.cuda().half()
            changed = {}
            for key, model in zip(EMB_KEYS[:2], models, strict=True):
                codes = model(pixel_values=pixels).image_embeds.float()
                if key == EMB_KEYS[1]:
                    codes = codes @ b32_map
                whole, *regions = codes.reshape(batch, 5, -1).unbind(1)
                branch_cosines[key].append(F.cosine_similarity(whole, released[key]).cpu())
                # Alter only the crop-induced displacement in each released semantic branch.
                changed[key] = (released[key][:, None] + torch.stack(regions, 1) - whole[:, None]).flatten(0, 1)
            changed["vae"] = released["vae"].repeat_interleave(4, dim=0)
            replay = fusion(**released).cpu()
            congruence = F.cosine_similarity(replay, torch.from_numpy(reference_full[indices].copy()))
            fused_cosines.append(congruence)
            require(float(congruence.min()) > .999, "Could not replay the cached whole-photo fusion")
            block = fusion(**changed).reshape(batch, 4, 1024)
            crops[start:start + batch] = block.cpu().half()
            if start % (batch_size * 50) == 0:
                print(phase, "crop targets", start + batch, "/", len(rows), flush=True)
    alignment = dict(replayed_fusion_minimum=float(torch.cat(fused_cosines).min()), checked=len(rows),
                     reencoded_median={k: float(torch.cat(v).median()) for k, v in branch_cosines.items()})
    print("Uncropped visual feature check", alignment, flush=True)
    require(min(alignment["reencoded_median"].values()) > .65,
            "The re-encoded semantic branches disagree with released features")
    require(torch.isfinite(crops).all(), "Nonfinite visual crops")
    files.verify()
    meta = dict(phase=phase, diagnostic_only=official, sources=proof, n=len(rows),
                crops_sha256=hashlib.sha256(crops.numpy().tobytes()).hexdigest(),
                ordered_image_content_sha256=image_hash.hexdigest(), uncropped_check=alignment)
    atomic_file(output, lambda stream: torch.save(dict(metadata=meta, rows=torch.from_numpy(rows), crops=crops), stream))
    save_json(output.with_suffix(".json"), meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "oracle"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    prepare(args.root, args.phase, args.archive, args.batch_size)
