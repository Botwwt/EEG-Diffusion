"""Sub-03 official R80 EEG retrieval against independent THINGSplus CC0 images.

Each dictionary entry has one fixed external image. Compare all anchors with
all official photos for duplication without consulting their categories.
Freeze the valid reference set, then rank its CLIP-H/14 projected image vectors
by cosine to the released EEG embeddings. Official labels enter only scoring.

The assets directory holds images_THINGSplus-CC0.zip from the pinned
poojasingh11/THINGS-images dataset and config.json, preprocessor_config.json
and model.safetensors from laion/CLIP-ViT-H-14-laion2B-s32B-b79K. Use the
revision and SHA constants below. The readout directory is the output of
official_object_readout.py with the test_images.zip and evaluation manifests.
"""

import argparse
import csv
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageOps

IMAGE_SHA = "ed86912f6848ac5cf8b405a727f875295cc081cdd64271e80ae4d607b7d447e3"
MODEL_SHA = "036e6e2bd49697511f4f8b8cb5ee465f93025f7a69a145eadeb9a881ace9b18d"
TEST_SHA = "57ed472cd88f68ccf88e9ecbadf3da91ac222d8548684d5680be95bcf47ec14a"
EEG_SHA = "15e96ed9d6626662b85c96dee41ebf1fdaf2d0e66b643b84e52955107a64137f"
IMAGE_REV = "fed41790a15a7a5d64c73564a74a1c9c6f7f720c"
MODEL_REV = "1c2b8495b28150b8a4922ee1c8edee224c284c0c"
BOOT_SEED = 20261125
FILTER = getattr(Image, "Resampling", Image)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, data):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def center(im):
    w, h = im.size
    n = min(w, h)
    left, top = round((w - n) / 2), round((h - n) / 2)
    return im.crop((left, top, left + n, top + n))


def thumb_signature(im):
    from scipy.fft import dctn

    gray = im.convert("L")
    thumb = np.asarray(gray.resize((32, 32), FILTER.LANCZOS), dtype=np.float64)
    low = dctn(thumb, norm="ortho")[:8, :8].reshape(-1)[1:]
    dh = np.asarray(gray.resize((9, 8), FILTER.LANCZOS))

    def bits(v):
        return int.from_bytes(np.packbits(v.ravel()).tobytes(), "big")

    return bits(low > np.median(low)), bits(dh[:, 1:] > dh[:, :-1])


def image_inputs(raw):
    with Image.open(io.BytesIO(raw)) as opened:
        im = ImageOps.exif_transpose(opened).convert("RGB")
    square = center(im)
    thumb = square.resize((128, 128), FILTER.BILINEAR)
    fingerprint = (
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(thumb.tobytes()).hexdigest(),
        thumb_signature(im),
        thumb_signature(square),
    )
    return im, fingerprint, np.asarray(thumb, dtype=np.uint8)


def matches(a, b):
    if a[0] == b[0]:
        return "identical_file"
    if a[1] == b[1]:
        return "identical_rgb_thumbnail"
    # Compare whole-image and center-crop views in both directions.
    if any(
        (p1 ^ p2).bit_count() <= 4 and (d1 ^ d2).bit_count() <= 6
        for p1, d1 in a[2:]
        for p2, d2 in b[2:]
    ):
        return "perceptual_thumbnail_match"
    return None


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or (norm < 1e-9).any():
        raise ValueError("Invalid embedding norm")
    return x / norm


def h14_pixels(im):
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    size = (224, int(224 * h / w)) if w <= h else (int(224 * w / h), 224)
    image = im.resize(size, FILTER.BICUBIC)
    x, y = round((size[0] - 224) / 2), round((size[1] - 224) / 2)
    return np.asarray(image.crop((x, y, x + 224, y + 224)), dtype=np.uint8)


def h14_batch(model, mean, std, pixels):
    import torch

    with torch.inference_mode():
        values = (
            torch.from_numpy(np.stack(pixels)).permute(0, 3, 1, 2).to(mean.device).float() / 255
        )
        output = (
            model(pixel_values=((values - mean) / std).to(model.dtype))
            .image_embeds.float()
            .cpu()
            .numpy()
        )
    return unit(output)


def load_h14(assets, device="cuda"):
    import torch
    from transformers import CLIPVisionModelWithProjection
    from transformers.utils import logging

    for path, expected in (
        (assets / "model.safetensors", MODEL_SHA),
        (
            assets / "config.json",
            "50150597fccadc90f3198d1c2caf2c41d88a40908c38fb428ea56c249b501d85",
        ),
        (
            assets / "preprocessor_config.json",
            "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
        ),
    ):
        if sha(path) != expected:
            raise ValueError(f"Unexpected SHA-256 for {path}")
    logging.set_verbosity_error()
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model, info = CLIPVisionModelWithProjection.from_pretrained(
        str(assets), torch_dtype=dtype, local_files_only=True, output_loading_info=True
    )
    if info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]:
        raise ValueError(f"Visual tower loading errors: {info}")
    config = json.loads((assets / "preprocessor_config.json").read_text())
    if config["size"] != 224 or config["crop_size"] != 224 or config["resample"] != 3:
        raise ValueError("Unexpected pinned H/14 preprocessing")
    mean = torch.tensor(config["image_mean"], device=device)[None, :, None, None]
    std = torch.tensor(config["image_std"], device=device)[None, :, None, None]
    model = model.eval().requires_grad_(False).to(device)
    return model, mean, std


def extract(args, ids):
    import torch
    import transformers

    assets = args.assets
    image_zip = assets / "images_THINGSplus-CC0.zip"
    official_zip = args.readout / "evaluation/test_images.zip"
    for p, expected in ((image_zip, IMAGE_SHA), (official_zip, TEST_SHA)):
        if sha(p) != expected:
            raise ValueError(f"Unexpected SHA-256 for {p}")
    data = read_csv(args.readout / "evaluation/originals_manifest.csv")
    if [int(r["eeg_row"]) for r in data] != list(range(200)):
        raise ValueError("Official original-photo manifest row mismatch")
    target_fingerprints, official = [], []
    with zipfile.ZipFile(official_zip) as z:
        for row in data:
            _, desc, thumb = image_inputs(z.read(row["archive_member"]))
            if desc[0] != row["image_sha256"]:
                raise ValueError("Unexpected official image bytes")
            target_fingerprints.append(desc)
            official.append(thumb)
    model, mean, std = load_h14(assets, args.device)
    enc, thumbs, source, duplicates = [], [], [], []
    start = time.monotonic()
    with zipfile.ZipFile(image_zip) as z, torch.inference_mode():
        lookup = {
            PurePosixPath(name).stem: name for name in z.namelist() if name.lower().endswith(".jpg")
        }
        if len(lookup) != 1854 or set(lookup) != set(ids):
            raise ValueError("CC0 image filenames differ from the frozen THINGS unique IDs")
        for lo in range(0, len(ids), args.batch_size):
            pixels = []
            for j, obj in enumerate(ids[lo : lo + args.batch_size], start=lo):
                im, desc, thumb = image_inputs(z.read(lookup[obj]))
                thumbs.append(thumb)
                source.append(
                    {
                        "index": j,
                        "unique_id": obj,
                        "member": lookup[obj],
                        "sha256": desc[0],
                        "thumbnail_sha256": desc[1],
                        "whole_phash": hex(desc[2][0]),
                        "whole_dhash": hex(desc[2][1]),
                        "crop_phash": hex(desc[3][0]),
                        "crop_dhash": hex(desc[3][1]),
                    }
                )
                for i, truth in enumerate(target_fingerprints):
                    match = matches(desc, truth)
                    if match:
                        duplicates.append({"index": j, "official_row": i, "reason": match})
                pixels.append(h14_pixels(im))
            enc.append(h14_batch(model, mean, std, pixels))
            if lo // args.batch_size % 10 == 0:
                print(
                    f"H/14 external anchors {lo + len(pixels)}/{len(ids)}, {time.monotonic() - start:.1f}s",
                    flush=True,
                )
    block = {r["index"] for r in duplicates}
    for key in ("sha256", "thumbnail_sha256"):
        seen = {}
        for r in source:
            other = seen.get(r[key])
            if other is not None:
                block.update((r["index"], other))
                duplicates.append(
                    {
                        "index": r["index"],
                        "other_external_index": other,
                        "reason": "external_" + key,
                    }
                )
            else:
                seen[r[key]] = r["index"]
    valid = ~np.isin(np.arange(len(ids)), list(block))
    vectors = np.concatenate(enc)
    if vectors.shape != (1854, 1024):
        raise ValueError("Expected 1854 projected 1024-dimensional image embeddings")
    np.savez(
        args.out / "external_anchors.npz",
        unique_ids=ids,
        embeddings=vectors,
        eligible=valid,
        source_sha256=IMAGE_SHA,
        model_sha256=MODEL_SHA,
    )
    np.save(args.out / "external_thumbnails.npy", np.stack(thumbs), allow_pickle=False)
    np.save(args.out / "official_thumbnails.npy", np.stack(official), allow_pickle=False)
    write_csv(args.out / "external_manifest.csv", source)
    dump(
        args.out / "collision_audit.json",
        dict(
            targets=200,
            external=len(ids),
            eligible=int(valid.sum()),
            comparisons=len(ids) * 200,
            global_excluded_indices=sorted(block),
            flagged=duplicates,
            thresholds="Remove globally for identical SHA or RGB128 thumb SHA, or Hamming pHash63 <=4 AND dHash64 <=6 on any full/center combination.",
            torch=torch.__version__,
            transformers=transformers.__version__,
            device=args.device,
            gpu=torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
            cuda_precision=str(model.dtype),
            weights=MODEL_SHA,
            image_archive=IMAGE_SHA,
            official_archive=TEST_SHA,
        ),
    )
    return vectors, valid, np.stack(thumbs), np.stack(official)


def rank(args, ids, words, categories, eeg, emb, eligible):
    sims = unit(eeg) @ unit(emb).T
    sims[:, ~eligible] = -np.inf
    ranking = np.argsort(-sims, axis=1, kind="stable")[:, : int(eligible.sum())]
    catset = [{v.strip() for v in str(c).split(",") if v.strip()} for c in categories]
    allcats = sorted(set.union(*catset))
    available = [
        c for c in allcats if any(c in s and ok for s, ok in zip(catset, eligible, strict=True))
    ]
    proto = unit(
        np.stack(
            [
                emb[[c in s and ok for s, ok in zip(catset, eligible, strict=True)]].mean(0)
                for c in available
            ]
        )
    )
    broad = np.asarray(available)[np.argmax(unit(eeg) @ proto.T, axis=1)]
    entries = [
        dict(
            eeg_row=i,
            predicted_id=ids[ix[0]],
            predicted_word=words[ix[0]],
            top5_ids="|".join(ids[ix[:5]]),
            broad_category=broad[i],
        )
        for i, ix in enumerate(ranking)
    ]
    write_csv(args.out / "predictions.csv", entries)
    np.savez(
        args.out / "readout.npz",
        ranked_ids=ids[ranking],
        broad_predictions=broad,
        category_names=available,
        eligible_ids=ids[eligible],
    )
    return ranking, broad, catset


def evaluate(args, ids, ranking, broad, categories, thumbs, original, emb, eligible_refs):
    from skimage.metrics import structural_similarity

    rows = read_csv(args.readout / "evaluation/scored.csv")
    if [int(r["eeg_row"]) for r in rows] != list(range(200)):
        raise ValueError("Expected 200 independently scored official EEG rows")
    manifest = read_csv(args.readout / "evaluation/originals_manifest.csv")
    if [r["unique_id"] for r in manifest] != [r["truth_id"] for r in rows] or [
        int(r["eeg_row"]) for r in manifest
    ] != list(range(200)):
        raise ValueError("Official scoring labels and original-photo manifest disagree")
    by_id = {v: i for i, v in enumerate(ids)}
    target = np.array([by_id[r["truth_id"]] for r in rows])
    rank_hits = ranking == target[:, None]
    rank = np.where(rank_hits.any(1), rank_hits.argmax(1) + 1, len(ids) + 1)
    text = read_csv(args.readout / "predictions.csv")
    if [int(r["eeg_row"]) for r in text] != list(range(200)):
        raise ValueError("Text baseline row mismatch")
    text_rank = np.array([int(r["truth_rank"]) for r in rows])
    text_index = np.array([by_id[r["predicted_id"]] for r in text])
    own = ranking[:, 0]
    wrong = np.arange(200)
    order = np.random.default_rng(BOOT_SEED).permutation(200)
    wrong[order] = np.roll(order, 1)
    boot = np.random.default_rng(BOOT_SEED).integers(200, size=(10000, 200))
    gray = np.array([0.299, 0.587, 0.114], dtype=np.float32)

    def pixel_metrics(idx):
        x = thumbs[idx].astype(np.float32) / 255
        y = original.astype(np.float32) / 255
        u, v = x.reshape(200, -1), y.reshape(200, -1)
        u = u - u.mean(1, keepdims=True)
        v = v - v.mean(1, keepdims=True)
        corr = np.divide(
            (u * v).sum(1), np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-12)
        )
        ss = np.array(
            [
                structural_similarity(
                    p @ gray,
                    q @ gray,
                    data_range=1,
                    gaussian_weights=True,
                    sigma=1.5,
                    use_sample_covariance=False,
                )
                for p, q in zip(x, y, strict=True)
            ]
        )
        return corr, ss

    corr, ss = pixel_metrics(own)
    corr_text, ss_text = pixel_metrics(text_index)
    corr_wrong, ss_wrong = pixel_metrics(own[wrong])

    def paired(d):
        d = np.asarray(d, dtype=np.float64)
        samples = (
            boot
            if len(d) == 200
            else np.random.default_rng(BOOT_SEED).integers(len(d), size=(10000, len(d)))
        )
        return dict(
            delta=float(d.mean()), ci95=np.quantile(d[samples].mean(1), [0.025, 0.975]).tolist()
        )

    eligible = np.array([bool(categories[i]) for i in target])
    broad_hits = np.array([c in categories[j] for c, j in zip(broad, target, strict=True)])
    text_broad = np.array(
        [r["broad_category"] in categories[j] for r, j in zip(text, target, strict=True)]
    )
    scored = [
        dict(
            eeg_row=i,
            truth_id=ids[j],
            reference_assisted_id=ids[own[i]],
            reference_assisted_word_rank=int(rank[i]),
            text_baseline_id=ids[text_index[i]],
            text_baseline_word_rank=int(text_rank[i]),
            broad_hit=int(broad_hits[i]) if eligible[i] else "",
            predicted_broad=broad[i],
            wrong_row=int(wrong[i]),
            external_rgb_corr=float(corr[i]),
            external_gray_ssim=float(ss[i]),
        )
        for i, j in enumerate(target)
    ]
    write_csv(args.out / "scored.csv", scored)
    summary = dict(
        n=200,
        vocabulary=len(ids),
        eligible_anchors=ranking.shape[1],
        masked_truth_rows=int((~rank_hits.any(1)).sum()),
        top1=int((rank == 1).sum()),
        top5=int((rank <= 5).sum()),
        top10=int((rank <= 10).sum()),
        median_rank=float(np.median(rank)),
        mean_reciprocal_rank=float(np.where(rank_hits.any(1), 1 / rank, 0).mean()),
        chance_top1=1 / ranking.shape[1],
        broad_correct=int(broad_hits[eligible].sum()),
        broad_evaluable=int(eligible.sum()),
        broad_text_correct=int(text_broad[eligible].sum()),
        broad_minus_text=paired(
            broad_hits[eligible].astype(float) - text_broad[eligible].astype(float)
        ),
        broad_task="EEG cosine against 27 unit category centroids of external-image anchors; any manual category of the true object receives credit.",
        wrong_eeg_top1=int((own[wrong] == target).sum()),
        wrong_eeg_top1_expected=float(
            (own[:, None] == target[None, :])[~np.eye(200, dtype=bool)].mean()
        ),
        paired_with_text={
            f"top{k}": paired((rank <= k).astype(float) - (text_rank <= k).astype(float))
            for k in (1, 5, 10)
        },
        correct_new_wrong_text=int(((rank == 1) & (text_rank != 1)).sum()),
        correct_text_wrong_new=int(((rank != 1) & (text_rank == 1)).sum()),
        top1_agreement_with_text=int((own == text_index).sum()),
        external_exemplar_pixels={
            "rgb_corr128": float(corr.mean()),
            "gray_ssim128": float(ss.mean()),
            "minus_wrong_eeg": {
                "rgb_corr128": paired(corr - corr_wrong),
                "gray_ssim128": paired(ss - ss_wrong),
            },
            "minus_text_word_exemplar": {
                "rgb_corr128": paired(corr - corr_text),
                "gray_ssim128": paired(ss - ss_text),
            },
        },
        scope="Single official R80 participant, reference-assisted prediction from a fixed external photo per dictionary word. Pixel metrics refer to the retrieved exemplars. Any generated-image scores require a separate run.",
        bootstrap="10000 paired official EEG rows, fixed reference dictionary and gallery",
        source={
            "image_revision": IMAGE_REV,
            "model_revision": MODEL_REV,
            "image_repo": "https://huggingface.co/datasets/poojasingh11/THINGS-images",
            "model_repo": "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            "data_publication": "https://doi.org/10.3758/s13428-023-02110-8",
            "model_sha256": MODEL_SHA,
            "image_sha256": IMAGE_SHA,
            "official_sha256": TEST_SHA,
            "eeg_npy_sha256": EEG_SHA,
        },
    )
    if args.audit_released_image_space:
        # An evaluation-only check of the two visual domains, after predictions.csv is frozen.
        import torch

        public_file = args.readout / "assets/text_test.pt"
        if sha(public_file) != "2d772e733ae9afe01f60547c38f0026a094347aa56fefd77cf375cf53667cdc9":
            raise ValueError("Unexpected author image-embedding release hash")
        gt_emb = torch.load(public_file, map_location="cpu", weights_only=True)[
            "img_features"
        ].numpy()
        visual = unit(gt_emb) @ unit(emb).T
        visual[:, ~eligible_refs] = -np.inf
        sorted_ref = np.argsort(-visual, axis=1, kind="stable")[:, : ranking.shape[1]]
        where = sorted_ref == target[:, None]
        ranks = np.where(where.any(1), where.argmax(1) + 1, len(ids) + 1)
        summary["official_photo_to_external_same_object_visual_diagnostic"] = {
            "top1": int((ranks == 1).sum()),
            "top5": int((ranks <= 5).sum()),
            "median_rank": float(np.median(ranks)),
            "eligible_anchors": ranking.shape[1],
            "purpose": "Post-readout cross-image category retrieval from released official H/14 photo features, with no EEG.",
        }
    dump(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--readout", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reuse-anchors", action="store_true")
    p.add_argument("--audit-released-image-space", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("Use a positive batch size")
    args.out.mkdir(exist_ok=True, parents=True)
    eeg_path = args.readout / "eeg_r80.npy"
    if sha(eeg_path) != EEG_SHA:
        raise ValueError("Expected the released sub-03 R80 EEG in the official row order")
    eeg = np.load(eeg_path, allow_pickle=False)
    with np.load(args.readout / "dictionary.npz", allow_pickle=False) as dx:
        ids, words, categories = (dx[k] for k in ("unique_ids", "words", "manual_categories"))
    if len(ids) != 1854 or len(set(ids)) != 1854:
        raise ValueError("Expected full THINGS dictionary")
    if args.reuse_anchors:
        with np.load(args.out / "external_anchors.npz", allow_pickle=False) as dx:
            if (
                not np.array_equal(ids, dx["unique_ids"])
                or str(dx["source_sha256"]) != IMAGE_SHA
                or str(dx["model_sha256"]) != MODEL_SHA
            ):
                raise ValueError("Cached anchor registration mismatch")
            emb, eligible = dx["embeddings"], dx["eligible"]
        if (
            emb.shape != (1854, 1024)
            or not np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-5)
            or eligible.dtype != np.bool_
            or eligible.shape != (1854,)
            or not eligible.any()
        ):
            raise ValueError("Malformed cached image features or eligibility mask")
        thumbs = np.load(args.out / "external_thumbnails.npy", allow_pickle=False)
        gt = np.load(args.out / "official_thumbnails.npy", allow_pickle=False)
    else:
        emb, eligible, thumbs, gt = extract(args, ids)
    if eeg.shape != (200, 1024):
        raise ValueError("Expected official sub-03 R80 EEG matrix")
    ranks, broad, catset = rank(args, ids, words, categories, eeg, emb, eligible)
    evaluate(args, ids, ranks, broad, catset, thumbs, gt, emb, eligible)


if __name__ == "__main__":
    main()
