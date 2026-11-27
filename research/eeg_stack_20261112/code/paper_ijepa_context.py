"""Extract training-only I-JEPA targets based on masked context prediction error."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from paper_middle_teacher import aligned_members
from paper_render_cache import PinnedFiles, atomic_file, require, save_json


MODES = ("mean-map", "photo-map", "same-object-map")
UPSTREAM = "52c1ae95d05f743e000e8f10a1f3a79b10cff048"
CHECKPOINT_BYTES = 10358004345
RANK = 512


def patch_indices(side, device=None):
    require(side > 0 and side % 2 == 0, "A quadrant grid needs an even side")
    grid = torch.arange(side**2, device=device).reshape(side, side)
    half = side // 2
    quadrants = grid.unfold(0, half, half).unfold(1, half, half).reshape(4, half**2)
    hidden = (grid.flatten()[None, :, None] == quadrants[:, None, :]).any(-1)
    visible = grid.flatten().expand(4, -1)[~hidden].reshape(4, -1)
    return visible, quadrants


def context_targets(pixels, encoder, teacher, predictor, side=16):
    """Apply the official context encoder to visible tokens before every block."""
    visible, hidden = patch_indices(side, pixels.device)
    n = len(pixels)
    context_masks = [p.expand(n, -1) for p in visible]
    target_masks = [p.expand(n, -1) for p in hidden]
    full = teacher(pixels).float()
    require(full.ndim == 3 and full.shape[:2] == (n, side**2), "Wrong target patch layout")
    full = F.layer_norm(full, (full.shape[-1],))
    context = encoder(pixels, context_masks)
    # One full batch per quadrant. Single mask lists pair each context with
    # its own missing region rather than forming a cross product of masks.
    guess = predictor(context, torch.cat(context_masks), torch.cat(target_masks)).float()
    guess = guess.reshape(4, n, hidden.shape[1], full.shape[-1])
    error = full.new_empty(n, side**2)
    for quadrant, selected in enumerate(hidden):
        error[:, selected] = F.smooth_l1_loss(guess[quadrant], full[:, selected], reduction="none").mean(-1)
    return full, error


def patch_pool(patches, error, average, scales, donors, foreground=None):
    """Pool the current photo in every arm, using different sources of patch weights."""
    position = (average - average.mean()) / average.std(unbiased=False).clamp_min(.01)
    specific = ((error - average) / scales).clamp(-4., 4.)
    scores = {
        "mean-map": position.expand_as(specific),
        "photo-map": position + specific,
        "same-object-map": position + specific[donors],
    }
    if foreground is None:
        distribution = {arm: (.5 / patches.shape[1]) + .5 * value.softmax(1) for arm, value in scores.items()}
    else:
        require(foreground.shape == error.shape and torch.all((foreground >= 0) & (foreground <= 1)),
                "Expected one soft foreground weight per patch")
        support = torch.where(foreground.sum(1, keepdim=True) > 0, foreground, torch.ones_like(foreground))
        support = support / support.sum(1, keepdim=True)
        log_base = support.log()
        distribution = {arm: .5 * support + .5 * (log_base + value).softmax(1) for arm, value in scores.items()}
    pooled = {arm: torch.einsum("np,npd->nd", weights, patches) for arm, weights in distribution.items()}
    return pooled, distribution


def register(prep, eeg_ids, files):
    split, labels, ids = (prep / name for name in ("paper_split.json", "train_concept.npy", "train_ids.json"))
    proof = dict(split=files.pin(split), concepts=files.pin(labels), photo_ids=files.pin(ids),
                 eeg_ids=files.pin(eeg_ids), code=files.pin(__file__))
    rows = np.asarray(json.loads(split.read_bytes())["train_rows"])
    photos = json.loads(ids.read_bytes())
    labels = np.load(labels, allow_pickle=False)
    folds = np.empty(1654, dtype=int)
    folds[np.random.default_rng(20261112).permutation(len(folds))] = np.arange(len(folds)) % 5
    identities = json.loads(eeg_ids.read_bytes())["ids"]
    members = aligned_members(rows, labels, folds, photos["ids"], photos["paths"], identities)
    proof["eeg_identity_list"] = hashlib.sha256("\n".join(identities).encode()).hexdigest()
    require(len(rows) == 13230 and labels.shape == (16540,), "Unexpected training registration size")
    groups = np.argsort(labels[rows], kind="stable").reshape(-1, 10)
    require(np.all(labels[rows][groups] == labels[rows][groups[:, :1]]), "Mixed registered object groups")
    return rows, members, groups, proof


def networks(upstream, checkpoint, digest, files, device):
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(upstream), "diff", "--quiet", "HEAD", "--", "src"], check=True)
    require(revision == UPSTREAM, "Expected the pinned I-JEPA release")
    require(checkpoint.stat().st_size == CHECKPOINT_BYTES and files.pin(checkpoint) == digest,
            "Weight file differs from the verified download")
    sys.path.insert(0, str(upstream))
    from src.models.vision_transformer import vit_huge, vit_predictor

    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    require(set(("encoder", "target_encoder", "predictor")).issubset(state), "A full I-JEPA checkpoint is required")
    models = []
    for key in ("encoder", "target_encoder", "predictor"):
        build = (lambda: vit_huge(patch_size=14)) if key != "predictor" else (
            lambda: vit_predictor(num_patches=256, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=16))
        model = build().eval().requires_grad_(False)
        model.load_state_dict({name.removeprefix("module."): tensor for name, tensor in state[key].items()},
                              strict=True, assign=True)
        models.append(model.to(device=device, dtype=torch.bfloat16))
    files.verify()
    return tuple(models)


def image_pixels(archive, members, indices, device, checksums):
    pixels = []
    for i in indices:
        raw = archive.read(members[i])
        checksums[int(i)] = hashlib.sha256(raw).digest()
        with Image.open(io.BytesIO(raw)) as source:
            require(source.size == (500, 500), "Expected an original square training photograph")
            img = source.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
            pixels.append(np.asarray(img))
    x = torch.from_numpy(np.stack(pixels)).to(device).permute(0, 3, 1, 2).float() / 255.
    mean, sigma = x.new_tensor([.485, .456, .406]), x.new_tensor([.229, .224, .225])
    return ((x - mean[None, :, None, None]) / sigma[None, :, None, None]).to(torch.bfloat16)


def shared_coordinates(raw, device):
    common = torch.cat([F.normalize(raw[arm].to(device), dim=1) for arm in MODES], dim=0).double()
    center = common.mean(0)
    common -= center
    values, axes = torch.linalg.eigh(common.T @ common / len(common))
    require(values[-RANK] > 0 and torch.isfinite(values).all(), "Invalid training visual target covariance")
    basis = axes[:, -RANK:].flip(1)
    signs = basis[basis.abs().argmax(0), torch.arange(RANK, device=device)].sign()
    projected = F.normalize((common @ basis) * signs, dim=1).float().cpu().chunk(len(MODES))
    return dict(zip(MODES, projected, strict=True)), float(values[-RANK:].sum() / values.sum())


def foreground_patches(path, rows, files, device):
    from paper_outline_routing import MASK_RECEIPTS

    require(files.pin(path) == MASK_RECEIPTS["train"], "Expected the previously audited training mask archive")
    with np.load(path, allow_pickle=False) as registered:
        mask = registered["masks"]
        require(np.array_equal(registered["rows"], rows) and registered["valid"].all()
                and mask.shape == (len(rows), 160, 160) and mask.dtype == np.uint8,
                "Foreground masks must match the registered training JPEGs")
    bins = mask.reshape(len(rows), 16, 10, 16, 10).mean(axis=(2, 4), dtype=np.float32)
    return torch.from_numpy(bins.reshape(len(rows), 256)).to(device) / 255.


@torch.inference_mode()
def prepare(args):
    device = torch.device(args.device)
    files = PinnedFiles()
    rows, members, groups, proof = register(args.prep, args.eeg_ids, files)
    models = networks(args.upstream, args.checkpoint, args.sha256, files, device)
    proof.update(dict(source_revision=UPSTREAM, checkpoint=args.sha256, dtype="bfloat16",
                      encoder_code=files.pin(args.upstream / "src/models/vision_transformer.py")))
    foreground = None
    if args.foreground:
        foreground = foreground_patches(args.foreground, rows, files, device)
        proof["foreground_masks"] = files.pin(args.foreground)
    start = time.monotonic()
    checksums = [None] * len(rows)
    with zipfile.ZipFile(args.images) as photos:
        from paper_middle_teacher import selected_zip_digest

        require(len(set(photos.namelist())) == len(photos.namelist()), "Duplicate names in photograph archive")
        proof["training_jpeg_members"] = selected_zip_digest(photos, members)
        if args.smoke:
            x = image_pixels(photos, members, groups[0, :args.batch_size], device, checksums)
            patches, err = context_targets(x, *models)
            require(torch.isfinite(patches).all() and torch.isfinite(err).all() and torch.any(err > 0),
                    "I-JEPA emitted no usable context features")
            print("Pinned checkpoint smoke", tuple(patches.shape), float(err.mean()), time.monotonic()-start, flush=True)
            return
        calibration = groups[np.arange(len(groups)), np.random.default_rng(20261127).integers(10, size=len(groups))]
        if args.calibration:
            proof["shared_calibration"] = files.pin(args.calibration)
            source = json.loads(args.calibration.read_bytes())
            keys = ("checkpoint", "split", "training_jpeg_members", "encoder_code")
            require(source["calibration_rows"] == rows[calibration].tolist()
                    and all(source["proof"][key] == proof[key] for key in keys),
                    "Calibration must come from the same training photographs and visual teacher")
            mean = torch.tensor(source["mean_patch_error"], device=device)
            scales = torch.tensor(source["patch_error_scale"], device=device)
        else:
            error_batches = []
            for indices in np.array_split(calibration, (len(calibration) + args.batch_size - 1) // args.batch_size):
                x = image_pixels(photos, members, indices, device, checksums)
                _, error = context_targets(x, *models)
                error_batches.append(error)
            reference = torch.cat(error_batches)
            mean = reference.mean(0)
            scales = reference.std(0, unbiased=False).clamp_min(.01)
        print("I-JEPA prediction error calibration", len(calibration), float(mean.mean()),
              time.monotonic()-start, flush=True)
        raw = {arm: torch.empty(len(rows), 1280) for arm in MODES}
        stats = torch.zeros(3, device=device)
        for pos in range(0, len(groups), 5):
            local = groups[pos:pos+5].flatten()
            features, errors = [], []
            for i in range(0, len(local), args.batch_size):
                x = image_pixels(photos, members, local[i:i+args.batch_size], device, checksums)
                patch, error = context_targets(x, *models)
                features.append(patch)
                errors.append(error)
            patch, err = torch.cat(features), torch.cat(errors)
            donor = torch.arange(len(local), device=device).reshape(-1, 10).roll(1, dims=1).flatten()
            current_mask = None if foreground is None else foreground[local]
            values, w = patch_pool(patch, err, mean, scales, donor, foreground=current_mask)
            for arm, value in values.items():
                raw[arm][local] = value.cpu()
            stats += torch.stack((err.mean(1).sum(), (w["photo-map"] - w["same-object-map"]).abs().sum(),
                                  (w["photo-map"] - w["mean-map"]).abs().sum()))
            if pos % 100 == 0 or pos+5 >= len(groups):
                print("I-JEPA training photo targets", min(pos+5, len(groups))*10, len(rows),
                      round(time.monotonic()-start, 1), flush=True)
    del models
    torch.cuda.empty_cache()
    require(all(checksums), "The registered training images were not all read")
    values, retained = shared_coordinates(raw, device)
    require(all(t.shape == (len(rows), RANK) and torch.isfinite(t).all() for t in values.values()),
            "Unexpected target features")
    proof["training_jpeg_sha256"] = hashlib.sha256(b"".join(checksums)).hexdigest()
    info = dict(proof=proof, n=int(len(rows)), retained_variance=retained, calibration_rows=rows[calibration].tolist(),
                mean_patch_error=mean.cpu().tolist(), patch_error_scale=scales.cpu().tolist(),
                error_and_map_differences=(stats / len(rows)).cpu().tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files.verify()
    atomic_file(args.output, lambda f: torch.save(dict(rows=torch.from_numpy(rows), targets=values, info=info), f))
    save_json(args.output.with_suffix(".json"), dict(info, packet_sha256=files.pin(args.output)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("prep", "eeg-ids", "upstream", "checkpoint", "images", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--sha256", required=True)
    cli.add_argument("--batch-size", default=10, type=int)
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--foreground", type=Path, help="Use registered training foreground masks as the patch prior")
    cli.add_argument("--calibration", type=Path, help="Reuse the audited training-patch calibration receipt")
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    if args.batch_size <= 0:
        cli.error("The feature extraction batch size must be positive")
    torch.set_num_threads(6)
    prepare(args)
