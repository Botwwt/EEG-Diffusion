"""Train matched middle/final CLIP-H visual heads alongside the native EEG-to-image-token loss.

Run --prepare once, then --subject 1 and --subject 3 with their usual personal EEG_RUNS_ROOTs.
The visual heads and both visual targets are fitting-only. Each saved state works directly with
paper_object_infer.py. The token-photo checkpoint supplies the registered zero-auxiliary control.
"""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from paper_object_tokens import BATCH, EPOCHS, SEED, cross_photo_targets, epoch_batches, group_index, sha


MODES = ("token-final-aux", "token-middle-aux")
MODEL = "CLIP-ViT-H-14-laion2B-s32B-b79K"
CONFIG_SHA = "50150597fccadc90f3198d1c2caf2c41d88a40908c38fb428ea56c249b501d85"
MODEL_SHAS = {"6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030",
              "036e6e2bd49697511f4f8b8cb5ee465f93025f7a69a145eadeb9a881ace9b18d"}
CACHE = "paper_h14_aux_training.pt"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def aligned_members(rows, labels, folds, ids, paths, eeg_ids):
    """Map just the registered complete training concepts to unique original JPEG members."""
    require(len(labels) == len(ids) == len(paths) == len(eeg_ids), "Incomplete training identity table")
    require(ids == eeg_ids and len(ids) == len(set(ids)), "Training photograph/EEG identities disagree")
    require(labels.ndim == 1 and labels.dtype.kind in "iu" and labels.min() >= 0
            and labels.max() < len(folds), "Unexpected training concept IDs")
    require(rows.ndim == 1 and rows.dtype.kind in "iu" and np.array_equal(rows, np.flatnonzero(folds[labels] != 0)),
            "Training rows must exactly cover the registered non-held-out concepts")
    classes, counts = np.unique(labels[rows], return_counts=True)
    require(np.array_equal(classes, np.flatnonzero(folds != 0)) and np.all(counts == 10),
            "Expected each complete ten-photo training concept")
    members = []
    directories = {}
    for i in rows:
        path = PurePosixPath(paths[i])
        require(len(path.parts) == 3 and path.parts[0] == "training_images" and path.suffix.lower() == ".jpg"
                and path.stem == ids[i], "Unexpected training archive member or photograph identity")
        label = int(labels[i])
        require(directories.setdefault(label, path.parent.name) == path.parent.name, "Mixed concept directory")
        members.append(str(path))
    require(len(set(directories.values())) == len(classes), "Two concept IDs share one image directory")
    return members


def registry(prep):
    from common import concept_folds

    rows = np.asarray(json.loads((prep / "paper_split.json").read_bytes())["train_rows"])
    labels = np.load(prep / "train_concept.npy", allow_pickle=False)
    photos = json.loads((prep / "train_ids.json").read_bytes())
    eeg_ids = json.loads((prep / "eeg_train_ids.json").read_bytes())["ids"]
    require(len(rows) == 13230 and len(labels) == 16540, "Unexpected size of registered training set")
    members = aligned_members(rows, labels, concept_folds(1654), photos["ids"], photos["paths"], eeg_ids)
    return rows, labels[rows], members


def visual_views(model, pixels):
    result = model(pixel_values=pixels, output_hidden_states=True, return_dict=True)
    # Hidden state 0 precedes the first block. Token 0 is the special CLS token.
    patches = result.hidden_states[12][:, 1:]
    return patches.float().mean(1), result.image_embeds.float()


def image_processor():
    from PIL import Image
    from transformers import CLIPImageProcessor

    return CLIPImageProcessor(size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224},
                              do_resize=True, resample=Image.Resampling.BICUBIC, do_center_crop=True,
                              do_convert_rgb=True, do_rescale=True, rescale_factor=1/255, do_normalize=True,
                              image_mean=[.48145466, .4578275, .40821073], image_std=[.26862954, .26130258, .27577711])


def pca_view(raw, width):
    """Fit and apply centered, unwhitened PCA using the supplied training rows only."""
    require(raw.ndim == 2 and 0 < width <= min(len(raw)-1, raw.shape[1]), "Unexpected PCA width")
    x = F.normalize(raw.double(), dim=1)
    x -= x.mean(0)
    values, axes = torch.linalg.eigh(x.T @ x / len(x))
    retained = values[-width:].sum() / values.sum()
    axes = axes[:, -width:].flip((1,))
    signs = axes[axes.abs().argmax(0), torch.arange(width, device=axes.device)].sign()
    y = F.normalize((x @ axes) * signs, dim=1).float()
    require(values[-width] > 0 and torch.isfinite(y).all() and torch.isfinite(retained), "Degenerate visual PCA")
    return y, float(retained)


def cache_inputs(prep, model_dir, rows, members):
    weight = sha(model_dir / "model.safetensors")
    cfg = sha(model_dir / "config.json")
    require(weight in MODEL_SHAS and cfg == CONFIG_SHA, "Unexpected pinned CLIP-H weights or config")
    return dict(model=weight, config=cfg, code=sha(Path(__file__)), split=sha(prep / "paper_split.json"),
                row_ids=hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
                member_ids=hashlib.sha256("\n".join(members).encode()).hexdigest())


def selected_zip_digest(archive, members):
    # Bind the 13,230 permitted photos. No compressed bytes of a held-out photo are read.
    h = hashlib.sha256()
    for member in members:
        info = archive.getinfo(member)
        h.update(f"{member}\t{info.file_size}\t{info.CRC}\n".encode())
    return h.hexdigest()


def load_cache(path, rows, expected=None):
    values = torch.load(path, map_location="cpu", weights_only=True)
    require(np.array_equal(values["rows"].numpy(), rows) and values["schema"] == 1,
            "Visual cache rows or format changed")
    if expected is not None:
        require(values["sources"] == expected, "Existing visual cache comes from different inputs")
    for mode in MODES:
        target = values[mode]
        require(target.shape == (len(rows), 1024) and target.dtype == torch.float32
                and torch.isfinite(target).all(), "Incomplete cached auxiliary features")
        require(hashlib.sha256(target.numpy().tobytes()).hexdigest() == values["feature_sha256"][mode],
                "Cached auxiliary feature data changed")
    return values


@torch.inference_mode()
def prepare(prep, model_dir, archive_path, cache_path, batch_size):
    from PIL import Image
    from transformers import CLIPVisionModelWithProjection

    rows, _, members = registry(prep)
    inputs = cache_inputs(prep, model_dir, rows, members)
    with zipfile.ZipFile(archive_path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "Ambiguous ZIP member names")
        inputs["selected_zip_entries"] = selected_zip_digest(archive, members)
        if cache_path.exists():
            load_cache(cache_path, rows, inputs)
            print("Reusing verified auxiliary features", cache_path, flush=True)
            return
        model = CLIPVisionModelWithProjection.from_pretrained(str(model_dir), local_files_only=True,
                                                               use_safetensors=True).float().cuda().eval().requires_grad_(False)
        cfg = model.config
        require((cfg.num_hidden_layers, cfg.hidden_size, cfg.projection_dim, cfg.patch_size, cfg.image_size)
                == (32, 1280, 1024, 14, 224), "Unexpected H/14 architecture")
        process = image_processor()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        middle, final = torch.empty(len(rows), 1280), torch.empty(len(rows), 1024)
        jpeg_sha = hashlib.sha256()
        for start in range(0, len(rows), batch_size):
            batch = members[start:start + batch_size]
            imgs = []
            for member in batch:
                raw = archive.read(member)
                jpeg_sha.update(hashlib.sha256(raw).digest())
                with Image.open(io.BytesIO(raw)) as image:
                    require(image.size == (500, 500), "Unexpected original photograph dimensions")
                    imgs.append(image.convert("RGB"))
            px = process(images=imgs, return_tensors="pt").pixel_values.cuda()
            m, f = visual_views(model, px)
            sl = slice(start, start + len(batch))
            middle[sl], final[sl] = m.cpu(), f.cpu()
            if start % (batch_size * 50) == 0:
                print("CLIP-H training photos", start + len(batch), "/", len(rows), flush=True)
    del model
    meta = dict(schema=1, sources=inputs, rows=torch.from_numpy(rows.copy()),
                ordered_training_jpeg_sha256=jpeg_sha.hexdigest(),
                pooling="block 12 mean of 256 patch tokens, excluding CLS; final projected CLS",
                transforms="RGB, PIL bicubic shortest edge 224, center crop 224, CLIP normalization",
                historical_pooling_exactly_verified=False, retained_variance={}, feature_sha256={})
    for mode, raw in zip(MODES, (final, middle)):
        target, retained = pca_view(raw, 1024)
        meta[mode] = target.cpu()
        meta["retained_variance"][mode] = retained
        meta["feature_sha256"][mode] = hashlib.sha256(target.numpy().tobytes()).hexdigest()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_path.parent) as td:
        stage = Path(td) / "training.pt"
        torch.save(meta, stage)
        try:
            os.link(stage, cache_path)
        except FileExistsError:
            load_cache(cache_path, rows, inputs)
    print("Training-only H14 targets", len(rows), meta["retained_variance"], cache_path, flush=True)


def symmetric_contrast(a, b):
    scores = F.normalize(a) @ F.normalize(b).T / .07
    labels = torch.arange(len(a), device=a.device)
    return .5 * (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels))


def losses(pred, target, center, variance, eeg_latent, auxiliary_visual):
    token = symmetric_contrast((pred-center).flatten(1), (target-center).flatten(1))
    token = token + ((pred-target).square().mean(-1) / variance).mean()
    return token, symmetric_contrast(eeg_latent, auxiliary_visual)


def completed_fit(file, proof, mode, subject):
    receipt = file.with_suffix(".json")
    if not file.is_file() or not receipt.is_file():
        return False
    rec = json.loads(receipt.read_bytes())
    require(rec["sources"] == proof and rec["checkpoint_sha256"] == sha(file) and rec["subject"] == subject
            and rec["mode"] == mode, "Existing fit has different sources or checkpoint")
    return True


def fit(subject, modes, prep, cache_path, model_dir, archive_path):
    from filelock import FileLock
    from common import HIVE, PRIOR, RUNS
    from paper_instance_residual import make_encoder
    import sys
    if str(HIVE) not in sys.path:
        sys.path.insert(0, str(HIVE))
    from main.layers import ResNet1DBlock
    from main.models_adapter import FusionEncoderModel, IPProjectionModel
    from paper_training import check_eeg

    require(RUNS.name == f"personal-{subject:02d}", "Expected this participant's independent fitting directory")
    rows, labels, members = registry(prep)
    inputs = cache_inputs(prep, model_dir, rows, members)
    with zipfile.ZipFile(archive_path) as archive:
        inputs["selected_zip_entries"] = selected_zip_digest(archive, members)
    targets = load_cache(cache_path, rows, inputs)
    native = RUNS / f"enc/native/sub-{subject:02d}/fold0_seed2025/brain_model.pt"
    dest = RUNS / f"enc/object_tokens/sub-{subject:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    eeg_path = prep / f"sub-{subject:02d}_train.npy"
    proof = dict(native=sha(native), code=sha(Path(__file__)), token_control_code=sha(Path(__file__).with_name("paper_object_tokens.py")),
                 split=inputs["split"], concept=sha(prep / "train_concept.npy"), eeg=sha(eeg_path), F=sha(prep / "F_train.npy"),
                 adapter=sha(PRIOR / "fusion_encoder/diffusion_pytorch_model.safetensors"),
                 projection=sha(PRIOR / "proj/diffusion_pytorch_model.safetensors"), auxiliary=sha(cache_path))
    pending = []
    for mode in modes:
        file = dest / f"{mode}.pt"
        with FileLock(str(file) + ".lock"):
            if not completed_fit(file, proof, mode, subject):
                pending.append(mode)
    if not pending:
        return
    check_eeg((subject,), "train")
    device = torch.device("cuda")
    inverse, ordered = group_index(torch.as_tensor(labels, device=device))
    raw = np.load(eeg_path, mmap_mode="r")
    eeg = torch.cat([torch.from_numpy(raw[r].copy()).to(device=device, dtype=torch.float32).mean(1)
                     for r in np.array_split(rows, math.ceil(len(rows)/128))])
    image = torch.from_numpy(np.load(prep / "F_train.npy", mmap_mode="r")[rows].copy()).to(device)
    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.cuda().float().eval().requires_grad_(False)
    proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").cuda().float().eval().requires_grad_(False)
    with torch.no_grad():
        tokens = torch.cat([proj(adapter(f.float())) for f in image.split(128)])
        require(tokens.shape == (len(rows), 4, 2048) and torch.isfinite(tokens).all(), "Unexpected image token geometry")
        center = tokens.mean(0)
        variance = (tokens-center).square().mean((0, 2)).clamp_min(1e-6)
        target = cross_photo_targets(tokens, inverse, ordered, torch.ones(4, device=device))
    state = torch.load(native, map_location="cpu", weights_only=True)
    steps = EPOCHS * ordered.shape[1] * math.ceil(len(ordered)/BATCH)
    for mode in pending:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        enc = make_encoder(device)
        enc.load_state_dict(state)
        enc.train()
        # The auxiliary head draws no dropout samples, preserving the native EEG dropout stream.
        aux = torch.nn.Sequential(torch.nn.Linear(1024, 1024),
                                  ResNet1DBlock(1024, 1024, num_layers=1, dropout=0.)).to(device)
        visual = targets[mode].to(device)
        params = list(enc.parameters()) + list(aux.parameters())
        opt = torch.optim.AdamW(params, lr=8e-5, betas=(.9, .995), weight_decay=.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-6)
        g = torch.Generator(device=device).manual_seed(SEED)
        log = []
        for epoch in range(EPOCHS):
            totals = np.zeros(2)
            for ix in epoch_batches(ordered, g):
                c = enc.backbone(eeg[ix]).float()
                token, auxiliary = losses(proj(adapter(c)), target[ix], center, variance, c, aux(visual[ix]))
                opt.zero_grad(set_to_none=True)
                (token + auxiliary).backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1., error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(aux.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                sched.step()
                totals += np.array([float(token.detach()), float(auxiliary.detach())]) * len(ix)
            log.append(dict(zip(("token", "auxiliary"), (totals / len(rows)).tolist())))
            print(mode, subject, epoch+1, log[-1], flush=True)
        file = dest / f"{mode}.pt"
        with tempfile.TemporaryDirectory(dir=dest) as td:
            stage = Path(td) / file.name
            torch.save({k: v.cpu() for k, v in enc.state_dict().items()}, stage)
            rec = dict(mode=mode, subject=subject, sources=proof, checkpoint_sha256=sha(stage), seed=SEED, steps=steps,
                       photos=len(rows), concepts=len(ordered), epochs=EPOCHS, loss=log, weight=1., temperature=.07,
                       auxiliary_parameters=sum(p.numel() for p in aux.parameters()), auxiliary_saved=False,
                       training_jpeg_sha256=targets["ordered_training_jpeg_sha256"],
                       training_supervision="frozen native IP tokens per photo and training-only CLIP-H visual target",
                       inference="EEG BrainMLP -> frozen adapter -> frozen token projection", frozen_generator=True)
            receipt = stage.with_suffix(".json")
            receipt.write_text(json.dumps(rec, indent=1, allow_nan=False))
            with FileLock(str(file) + ".lock"):
                if not completed_fit(file, proof, mode, subject):
                    stage.replace(file)
                    receipt.replace(file.with_suffix(".json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--subject", type=int, choices=range(1, 11))
    p.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    p.add_argument("--batch-size", type=int, default=24, help="Visual extraction microbatch")
    args = p.parse_args()
    from common import PREP, ROOT
    model_dir = ROOT / "models" / MODEL
    archive_path = Path(os.environ["ARCHIVE"]) / "assets/things_images_v254/training_images.zip"
    cache_path = PREP / CACHE
    if args.prepare:
        require(args.batch_size > 0, "Visual batch size must be positive")
        prepare(PREP, model_dir, archive_path, cache_path, args.batch_size)
    else:
        fit(args.subject, args.modes, PREP, cache_path, model_dir, archive_path)
