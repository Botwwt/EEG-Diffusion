"""Train a four-trial EEG readout with sharp or multiscale visual supervision."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, atomic_file, require, save_json


MODES = ("sharp", "multiscale")
ARMS = ("shared-d2", *MODES, "multiscale-wrong")
SUBJECTS = tuple(range(1, 11))
KERNELS = (1, 9, 21, 39)
EPOCHS = 32
BATCH = 384
SEED = 20261129


def combine_views(views, weights, mode):
    if mode == "sharp":
        views = views[:, :1].expand_as(views)
    return (views.float() * weights.softmax(dim=0)).sum(dim=1)


def tangent(values, axis):
    orthogonal = values.float() - (values.float() @ axis)[:, None] * axis
    return F.normalize(orthogonal, dim=-1, eps=1e-6)


def object_logits(prediction, photos, sums, labels):
    logits = prediction @ F.normalize(sums).T
    own = F.normalize(sums[labels] - photos)
    return logits.scatter(1, labels[:, None], (prediction * own).sum(1, keepdim=True))


class BlurCondition(nn.Module):
    def __init__(self, mean, scale, axis, views=len(KERNELS)):
        super().__init__()
        people, channels, samples = mean.shape
        width = axis.numel()
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_scale", scale)
        self.register_buffer("axis", axis)
        virtual, time_width = 25, 200
        spatial = torch.empty(people, virtual, channels)
        nn.init.normal_(spatial, mean=-.5, std=.12)
        spatial[:, torch.arange(virtual), torch.arange(virtual) % channels] += 1.4
        self.spatial = nn.Parameter(spatial)
        self.eeg_encoder = nn.Sequential(
            nn.BatchNorm1d(virtual),
            nn.Linear(samples, time_width), nn.ELU(), nn.Dropout(.25),
            nn.Linear(time_width, time_width), nn.ELU(), nn.Dropout(.65),
            nn.Flatten(), nn.Linear(virtual * time_width, width), nn.LayerNorm(width),
        )
        self.blur_weights = nn.Parameter(torch.zeros(views, width))
        self.visual_adapter = nn.Sequential(
            nn.Linear(width, 768), nn.ELU(), nn.Dropout(.70),
            nn.Linear(768, width), nn.LayerNorm(width),
        )
        self.to_native = nn.Linear(width, width)

    def encode_eeg(self, eeg, subject):
        eeg = (eeg.float() - self.input_mean[subject]) / self.input_scale[subject]
        mix = self.spatial[subject].softmax(dim=-1)
        return F.normalize(self.eeg_encoder(mix @ eeg).float())

    def encode_photo(self, views, mode):
        visual = combine_views(views, self.blur_weights, mode)
        return F.normalize(self.visual_adapter(visual).float())

    def decode(self, encoded):
        return tangent(self.to_native(encoded), self.axis)

    def forward(self, eeg, subject):
        return self.decode(self.encode_eeg(eeg, subject))


def batch_alignment(a, b, temperature=.08):
    scores = a @ b.T / temperature
    correct = torch.arange(len(a), device=a.device)
    return .5 * (F.cross_entropy(scores, correct) + F.cross_entropy(scores.T, correct))


def bind_cache(bank, rows, inputs):
    values = bank["features"]
    require(bank["inputs"] == inputs and np.array_equal(bank["rows"].numpy(), rows)
            and values.shape == (len(rows), len(KERNELS), 1024)
            and values.dtype == torch.float16 and torch.isfinite(values).all(), "Invalid visual training bank")
    return values


@torch.inference_mode()
def prepare(output):
    from PIL import Image
    from torchvision.transforms.functional import gaussian_blur
    from transformers import CLIPVisionModelWithProjection
    from common import PREP, ROOT
    from paper_middle_teacher import MODEL, cache_inputs, image_processor, registry, selected_zip_digest

    files = PinnedFiles()
    rows, _, members = registry(PREP)
    model_dir = ROOT / "models" / MODEL
    archive = Path(os.environ["ARCHIVE"]) / "assets/things_images_v254/training_images.zip"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "training-blur-features.pt"
    inputs = dict(cache_inputs(PREP, model_dir, rows, members), builder=files.pin(Path(__file__)),
                  kernels=list(KERNELS), resize=224, clip_dtype="torch.float16")
    with zipfile.ZipFile(archive) as images:
        require(len(images.namelist()) == len(set(images.namelist())), "Ambiguous image archive entries")
        inputs["selected_zip_entries"] = selected_zip_digest(images, members)
        if path.is_file():
            bind_cache(torch.load(path, map_location="cpu", weights_only=True), rows, inputs)
            return
        model = CLIPVisionModelWithProjection.from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True, torch_dtype=torch.float16
        ).cuda().eval()
        processor = image_processor()
        features = torch.empty(len(rows), len(KERNELS), 1024, dtype=torch.float16)
        jpeg_digest = hashlib.sha256()
        batch = 24
        for start in range(0, len(rows), batch):
            pictures = []
            for name in members[start:start + batch]:
                data = images.read(name)
                jpeg_digest.update(hashlib.sha256(data).digest())
                with Image.open(io.BytesIO(data)) as image:
                    require(image.size == (500, 500), "Unexpected training photograph dimensions")
                    pictures.append(image.convert("RGB"))
            pixels = processor(images=pictures, return_tensors="pt").pixel_values.cuda()
            blur = [pixels, *(gaussian_blur(pixels, [k, k]) for k in KERNELS[1:])]
            z = model(pixel_values=torch.cat(blur).half()).image_embeds.float()
            z = F.normalize(z).reshape(len(KERNELS), len(pictures), 1024).transpose(0, 1)
            features[start:start + len(pictures)] = z.cpu().half()
            if (start // batch) % 25 == 0:
                print("Visual targets", start + len(pictures), "/", len(rows), flush=True)
    require(torch.isfinite(features).all(), "Nonfinite visual target")
    files.verify()
    atomic_file(path, lambda stream: torch.save(
        dict(inputs=inputs, rows=torch.from_numpy(rows.copy()), features=features,
             ordered_training_jpeg_sha256=jpeg_digest.hexdigest()), stream
    ))
    print("Complete four-scale visual targets", len(rows), path, flush=True)


def measured_mean(eeg, rows):
    out = []
    for start in range(0, len(rows), 256):
        wave = torch.from_numpy(eeg[rows[start:start + 256], :4].copy()).cuda().float().mean(1)
        out.append(wave.half())
    return torch.cat(out)


def fit(mode, output, epochs):
    from common import PREP, RUNS
    from paper_d2_object_swap import d2_models
    from paper_middle_teacher import registry
    from paper_render import physical_eeg

    require(RUNS.name == "shared", "Select the registered shared D2 environment")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    files = PinnedFiles()
    rows, labels, _ = registry(PREP)
    split = files.pin(PREP / "paper_split.json")
    source = output / "training-blur-features.pt"
    photo = PREP / "F_train.npy"
    sources = {p.name: files.pin(p) for p in (source, photo, Path(__file__))}
    bank = torch.load(source, map_location="cpu", weights_only=True)
    require(bank["inputs"]["builder"] == sources[Path(__file__).name] and bank["inputs"]["split"] == split
            and bank["inputs"]["kernels"] == list(KERNELS), "Visual features were built for a different fit")
    views = bind_cache(bank, rows, bank["inputs"]).cuda()
    del bank
    encoder, residual, cone, d2 = d2_models(files)
    axis = F.normalize(cone["u_tr"].reshape(-1).cuda().float(), dim=0)
    del encoder, residual
    native = torch.from_numpy(np.load(photo, mmap_mode="r", allow_pickle=False)[rows].copy()).cuda().float()
    target = tangent(native, axis)
    _, inverse = np.unique(labels, return_inverse=True)
    concepts = torch.from_numpy(inverse).cuda().long()
    sums = target.new_zeros((int(concepts.max()) + 1, target.shape[1])).index_add_(0, concepts, target)

    eeg, means, scales, receipts = [], [], [], []
    for subject in SUBJECTS:
        raw, receipt = physical_eeg(subject, "train", files)
        waves = measured_mean(raw, rows)
        values = waves.float()
        means.append(values.mean(0))
        scales.append(values.std((0, 2)).clamp_min(1e-5)[:, None])
        eeg.append(waves)
        receipts.append(receipt)
    proof = dict(mode=mode, epochs=epochs, batch=BATCH, seed=SEED, sources=sources,
                 d2=d2, split=split, training_eeg=receipts, kernels=list(KERNELS))
    directory = output / "fits"
    directory.mkdir(exist_ok=True)
    destination = directory / f"{mode}.pt"
    if destination.is_file():
        fit = torch.load(destination, map_location="cpu", weights_only=True)
        require(fit["proof"] == proof and len(fit["history"]) == epochs, "Existing fit has different sources")
        save_json(destination.with_suffix(".json"), dict(fit["report"], checkpoint_sha256=files.pin(destination)))
        return
    model = BlurCondition(torch.stack(means), torch.stack(scales), axis).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(.9, .99), weight_decay=.04)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)
    progress = destination.with_suffix(".progress.pt")
    history = []
    if progress.is_file():
        saved = torch.load(progress, map_location="cpu", weights_only=True)
        require(saved["proof"] == proof, "Partial fit uses different training inputs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["schedule"])
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        history = saved["history"]
    for epoch in range(len(history), epochs):
        rng = np.random.default_rng(SEED + epoch)
        diagnostics = torch.zeros(5, device="cuda")
        steps = 0
        for person in rng.permutation(len(SUBJECTS)):
            order = torch.from_numpy(rng.permutation(len(rows))).cuda()
            for batch in order.split(BATCH):
                wave, visual = eeg[person][batch], views[batch]
                person_ids = torch.full_like(batch, int(person))
                q, v = model.encode_eeg(wave, person_ids), model.encode_photo(visual, mode)
                q_native, v_native = model.decode(q), model.decode(v)
                y = target[batch]
                alignment = batch_alignment(q, v)
                recovery = batch_alignment(q_native, y)
                visual_inverse = 1. - (v_native * y).sum(1).mean()
                objects = F.cross_entropy(object_logits(q_native, y, sums, concepts[batch]) / .08, concepts[batch])
                loss = alignment + .50 * recovery + .50 * visual_inverse + .25 * objects
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
                optimizer.step()
                diagnostics += torch.stack((loss, alignment, recovery, visual_inverse, objects)).detach()
                steps += 1
        schedule.step()
        record = dict(zip(("loss", "alignment", "recovery", "visual_inverse", "objects"),
                          (diagnostics / steps).tolist(), strict=True))
        require(all(np.isfinite(v) for v in record.values()), "Fitting produced a nonfinite loss")
        history.append(record)
        print("Blur readout", mode, epoch + 1, "/", epochs, record, flush=True)
        if (epoch + 1) % 4 == 0 and epoch + 1 < epochs:
            files.verify()
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), schedule=schedule.state_dict(),
                         cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), proof=proof, history=history)
            atomic_file(progress, lambda stream: torch.save(state, stream))
    files.verify()
    model.cpu()
    report = dict(
        proof=proof, history=history,
        parameters=sum(param.numel() for param in model.parameters()),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        per_scale_mean_weight=model.blur_weights.detach().softmax(0).mean(1).tolist(),
    )
    state = dict(model=model.state_dict(), proof=proof, history=history, report=report)
    atomic_file(destination, lambda stream: torch.save(state, stream))
    save_json(destination.with_suffix(".json"), dict(report, checkpoint_sha256=files.pin(destination)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("prepare")
    training = commands.add_parser("fit")
    training.add_argument("--mode", choices=MODES, required=True)
    training.add_argument("--epochs", type=int, choices=(1, EPOCHS), default=EPOCHS)
    for command in (commands.choices["prepare"], training):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output)
    else:
        fit(args.mode, args.output, args.epochs)
