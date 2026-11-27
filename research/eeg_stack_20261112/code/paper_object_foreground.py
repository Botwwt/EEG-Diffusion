"""Learn to recognize the same object across training EEG from different photographs."""
import argparse
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_background_pairs import photo_pairs


MODES = ("uniform", "background")
SEED = 20261201
EPOCHS = 12
OBJECTS_PER_BATCH = 96
TEMPERATURE = .075


def closest_other_context(context, labels, *, use_context):
    if use_context:
        cost = torch.cdist(context, context)
    else:
        order = torch.arange(len(labels), device=labels.device)
        cost = ((order[None] - order[:, None] - len(labels) // 2) % len(labels)).float()
    return cost.masked_fill(labels[:, None] == labels, torch.inf).argmin(1)


def assignment_loss(predicted, target, donor):
    pred = F.normalize(predicted, dim=-1)
    truth = F.normalize(target, dim=-1)
    matched = (pred * truth).sum(-1)
    swapped = (pred * truth[donor] + pred[donor] * truth).sum(-1)
    return F.softplus((swapped - matched - matched[donor]) / (2 * TEMPERATURE)).mean()


class ObjectEncoder(nn.Module):
    def __init__(self, backbone, dimensions):
        super().__init__()
        self.backbone = backbone
        self.semantic = nn.Linear(1024, dimensions)

    def forward(self, voltage):
        return self.semantic(self.backbone(voltage))

    @torch.no_grad()
    def initialize_from_photo(self, fit, target_scale):
        direction = fit["weights"] / fit["scale"][:, None] * fit["gain"] / target_scale
        self.semantic.weight.copy_(direction.T)
        self.semantic.bias.copy_(-fit["center"] @ direction)


def root():
    return Path(os.environ["EEG_CROSSFIT_ROOT"])


def text_targets(files):
    from common import ROOT
    from paper_render_cache import require
    from paper_soft_text import registry

    rows, labels, names, registration = registry()
    path = ROOT / "repeat_semantic_rank/predictable-text-space.pt"
    digest = files.pin(path)
    packet = torch.load(path, map_location="cpu", weights_only=True)
    target = packet["final_codes"].float()
    require(packet["meta"]["inputs"] == registration and target.shape == (1323, 160),
            "Unexpected object coordinate space or training photo split")
    return rows, labels, names, target, packet["states"], digest


def predict_at_budget(encoder, raw, repeats):
    selected = sum(repeats, [])
    if len(selected) not in (4, 80) or len(set(selected)) != len(selected):
        raise ValueError("Each readout needs four or eighty distinct physical EEG trials")
    results = []
    for start in range(0, len(raw), 20):
        batch = torch.from_numpy(np.array(raw[start:start + 20], copy=True)).cuda().float()
        results.append(encoder(batch[:, selected].mean(1)))
    return torch.cat(results)


@torch.inference_mode()
def load_encoder(subject, mode, basis, native, files):
    from paper_instance_residual import make_encoder
    from paper_render_cache import require

    path = root() / "fits" / f"sub-{subject:02d}-{mode}.pt"
    digest = files.pin(path)
    packet = torch.load(path, map_location="cpu", weights_only=True)
    require(packet["subject"] == subject and packet["mode"] == mode
            and packet["proof"]["basis"] == basis and packet["proof"]["native"] == native,
            "Object encoder does not use the same participant, basis or native checkpoint")
    model = ObjectEncoder(make_encoder("cuda").backbone, 160).cuda().eval()
    model.load_state_dict(packet["state"])
    return model, packet["target_scale"], digest


def train(subject, modes):
    from common import PREP
    from paper_oof_conditions import fitted_native
    from paper_render import physical_eeg
    from paper_render_cache import PinnedFiles, atomic_file, require, save_json
    from paper_repeat_semantic_bridge import fit_head

    torch.set_float32_matmul_precision("high")
    files = PinnedFiles()
    rows, labels, names, target, _, basis = text_targets(files)
    raw, eeg_proof = physical_eeg(subject, "train", files)
    initial, native = fitted_native(subject, "cuda")
    initial = {k: v.cpu() for k, v in initial.backbone.state_dict().items()}
    context_path = root() / "training-borders.pt"
    photograph = PREP / "F_train.npy"
    proof = dict(native=files.pin(native), basis=basis, eeg=eeg_proof, photograph=files.pin(photograph),
                 border=files.pin(context_path), code=files.pin(Path(__file__)),
                 pair_code=files.pin(Path(__file__).with_name("paper_background_pairs.py")),
                 split=files.pin(PREP / "paper_split.json"))
    context = torch.load(context_path, map_location="cpu", weights_only=True)
    require(np.array_equal(context["rows"].numpy(), rows) and context["proof"]["split"] == proof["split"],
            "Border features and EEG training photos follow different registrations")
    border = context["border"].numpy()
    require(border.shape == (len(rows), 72) and torch.isfinite(context["border"]).all(),
            "Invalid color and texture statistics")
    members = labels.argsort(stable=True).numpy().reshape(-1, 10)
    require(np.array_equal(labels[members].numpy(), np.broadcast_to(np.arange(len(target))[:, None], members.shape)),
            "Expected ten rows from each object")
    _, unique, class_ids = np.unique(names, return_index=True, return_inverse=True)
    class_ids = torch.from_numpy(class_ids).cuda()
    target = target.cuda()
    labels = labels.cuda()
    scale = target.square().mean().sqrt().item()
    require(scale > 0., "Zero-scale text coordinates")
    targets = target / scale
    by_name = F.normalize(targets[unique])
    photos = torch.from_numpy(np.load(photograph, mmap_mode="r", allow_pickle=False)[rows].copy()).float().cuda()
    photo_head = fit_head(photos, target, labels)
    del photos
    sensor = torch.empty(len(rows), 17, 250, device="cuda")
    for sl in range(0, len(rows), 128):
        section = slice(sl, sl + 128)
        sensor[section] = torch.from_numpy(raw[rows[section]].copy()).float().cuda().mean(1)
    require(torch.isfinite(sensor).all(), "Non-finite physical training EEG")
    edge = torch.from_numpy(border).cuda()
    n_steps = EPOCHS * members.shape[1] // 2 * math.ceil(len(members) / OBJECTS_PER_BATCH)
    destination = root() / "fits"
    destination.mkdir(parents=True, exist_ok=True)
    from paper_instance_residual import make_encoder

    for mode in modes:
        path = destination / f"sub-{subject:02d}-{mode}.pt"
        if path.exists():
            frozen = torch.load(path, map_location="cpu", weights_only=True)
            require(frozen["proof"] == proof and frozen["subject"] == subject and frozen["mode"] == mode,
                    "Existing trained encoder is tied to different inputs")
            continue
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = ObjectEncoder(make_encoder("cuda").backbone, 160).cuda().train()
        model.backbone.load_state_dict(initial)
        model.initialize_from_photo(photo_head, scale)
        optimize = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": 8e-5},
                                      {"params": model.semantic.parameters(), "lr": 3e-4}],
                                     betas=(.9, .995), weight_decay=.05)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimize, T_max=n_steps, eta_min=5e-6)
        order = np.random.default_rng(SEED)
        losses, pairing_digest, order_digest = [], hashlib.sha256(), hashlib.sha256()
        for epoch in range(EPOCHS):
            pairing = photo_pairs(members, border, SEED + epoch, distant=mode == "background")
            pairing_digest.update(pairing.astype("<i8", copy=False).tobytes())
            gap = float(((border[pairing[..., 0]] - border[pairing[..., 1]]) ** 2).mean())
            epoch_stats = []
            for slot in range(pairing.shape[1]):
                classes = order.permutation(len(pairing))
                order_digest.update(classes.astype("<i8", copy=False).tobytes())
                for start in range(0, len(classes), OBJECTS_PER_BATCH):
                    idx = classes[start:start + OBJECTS_PER_BATCH]
                    a, b = torch.from_numpy(pairing[idx, slot].T.copy()).cuda()
                    c = labels[a]
                    reference = targets[c]
                    matched = class_ids[c]
                    first, second = sensor[a], sensor[b]
                    one, two, mixed = model(torch.cat((first, second, (first + second) * .5))).chunk(3)
                    logits = F.normalize(torch.cat((one, two, mixed)), dim=-1) @ by_name.T / TEMPERATURE
                    recognition = F.cross_entropy(logits, matched.repeat(3))
                    coordinate_loss = F.mse_loss(torch.stack((one, two, mixed)), reference.expand(3, -1, -1))
                    cross_photo = (1. - F.cosine_similarity(one, two)).mean()
                    adversary = closest_other_context(edge[a], matched, use_context=mode == "background")
                    match = assignment_loss(one, reference, adversary)
                    objective = recognition + .30 * coordinate_loss + .15 * cross_photo + .35 * match
                    optimize.zero_grad(set_to_none=True)
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimize.step()
                    schedule.step()
                    epoch_stats.append([float(value.detach()) for value in (recognition, coordinate_loss, cross_photo, match)])
            losses.append(dict(epoch=epoch + 1, objective=np.mean(epoch_stats, axis=0).tolist(),
                               updates=len(epoch_stats), squared_background_gap=gap))
            print("Object EEG", subject, mode, losses[-1], flush=True)
        files.verify()
        packet = dict(subject=subject, mode=mode, proof=proof, target_scale=scale,
                      state={k: v.cpu() for k, v in model.state_dict().items()})
        atomic_file(path, lambda f: torch.save(packet, f))
        save_json(path.with_suffix(".json"), dict(subject=subject, mode=mode, proof=proof,
                  checkpoint_sha256=files.pin(path), pairing_sha256=pairing_digest.hexdigest(),
                  batch_order_sha256=order_digest.hexdigest(), epochs=losses))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("train",))
    cli.add_argument("subject", type=int, choices=range(1, 11))
    cli.add_argument("--modes", choices=MODES, nargs="+", default=MODES)
    args = cli.parse_args()
    train(args.subject, args.modes)
