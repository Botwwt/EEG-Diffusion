"""Fine-tune CBraMod with joint or separate attention over four observed EEG repeats."""
import argparse
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, atomic_file, require, save_json


FIT_ARMS = ("joint", "separate", "random-joint")
IMAGE_ARMS = ("native-paper", *FIT_ARMS, "joint-wrong")
SOURCE_FILES = {
    "models/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "models/cbramod.py": "57019e52336137c7dee253ec38663f78905553370af596a125fe4aae142c3357",
    "models/criss_cross_transformer.py": "5eb3c4c93c1b530a8cbe3ad3d0e217c63d9bd6419553fdfd8eae46223df7950a",
    "pretrained_weights/pretrained_weights.pth": "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178",
}
WEIGHTS = "pretrained_weights/pretrained_weights.pth"
SEED, EPOCHS, BATCH = 20261127, 8, 128
REPEATS = [0, 1, 2, 3]
SEMANTIC_WEIGHT = .25


def to_200hz(raw):
    from scipy.signal import resample_poly

    require(raw.ndim == 4 and raw.shape[1:] == (4, 17, 250), "Expected four one-second 250 Hz EEG presentations")
    return resample_poly(raw.astype(np.float32, copy=False), 4, 5, axis=-1, padtype="line")


def measured_trials(raw, rows, device="cuda"):
    blocks = [torch.from_numpy(to_200hz(raw[rows[i:i + 128]][:, REPEATS])).to(device)
              for i in range(0, len(rows), 128)]
    trials = torch.cat(blocks)
    require(trials.shape == (len(rows), 4, 17, 200) and torch.isfinite(trials).all(),
            "Resampled measurements differ from the registered sequence")
    return trials


def signal_scale(trials):
    center = trials.mean((0, 1, 3), keepdim=True)
    rms = (trials - center).square().mean(-1).sqrt().flatten(0, 1)
    scale = (rms.median(0).values.clamp_min(.01) / .2).reshape(1, 1, trials.shape[2], 1)
    return center, scale


def encode_repeats(backbone, trials, representation):
    """Keep patch extraction per trial and allow repeat interaction only in the encoder."""
    require(representation in ("joint", "separate"), "Invalid repeat attention control")
    n, r, _, _ = trials.shape
    flat = backbone.patch_embedding(trials.flatten(0, 1).unsqueeze(2))
    if representation == "separate":
        contextual = backbone.encoder(flat).squeeze(2).unflatten(0, (n, r)).mean(1)
    else:
        packed = flat.squeeze(2).unflatten(0, (n, r)).transpose(1, 2)
        contextual = backbone.encoder(packed).mean(2)
    return contextual


def upstream_model(root, pretrained, files):
    proof = {name: files.pin(root / name) for name in SOURCE_FILES}
    require(proof == SOURCE_FILES, "Released CBraMod source or weights differ from the pinned bytes")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    code = importlib.import_module("models.cbramod")
    require(Path(code.__file__).resolve() == (root / "models/cbramod.py").resolve(), "Unexpected CBraMod import")
    model = code.CBraMod()
    if pretrained:
        parameters = torch.load(root / WEIGHTS, map_location="cpu", weights_only=True)
        model.load_state_dict(parameters, strict=True)
    model.proj_out = nn.Identity()
    return model


class PhotoCondition(nn.Module):
    def __init__(self, backbone, arm, input_center, input_scale, photo_center, photo_scale):
        super().__init__()
        require(arm in FIT_ARMS, "Invalid CBraMod fit arm")
        self.backbone = backbone
        self.representation = "separate" if arm == "separate" else "joint"
        self.register_buffer("input_center", input_center)
        self.register_buffer("input_scale", input_scale)
        self.register_buffer("photo_center", photo_center)
        self.register_buffer("photo_scale", photo_scale)
        self.head = nn.Sequential(nn.LayerNorm(17 * 200), nn.Linear(17 * 200, 512), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(512, len(photo_center)))

    def forward(self, trials):
        signal = (trials - self.input_center) / self.input_scale
        state = encode_repeats(self.backbone, signal, self.representation)
        return self.photo_center + self.photo_scale * self.head(state.flatten(1)).float()


def model_dir(root, subject, arm):
    return root / "fit" / f"sub-{subject:02d}" / arm


def fit(subject, upstream, source, arm):
    from common import PREP
    from paper_object_margin_encoder import visual_transfer
    from paper_object_tokens import epoch_batches, group_index
    from paper_oof_conditions import registered_rows, study_root
    from paper_render import physical_eeg
    from paper_repeat_semantics import photo_loss

    files = PinnedFiles()
    rows, labels, _ = registered_rows(PREP)
    raw, receipt = physical_eeg(subject, "train", files)
    photo_path = PREP / "F_train.npy"
    dino_path = source / "object-identity/training-photo-dino.pt"
    split_sha = files.pin(PREP / "paper_split.json")
    subjects_seed = SEED + subject
    proof = dict(physical_eeg=receipt, split_sha256=split_sha, photo_sha256=files.pin(photo_path),
                 dino_sha256=files.pin(dino_path), code_sha256=files.pin(Path(__file__)),
                 photo_loss_sha256=files.pin(Path(inspect.getfile(photo_loss))),
                 batch_code_sha256=files.pin(Path(inspect.getfile(epoch_batches))),
                 visual_map_sha256=files.pin(Path(inspect.getfile(visual_transfer))),
                 seed=subjects_seed, epochs=EPOCHS, batch_size=BATCH, semantic_weight=SEMANTIC_WEIGHT,
                 rows=len(rows), upstream={k: files.pin(upstream / k) for k in SOURCE_FILES})
    require(proof["upstream"] == SOURCE_FILES, "Wrong upstream CBraMod")
    root = model_dir(study_root(), subject, arm)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "model.pt"
    if checkpoint.is_file():
        old = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(old["proof"] == proof and old["arm"] == arm and old["subject"] == subject, "Different CBraMod fit exists")
        return
    target = torch.load(dino_path, map_location="cpu", weights_only=True)
    require(np.array_equal(target["rows"].numpy(), rows) and target["proof"]["split"] == split_sha
            and np.array_equal(target["concepts"][target["labels"]].numpy(), labels),
            "DINO training features must describe the same photo rows")
    photos = torch.from_numpy(np.load(photo_path, mmap_mode="r", allow_pickle=False)[rows].copy()).cuda()
    images = target["image_features"].float().cuda()
    photo_center, dino_center, mapping = visual_transfer(photos, images)
    leave_one_out = target["sums"][target["labels"]].cuda() - images
    semantic = F.normalize(F.normalize(leave_one_out) - dino_center)
    variance = (photos - photo_center).square().mean()
    trials = measured_trials(raw, rows)
    center, scale = signal_scale(trials)
    torch.manual_seed(subjects_seed)
    sampler = torch.Generator(device="cpu").manual_seed(subjects_seed)
    _, groups = group_index(torch.from_numpy(labels))
    trunk = upstream_model(upstream, arm != "random-joint", files)
    network = PhotoCondition(trunk, arm, center, scale, photo_center, variance.sqrt()).cuda().train()
    parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([
        {"params": network.backbone.parameters(), "lr": .00015},
        {"params": network.head.parameters(), "lr": .0003},
    ], betas=(.9, .995), weight_decay=.05)
    steps = EPOCHS * math.ceil(len(groups) / BATCH) * 10
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=.000015)
    order = hashlib.sha256()
    history = []
    for epoch in range(EPOCHS):
        epoch_loss = torch.zeros(2, device="cuda")
        for batch in epoch_batches(groups, sampler, batch_size=BATCH):
            order.update(batch.numpy().astype("<i8").tobytes())
            index = batch.cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = network(trials[index])
                base = photo_loss(predicted, photos[index], photo_center, variance)
                vision = F.normalize((predicted - photo_center) @ mapping)
                score = vision @ semantic[index].T / .07
                correct = torch.arange(len(index), device="cuda")
                auxiliary = .5 * (F.cross_entropy(score.float(), correct) + F.cross_entropy(score.T.float(), correct))
                loss = base + SEMANTIC_WEIGHT * auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            epoch_loss += len(index) * torch.stack((base.detach(), auxiliary.detach()))
        summary = (epoch_loss / len(rows)).tolist()
        require(np.isfinite(summary).all(), "Nonfinite CBraMod training")
        history.append(summary)
        print("CBraMod fit", subject, arm, epoch + 1, "photo and semantic loss", summary, flush=True)
    files.verify()
    state = {key: value.detach().cpu() for key, value in network.state_dict().items()}
    details = dict(subject=subject, arm=arm, proof=proof, parameters=parameters, steps=steps,
                   batch_order_sha256=order.hexdigest(), history=history)
    atomic_file(checkpoint, lambda stream: torch.save(dict(details, state=state), stream))
    save_json(root / "fit.json", dict(details, checkpoint_sha256=files.pin(checkpoint)))


@torch.inference_mode()
def render(subject, upstream, arms):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_oof_conditions import study_root
    from paper_render import condition, draw_images, generator_inputs, groups, physical_eeg, query_seed
    from paper_token_evidence import project_tokens, render_arms, score_outputs
    from paper_visual_ceiling import official_features

    root, files = study_root(), PinnedFiles()
    split = PREP / "paper_split.json"
    split_sha = files.pin(split)
    rows = json.loads(split.read_bytes())["official_rows"]
    require(rows == list(range(200)), "Only the official 200-stimulus panel is registered")
    eeg, physical = physical_eeg(subject, "test", files)
    samples = measured_trials(eeg, rows)
    repeats = groups("official-r4")
    require(repeats == [REPEATS], "All conditions must use the same measured repeats")
    native, native_sha = condition("native-paper", subject, eeg[rows], repeats, features_only=True, files=files)
    photo_sha = files.pin(PREP / "F_train.npy")
    expected_arms = {arm.removesuffix("-wrong") for arm in arms if arm != "native-paper"}
    code_sha = files.pin(Path(__file__))
    conditions, fits, diagnostics = {"native-paper": native.cuda()}, {}, {}
    normalizer = None
    for arm in sorted(expected_arms):
        path = model_dir(root, subject, arm) / "model.pt"
        fit_sha = files.pin(path)
        pack = torch.load(path, map_location="cpu", weights_only=True)
        proof = pack["proof"]
        require((pack["subject"], pack["arm"], proof["split_sha256"], proof["photo_sha256"], proof["code_sha256"])
                == (subject, arm, split_sha, photo_sha, code_sha), "CBraMod checkpoint uses different evidence")
        backbone = upstream_model(upstream, False, files)
        weights = pack["state"]
        args = [weights[name] for name in ("input_center", "input_scale", "photo_center", "photo_scale")]
        model = PhotoCondition(backbone, arm, *args)
        model.load_state_dict(weights, strict=True)
        model.cuda().eval().requires_grad_(False)
        normalizer = model.photo_center
        conditions[arm] = torch.cat([model(batch) for batch in samples.split(32)])
        require(torch.isfinite(conditions[arm]).all(), "CBraMod conditions are nonfinite")
        if arm == "joint":
            permutation = model(samples[:16, [2, 0, 3, 1]])
            diagnostics["joint_repeat_permutation_max_abs"] = float((conditions[arm][:16] - permutation).abs().max())
        fits[arm] = dict(model_sha256=fit_sha, training_physical_eeg=proof["physical_eeg"],
                         dino_training_sha256=proof["dino_sha256"], batch_order_sha256=pack["batch_order_sha256"],
                         steps=pack["steps"], parameters=pack["parameters"])
        del model, backbone
    order = np.arange(len(rows))
    swapped = np.roll(order, len(order) // 2)
    generator = generator_inputs(files)
    pipeline = load_generator(torch.float16, False)
    _, adapter, project, _, vae = pipeline
    variants = []
    for arm in arms:
        key = arm.removesuffix("-wrong")
        donor = swapped if arm.endswith("-wrong") else order
        tokens = project_tokens(conditions[key][donor], adapter, project).cpu().half()
        variants.append((arm, tokens, dict(eeg_donor=donor.tolist(), fit=fits.get(key))))
    shared = dict(subject=subject, budget="official-r4", repeats=repeats, physical_eeg=physical, split_sha256=split_sha,
                  native_checkpoint_sha256=native_sha, generator_files_sha256=generator, code_sha256=code_sha,
                  renderer_sha256=files.pin(Path(inspect.getfile(draw_images))), upstream_files_sha256=SOURCE_FILES,
                  diagnostic_only=False)
    seeds = [query_seed("official", i) for i in order]
    outputs = render_arms(root / "images" / f"sub-{subject:02d}", variants, rows, seeds, shared, files, pipeline)
    del pipeline, adapter, project, vae
    torch.cuda.empty_cache()
    score_outputs(outputs, "official")
    ref, held_rows, gallery = official_features(PREP, files)
    require(held_rows == rows, "The offline scoring gallery rows differ")
    center = normalizer if normalizer is not None else ref.cuda().mean(0)
    ref = F.normalize(ref.cuda() - center)
    diagnostics["feature_match"] = {}
    for arm, feat in conditions.items():
        sim = F.normalize(feat - center) @ ref.T
        hit = (sim.argmax(1) == torch.arange(len(ref), device="cuda")).sum()
        diagnostics["feature_match"][arm] = dict(hits=int(hit), mean_target_cosine=float(sim.diagonal().mean()))
    files.verify()
    save_json(root / f"pre-gen-sub-{subject:02d}.json", dict(diagnostics, photos=gallery))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    training = commands.add_parser("fit")
    generation = commands.add_parser("render")
    for command in (training, generation):
        command.add_argument("subject", type=int, choices=range(1, 11))
        command.add_argument("--upstream", type=Path, required=True)
    training.add_argument("--source-root", type=Path, required=True)
    training.add_argument("--arm", choices=FIT_ARMS, required=True)
    generation.add_argument("--arms", nargs="+", default=IMAGE_ARMS, choices=IMAGE_ARMS)
    args = parser.parse_args()
    if args.action == "fit":
        fit(args.subject, args.upstream, args.source_root, args.arm)
    else:
        render(args.subject, args.upstream, args.arms)
