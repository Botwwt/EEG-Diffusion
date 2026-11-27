"""Train a shared EEG encoder on paired photographs viewed by different participants."""

import argparse
import hashlib
import inspect
from itertools import combinations
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, atomic_file, render_transaction, require, save_json


SUBJECTS = tuple(range(1, 11))
MODES = ("shared-photo", "shared-object-detail")
RENDER_ARMS = (*MODES, "shared-object-detail-wrong")
STAGED_ARMS = ("scaled-full", "object-only", "object-first", "object-first-wrong-object", "object-first-wrong-photo")
SEED = 20261126
EPOCHS, CONCEPTS_PER_BATCH = 100, 64
REPEATS = [0, 1, 2, 3]
CHANNELS, BINS, SAMPLES_PER_BIN = 17, 40, 5


def matching_batches(members, people, epochs, concepts_per_batch, seed):
    require(members.ndim == 2 and members.shape[1] >= 2 and people >= 2,
            "Two viewers and two photographs per object are required")
    rng = np.random.default_rng(seed)
    pairs = np.asarray(list(combinations(range(people), 2)))
    shuffled_pairs = np.empty(0, dtype=np.int64)
    pos = 0
    for _ in range(epochs):
        concepts = rng.permutation(len(members))
        for start in range(0, len(concepts), concepts_per_batch):
            selected = members[concepts[start:start + concepts_per_batch]]
            first_two = rng.random(selected.shape).argsort(1)[:, :2]
            if pos == len(shuffled_pairs):
                shuffled_pairs = rng.permutation(len(pairs))
                pos = 0
            first, second = pairs[shuffled_pairs[pos]]
            pos += 1
            if rng.integers(2):
                first, second = second, first
            yield int(first), int(second), np.take_along_axis(selected, first_two, axis=1)


def visual_components(fused, members):
    require(members.numel() == len(fused) and torch.equal(members.flatten().sort().values,
            torch.arange(len(fused), device=fused.device)), "Training object groups must partition the photographs")
    center = fused.mean(0)
    object_average = fused[members].mean(1)
    common = torch.empty_like(fused)
    common[members] = (object_average - center)[:, None, :]
    return center, common, fused - center - common


def contrast(a, b, temperature=.1):
    logits = F.normalize(a.float()) @ F.normalize(b.float()).T / temperature
    index = torch.arange(len(a), device=a.device)
    return .5 * (F.cross_entropy(logits, index) + F.cross_entropy(logits.T, index))


def factorized_pair_loss(common, residual):
    a, b = common
    class_loss = .5 * (contrast(a[:, 0], b[:, 1]) + contrast(a[:, 1], b[:, 0]))
    dx, dy = F.normalize((residual[:, :, 0] - residual[:, :, 1]).float(), dim=-1).unbind()
    identity = torch.arange(len(dx), device=dx.device)
    score = dx @ dy.T / .1
    # Both directions also score reversed photograph assignments as negatives.
    detail = .5 * (F.cross_entropy(torch.cat((score, -score), 1), identity)
                   + F.cross_entropy(torch.cat((score.T, -score.T), 1), identity))
    return .5 * (class_loss + detail)


def recording_basis(incidence, differences, scale):
    gram = incidence.T @ incidence
    gram.diagonal().add_(.01 * gram.diagonal().mean())
    bias = torch.linalg.solve(gram, incidence.T @ (differences / scale))
    _, singular, directions = torch.linalg.svd(bias, full_matrices=False)
    return (directions[:3].T * (singular[:3] > 1e-6)).contiguous()


def scale_condition(center, direction, radius):
    return center + radius * F.normalize(direction, dim=-1, eps=1e-5)


class SpatiotemporalEncoder(nn.Module):
    hidden = 768

    def __init__(self, mean, scale, session_basis=None):
        super().__init__()
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_scale", scale)
        self.register_buffer("session_basis", session_basis)
        self.montage = nn.Parameter(torch.eye(CHANNELS).repeat(len(mean), 1, 1))
        width, hidden = 64, self.hidden
        self.temporal = nn.Sequential(nn.Conv1d(CHANNELS, width, 7, padding=3),
                                      nn.GroupNorm(8, width), nn.GELU())
        self.context = nn.Sequential(nn.Conv1d(width, width, 5, padding=4, dilation=2),
                                     nn.GroupNorm(8, width), nn.GELU(),
                                     nn.Conv1d(width, width, 3, padding=1))
        self.pool = nn.Sequential(nn.Flatten(), nn.Linear(width * BINS, hidden),
                                  nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(.15))
        self.mix = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(.15))

    def features(self, eeg, subject):
        eeg = (eeg - self.input_mean[subject]) / self.input_scale[subject]
        if self.session_basis is not None:
            basis = self.session_basis[subject]
            flat = eeg.flatten(1).unsqueeze(1)
            eeg = (flat - (flat @ basis) @ basis.transpose(1, 2)).reshape_as(eeg)
        eeg = torch.bmm(self.montage[subject], eeg)
        local = self.temporal(eeg)
        wide = self.pool(local + self.context(local))
        return wide + self.mix(wide)


class PairedEncoder(SpatiotemporalEncoder):
    def __init__(self, mean, scale, center, head_scale, session_basis=None):
        super().__init__(mean, scale, session_basis)
        self.register_buffer("visual_center", center)
        self.register_buffer("head_scale", head_scale)
        self.head = nn.Sequential(nn.LayerNorm(self.hidden), nn.Linear(self.hidden, 2 * center.numel()))

    def forward(self, eeg, subject):
        common, residual = self.head(self.features(eeg, subject)).float().chunk(2, dim=-1)
        return common * self.head_scale[0], residual * self.head_scale[1]


def prepare_eeg(raw, rows):
    blocks = []
    for start in range(0, len(rows), 128):
        x = torch.from_numpy(raw[rows[start:start + 128]][:, REPEATS, :, :BINS * SAMPLES_PER_BIN].copy())
        x = x.cuda().float().mean(1)
        blocks.append(x.reshape(-1, CHANNELS, BINS, SAMPLES_PER_BIN).mean(-1))
    return torch.cat(blocks)


def fit_recording_basis(raw, rows, sessions, scale):
    ids = torch.from_numpy(sessions[rows].astype(np.int64, copy=False)).cuda()
    require(ids.shape == (len(rows), 4) and ids.min() >= 0 and ids.max() < 4,
            "Invalid training session indices")
    early = ids < 2
    require(torch.all(early.sum(1) == 2), "Each photograph needs two early and two late session observations")
    weights = early.float() - .5
    incidence = (F.one_hot(ids, 4).float() * weights[:, :, None]).sum(1)
    blocks = []
    for start in range(0, len(rows), 128):
        sl = slice(start, start + 128)
        eeg = torch.from_numpy(raw[rows[sl]][:, :, :, :BINS * SAMPLES_PER_BIN].copy()).cuda().float()
        difference = (eeg * weights[sl, :, None, None]).sum(1)
        blocks.append(difference.reshape(-1, CHANNELS, BINS, SAMPLES_PER_BIN).mean(-1).flatten(1))
    return recording_basis(incidence, torch.cat(blocks), scale.flatten())


def losses(common, detail, full, target_common, target_detail, scale, mode):
    target_full = target_common + target_detail
    photo_loss = torch.stack([contrast(v.flatten(0, 1), target_full.flatten(0, 1), .07) for v in full]).mean()
    calibration = ((common - target_common) / scale[0]).square().mean()
    calibration += ((detail - target_detail) / scale[1]).square().mean()
    if mode == "shared-object-detail":
        shared = factorized_pair_loss(common, detail)
    else:
        shared = contrast(full[0].flatten(0, 1), full[1].flatten(0, 1))
    return photo_loss, calibration, shared


def fit(mode, sessions_root=None):
    from common import PREP
    from paper_oof_conditions import registered_rows, study_root
    from paper_render import physical_eeg

    root = study_root()
    root.mkdir(parents=True, exist_ok=True)
    pins = PinnedFiles()
    rows, labels, _ = registered_rows(PREP)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    require(len(counts) == 1323 and np.all(counts == 10), "Expected 1,323 complete training objects")
    members = inverse.argsort(kind="stable").reshape(-1, 10)
    photo_file = PREP / "F_train.npy"
    proof = dict(split=pins.pin(PREP / "paper_split.json"), labels=pins.pin(PREP / "train_concept.npy"),
                 photos=pins.pin(photo_file), code=pins.pin(Path(__file__)), training_eeg={})
    raw_views = []
    session_files = []
    for subject in SUBJECTS:
        raw, receipt = physical_eeg(subject, "train", pins)
        proof["training_eeg"][str(subject)] = receipt
        raw_views.append(raw)
        if sessions_root is not None:
            original = json.loads((PREP / f"sub-{subject:02d}_train.json").read_bytes())
            path = sessions_root / f"sub-{subject:02d}_train.npy"
            digest = pins.pin(path)
            require(original["session_sha256"] == digest, "Unregistered session ordering")
            proof.setdefault("session_indices", {})[str(subject)] = digest
            session_files.append(path)
    dest = root / f"{mode}.pt"
    if dest.is_file():
        state = torch.load(dest, map_location="cpu", weights_only=True)
        require(state["proof"] == proof and state["mode"] == mode, "Existing paired encoder used different data")
        return
    eeg = torch.stack([prepare_eeg(raw, rows) for raw in raw_views])
    require(torch.isfinite(eeg).all(), "Nonfinite training brain responses")
    fused = torch.from_numpy(np.load(photo_file, mmap_mode="r", allow_pickle=False)[rows].copy()).cuda().float()
    require(fused.shape == (len(rows), 1024) and torch.isfinite(fused).all(), "Invalid native visual targets")
    center, obj, detail = visual_components(fused, torch.from_numpy(members).cuda())
    scales = torch.stack((obj.square().mean().sqrt(), detail.square().mean().sqrt())).clamp_min(1e-5)
    scale = eeg.std(1).clamp_min(1e-5)
    nuisance = None
    if sessions_root is not None:
        nuisance = torch.stack([
            fit_recording_basis(raw, rows, np.load(path, allow_pickle=False), std)
            for raw, path, std in zip(raw_views, session_files, scale, strict=True)
        ])
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = PairedEncoder(eeg.mean(1), scale, center, scales, nuisance).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(.9, .98), weight_decay=.05)
    steps = EPOCHS * math.ceil(len(members) / CONCEPTS_PER_BATCH)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-5)
    totals = torch.zeros(3, device="cuda")
    history = []
    plan_hash = hashlib.sha256()
    unique = np.zeros((len(eeg), len(rows)), dtype=bool)
    batches = matching_batches(members, len(SUBJECTS), EPOCHS, CONCEPTS_PER_BATCH, SEED)
    for step, (first, second, chosen) in enumerate(batches, 1):
        unique[[first, second], chosen.ravel()[:, None]] = True
        plan_hash.update(np.asarray([first, second, len(chosen)], dtype="<i8").tobytes() + chosen.astype("<i8").tobytes())
        ids = torch.from_numpy(chosen).cuda()
        people = torch.tensor([first, second], device="cuda")[:, None, None].expand(2, *ids.shape)
        x = eeg[people, ids[None, :, :]].flatten(0, 2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, b = model(x, people.flatten())
        a, b = (value.reshape(2, *ids.shape, -1) for value in (a, b))
        terms = losses(a, b, a + b, obj[ids], detail[ids], scales, mode)
        opt.zero_grad(set_to_none=True)
        (terms[0] + .5 * terms[1] + .25 * terms[2]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        schedule.step()
        totals += torch.stack([value.detach() for value in terms])
        if step % 100 == 0 or step == steps:
            window = step % 100 or 100
            record = dict(step=step, losses=(totals / window).tolist())
            history.append(record)
            totals.zero_()
            print("paired encoder", mode, record, flush=True)
    pins.verify()
    require(step == steps, "Training schedule did not finish")
    metadata = dict(mode=mode, proof=proof, seed=SEED, steps=steps, schedule_sha256=plan_hash.hexdigest(),
                    distinct_viewer_photo_pairs=int(unique.sum()), history=history)
    atomic_file(dest, lambda f: torch.save(dict(metadata, state={k: v.cpu() for k, v in model.state_dict().items()}), f))
    save_json(dest.with_suffix(".json"), metadata)


@torch.inference_mode()
def render(subject, panel, arms):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_oof_conditions import study_root
    from paper_render import donor_rows, draw_images, generator_inputs, physical_eeg, query_seed
    from paper_score import score

    root = study_root()
    pins = PinnedFiles()
    split_path = PREP / "paper_split.json"
    split = pins.pin(split_path)
    registered = json.loads(split_path.read_bytes())
    development = None
    if panel == "dev":
        dev_path = PREP / "paper_development.json"
        development = pins.pin(dev_path)
        plan = json.loads(dev_path.read_bytes())
        require(subject in plan["subjects"], "Use a registered development participant")
        rows = plan["dev_rows"]
    else:
        rows = registered[f"{panel}_rows"]
    require(len(rows) == len(set(rows)) == (500 if panel == "fresh" else 200), "Registration changed")
    raw, physical = physical_eeg(subject, "test" if panel == "official" else "train", pins)
    exchange = same_object_exchange = None
    swap_object = any(arm.endswith(("-wrong", "-wrong-object")) for arm in arms)
    swap_photo = "object-first-wrong-photo" in arms
    if panel != "official":
        require(not np.intersect1d(rows, registered["train_rows"]).size,
                "An evaluation photograph entered encoder training")
        if swap_object or swap_photo:
            label_file = PREP / "train_concept.npy"
            pins.pin(label_file)
            labels = np.load(label_file, allow_pickle=False)[rows]
            if swap_object:
                exchange = donor_rows("token-reliable-wrong", panel, labels)
            if swap_photo:
                same_object_exchange = np.arange(len(rows))
                for label in np.unique(labels):
                    subset = np.flatnonzero(labels == label)
                    same_object_exchange[subset] = np.roll(subset, 1)
    else:
        require(rows == list(range(len(raw))), "Official images must follow the physical EEG order")
        require(not swap_photo, "The within-object exchange needs multiple photos per object")
        if swap_object:
            exchange = np.roll(np.arange(len(rows)), len(rows) // 2)
    eeg = prepare_eeg(raw, rows)
    generator_files = generator_inputs(pins)
    pipe, adapter, projection, prompts, vae = load_generator(torch.float16, False)
    seeds = [query_seed(panel, row) for row in range(len(rows))]
    person = torch.full((len(rows),), subject - 1, device="cuda", dtype=torch.long)
    output = []
    fits = {}
    for arm in arms:
        staged = arm in STAGED_ARMS
        mode = "shared-object-detail" if staged else arm.removesuffix("-wrong")
        fit_path = root / f"{mode}.pt"
        signature = pins.pin(fit_path)
        if mode not in fits:
            fit = torch.load(fit_path, map_location="cuda", weights_only=True)
            require(fit["mode"] == mode and fit["proof"]["split"] == split
                    and tuple(fit["proof"]["training_eeg"]) == tuple(str(s) for s in SUBJECTS),
                    "Encoder checkpoint registration differs")
            state = fit["state"]
            model = PairedEncoder(state["input_mean"], state["input_scale"], state["visual_center"],
                                  state["head_scale"], state.get("session_basis")).cuda().eval()
            model.load_state_dict(state)
            mean, detail = model(eeg, person)
            training_radius = model.head_scale.norm() * math.sqrt(mean.shape[1])
            fits[mode] = model.visual_center, mean, detail, training_radius
            del model, fit, state
        center, common, residual, radius = fits[mode]
        own = np.arange(len(rows))
        donors = exchange if arm.endswith(("-wrong", "-wrong-object")) else own
        features = center + common[donors] + residual[donors]
        detail_donors = donors
        if arm == "object-first-wrong-photo":
            detail_donors = same_object_exchange
        if staged:
            main_features = scale_condition(center, common[donors], radius)
            full_features = scale_condition(center, common[donors] + residual[detail_donors], radius)
            features = main_features if arm == "object-only" else full_features
        def project(feature):
            return torch.cat([projection(adapter(batch.half())).cpu() for batch in feature.split(20)])

        tokens = project(features)
        if arm.startswith("object-first"):
            first_steps = project(main_features)
            tokens = first_steps, first_steps, tokens, tokens
        dest = root / "images" / arm / f"{panel}-r4" / f"sub-{subject:02d}"
        request = dict(render_cache_version=2, arm=arm, subject=subject, n=len(rows), row=rows,
                       budget=f"{panel}-r4", seeds=seeds, repeats=[REPEATS], prompt="", draws=1,
                       unet_calls_per_draw=4, eeg_donor=donors.tolist(), split_sha256=split,
                       development_split_sha256=development, physical_eeg=physical, checkpoint_sha256=signature,
                       generator_files_sha256=generator_files, code_sha256=pins.pin(Path(__file__)))
        if staged:
            request.update(object_eeg_donor=donors.tolist(), residual_eeg_donor=detail_donors.tolist(),
                           training_radius=float(radius), denoising_conditions=(
                               ["object"] * 2 + ["object-and-photo"] * 2 if arm.startswith("object-first")
                               else (["object"] if arm == "object-only" else ["object-and-photo"]) * 4),
                           render_code_sha256=pins.pin(Path(inspect.getfile(draw_images))))

        def generate(stage, starts, complete):
            draw_images(stage, pipe, vae, prompts, tokens, seeds,
                        batch_starts=starts, batch_complete=complete)

        render_transaction(dest, request, generate, pins.verify)
        print("paired encoder images", subject, panel, arm, len(rows), flush=True)
        output.append(dest)
    del pipe, vae, adapter, projection
    torch.cuda.empty_cache()
    for dest in output:
        if not (dest / "scores.json").is_file():
            score(dest, panel, Path(os.environ["ARCHIVE"]), dest)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="action", required=True)
    train_cli = sub.add_parser("fit")
    train_cli.add_argument("mode", choices=MODES)
    train_cli.add_argument("--sessions-root", type=Path)
    render_cli = sub.add_parser("render")
    render_cli.add_argument("--subject", type=int, choices=SUBJECTS, required=True)
    render_cli.add_argument("--panel", choices=("dev", "fresh", "official"), default="dev")
    render_cli.add_argument("--arms", nargs="+", choices=(*RENDER_ARMS, *STAGED_ARMS), default=RENDER_ARMS)
    args = cli.parse_args()
    if args.action == "fit":
        fit(args.mode, args.sessions_root)
    else:
        render(args.subject, args.panel, args.arms)
