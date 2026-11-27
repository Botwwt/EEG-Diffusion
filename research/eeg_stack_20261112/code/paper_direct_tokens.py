"""Direct supervision of Brain-HiVE's relative image tokens on physical four-repeat EEG."""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_paired_encoder import REPEATS, SUBJECTS, SpatiotemporalEncoder, prepare_eeg
from paper_render_cache import PinnedFiles, atomic_file, require, save_json
from paper_token_evidence import project_tokens, ridge_weights, token_parts

SEED = 20261127
RANK = 512
EPOCHS, BATCH, VIEWERS = 60, 64, 4
FIT_ARMS = ("uniform-relative", "weighted-relative")
ARMS = ("native-paper", *FIT_ARMS, "weighted-relative-wrong")


class RelativeTokenEncoder(SpatiotemporalEncoder):
    def __init__(self, mean, scale, dimensions):
        super().__init__(mean, scale)
        self.head = nn.Sequential(nn.LayerNorm(self.hidden), nn.Linear(self.hidden, dimensions))

    def forward(self, eeg, subject):
        return self.head(self.features(eeg, subject)).float()


def visual_basis(relative, dimensions):
    center = relative.mean(0)
    flat = (relative - center).flatten(1)
    q = min(dimensions + 24, *flat.shape)
    require(0 < dimensions <= q, "Invalid visual basis rank")
    devices = [relative.device] if relative.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(SEED)
        _, singular, basis = torch.pca_lowrank(flat, q=q, center=False, niter=5)
    axes = basis[:, :dimensions].contiguous()
    require(singular[dimensions - 1] > 1e-5, "Visual targets do not have the registered rank")
    scores = flat @ axes
    scale = scores.square().mean(0).sqrt().clamp_min(1e-5)
    return dict(center=center, axes=axes, scale=scale,
                radius=scores.square().sum(1).mean().sqrt(),
                variance_kept=scores.square().sum() / flat.square().sum()), scores / scale


def restore_relative(coordinates, basis, *, calibrate=False):
    measured = coordinates * basis["scale"]
    if calibrate:
        measured = F.normalize(measured, eps=1e-5, dim=-1) * basis["radius"]
    result = basis["center"] + (measured @ basis["axes"].T).reshape(-1, *basis["center"].shape)
    return token_parts(result)[1]


def coordinate_weights(recovery, object_variance):
    evidence = recovery.clamp_min(0) * object_variance.clamp(0, 1).sqrt()
    strength = evidence / evidence.mean().clamp_min(1e-8)
    weights = .5 + .5 * strength.clamp_max(4)
    return weights / weights.mean()


def held_object_recovery(eeg, coordinates, folds):
    correlations = []
    for viewer in eeg.flatten(2):
        prediction = torch.empty_like(coordinates)
        for fold in folds.unique():
            held = folds == fold
            train = viewer[~held]
            mean = train.mean(0)
            scale = train.std(0).clamp_min(1e-5)
            target = coordinates[~held]
            center = target.mean(0)
            coefficient = ridge_weights((train - mean) / scale, target - center, penalty=.15)
            prediction[held] = (viewer[held] - mean) / scale @ coefficient + center
        correlations.append((F.normalize(prediction - prediction.mean(0), dim=0) *
                             F.normalize(coordinates - coordinates.mean(0), dim=0)).sum(0))
    return torch.stack(correlations)


def photo_batches(members, people, epochs, concepts_per_batch, seed):
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        if epoch % members.shape[1] == 0:
            photos = np.take_along_axis(members, rng.random(members.shape).argsort(axis=1), axis=1)
        selected = photos[:, epoch % members.shape[1]][rng.permutation(len(members))]
        for start in range(0, len(members), concepts_per_batch):
            viewers = rng.choice(people, size=VIEWERS, replace=False)
            yield viewers, selected[start:start + concepts_per_batch]


def coefficient_loss(prediction, target, weights):
    root = weights.sqrt()
    scored_prediction = prediction.flatten(0, 1) * root
    scored_target = target * root
    logits = F.normalize(scored_prediction) @ F.normalize(scored_target).T / .07
    correct = torch.arange(len(target), device=target.device).repeat(len(prediction))
    squared = ((prediction - target).square() * weights).mean()
    return F.cross_entropy(logits, correct) + .25 * squared


def training_inputs(prep, pins):
    from paper_oof_conditions import registered_rows
    from paper_render import physical_eeg

    split, labels, photo, available = [prep / name for name in
                                      ("paper_split.json", "train_concept.npy", "F_train.npy", "lat512_train_ok.npy")]
    sources = dict(split=pins.pin(split), photos=pins.pin(photo), labels=pins.pin(labels),
                   available=pins.pin(available), eeg={}, code={})
    for function in (training_inputs, registered_rows, prepare_eeg, project_tokens, physical_eeg, ridge_weights):
        path = Path(inspect.getfile(function))
        sources["code"][path.name] = pins.pin(path)
    rows, object_ids, folds = registered_rows(prep)
    _, inverse, counts = np.unique(object_ids, return_inverse=True, return_counts=True)
    require(len(counts) == 1323 and np.all(counts == 10), "Expected ten photographs of each training object")
    members = inverse.argsort(kind="stable").reshape(-1, 10)
    require(np.all(folds[members] == folds[members[:, :1]]) and set(folds) == {1, 2, 3, 4},
            "Every internal fold must hold out entire training objects")
    eeg = []
    for subject in SUBJECTS:
        raw, sources["eeg"][str(subject)] = physical_eeg(subject, "train", pins)
        eeg.append(prepare_eeg(raw, rows))
    eeg = torch.stack(eeg)
    require(torch.isfinite(eeg).all(), "Nonfinite training EEG")
    return eeg, rows, members, folds, sources


@torch.inference_mode()
def prepare():
    from common import PREP, PRIOR
    from main.models_adapter import FusionEncoderModel, IPProjectionModel
    from paper_oof_conditions import study_root

    root = study_root()
    root.mkdir(parents=True, exist_ok=True)
    pins = PinnedFiles()
    eeg, rows, members, folds, sources = training_inputs(PREP, pins)
    paths = [p for name in ("fusion_encoder", "proj") for p in sorted((PRIOR / name).rglob("*"))
             if p.is_file() and not any(part.startswith(".") for part in p.relative_to(PRIOR).parts)]
    sources["projection"] = {str(p.relative_to(PRIOR)): pins.pin(p) for p in paths}
    for cls in (FusionEncoderModel, IPProjectionModel):
        path = Path(inspect.getfile(cls))
        sources["code"][path.name] = pins.pin(path)
    target_file = root / "training-targets.pt"
    if target_file.is_file():
        saved = torch.load(target_file, map_location="cpu", weights_only=True)
        require(saved["sources"] == sources and saved["rank"] == RANK, "Existing visual target bank used different inputs")
        return
    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.cuda().half().eval()
    projection = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").cuda().half().eval()
    path = PREP / "F_train.npy"
    visual = torch.from_numpy(np.load(path, mmap_mode="r", allow_pickle=False)[rows].copy())
    _, relative = token_parts(project_tokens(visual, adapter, projection))
    require(relative.ndim == 3 and relative.shape[1] == 4 and torch.isfinite(relative).all(),
            "Invalid native four-token visual bank")
    basis, coordinates = visual_basis(relative, RANK)
    members = torch.from_numpy(members).cuda()
    object_variance = coordinates[members].mean(1).var(0) / coordinates.var(0).clamp_min(1e-6)
    recovery = held_object_recovery(eeg, coordinates, torch.from_numpy(folds).cuda())
    weight = coordinate_weights(recovery.mean(0), object_variance)
    require(all(torch.isfinite(x).all() for x in (coordinates, recovery, object_variance, weight)),
            "Nonfinite visual coefficients or evidence weights")
    package = dict(sources=sources, rank=RANK, rows=torch.from_numpy(rows),
                   basis={name: t.cpu() for name, t in basis.items()}, coordinates=coordinates.cpu(),
                   weights=weight.cpu(), recovery=recovery.cpu(), object_variance=object_variance.cpu())
    pins.verify()
    atomic_file(target_file, lambda f: torch.save(package, f))
    stats = dict(rows=len(rows), dimensions=RANK, photo_variance_retained=basis["variance_kept"].item(),
                 recovery_mean=recovery.mean(1).tolist(),
                 weight_quantiles=weight.quantile(torch.tensor([0., .25, .5, .75, 1.], device="cuda")).tolist())
    save_json(root / "training-targets.json", stats)
    print(json.dumps(stats), flush=True)


def fit(arm):
    from common import PREP
    from paper_oof_conditions import study_root

    root, pins = study_root(), PinnedFiles()
    target_file = root / "training-targets.pt"
    target_digest = pins.pin(target_file)
    package = torch.load(target_file, map_location="cuda", weights_only=True)
    eeg, rows, members, _, sources = training_inputs(PREP, pins)
    expected = package["sources"]
    require(all(expected[k] == v if k != "code" else all(expected[k][n] == d for n, d in v.items())
                for k, v in sources.items())
            and np.array_equal(rows, package["rows"].cpu().numpy()) and package["rank"] == RANK,
            "EEG and exact-photo supervision have different registered training data")
    model_file = root / f"{arm}.pt"
    if model_file.is_file():
        state = torch.load(model_file, map_location="cpu", weights_only=True)
        require(state["target_sha256"] == target_digest and state["arm"] == arm, "Prior fit differs")
        return
    torch.manual_seed(SEED)
    model = RelativeTokenEncoder(eeg.mean(1), eeg.std(1).clamp_min(1e-5), RANK).cuda().train()
    fit_weights = package["weights"] if arm == "weighted-relative" else torch.ones_like(package["weights"])
    coordinates = package["coordinates"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(.9, .98), weight_decay=.05)
    steps = EPOCHS * math.ceil(len(members) / BATCH)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=2e-5)
    batches = photo_batches(members, len(SUBJECTS), EPOCHS, BATCH, SEED)
    plan = hashlib.sha256()
    history = []
    total = torch.zeros((), device="cuda")
    for step, (viewers, selected) in enumerate(batches, 1):
        plan.update(np.asarray([len(selected), *viewers, *selected], dtype="<i8").tobytes())
        ids = torch.from_numpy(selected).cuda()
        people = torch.from_numpy(viewers).cuda()[:, None].expand(-1, len(ids))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(eeg[people, ids].flatten(0, 1), people.flatten()).reshape(VIEWERS, len(ids), RANK)
        loss = coefficient_loss(prediction, coordinates[ids], fit_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        total += loss.detach()
        if step % 100 == 0 or step == steps:
            entry = dict(step=step, loss=(total / (step % 100 or 100)).item())
            history.append(entry)
            total.zero_()
            print(arm, entry, flush=True)
    receipt = dict(arm=arm, target_sha256=target_digest, plan_sha256=plan.hexdigest(),
                   steps=steps, history=history)
    pins.verify()
    atomic_file(model_file, lambda f: torch.save(dict(**receipt, model=model.cpu().state_dict()), f))
    save_json(root / f"{arm}.json", receipt)


@torch.inference_mode()
def render(subject, panel, arms):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_oof_conditions import study_root
    from paper_render import condition, donor_rows, draw_images, generator_inputs, groups, physical_eeg, query_seed
    from paper_token_evidence import development, render_arms, score_outputs

    root, pins = study_root(), PinnedFiles()
    target_file = root / "training-targets.pt"
    target_digest = pins.pin(target_file)
    package = torch.load(target_file, map_location="cuda", weights_only=True)
    split = PREP / "paper_split.json"
    split_digest = pins.pin(split)
    require(package["sources"]["split"] == split_digest, "Token target split differs from render split")
    require(package["sources"]["code"][Path(__file__).name] == pins.pin(Path(__file__)), "Token encoder source changed")
    phase, dev_digest = "train", None
    if panel == "dev":
        plan, rows = development(PREP, pins)
        require(subject in plan["subjects"], "Use the registered development participants")
        dev_digest = pins.pin(PREP / "paper_development.json")
    else:
        rows = json.loads(split.read_bytes())[f"{panel}_rows"]
        phase = "test" if panel == "official" else "train"
        require(len(rows) == len(set(rows)) == (200 if panel == "official" else 500),
                "Expected the complete registered confirmation panel")
        require(panel == "official" or not np.intersect1d(rows, package["rows"].cpu().numpy()).size,
                "Fresh concepts must be absent from fitting")
    repeats = groups(f"{panel}-r4")
    require(repeats == [REPEATS], "Candidate and native inputs must use the same physical repeats")
    raw, physical = physical_eeg(subject, phase, pins)
    native_features, native_checkpoint = condition("native-paper", subject, raw[rows], repeats,
                                                   features_only=True, files=pins)
    assets = generator_inputs(pins)
    for name, digest in package["sources"]["projection"].items():
        require(assets[f"prior/{name}"] == digest, "Frozen visual projection changed after fitting")
    pipeline = load_generator(torch.float16, False)
    _, adapter, projection, _, _ = pipeline
    native = project_tokens(native_features, adapter, projection)
    native_mean, _ = token_parts(native)
    eeg = prepare_eeg(raw, rows)
    people = torch.full((len(rows),), subject - 1, device="cuda", dtype=torch.long)
    variants = []
    relatives = {}
    own = list(range(len(rows)))
    for arm in arms:
        if arm == "native-paper":
            variants.append((arm, native.half().cpu(), dict(relative_eeg_row=own)))
            continue
        name = arm.removesuffix("-wrong")
        if name not in relatives:
            path = root / f"{name}.pt"
            digest = pins.pin(path)
            state = torch.load(path, map_location="cuda", weights_only=True)
            require(state["target_sha256"] == target_digest and state["arm"] == name, "Token checkpoint differs")
            buffers = state["model"]
            model = RelativeTokenEncoder(buffers["input_mean"], buffers["input_scale"], package["rank"]).cuda()
            model.load_state_dict(buffers)
            model.eval()
            prediction = torch.cat([model(eeg[i:i+128], people[i:i+128]) for i in range(0, len(rows), 128)])
            relatives[name] = restore_relative(prediction, package["basis"], calibrate=True), digest, state["plan_sha256"]
        detail, digest, plan_digest = relatives[name]
        donor = own
        if arm.endswith("-wrong"):
            if panel == "official":
                donor = np.roll(own, len(rows) // 2).tolist()
            else:
                labels = PREP / "train_concept.npy"
                pins.pin(labels)
                donor = donor_rows("token-reliable-wrong", panel, np.load(labels, allow_pickle=False)[rows]).tolist()
        tokens = (native_mean + detail[donor]).half()
        require(torch.isfinite(tokens).all(), "Nonfinite relative token render input")
        variants.append((arm, tokens.cpu(), dict(relative_eeg_row=donor, checkpoint_sha256=digest,
                                                fit_plan_sha256=plan_digest)))
    if len(relatives) > 1:
        require(len({value[2] for value in relatives.values()}) == 1, "Matched encoders used different minibatches")
    seeds = [query_seed(panel, n) for n in own]
    shared = dict(subject=subject, budget=f"{panel}-r4", physical_eeg=physical, repeats=repeats,
                  split_sha256=split_digest, development_split_sha256=dev_digest, training_targets_sha256=target_digest,
                  native_checkpoint_sha256=native_checkpoint, generator_files_sha256=assets,
                  code_sha256=pins.pin(Path(__file__)), render_source_sha256=pins.pin(Path(inspect.getfile(draw_images))))
    outputs = render_arms(root / "images" / panel / f"sub-{subject:02d}", variants, rows, seeds, shared, pins, pipeline)
    del pipeline, adapter, projection
    torch.cuda.empty_cache()
    score_outputs(outputs, panel)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="task", required=True)
    actions.add_parser("prepare")
    fit_args = actions.add_parser("fit")
    fit_args.add_argument("arm", choices=FIT_ARMS)
    render_args = actions.add_parser("render")
    render_args.add_argument("subject", type=int, choices=SUBJECTS)
    render_args.add_argument("--panel", choices=("dev", "fresh", "official"), default="dev")
    render_args.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    args = cli.parse_args()
    if args.task == "prepare":
        prepare()
    elif args.task == "fit":
        fit(args.arm)
    else:
        render(args.subject, args.panel, args.arms)
