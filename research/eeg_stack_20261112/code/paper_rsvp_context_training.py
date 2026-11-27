"""Train an EEG encoder on the same photograph under exchanged stimulus contexts."""
import argparse
from inspect import getfile
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


SEED = 20261125
TEMPERATURE = .07
MODES = ("repeated", "shuffled-context", "exchanged-context")
ARMS = ("native-paper", *MODES, "exchanged-wrong")
BUDGETS = ("dev-r4", "official-r4", "official-r80")


def context_donors(groups, concepts):
    first = np.empty(len(groups), dtype=np.int64)
    second = np.empty_like(first)
    rng = np.random.default_rng(SEED)
    for group in np.unique(groups):
        columns = np.flatnonzero(groups == group)
        order = columns[np.lexsort((rng.random(len(columns)), concepts[columns]))]
        step = len(order) // 3
        if step < np.unique(concepts[order], return_counts=True)[1].max():
            raise ValueError("Could not assign two distinct other concepts from the same recording group")
        first[order] = np.roll(order, step)
        second[order] = np.roll(order, 2 * step)
    return first, second


def context_changes(context, replacement, placebo):
    swapped = context[replacement] - context
    shuffled = context[replacement] - context[placebo]
    shuffled = F.normalize(shuffled.flatten(1)).reshape_as(shuffled)
    norm = swapped.flatten(1).norm(dim=1).reshape(-1, *([1] * (context.ndim - 1)))
    return swapped, shuffled * norm


def lift_changes(values):
    blocks = values.reshape(len(values), 2, 17, 10).transpose(1, 2)
    voltage = blocks.flatten(2).repeat_interleave(5, dim=-1)
    return F.pad(voltage, (0, 150))


def root():
    return Path(os.environ["EEG_CROSSFIT_ROOT"])


@torch.inference_mode()
def perturbations(raw, rows, classes, folds, photo, edges, sessions):
    from paper_render_cache import require
    from paper_rsvp_deconvolution import complete_context, fit_kernels, photo_coordinates, wave_bins

    events = complete_context(edges)
    current, repeat, prev, nxt = events.T
    lookup = np.full(len(raw), -1, dtype=np.int64)
    lookup[rows] = np.arange(len(rows))
    registered = lookup[events[:, [0, 2, 3]]]
    require(registered.min() >= 0, "A photograph outside the registered training split entered the context model")
    photo_local, prev_local, next_local = registered.T
    event_folds = folds[registered]
    voltage = torch.cat([torch.from_numpy(raw[part].copy()).cuda().float() for part in np.array_split(rows, 128)])
    wave = wave_bins(voltage[photo_local, repeat])
    predicted = torch.empty_like(wave)
    for fold in np.unique(folds):
        train = np.all(event_folds != fold, axis=1)
        held = event_folds[:, 0] == fold
        source = photo_coordinates(photo, rows[folds != fold])
        own, before, after = source[current], source[prev], source[nxt]
        effects = wave[train] - wave[train].mean(0)
        early, late = fit_kernels(own[train], before[train], after[train], *effects.unbind(1))
        predicted[held] = torch.stack((before[held] @ late, after[held] @ early), dim=1)
    group = 4 * (folds[photo_local] - 1) + sessions[current, repeat]
    donor, sham = context_donors(group, classes[photo_local])
    changes = context_changes(predicted, donor, sham)
    stats = dict(contexts=len(events), photos_with_context=len(np.unique(current)),
                 eeg_rms_0_400=float(wave.square().mean().sqrt()),
                 model_context_rms=float(predicted.square().mean().sqrt()),
                 delta_rms=float(changes[0].square().mean().sqrt()),
                 control_delta_rms=float(changes[1].square().mean().sqrt()))
    require(torch.isfinite(predicted).all() and abs(stats["delta_rms"] - stats["control_delta_rms"]) < 1e-6,
            "Train augmentations must match their placebo strength")
    corrected = {}
    for mode, delta in zip(("exchanged-context", "shuffled-context"), changes, strict=True):
        total = voltage.new_zeros(len(rows), 2, 170).index_add_(0, torch.as_tensor(photo_local, device="cuda"), delta)
        corrected[mode] = lift_changes(total / 4)
    return voltage.mean(1), corrected, stats


def photograph_loss(predicted, target, center, spread):
    logits = F.normalize(predicted - center) @ F.normalize(target - center).T / TEMPERATURE
    ids = torch.arange(len(target), device=target.device)
    contrast = .5 * (F.cross_entropy(logits, ids) + F.cross_entropy(logits.T, ids))
    return contrast + (predicted - target).square().mean() / spread


def fit(subject, context_dir):
    from common import PREP, ROOT
    from paper_object_margin_encoder import confuser_log_prior, object_loss, visual_transfer
    from paper_object_tokens import BATCH, EPOCHS, epoch_batches, group_index
    from paper_oof_conditions import fitted_native, registered_rows
    from paper_render import physical_eeg
    from paper_render_cache import PinnedFiles, atomic_file, require, save_json
    from paper_rsvp_deconvolution import fit_kernels

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    files = PinnedFiles()
    rows, classes, folds = registered_rows(PREP)
    native, native_path = fitted_native(subject, "cuda")
    raw, eeg = physical_eeg(subject, "train", files)
    context = context_dir / f"sub-{subject:02d}-training-context.npy"
    audit = context_dir / f"sub-{subject:02d}-training-context-audit.json"
    session = context_dir / f"sub-{subject:02d}_train.npy"
    photographs = PREP / "F_train.npy"
    dino = ROOT / "crossfit_generator/object-identity/training-photo-dino.pt"
    proof = dict(eeg=eeg, split=files.pin(PREP / "paper_split.json"), native=files.pin(native_path),
                 context=files.pin(context), audit=files.pin(audit), sessions=files.pin(session), photos=files.pin(photographs),
                 dino=files.pin(dino), code=files.pin(__file__), timing_model=files.pin(getfile(fit_kernels)),
                 object_loss=files.pin(getfile(object_loss)), batch_order=files.pin(getfile(epoch_batches)))
    validation = json.loads(audit.read_bytes())
    require(validation["subject"] == subject and validation["verified_eeg_sha256"] == eeg["array_sha256"]
            and validation["exact_order_better"] > .99 and validation["input"]["split_sha256"] == proof["split"]
            and validation["input"]["session_sha256"] == proof["sessions"]
            and validation["input"]["context_array_sha256"] == proof["context"], "Mismatched RSVP event or EEG input")
    sessions = np.load(session, allow_pickle=False)
    require(sessions.shape == (len(raw), 4) and sessions.min() == 0 and sessions.max() == 3,
            "Unexpected physical session order")
    edges = np.load(context, allow_pickle=False)
    features = torch.from_numpy(np.load(photographs, allow_pickle=False)).cuda().float()
    mean, deltas, stats = perturbations(raw, rows, classes, folds, features, edges, sessions)
    photo = features[rows]
    print("Verified training context augmentation", subject, stats, flush=True)
    teacher = torch.load(dino, weights_only=True, map_location="cuda")
    labels, dino_features, sums = (teacher[key] for key in ("labels", "image_features", "sums"))
    require(teacher["proof"]["split"] == proof["split"] and np.array_equal(rows, teacher["rows"].cpu())
            and np.array_equal(classes, teacher["concepts"][labels].cpu()), "Mismatched object image teacher")
    _, ordered = group_index(labels)
    center, dino_center, project = visual_transfer(photo, dino_features)
    exact = F.normalize(dino_features - dino_center)
    prototypes = F.normalize(sums / ordered.shape[1] - dino_center)
    log_prior = confuser_log_prior(exact, prototypes, labels, 32, 1.)
    spread = (photo - center).square().mean().clamp_min(1e-6)
    initial = {key: value.cpu().clone() for key, value in native.state_dict().items()}
    native.backbone.requires_grad_(True)
    steps = EPOCHS * ordered.shape[1] * math.ceil(len(ordered) / BATCH)
    output = root() / "fits" / f"sub-{subject:02d}"
    output.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        rng = torch.Generator(device="cuda").manual_seed(SEED)
        torch.manual_seed(SEED)
        native.load_state_dict(initial)
        native.train()
        optimizer = torch.optim.AdamW(native.backbone.parameters(), lr=6e-5, betas=(.9, .995), weight_decay=.05)
        decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=5e-6)
        augmented = mean if mode == "repeated" else mean + deltas[mode]
        history = []
        for epoch in range(EPOCHS):
            total = np.zeros(3)
            for chosen in epoch_batches(ordered, rng):
                normal, altered = native.backbone(torch.cat((mean[chosen], augmented[chosen]))).chunk(2)
                truth = photo[chosen]
                core = .5 * (photograph_loss(normal, truth, center, spread) +
                             photograph_loss(altered, truth, center, spread))
                semantic = torch.cat((normal, altered))
                same = chosen.repeat(2)
                objects = object_loss((semantic - center) @ project, exact[same], prototypes,
                                      labels[same], log_prior[same])
                consistency = (1. - F.cosine_similarity(normal - center, altered - center)).mean()
                optimizer.zero_grad(set_to_none=True)
                (core + .25 * objects + .1 * consistency).backward()
                torch.nn.utils.clip_grad_norm_(native.backbone.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                decay.step()
                total += len(chosen) * np.array([float(core.detach()), float(objects.detach()), float(consistency.detach())])
            history.append((total / len(rows)).tolist())
            print("Context EEG fit", subject, mode, epoch + 1, history[-1], flush=True)
        model = output / f"{mode}.pt"
        files.verify()
        atomic_file(model, lambda stream: torch.save({key: value.cpu() for key, value in native.state_dict().items()}, stream))
        save_json(model.with_suffix(".json"), dict(subject=subject, mode=mode, photos=len(rows), proof=proof, stats=stats,
                                                  steps=steps, history=history, weights=files.pin(model)))


@torch.inference_mode()
def infer(subject, budgets, render_budgets):
    from common import PREP
    from paper_object_foreground import predict_at_budget, text_targets
    from paper_object_foreground_render import split_input
    from paper_oof_conditions import fitted_native
    from paper_render import groups
    from paper_render_cache import PinnedFiles, atomic_file, require
    from paper_repeat_semantic_bridge import fit_head, predict
    from paper_soft_text import wrong_donors

    files = PinnedFiles()
    native, baseline = fitted_native(subject, "cuda")
    checkpoints = {"native-paper": (baseline, files.pin(baseline))}
    split = files.pin(PREP / "paper_split.json")
    photo = PREP / "F_train.npy"
    photo_digest = files.pin(photo)
    for mode in MODES:
        path = root() / "fits" / f"sub-{subject:02d}" / f"{mode}.pt"
        audit = path.with_suffix(".json")
        files.pin(audit)
        recorded, digest = json.loads(audit.read_bytes()), files.pin(path)
        require(recorded["weights"] == digest and recorded["mode"] == mode and recorded["subject"] == subject
                and recorded["proof"]["native"] == checkpoints["native-paper"][1]
                and recorded["proof"]["split"] == split and recorded["proof"]["photos"] == photo_digest,
                "Unexpected tuned encoder weights")
        checkpoints[mode] = (path, digest)
    training, labels, _, targets, states, basis = text_targets(files)
    image = torch.from_numpy(np.load(photo, mmap_mode="r", allow_pickle=False)[training].copy()).float().cuda()
    head = fit_head(image, targets.cuda(), labels.cuda())
    pipeline = None
    for budget in budgets:
        raw, proof, rows = split_input(subject, budget, files)
        repeats = groups(budget)
        features = {}
        for mode, (path, _) in checkpoints.items():
            native.load_state_dict(torch.load(path, weights_only=True, map_location="cuda"))
            features[mode] = predict_at_budget(native.backbone, raw, repeats)
        donors = wrong_donors(len(rows)).tolist()
        features["exchanged-wrong"] = features["exchanged-context"][donors]
        shared = dict(subject=subject, budget=budget, row=rows, repeats=repeats, physical_eeg=proof,
                      native_checkpoint_sha256=checkpoints["native-paper"][1],
                      encoder_sha256={mode: item[1] for mode, item in checkpoints.items()},
                      split_sha256=split, images_during_training=photo_digest, semantic_basis_sha256=basis,
                      code_sha256=files.pin(__file__))
        codes = {mode: predict(head, z).cpu() for mode, z in features.items()}
        vectors = {mode: states["mean_pooled"] + value @ states["pooled_basis"] for mode, value in codes.items()}
        out = root() / "images" / budget / f"sub-{subject:02d}"
        out.mkdir(parents=True, exist_ok=True)
        files.verify()
        atomic_file(out / "semantic-readout.pt", lambda file: torch.save(dict(proof=shared, pool=vectors, donors=donors), file))
        print("Context-trained EEG readout", subject, budget, len(rows), flush=True)
        if budget in render_budgets:
            if pipeline is None:
                from paper_identity_render import load_generator

                pipeline = load_generator(torch.float16, False)
            render(out, features, pipeline, shared, files, donors)


def render(folder, features, pipeline, shared, files, donors):
    from paper_render import draw_images, generator_inputs, query_seed, tensor_fingerprint
    from paper_render_cache import render_transaction
    from paper_token_evidence import project_tokens

    pipe, adapter, project, prompt, vae = pipeline
    panel = shared["budget"].split("-")[0]
    indices = list(range(len(features["native-paper"])))
    seeds = [query_seed(panel, row) for row in indices]
    assets = generator_inputs(files)
    for mode, visual in features.items():
        tokens = project_tokens(visual, adapter, project).half().cpu()
        frozen = dict(render_cache_version=2, arm=mode, n=len(seeds), seeds=seeds, draws=1, unet_calls_per_draw=4,
                      generator_inputs=assets, image_token_sha256=tensor_fingerprint(tokens),
                      image_eeg_donor=donors if mode == "exchanged-wrong" else indices, **shared)

        def sample(path, starts, complete):
            draw_images(path, pipe, vae, prompt, tokens, seeds, batch_starts=starts, batch_complete=complete)

        render_transaction(folder / mode, frozen, sample, files.verify)
        print("Context-trained EEG render", shared["subject"], shared["budget"], mode, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("fit", "infer", "score"):
        stage = actions.add_parser(name)
        stage.add_argument("subject", type=int, choices=(1, 3))
        if name == "fit":
            stage.add_argument("--context", type=Path, required=True)
        else:
            stage.add_argument("--budgets", choices=BUDGETS, nargs="+", default=BUDGETS)
            stage.add_argument("--render-budgets", choices=BUDGETS, nargs="+", default=[])
    args = parser.parse_args()
    if args.action == "fit":
        fit(args.subject, args.context)
    else:
        if not set(args.render_budgets).issubset(args.budgets):
            parser.error("Render budgets must also be EEG readout budgets")
        if args.action == "infer":
            infer(args.subject, args.budgets, args.render_budgets)
        else:
            from paper_object_foreground_render import score

            score(args.subject, args.budgets, args.render_budgets, ARMS)
