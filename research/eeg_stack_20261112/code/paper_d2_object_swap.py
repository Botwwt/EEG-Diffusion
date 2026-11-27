"""Train shared D2 to distinguish objects when only the EEG condition is exchanged."""

import argparse
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from paper_render_cache import PinnedFiles, atomic_file, require, save_json


SUBJECTS = tuple(range(1, 11))
PANEL = (4, 7)
DONOR = (2, 3, 0, 1)
MODES = ("object-anchor", "object-swap")
ARMS = ("shared-d2", *MODES, "object-swap-wrong", "shared-d2-wrong")
SEED, STEPS, TEMPERATURE = 20261128, 120, .07
ANCHOR_WEIGHT, SWAP_WEIGHT = .35, .50
LAYOUT_STRENGTH, LAYOUT_FILTER = .375, (8, .11, .02)


def make_episodes(groups, appearance, steps=STEPS):
    """Use the same people, photograph pairs, and noise in the two fits."""
    require(groups.ndim == 2 and min(groups.shape) >= 2 and len(appearance) == len(groups),
            "Each visual object requires at least two photographs")
    rng = np.random.default_rng(SEED)
    unit = F.normalize(appearance.float() - appearance.float().mean(0))
    proximity = unit @ unit.T
    proximity.fill_diagonal_(-torch.inf)
    choices = torch.argsort(proximity, descending=True, stable=True)[:, :min(8, len(groups) - 1)].cpu().numpy()
    subjects = np.concatenate([rng.permutation(len(SUBJECTS)) for _ in range((steps + 9) // 10)])[:steps]
    first = rng.integers(len(groups), size=steps)
    second = choices[first, rng.integers(choices.shape[1], size=steps)]
    rows = np.stack([np.concatenate([rng.choice(groups[c], size=2, replace=False) for c in (a, b)])
                     for a, b in zip(first, second, strict=True)])
    seeds = rng.integers(0, 2**63 - 1, size=steps, dtype=np.int64)
    return subjects, rows, seeds


def photo_loss(similarity):
    same_object = (torch.arange(4, device=similarity.device) // 2)
    masked = (15. * similarity).masked_fill(same_object[:, None] != same_object[None, :], -torch.inf)
    own = torch.arange(4, device=similarity.device)
    return .5 * (F.cross_entropy(masked, own) + F.cross_entropy(masked.T, own) - similarity.diagonal().mean())


def object_margin(logits, target, donor):
    return (logits.gather(1, target[:, None]) - logits.gather(1, donor[:, None])).flatten()


def exchange_loss(original, exchanged, target, donor):
    original_margin = object_margin(original, target, donor)
    swapped_margin = object_margin(exchanged, target, donor)
    return F.softplus((swapped_margin - original_margin) / (2 * TEMPERATURE)).mean()


def d2_models(files, prepared=None):
    from paper_render import CrossSubjectHorizonEncoder, ResidualIPCondLayer, condition_files

    path, backbone, sources = condition_files("d2", SUBJECTS[0], files) if prepared is None else prepared
    require(backbone is not None, "Expected the archived shared D2 residual and its EEG backbone")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    parameters = torch.load(backbone, map_location="cpu", weights_only=True)
    encoder = CrossSubjectHorizonEncoder().cuda().float().eval().requires_grad_(False)
    encoder.load_state_dict(parameters["state_dict"], strict=True)
    original = ResidualIPCondLayer().cuda().float().eval().requires_grad_(False)
    original.load_state_dict(checkpoint["cond_layer"], strict=True)
    return encoder, original, checkpoint["cone"], sources


@torch.no_grad()
def repeat_hidden(encoder, eeg, subject):
    from paper_render import forward_branches_and_trunk, prec_erp_analytic

    _, grouped, _ = forward_branches_and_trunk(encoder, prec_erp_analytic(eeg), subject - 1, 4.)
    _, individual, _ = forward_branches_and_trunk(encoder, eeg.flatten(0, 1), subject - 1, 1.)
    return grouped, individual


def image_tokens(head, residual, hidden, cone, adapter, projection):
    from paper_render import per_sample_cone

    grouped, individual = hidden
    count = len(grouped)
    f = .72 * head(grouped) + .28 * head(individual).reshape(count, 4, -1).mean(1)
    h = .72 * grouped + .28 * individual.reshape(count, 4, -1).mean(1)
    fused = per_sample_cone(f, cone["R"], cone["alpha"], cone["beta"], cone["u_tr"])
    return residual(h, projection(adapter(fused.to(next(adapter.parameters()).dtype))))


def compile_targets(prep, rows, labels, dino_bank, files):
    from common import EMB_KEYS
    from paper_dino import MODEL_FILES

    visual_path = prep / "emb_train.pt"
    layout_path = prep / "lat512_train.npy"
    alignment = prep / "b32_fix.pt"
    proof = {p.name: files.pin(p) for p in (visual_path, layout_path, alignment, dino_bank)}
    targets = torch.load(dino_bank, map_location="cpu", weights_only=True)
    require(np.array_equal(targets["rows"].numpy(), rows)
            and np.array_equal(targets["concepts"][targets["labels"]].numpy(), labels)
            and targets["proof"]["split"] == files.pin(prep / "paper_split.json")
            and targets["proof"]["dino"] == MODEL_FILES, "DINOv2 photo targets use a different training split")
    photos = torch.load(visual_path, map_location="cpu", weights_only=True)
    latent = np.load(layout_path, mmap_mode="r", allow_pickle=False)
    pooled = torch.cat([F.avg_pool2d(torch.from_numpy(latent[batch].copy()).cuda().float(), 4)
                        for batch in np.array_split(rows, (len(rows) + 127) // 128)])
    h14, b32 = (F.normalize(photos[key][rows].cuda().float()) for key in EMB_KEYS[:2])
    return dict(proof=proof, labels=targets["labels"].cuda(), photo=targets["image_features"].cuda().float(),
                sums=targets["sums"].cuda().float(), h14=h14, b32=b32, latent=F.normalize(pooled.flatten(1)),
                appearance=F.avg_pool2d(pooled, 2).flatten(1),
                b32_map=torch.load(alignment, map_location="cpu", weights_only=True).cuda().float())


def class_logits(correct, swapped, labels, given, sums):
    prototypes = F.normalize(sums.index_add(0, labels, -given))
    return correct @ prototypes.T, swapped @ prototypes.T


def train_images(pipe, vae, prompts, tokens, layout, noise):
    from paper_render import rollout_4step_step_ips
    from paper_rgb_object import CheckpointedUNet

    first, *steps = noise.unbind(1)
    latent = rollout_4step_step_ips(CheckpointedUNet(pipe.unet), pipe.scheduler, [tokens] * 4,
                                   layout, first, steps, *prompts, LAYOUT_STRENGTH, LAYOUT_FILTER)
    z = latent.float() / vae.config.scaling_factor
    rgb = checkpoint(lambda value: vae.decode(value).sample, z, use_reentrant=False, preserve_rng_state=False)
    return (rgb * .5 + .5).clamp(0., 1.), latent


def photo_similarity(rgb, latent, rows, targets, h14, b32):
    view = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
    mean = view.new_tensor((.48145466, .4578275, .40821073))[None, :, None, None]
    std = view.new_tensor((.26862954, .26130258, .27577711))[None, :, None, None]
    view = ((view - mean) / std).to(h14.dtype)
    large = checkpoint(lambda x: h14(pixel_values=x).image_embeds.float(), view, use_reentrant=False, preserve_rng_state=False)
    small = checkpoint(lambda x: b32(pixel_values=x).image_embeds.float(), view, use_reentrant=False, preserve_rng_state=False)
    high = F.normalize(large) @ targets["h14"][rows].T
    low = F.normalize(small @ targets["b32_map"]) @ targets["b32"][rows].T
    structural = F.normalize(F.avg_pool2d(latent.float(), 4).flatten(1)) @ targets["latent"][rows].T
    return .60 * high + .25 * low + .15 * structural


def fit(mode, bank, output, steps):
    from common import EMB_KEYS, PREP, ROOT, RUNS
    from exp0_intra10_swap_gate import get_concept_splits
    from exp6_draft_intra4 import DRAFT_DIR
    from paper_dino import MODEL_FILES
    from paper_identity_bank import visual_model
    from paper_identity_render import load_generator
    from paper_oof_conditions import registered_rows
    from paper_render import generator_inputs, physical_eeg, render_code
    from paper_rgb_object import dino_cls
    from paper_training import vision

    require(RUNS.name == "shared", "The two fits start from the shared D2 model")
    files = PinnedFiles()
    rows, labels, _ = registered_rows(PREP)
    _, _, archived_rows, *_ = get_concept_splits()
    require(np.array_equal(rows, archived_rows), "The restored D2 layout order differs from the registered training rows")
    groups = np.argsort(labels, kind="stable").reshape(-1, 10)
    dino_path = ROOT / "models/paper-dinov2-small"
    eeg = [physical_eeg(subject, "train", files) for subject in SUBJECTS]
    encoder, residual, cone, shared = d2_models(files)
    head = deepcopy(encoder.head_f).requires_grad_(True)
    trainable = deepcopy(residual).requires_grad_(True)
    oof_path = DRAFT_DIR / "z0_oof_tr14_256.pt"
    proof = dict(split=files.pin(PREP / "paper_split.json"), d2=shared, layout=files.pin(oof_path),
                 eeg=[p for _, p in eeg], generator=generator_inputs(files), render_sources=render_code(files, "d2"),
                 visual_teacher={key: files.pin(dino_path / key) for key in MODEL_FILES},
                 source_code=files.pin(Path(__file__)), upstream_rollout=files.pin(Path(inspect.getfile(load_generator))),
                 steps=steps, seed=SEED, anchor_weight=ANCHOR_WEIGHT, swap_weight=SWAP_WEIGHT if mode == "object-swap" else 0.)
    require(proof["visual_teacher"] == MODEL_FILES, "Unexpected DINOv2 training model")
    for key in EMB_KEYS[:2]:
        vision_dir = ROOT / "models" / key
        proof[key] = {p.name: files.pin(p) for p in sorted(vision_dir.iterdir()) if p.is_file() and not p.name.startswith(".")}
    proof["targets"] = {p.name: files.pin(p) for p in (PREP / "emb_train.pt", PREP / "lat512_train.npy", PREP / "b32_fix.pt", bank)}
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{mode}.pt"
    if destination.is_file():
        old = torch.load(destination, map_location="cpu", weights_only=True)
        require(old["proof"] == proof and old["mode"] == mode, "An incompatible trained checkpoint exists")
        if not destination.with_suffix(".json").is_file():
            save_json(destination.with_suffix(".json"), dict(old["report"], checkpoint_sha256=files.pin(destination)))
        return
    targets = compile_targets(PREP, rows, labels, bank, files)
    require(targets["proof"] == proof["targets"], "Visual targets changed before fit")
    training = make_episodes(groups, targets["appearance"][torch.from_numpy(groups).cuda()].mean(1).cpu(), steps)
    order = hashlib.sha256(b"".join(x.astype("<i8").tobytes() for x in training)).hexdigest()
    oof = torch.load(oof_path, map_location="cpu", weights_only=True)
    require(oof["Z_oof"].shape == (10, len(rows), 256) and oof["V_pca"].shape == (16384, 256)
            and oof["my"].shape == (16384,), "The archived layout has incompatible axes")
    basis, mean = oof["V_pca"].cuda().float(), oof["my"].cuda().float()
    dtype = torch.bfloat16
    pipe, adapter, project, prompts, vae = load_generator(dtype, True)
    h14, b32 = (vision(key, dtype) for key in EMB_KEYS[:2])
    dino = visual_model(dino_path)
    parameters = [*head.parameters(), *trainable.parameters()]
    optimizer = torch.optim.AdamW([
        {"params": head.parameters(), "lr": 1e-5}, {"params": trainable.parameters(), "lr": 8e-5},
    ], betas=(.9, .995), weight_decay=.05)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=2e-6)
    history = []
    donors = list(DONOR)
    for update, (sid, selected, seed) in enumerate(zip(*training, strict=True), 1):
        local = torch.from_numpy(selected).cuda()
        observed = torch.from_numpy(eeg[sid][0][rows[selected]].copy()).cuda().float()
        hidden = repeat_hidden(encoder, observed, SUBJECTS[sid])
        layout = (oof["Z_oof"][sid, selected].cuda().float() @ basis.T + mean).reshape(4, 4, 64, 64)
        noise = torch.randn(1, 5, 4, 64, 64, generator=torch.Generator().manual_seed(int(seed))).cuda().expand(4, -1, -1, -1, -1)
        with torch.no_grad():
            reference = image_tokens(encoder.head_f, residual, hidden, cone, adapter, project).float()
        tokens = image_tokens(head, trainable, hidden, cone, adapter, project)
        rgb, latent = train_images(pipe, vae, prompts, tokens, layout, noise)
        swapped_rgb, _ = train_images(pipe, vae, prompts, tokens[donors], layout, noise)
        projected = checkpoint(lambda x: dino_cls(dino, x), rgb, use_reentrant=False, preserve_rng_state=False)
        swapped = checkpoint(lambda x: dino_cls(dino, x), swapped_rgb, use_reentrant=False, preserve_rng_state=False)
        names = targets["labels"][local]
        good, bad = class_logits(projected, swapped, names, targets["photo"][local], targets["sums"])
        intra = photo_loss(photo_similarity(rgb, latent, local, targets, h14, b32))
        recognition = F.cross_entropy(good / TEMPERATURE, names)
        exchange = exchange_loss(good, bad, names, names[donors])
        trust = (tokens.float() - reference).square().mean() / reference.square().mean().clamp_min(1e-8)
        loss = intra + ANCHOR_WEIGHT * recognition + proof["swap_weight"] * exchange + .05 * trust
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        require(grad > 0 and torch.isfinite(loss), "Training RGB did not provide a finite EEG gradient")
        optimizer.step()
        schedule.step()
        difference = object_margin(good, names, names[donors]) - object_margin(bad, names, names[donors])
        record = dict(loss=float(loss.detach()), photo=float(intra.detach()), identity=float(recognition.detach()),
                      exchange=float(exchange.detach()), pairing_win=float((difference > 0).float().mean()),
                      pairing_margin=float(difference.detach().mean()), gradient=float(grad))
        history.append(record)
        if update == 1 or update % 10 == 0 or update == steps:
            prior = history[-10:]
            print(mode, update, steps, {k: np.mean([x[k] for x in prior]) for k in record}, flush=True)
    files.verify()
    report = dict(mode=mode, steps=steps, proof=proof, training_order_sha256=order, history=history,
                  parameters=sum(p.numel() for p in parameters), peak_cuda_bytes=torch.cuda.max_memory_allocated())
    state = {"head": head.cpu().state_dict(), "residual": trainable.cpu().state_dict()}
    atomic_file(destination, lambda stream: torch.save(dict(mode=mode, proof=proof, state=state, report=report), stream))
    save_json(destination.with_suffix(".json"), dict(report, checkpoint_sha256=files.pin(destination)))


@torch.inference_mode()
def render(subject, output):
    from common import PREP, RUNS
    from paper_identity_render import load_generator
    from paper_render import draw_images, fit_layout, generator_inputs, groups, layout, physical_eeg, query_seed
    from paper_render_cache import render_transaction
    from paper_token_evidence import score_outputs

    require(RUNS.name == "shared" and subject in PANEL, "Use the registered shared D2 comparison")
    files = PinnedFiles()
    split = PREP / "paper_split.json"
    split_sha = files.pin(split)
    plan = json.loads(split.read_bytes())
    rows = plan["official_rows"]
    repeat = groups("official-r4")
    train, training = physical_eeg(subject, "train", files)
    test, physical = physical_eeg(subject, "test", files)
    require(rows == list(range(200)) and repeat == [[0, 1, 2, 3]], "Wrong official stimulus or EEG ordering")
    enc, residual, cone, sources = d2_models(files)
    generator = generator_inputs(files)
    pipe, adapter, project, prompts, vae = load_generator(torch.float16, False)
    ridge, layout_source = fit_layout(subject, plan, train, files=files, train_proof=training, with_source=True)
    starts = layout(ridge, test, repeat)
    blocks = [repeat_hidden(enc, torch.from_numpy(test[lo:lo + 20, :4].copy()).cuda().float(), subject)
              for lo in range(0, len(rows), 20)]
    values = {"shared-d2": torch.cat([image_tokens(enc.head_f, residual, h, cone, adapter, project).cpu() for h in blocks])}
    fits = {}
    for mode in MODES:
        fit_path = output / "fits" / f"{mode}.pt"
        report_path = fit_path.with_suffix(".json")
        fit_sha = files.pin(fit_path)
        report = json.loads(report_path.read_bytes())
        package = torch.load(fit_path, map_location="cpu", weights_only=True)
        proof = package["proof"]
        require(package["mode"] == mode and proof["steps"] == STEPS and proof["source_code"] == files.pin(Path(__file__))
                and proof["split"] == split_sha and proof["d2"] == sources and proof["generator"] == generator
                and report["checkpoint_sha256"] == fit_sha, "Wrong fitted EEG decoder or generator")
        head, adjusted = deepcopy(enc.head_f), deepcopy(residual)
        head.load_state_dict(package["state"]["head"], strict=True)
        adjusted.load_state_dict(package["state"]["residual"], strict=True)
        values[mode] = torch.cat([image_tokens(head, adjusted, h, cone, adapter, project).cpu() for h in blocks])
        fits[mode] = dict(checkpoint_sha256=fit_sha, report_sha256=files.pin(report_path),
                          training_order_sha256=report["training_order_sha256"], steps=proof["steps"])
    require(fits[MODES[0]]["training_order_sha256"] == fits[MODES[1]]["training_order_sha256"], "Training plans differ")
    seeds = [query_seed("official", row) for row in rows]
    donors = np.roll(rows, len(rows) // 2).tolist()
    shared = dict(render_cache_version=2, subject=subject, row=rows, n=len(rows), budget="official-r4", repeats=repeat,
                  physical_eeg=physical, split_sha256=split_sha, generator_files_sha256=generator, layout_source=layout_source,
                  d2_sources=sources, code_sha256=files.pin(Path(__file__)), seeds=seeds, draws=1,
                  unet_calls_per_draw=4, prompt="", diagnostic_only=False)
    folders = []
    for arm in ARMS:
        source_arm = arm.removesuffix("-wrong")
        donor = donors if arm.endswith("-wrong") else rows
        tokens = values[source_arm][donor]
        folder = output / "images" / f"sub-{subject:02d}" / arm
        request = dict(shared, arm=arm, eeg_donor=donor, layout_donor=rows, fit=fits.get(source_arm))

        def generate(stage, batches, complete):
            draw_images(stage, pipe, vae, prompts, tokens, seeds, starts, batch_starts=batches, batch_complete=complete)

        render_transaction(folder, request, generate, files.verify)
        folders.append(folder)
        print("Cross-object render", subject, arm, len(rows), flush=True)
    del pipe, adapter, project, vae
    torch.cuda.empty_cache()
    score_outputs(folders, "official")
    files.verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    training = sub.add_parser("fit")
    training.add_argument("--mode", choices=MODES, required=True)
    training.add_argument("--bank", type=Path, required=True)
    training.add_argument("--steps", type=int, default=STEPS, choices=(2, STEPS))
    images = sub.add_parser("render")
    images.add_argument("subject", type=int, choices=PANEL)
    for task in (training, images):
        task.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "fit":
        fit(args.mode, args.bank, args.output, args.steps)
    else:
        render(args.subject, args.output)
