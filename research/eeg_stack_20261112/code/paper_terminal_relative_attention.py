"""Compare image value updates learned from photographs and from objects."""

import argparse
from contextlib import contextmanager
import hashlib
from inspect import getfile
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from exp6_draft_intra4 import DRAFT_DIR, rollout_4step_step_ips
from paper_d2_object_swap import SUBJECTS, d2_models, image_tokens, make_episodes, repeat_hidden
from paper_eeg_attention import ImageProjectionUpdate, image_layers
from paper_render_cache import PinnedFiles, atomic_file, render_transaction, require, save_json
from paper_rgb_object import CheckpointedUNet, dino_pixels


MODES = ("photo", "sibling", "foreground")
ARMS = ("d2", *MODES, "foreground-wrong")
PLACEMENTS = ("terminal", "all")
RANK, STEPS, SEED = 4, 120, 20261127
LAST, TEMPERATURE = 249, .07
TIMESTEPS = (999, 749, 499, LAST)


def relative(tokens):
    tokens = tokens.float()
    return tokens - tokens.mean(dim=-2, keepdim=True)


class RelativeValueUpdate(ImageProjectionUpdate):
    source = None

    def forward(self, tokens):
        original = self.base(tokens)
        if not self.enabled:
            return original
        direction = relative(tokens) if self.source is None else self.source
        require(direction.shape == tokens.shape, "The exchanged relative input must match the attention tokens")
        return original + F.linear(F.linear(direction, self.down), self.up).to(original.dtype)


def add_value_updates(unet, placement):
    updates = {}
    for name, (processor, attr) in image_layers(unet).items():
        if attr == "to_v_ip" and (placement == "all" or name.startswith("up_blocks.0.")):
            updates[name] = RelativeValueUpdate(getattr(processor, attr), RANK)
            setattr(processor, attr, updates[name])
    require(updates, "No image value projections in the selected SDXL blocks")
    return updates


@contextmanager
def update_context(updates, *, enabled, source=None):
    previous = [(m.enabled, m.source) for m in updates.values()]
    try:
        for module in updates.values():
            module.enabled, module.source = enabled, source
        yield
    finally:
        for module, state in zip(updates.values(), previous, strict=True):
            module.enabled, module.source = state


class ValueUpdateUNet:
    def __init__(self, unet, updates, source, timesteps):
        self.unet, self.updates, self.source, self.timesteps = unet, updates, source, timesteps
        self.dtype = unet.dtype

    def __call__(self, sample, timestep, **kwargs):
        with update_context(self.updates, enabled=int(timestep[0]) in self.timesteps, source=self.source):
            return self.unet(sample, timestep, **kwargs)


def target_prototypes(bank, labels):
    count = int(labels.max()) + 1
    require(torch.all(torch.bincount(labels) == 10).item(), "Every training concept must have ten photographs")
    prototypes = {"photo": bank["uniform"]}
    for mode, key in (("sibling", "uniform"), ("foreground", "foreground")):
        vectors = bank[key]
        summed = vectors.new_zeros(count, vectors.shape[1]).index_add_(0, labels, vectors)
        prototypes[mode] = F.normalize(summed[labels] - vectors)
    return prototypes


def object_scores(model, rgb, references):
    patches = model(pixel_values=dino_pixels(rgb)).last_hidden_state[:, 1:].float()
    similarities = F.normalize(patches, dim=-1) @ references.T
    return TEMPERATURE * ((similarities / TEMPERATURE).logsumexp(dim=1) - math.log(patches.shape[1]))


def episode_conditions(training, rows, eeg, encoder, residual, cone, adapter, projection, oof):
    basis, mean = oof["V_pca"].cuda().float(), oof["my"].cuda().float()
    tokens, layouts, seeds = [], [], []
    for sid, selected, seed in zip(*training, strict=True):
        photo = selected[[0, 2]]
        physical = torch.from_numpy(eeg[sid][0][rows[photo]].copy()).cuda().float()
        with torch.no_grad():
            signal = repeat_hidden(encoder, physical, SUBJECTS[sid])
            token = image_tokens(encoder.head_f, residual, signal, cone, adapter, projection)
        tokens.append(token.detach())
        layouts.append((oof["Z_oof"][sid, photo].cuda().float() @ basis.T + mean).reshape(2, 4, 64, 64))
        seeds.append(int(seed))
    return tokens, layouts, seeds


def fit(output, bank_path, steps, placement="terminal", modes=MODES):
    from common import PREP, ROOT, RUNS
    from exp0_intra10_swap_gate import get_concept_splits
    from paper_dino import MODEL_FILES
    from paper_identity_bank import visual_model
    from paper_identity_render import load_generator
    from paper_oof_conditions import registered_rows
    from paper_photo_top1 import DEFAULT_LAYOUT
    from paper_render import generator_inputs, physical_eeg, render_code

    require(RUNS.name == "shared" and steps >= 10, "Expected shared D2 and a schedule covering all ten participants")
    pins = PinnedFiles()
    require(Path(getfile(rollout_4step_step_ips)).resolve() == Path(__file__).with_name("exp6_draft_intra4.py").resolve(),
            "The release must load the bundled continuation before the archived experiment code")
    rows, labels, _ = registered_rows(PREP)
    require(np.array_equal(rows, get_concept_splits()[2]), "The D2 layout cache uses a different training order")
    label_ids, inverse = np.unique(labels, return_inverse=True)
    groups = np.argsort(inverse, kind="stable").reshape(len(label_ids), 10)
    oof_path = DRAFT_DIR / "z0_oof_tr14_256.pt"
    latent_path = PREP / "lat512_train.npy"
    dino_path = ROOT / "models/paper-dinov2-small"
    split_hash = pins.pin(PREP / "paper_split.json")
    bank_hash = pins.pin(bank_path)
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    require(np.array_equal(bank["rows"].numpy(), rows) and bank["proof"]["split"] == split_hash
            and bank["proof"]["dino"] == MODEL_FILES, "The object teacher uses a different split or checkpoint")
    prototypes = target_prototypes({key: bank["views"][key].cuda().float() for key in ("uniform", "foreground")},
                                   torch.from_numpy(inverse).cuda())
    del bank
    eeg = [physical_eeg(subject, "train", pins) for subject in SUBJECTS]
    encoder, residual, cone, model_source = d2_models(pins)
    proof = dict(split=split_hash, placement=placement, train_eeg=[source for _, source in eeg], d2=model_source,
                 layout=pins.pin(oof_path), reference_layout=pins.pin(latent_path), targets=bank_hash,
                 generator=generator_inputs(pins), render_sources=render_code(pins, "d2"),
                 dino={name: pins.pin(dino_path / name) for name in MODEL_FILES},
                 code={name: pins.pin(Path(__file__).with_name(name)) for name in (Path(__file__).name,
                       "paper_d2_object_swap.py", "paper_eeg_attention.py", "paper_rgb_object.py", "exp6_draft_intra4.py")})
    require(proof["dino"] == MODEL_FILES, "Unexpected DINOv2 weights")
    print("Pinned training inputs", len(rows), len(groups), flush=True)
    latent = np.load(latent_path, mmap_mode="r", allow_pickle=False)
    appearance = torch.cat([F.avg_pool2d(torch.from_numpy(latent[index].copy()).float(), 8).flatten(1)
                            for index in np.array_split(rows, 64)])
    training = make_episodes(groups, appearance[groups].mean(1), steps)
    schedule = hashlib.sha256(b"".join(x.astype("<i8").tobytes() for x in training)).hexdigest()
    oof = torch.load(oof_path, map_location="cpu", weights_only=True)
    require(oof["Z_oof"].shape == (10, len(rows), 256) and oof["V_pca"].shape == (16384, 256)
            and oof["my"].shape == (16384,), "Unexpected fold-out layout axes")
    pipe, adapter, projection, prompts, vae = load_generator(torch.bfloat16, True)
    require(tuple(pipe.scheduler.timesteps.tolist()) == TIMESTEPS, "Unexpected four-step schedule")
    tokens, layouts, seeds = episode_conditions(training, rows, eeg, encoder, residual, cone, adapter, projection, oof)
    del encoder, residual, cone, adapter, projection, oof
    dino = visual_model(dino_path)
    torch.manual_seed(SEED)
    updates = add_value_updates(pipe.unet, placement)
    original = {name: module.export() for name, module in updates.items()}
    now = CheckpointedUNet(pipe.unet)
    params = [p for module in updates.values() for p in (module.down, module.up)]
    print("Trainable image values", len(updates), sum(p.numel() for p in params), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    reference_index = torch.arange(2, device="cuda")
    roll = rollout_4step_step_ips
    for mode in modes:
        destination = output / f"{mode}.pt"
        if destination.is_file():
            old = torch.load(destination, map_location="cpu", weights_only=True)
            require(old["proof"] == proof and old["schedule_sha256"] == schedule and old["mode"] == mode,
                    "An incompatible terminal attention fit exists")
            continue
        with torch.no_grad():
            for name, module in updates.items():
                module.up.copy_(original[name]["up"])
                module.down.copy_(original[name]["down"])
        optimizer = torch.optim.AdamW(params, lr=5e-4, betas=(.9, .995), weight_decay=.01)
        history = []
        for i, (token, layout, seed, photos) in enumerate(zip(tokens, layouts, seeds, training[1], strict=True)):
            noise = torch.randn(5, 4, 64, 64, generator=torch.Generator().manual_seed(seed)).cuda()
            noises = [p.expand_as(layout) for p in noise.unbind(0)]
            eps, *step_noises = noises
            strength, keep = DEFAULT_LAYOUT
            args = (pipe.scheduler, [token] * 4, layout, eps, step_noises, *prompts, strength, keep)
            with torch.no_grad(), update_context(updates, enabled=False):
                prefix = roll(pipe.unet, *args, stop_step=3) if placement == "terminal" else None
                resume = {} if prefix is None else {"start_latent": prefix, "start_step": 3}
                base = roll(pipe.unet, *args, **resume)
            if i == 0:
                with torch.no_grad():
                    initial = roll(pipe.unet, *args, **resume)
                require(torch.equal(base, initial), "Zero-initialized values changed the baseline render")

            def duplicate(x):
                return torch.cat((x, x))

            input_tokens = duplicate(token)
            directions = relative(token.to(pipe.unet.dtype))
            exchanged = torch.cat((directions, directions.flip(0)))
            ref = prototypes[mode][photos[[0, 2]]]
            if prefix is not None:
                resume["start_latent"] = duplicate(prefix)
            with update_context(updates, enabled=True, source=exchanged):
                latent_out = roll(now, pipe.scheduler, [input_tokens] * 4, duplicate(layout), duplicate(eps),
                                  [duplicate(n) for n in step_noises], *prompts, strength, keep, **resume)
                rgb = checkpoint(lambda z: vae.decode(z / vae.config.scaling_factor).sample,
                                 latent_out, use_reentrant=False, preserve_rng_state=False)
                scores = checkpoint(lambda pixels: object_scores(dino, pixels, ref),
                                    (rgb.float() * .5 + .5).clamp(0., 1.), use_reentrant=False, preserve_rng_state=False)
                own, wrong = scores.split(2)
                margin = own.diag() - own.flip(1).diag()
                wrong_margin = wrong.diag() - wrong.flip(1).diag()
                pairing = F.softplus((wrong_margin - margin) / TEMPERATURE).mean()
                anchor = F.avg_pool2d(latent_out[:2] - base, 8).square().mean()
                identity = F.cross_entropy(own / TEMPERATURE, reference_index)
                total = identity + .5 * pairing + .5 * anchor
                optimizer.zero_grad(set_to_none=True)
                total.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            require(i != 0 or float(grad) > 0, "RGB supervision cannot reach the value updates")
            optimizer.step()
            history.append([float(x.detach()) for x in (identity, pairing, anchor, grad)])
            if i == 0 or (i + 1) % 16 == 0:
                print(mode, i + 1, np.mean(history[-16:], axis=0).tolist(), flush=True)
        saved = dict(mode=mode, proof=proof, schedule_sha256=schedule, steps=steps, rank=RANK,
                     placement=placement, terminal_step=LAST if placement == "terminal" else None, history=history,
                     values={name: module.export() for name, module in updates.items()})
        pins.verify()
        atomic_file(destination, lambda f: torch.save(saved, f))
        save_json(destination.with_suffix(".json"), dict(mode=mode, proof=proof, schedule_sha256=schedule,
                                                        steps=steps, rank=RANK, checkpoint_sha256=pins.pin(destination)))


@torch.inference_mode()
def render(subject, output, checkpoints, placement="terminal"):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_photo_top1 import DEFAULT_LAYOUT, render_inputs
    from paper_render import condition, donor_rows, draw_images, generator_inputs, layout, query_seed, runtime, tensor_fingerprint
    from paper_token_evidence import score_outputs
    from paper_uniform_baselines import reference_layout

    pins = PinnedFiles()
    raw, rows, _, budget, groups, _, split, dev, eeg = render_inputs(subject, "dev", pins)
    labels = np.load(PREP / "train_concept.npy", allow_pickle=False)[rows]
    own = np.arange(len(rows))
    donors = donor_rows("token-reliable-wrong", "dev", labels)
    fitted, layout_source = reference_layout(subject, split, pins)
    sketch = layout(fitted, raw, groups)
    pipe, adapter, projection, prompts, vae = load_generator(torch.float16, False)
    tokens, checkpoint_sha, d2 = condition("d2", subject, raw, groups, adapter, projection, files=pins, return_source=True)
    del adapter, projection
    weights = generator_inputs(pins)
    seeds = [query_seed("dev", int(i)) for i in own]
    require(tuple(pipe.scheduler.timesteps.tolist()) == TIMESTEPS, "Unexpected four-step schedule")
    updates = add_value_updates(pipe.unet, placement)
    timesteps = (LAST,) if placement == "terminal" else TIMESTEPS
    trial = f"{placement}-relative"
    strength, keep = DEFAULT_LAYOUT
    base = dict(render_cache_version=2, training=trial, subject=subject, budget=budget, n=len(rows), row=rows,
                eeg_donor=own.tolist(), repeats=groups, seeds=seeds, prompt="", draws=1, unet_calls_per_draw=4,
                split_sha256=split, development_split_sha256=dev, eeg=eeg, d2=d2, checkpoint_sha256=checkpoint_sha,
                layout_source=layout_source, layout_sha256=tensor_fingerprint(sketch), token_sha256=tensor_fingerprint(tokens),
                generator=weights, runtime=runtime(), code=pins.pin(Path(__file__)))
    paths = []
    for arm in ARMS:
        is_base, is_wrong = arm == "d2", arm.endswith("-wrong")
        branch = donors if is_wrong else own
        source = relative(tokens[branch]) if is_wrong else None
        digest = None
        if not is_base:
            mode = arm.removesuffix("-wrong")
            artifact = checkpoints / f"{mode}.pt"
            digest = pins.pin(artifact)
            pkg = torch.load(artifact, map_location="cpu", weights_only=True)
            require(pkg["mode"] == mode and pkg["proof"]["split"] == split and pkg["proof"]["generator"] == weights
                    and pkg["rank"] == RANK and pkg.get("placement", "terminal") == placement
                    and set(pkg["values"]) == set(updates), "The value fit differs from the render configuration")
            for name, module in updates.items():
                state = pkg["values"][name]
                module.up.copy_(state["up"])
                module.down.copy_(state["down"])
        folder = output / "images" / trial / arm / budget / f"sub-{subject:02d}"
        step_report = {} if placement == "terminal" else {"applied_timesteps": [] if is_base else timesteps}
        manifest = dict(base, arm=arm, update_sha256=digest,
                        applied_timestep=None if is_base or placement != "terminal" else LAST, **step_report,
                        branch_donors=None if is_base else branch.tolist())

        def draw(stage, starts, complete):
            for start in starts:
                donor = None if source is None else source[start:start + 8].cuda()
                unet = ValueUpdateUNet(pipe.unet, {} if is_base else updates, donor, timesteps)
                renderer = SimpleNamespace(unet=unet, scheduler=pipe.scheduler)
                with update_context(updates, enabled=False):
                    draw_images(stage, renderer, vae, prompts, tokens, seeds, sketch, batch_starts=[start],
                                batch_complete=complete, layout_strength=strength, layout_low_keep=keep)

        render_transaction(folder, manifest, draw, pins.verify)
        paths.append(folder)
        print(arm, subject, len(rows), flush=True)
    del pipe, vae, updates
    torch.cuda.empty_cache()
    score_outputs(paths, "dev")
    save_json(output / f"{trial}.json", dict(subject=subject, arms=list(ARMS), input_proof=base))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    learn = commands.add_parser("fit")
    learn.add_argument("--steps", type=int, default=STEPS)
    learn.add_argument("--targets", type=Path, required=True)
    learn.add_argument("--output", type=Path, required=True)
    learn.add_argument("--placement", choices=PLACEMENTS, default="terminal")
    learn.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    draw = commands.add_parser("render")
    draw.add_argument("--subject", type=int, required=True)
    draw.add_argument("--checkpoints", type=Path, required=True)
    draw.add_argument("--output", type=Path, required=True)
    draw.add_argument("--placement", choices=PLACEMENTS, default="terminal")
    args = parser.parse_args()
    if args.command == "fit":
        fit(args.output, args.targets, args.steps, args.placement, args.modes)
    else:
        render(args.subject, args.output, args.checkpoints, args.placement)
