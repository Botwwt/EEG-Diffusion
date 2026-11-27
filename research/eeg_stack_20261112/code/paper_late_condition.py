"""Fit paired visual response supervision on a running R4 reconstruction trajectory."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from paper_d2_object_swap import (DONOR, LAYOUT_FILTER, LAYOUT_STRENGTH, SUBJECTS, d2_models,
                                  image_tokens, make_episodes, photo_loss, repeat_hidden)
from paper_render_cache import PinnedFiles, atomic_file, require, save_json


STEPS = 240
SEED = 20261128
TIMES = (999, 749, 499, 249)
MODES = ("late-absolute", "late-paired")
ARMS = ("shared-d2", *MODES, "late-paired-wrong")
R4 = [[0, 1, 2, 3]]
PAIR_WEIGHT = .50
OBJECT_WEIGHT = .35
PHOTO_WEIGHT = .25
TRUST_WEIGHT = .05
TEMPERATURE = .07


class LateCondition(nn.Module):
    def __init__(self, width=1024, rank=128, tokens=4, channels=2048):
        super().__init__()
        self.token_shape = (tokens, channels)
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank)
        self.outputs = nn.ModuleList(nn.Linear(rank, tokens * channels, bias=False) for _ in TIMES[2:])
        for output in self.outputs:
            nn.init.zeros_(output.weight)

    def forward(self, hidden, base):
        features = F.silu(self.down(self.norm(hidden.float())))
        proposals = torch.stack([output(features).reshape(-1, *self.token_shape) for output in self.outputs])
        strength = proposals.flatten(2).norm(dim=-1)
        limit = .60 * base.float().flatten(1).norm(dim=-1).clamp_min(1e-4)
        return proposals / (strength / limit).clamp_min(1.)[..., None, None]


def make_step_tokens(base, residuals, selected=(True, True)):
    late = tuple((base.float() + delta).to(base.dtype) if active else base
                 for delta, active in zip(residuals, selected, strict=True))
    return (base, base, *late)


def response_gaps(correct, swapped, teacher_correct, teacher_swapped):
    correct, swapped = correct.float().flatten(1), swapped.float().flatten(1)
    own, other = teacher_correct.float().flatten(1), teacher_swapped.float().flatten(1)
    difference = own - other
    scale = difference.square().mean(1).clamp_min(.05 ** 2)
    absolute = ((correct - own).square().mean(1) / scale).mean()
    paired = (((correct - swapped) - difference).square().mean(1) / scale).mean()
    return absolute, paired


def pooled_hidden(hidden):
    grouped, individual = hidden
    return .72 * grouped + .28 * individual.reshape(len(grouped), 4, -1).mean(1)


class MeasuredUNet:
    """Intervene on the condition at the student's actual state, keeping the state fixed."""

    def __init__(self, unet, teacher, wrong):
        from paper_rgb_object import CheckpointedUNet

        self.unet = unet
        self.student = CheckpointedUNet(unet)
        self.dtype = unet.dtype
        self.teacher = teacher
        self.wrong = wrong
        self.steps = 0
        self.losses = []

    def __call__(self, state, time, *, encoder_hidden_states, added_cond_kwargs, cross_attention_kwargs):
        step = self.steps
        self.steps += 1
        prompts = dict(encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
        if step < 2:
            return self.unet(state, time, **prompts, cross_attention_kwargs=cross_attention_kwargs)
        correct = self.student(state, time, **prompts, cross_attention_kwargs=cross_attention_kwargs)
        swapped = self.student(state.detach(), time, **prompts,
                               cross_attention_kwargs={"ip_hidden_states": self.wrong[step]})
        with torch.no_grad():
            twice = lambda value: torch.cat((value, value))
            teachers = self.unet(twice(state.detach()), twice(time), twice(encoder_hidden_states),
                                 added_cond_kwargs={key: twice(value) for key, value in added_cond_kwargs.items()},
                                 cross_attention_kwargs={"ip_hidden_states": self.teacher}).sample
        self.losses.append(response_gaps(correct.sample, swapped.sample, *teachers.chunk(2)))
        return correct


def sources():
    from paper_identity_render import load_generator
    from paper_render import draw_images, render_code

    files = PinnedFiles()
    code = render_code(files, "d2")
    for origin in (Path(__file__), *(Path(inspect.getfile(fn)) for fn in (image_tokens, load_generator, draw_images))):
        code[origin.name] = files.pin(origin)
    return files, code


def training_assets(prep, rows, labels, dino_path, source, files):
    from paper_dino import MODEL_FILES
    from paper_identity_bank import visual_model

    targets = torch.load(source, map_location="cpu", weights_only=True)
    weights = {name: files.pin(dino_path / name) for name in MODEL_FILES}
    require(weights == targets["proof"]["dino"] == MODEL_FILES
            and targets["proof"]["split"] == files.pin(prep / "paper_split.json")
            and np.array_equal(targets["rows"].numpy(), rows)
            and np.array_equal(targets["concepts"][targets["labels"]].numpy(), labels), "Wrong visual training registry")
    features = np.load(prep / "F_train.npy", mmap_mode="r", allow_pickle=False)
    latent = np.load(prep / "lat512_train.npy", mmap_mode="r", allow_pickle=False)
    coarse = torch.cat([F.avg_pool2d(torch.from_numpy(latent[group].copy()).cuda().float(), 8).flatten(1).cpu()
                        for group in np.array_split(rows, (len(rows) + 127) // 128)])
    groups = np.argsort(labels, kind="stable").reshape(-1, 10)
    class_layout = coarse[groups].mean(1)
    return visual_model(dino_path), torch.from_numpy(features[rows].copy()).cuda(), class_layout, groups, targets


def write_checkpoint(path, state):
    atomic_file(path, lambda stream: torch.save(state, stream))


def fit(arm, output, bank, steps):
    from common import PREP, ROOT, RUNS
    from exp0_intra10_swap_gate import get_concept_splits
    from exp6_draft_intra4 import DRAFT_DIR
    from paper_identity_render import load_generator
    from paper_oof_conditions import registered_rows
    from paper_render import generator_inputs, physical_eeg, rollout_4step_step_ips
    from paper_rgb_object import dino_cls

    require(RUNS.name == "shared", "Use the registered shared D2 fit")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    files, code = sources()
    rows, labels, _ = registered_rows(PREP)
    require(np.array_equal(rows, get_concept_splits()[2]), "Fold-out layouts cover a different training split")
    encoder, baseline, cone, d2_proof = d2_models(files)
    eeg = [physical_eeg(subject, "train", files) for subject in SUBJECTS]
    oof_path = DRAFT_DIR / "z0_oof_tr14_256.pt"
    pins = (bank, oof_path, PREP / "F_train.npy", PREP / "lat512_train.npy", PREP / "paper_split.json")
    proof = dict(arm=arm, steps=steps, seed=SEED, d2=d2_proof, generators=generator_inputs(files), code=code,
                 eeg=[item for _, item in eeg], inputs={path.name: files.pin(path) for path in pins},
                 pair_weight=PAIR_WEIGHT if arm == "late-paired" else 0., object_weight=OBJECT_WEIGHT,
                 photo_weight=PHOTO_WEIGHT, trust_weight=TRUST_WEIGHT)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{arm}.pt"
    if destination.is_file():
        final = torch.load(destination, map_location="cpu", weights_only=True)
        require(final["proof"] == proof and len(final["history"]) == steps, "Existing fit uses different inputs")
        save_json(destination.with_suffix(".json"), dict(final["report"], checkpoint_sha256=files.pin(destination)))
        return
    dino, visual, class_layout, groups, teacher = training_assets(
        PREP, rows, labels, ROOT / "models/paper-dinov2-small", bank, files)
    true_photo, true_label, sums = (teacher[key].cuda() for key in ("image_features", "labels", "sums"))
    training = make_episodes(groups, class_layout, steps=steps)
    order_sha = hashlib.sha256(b"".join(value.astype("<i8").tobytes() for value in training)).hexdigest()
    oof = torch.load(oof_path, map_location="cpu", weights_only=True)
    require(oof["Z_oof"].shape == (10, len(rows), 256), "Wrong fold-out layout inventory")
    basis, mean = oof["V_pca"].cuda().float(), oof["my"].cuda().float()
    pipe, adapter, projection, prompts, vae = load_generator(torch.bfloat16, True)
    require(tuple(pipe.scheduler.timesteps.tolist()) == TIMES, "Wrong registered denoising steps")
    model = LateCondition().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(.9, .995), weight_decay=.04)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    history = []
    partial = destination.with_suffix(".progress.pt")
    if partial.is_file():
        state = torch.load(partial, map_location="cpu", weights_only=True)
        require(state["proof"] == proof and state["training_order_sha256"] == order_sha,
                "Interrupted fit uses different sources or training order")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        history = state["history"]
    donor = list(DONOR)
    for index in range(len(history), steps):
        sid, selected, seed = (value[index] for value in training)
        local = torch.from_numpy(selected).cuda()
        with torch.no_grad():
            raw = torch.from_numpy(eeg[sid][0][rows[selected]].copy()).cuda().float()
            encoded = repeat_hidden(encoder, raw, SUBJECTS[sid])
            hidden = pooled_hidden(encoded)
            base = image_tokens(encoder.head_f, baseline, encoded, cone, adapter, projection)
            photo = projection(adapter(visual[local].to(torch.bfloat16)))
            teachers = torch.cat((photo, photo[donor]))
            layout = (oof["Z_oof"][sid, selected].cuda().float() @ basis.T + mean).reshape(4, 4, 64, 64)
            noise = torch.randn(1, 5, 4, 64, 64, generator=torch.Generator().manual_seed(int(seed))).cuda()
            initial, *additional = noise.expand(len(selected), -1, -1, -1, -1).unbind(1)
        delta = model(hidden, base)
        measured = MeasuredUNet(pipe.unet, teachers, make_step_tokens(base, delta[:, donor]))
        latent = rollout_4step_step_ips(measured, pipe.scheduler, make_step_tokens(base, delta), layout,
                                       initial, additional, *prompts, LAYOUT_STRENGTH, LAYOUT_FILTER)
        require(measured.steps == 4 and len(measured.losses) == 2, "Did not supervise both final steps")
        absolute, paired = (torch.stack(values).mean() for values in zip(*measured.losses, strict=True))
        rgb = checkpoint(lambda z: vae.decode(z).sample, latent.float() / vae.config.scaling_factor,
                         use_reentrant=False, preserve_rng_state=False)
        pred = checkpoint(lambda pixels: dino_cls(dino, pixels), (rgb * .5 + .5).clamp(0, 1),
                          use_reentrant=False, preserve_rng_state=False)
        names = true_label[local]
        prototypes = F.normalize(sums.index_add(0, names, -true_photo[local]))
        identity = F.cross_entropy(pred @ prototypes.T / TEMPERATURE, names)
        matched = photo_loss(pred @ true_photo[local].T)
        trust = (delta.square().mean((0, 2, 3)) / base.float().square().mean((1, 2)).clamp_min(1e-6)).mean()
        loss = (absolute + proof["pair_weight"] * paired + OBJECT_WEIGHT * identity
                + PHOTO_WEIGHT * matched + TRUST_WEIGHT * trust)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        require(torch.isfinite(loss) and gradient > 0, "No finite gradient reached the late EEG decoder")
        optimizer.step()
        schedule.step()
        record = dict(loss=float(loss.detach()), absolute=float(absolute.detach()), paired=float(paired.detach()),
                      identity=float(identity.detach()), photo=float(matched.detach()), gradient=float(gradient),
                      trust=float(trust.detach()))
        history.append(record)
        step = index + 1
        if step == 1 or step % 10 == 0 or step == steps:
            print(arm, step, steps, record, flush=True)
        if step % 40 == 0 and step < steps:
            files.verify()
            write_checkpoint(partial, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                           schedule=schedule.state_dict(), training_order_sha256=order_sha,
                                           history=history, proof=proof))
    files.verify()
    report = dict(training_order_sha256=order_sha, parameters=sum(value.numel() for value in model.parameters()),
                  proof=proof, history=history, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    write_checkpoint(destination, dict(model=model.cpu().state_dict(), history=history, proof=proof, report=report))
    save_json(destination.with_suffix(".json"), dict(report, checkpoint_sha256=files.pin(destination)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=MODES, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(2, STEPS), default=STEPS)
    args = parser.parse_args()
    fit(args.arm, args.output, args.bank, args.steps)
