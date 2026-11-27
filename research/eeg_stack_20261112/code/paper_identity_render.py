"""Supervise the object recognized in the final RGB image produced from real EEG."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_identity_bank import bank_root, object_logits, visual_model
from paper_oof_conditions import HELD_OUT, SEED, SOURCES, fitted_native, native_features, registered_rows, require, study_root


STEPS, BATCH = 512, 2
FIT_MODES = ("photo", "identity")
ARMS = (*FIT_MODES, "identity-wrong")
TAU = .07


class ConditionOffset(nn.Module):
    def __init__(self, center, scale):
        super().__init__()
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.offset = nn.Linear(center.numel(), center.numel())
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x):
        centered = x - self.center
        delta = self.scale * self.offset(centered / self.scale)
        limit = (.5 * centered.norm(dim=1, keepdim=True)).clamp_min(1e-4)
        return x + delta / (delta.norm(dim=1, keepdim=True) / limit).clamp_min(1.)


def load_generator(dtype, training):
    from common import TURBO
    from diffusers import AutoencoderKL
    from paper_training import generator

    pipe, adapter, project, prompt, pooled, dims = generator(dtype, training=training)
    pipe.vae = None
    vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16",
                                        torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    vae.enable_slicing()
    return pipe, adapter, project, (prompt, pooled, dims), vae


def schedule(n):
    require(n >= STEPS // 2, "The source participants need enough distinct training photographs")
    rng = np.random.default_rng(SEED)
    selected = [rng.permutation(n)[:STEPS // 2] for _ in SOURCES]
    source = np.arange(STEPS * BATCH).reshape(STEPS, BATCH) % len(SOURCES)
    rows = np.stack(selected, axis=1).reshape(STEPS, BATCH)
    seeds = rng.integers(0, 2**63 - 1, size=(STEPS, BATCH), dtype=np.int64)
    return rows, source, seeds


def fit(mode):
    from common import PREP, ROOT
    from paper_dino import MODEL_FILES
    from paper_render import generator_inputs, rollout_4step_step_ips
    from paper_render_cache import PinnedFiles, atomic_file, save_json
    from paper_rgb_object import dino_cls, render_rgb

    root = bank_root()
    path = root / "training-photo-dino.pt"
    conditions = [study_root() / f"conditions-sub-{s:02d}.pt" for s in SOURCES]
    pins = PinnedFiles()
    split_hash = pins.pin(PREP / "paper_split.json")
    dino_path = ROOT / "models/paper-dinov2-small"
    proof = dict(training_photos=pins.pin(path), split=split_hash, generator=generator_inputs(pins),
                 conditions=[pins.pin(p) for p in conditions], code=pins.pin(Path(__file__)),
                 renderer_code=pins.pin(Path(__file__).with_name("paper_rgb_object.py")),
                 dino={key: pins.pin(dino_path / key) for key in MODEL_FILES})
    dest = root / f"condition-{mode}.pt"
    if dest.is_file():
        pack = torch.load(dest, map_location="cpu", weights_only=True)
        require(pack["proof"] == proof and pack["mode"] == mode, "Existing object fit used different supervision")
        return
    rows, _, folds = registered_rows(PREP)
    queries = []
    for subject, source in zip(SOURCES, conditions, strict=True):
        data = torch.load(source, map_location="cpu", weights_only=True)
        require(data["sources"]["split"] == split_hash and data["sources"]["subject"] == subject
                and np.array_equal(data["rows"].numpy(), rows) and np.array_equal(data["folds"].numpy(), folds),
                "Out-of-concept training EEG rows differ")
        queries.append(data["out_of_fold"])
    eeg = torch.stack(queries, dim=1).cuda().float()
    target = torch.load(path, map_location="cpu", weights_only=True)
    require(np.array_equal(target["rows"].numpy(), rows) and target["proof"]["split"] == split_hash
            and target["proof"]["dino"] == proof["dino"] == MODEL_FILES,
            "Object teacher saw a different image index")
    labels, photos, sums = (target[key].cuda() for key in ("labels", "image_features", "sums"))
    require(tuple(photos.shape) == (len(rows), 384) and tuple(sums.shape) == (1323, 384),
            "Unexpected source object recognition dimensions")
    dino = visual_model(dino_path)
    pipe, adapter, project, prompts, vae = load_generator(torch.float16, True)
    x = eeg.flatten(0, 1)
    model = ConditionOffset(x.mean(0), x.std(0).clamp_min(1e-4)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(.9, .995), weight_decay=.05)
    shrink = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-5)
    plan, source_ids, seeds = schedule(len(rows))
    outcomes = []
    for step, (indices, subjects, noise_seeds) in enumerate(zip(plan, source_ids, seeds, strict=True), 1):
        current = eeg[indices, subjects]
        adjusted = model(current)
        tokens = project(adapter(adjusted.half()))
        noise = torch.stack([torch.randn(5, 4, 64, 64, generator=torch.Generator().manual_seed(int(seed)))
                             for seed in noise_seeds]).cuda()
        image = render_rgb(pipe.unet, pipe.scheduler, vae, prompts, tokens, noise, rollout_4step_step_ips)
        views = [dino_cls(dino, sample) for sample in (image, image[:, :, 32:-32, 32:-32])]
        recog = torch.stack([object_logits(v, photos[indices], labels[indices], sums, mode) for v in views])
        targets = labels[indices]
        objective = F.cross_entropy(recog.flatten(0, 1) / TAU, targets.repeat(len(views)))
        delta = ((adjusted - current) / model.scale).square().mean()
        loss = objective + .10 * delta
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        shrink.step()
        outcomes.append([float(objective.detach()), float(delta.detach()),
                         int((recog.mean(0).argmax(1) == targets).sum())])
        if step == 1 or step % 16 == 0:
            recent = np.asarray(outcomes[-16:])
            print("visible object", mode, step, recent.mean(0).tolist(), flush=True)
    plan_sha = hashlib.sha256(plan.astype("<i8").tobytes() + source_ids.astype("<i8").tobytes()
                              + seeds.astype("<i8").tobytes()).hexdigest()
    pack = dict(mode=mode, proof=proof, source_subjects=list(SOURCES), steps=STEPS, batch=BATCH, schedule_sha256=plan_sha,
                training_outcomes=outcomes, state={key: value.cpu() for key, value in model.state_dict().items()})
    pins.verify()
    atomic_file(dest, lambda stream: torch.save(pack, stream))
    save_json(dest.with_suffix(".json"), dict(mode=mode, proof=proof, source_subjects=list(SOURCES), steps=STEPS,
              batch=BATCH, schedule_sha256=plan_sha, training_teacher=target["recognition"], final_record=outcomes[-1]))


@torch.inference_mode()
def render(subject, panel, arms):
    from common import PREP
    from paper_render import draw_images, generator_inputs, physical_eeg, query_seed
    from paper_render_cache import PinnedFiles, render_transaction
    from paper_score import score

    pins = PinnedFiles()
    split_path = PREP / "paper_split.json"
    split = pins.pin(split_path)
    rows = json.loads(split_path.read_bytes())[f"{panel}_rows"]
    raw, physical = physical_eeg(subject, "test" if panel == "official" else "train", pins)
    require(len(set(rows)) == len(rows) == (200 if panel == "official" else 500), "Invalid registered query panel")
    if panel == "fresh":
        train, _, _ = registered_rows(PREP)
        require(not np.intersect1d(rows, train).size, "Fresh queries overlap training")
    repeats = [0, 1, 2, 3]
    native, native_path = fitted_native(subject, "cuda")
    native_hash = pins.pin(native_path)
    mean_eeg = torch.from_numpy(raw[rows][:, repeats].copy()).cuda().float().mean(1)
    conditions = native_features(native, mean_eeg).cuda()
    del native, mean_eeg
    assets = generator_inputs(pins)
    pipe, adapter, project, prompts, vae = load_generator(torch.float16, False)
    seeds = [query_seed(panel, i) for i in range(len(rows))]
    outputs = []
    for arm in arms:
        path = bank_root() / f"condition-{arm.removesuffix('-wrong')}.pt"
        pin = pins.pin(path)
        learned = torch.load(path, map_location="cpu", weights_only=True)
        require(learned["proof"]["split"] == split and learned["proof"]["generator"] == assets
                and set(learned["source_subjects"]).isdisjoint(HELD_OUT), "Checkpoint and inference registration differ")
        w = learned["state"]
        correct = ConditionOffset(w["center"], w["scale"])
        correct.load_state_dict(w)
        adjusted = correct.cuda()(conditions)
        tokens = torch.cat([project(adapter(x.half())).cpu() for x in adjusted.split(20)])
        donors = np.arange(len(rows))
        if arm.endswith("-wrong"):
            donors = np.roll(donors, len(rows) // 2)
        dest = bank_root() / "images" / arm / f"{panel}-r4" / f"sub-{subject:02d}"
        setup = dict(render_cache_version=2, arm=arm, subject=subject, n=len(rows), training="shared-image-identity",
                     budget=f"{panel}-r4", row=rows, repeats=[repeats], eeg_donor=donors.tolist(), seeds=seeds,
                     draws=1, unet_calls_per_draw=4, prompt="", split_sha256=split, physical_eeg=physical,
                     native_checkpoint_sha256=native_hash, checkpoint_sha256=pin, generator_files_sha256=assets,
                     render_source_sha256=pins.pin(Path(__file__)))

        def generate(work, starts, complete):
            draw_images(work, pipe, vae, prompts, tokens[donors], seeds, batch_starts=starts, batch_complete=complete)

        render_transaction(dest, setup, generate, pins.verify)
        print("identity render ready", subject, panel, arm, flush=True)
        outputs.append(dest)
    del pipe, adapter, project, vae
    torch.cuda.empty_cache()
    for dest in outputs:
        if not (dest / "scores.json").is_file():
            score(dest, panel, Path(os.environ["ARCHIVE"]), dest)


def report(panel):
    from paper_aggregate import bootstrap_ci, units
    from paper_render_cache import atomic_file, save_json, sha

    root = bank_root()
    fits = {mode: torch.load(root / f"condition-{mode}.pt", map_location="cpu", weights_only=True) for mode in FIT_MODES}
    hashes = {mode: sha(root / f"condition-{mode}.pt") for mode in FIT_MODES}
    require(fits["photo"]["proof"] == fits["identity"]["proof"]
            and fits["photo"]["schedule_sha256"] == fits["identity"]["schedule_sha256"], "Image identity loss controls differ")
    fields = ("generated_name1854", "clip_image_top1", "ssim") + (("clip_within10",) if panel == "fresh" else ())
    values, sources = {arm: [] for arm in ("native", *ARMS)}, {}
    classes, rows = None, None
    for person in HELD_OUT:
        for arm in values:
            parent = study_root() if arm == "native" else root
            path = parent / "images" / arm / f"{panel}-r4" / f"sub-{person:02d}"
            meta = json.loads((path / "render.json").read_bytes())
            score = json.loads((path / "scores.json").read_bytes())
            require(score["render_sha256"] == sha(path / "render.json")
                    and score["observations_sha256"] == sha(path / "observations.npz"), "Image scoring receipt differs")
            if arm == "native":
                base = meta
            else:
                keys = ("n", "budget", "row", "seeds", "prompt", "draws", "unet_calls_per_draw", "repeats",
                        "split_sha256", "native_checkpoint_sha256", "generator_files_sha256", "physical_eeg")
                require(all(meta[key] == base[key] for key in keys), "An identity arm used different EEG or image sampling")
                donors = np.roll(np.arange(meta["n"]), meta["n"] // 2) if arm.endswith("-wrong") else np.arange(meta["n"])
                require(meta["eeg_donor"] == donors.tolist() and meta["arm"] == arm
                        and meta["checkpoint_sha256"] == hashes[arm.removesuffix("-wrong")],
                        "Unexpected EEG assignment or model weights")
            with np.load(path / "observations.npz", allow_pickle=False) as o:
                if rows is None:
                    classes, rows = o["concept"], o["row"]
                require(np.array_equal(rows, o["row"]) and np.array_equal(classes, o["concept"])
                        and rows.tolist() == meta["row"], "A score file references different original photographs")
                values[arm].append(np.stack([o[key] for key in fields], -1))
            sources.setdefault(arm, {})[str(person)] = {"score": sha(path / "scores.json"),
                                                        "png_sequence": score["ordered_image_hashes_sha256"]}
    measures = {arm: np.stack(a) for arm, a in values.items()}
    paired = {}
    for other in ("native", "photo", "identity-wrong"):
        delta = measures["identity"] - measures[other]
        interval = bootstrap_ci(units(delta, classes, panel))
        paired[other] = {key: {"change": float(delta[:, :, i].mean()), "ci95": interval[:, i].tolist()}
                         for i, key in enumerate(fields)}
    bank = root / f"summary-{panel}.npz"
    atomic_file(bank, lambda f: np.savez_compressed(f, arms=np.array(list(measures)), metrics=np.array(fields),
                rows=rows, concepts=classes, subjects=np.array(HELD_OUT), values=np.stack(list(measures.values()))))
    save_json(root / f"summary-{panel}.json",
              dict(panel=panel, people=list(HELD_OUT), rows_per_person=len(rows), contrasts=paired, source_sha256=sources,
                   means={arm: dict(zip(fields, v.mean((0, 1)).tolist(), strict=True)) for arm, v in measures.items()},
                   observations_sha256=sha(bank)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_subparsers(dest="action", required=True)
    train = action.add_parser("fit")
    train.add_argument("mode", choices=FIT_MODES)
    draw = action.add_parser("render")
    draw.add_argument("--subject", type=int, choices=HELD_OUT, required=True)
    draw.add_argument("--panel", choices=("official", "fresh"), default="official")
    draw.add_argument("--arms", choices=ARMS, nargs="+", default=ARMS)
    score = action.add_parser("report")
    score.add_argument("panel", choices=("official", "fresh"))
    args = p.parse_args()
    if args.action == "fit":
        fit(args.mode)
    elif args.action == "render":
        render(args.subject, args.panel, args.arms)
    else:
        report(args.panel)
