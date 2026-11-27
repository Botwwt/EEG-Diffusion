"""Learn an EEG-conditional flow over the relative image tokens used by Brain-HiVE."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SEED = 20261127
STEPS = 3000
BATCH_PAIRS = 128
FLOW_STEPS = 8
FIT_ARMS = ("flow", "paired-flow")
ARMS = ("native-paper", *FIT_ARMS, "paired-flow-wrong")
BUDGETS = ("dev-r4", "official-r4", "official-r80")


def paired_path(noise, target, times):
    noise = noise[:, None]
    times = times[:, None, None]
    return torch.lerp(noise, target, times), target - noise


def pair_penalty(matched, crossed, target, temperature=.15):
    own = (matched - target).square().mean(-1).sum(1)
    other = (crossed - target).square().mean(-1).sum(1)
    return F.softplus((own - other) / (2 * temperature))


@torch.no_grad()
def choose_confusers(visual, labels, count=12):
    x = F.normalize(visual)
    blocks = []
    for start in range(0, len(x), 256):
        scores = x[start:start+256] @ x.T
        scores.masked_fill_(labels[start:start+256, None] == labels, float("-inf"))
        values, indices = scores.topk(count)
        if not torch.isfinite(values).all():
            raise ValueError("Each photograph needs enough other-object training photographs")
        blocks.append(indices)
    return torch.cat(blocks)


def solve_flow(field, state, condition, steps=FLOW_STEPS):
    """Heun integration for a single token sample per query."""
    dt = 1. / steps
    times = state.new_empty(len(state))
    for i in range(steps):
        first = field(state, times.fill_(i * dt), condition)
        second = field(state + dt * first, times.fill_((i + 1) * dt), condition)
        state = state + .5 * dt * (first + second)
    return state


class TokenFlow(nn.Module):
    def __init__(self, coordinates, condition_dim):
        super().__init__()
        hidden = 1024
        self.input = nn.Linear(coordinates + condition_dim + 4, hidden)
        self.residual = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, coordinates))

    def forward(self, state, time, condition):
        t = time[:, None]
        location = torch.cat((t, t.square(), torch.sin(torch.pi * t), torch.cos(torch.pi * t)), dim=1)
        features = self.input(torch.cat((state, condition, location), dim=1))
        return self.output(features + self.residual(features)).float()


def output_root():
    return Path(os.environ["EEG_CROSSFIT_ROOT"])


def training_inputs(files):
    from common import PREP, ROOT
    from paper_oof_conditions import SOURCES, registered_rows
    from paper_render_cache import require

    rows, objects, folds = registered_rows(PREP)
    target_file = ROOT / "direct_token_encoder/training-targets.pt"
    photo_file = ROOT / "crossfit_generator/object-identity/training-photo-dino.pt"
    proof = dict(target=files.pin(target_file), photo_teacher=files.pin(photo_file),
                 train_split=files.pin(PREP / "paper_split.json"), code=files.pin(__file__),
                 conditions={})
    target = torch.load(target_file, weights_only=True, map_location="cpu")
    teacher = torch.load(photo_file, weights_only=True, map_location="cpu")
    require(np.array_equal(target["rows"].numpy(), rows) and np.array_equal(teacher["rows"].numpy(), rows)
            and np.array_equal(teacher["concepts"][teacher["labels"]], objects)
            and target["sources"]["split"] == teacher["proof"]["split"] == proof["train_split"],
            "The token targets and DINO teacher must have the exact registered training photos")
    predicts = []
    for subject in SOURCES:
        path = ROOT / "crossfit_generator" / f"conditions-sub-{subject:02d}.pt"
        proof["conditions"][str(subject)] = files.pin(path)
        packet = torch.load(path, weights_only=True, map_location="cpu")
        require(np.array_equal(packet["rows"].numpy(), rows) and np.array_equal(packet["folds"].numpy(), folds)
                and packet["sources"]["subject"] == subject and packet["sources"]["split"] == proof["train_split"]
                and packet["sources"]["images"] == target["sources"]["photos"]
                and [fold["fold"] for fold in packet["split_audits"]] == [1, 2, 3, 4],
                "EEG conditions must come from encoders trained outside the corresponding object fold")
        predicts.append(packet["out_of_fold"].float())
    conditions = torch.stack(predicts).cuda()
    coords = target["coordinates"].cuda().float()
    dino = teacher["image_features"].cuda().float()
    require(conditions.shape == (4, len(rows), 1024) and coords.shape == (len(rows), 512)
            and dino.shape == (len(rows), 384) and all(torch.isfinite(x).all() for x in (conditions, coords, dino)),
            "Unexpected nonfinite or unaligned training inputs")
    return conditions, coords, dino, teacher["labels"].cuda(), proof


def fit():
    from paper_render_cache import PinnedFiles, atomic_file, require, save_json
    from paper_token_evidence import ridge_weights

    files = PinnedFiles()
    conditions, target, dino, labels, proof = training_inputs(files)
    center = conditions.mean((0, 1))
    scale = conditions.flatten(0, 1).std(0, unbiased=False).clamp_min(.05)
    conditions = ((conditions - center) / scale).clamp(-6, 6)
    confusers = choose_confusers(dino, labels).cpu().numpy()
    guidance = (dino - dino.mean(0)) / dino.std(0, unbiased=False).clamp_min(.02)
    matrix = ridge_weights(target, guidance, penalty=.15)
    paths = output_root()
    paths.mkdir(parents=True, exist_ok=True)
    for arm in FIT_ARMS:
        path = paths / f"{arm}.pt"
        if path.exists():
            cached = torch.load(path, weights_only=True, map_location="cpu")
            require(cached["proof"] == proof and cached["arm"] == arm and cached["steps"] == STEPS,
                    "An existing flow uses a different training registration")
            continue
        torch.manual_seed(SEED)
        generator = torch.Generator(device="cuda").manual_seed(SEED + 1)
        random_pairs = np.random.default_rng(SEED + 2)
        model = TokenFlow(target.shape[1], conditions.shape[2]).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, betas=(.9, .99), weight_decay=.02)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=3e-5)
        batch_digest = hashlib.sha256()
        history = []
        metrics = target.new_zeros(4)
        pair_weight = .4 if arm == "paired-flow" else 0.
        for step in range(1, STEPS + 1):
            first = random_pairs.integers(len(target), size=BATCH_PAIRS)
            second = confusers[first, random_pairs.integers(confusers.shape[1], size=BATCH_PAIRS)]
            people = random_pairs.integers(len(conditions), size=BATCH_PAIRS)
            ids = np.stack((first, second), axis=1)
            batch_digest.update(np.stack((people, first, second)).astype("<i8").tobytes())
            photo = target[ids]
            condition = conditions[people[:, None], ids]
            noise = torch.randn(BATCH_PAIRS, target.shape[1], device="cuda", generator=generator)
            time = torch.rand(BATCH_PAIRS, device="cuda", generator=generator).square()
            state, speed = paired_path(noise, photo, time)
            inputs = torch.cat((condition, condition.flip(1))).flatten(0, 1)
            location = time.repeat_interleave(2).repeat(2)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                normal, crossed = model(state.flatten(0, 1).repeat(2, 1), location, inputs).reshape(-1, 2, target.shape[1]).chunk(2)
            remaining = 1 - time[:, None, None]
            own = (state + remaining * normal) @ matrix
            wrong = (state + remaining * crossed) @ matrix
            exact = guidance[ids]
            velocity_loss = F.mse_loss(normal, speed)
            object_loss = F.mse_loss(own, exact)
            pair_loss = ((1. - time) * pair_penalty(own, wrong, exact)).mean()
            loss = velocity_loss + .15 * object_loss + pair_weight * pair_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            with torch.no_grad():
                wins = (((own - exact).square().sum((1, 2))) < ((wrong - exact).square().sum((1, 2)))).float().mean()
                metrics += torch.stack((velocity_loss, object_loss, pair_loss, wins))
            if step % 250 == 0:
                log = dict(step=step, velocity=float(metrics[0]/250), object=float(metrics[1]/250),
                           pair=float(metrics[2]/250), ordered_fraction=float(metrics[3]/250))
                history.append(log)
                metrics.zero_()
                print("Token flow training", arm, log, flush=True)
        model = model.cpu()
        payload = dict(arm=arm, proof=proof, steps=STEPS, batch_sha256=batch_digest.hexdigest(),
                       pair_weight=pair_weight, history=history, center=center.cpu(), scale=scale.cpu(),
                       weights=model.state_dict())
        files.verify()
        atomic_file(path, lambda stream: torch.save(payload, stream))
        summary = {key: payload[key] for key in ("arm", "proof", "steps", "batch_sha256", "pair_weight", "history")}
        save_json(path.with_suffix(".json"), dict(summary, checkpoint=files.pin(path),
                  parameters=sum(p.numel() for p in model.parameters()), training_queries=conditions.shape[0]*conditions.shape[1]))


def seeded_initial(seeds, width, device):
    rows = []
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed ^ 0x31415D88)
        rows.append(torch.randn(width, generator=generator))
    return torch.stack(rows).to(device)


@torch.inference_mode()
def infer(subject, budgets, render_budgets):
    from common import PREP, ROOT
    from paper_direct_tokens import restore_relative
    from paper_identity_render import load_generator
    from paper_object_foreground import predict_at_budget, text_targets
    from paper_object_foreground_render import split_input
    from paper_oof_conditions import fitted_native
    from paper_render import generator_inputs, groups, query_seed, tensor_fingerprint
    from paper_render_cache import PinnedFiles, atomic_file, require
    from paper_repeat_semantic_bridge import fit_head, predict
    from paper_soft_text import wrong_donors
    from paper_token_evidence import mix_tokens, project_tokens, render_arms, token_parts

    files = PinnedFiles()
    rows, labels, _, text, states, basis_hash = text_targets(files)
    target_path = ROOT / "direct_token_encoder/training-targets.pt"
    target_hash = files.pin(target_path)
    target = torch.load(target_path, weights_only=True, map_location="cuda")
    coords, basis = target["coordinates"], target["basis"]
    require(np.array_equal(target["rows"].cpu().numpy(), rows), "The language diagnostic uses another training split")
    mapper = fit_head(coords, text.cuda(), labels.cuda())
    encoder, checkpoint = fitted_native(subject, "cuda")
    native_hash = files.pin(checkpoint)
    fits = {}
    models = {}
    for arm in FIT_ARMS:
        path = output_root() / f"{arm}.pt"
        fits[arm] = files.pin(path)
        packet = torch.load(path, weights_only=True, map_location="cuda")
        require(packet["arm"] == arm and packet["proof"]["target"] == target_hash and packet["steps"] == STEPS,
                "Unexpected token flow checkpoint")
        model = TokenFlow(coords.shape[1], len(packet["scale"])).cuda().eval()
        model.load_state_dict(packet["weights"])
        models[arm] = (model, packet["center"], packet["scale"], packet["batch_sha256"])
    require(models["flow"][3] == models["paired-flow"][3], "The paired comparison needs identical training batches")
    assets = generator_inputs(files)
    pipeline = load_generator(torch.float16, False)
    for budget in budgets:
        raw, receipt, query_rows = split_input(subject, budget, files)
        repetitions = groups(budget)
        features = predict_at_budget(encoder.backbone, raw, repetitions)
        tokens = project_tokens(features, pipeline[1], pipeline[2])
        global_token, relative = token_parts(tokens)
        panel = budget.split("-")[0]
        seeds = [query_seed(panel, row) for row in range(len(query_rows))]
        noise = seeded_initial(seeds, coords.shape[1], features.device)
        donors = wrong_donors(len(query_rows)).tolist()
        changes = {"native-paper": relative}
        for arm, (model, mean, std, _) in models.items():
            condition = ((features - mean) / std).clamp(-6, 6)
            prediction = solve_flow(model, noise, condition)
            changes[arm] = restore_relative(prediction, basis, calibrate=True)
            if arm == "paired-flow":
                wrong = solve_flow(model, noise, condition[donors])
                changes["paired-flow-wrong"] = restore_relative(wrong, basis, calibrate=True)
        codes = {arm: ((delta - basis["center"]).flatten(1) @ basis["axes"]) / basis["scale"]
                 for arm, delta in changes.items()}
        pooled = {arm: states["mean_pooled"] + predict(mapper, code).cpu() @ states["pooled_basis"]
                  for arm, code in codes.items()}
        dest = output_root() / "images" / budget / f"sub-{subject:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        proof = dict(subject=subject, budget=budget, repeats=repetitions, physical_eeg=receipt,
                     native_checkpoint_sha256=native_hash, flow_checkpoints=fits, training_target_sha256=target_hash,
                     semantic_basis_sha256=basis_hash, code_sha256=files.pin(__file__), generator_inputs=assets,
                     latent_solver_evaluations=2 * FLOW_STEPS, latent_initial_sha256=tensor_fingerprint(noise.cpu()))
        files.verify()
        atomic_file(dest / "semantic-readout.pt", lambda stream:
                    torch.save(dict(proof=dict(proof, row=query_rows), pool=pooled, donors=donors), stream))
        print("Paired token flow readout", subject, budget, len(query_rows), flush=True)
        if budget in render_budgets:
            variants = []
            own = list(range(len(query_rows)))
            for arm, delta in changes.items():
                actual = tokens.half() if arm == "native-paper" else mix_tokens(global_token, delta, torch.float16)
                donor = donors if arm == "paired-flow-wrong" else own
                variants.append((arm, actual.cpu(), dict(image_token_sha256=tensor_fingerprint(actual.cpu()),
                                 mean_eeg_donor=own, relative_eeg_donor=donor)))
            render_arms(dest, variants, query_rows, seeds, proof, files, pipeline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("fit")
    for name in ("infer", "score"):
        stage = commands.add_parser(name)
        stage.add_argument("subject", type=int, choices=range(1, 11))
        stage.add_argument("--budgets", nargs="+", choices=BUDGETS, default=BUDGETS)
        stage.add_argument("--render-budgets", nargs="+", choices=BUDGETS, default=[])
    args = parser.parse_args()
    if args.phase == "fit":
        fit()
    else:
        if not set(args.render_budgets).issubset(args.budgets):
            parser.error("Render budgets must also be included in the EEG readout")
        if args.phase == "infer":
            infer(args.subject, args.budgets, args.render_budgets)
        else:
            from paper_object_foreground_render import score

            score(args.subject, args.budgets, args.render_budgets, ARMS)
