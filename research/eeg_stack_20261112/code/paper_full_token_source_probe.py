"""Fit a direct full-token image prior and a covariance-matched Gaussian control."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from paper_aggregate import bootstrap_ci, units
from paper_clean_image_flow import EPOCHS, ODE_STEPS, SEED, SOURCE_COUNT, STEPS, Velocity
from paper_fusion_siglip_probe import train_coordinates
from paper_raw_semantic_readout import METRICS, measure, normalize
from paper_render_cache import PinnedFiles, atomic_file, require, save_json
from paper_tangent_calibration import read_inputs


PEOPLE = (1, 3)
TOKEN_SHAPE = (4, 2048)
VARIANCE_RIDGE = .02
INVERSE_RIDGE = .02


@torch.no_grad()
def projection(benchmark, features, files, device):
    from main.models_adapter import FusionEncoderModel, IPProjectionModel

    prior = benchmark / "inputs/fusion-prior-source"
    rank_meta = benchmark / "token-rank-check/token-rank.json"
    files.pin(rank_meta)
    ranked = json.loads(rank_meta.read_bytes())["design"]["generator_files_sha256"]
    pins = {f"prior/{component}/{name}": files.pin(prior / component / name)
            for component in ("fusion_encoder", "proj") for name in ("config.json", "diffusion_pytorch_model.safetensors")}
    require(all(ranked[k] == v for k, v in pins.items()), "The projector differs from the rank intervention")
    fused = FusionEncoderModel.from_pretrained(prior, subfolder="fusion_encoder", local_files_only=True)
    adapter = fused.adapter.to(device, dtype=torch.float16).eval()
    projector = IPProjectionModel.from_pretrained(prior, subfolder="proj", local_files_only=True)
    projector.to(device, dtype=torch.float16).eval()
    tokens = torch.cat([projector(adapter(x.half())).float() for x in features.split(256)])
    require(tuple(tokens.shape[1:]) == TOKEN_SHAPE and bool(tokens.isfinite().all()), "Invalid external image tokens")
    return tokens.flatten(1), pins


@torch.no_grad()
def charts(tokens, features):
    n, dim = tokens.shape
    mean = tokens.mean(0)
    centered = tokens - mean
    cov = centered.T @ centered / n
    average = cov.diagonal().mean()
    cov.diagonal().add_(VARIANCE_RIDGE * average)
    scale = torch.linalg.cholesky(cov)
    clean = torch.linalg.solve_triangular(scale, centered.T, upper=False).T.contiguous()
    gram = clean.T @ clean / n
    gram.diagonal().add_(INVERSE_RIDGE)
    feature_mean = features.mean(0)
    inverse = torch.linalg.solve(gram, clean.T @ (features - feature_mean) / n)
    require(clean.shape == (n, dim) and bool(torch.isfinite(inverse).all()), "Unstable full-token coordinate transform")
    return mean, scale, clean, inverse, feature_mean


def train_flow(clean, device):
    velocity = Velocity(clean.shape[-1]).to(device)
    opt = torch.optim.AdamW(velocity.parameters(), lr=6e-4, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=8e-5)
    for epoch in range(EPOCHS):
        batch_losses = []
        for rows in torch.randperm(len(clean), device=device).split(256):
            y = clean[rows]
            z = torch.randn_like(y)
            t = torch.rand((len(rows), 1), device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = velocity(torch.lerp(z, y, t), t)
            loss = F.mse_loss(prediction.float(), y - z)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(velocity.parameters(), 2., error_if_nonfinite=True)
            opt.step()
            batch_losses.append(loss.detach())
        schedule.step()
        print("Full-token flow epoch", epoch + 1, float(torch.stack(batch_losses).mean()), flush=True)
    return velocity.eval().requires_grad_(False)


def endpoint(velocity, source):
    if velocity is None:
        return source
    dt = 1. / ODE_STEPS
    value = source
    for k in range(ODE_STEPS):
        with torch.autocast(value.device.type, dtype=torch.bfloat16, enabled=value.is_cuda):
            update = velocity(value, value.new_full((len(value), 1), k * dt))
        value = value + dt * update.float()
    return value


def condition(velocity, query, starts, inverse, feature_mean):
    trials, dim = starts.shape
    radii = starts.norm(dim=-1, keepdim=True)
    retained = []
    for q in query.split(48):
        state = starts.expand(len(q), trials, dim).clone().requires_grad_()
        opt = torch.optim.Adam([state], lr=.12)

        def evaluate():
            z = F.normalize(state, dim=-1) * radii
            t = endpoint(velocity, z.flatten(0, 1)).reshape_as(z)
            predicted = F.normalize(t @ inverse + feature_mean, dim=-1)
            return t, (predicted * q[:, None]).sum(-1)

        for _ in range(STEPS):
            _, scores = evaluate()
            opt.zero_grad(set_to_none=True)
            (-scores.mean()).backward()
            opt.step()
        with torch.no_grad():
            values, scores = evaluate()
            retained.append(values[torch.arange(len(q), device=q.device), scores.argmax(1)])
    return torch.cat(retained)


def relative_magnitude(flat):
    tokens = flat.reshape(-1, *TOKEN_SHAPE)
    relative = tokens - tokens.mean(1, keepdim=True)
    return relative.square().sum(-1).mean().sqrt().item()


def run(benchmark, output, device):
    torch.manual_seed(SEED)
    files = PinnedFiles()
    source, train, dev, photo, _, dev_labels, text_rows, text, proof, audit = read_inputs(benchmark, files)
    _, queries, _, eeg_receipts = train_coordinates(source, train, dev, proof, audit, files, device)
    query = queries.flatten(0, 1)
    photo_features = torch.from_numpy(np.array(photo[np.r_[train, dev]], copy=True)).to(device)
    tokens, generator = projection(benchmark, photo_features, files, device)
    token_train, token_eval = tokens[:len(train)], tokens[len(train):]
    features = F.normalize(photo_features, dim=-1)
    mean, scale, clean, inverse, feature_mean = charts(token_train, features[:len(train)])
    held_coords = torch.linalg.solve_triangular(scale, (token_eval - mean).T, upper=False).T
    held_hat = F.normalize(held_coords @ inverse + feature_mean, dim=-1)
    held_vision = features[len(train):]
    inverse_cosine = float((held_hat * held_vision).sum(-1).mean())
    inverse_top1 = float(((held_hat @ held_vision.T).argmax(1) == torch.arange(len(dev), device=device)).float().mean())
    print("Held photo inverse", inverse_cosine, inverse_top1, flush=True)
    fit_start = time.perf_counter()
    velocity = train_flow(clean, device)
    if query.is_cuda:
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - fit_start
    starts = torch.randn((SOURCE_COUNT, len(mean)), device=device,
                        generator=torch.Generator(device=device).manual_seed(SEED + 1))
    donors = np.roll(np.arange(len(dev)), next(k for k in range(1, len(dev))
                     if not np.any(np.roll(dev_labels, k) == dev_labels)))
    estimates = {"native-query": query.reshape(10, len(dev), -1).cpu().numpy()}
    raw_inputs = {}
    times = {}
    magnitude = {"actual_training_tokens": relative_magnitude(token_train),
                 "held_photograph_tokens": relative_magnitude(token_eval)}
    for name, transport in (("full-token-gaussian", None), ("full-token-flow", velocity)):
        before = time.perf_counter()
        latent = condition(transport, query, starts, inverse, feature_mean)
        if query.is_cuda:
            torch.cuda.synchronize()
        times[name] = time.perf_counter() - before
        estimates[name] = (latent @ inverse + feature_mean).reshape(10, len(dev), -1).cpu().numpy()
        decoded = latent @ scale.T + mean
        require(bool(decoded.isfinite().all()), "Nonfinite full-token source result")
        magnitude[name] = relative_magnitude(decoded)
        selected = decoded.reshape(10, len(dev), *TOKEN_SHAPE)[[i - 1 for i in PEOPLE]].half().cpu().numpy()
        raw_inputs[name] = selected
        print(name, "optimization seconds", times[name], "relative RMS", magnitude[name], flush=True)
    raw_inputs["full-token-flow-wrong"] = raw_inputs["full-token-flow"][:, donors]
    estimates["full-token-flow-wrong"] = estimates["full-token-flow"][:, donors]
    bank = normalize(text)
    gallery = held_vision.cpu().numpy()
    rows = {name: measure(value, bank, gallery, text_rows) for name, value in estimates.items()}
    comparisons = {}
    for a, b in (("full-token-flow", "native-query"), ("full-token-flow", "full-token-gaussian"),
                 ("full-token-flow", "full-token-flow-wrong")):
        delta = rows[a] - rows[b]
        lo, hi = bootstrap_ci(units(delta, dev_labels, "dev"))
        comparisons[f"{a} minus {b}"] = dict(zip(METRICS, np.stack((delta.mean((0, 1)), lo, hi), axis=1).tolist(), strict=True))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "full-token-conditions.npz"
    atomic_file(path, lambda out: np.savez_compressed(out, people=PEOPLE, rows=dev, donors=donors, **raw_inputs))
    observation = output / "full-token-scores.npz"
    atomic_file(observation, lambda out: np.savez_compressed(out, rows=dev, concept=dev_labels, **rows))
    report = dict(n=10 * len(dev), photos=len(train), token_shape=TOKEN_SHAPE, sources=SOURCE_COUNT, source_updates=STEPS,
                  ode_evaluations=ODE_STEPS, epochs=EPOCHS, covariance_ridge=VARIANCE_RIDGE, inverse_ridge=INVERSE_RIDGE,
                  train_seconds=train_seconds, optim_seconds=times, seeds=(SEED, SEED + 1), inverse_cosine=inverse_cosine,
                  inverse_top1=inverse_top1, magnitude=magnitude, proof=proof, generator=generator, eeg=eeg_receipts,
                  conditions_sha256=files.pin(path), observations_sha256=files.pin(observation), paired=comparisons,
                  code_sha256=files.pin(Path(__file__)),
                  results={name: dict(zip(METRICS, v.sum((0, 1)).tolist(), strict=True)) for name, v in rows.items()})
    files.verify()
    save_json(output / "full-token-source.json", report)
    print(json.dumps(report["results"], indent=2), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("benchmark", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = cli.parse_args()
    torch.set_num_threads(4)
    run(args.benchmark, args.output, torch.device(args.device))
