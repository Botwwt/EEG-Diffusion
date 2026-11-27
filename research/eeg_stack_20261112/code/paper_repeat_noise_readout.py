"""Estimate stimulus and repeat variance separately for a continuous EEG object readout."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize


RIDGE = .20
PHOTO_PENALTY = .75
MIX = .50
ARMS = ("native-paper", "native-map", "equal-magnitude", "repeat-aware", "repeat-wrong")
BUDGETS = ("dev-r4", "official-r4", "official-r80")


def repeat_moments(trials, labels):
    count = torch.bincount(labels)
    if trials.ndim != 3 or trials.shape[1] != 4 or count.min() < 2 or count.unique().numel() != 1:
        raise ValueError("Expected four physical repeats and equally sized multi-photo objects")
    n, r, width = trials.shape
    mean = trials.mean(dim=1)
    class_mean = trials.new_zeros(len(count), width)
    class_mean.index_add_(0, labels, mean)
    class_mean /= count[:, None]
    residual = (trials - mean[:, None]).flatten(0, 1)
    noise = (residual.T @ residual) / (n * r * (r - 1))
    within = mean - class_mean[labels]
    within = (within.T @ within) / (n - len(count))
    centered = mean - mean.mean(0)
    covariance = (centered.T @ centered) / n
    return dict(noise=noise, photo=within - noise, covariance=covariance, class_mean=class_mean)


def semantic_weights(noise, photograph, cross_target, repeats, photo_penalty):
    if repeats < 4:
        raise ValueError("An EEG query must contain at least four different physical repeats")
    system = torch.eye(len(noise), device=noise.device, dtype=noise.dtype)
    system = system - (1. - 4. / repeats) * noise + photo_penalty * photograph
    return torch.linalg.solve(system, cross_target)


def bound(coordinates, maximum):
    return coordinates / (coordinates.norm(dim=1, keepdim=True) / maximum).clamp_min(1.)


def subject_condition(native, plain, corrected, donors, maximum):
    direct = bound((1. - MIX) * native + MIX * plain, maximum)
    adjusted = bound((1. - MIX) * native + MIX * corrected, maximum)
    exchanged = bound((1. - MIX) * native + MIX * corrected[donors], maximum)
    return {"native-map": bound(native, maximum), "plain-mixture": direct,
            "equal-magnitude": normalize(direct) * adjusted.norm(dim=1, keepdim=True),
            "repeat-aware": adjusted, "repeat-wrong": exchanged}


def spectrum(matrix, maximum):
    vals, axes = torch.linalg.eigh((matrix + matrix.T).cpu().double() * .5)
    vals = vals.clamp(0., maximum).to(matrix)
    axes = axes.to(matrix)
    return (axes * vals) @ axes.T, vals, axes


def fit_readout(trials, label, targets):
    averages = trials.mean(1)
    center = averages.mean(0)
    scale = averages.std(0, unbiased=False).clamp_min(1e-5)
    moments = repeat_moments((trials - center) / scale, label)
    cov = moments["covariance"]
    ridge = cov + torch.eye(len(cov), device=cov.device) * RIDGE
    lower = torch.linalg.cholesky(ridge)
    whiten = torch.linalg.solve_triangular(lower, torch.eye(len(cov), device=cov.device), upper=False)
    noise, vals, axes = spectrum(whiten @ moments["noise"] @ whiten.T, .95)
    photo, photo_vals, _ = spectrum(whiten @ moments["photo"] @ whiten.T, 1.)
    means = moments["class_mean"] @ whiten.T
    cross = means.T @ targets / len(targets)
    signal = (means @ cross).square().mean().clamp_min(1e-12)
    gain = min(3., float((targets.square().mean() / signal).sqrt()))
    stats = dict(feature_count=len(cov), noise_median=float(vals.median()), noise_mean=float(vals.mean()),
                 photo_variance_mean=float(photo_vals.mean()), gain=gain)
    return dict(center=center, scale=scale, whiten=whiten, noise=noise, photo=photo,
                cross=cross, noise_axes=axes, gain=gain), stats


def eeg_projection(voltage, native_projection):
    lead = voltage.shape[:-2]
    data = voltage.contiguous().reshape(-1, 17, 250)
    temporal = data[..., :200].reshape(-1, 17, 40, 5).mean(-1).flatten(1)
    native = native_projection(data.flatten(1))
    return torch.cat((native, temporal), dim=-1).reshape(*lead, -1)


def standardized(features, fit):
    return ((features - fit["center"]) / fit["scale"]) @ fit["whiten"].T


def readout_path(subject, root):
    return root / "fits" / f"sub-{subject:02d}.pt"


@torch.inference_mode()
def train(subject):
    from common import PREP
    from paper_object_foreground import root, text_targets
    from paper_oof_conditions import fitted_native
    from paper_render import physical_eeg
    from paper_render_cache import PinnedFiles, atomic_file, require, save_json

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    files = PinnedFiles()
    rows, labels, _, target, _, basis = text_targets(files)
    native, prior = fitted_native(subject, "cuda")
    raw, physical = physical_eeg(subject, "train", files)
    proof = dict(physical_eeg=physical, native_checkpoint_sha256=files.pin(prior),
                 semantic_basis_sha256=basis, split_sha256=files.pin(PREP / "paper_split.json"),
                 code_sha256=files.pin(__file__))
    location = readout_path(subject, root())
    if location.exists():
        packet = torch.load(location, weights_only=True, map_location="cpu")
        require(packet["proof"] == proof and packet["subject"] == subject, "Existing fit used different physical input")
        return
    features = []
    for start in range(0, len(rows), 96):
        signal = torch.from_numpy(raw[rows[start:start + 96]].copy()).cuda().float()
        features.append(eeg_projection(signal, native.backbone.proj_in))
    joined = torch.cat(features)
    require(joined.shape == (len(rows), 4, 1704) and torch.isfinite(joined).all(), "Invalid physical trial features")
    fit, stats = fit_readout(joined, labels.cuda(), target.cuda())
    state = {key: x.cpu() if isinstance(x, torch.Tensor) else x for key, x in fit.items()}
    location.parent.mkdir(parents=True, exist_ok=True)
    files.verify()
    atomic_file(location, lambda f: torch.save(dict(state=state, proof=proof, subject=subject), f))
    save_json(location.with_suffix(".json"), dict(stats=stats, proof=proof, subject=subject,
              photos=len(rows), concepts=len(target), checkpoint_sha256=files.pin(location)))
    print("Repeat noise fit", subject, stats, flush=True)


def sample_features(raw, repeats, native, fit):
    slots = sum(repeats, [])
    if len(set(slots)) != len(slots) or len(slots) not in (4, 80):
        raise ValueError("Repeat budgets must use distinct physical trials")
    volt, observations = [], []
    for start in range(0, len(raw), 20):
        chunk = torch.from_numpy(np.array(raw[start:start + 20], copy=True)).cuda().float()
        trials = chunk[:, slots]
        volt.append(trials.mean(1))
        if len(slots) == 80:
            observations.append(torch.stack((trials[:, ::2].mean(1), trials[:, 1::2].mean(1)), dim=1))
    pooled = torch.cat(volt)
    original = torch.cat([native.backbone(x) for x in pooled.split(20)])
    whitened = standardized(eeg_projection(pooled, native.backbone.proj_in), fit)
    stability = {}
    if observations:
        dual = standardized(eeg_projection(torch.cat(observations), native.backbone.proj_in), fit)
        axes = dual @ fit["noise_axes"]
        for index, plane in enumerate(axes.chunk(4, dim=-1), 1):
            a, b = (plane - plane.mean(0)).unbind(1)
            stability[str(index)] = float((2 * (a * b).sum() / (a.square().sum() + b.square().sum())).cpu())
    return original, whitened, stability


@torch.inference_mode()
def infer(subject, budgets, render_budgets, arms):
    from common import PREP
    from paper_object_foreground import root, text_targets
    from paper_object_foreground_render import generate, split_input
    from paper_oof_conditions import fitted_native
    from paper_render import groups
    from paper_render_cache import PinnedFiles, atomic_file, require, save_json
    from paper_repeat_semantic_bridge import fit_head, predict
    from paper_soft_text import wrong_donors

    files = PinnedFiles()
    rows, labels, _, target, states, basis = text_targets(files)
    native, prior = fitted_native(subject, "cuda")
    native_hash = files.pin(prior)
    saved = readout_path(subject, root())
    fit_hash = files.pin(saved)
    learned = torch.load(saved, map_location="cpu", weights_only=True)
    require(learned["subject"] == subject and learned["proof"]["native_checkpoint_sha256"] == native_hash
            and learned["proof"]["semantic_basis_sha256"] == basis, "Readout used different weights or supervision")
    fit = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in learned["state"].items()}
    source = PREP / "F_train.npy"
    photo_sha = files.pin(source)
    photo = torch.from_numpy(np.load(source, mmap_mode="r", allow_pickle=False)[rows].copy()).float().cuda()
    target = target.cuda()
    prior_head = fit_head(photo, target, labels.cuda())
    del photo
    maximum = target.norm(dim=-1).quantile(.95)
    pipeline = None
    for budget in budgets:
        waveform, physical, query = split_input(subject, budget, files)
        repeats = groups(budget)
        original, feature, stability = sample_features(waveform, repeats, native, fit)
        plain = (feature @ fit["cross"]) * fit["gain"]
        coeff = semantic_weights(fit["noise"], fit["photo"], fit["cross"], sum(map(len, repeats)), PHOTO_PENALTY)
        adjusted = (feature @ coeff) * fit["gain"]
        base = predict(prior_head, original)
        donors = wrong_donors(len(query)).tolist()
        codes = subject_condition(base, plain, adjusted, donors, maximum)
        codes = {name: vector.cpu() for name, vector in codes.items()}
        vectors = {name: states["mean_pooled"] + code @ states["pooled_basis"] for name, code in codes.items()}
        shared = dict(subject=subject, budget=budget, row=query, repeats=repeats, physical_eeg=physical,
                      native_checkpoint_sha256=native_hash, semantic_basis_sha256=basis, readout_sha256=fit_hash,
                      training_photo_features_sha256=photo_sha, split_sha256=files.pin(PREP / "paper_split.json"),
                      code_sha256=files.pin(__file__), photo_penalty=PHOTO_PENALTY, native_mix=1. - MIX)
        folder = root() / "images" / budget / f"sub-{subject:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        files.verify()
        atomic_file(folder / "semantic-readout.pt", lambda f: torch.save(dict(proof=shared, pool=vectors, donors=donors), f))
        save_json(folder / "repeat-reproducibility.json", dict(same_photograph_split_correlation=stability,
                  norm={name: float(z.norm(dim=-1).median()) for name, z in codes.items()}, proof=shared))
        print("Repeat noise readout", subject, budget, stability, flush=True)
        if budget in render_budgets:
            if pipeline is None:
                from paper_identity_render import load_generator

                pipeline = load_generator(torch.float16, False)
            generate(folder, codes, states, original, pipeline, shared, files, donors, arms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "infer", "score"))
    parser.add_argument("subject", type=int, choices=range(1, 11))
    parser.add_argument("--budgets", nargs="+", choices=BUDGETS, default=BUDGETS)
    parser.add_argument("--render-budgets", nargs="+", choices=BUDGETS, default=[])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    args = parser.parse_args()
    if not set(args.render_budgets).issubset(args.budgets):
        parser.error("Image budgets must also have frozen EEG readouts")
    if args.action == "train":
        train(args.subject)
    elif args.action == "infer":
        infer(args.subject, args.budgets, args.render_budgets, args.arms)
    else:
        from paper_object_foreground_render import score

        score(args.subject, args.budgets, args.render_budgets, args.arms)
