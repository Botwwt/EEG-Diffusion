"""Align photo-dependent EEG responses using only repeats of the current query."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, atomic_file, require, save_json


SHIFTS = (-4, -2, 0, 2, 4)
START, STOP = 25, 90
RAMP = 9
STRENGTHS = (4., 8., 16., 32., 64.)
POOLS = ("mean", "precision", "symmetric", "aligned")


def shift_marginals(features, strength):
    """Marginalize a small joint distribution of synchronous-repeat offsets."""
    n, repeats, _, samples = features.shape
    require(repeats in (2, 4) and samples >= STOP + max(SHIFTS), "Expected two or four physical epochs")
    shifts = torch.as_tensor(SHIFTS, device=features.device)
    choices = torch.arange(len(shifts), device=features.device)
    assignments = torch.cartesian_prod(*[choices] * repeats)
    time = torch.arange(START, STOP, device=features.device)
    shifted = features[..., (time[:, None] + shifts)].permute(0, 1, 4, 2, 3).flatten(3)
    normalized = F.normalize(shifted, dim=-1)
    log_prior = -.5 * (shifts[assignments].float() / 2.5).square().sum(1)
    score = normalized.new_zeros((n, len(assignments)))
    for first in range(repeats):
        for second in range(first + 1, repeats):
            correlation = normalized[:, first] @ normalized[:, second].transpose(-1, -2)
            score += correlation[:, assignments[:, first], assignments[:, second]]
    probability = (score * (2. * strength / repeats) + log_prior).softmax(dim=1)
    choices_per_trial = F.one_hot(assignments, len(shifts)).flatten(1).to(probability)
    return (probability @ choices_per_trial).reshape(n, repeats, len(shifts))


def aggregate(trials, center, axes, strength):
    """Pool the same voltage observations with unsigned and signed shift controls."""
    mean = trials.mean(1)
    weights = trials.square().mean(-1, keepdim=True).clamp_min(1e-6).rsqrt()
    weights = weights / weights.sum(1, keepdim=True)
    precision = (trials * weights).sum(1)
    pools = dict(mean=mean, precision=precision)
    if strength == 0.:
        return dict(pools, symmetric=precision, aligned=precision)

    residual = trials - center
    features = axes.T @ residual
    assignments = shift_marginals(features, strength)
    shifts = torch.as_tensor(SHIFTS, device=trials.device)
    time = torch.arange(trials.shape[-1], device=trials.device)
    offset = (time[:, None] + shifts).clamp(0, len(time) - 1)
    variants = residual[..., offset]
    envelope = ((time - (START - RAMP)).clamp(0, RAMP)
                .minimum((STOP + RAMP - 1 - time).clamp(0, RAMP)) / RAMP)
    for name, probabilities in (("symmetric", .5 * (assignments + assignments.flip(-1))),
                                ("aligned", assignments)):
        synchronized = torch.einsum("brctl,brl->brct", variants, probabilities)
        change = ((synchronized - residual) * weights).sum(1)
        pools[name] = precision + envelope * change
    return pools


def fit_axes(ordered, sessions, ranks=8):
    """Estimate spatial modes reproducible between separate acquisition sessions."""
    first, second = ordered[:, :2].mean(1), ordered[:, 2:].mean(1)
    center = .5 * (first.mean(0) + second.mean(0))
    means = torch.cat((first, second))
    ids = torch.cat((sessions[:, 0], sessions[:, 2]))
    session_centers = torch.stack([means[ids == k].mean(0) for k in range(4)])
    a = (first - session_centers[sessions[:, 0]])[:, :, START:STOP]
    b = (second - session_centers[sessions[:, 2]])[:, :, START:STOP]
    a, b = (t.transpose(1, 2).reshape(-1, ordered.shape[2]) for t in (a, b))
    signal = a.T @ b / len(a)
    signal = .5 * (signal + signal.T)
    noise = (ordered[:, ::2, :, START:STOP] - ordered[:, 1::2, :, START:STOP]) / np.sqrt(2.)
    noise = noise.transpose(-1, -2).reshape(-1, ordered.shape[2])
    covariance = noise.T @ noise / len(noise)
    covariance.diagonal().add_(.25 * covariance.diagonal().mean())
    values, directions = torch.linalg.eigh(covariance)
    whitener = (directions / values.clamp_min(1e-6).sqrt()) @ directions.T
    reproducibility, axes = torch.linalg.eigh(whitener @ signal @ whitener)
    return center, whitener @ axes[:, -ranks:], reproducibility.flip(0)


def pooled_batches(trials, center, axes, strength, batch=96):
    estimates = {arm: [] for arm in POOLS}
    for block in trials.split(batch):
        for arm, result in aggregate(block, center, axes, strength).items():
            estimates[arm].append(result)
    return {arm: torch.cat(values) for arm, values in estimates.items()}


def match_report(pools, reference, center, axes, wrong):
    def normalized(x):
        return F.normalize((axes.T @ (x - center))[..., START:STOP].flatten(1))

    target = normalized(reference)
    report = {}
    for arm, waveform in pools.items():
        score = normalized(waveform) @ target.T
        correct = score.diagonal()
        margin = correct - score[torch.arange(len(target), device=target.device), wrong]
        report[arm] = dict(top1=int((score.argmax(1) == torch.arange(len(target), device=target.device)).sum()),
                           paired=int((margin > 0).sum()), cosine=float(correct.mean()), margin=float(margin.mean()))
    return report


def session_controls(labels, sessions, seed):
    rng = np.random.default_rng(seed)
    sessions = np.asarray(sessions)
    donor = np.empty(len(labels), dtype=np.int64)
    for pair in np.unique(sessions, axis=0):
        group = np.flatnonzero((sessions == pair).all(1))
        for i in group:
            candidates = group[labels[group] != labels[i]]
            require(len(candidates) > 0, "Each session needs another registered object")
            donor[i] = rng.choice(candidates)
    return donor


def load_record(raw_root, subject, phase, files):
    stem = raw_root / f"sub-{subject:02d}_{phase}"
    array, meta = stem.with_suffix(".npy"), stem.with_suffix(".json")
    session = raw_root / "sessions" / array.name
    proof = {name: files.pin(path) for name, path in (("eeg", array), ("receipt", meta), ("session", session))}
    receipt = json.loads(meta.read_bytes())
    ids = raw_root / f"{phase}_ids.json"
    proof["ids"] = files.pin(ids)
    require(receipt["subject"] == f"sub-{subject:02d}" and receipt["split"] == phase
            and receipt["session_sha256"] == proof["session"] and receipt["output_sha256"] == proof["eeg"]
            and receipt["ids_sha256"] == proof["ids"] and receipt["sample_rate_hz"] == 250,
            "Acquisition bytes, identities or timing disagree with the physical receipt")
    raw = np.load(array, allow_pickle=False, mmap_mode="r")
    sessions = np.load(session, allow_pickle=False)
    require(tuple(receipt["shape"]) == raw.shape and raw.shape[:2] == sessions.shape
            and raw.shape[2:] == (17, 250), "Unexpected EEG layout")
    return raw, sessions, proof


@torch.inference_mode()
def calibrate(subject, prep, raw_root, out, device):
    files = PinnedFiles()
    split_path, label_path = prep / "paper_split.json", prep / "train_concept.npy"
    proof = dict(subject=subject, split=files.pin(split_path), labels=files.pin(label_path), code=files.pin(__file__))
    split = json.loads(split_path.read_bytes())
    rows = np.asarray(split["train_rows"])
    labels = np.load(label_path, allow_pickle=False)[rows]
    objects, counts = np.unique(labels, return_counts=True)
    permutation = np.random.default_rng(20261112).permutation(1654)
    folds = np.empty(1654, dtype=int)
    folds[permutation] = np.arange(1654) % 5
    require(len(objects) == 1323 and np.all(counts == 10) and np.all(folds[labels] != 0)
            and np.array_equal(rows, np.unique(rows)) and not np.intersect1d(rows, split["fresh_rows"]).size,
            "Training must use exactly the registered complete objects")
    raw, sessions, proof["train_eeg"] = load_record(raw_root, subject, "train", files)
    order = np.argsort(sessions[rows], axis=1, kind="stable")
    sessions = np.take_along_axis(sessions[rows], order, axis=1)
    require(np.isin(sessions, np.arange(4)).all() and np.array_equal(sessions[:, ::2], sessions[:, 1::2])
            and np.all(sessions[:, 0] != sessions[:, 2]), "Expected two separate sessions per training image")
    x = torch.from_numpy(raw[rows[:, None], order].copy()).to(device=device, dtype=torch.float32)
    s = torch.from_numpy(sessions.astype(np.int64)).to(device)
    fit, held = np.flatnonzero(folds[labels] != 4), np.flatnonzero(folds[labels] == 4)
    center, axes, _ = fit_axes(x[fit], s[fit])
    calibration = x[held]
    wrong = torch.from_numpy(session_controls(labels[held], sessions[held, ::2], 20261127)).to(device)
    half = (calibration[:, :2], calibration[:, 2:])
    report = {}
    for strength in (*STRENGTHS, 0.):
        report[strength] = [match_report(pooled_batches(own, center, axes, strength), other.mean(1),
                                         center, axes, wrong) for own, other in (half, half[::-1])]
    selected = max(STRENGTHS, key=lambda value: sum(r["aligned"]["margin"] for r in report[value]))
    center, axes, reproducibility = fit_axes(x, s)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"sub-{subject:02d}.pt"
    files.verify()
    atomic_file(path, lambda f: torch.save(dict(center=center.cpu(), axes=axes.cpu(), strength=selected,
                                                proof=proof), f))
    summary = dict(calibration_rows=len(held), fitted_rows=len(rows), strength=selected,
                   reproducibility=reproducibility.cpu().tolist(), cross_session=report,
                   proof=proof, checkpoint=files.pin(path))
    save_json(path.with_suffix(".json"), summary)
    print("Cross-session repeat calibration", subject, selected, report[selected], flush=True)


@torch.inference_mode()
def diagnose(subject, raw_root, out, device):
    files = PinnedFiles()
    path = out / f"sub-{subject:02d}.pt"
    packet = torch.load(path, weights_only=True, map_location=device)
    pinned = files.pin(path)
    raw, sessions, proof = load_record(raw_root, subject, "test", files)
    require(packet["proof"]["subject"] == subject and raw.shape[:2] == (200, 80)
            and np.array_equal(sessions[0], sessions[1:][0]) and np.all(sessions == sessions[:1]),
            "Expected the registered 80-repeat official acquisition")
    epochs = torch.from_numpy(raw.copy()).to(device=device, dtype=torch.float32)
    center, axes = packet["center"], packet["axes"]
    # Only the first four physical observations enter any estimate.
    estimates = pooled_batches(epochs[:, :4], center, axes, packet["strength"])
    other_observations = epochs[:, 4:].mean(1)
    donor = torch.arange(len(epochs), device=device).roll(67)
    scores = match_report(estimates, other_observations, center, axes, donor)
    files.verify()
    save_json(out / f"sub-{subject:02d}-repeat-check.json", dict(
        n=len(raw), train_checkpoint=pinned, test_eeg=proof, estimator=list(range(4)),
        reference=list(range(4, 80)), scores=scores))
    print("Independent repeated EEG", subject, scores, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("calibrate", "diagnose"))
    cli.add_argument("subject", type=int, choices=range(1, 11))
    cli.add_argument("--prep", type=Path, required=True)
    cli.add_argument("--recordings", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--device", default="cuda")
    args = cli.parse_args()
    torch.set_num_threads(6)
    if args.phase == "calibrate":
        calibrate(args.subject, args.prep, args.recordings, args.out, args.device)
    else:
        diagnose(args.subject, args.recordings, args.out, args.device)
