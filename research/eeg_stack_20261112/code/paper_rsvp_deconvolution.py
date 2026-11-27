"""Learn visual EEG kernels from training events, then read out a query without its neighbors."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


ARMS = ("native-paper", "native-map", "plain-kernel", "unmixed-kernel", "unmixed-wrong")
BUDGETS = ("dev-r4", "official-r4", "official-r80")


def complete_context(edges):
    current = edges[:, 0] * 4 + edges[:, 1]
    successor = edges[:, 2] * 4 + edges[:, 3]
    middle, previous, following = np.intersect1d(successor, current, assume_unique=True, return_indices=True)
    return np.column_stack((middle // 4, middle % 4, current[previous] // 4, successor[following] // 4))


def fit_kernels(own, previous, successor, early, late, penalty=.15):
    first = torch.cat((own, previous), dim=-1)
    second = torch.cat((successor, own), dim=-1)
    x = torch.cat((first, second))
    y = torch.cat((early, late))
    cov = (x.T @ x) / len(x)
    cov.diagonal().add_(penalty)
    result = torch.linalg.solve(cov, (x.T @ y) / len(x))
    return result.chunk(2)


def wave_bins(eeg):
    # Two 200 ms intervals at 250 Hz with ten 20 ms values for each electrode.
    values = eeg[..., :100].reshape(*eeg.shape[:-1], 2, 10, 5).mean(-1)
    return values.transpose(-2, -3).reshape(*eeg.shape[:-2], 2, 170)


def photo_coordinates(photo, fit_rows):
    source = photo[fit_rows]
    center = source.mean(0)
    selected = source - center
    vals, axes = torch.linalg.eigh((selected.T @ selected / len(source)).cpu().double())
    space = axes[:, -128:].to(photo) / vals[-128:].sqrt().to(photo).clamp_min(1e-5)
    return (photo - center) @ space


def kernels_for(wave, photos, event_rows):
    own, _, previous, successor = event_rows.T
    current = photos[own]
    input_mean = wave.mean(0)
    early, late = (wave - input_mean).unbind(1)
    zero = torch.zeros_like(current)
    return {
        "plain-kernel": fit_kernels(current, zero, zero, early, late),
        "unmixed-kernel": fit_kernels(current, photos[previous], photos[successor], early, late),
    }, input_mean


def predicted_rows(kernels, photos, triplets, use_context):
    center, _, prev, next_image = triplets.T
    early, late = kernels
    current = photos[center]
    first, second = current @ early, current @ late
    if use_context:
        first = first + photos[prev] @ late
        second = second + photos[next_image] @ early
    return torch.stack((first, second), dim=1)


@torch.inference_mode()
def fit(subject, context):
    from common import PREP
    from paper_object_foreground import root, text_targets
    from paper_oof_conditions import registered_rows
    from paper_render import physical_eeg
    from paper_render_cache import PinnedFiles, atomic_file, require, save_json
    from paper_repeat_noise_readout import fit_readout

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    files = PinnedFiles()
    training, labels, _, target, _, basis = text_targets(files)
    ordered, _, folds = registered_rows(PREP)
    require(np.array_equal(training, ordered), "Text and RSVP training registrations differ")
    raw, eeg_proof = physical_eeg(subject, "train", files)
    mapped = context / f"sub-{subject:02d}-training-context.npy"
    verified = mapped.with_name(mapped.stem + "-audit.json")
    events_hash, validation = files.pin(mapped), files.pin(verified)
    audit = json.loads(verified.read_bytes())
    split = files.pin(PREP / "paper_split.json")
    require(audit["input"]["context_array_sha256"] == events_hash and audit["verified_eeg_sha256"] == eeg_proof["array_sha256"]
            and audit["input"]["split_sha256"] == split, "RSVP event mapping has a different EEG or split")
    require(audit["subject"] == subject and audit["exact_order_better"] > .99, "Actual voltages do not verify chronological order")
    event_rows = complete_context(np.load(mapped, allow_pickle=False))
    fold_by_photo = np.full(len(raw), -1, dtype=np.int32)
    fold_by_photo[training] = folds
    scores = fold_by_photo[event_rows[:, [0, 2, 3]]]
    require(np.min(scores) > 0, "A non-training photograph entered the event design")
    fit_only = np.all(scores != 4, axis=1)
    held = scores[:, 0] == 4
    require(fit_only.any() and held.any(), "No class-disjoint context rows")
    training_wave = []
    for start in range(0, len(training), 128):
        batch = torch.from_numpy(raw[training[start:start + 128]].copy()).cuda().float()
        training_wave.append(wave_bins(batch))
    voltage = torch.cat(training_wave)
    inverse = np.full(len(raw), -1, dtype=int)
    inverse[training] = np.arange(len(training))
    all_wave = voltage[inverse[event_rows[:, 0]], event_rows[:, 1]]
    photo_file = PREP / "F_train.npy"
    photos_sha = files.pin(photo_file)
    photo = torch.from_numpy(np.load(photo_file, mmap_mode="r", allow_pickle=False).copy()).float().cuda()
    partial_photo = photo_coordinates(photo, training[folds != 4])
    estimate, center = kernels_for(all_wave[fit_only], partial_photo, event_rows[fit_only])
    outcomes = {}
    observed = all_wave[held] - center
    held_rows = event_rows[held]
    permuted = held_rows.copy()
    permuted[:, 2:] = np.roll(permuted[:, 2:], len(permuted) // 2, axis=0)
    for name, pulses in estimate.items():
        own = predicted_rows(pulses, partial_photo, held_rows, use_context=name == "unmixed-kernel")
        swapped = predicted_rows(pulses, partial_photo, permuted, use_context=name == "unmixed-kernel")
        own_error = (own - observed).square().mean((1, 2))
        wrong_error = (swapped - observed).square().mean((1, 2))
        outcomes[name] = dict(mse=float(own_error.mean()), wrong_context_mse=float(wrong_error.mean()),
                              context_pair_wins=float((own_error < wrong_error).float().sum()
                                                       + .5 * (own_error == wrong_error).float().sum()))
    features = photo_coordinates(photo, training)
    learned, _ = kernels_for(all_wave, features, event_rows)
    # Use identical EEG covariance for the two inverse readouts, and differ only in the visual-to-neural target.
    target = target.cuda()
    labels = labels.cuda()
    measured = voltage.flatten(-2)
    base, _ = fit_readout(measured, labels, target)
    transformations = {}
    for name, (early, late) in learned.items():
        expected = torch.cat((features[training] @ early, features[training] @ late), dim=1)
        expected = ((expected - expected.mean(0)) / base["scale"]) @ base["whiten"].T
        teacher = target[labels]
        cross = expected.T @ teacher / len(expected)
        predicted = expected @ cross
        gain = min(6., float((teacher.square().mean() / predicted.square().mean().clamp_min(1e-12)).sqrt()))
        transformations[name] = {"cross": cross.cpu(), "gain": gain}
    save = root() / "fits" / f"sub-{subject:02d}.pt"
    save.parent.mkdir(parents=True, exist_ok=True)
    common = {name: base[name].cpu() for name in ("center", "scale", "whiten", "noise")}
    proof = dict(context=events_hash, context_waveform_verification=validation, physical_eeg=eeg_proof,
                 semantic_basis_sha256=basis, split_sha256=split, photo_sha256=photos_sha, code=files.pin(__file__))
    files.verify()
    atomic_file(save, lambda f: torch.save(dict(common=common, modes=transformations, subject=subject, proof=proof), f))
    save_json(save.with_suffix(".json"), dict(subject=subject, proof=proof, training_triples=len(event_rows),
              fold4_fit_triples=int(fit_only.sum()), fold4_eval_triples=int(held.sum()), held=outcomes,
              digest=files.pin(save)))
    print("RSVP deconvolution fit", subject, len(event_rows), outcomes, flush=True)


@torch.inference_mode()
def infer(subject, budgets, render_budgets):
    from common import PREP
    from paper_object_foreground import predict_at_budget, root, text_targets
    from paper_object_foreground_render import generate, split_input
    from paper_oof_conditions import fitted_native
    from paper_render import groups
    from paper_render_cache import PinnedFiles, atomic_file, require
    from paper_repeat_noise_readout import bound, semantic_weights, standardized
    from paper_repeat_semantic_bridge import fit_head, predict
    from paper_soft_text import wrong_donors

    files = PinnedFiles()
    rows, labels, _, targets, state, basis = text_targets(files)
    model, checkpoint = fitted_native(subject, "cuda")
    path = root() / "fits" / f"sub-{subject:02d}.pt"
    digest = files.pin(path)
    packet = torch.load(path, map_location="cuda", weights_only=True)
    photo_path = PREP / "F_train.npy"
    photo_hash = files.pin(photo_path)
    require(packet["subject"] == subject and packet["proof"]["semantic_basis_sha256"] == basis
            and packet["proof"]["photo_sha256"] == photo_hash, "EEG decoder has different visual supervision")
    targets = targets.cuda()
    prior = fit_head(torch.from_numpy(np.load(photo_path, mmap_mode="r", allow_pickle=False)[rows].copy()).float().cuda(),
                     targets, labels.cuda())
    maximum = targets.norm(dim=1).quantile(.95)
    transform = packet["common"]
    pipeline = None
    for budget in budgets:
        raw, proof, rows = split_input(subject, budget, files)
        repeats = groups(budget)
        number = sum(map(len, repeats))
        chosen = sum(repeats, [])
        pooled = [torch.from_numpy(np.array(raw[s:s + 20, chosen], copy=True)).cuda().float().mean(1)
                  for s in range(0, len(raw), 20)]
        signals = standardized(wave_bins(torch.cat(pooled)).flatten(1), transform)
        original = predict_at_budget(model.backbone, raw, repeats)
        base = predict(prior, original)
        variants = {}
        for mode, parameters in packet["modes"].items():
            coeff = semantic_weights(transform["noise"], torch.zeros_like(transform["noise"]),
                                     parameters["cross"], number, 0.)
            variants[mode] = (signals @ coeff) * parameters["gain"]
        donors = wrong_donors(len(rows)).tolist()
        variants["unmixed-wrong"] = variants["unmixed-kernel"][donors]
        codes = {name: bound(.5 * (base + selected), maximum).cpu() for name, selected in variants.items()}
        codes["native-map"] = bound(base, maximum).cpu()
        shared = dict(subject=subject, budget=budget, row=rows, repeats=repeats, physical_eeg=proof,
                      native_checkpoint_sha256=files.pin(checkpoint), semantic_basis_sha256=basis, timing_readout=digest,
                      split_sha256=files.pin(PREP / "paper_split.json"), code_sha256=files.pin(__file__))
        folder = root() / "images" / budget / f"sub-{subject:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        vectors = {name: state["mean_pooled"] + code @ state["pooled_basis"] for name, code in codes.items()}
        atomic_file(folder / "semantic-readout.pt", lambda f: torch.save(dict(proof=shared, pool=vectors, donors=donors), f))
        print("RSVP blind EEG probe", subject, budget, len(rows), flush=True)
        if budget in render_budgets:
            if pipeline is None:
                from paper_identity_render import load_generator

                pipeline = load_generator(torch.float16, False)
            generate(folder, codes, state, original, pipeline, shared, files, donors, ARMS)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("fit", "infer", "score"))
    cli.add_argument("subject", type=int, choices=(1, 3))
    cli.add_argument("--context", type=Path)
    cli.add_argument("--budgets", choices=BUDGETS, nargs="+", default=BUDGETS)
    cli.add_argument("--render-budgets", choices=BUDGETS, nargs="+", default=[])
    args = cli.parse_args()
    if not set(args.render_budgets).issubset(args.budgets):
        cli.error("Render budgets must have EEG readouts")
    if args.action == "fit":
        if args.context is None:
            cli.error("--context is required for fitting")
        fit(args.subject, args.context)
    elif args.action == "infer":
        infer(args.subject, args.budgets, args.render_budgets)
    else:
        from paper_object_foreground_render import score

        score(args.subject, args.budgets, args.render_budgets, ARMS)
