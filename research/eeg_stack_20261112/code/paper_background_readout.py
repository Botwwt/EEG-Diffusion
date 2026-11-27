"""Fit EEG object readouts with training-only foreground/background interventions."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_background_invariant import ARMS, RIDGE, object_directions
from paper_horizon_readout import registration
from paper_raw_semantic_readout import leave_photo_out, normalize, read_json
from paper_render_cache import PinnedFiles, atomic_file, require, save_json
from paper_semantic_contrast import data_registry, matched_montages
from paper_temporal_readout import BATCH, SAMPLES_PER_BIN, SEED, TemporalReadout, held_concepts, paired_batches, predict, score, visual_loss


CONTROL = "image-and-text"
EPOCHS = 12
SEGMENTS = 4
PAIR_WEIGHT = .4
TEMPERATURE = .07


def load_pairs(path, train, labels, registry, roots, files):
    receipt = path.with_suffix(".json")
    proof = read_json(receipt)
    require(proof["output_sha256"] == files.pin(path) and proof["scope"] == "training-only"
            and proof["registration"] == registry and proof["layout"] == roots["visual"],
            "Foreground targets must come from the registered EEG training photos")
    with np.load(path, allow_pickle=False) as bank:
        require(np.array_equal(bank["rows"], train), "Foreground target rows changed")
        pairs, background, views = (bank[k] for k in ("pair", "background", "views"))
    require(pairs.shape == background.shape == (len(train) // 2, 2)
            and np.array_equal(np.sort(pairs.ravel()), np.arange(len(train)))
            and np.all((background >= 0) & (background < len(train))) and len(views) == len(pairs)
            and views.shape == (len(pairs), 2, 2, 1024) and np.isfinite(views).all(), "Incomplete matched-background views")
    concepts = labels[train]
    a, b = concepts[pairs].T
    bg = concepts[background]
    require(np.all(a != b) and np.all(bg[:, 0] != bg[:, 1]) and np.all(bg != a[:, None]) and np.all(bg != b[:, None]),
            "Every counterfactual shares two unrelated backgrounds")
    return pairs, views, files.pin(receipt)


def fit(arm, initial, eeg, held, image, prototype, text, groups, partners, direction, weights):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    person_for_photo = rng.integers(len(eeg), size=eeg.shape[1])
    device = eeg.device
    model = TemporalReadout(segments=SEGMENTS).to(device)
    model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=.025)
    steps = EPOCHS * groups.shape[1] * ((len(groups) + BATCH - 1) // BATCH)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=4e-5)
    batch_ids, updates, history = sha256(), 0, []
    for epoch in range(EPOCHS):
        summary = torch.zeros(3, device=device)
        for subject, row in paired_batches(groups, person_for_photo, epoch, rng):
            batch_ids.update(subject[0].astype("<i8").tobytes())
            batch_ids.update(row[0].astype("<i8").tobytes())
            people, rows = (torch.as_tensor(values[0], device=device) for values in (subject, row))
            other = partners[rows]
            batch = torch.cat((eeg[people, rows], eeg[people, other])).float()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                photo, semantic = model(batch)
                at, away = semantic.chunk(2)
                joint = visual_loss(photo[:len(rows)], at, image[rows], prototype[rows], text[rows])
            margin = ((at - away) * direction[rows]).sum(-1)
            pair = F.softplus(-margin / TEMPERATURE).mean()
            loss = joint + (0. if arm == CONTROL else PAIR_WEIGHT) * pair
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            summary += torch.stack((joint.detach(), pair.detach(), (margin.detach() > 0).float().mean())) * len(rows)
            updates += 1
        history.append((summary / eeg.shape[1]).cpu().tolist())
        print(arm, epoch + 1, history[-1], flush=True)
    require(updates == steps, "A training arm missed scheduled photos")
    atomic_file(weights, lambda stream: torch.save({k: value.detach().cpu() for k, value in model.state_dict().items()}, stream))
    photo, semantic = predict(model, held)
    return photo, semantic, dict(updates=updates, schedule_sha256=batch_ids.hexdigest(), losses=history)


def run(benchmark, bank, output, device):
    files = PinnedFiles()
    prep = benchmark / "verified/data/prep"
    (registered, _, _), _, registration_proof = registration(prep, files)
    raw, train, dev, labels, category, image, text, layout, roots = data_registry(benchmark)
    require(np.array_equal(train, registered), "Different EEG rows for the candidate and controls")
    held = held_concepts(benchmark, train, dev, labels)
    pair, views, view_receipt = load_pairs(bank, train, labels, registration_proof, roots, files)
    visual = image[train]
    original = torch.from_numpy(visual[pair]).to(device)
    views = torch.from_numpy(views).to(device)
    natural, foreground, residual = object_directions(views, original)
    permutation = torch.arange(len(pair), device=device).roll(23)
    _, _, rotated = object_directions(views, original, nuisance_indices=permutation)
    require(all(float(d.norm(dim=-1).min()) > .999 for d in (natural, foreground, residual, rotated)),
            "An image pair has no distinct object direction")
    pair_change = float((foreground * residual).sum(-1).mean().cpu())
    choices = dict(zip((CONTROL, *ARMS), (natural, natural, foreground, residual, rotated), strict=True))
    partners = torch.empty(len(train), dtype=torch.int64, device=device)
    indices = torch.from_numpy(pair).to(device)
    partners[indices[:, 0]] = indices[:, 1]
    partners[indices[:, 1]] = indices[:, 0]
    full = benchmark / "native_baseline_candidate/enigma/prep"
    data, queries, _, recording, norm = matched_montages(raw, full, train, held, roots["photo_registry"],
                                                         bin_samples=SAMPLES_PER_BIN, dtype=np.float16, with_normalization=True)
    eeg, queries = torch.from_numpy(data).to(device), torch.from_numpy(queries).to(device)
    del data
    initial_path = benchmark / "semantic-energy-check/temporal-segments-20261127/visual-targets.pt"
    receipt = read_json(initial_path.with_suffix(".json"))
    normalizer = initial_path.with_name("normalization.npy")
    require(receipt["sources"]["data"] == roots and receipt["sources"]["eeg"] == recording
            and receipt["sources"]["segments"] == SEGMENTS and files.pin(initial_path) == receipt["weights_sha256"]
            and files.pin(normalizer) == receipt["sources"]["normalization_sha256"]
            and np.array_equal(norm, np.load(normalizer, allow_pickle=False)), "Incompatible EEG training initialization")
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    groups = np.unique(labels[train], return_inverse=True)[1]
    grouped = np.argsort(groups, kind="stable").reshape(-1, 10)
    supervise = tuple(torch.from_numpy(x).to(device=device, dtype=torch.float32) for x in
                      (visual, normalize(leave_photo_out(visual, groups)), text[labels[train]]))
    proof = dict(data=roots, eeg=recording, registration=registration_proof, view_receipt=view_receipt,
                 view_sha256=files.pin(bank), initialization_sha256=files.pin(initial_path), normalization_sha256=files.pin(normalizer),
                 epochs=EPOCHS, segments=SEGMENTS, repeats=[0, 1, 2, 3], people=10, batch_size=BATCH, seed=SEED,
                 weight=PAIR_WEIGHT, temperature=TEMPERATURE, residual_ridge=RIDGE, foreground_and_residual_cosine=pair_change,
                 code={name: files.pin(Path(__file__).with_name(name)) for name in (
                     Path(__file__).name, "paper_background_invariant.py", "paper_temporal_readout.py", "paper_semantic_contrast.py")})
    output.mkdir(parents=True, exist_ok=True)
    for arm, margin in choices.items():
        archive = output / f"{arm}.npz"
        weights, metadata = archive.with_suffix(".pt"), archive.with_suffix(".json")
        if metadata.exists():
            previous = read_json(metadata)
            require(previous["arm"] == arm and previous["sources"] == proof
                    and previous["weights_sha256"] == files.pin(weights) and previous["predictions_sha256"] == files.pin(archive),
                    "Existing readout receipts differ from registered inputs")
            continue
        directions = torch.empty((len(train), margin.shape[-1]), device=device)
        directions[indices[:, 0]], directions[indices[:, 1]] = margin, -margin
        photo, semantic, log = fit(arm, initial, eeg, queries, *supervise, grouped, partners, directions, weights)
        atomic_file(archive, lambda stream: np.savez_compressed(stream, row=held, concept=category[labels[held]],
                                                               embedding=semantic, photo_embedding=photo))
        files.verify()
        save_json(metadata, dict(arm=arm, sources=proof, predictions_sha256=files.pin(archive), weights_sha256=files.pin(weights), **log))
    comparisons = [("background-residual", a) for a in choices if a != "background-residual"]
    score(output, benchmark, held, labels, category, image, layout, proof, tuple(choices),
          comparisons=comparisons, report_name="background-readout-verified.json")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("benchmark", type=Path)
    cli.add_argument("visual_bank", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = cli.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    run(args.benchmark, args.visual_bank, args.output, args.device)
