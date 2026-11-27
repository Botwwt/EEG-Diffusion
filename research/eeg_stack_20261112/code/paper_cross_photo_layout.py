"""Train a semantic EEG readout with layout-matched cross-photo references."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_raw_semantic_readout import leave_photo_out, normalize, read_json, sha
from paper_render_cache import atomic_file, require, save_json
from paper_semantic_contrast import data_registry, matched_montages, nearest_other
from paper_temporal_readout import (
    BATCH, SAMPLES_PER_BIN, TemporalReadout, held_concepts, paired_batches, predict, score, visual_loss,
)


ARMS = ("visual-control", "crossed-references", "matched-references")
CONTEXT_ARMS = ("visual-control", "same-layout-context", "opposite-layout-context")
EPOCHS = 12
SEED = 20261204
TEMPERATURE = .07
PAIR_WEIGHT = .4
SEGMENTS = 4


def assigned_references(cost):
    """Select two disjoint reference pairs, then minimize cost over the two assignments."""
    n, count, _ = cost.shape
    episode = torch.arange(n, device=cost.device)
    positions = torch.arange(count, device=cost.device)
    first = cost.flatten(1).argmin(-1)
    a0, b0 = first // count, first % count
    excluded = (positions[None, :, None] == a0[:, None, None]) | (positions[None, None, :] == b0[:, None, None])
    second = cost.masked_fill(excluded, torch.inf).flatten(1).argmin(-1)
    a1, b1 = second // count, second % count
    alternate = cost[episode, a0, b1] + cost[episode, a1, b0] < cost[episode, a0, b0] + cost[episode, a1, b1]
    return torch.stack((a0, a1), dim=1), torch.stack((torch.where(alternate, b1, b0),
                                                     torch.where(alternate, b0, b1)), dim=1)


def paired_margin(anchor, donor, reference_a, reference_b):
    eeg_difference = anchor - donor
    if eeg_difference.ndim == 2:
        eeg_difference = eeg_difference[:, None]
    return (eeg_difference * (reference_a - reference_b)).sum(-1)


@torch.no_grad()
def episodes(layout, group, groups, device):
    query = np.arange(len(layout))
    donor = nearest_other(layout, group)
    choices = groups[group]
    sibling = choices[choices != query[:, None]].reshape(len(query), -1)
    descriptor = torch.from_numpy(layout).to(device)
    first, second, similar, crossed = [], [], [], []
    for start in range(0, len(query), 256):
        local = slice(start, start + 256)
        a, b = (torch.from_numpy(value).to(device) for value in (sibling[local], sibling[donor[local]]))
        cost = torch.cdist(descriptor[a], descriptor[b]).square() / descriptor.shape[1]
        ai, bi = assigned_references(cost)
        first.append(a.gather(1, ai).cpu().numpy())
        second.append(b.gather(1, bi).cpu().numpy())
        rows = torch.arange(len(a), device=device)[:, None]
        similar.append(cost[rows, ai, bi].cpu().numpy())
        crossed.append(cost[rows, ai, bi.flip(-1)].cpu().numpy())
    ra, rb = np.concatenate(first), np.concatenate(second)
    similar, crossed = np.concatenate(similar), np.concatenate(crossed)
    require(np.all(group[query] != group[donor]) and np.all(group[ra] == group[query, None])
            and np.all(group[rb] == group[donor, None]), "Reference concept pairing differs from the EEG pair")
    require(np.all(ra != query[:, None]) and np.all(rb != donor[:, None])
            and np.all(ra[:, 0] != ra[:, 1]) and np.all(rb[:, 0] != rb[:, 1]), "Repeated stimulus in a sibling reference")
    require(np.all(similar.sum(1) <= crossed.sum(1) + 1e-6), "Matched visual references have higher cost")
    gap = crossed.mean() - similar.mean()
    require(gap > 0 and np.isfinite(gap), "There was no appearance difference between the assignments")
    metrics = dict(eeg_layout_distance=float(((layout - layout[donor]) ** 2).mean()),
                   matched_reference_distance=float(similar.mean()), crossed_reference_distance=float(crossed.mean()),
                   different_reference_cost=float((crossed.sum(1) - similar.sum(1) > 1e-6).mean()))
    return donor, ra, rb, metrics


@torch.no_grad()
def contextual_episodes(layout, group, groups, device):
    """Build two matched EEG contexts and two independently photographed reference pairs."""
    anchor = np.arange(len(layout))
    donor = nearest_other(layout, group)
    all_photos = groups[group]
    neighbors = all_photos[all_photos != anchor[:, None]].reshape(len(anchor), -1)
    scene = torch.from_numpy(layout).to(device)
    count = neighbors.shape[1]
    slot = torch.arange(count, device=device)
    episodes = []
    diagnostics = []

    def distance(a, b):
        return (a - b).square().mean(-1)

    for start in range(0, len(anchor), 256):
        local = slice(start, start + 256)
        a0, b0 = (torch.as_tensor(rows[local], device=device) for rows in (anchor, donor))
        a, b = (torch.as_tensor(rows, device=device) for rows in (neighbors[local], neighbors[donor[local]]))
        row = torch.arange(len(a), device=device)
        xa, xb = scene[a], scene[b]
        within = torch.cdist(xa, xb).square() / scene.shape[1]
        da0, db0 = distance(xa, scene[a0, None]), distance(xb, scene[b0, None])
        offset0 = da0[:, :, None] + db0[:, None, :]
        # Seek another internally matched pair that differs in layout from the first.
        second = (within - .25 * offset0).flatten(1).argmin(-1)
        ai, bi = second // count, second % count
        a1, b1 = a[row, ai], b[row, bi]
        exclusion = (slot[None, :, None] == ai[:, None, None]) | (slot[None, None, :] == bi[:, None, None])
        first_ref = (within + .5 * offset0).masked_fill(exclusion, torch.inf).flatten(1).argmin(-1)
        ar, br = first_ref // count, first_ref % count
        exclusion |= (slot[None, :, None] == ar[:, None, None]) | (slot[None, None, :] == br[:, None, None])
        da1, db1 = distance(xa, scene[a1, None]), distance(xb, scene[b1, None])
        offset1 = da1[:, :, None] + db1[:, None, :]
        second_ref = (within + .5 * offset1).masked_fill(exclusion, torch.inf).flatten(1).argmin(-1)
        at, bt = second_ref // count, second_ref % count
        same = offset0[row, ar, br] + offset1[row, at, bt]
        opposite = offset0[row, at, bt] + offset1[row, ar, br]
        ra, rb = torch.stack((a[row, ar], a[row, at]), 1), torch.stack((b[row, br], b[row, bt]), 1)
        swap = (opposite < same)[:, None]
        ra, rb = torch.where(swap, ra.flip(1), ra), torch.where(swap, rb.flip(1), rb)
        left, right = torch.stack((a0, a1), 1), torch.stack((b0, b1), 1)
        episodes.append(torch.stack((left, right, ra, rb)).cpu().numpy())
        diagnostics.append(torch.stack((distance(scene[left], scene[right]).mean(1),
                                        distance(scene[left[:, 0]], scene[left[:, 1]]) * .5
                                        + distance(scene[right[:, 0]], scene[right[:, 1]]) * .5,
                                        torch.minimum(same, opposite) / 4, torch.maximum(same, opposite) / 4)).cpu().numpy())
    left, right, ra, rb = np.concatenate(episodes, axis=1)
    for eeg, refs in ((left, ra), (right, rb)):
        selected = np.concatenate((eeg, refs), axis=1)
        require(np.all(group[selected] == group[eeg[:, :1]]) and np.all(np.diff(np.sort(selected, axis=1), axis=1) > 0),
                "Cross-context references must exclude both photographs shown during EEG recording")
    require(np.all(group[left[:, 0]] != group[right[:, 0]]) and np.array_equal(left[:, 0], anchor),
            "Cross-context concept pairing changed")
    values = np.concatenate(diagnostics, axis=1)
    metrics = dict(zip(("within_eeg_pair_layout_distance", "between_eeg_context_layout_distance",
                        "same_context_reference_distance", "opposite_context_reference_distance"),
                       values.mean(1).tolist(), strict=True))
    require(metrics["between_eeg_context_layout_distance"] > metrics["within_eeg_pair_layout_distance"]
            and metrics["opposite_context_reference_distance"] > metrics["same_context_reference_distance"],
            "The registered layouts do not distinguish the two EEG contexts")
    return (left, right, ra, rb), metrics


def fit(arm, initial, eeg, held, target, groups, pair, out):
    device = eeg.device
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    offsets = rng.integers(len(eeg), size=eeg.shape[1])
    model = TemporalReadout(segments=SEGMENTS).to(device)
    model.load_state_dict(initial, strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=.025)
    updates = EPOCHS * groups.shape[1] * ((len(groups) + BATCH - 1) // BATCH)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates, eta_min=4e-5)
    visual, prototype, words = target
    left, right, ref_a, ref_b = (torch.as_tensor(a, device=device) for a in pair)
    if arm == "crossed-references":
        ref_b = ref_b.flip(-1)
    if arm == "opposite-layout-context":
        ref_a, ref_b = ref_a.flip(-1), ref_b.flip(-1)
    contexts = left.shape[1]
    batch_hash = sha256()
    history = []
    count = 0
    for epoch in range(EPOCHS):
        totals = torch.zeros(3, device=device)
        for person, row in paired_batches(groups, offsets, epoch, rng):
            batch_hash.update(person[0].astype("<i8").tobytes())
            batch_hash.update(row[0].astype("<i8").tobytes())
            s, i = torch.as_tensor(person[0], device=device), torch.as_tensor(row[0], device=device)
            own = eeg[s[:, None], left[i]].flatten(0, 1)
            other = eeg[s[:, None], right[i]].flatten(0, 1)
            batch = torch.cat((own, other), dim=0).float()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                photo, semantic = model(batch)
                anchor, donor = (value.reshape(len(i), contexts, -1) for value in semantic.chunk(2))
                base = visual_loss(photo[:len(own):contexts], anchor[:, 0], visual[i], prototype[i], words[i])
            margin = paired_margin(anchor, donor, visual[ref_a[i]], visual[ref_b[i]])
            assignment = F.softplus(-margin / (2 * TEMPERATURE)).mean()
            loss = base + (0. if arm == "visual-control" else PAIR_WEIGHT) * assignment
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            opt.step()
            schedule.step()
            totals += torch.stack((base.detach(), assignment.detach(), (margin.detach() > 0).float().mean())) * len(i)
            count += 1
        history.append((totals / eeg.shape[1]).cpu().tolist())
        print(arm, epoch + 1, history[-1], flush=True)
    require(count == updates, "Incomplete matched-reference training schedule")
    q, z = predict(model, held)
    atomic_file(out, lambda stream: torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, stream))
    return q, z, dict(updates=count, schedule_sha256=batch_hash.hexdigest(), losses=history)


def run(benchmark, output, device, contextual=False):
    raw, train, dev, labels, category, image, text, appearance, roots = data_registry(benchmark)
    held = held_concepts(benchmark, train, dev, labels)
    e = benchmark / "native_baseline_candidate/enigma/prep"
    data, queries, _, recording, norm = matched_montages(raw, e, train, held, roots["photo_registry"],
                                                         bin_samples=SAMPLES_PER_BIN, dtype=np.float16, with_normalization=True)
    eeg, queries = torch.from_numpy(data).to(device), torch.from_numpy(queries).to(device)
    del data
    output.mkdir(parents=True, exist_ok=True)
    initial_path = benchmark / "semantic-energy-check/temporal-segments-20261127/visual-targets.pt"
    initial_receipt = read_json(initial_path.with_suffix(".json"))
    initial_norm = initial_path.with_name("normalization.npy")
    require(initial_receipt["sources"]["data"] == roots and initial_receipt["sources"]["eeg"] == recording
            and initial_receipt["sources"]["segments"] == SEGMENTS
            and sha(initial_path) == initial_receipt["weights_sha256"]
            and sha(initial_norm) == initial_receipt["sources"]["normalization_sha256"]
            and np.array_equal(norm, np.load(initial_norm, allow_pickle=False)), "Changed frozen EEG starting point")
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    group = np.unique(labels[train], return_inverse=True)[1]
    require(np.array_equal(np.bincount(group), np.full(1323, 10)), "The cross-photo design needs all ten registered photos")
    groups = np.argsort(group, kind="stable").reshape(1323, 10)
    if contextual:
        pair, distance = contextual_episodes(appearance[train], group, groups, device)
        arms = CONTEXT_ARMS
    else:
        donor, first, second, distance = episodes(appearance[train], group, groups, device)
        pair = (np.arange(len(train))[:, None], donor[:, None], first, second)
        arms = ARMS
    path = output / "registered-pairs.npz"
    tables = dict(zip(("train", "eeg_left", "eeg_right", "reference_a", "reference_b"), (train, *pair), strict=True))
    if path.is_file():
        with np.load(path, allow_pickle=False) as values:
            require(all(np.array_equal(values[key], a) for key, a in tables.items()),
                "Train-only reference pairs changed")
    else:
        atomic_file(path, lambda f: np.savez_compressed(f, **tables))
    proof = dict(data=roots, eeg=recording, training_photos=len(train), segments=SEGMENTS, repeats=4, subjects=10,
                 initial_sha256=sha(initial_path), pairs_sha256=sha(path), code_sha256=sha(Path(__file__)),
                 seed=SEED, epochs=EPOCHS, temperature=TEMPERATURE, pair_weight=PAIR_WEIGHT, distances=distance,
                 eeg_contexts=pair[0].shape[1])
    print(json.dumps(distance), flush=True)
    photos = image[train]
    target = tuple(torch.from_numpy(a).to(device) for a in (photos, normalize(leave_photo_out(photos, group)).astype(np.float32),
                                                           text[labels[train]]))
    for arm in arms:
        weights, estimates = (output / (arm + suffix) for suffix in (".pt", ".npz"))
        receipt = estimates.with_suffix(".json")
        if receipt.is_file():
            existing = read_json(receipt)
            require(existing["arm"] == arm and existing["sources"] == proof
                    and sha(estimates) == existing["predictions_sha256"] and sha(weights) == existing["weights_sha256"],
                    "Existing layout episode result differs from the requested experiment")
            continue
        q, z, history = fit(arm, initial, eeg, queries, target, groups, pair, weights)
        atomic_file(estimates, lambda f: np.savez_compressed(f, embedding=z, photo_embedding=q, row=held, concept=category[labels[held]]))
        save_json(receipt, dict(arm=arm, sources=proof, weights_sha256=sha(weights), predictions_sha256=sha(estimates), **history))
    score(output, benchmark, held, labels, category, image, appearance, proof, arms,
          comparisons=((arms[2], arms[1]), (arms[2], arms[0]), (arms[1], arms[0])),
          report_name="cross-photo-layout-verified.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--two-contexts", action="store_true")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    run(arguments.benchmark, arguments.output, torch.device(arguments.device), arguments.two_contexts)
