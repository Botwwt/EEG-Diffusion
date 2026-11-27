"""Test exact same-photograph cross-person EEG signal directions on held-out objects."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_paired_encoder import SUBJECTS, prepare_eeg
from paper_render_cache import PinnedFiles, atomic_file, require, save_json

VIEWERS = (1, 3, 5, 6)
ARMS = ("ridge", "same-photo-peers", "other-photo-peers")
PENALTY, RANK = .15, 128


def whitening(training, test):
    mean = training.mean(0)
    scale = training.std(0).clamp_min(1e-5)
    x = (training - mean) / scale
    eigen, axes = torch.linalg.eigh(x.T @ x / len(x))
    preconditioner = axes / (eigen.clamp_min(0.) + PENALTY).sqrt()
    return x @ preconditioner, ((test - mean) / scale) @ preconditioner


def rotate_photographs(labels, offset):
    _, inverse, counts = labels.unique(return_inverse=True, return_counts=True)
    require(torch.all(counts == counts[0]) and 0 < offset < int(counts[0]),
            "Control requires an equal count of training photos per object and a nonzero rotation")
    by_object = inverse.argsort(stable=True).reshape(-1, int(counts[0]))
    donor = torch.empty_like(inverse)
    donor[by_object.flatten()] = by_object.roll(offset, dims=1).flatten()
    return donor


def peer_directions(observed, peers, rank):
    covariance = observed.new_zeros((observed.shape[1], observed.shape[1]))
    for peer in peers:
        correlation = observed.T @ peer / len(observed)
        covariance.addmm_(correlation, correlation.T)
    _, modes = torch.linalg.eigh(covariance)
    return modes[:, -rank:].contiguous()


def boost_shared(coefficients, axes):
    shared = axes @ (axes.T @ coefficients)
    return .5 * (coefficients + shared)


def scoring(prediction, visual, labels, sessions):
    _, inverse, counts = labels.unique(return_inverse=True, return_counts=True)
    require(torch.all(counts == 10), "Evaluation objects must keep ten reference photos")
    photos = inverse.argsort(stable=True).reshape(-1, 10)
    require(torch.all(sessions[photos] == sessions[photos[:, :1]]), "An object spans several recording pairs")
    query = F.normalize(prediction)
    sums = visual[photos].sum(1)
    positive = (query * F.normalize(sums[inverse] - visual)).sum(1)
    comparison = (query @ F.normalize(sums).T).scatter_(1, inverse[:, None], -torch.inf)
    same_session = sessions[:, None] == sessions[photos[:, 0]][None, :]
    matched = comparison.masked_fill(~same_session, -torch.inf)
    photograph = (query[:, None, :] * F.normalize(visual[photos[inverse]], dim=-1)).sum(2)
    order = torch.arange(len(query), device=query.device)
    return {
        "full_object": positive > comparison.max(1).values,
        "same_session_object": positive > matched.max(1).values,
        "full_rank": 1 + (comparison >= positive[:, None]).sum(1),
        "same_object_photo": photos[inverse, photograph.argmax(1)] == order,
    }


@torch.inference_mode()
def run(args):
    from common import concept_folds

    files = PinnedFiles()
    split_file = args.registration / "paper_split.json"
    labels_file = args.registration / "train_concept.npy"
    split = json.loads(split_file.read_bytes())
    rows = np.asarray(split["train_rows"], dtype=np.int64)
    proof = dict(split=files.pin(split_file), labels=files.pin(labels_file), fused=files.pin(args.fused),
                 target_bank=files.pin(args.target_bank), eeg={}, code=files.pin(Path(__file__)))
    bank = torch.load(args.target_bank, map_location="cpu", weights_only=True)
    source = bank["sources"]
    require(all(proof[k] == source[k] for k in ("split", "labels")) and proof["fused"] == source["photos"]
            and np.array_equal(rows, bank["rows"].numpy()), "Readout used an unregistered visual target")
    category = np.load(labels_file, allow_pickle=False)[rows]
    folds = concept_folds(1654)[category]
    require(len(rows) == 13230 and len(np.unique(category)) == 1323 and set(folds) == {1, 2, 3, 4}
            and not np.intersect1d(rows, split["fresh_rows"]).size, "Expected full held-concept partitions")
    labels = torch.from_numpy(category).cuda()
    fold_ids = torch.from_numpy(folds).cuda()
    recordings, sessions = {}, {}
    for person in SUBJECTS:
        prefix = f"sub-{person:02d}_train"
        path = args.eeg / f"{prefix}.npy"
        proof["eeg"][str(person)] = files.pin(path)
        require(proof["eeg"][str(person)] == source["eeg"][str(person)]["array_sha256"],
                "A peer array differs from the verified Brain-HiVE physical inputs")
        feed = np.load(path, mmap_mode="r", allow_pickle=False)
        recordings[person] = prepare_eeg(feed, rows).flatten(1)
        if person in VIEWERS:
            session_file = args.eeg / "sessions" / f"{prefix}.npy"
            source_definition = json.loads((args.eeg / f"{prefix}.json").read_bytes())
            expected = files.pin(session_file)
            require(expected == source_definition["session_sha256"], "Source session order changed")
            proof.setdefault("sessions", {})[str(person)] = expected
            ordered = np.sort(np.load(session_file, allow_pickle=False)[rows], axis=1).astype(np.int64)
            require(np.all(ordered[:, [0, 2]] == ordered[:, [1, 3]]) and np.all(ordered[:, 0] != ordered[:, 2]),
                    "Expected two repeats in each of two observed sessions")
            sessions[person] = torch.from_numpy(ordered[:, 0] * 4 + ordered[:, 2]).cuda()
    original_f = torch.from_numpy(np.load(args.fused, mmap_mode="r", allow_pickle=False)[rows].copy()).cuda().float()
    d = bank["coordinates"].cuda()
    visual = torch.cat((original_f - original_f.mean(0), d), dim=1)
    relative_scale = bank["basis"]["scale"].cuda()
    results = {s: {arm: {key: [] for key in ("fused", "relative")} for arm in ARMS} for s in VIEWERS}
    saved = {"registered_rows": rows, "objects": category, "folds": folds}
    for k in (1, 2, 3, 4):
        hold = fold_ids == k
        transformed = {s: whitening(x[~hold], x[hold]) for s, x in recordings.items()}
        target_center = visual[~hold].mean(0)
        y = visual[~hold] - target_center
        normalized_targets = [visual[hold, :original_f.shape[1]], d[hold] * relative_scale]
        validate = labels[hold]
        exchange = {s: rotate_photographs(labels[~hold], 1 + (s - 1) % 9) for s in SUBJECTS}
        for s in VIEWERS:
            x_train, x_test = transformed[s]
            peers = [t for t in SUBJECTS if t != s]
            real = peer_directions(x_train, (transformed[t][0] for t in peers), RANK)
            wrong = peer_directions(x_train, (transformed[t][0][exchange[t]] for t in peers), RANK)
            solve = x_train.T @ y / len(y)
            for arm, weight in zip(ARMS, (solve, boost_shared(solve, real), boost_shared(solve, wrong)), strict=True):
                predicted = x_test @ weight + target_center
                fused, condition = predicted.split((original_f.shape[1], d.shape[1]), dim=1)
                for key, tensor, truth in zip(("fused", "relative"), (fused, condition * relative_scale),
                                              normalized_targets, strict=True):
                    metrics = scoring(tensor, truth, validate, sessions[s][hold])
                    results[s][arm][key].append(dict(n=len(validate), **{
                        name: float(x.sum()) if name == "full_rank" else int(x.sum())
                        for name, x in metrics.items()
                    }))
                    for name, x in metrics.items():
                        saved[f"p{s:02d}_fold{k}_{arm}_{key}_{name}"] = x.cpu().numpy()
        print("shared signal held fold", k, len(validate), flush=True)
    result = {}
    for s, person in results.items():
        result[s] = {}
        for arm, conditions in person.items():
            result[s][arm] = {}
            for domain, by_fold in conditions.items():
                result[s][arm][domain] = {name: sum(z[name] for z in by_fold) for name in by_fold[0]}
                result[s][arm][domain]["mean_full_rank"] = (
                    result[s][arm][domain].pop("full_rank") / result[s][arm][domain]["n"])
    files.verify()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "shared-signal-observations.npz"
    atomic_file(archive, lambda out: np.savez_compressed(out, **saved))
    save_json(args.output / "shared-signal.json", dict(subjects=result, proof=proof,
               observation_sha256=files.pin(archive), penalty=PENALTY, rank=RANK))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("registration", "eeg", "target-bank", "fused", "output"):
        p.add_argument(f"--{field}", type=Path, required=True)
    run(p.parse_args())
