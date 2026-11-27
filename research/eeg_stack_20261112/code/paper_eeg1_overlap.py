"""Remove predicted responses to nearby stimuli from registered EEG1 training epochs."""

import argparse
from bisect import bisect_left, bisect_right
import csv
import json
from pathlib import Path

import numpy as np
import torch

from paper_eeg1_cross_recording import same_object_pairs, standardize
from paper_external_teacher import covered_donors, safe_alignment
from paper_paired_encoder import SUBJECTS
from paper_render_cache import PinnedFiles, require, save_json
from paper_token_evidence import ridge_weights
from prepare_eeg1_context import FUTURE, PAST, path_key

BINS, LAGS, DIMENSION = 12, 15, 32
ARMS = ("raw", "remove-neighbors", "remove-siblings")


def response_design(n, geometry, features, allowed):
    epoch, rows, offset, center = geometry
    values = .02 * np.arange(BINS) + .01
    lag = np.floor((values[None, :] - offset[:, None]) / .02 + 1e-5).astype(np.int64) - 1
    permitted = allowed[rows, None] & (lag >= 0) & (lag < LAGS)
    rec, times = np.nonzero(permitted)
    bins = (epoch[rec] * BINS + times) * LAGS + lag[rec, times]
    design = features.new_zeros((n * BINS * LAGS, features.shape[1]))
    rows_tensor = torch.as_tensor(rows[rec], device=features.device)
    bins_tensor = torch.as_tensor(bins, device=features.device)
    neighbors_mask = torch.as_tensor(~center[rec], device=features.device)
    vectors = features[rows_tensor]
    design.index_add_(0, bins_tensor[neighbors_mask], vectors[neighbors_mask])
    neighbor = design.reshape(n * BINS, -1)
    design = design.clone()
    design.index_add_(0, bins_tensor[~neighbors_mask], vectors[~neighbors_mask])
    return design.reshape(n * BINS, -1), neighbor


def context_geometry(events, original, photo_table, rows):
    times = [float(r["onset"]) for r in events]
    local = [photo_table.get(path_key(r["stim"], False), -1) for r in events]
    require(np.array_equal(np.asarray(local)[original], rows), "Continuous event timestamps name another photo")
    indices, pictures, deltas, center = [], [], [], []
    for e, event in enumerate(original):
        onset = times[event]
        begin = bisect_left(times, onset - PAST)
        end = bisect_right(times, onset + FUTURE)
        for j in range(begin, end):
            if local[j] >= 0 and (j == event or local[j] != local[event]):
                indices.append(e)
                pictures.append(local[j])
                deltas.append(times[j] - onset)
                center.append(j == event)
    return tuple(np.asarray(x) for x in (indices, pictures, deltas, center))


def visual_coordinates(original, take):
    x = original[take]
    center = x.mean(0)
    x = x - center
    variance, axes = torch.linalg.eigh(x.T @ x / len(x))
    transform = axes[:, -DIMENSION:] / variance[-DIMENSION:].clamp_min(1e-8).sqrt()
    return (original - center) @ transform


def project(e1, e2, fit, held):
    a = standardize(e1, fit)
    fit_group = e2[fit]
    target_mean = fit_group.mean(0)
    coefficients = ridge_weights(a[fit], fit_group - target_mean, .5)
    return a[held] @ coefficients + target_mean


@torch.inference_mode()
def run(args):
    from common import concept_folds

    pins = PinnedFiles()
    split = args.registration / "paper_split.json"
    categories = args.registration / "train_concept.npy"
    index = args.registration / "train_ids.json"
    photo = args.fused
    bank_path = args.target_bank
    rows = np.asarray(json.loads(split.read_bytes())["train_rows"], dtype=np.int64)
    proof = dict(split=pins.pin(split), labels=pins.pin(categories), index=pins.pin(index), photos=pins.pin(photo),
                 target_bank=pins.pin(bank_path), code=pins.pin(Path(__file__)), eeg2={}, eeg1={})
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    require(np.array_equal(rows, bank["rows"].numpy()) and bank["sources"]["photos"] == proof["photos"]
            and bank["sources"]["split"] == proof["split"] and bank["sources"]["labels"] == proof["labels"],
            "Overlap correction must use the registered outer training bank")
    targets = np.load(categories, allow_pickle=False)[rows]
    folds = concept_folds(1654)[targets]
    paths = json.loads(index.read_bytes())["paths"]
    photo_table = {path_key(paths[i], True): j for j, i in enumerate(rows)}
    sibling, _ = covered_donors(targets, np.ones(len(targets), dtype=np.int64))
    reference = torch.from_numpy(np.load(photo, mmap_mode="r", allow_pickle=False)[rows].copy()).cuda().float()
    internal = []
    for person in SUBJECTS:
        path = args.eeg2 / f"sub-{person:02d}_train.npy"
        digest = pins.pin(path)
        require(digest == bank["sources"]["eeg"][str(person)]["array_sha256"], "The physical E2 recordings differ")
        proof["eeg2"][str(person)] = digest
        wave = np.load(path, mmap_mode="r", allow_pickle=False)
        internal.append(torch.cat([torch.from_numpy(wave[j][..., :60].copy()).cuda().float()
                                   .mean(1).reshape(-1, 17, BINS, 5).mean(-1).flatten(1)
                                   for j in np.array_split(rows, 55)]))
    embed = {k: visual_coordinates(reference, folds != k) for k in (1, 2, 3, 4)}
    results = {}
    for p in args.subjects:
        person = f"sub-{p:02d}"
        prefix = args.epochs / f"{person}-registered-context.npz"
        meta = prefix.with_suffix(".json")
        source = json.loads(meta.read_bytes())
        tsv_name, = [filename for filename in source["files"]["files"] if filename.endswith(".tsv")]
        tsv = args.events / tsv_name
        evidence = dict(epoch=pins.pin(prefix), receipt=pins.pin(meta), event=pins.pin(tsv))
        require(evidence["epoch"] == source["sha256"] and evidence["event"] == source["files"]["files"][tsv_name]
                and source["files"]["split"] == proof["split"] and source["files"]["photo_index"] == proof["index"],
                "Context source or split changed")
        proof["eeg1"][person] = evidence
        with np.load(prefix, allow_pickle=False) as f:
            idx = np.searchsorted(rows, f["train_row"])
            require(np.all(idx < len(rows)) and np.array_equal(rows[idx], f["train_row"]),
                    "An external image falls outside the visual training bank")
            where, context = f["source_event"], f["context_fold_mask"]
            eeg = torch.from_numpy(f["eeg"]).cuda().reshape(-1, 17, BINS, 5).mean(-1)
        n = len(eeg)
        with tsv.open(newline="", encoding="utf-8") as h:
            geometry = context_geometry(list(csv.DictReader(h, delimiter="\t")), where, photo_table, idx)
        epochs, local, delta, is_center = geometry
        wrong_geometry = (epochs, np.where(is_center, local, sibling[local]), delta, is_center)
        paired = [v[idx] for v in internal]
        truth = torch.from_numpy(targets[idx]).cuda()
        observed = eeg.transpose(1, 2).reshape(n * BINS, 17)
        score = {a: [] for a in ARMS}
        for fold in (1, 2, 3, 4):
            fit, held = safe_alignment(context, folds[idx], fold)
            take = np.repeat(fit, BINS)
            x, real = response_design(n, geometry, embed[fold], folds != fold)
            _, false = response_design(n, wrong_geometry, embed[fold], folds != fold)
            design = x[take] - x[take].mean(0)
            y = observed[take] - observed[take].mean(0)
            transfer = ridge_weights(design, y, .25)
            e2 = torch.stack([standardize(v, fit) for v in paired]).mean(0)
            for method, cleaned in zip(ARMS, (observed, observed-real@transfer, observed-false@transfer), strict=True):
                e1 = cleaned.reshape(n, BINS, 17).transpose(1, 2).reshape(n, -1)
                corr, delta, _ = same_object_pairs(project(e1, e2, fit, held), e2[held], truth[held])
                correct = (delta > 0).sum() + .5 * (delta == 0).sum()
                score[method].append(dict(fold=fold, training=int(fit.sum()), test=int(held.sum()),
                                          pairs=len(delta), correct=float(correct), cosine=corr))
        results[person] = score
        print(person, {k: (sum(v["correct"] for v in x), sum(v["pairs"] for v in x)) for k, x in score.items()},
              flush=True)
    pins.verify()
    save_json(args.output, dict(results=results, sources=proof, overlap_lag_ms=[20, 320], visual_dimensions=DIMENSION))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("epochs", "events", "eeg2", "registration", "fused", "target-bank", "output"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--subjects", type=int, choices=range(1, 13), nargs="+", default=(1, 2, 3))
    run(cli.parse_args())
