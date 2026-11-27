"""Measure exact photograph correspondence between independently recorded EEG datasets."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_paired_encoder import SUBJECTS
from paper_render_cache import PinnedFiles, require, save_json
from paper_token_evidence import ridge_weights


def standardize(x, training):
    reference = x[training]
    return (x - reference.mean(0)) / reference.std(0).clamp_min(1e-5)


def same_object_pairs(predicted, observed, labels):
    matching = F.normalize(predicted) @ F.normalize(observed).T
    own = matching.diag()
    compare = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    margin = (own[:, None] + own[None, :] - matching - matching.T)[compare]
    groups = labels[:, None].expand_as(compare)[compare]
    return own.mean().item(), margin.cpu().numpy(), groups.cpu().numpy()


@torch.inference_mode()
def run(args):
    from common import concept_folds

    pins = PinnedFiles()
    labels_file = args.registration / "train_concept.npy"
    split_file = args.registration / "paper_split.json"
    source_info = args.eeg1.with_suffix(".json")
    data = json.loads(split_file.read_bytes())
    proof = dict(external_eeg=pins.pin(args.eeg1), external_receipt=pins.pin(source_info),
                 labels=pins.pin(labels_file), split=pins.pin(split_file), visual_bank=pins.pin(args.target_bank),
                 code=pins.pin(Path(__file__)), eeg2={})
    target = torch.load(args.target_bank, map_location="cpu", weights_only=True)
    external = json.loads(source_info.read_bytes())
    require(proof["external_eeg"] == external["sha256"]
            and target["sources"]["labels"] == proof["labels"] == external["files"]["concept_ids"]
            and target["sources"]["split"] == proof["split"] == external["files"]["split"]
            and target["rows"].numpy().tolist() == data["train_rows"],
            "External and internal EEG must use the same registered training identities")
    category = np.load(labels_file, allow_pickle=False)
    with np.load(args.eeg1, allow_pickle=False) as packet:
        rows, context = packet["train_row"], packet["context_fold_mask"]
        eeg1 = torch.from_numpy(packet["eeg"]).cuda().reshape(-1, 17, 12, 5).mean(-1).flatten(1)
    require(np.isin(rows, data["train_rows"]).all()
            and np.all((context & (1 << concept_folds(1654)[category[rows]])) > 0),
            "An external epoch or its surroundings are not registered for training")
    labels = torch.from_numpy(category[rows]).cuda()
    folds = concept_folds(1654)[category[rows]]
    recordings = []
    for person in SUBJECTS:
        path = args.eeg2 / f"sub-{person:02d}_train.npy"
        digest = pins.pin(path)
        require(digest == target["sources"]["eeg"][str(person)]["array_sha256"], "Physical internal EEG differs")
        proof["eeg2"][str(person)] = digest
        x = np.load(path, mmap_mode="r", allow_pickle=False)
        waveforms = torch.from_numpy(x[rows][..., :60].copy()).cuda().float().mean(1)
        recordings.append(waveforms.reshape(-1, 17, 12, 5).mean(-1).flatten(1))
    panels, all_margins, all_groups = [], [], []
    for fold in (1, 2, 3, 4):
        train = torch.from_numpy((context & (1 << fold)) == 0).cuda()
        held = torch.from_numpy(folds == fold).cuda()
        require(not torch.any(train & held) and train.any() and held.any(), "Held-object fit partition differs")
        external_eeg = standardize(eeg1, train)
        peer_mean = torch.stack([standardize(x, train) for x in recordings]).mean(0)
        weight = ridge_weights(external_eeg[train], peer_mean[train], .5)
        cosine, margins, groups = same_object_pairs(external_eeg[held] @ weight, peer_mean[held], labels[held])
        wins = (margins > 0).astype(float) + .5 * (margins == 0)
        panels.append(dict(fold=fold, training=int(train.sum()), queries=int(held.sum()), pairs=len(wins),
                           correct_pairs=float(wins.sum()), mean_matching_cosine=cosine))
        all_margins.append(wins)
        all_groups.append(groups)
    wins, object_ids = np.concatenate(all_margins), np.concatenate(all_groups)
    _, group, size = np.unique(object_ids, return_inverse=True, return_counts=True)
    counts = np.bincount(group, weights=wins)
    rng = np.random.default_rng(20261128)
    sample = rng.integers(len(size), size=(10000, len(size)))
    rates = counts[sample].sum(1) / size[sample].sum(1)
    result = dict(n=len(wins), success=float(wins.sum()), paired_ci=np.quantile(rates, [.025, .975]).tolist(),
                  source_participant=external["subject"], eeg2_peer_count=len(SUBJECTS), panels=panels, provenance=proof)
    pins.verify()
    save_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("eeg1", "eeg2", "registration", "target-bank", "output"):
        cli.add_argument(f"--{key}", type=Path, required=True)
    run(cli.parse_args())
