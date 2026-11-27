"""Measure which frozen visual target carries object information through a held-out EEG fit."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_foreground_targets import eeg_matrix, estimate, regression
from paper_identity_bank import class_sums
from paper_ijepa_context import MODES, register
from paper_repeat_latency import load_record
from paper_render_cache import PinnedFiles, require, save_json


def object_scores(prediction, labels, truth, sums, bank):
    prediction = F.normalize(prediction)
    own = (prediction * F.normalize(sums[labels] - truth)).sum(1, keepdim=True)
    scores = (prediction @ bank.T).scatter(1, labels[:, None], own)
    rank = 1 + (scores > own).sum(1)
    return dict(top1=int((rank == 1).sum()), top5=int((rank <= 5).sum()), correct_similarity=float(own.mean()))


@torch.inference_mode()
def run(args):
    files = PinnedFiles()
    _, _, _, registration = register(args.prep, args.eeg / "train_ids.json", files)
    targets_hash, dino_hash = files.pin(args.targets), files.pin(args.dino)
    targets = torch.load(args.targets, map_location="cpu", weights_only=True)
    photo = torch.load(args.dino, map_location="cpu", weights_only=True)
    rows = targets["rows"].numpy()
    concepts = np.load(args.prep / "train_concept.npy", allow_pickle=False)[rows]
    proof = targets["info"]["proof"]
    require(np.array_equal(photo["rows"].numpy(), rows) and np.array_equal(photo["concepts"][photo["labels"]], concepts)
            and proof["split"] == registration["split"] == photo["proof"]["split"]
            and proof["eeg_identity_list"] == registration["eeg_identity_list"], "Visual targets use other training photos")
    labels = photo["labels"].long().cuda()
    dino = F.normalize(photo["image_features"].float().cuda())
    sums = class_sums(dino, labels)
    bank = F.normalize(sums)
    fold = np.empty(1654, dtype=int)
    fold[np.random.default_rng(20261112).permutation(1654)] = np.arange(1654) % 5
    folds = fold[concepts]
    views = {arm: targets["targets"][arm].cuda().float() for arm in MODES}
    reports = {}
    for person in args.subjects:
        array, _, acquisition = load_record(args.eeg, person, "train", files)
        signal = eeg_matrix(array[rows], [0, 1, 2, 3])
        result = {}
        for held in (1, 2, 3, 4):
            train = torch.from_numpy(folds != held).cuda()
            test = ~train
            features, truth, category = signal[test], dino[test], labels[test]
            result[held] = {}
            for arm, tokens in views.items():
                readout = regression(signal[train], tokens[train])
                visual = regression(tokens[train], dino[train])
                actual = estimate(visual, estimate(readout, features))
                full = estimate(visual, tokens[test])
                result[held][arm] = dict(
                    eeg=object_scores(actual, category, truth, sums, bank),
                    photo=object_scores(full, category, truth, sums, bank),
                )
            direct = estimate(regression(signal[train], dino[train]), features)
            result[held]["direct-dino"] = dict(eeg=object_scores(direct, category, truth, sums, bank))
        totals = {arm: {k: sum(f[arm]["eeg"][k] for f in result.values()) for k in ("top1", "top5")}
                  for arm in (*MODES, "direct-dino")}
        reports[person] = dict(scores=result, totals=totals, source=acquisition)
        print("Held-object I-JEPA EEG readout", person, totals, flush=True)
    files.verify()
    save_json(args.output, dict(views=reports, proof=dict(registered=registration, targets=targets_hash, dino=dino_hash)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("prep", "eeg", "targets", "dino", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--subjects", nargs="+", choices=range(1, 11), default=[1, 2, 3, 4], type=int)
    run(cli.parse_args())
