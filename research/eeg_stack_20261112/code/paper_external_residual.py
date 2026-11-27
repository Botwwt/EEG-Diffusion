"""Test whether cross-acquisition EEG predicts errors left by an internal EEG teacher."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_external_teacher import align_half, covered_donors, early_bins, read_external
from paper_paired_encoder import SUBJECTS
from paper_render_cache import PinnedFiles, require, save_json
from paper_token_evidence import ridge_weights


def nested_prediction(external, peer, folds, outer):
    sources = torch.zeros_like(peer)
    counts = torch.zeros(len(peer), device=peer.device)
    for inner in (x for x in (1, 2, 3, 4) if x != outer):
        excluded = (1 << outer) | (1 << inner)
        for values, idx, context in external:
            fit = (context & excluded) == 0
            take = (folds[idx] == inner) | (folds[idx] == outer)
            require(not np.any(fit & take) and fit.sum() >= 2 and take.any(),
                    "Missing doubly held-out cross-acquisition support")
            pred = align_half(values, peer[idx], fit, take)
            target = torch.from_numpy(idx[take]).to(peer.device)
            sources.index_add_(0, target, pred)
            counts.index_add_(0, target, torch.ones_like(target, dtype=counts.dtype))
    estimate = sources / counts.clamp_min(1)[:, None]
    counts[folds == outer] /= 3
    return estimate, counts


def retrieval(estimated, visual, labels, valid):
    _, inverse, counts = labels.unique(return_inverse=True, return_counts=True)
    require(torch.all(counts == 10), "Expected ten photographs per training object")
    members = inverse.argsort(stable=True).reshape(-1, 10)
    query = F.normalize(estimated[valid])
    image = F.normalize(visual)
    categories = inverse[valid]
    reference = visual[members].sum(1)
    own = (query * F.normalize(reference[categories] - visual[valid])).sum(1)
    other = (query @ F.normalize(reference).T).scatter_(1, categories[:, None], -torch.inf).amax(1)
    hits = members[categories].gather(1, (query[:, None] * image[members[categories]]).sum(2).argmax(1, keepdim=True))
    indices = torch.where(valid)[0]
    return {"full_object": own > other, "same_object_photo": hits[:, 0] == indices}


def paired_interval(improvement, categories):
    _, group, sizes = np.unique(categories, return_inverse=True, return_counts=True)
    deltas = np.bincount(group, weights=improvement)
    g = np.random.default_rng(20261129)
    draw = g.integers(len(sizes), size=(10000, len(sizes)))
    return np.quantile(deltas[draw].sum(1) / sizes[draw].sum(1), [.025, .975]).tolist()


@torch.inference_mode()
def run(args):
    from common import CH17, concept_folds

    pins = PinnedFiles()
    split = args.registration / "paper_split.json"
    labels_file = args.registration / "train_concept.npy"
    index_file = args.registration / "train_ids.json"
    rows = np.asarray(json.loads(split.read_bytes())["train_rows"], dtype=np.int64)
    labels = np.load(labels_file, allow_pickle=False)[rows]
    folds = concept_folds(1654)[labels]
    proof = dict(split=pins.pin(split), labels=pins.pin(labels_file), photo_index=pins.pin(index_file),
                 photos=pins.pin(args.fused), target_bank=pins.pin(args.target_bank),
                 channels=CH17, external={}, internal={}, code=pins.pin(Path(__file__)))
    target_bank = torch.load(args.target_bank, map_location="cpu", weights_only=True)
    sources = target_bank["sources"]
    require(np.array_equal(rows, target_bank["rows"].numpy())
            and all(proof[k] == sources[k] for k in ("split", "labels", "photos")),
            "Residual estimation used a different image bank")
    ext = read_external(args.epochs, rows, folds, proof, pins)
    eeg2 = {}
    for p in SUBJECTS:
        loc = args.eeg2 / f"sub-{p:02d}_train.npy"
        digest = pins.pin(loc)
        require(digest == sources["eeg"][str(p)]["array_sha256"], "A physical waveform differs")
        proof["internal"][str(p)] = digest
        data = np.load(loc, mmap_mode="r", allow_pickle=False)
        eeg2[p] = torch.cat([early_bins(torch.from_numpy(data[ii][..., :60].copy()).cuda().float().mean(1))
                             for ii in np.array_split(rows, 55)])
    target = torch.from_numpy(np.load(args.fused, mmap_mode="r", allow_pickle=False)[rows].copy()).cuda().float()
    concepts = torch.from_numpy(labels).cuda()
    centered = target - target.mean(0)
    names = ("internal-peer", "external-residual", "sibling-residual")
    summary = {}
    for recipient in (1, 3):
        raw = [eeg2[p] for p in SUBJECTS if p != recipient]
        valid = torch.zeros(len(rows), device="cuda", dtype=torch.bool)
        output = {key: torch.zeros_like(target) for key in names}
        by_fold = {}
        for k in (1, 2, 3, 4):
            test = folds == k
            train = ~test
            h = torch.stack([(v - v[train].mean(0)) / v[train].std(0).clamp_min(1e-5) for v in raw]).mean(0)
            h -= h[train].mean(0)
            f = target - target[train].mean(0)
            b = h @ ridge_weights(h[train], f[train], .3)
            q, presence = nested_prediction(ext, h, folds, k)
            donor, coverage = covered_donors(labels, presence.cpu().numpy().astype(np.int64))
            support = coverage > 0
            fit = train & support
            cal_h = h - h[fit].mean(0)
            cal_q = q - q[fit].mean(0)
            residual = cal_q - cal_h @ ridge_weights(cal_h[fit], cal_q[fit], .3)
            err = f - b
            err = err - err[fit].mean(0)
            offset = residual @ ridge_weights(residual[fit], err[fit], .75)
            changed = offset[torch.from_numpy(donor.clip(min=0)).cuda()]
            weight = torch.from_numpy((coverage / 3).clip(max=1)).cuda().float()[:, None]
            output[names[0]][test] = b[test]
            output[names[1]][test] = (b + weight * offset)[test]
            output[names[2]][test] = (b + weight * changed)[test]
            valid[test] = torch.from_numpy(support[test]).cuda()
            by_fold[str(k)] = {"train_joint": int(fit.sum()), "held_joint": int((test & support).sum())}
        outcomes = {a: retrieval(prediction, centered, concepts, valid) for a, prediction in output.items()}
        result = {a: {"queries": int(valid.sum()),
                      **{key: int(score.sum()) for key, score in v.items()}} for a, v in outcomes.items()}
        comparisons = {}
        for control in (names[0], names[2]):
            comparisons[control] = {}
            for key, estimates in outcomes[names[1]].items():
                change = (estimates.float() - outcomes[control][key].float()).cpu().numpy()
                categories = concepts[valid].cpu().numpy()
                comparisons[control][key] = dict(difference=float(change.mean()),
                                                  interval=paired_interval(change, categories))
        summary[str(recipient)] = dict(folds=by_fold, diagnostic=result, comparisons=comparisons)
        print(recipient, json.dumps(result), flush=True)
    pins.verify()
    save_json(args.output, dict(subjects=summary, sources=proof, additional_penalty=.75))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("epochs", "eeg2", "registration", "fused", "target-bank", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    run(p.parse_args())
