"""Identify the stimulus from the continuous EEG condition, before diffusion sampling."""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import PREP, ROOT, RUNS, write_json
from paper_render import condition, groups, sha
from paper_training import check_eeg


def credit(scores, k):
    positive = scores.diagonal()[:, None]
    above = (scores > positive).sum(1)
    tied = (scores == positive).sum(1)
    return ((k - above).clamp_min(0) / tied).clamp_max(1).cpu().numpy()


@torch.inference_mode()
def measure(subject, arm, budgets):
    track = "native-full" if RUNS.name.startswith("native-full-") else ("shared" if RUNS.name == "shared" else "personal")
    plan = json.loads((PREP / "paper_split.json").read_bytes())
    for phase in ("train", "test"):
        check_eeg((subject,), phase)
    if len(set(budgets)) != len(budgets) or (track == "native-full" and "fresh-r4" in budgets):
        raise ValueError("Invalid retrieval budget for this training split")
    for budget in budgets:
        panel = budget.split("-")[0]
        split = "test" if panel == "official" else "train"
        rows = np.array(plan["official_rows" if split == "test" else "fresh_rows"])
        target_file = PREP / f"F_{split}.npy"
        if split == "test":
            proof = json.loads((PREP / "paper_test_F_receipt.json").read_bytes())
            ids = json.loads((PREP / "test_ids.json").read_bytes())["ids"]
            if proof["ordered_ids"] != ids or proof["F_test_sha256"] != sha(target_file):
                raise ValueError("Official test photo identities or fused visual target checksum changed")
        eeg = np.load(PREP / f"sub-{subject:02d}_{split}.npy", mmap_mode="r")
        repeats = groups(budget)
        embedding, checkpoint = condition(arm, subject, eeg[rows], repeats, features_only=True)
        target = torch.from_numpy(np.load(target_file, mmap_mode="r")[rows].copy()).float()
        if embedding.shape != target.shape or not torch.isfinite(embedding).all() or not torch.isfinite(target).all():
            raise ValueError("Incompatible or non-finite EEG / stimulus embeddings")
        similarity = F.normalize(embedding.float()) @ F.normalize(target).T
        rank = 1 + (similarity > similarity.diagonal()[:, None]).sum(1).numpy()
        values = {"feature_top1": credit(similarity, 1), "feature_top5": credit(similarity, 5),
                  "positive_cosine": similarity.diagonal().numpy(), "rank": rank}
        if panel == "fresh":
            concepts = np.load(PREP / "train_concept.npy")[rows]
            within = np.zeros(len(rows))
            for concept in np.unique(concepts):
                idx = np.flatnonzero(concepts == concept)
                if len(idx) != 10:
                    raise ValueError("Incomplete within-concept feature gallery")
                within[idx] = credit(similarity[idx][:, idx], 1)
            values["feature_within10"] = within
        out = ROOT / f"features/{track}/{arm}/{budget}/sub-{subject:02d}"
        out.mkdir(parents=True, exist_ok=True)
        payload = out / "features.npz"
        np.savez_compressed(out / "features.partial.npz", row=rows, **values)
        (out / "features.partial.npz").replace(payload)
        means = {name: float(np.mean(v)) for name, v in values.items()}
        write_json(out / "features.json", dict(training=track, subject=subject, arm=arm, budget=budget, n=len(rows),
            repeats=repeats, gallery=len(rows), stimulus_split_sha256=sha(PREP / "paper_split.json"),
            checkpoint_sha256=checkpoint, target_sha256=sha(target_file), observations_sha256=sha(payload), mean=means,
            representation="Frozen brain encoder output" if arm == "native-paper" else "M1a brain embedding after training-cone calibration, before token residual"))
        print(track, arm, subject, budget, means, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True, type=int, choices=range(1, 11))
    p.add_argument("--arm", required=True, choices=("native-paper", "d2"))
    p.add_argument("--budgets", nargs="+", required=True, choices=("official-r4", "official-r4-balanced", "official-r80", "fresh-r4"))
    args = p.parse_args()
    measure(args.subject, args.arm, args.budgets)
