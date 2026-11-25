"""Identify the stimulus from the continuous EEG condition, before diffusion sampling."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from filelock import FileLock

from common import PREP, ROOT, RUNS
from paper_render import condition, groups, physical_eeg, render_code
from paper_render_cache import PinnedFiles, atomic_file, key, require, save_json, sha, track_for


def credit(scores, k):
    positive = scores.diagonal()[:, None]
    above = (scores > positive).sum(1)
    tied = (scores == positive).sum(1)
    return ((k - above).clamp_min(0) / tied).clamp_max(1).cpu().numpy()


@torch.inference_mode()
def measure(subject, arm, budgets):
    track = track_for(RUNS, subject)
    files = PinnedFiles()
    split_file = PREP / "paper_split.json"
    split_sha = files.pin(split_file)
    plan = json.loads(split_file.read_bytes())
    if len(set(budgets)) != len(budgets) or (track == "native-full" and "fresh-r4" in budgets):
        raise ValueError("Invalid retrieval budget for this training split")
    physical = {s: physical_eeg(subject, s, files) for s in sorted({"test" if b.startswith("official-") else "train" for b in budgets})}
    for budget in budgets:
        panel = budget.split("-")[0]
        split = "test" if panel == "official" else "train"
        rows = np.array(plan["official_rows" if split == "test" else "fresh_rows"])
        target_file = PREP / f"F_{split}.npy"
        target_sha = files.pin(target_file)
        target_receipt_sha = None
        if split == "test":
            target_receipt = PREP / "paper_test_F_receipt.json"
            target_receipt_sha = files.pin(target_receipt)
            proof = json.loads(target_receipt.read_bytes())
            ids = json.loads((PREP / "test_ids.json").read_bytes())["ids"]
            if proof["ordered_ids"] != ids or proof["F_test_sha256"] != target_sha:
                raise ValueError("Official test photo identities or fused visual target checksum changed")
        eeg, input_proof = physical[split]
        repeats = groups(budget)
        embedding, checkpoint, condition_source = condition(arm, subject, eeg[rows], repeats, features_only=True, files=files, return_source=True)
        target = torch.from_numpy(np.load(target_file, mmap_mode="r")[rows].copy()).float()
        if embedding.shape != target.shape or not torch.isfinite(embedding).all() or not torch.isfinite(target).all():
            raise ValueError("Incompatible or non-finite EEG / stimulus embeddings")
        similarity = F.normalize(embedding.float()) @ F.normalize(target).T
        rank = 1 + (similarity > similarity.diagonal()[:, None]).sum(1).numpy()
        values = {"feature_top1": credit(similarity, 1), "feature_top5": credit(similarity, 5),
                  "positive_cosine": similarity.diagonal().numpy(), "rank": rank}
        if panel == "fresh":
            files.pin(PREP / "train_concept.npy")
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
        payload, manifest = out / "features.npz", out / "features.json"
        means = {name: float(np.mean(v)) for name, v in values.items()}
        source = {"version": 2, "condition_sources": condition_source, "physical_eeg": input_proof, "rows_sha256": key(rows.tolist()),
                  "target_receipt_sha256": target_receipt_sha, "code_sha256": {**render_code(files, arm), "retrieval": files.pin(Path(__file__))},
                  "training_concept_sha256": files.pin(PREP / "train_concept.npy") if panel == "fresh" else None}
        request = dict(training=track, subject=subject, arm=arm, budget=budget, n=len(rows), repeats=repeats, gallery=len(rows),
                       stimulus_split_sha256=split_sha, checkpoint_sha256=checkpoint, target_sha256=target_sha, feature_source=source,
                       representation="Frozen brain encoder output" if arm == "native-paper" else "M1a brain embedding after training-cone calibration, before token residual")
        with FileLock(str(out / ".features-v2.lock")):
            files.verify()
            marker = out / ".features-v2.request.json"
            request_key = key(request)
            if marker.exists():
                require(json.loads(marker.read_bytes()) == request_key, f"Existing feature transaction uses other inputs: {out}")
            existing = json.loads(manifest.read_bytes()) if manifest.is_file() else None
            if existing is not None:
                require({k: v for k, v in existing.items() if k not in ("mean", "observations_sha256")} == request,
                        f"Existing feature receipt is historical or uses other inputs: {out}")
                if payload.is_file() and sha(payload) == existing["observations_sha256"]:
                    continue
            require(existing is not None or marker.exists() or not payload.exists(), f"Unbound existing feature payload: {out}")
            stream = io.BytesIO()
            np.savez_compressed(stream, row=rows, **values)
            raw = stream.getvalue()
            digest = hashlib.sha256(raw).hexdigest()
            require(existing is None or digest == existing["observations_sha256"], "Feature repair differs from its original payload")
            save_json(marker, request_key)
            atomic_file(payload, lambda f: f.write(raw))
            save_json(manifest, dict(request, observations_sha256=digest, mean=means))
        print(track, arm, subject, budget, means, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True, type=int, choices=range(1, 11))
    p.add_argument("--arm", required=True, choices=("native-paper", "d2"))
    p.add_argument("--budgets", nargs="+", required=True, choices=("official-r4", "official-r4-balanced", "official-r80", "fresh-r4"))
    args = p.parse_args()
    measure(args.subject, args.arm, args.budgets)
