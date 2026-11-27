"""Replay the ten verified D2 R80 retrieval conditions and export their fused vectors."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize

from paper_render_cache import PinnedFiles, require, save_json
from paper_retrieval import credit


@torch.inference_mode()
def run(output, scores_root):
    from common import PREP, RUNS
    from paper_render import condition, groups, physical_eeg, runtime

    require(RUNS.name == "shared", "Use the archived shared D2 fit")
    files = PinnedFiles()
    split = PREP / "paper_split.json"
    split_sha = files.pin(split)
    rows = np.asarray(json.loads(split.read_bytes())["official_rows"])
    require(np.array_equal(rows, np.arange(200)), "Changed official row order")
    target_path = PREP / "F_test.npy"
    target_sha = files.pin(target_path)
    target_receipt = PREP / "paper_test_F_receipt.json"
    target_receipt_sha = files.pin(target_receipt)
    test_ids = PREP / "test_ids.json"
    files.pin(test_ids)
    target_source = json.loads(target_receipt.read_bytes())
    require(target_source["F_test_sha256"] == target_sha and
            target_source["ordered_ids"] == json.loads(test_ids.read_bytes())["ids"], "Changed official gallery")
    target = torch.from_numpy(np.load(target_path, allow_pickle=False)).float()
    repeats = groups("official-r80")
    embeddings, hits, ranks, receipts = [], [], [], {}
    for subject in range(1, 11):
        archive = scores_root / "shared/d2/official-r80" / f"sub-{subject:02d}"
        metadata, observations = archive / "features.json", archive / "features.npz"
        metadata_sha, rows_sha = files.pin(metadata), files.pin(observations)
        expected = json.loads(metadata.read_bytes())
        require(expected["subject"] == subject and expected["repeats"] == repeats and expected["n"] == 200
                and expected["stimulus_split_sha256"] == split_sha and expected["target_sha256"] == target_sha
                and expected["observations_sha256"] == rows_sha, "Changed archived feature protocol")
        raw, eeg = physical_eeg(subject, "test", files)
        embedding, checkpoint, sources = condition("d2", subject, raw[rows], repeats, features_only=True,
                                                   files=files, return_source=True)
        require(checkpoint == expected["checkpoint_sha256"], "Changed D2 fit")
        embedding = embedding.float().cpu()
        require(embedding.shape == target.shape and bool(torch.isfinite(embedding).all()), "Invalid fused conditions")
        cosine = normalize(embedding) @ normalize(target).mT
        hit = credit(cosine, 1)
        rank = (1 + (cosine > cosine.diagonal()[:, None]).sum(1)).numpy()
        with np.load(observations, allow_pickle=False) as archive_rows:
            require(np.array_equal(archive_rows["row"], rows) and np.array_equal(archive_rows["feature_top1"], hit)
                    and np.array_equal(archive_rows["rank"], rank), "Replayed ranks differ from the archived 937-hit cohort")
            require(np.allclose(archive_rows["positive_cosine"], cosine.diagonal().numpy(), atol=5e-4),
                    "Replayed fused cosine differs from the archived conditions")
        embeddings.append(embedding.numpy())
        hits.append(hit)
        ranks.append(rank)
        receipts[str(subject)] = dict(scores=rows_sha, metadata=metadata_sha, checkpoint=checkpoint,
                                      eeg=eeg, condition=sources, replayed_top1=float(hit.sum()))
        print("Replayed R80 features", subject, int(hit.sum()), flush=True)
    require(float(np.asarray(hits).sum()) == 937, "Expected 937 of 2,000 archived top-one photo features")
    files.verify()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "official-r80-fused.npz"
    np.savez_compressed(destination, embedding=np.stack(embeddings), target=target.numpy(), row=rows,
                        feature_top1=np.stack(hits), rank=np.stack(ranks), subjects=np.arange(1, 11))
    save_json(output / "official-r80-fused.json", dict(output_sha256=files.pin(destination), n=2000, reproduced_top1=937,
              repeats=repeats, split=split_sha, target_sha256=target_sha, target_receipt=target_receipt_sha,
              code_sha256=files.pin(Path(__file__)), runtime=runtime(), per_subject=receipts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.scores_root)
