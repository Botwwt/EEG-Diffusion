"""Fixed-final, concept-OOF trial-set *representation pilot*; never an image result.

Default arms have identical trainable parameters and initialization. ``mean``
encodes the physical R4 mean; ``each_mean`` encodes each of the same four trials
before pooling. Optional learned pooling has extra parameters, reported exactly.
No held-fold metric controls epochs, learning rate, checkpoint choice, or arms.
The launcher owns GPU exclusivity and raw-trial / existing-mean identity checks.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


VERSION = "BrainHIVE_trialset_fastscreen_20261027_v1"
ARMS = ("mean", "each_mean", "learned_pool")


class Deadline(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_torch(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def atomic_npy(path, array):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temp, path)


def read_index(path):
    path = Path(path)
    if path.suffix.lower() in (".csv", ".tsv"):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ","))
    elif path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(rows, dict):
            rows = rows["rows"]
    result = []
    for item in rows:
        row = dict(item)
        row["row"] = int(row.get("row", row.get("array_row", -1)))
        row["target_row"] = int(row.get("target_row", row["row"]))
        row["concept_id"] = str(row.get("concept_id", row.get("concept", "")))
        row["fold"] = int(row["fold"])
        row["subject_id"] = str(row.get("subject_id", row.get("subject", "")))
        if "partition" in row and row["partition"] != "T":
            raise ValueError("Only T index rows are permitted")
        result.append(row)
    return sorted(result, key=lambda row: row["row"])


def validate_index(rows, trials_shape, target_shape, expected_rows):
    if len(rows) != expected_rows:
        raise ValueError(f"Expected full registered denominator {expected_rows}, got {len(rows)}")
    if len(trials_shape) != 4 or tuple(trials_shape[1:]) != (4, 17, 250):
        raise ValueError(f"Expected [N,4,17,250], got {trials_shape}")
    if len(target_shape) != 2 or target_shape[1] != 1024:
        raise ValueError(f"Expected [N,1024] targets, got {target_shape}")
    if {row["fold"] for row in rows} != {0, 1}:
        raise ValueError("Index must contain exactly the two outer concept folds 0 and 1")
    subjects = {row["subject_id"] for row in rows}
    if len(subjects) != 1 or "" in subjects:
        raise ValueError("This pilot requires one explicit subject, never pooled subjects")
    if len({row["row"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source row")
    if len({row["target_row"] for row in rows}) != len(rows):
        raise ValueError("Duplicate target row")
    concepts = {}
    for row in rows:
        if not 0 <= row["row"] < trials_shape[0] or not 0 <= row["target_row"] < target_shape[0]:
            raise ValueError("Array row outside actual source")
        if not row["concept_id"]:
            raise ValueError("Missing concept ID")
        concepts.setdefault(row["concept_id"], []).append(row)
    for concept, group in concepts.items():
        if len(group) != 10 or len({row["fold"] for row in group}) != 1:
            raise ValueError(f"Concept {concept} must have all 10 stimuli in exactly one fold")
    return {"subject_id": next(iter(subjects)), "rows": len(rows), "concepts": len(concepts),
            "fold_rows": {str(fold): sum(row["fold"] == fold for row in rows) for fold in (0, 1)}}


class TrialSetPredictor(nn.Module):
    def __init__(self, arm, hidden=256):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.encoder = nn.Sequential(nn.Linear(17 * 250, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.head = nn.Sequential(nn.Linear(hidden, 1024), nn.LayerNorm(1024))
        if arm == "learned_pool":
            # Depend on content and its deviation from the query's mean feature.
            # Nonlinearity prevents the residual term cancelling in softmax.
            self.pool_score = nn.Sequential(nn.Linear(2 * hidden, 32), nn.GELU(), nn.Linear(32, 1))
            nn.init.zeros_(self.pool_score[-1].weight)
            nn.init.zeros_(self.pool_score[-1].bias)

    def forward(self, trials):
        flat = trials.flatten(2)
        if self.arm == "mean":
            pooled = self.encoder(flat.mean(1))
        else:
            features = self.encoder(flat)
            if self.arm == "each_mean":
                pooled = features.mean(1)
            else:
                residual = features - features.mean(1, keepdim=True)
                weights = self.pool_score(torch.cat((features, residual), -1)).softmax(1)
                pooled = (features * weights).sum(1)
        return self.head(pooled)


def within_concept(predictions, targets, rows):
    pred = F.normalize(predictions.float(), dim=1)
    truth = F.normalize(targets.float(), dim=1)
    correct = torch.zeros(len(rows), dtype=torch.bool, device=pred.device)
    margin = torch.empty(len(rows), dtype=torch.float32, device=pred.device)
    groups = {}
    for position, row in enumerate(rows):
        groups.setdefault(row["concept_id"], []).append(position)
    for positions in groups.values():
        ids = torch.tensor(positions, device=pred.device)
        if len(positions) != 10:
            raise ValueError("Metric requires the full ten-stimulus concept")
        similarity = pred[ids] @ truth[ids].T
        expected = torch.arange(10, device=pred.device)
        # argmax resolves exact ties by registered row order, which is logged.
        correct[ids] = similarity.argmax(1) == expected
        positive = similarity.diagonal().clone()
        similarity.fill_diagonal_(-float("inf"))
        margin[ids] = positive - similarity.max(1).values
    return correct, margin


@torch.no_grad()
def evaluate(model, x, targets, ids, rows, mean, scale, batch, check_deadline):
    model.eval()
    predictions = []
    for start in range(0, len(ids), batch):
        check_deadline()
        selected = ids[start:start + batch]
        predictions.append(model((x[selected] - mean) / scale).float())
    predictions = torch.cat(predictions)
    truth = targets[ids]
    mse = (predictions - truth).square().mean(1)
    cosine = F.cosine_similarity(predictions, truth, dim=1)
    correct, margin = within_concept(predictions, truth, rows)
    return {"mse": mse, "cosine": cosine, "same_concept_top1": correct,
            "same_concept_margin": margin}, predictions


def summarize(metrics):
    return {key: float(value.float().mean().item()) for key, value in metrics.items()}


def save_per_query(path, metrics, rows, fold, arm):
    cpu = {key: value.detach().cpu().tolist() for key, value in metrics.items()}
    temp = path.with_name(path.name + ".tmp")
    fields = ["subject_id", "row", "target_row", "source_row", "concept_id", "fold", "arm", *cpu.keys()]
    with gzip.open(temp, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for position, row in enumerate(rows):
            writer.writerow({"subject_id": row["subject_id"], "row": row["row"],
                             "target_row": row["target_row"], "source_row": row.get("source_row", ""),
                             "concept_id": row["concept_id"], "fold": fold, "arm": arm,
                             **{key: values[position] for key, values in cpu.items()}})
    os.replace(temp, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", "--input-trials", dest="trials", required=True)
    parser.add_argument("--targets", "--target", dest="targets", required=True)
    parser.add_argument("--index", required=True, help="T-only row / target_row / concept_id / fold / subject_id")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", "--max_seconds", dest="max_seconds", type=float, default=1200)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--expected-rows", type=int, default=13230, help="Change only for explicit synthetic smoke tests")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=["mean", "each_mean"])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.epochs, args.batch, args.hidden, args.expected_rows) <= 0 or args.max_seconds <= 0:
        raise ValueError("Positive epochs, batch, hidden, expected-rows and deadline required")
    if len(set(args.arms)) != len(args.arms) or "mean" not in args.arms or len(args.arms) < 2:
        raise ValueError("Specify mean plus at least one distinct trial-set arm")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output must be empty; never overwrite an earlier pilot or choose a result")
    started = time.monotonic()
    deadline = started + args.max_seconds
    def check_deadline():
        if time.monotonic() >= deadline:
            raise Deadline("Fixed wall-clock limit reached; pilot remains incomplete")

    event_path = output / "epochs.jsonl"
    status = {"version": VERSION, "status": "INITIALIZING", "representation_pilot_only": True,
              "not_image_evidence": True, "not_ten_subject_evidence": True,
              "held_metrics_used_for_selection": False, "checkpoint_policy": "fixed_last_epoch",
              "args": vars(args), "completed_arms": [], "started_unix": time.time()}
    atomic_json(output / "status.json", status)
    provenance = None
    active_model = active_optimizer = None
    active = {}
    try:
        paths = {"trials": Path(args.trials).resolve(), "targets": Path(args.targets).resolve(),
                 "index": Path(args.index).resolve(), "script": Path(__file__).resolve()}
        provenance = {name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                      for name, path in paths.items()}
        source_x = np.load(paths["trials"], mmap_mode="r", allow_pickle=False)
        source_y = np.load(paths["targets"], mmap_mode="r", allow_pickle=False)
        if source_x.dtype != np.float32 or source_y.dtype != np.float32:
            raise ValueError("Trials and targets must be real float32 .npy arrays")
        rows = read_index(paths["index"])
        contract = validate_index(rows, source_x.shape, source_y.shape, args.expected_rows)
        if str(args.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        x = torch.from_numpy(np.array(source_x[[row["row"] for row in rows]], copy=True)).to(device)
        targets = torch.from_numpy(np.array(source_y[[row["target_row"] for row in rows]], copy=True)).to(device)
        if not torch.isfinite(x).all().item() or not torch.isfinite(targets).all().item():
            raise ValueError("Non-finite source value; no selective row dropping")
        if not (targets.norm(dim=1) > 0).all().item():
            raise ValueError("Zero target feature")
        del source_x, source_y
        all_results = {}
        provenance.update({"contract": contract, "torch": torch.__version__, "numpy": np.__version__,
                           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                           "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                           "normalization": "train-fold physical-mean feature mean/std, same transform for every arm/trial",
                           "loss": "raw F-target MSE only", "inference_inputs": "same four physical trials; one subject",
                           "prediction_coordinates": "raw Linear+learned LayerNorm output in fitted F-target coordinates; no L2 renormalization or query-target scaling",
                           "compute_limit": "same rows/epochs/batches and UNet sampling later; each_mean/learned_pool encode four trials versus one mean, so encoder FLOPs are not matched",
                           "outer_folds": "all concepts/ten stimuli stay together; no V/D/official test reads",
                           "source_mean_identity": "launcher responsibility; not asserted by this script",
                           "candidate_tie_policy": "argmax uses ascending registered row order"})
        atomic_json(output / "provenance.json", provenance)
        for fold in (0, 1):
            check_deadline()
            train_positions = [i for i, row in enumerate(rows) if row["fold"] != fold]
            held_positions = [i for i, row in enumerate(rows) if row["fold"] == fold]
            train_ids = torch.tensor(train_positions, device=device)
            held_ids = torch.tensor(held_positions, device=device)
            train_rows = [rows[i] for i in train_positions]
            held_rows = [rows[i] for i in held_positions]
            # Chunked moments avoid a second entire trial-array copy on GPU.
            sum_x = torch.zeros((1, 1, 17, 250), device=device)
            sum_x2 = torch.zeros_like(sum_x)
            for start in range(0, len(train_ids), args.batch):
                means = x[train_ids[start:start + args.batch]].mean(1, keepdim=True)
                sum_x += means.sum(0, keepdim=True)
                sum_x2 += means.square().sum(0, keepdim=True)
            center = sum_x / len(train_ids)
            scale = (sum_x2 / len(train_ids) - center.square()).clamp_min(0).sqrt().clamp_min(1e-4)
            for arm in args.arms:
                check_deadline()
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(args.seed)
                model = TrialSetPredictor(arm, args.hidden).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                parameter_count = sum(value.numel() for value in model.parameters())
                base_count = sum(value.numel() for value in TrialSetPredictor("mean", args.hidden).parameters())
                active_model, active_optimizer = model, optimizer
                active = {"fold": fold, "arm": arm, "epoch_completed": 0, "updates": 0,
                          "parameter_count": parameter_count, "mean_parameter_count": base_count,
                          "parameter_delta": parameter_count - base_count,
                          "exact_parameter_match": parameter_count == base_count}
                status.update({"status": "RUNNING", "active": active})
                atomic_json(output / "status.json", status)
                # Profile has no parameter update and does not inspect held scores.
                model.train()
                profile_ids = train_ids[:min(args.batch, len(train_ids))]
                if device.type == "cuda": torch.cuda.synchronize(device)
                profile_start = time.monotonic()
                loss = F.mse_loss(model((x[profile_ids] - center) / scale), targets[profile_ids])
                loss.backward()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda": torch.cuda.synchronize(device)
                profile = {**active, "profile_rows": len(profile_ids), "profile_forward_backward_seconds": time.monotonic() - profile_start,
                           "profile_loss": float(loss.detach().item()),
                           "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
                atomic_json(output / f"fold{fold}_{arm}_profile.json", profile)
                for epoch in range(1, args.epochs + 1):
                    epoch_start = time.monotonic()
                    # A separate CPU generator gives exactly the same orders to all arms.
                    generator = torch.Generator(device="cpu").manual_seed(args.seed + fold * 100000 + epoch)
                    order = torch.randperm(len(train_ids), generator=generator).to(device)
                    model.train()
                    train_mse_sum, train_cos_sum, count = 0.0, 0.0, 0
                    for start in range(0, len(order), args.batch):
                        check_deadline()
                        ids = train_ids[order[start:start + args.batch]]
                        prediction = model((x[ids] - center) / scale)
                        loss = F.mse_loss(prediction, targets[ids])
                        if not torch.isfinite(loss).item():
                            raise RuntimeError("Non-finite loss; keep failure, never skip a query")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                        optimizer.step()
                        train_mse_sum += float(loss.detach().item()) * len(ids)
                        train_cos_sum += float(F.cosine_similarity(prediction.detach(), targets[ids], dim=1).sum().item())
                        count += len(ids)
                        active["updates"] += 1
                    if count != len(train_ids):
                        raise RuntimeError("Training denominator mismatch")
                    metrics, predictions = evaluate(model, x, targets, held_ids, held_rows, center, scale, args.batch, check_deadline)
                    active["epoch_completed"] = epoch
                    record = {"fold": fold, "arm": arm, "epoch": epoch, "updates": active["updates"],
                              "train_rows": count, "held_rows": len(held_ids),
                              "train_online_mse": train_mse_sum / count, "train_online_cosine": train_cos_sum / count,
                              "held": summarize(metrics), "epoch_seconds": time.monotonic() - epoch_start,
                              "elapsed_seconds": time.monotonic() - started,
                              "selection_used": False}
                    with event_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        stream.flush()
                    print(json.dumps(record, allow_nan=False), flush=True)
                    checkpoint = {"version": VERSION, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                  "center": center.cpu(), "scale": scale.cpu(), "active": dict(active),
                                  "args": vars(args), "source_sha256": {name: value["sha256"] for name, value in provenance.items() if isinstance(value, dict) and "sha256" in value},
                                  "train_rows": [row["row"] for row in train_rows], "held_rows": [row["row"] for row in held_rows],
                                  "checkpoint_policy": "latest_epoch_for_recovery; only requested final epoch is a result"}
                    atomic_torch(output / f"fold{fold}_{arm}_latest.pt", checkpoint)
                    status["active"] = dict(active)
                    atomic_json(output / "status.json", status)
                result_metrics = {key: value.detach().cpu() for key, value in metrics.items()}
                all_results[(fold, arm)] = {"metrics": result_metrics, "rows": held_rows,
                                            "predictions": predictions.detach().cpu()}
                per_query = output / f"fold{fold}_{arm}_per_query.csv.gz"
                save_per_query(per_query, result_metrics, held_rows, fold, arm)
                status["completed_arms"].append({**active, "final": summarize(metrics),
                                                  "per_query_sha256": sha256(per_query),
                                                  "checkpoint_sha256": sha256(output / f"fold{fold}_{arm}_latest.pt")})
                atomic_json(output / "status.json", status)
                active_model = active_optimizer = None
                del predictions, metrics, model, optimizer
        # Only complete outer-fold coverage can produce a final comparison.
        summary = {"status": "COMPLETE_REPRESENTATION_PILOT", "rows_per_arm": len(rows), "subject_id": contract["subject_id"],
                   "fixed_final_epoch": args.epochs, "selection_used": False, "arms": {}, "paired_vs_mean": {},
                   "claim_limit": "single-subject T-only OOF representation screening; no image/latency/physiology/SOTA conclusion"}
        prediction_index = output / "prediction_index.jsonl"
        prediction_index.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        summary["prediction_index"] = str(prediction_index)
        summary["prediction_index_sha256"] = sha256(prediction_index)
        positions_by_row = {row["row"]: i for i, row in enumerate(rows)}
        for arm in args.arms:
            combined = {key: torch.cat([all_results[(fold, arm)]["metrics"][key] for fold in (0, 1)])
                        for key in all_results[(0, arm)]["metrics"]}
            summary["arms"][arm] = summarize(combined)
            parameter_receipt = next(item for item in status["completed_arms"] if item["arm"] == arm)
            summary["arms"][arm].update({key: parameter_receipt[key] for key in (
                "parameter_count", "mean_parameter_count", "parameter_delta", "exact_parameter_match")})
            summary["arms"][arm]["encoder_calls_per_query"] = 1 if arm == "mean" else 4
            raw_predictions = np.empty((len(rows), 1024), dtype=np.float32)
            prediction_coverage = np.zeros(len(rows), dtype=np.int32)
            for fold in (0, 1):
                result = all_results[(fold, arm)]
                positions = [positions_by_row[row["row"]] for row in result["rows"]]
                raw_predictions[positions] = result["predictions"].numpy()
                prediction_coverage[positions] += 1
            if not (prediction_coverage == 1).all() or not np.isfinite(raw_predictions).all():
                raise RuntimeError("OOF prediction coverage/finite check failed")
            arm_directory = output / arm
            arm_directory.mkdir()
            prediction_path = arm_directory / "predictions.npy"
            atomic_npy(prediction_path, raw_predictions)
            summary["arms"][arm].update({"predictions_path": str(prediction_path),
                                           "predictions_sha256": sha256(prediction_path),
                                           "predictions_shape": list(raw_predictions.shape),
                                           "predictions_norm_mean": float(np.linalg.norm(raw_predictions, axis=1).mean()),
                                           "prediction_coordinates": provenance["prediction_coordinates"],
                                           "oof_coverage_min": int(prediction_coverage.min()),
                                           "oof_coverage_max": int(prediction_coverage.max())})
            if arm != "mean":
                differences = {key: torch.cat([all_results[(fold, arm)]["metrics"][key].float() - all_results[(fold, "mean")]["metrics"][key].float() for fold in (0, 1)])
                               for key in combined}
                summary["paired_vs_mean"][arm] = summarize(differences)
        after_hashes = {name: sha256(path) for name, path in paths.items()}
        if any(after_hashes[name] != provenance[name]["sha256"] for name in paths):
            raise RuntimeError("Input/script changed during the run")
        summary["input_after_sha256"] = after_hashes
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        atomic_json(output / "summary.json", summary)
        status.update({"status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"],
                       "summary_sha256": sha256(output / "summary.json")})
        atomic_json(output / "status.json", status)
        return 0
    except BaseException as error:
        # Preserve failures and completed folds without selecting the shorter run.
        if active_model is not None:
            atomic_torch(output / "interrupted.pt", {"model": active_model.state_dict(),
                                                       "optimizer": active_optimizer.state_dict(), "active": active,
                                                       "not_a_final_result": True})
        status.update({"status": "INCOMPLETE_TIMEOUT" if isinstance(error, Deadline) else "FAILED",
                       "error": repr(error), "traceback": traceback.format_exc(), "active": active,
                       "elapsed_seconds": time.monotonic() - started, "not_a_final_comparison": True})
        atomic_json(output / "status.json", status)
        print(status["traceback"], file=sys.stderr, flush=True)
        return 124 if isinstance(error, Deadline) else 1


if __name__ == "__main__":
    raise SystemExit(main())
