"""Compare photo and cross-photo targets on physical EEG from held-out concepts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from paper_aggregate import bootstrap_ci, require, units


# Twenty millisecond ERP averages. The same windows are used for both targets.
WINDOWS = {"early": (60, 200), "late": (160, 600), "full": (0, 800)}
METRICS = ("name_top1", "name_top5", "photo_top1")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def normalize(a):
    norm = np.linalg.norm(a, axis=-1, keepdims=True)
    require(np.isfinite(norm).all() and (norm > 0).all(), "Invalid feature norms")
    return a / norm


def pool_trials(raw, rows):
    pooled = np.empty((len(rows), 17, 40), dtype=np.float32)
    for chunk in np.array_split(np.arange(len(rows)), (len(rows) + 255) // 256):
        trials = raw[rows[chunk], :, :, :200].astype(np.float32)
        pooled[chunk] = trials.reshape(len(chunk), 4, 17, 40, 5).mean(axis=(1, 4))
    require(np.isfinite(pooled).all(), "Nonfinite EEG")
    return pooled


def window_features(tensor, bounds):
    start, end = (t // 20 for t in bounds)
    return tensor[..., start:end].reshape(len(tensor), -1)


def leave_photo_out(visual, group):
    counts = np.bincount(group)
    require(np.all(counts == 10), "Expected ten separate photos per concept")
    sums = np.zeros((len(counts), visual.shape[1]), dtype=visual.dtype)
    np.add.at(sums, group, visual)
    return (sums[group] - visual) / (counts[group, None] - 1)


def top_credit(scores, target, k):
    query = np.broadcast_to(target, scores.shape[:-1])
    truth = np.take_along_axis(scores, query[..., None], axis=-1)
    return np.clip((k - (scores > truth).sum(-1)) / (scores == truth).sum(-1), 0, 1)


def measure(features, bank, truth, names):
    x = normalize(features)
    name_scores = x @ bank.T
    photo_scores = x @ truth.T
    return np.stack((top_credit(name_scores, names, 1), top_credit(name_scores, names, 5),
                     top_credit(photo_scores, np.arange(len(names)), 1)), axis=-1)


def run(benchmark, photo_path, output):
    prep = benchmark / "verified/data/prep"
    raw_dir = benchmark / "inputs/prep17"
    bank_dir = benchmark / "token-rank-check"
    d2_dir = benchmark / "semantic-energy-check"
    sources = dict(split=prep / "paper_split.json", development=prep / "paper_development.json",
                   labels=prep / "train_concept.npy", bank=bank_dir / "naming-frozen-bank.npz",
                   truth=photo_path, registry=benchmark / "inputs/image_set/train_photos.csv",
                   physical_ids=raw_dir / "train_ids.json")
    proof = {key: sha(path) for key, path in sources.items()}
    reference = read_json(d2_dir / "matching-targets-verified.json")["input"]
    require(all(proof[k] == checksum for k, checksum in reference.items()),
            "The cached D2 comparison used different data")
    require(proof["bank"] == read_json(bank_dir / "naming-frozen-bank.json")["bank_sha256"],
            "Frozen evaluation bank has changed")
    train = np.asarray(read_json(sources["split"])["train_rows"])
    held = np.asarray(read_json(sources["development"])["dev_rows"])
    labels = np.load(sources["labels"], allow_pickle=False)
    require(len(np.unique(train)) == len(train) == 13230 and len(np.unique(held)) == len(held) == 200
            and set(labels[train]).isdisjoint(labels[held]), "Expected a whole-concept holdout")
    with sources["registry"].open(encoding="utf-8-sig", newline="") as stream:
        registry = list(csv.DictReader(stream))
    require(len(registry) == 16540 and all(int(r["row0"]) == i for i, r in enumerate(registry)),
            "Expected the official training image order")
    require([r["image_id"] for r in registry] == read_json(sources["physical_ids"])["ids"],
            "Physical trial and photo orders differ")
    concepts = np.array([registry[row]["unique_id"] for row in held])
    with np.load(sources["bank"], allow_pickle=False) as data:
        bank_labels, bank = data["concept_id"].tolist(), normalize(data["h14_only_vae_mean"])
    require(len(set(bank_labels)) == 1854 and bank.shape == (1854, 1024), "Invalid naming bank")
    dictionary = {name: i for i, name in enumerate(bank_labels)}
    names = np.array([dictionary[c] for c in concepts])
    image = np.load(photo_path, mmap_mode="r", allow_pickle=False)
    require(image.shape == (16540, 1024), "Invalid frozen visual features")
    fit_images, held_images = image[train], normalize(image[held])
    group = np.unique(labels[train], return_inverse=True)[1]
    targets = np.concatenate((fit_images, leave_photo_out(fit_images, group)), axis=-1).astype(np.float32)
    intercept = targets.mean(0)
    train_eeg_sum = np.zeros((len(train), 17, 40), dtype=np.float32)
    held_eeg = []
    fit_gram = {key: np.zeros((17 * ((end - start) // 20),) * 2, dtype=np.float64)
                for key, (start, end) in WINDOWS.items()}
    raw_receipts = {}
    d2_values = []
    for subject in range(1, 11):
        raw_path = raw_dir / f"sub-{subject:02}_train.npy"
        physical = read_json(raw_path.with_suffix(".json"))
        raw_receipts[str(subject)] = digest = sha(raw_path)
        require(physical["output_sha256"] == digest and physical["ids_sha256"] == proof["physical_ids"]
                and physical["sample_rate_hz"] == 250 and physical["checked_physical_photo_ids"] == 66160,
                "Physical EEG provenance changed")
        raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        require(raw.shape == (16540, 4, 17, 250) and raw.dtype == np.float16, "Invalid physical EEG")
        fit = pool_trials(raw, train)
        center = fit.mean(0)
        fit -= center
        train_eeg_sum += fit
        held_eeg.append(pool_trials(raw, held) - center)
        for key, bounds in WINDOWS.items():
            z = window_features(fit, bounds)
            fit_gram[key] += z.T @ z
        d2_path = d2_dir / f"d2-train-dev-sub-{subject:02}.npz"
        require(sha(d2_path) == read_json(d2_path.with_suffix(".json"))["data_sha256"], "D2 data changed")
        with np.load(d2_path, allow_pickle=False) as d2:
            require(np.array_equal(d2["dev_rows"], held), "D2 condition is not paired")
            d2_values.append(d2["dev_fused"])
        print(f"Verified and pooled subject {subject:02}", flush=True)
    count = len(train) * 10
    held_eeg = np.stack(held_eeg)
    scores = {"d2": measure(np.stack(d2_values), bank, held_images, names)}
    predictions, penalties = {}, {}
    for key, bounds in WINDOWS.items():
        cov = fit_gram[key] / count
        ridge = .1 * np.trace(cov) / len(cov)
        penalties[key] = float(ridge)
        cov.flat[::len(cov) + 1] += ridge
        rhs = window_features(train_eeg_sum, bounds).T @ (targets - intercept) / count
        weights = np.linalg.solve(cov, rhs).astype(np.float32)
        x = window_features(held_eeg.reshape(-1, 17, 40), bounds)
        v = (x @ weights + intercept).reshape(10, len(held), -1)
        for label, prediction in zip(("photo", "other-nine"), np.split(v, 2, axis=-1), strict=True):
            arm = f"{key}-{label}"
            predictions[arm] = prediction
            scores[arm] = measure(prediction, bank, held_images, names)
    contrasts = [(f"{key}-other-nine", f"{key}-photo") for key in WINDOWS]
    contrasts += [(f"{key}-other-nine", "d2") for key in WINDOWS]
    paired = {}
    for a, b in contrasts:
        delta = scores[a] - scores[b]
        lo, hi = bootstrap_ci(units(delta, concepts, "dev"))
        paired[f"{a} minus {b}"] = {
            key: [float(delta[:, :, i].mean()), float(lo[i]), float(hi[i])]
            for i, key in enumerate(METRICS)
        }
    output.mkdir(parents=True, exist_ok=True)
    arrays = output / "raw-semantic-rows.npz"
    np.savez_compressed(arrays, row=held, concept=concepts, **predictions)
    report = dict(protocol="Physical R4 ERP to frozen fused features on whole held-out concepts",
                  subjects=10, queries=2000, held_concepts=len(set(concepts)), input=proof,
                  raw_sha256=raw_receipts, code_sha256=sha(Path(__file__)), predictions_sha256=sha(arrays),
                  windows_ms=WINDOWS, ridge=penalties, paired=paired,
                  hits={arm: dict(zip(METRICS, values.sum((0, 1)).tolist(), strict=True))
                        for arm, values in scores.items()})
    (output / "raw-semantic-verified.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["hits"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("fused_training_photos", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.benchmark, args.fused_training_photos, args.output)
