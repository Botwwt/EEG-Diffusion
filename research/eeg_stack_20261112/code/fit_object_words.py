"""Predict open-vocabulary object prompts from sub-03 four-repeat EEG using all 1,854 THINGS words."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from fit_photo_conditions import FOLDS, RIDGE, SEED, multiscale_features, normalize, panel_rows
from predict_instance_starts import read_registration, read_signals

N_WORDS = 1854
H14 = "CLIP-ViT-H-14-laion2B-s32B-b79K"
METHODS = ("text_ridge", "image_ridge", "image_text_average")


def dictionary(path):
    with np.load(path, allow_pickle=False) as data:
        ids = data["unique_ids"].astype(str)
        words = data["words"].astype(str)
        things_ids = data["things_ids"]
        emb = np.asarray(data["text_embeddings"], dtype=np.float32)
        model = str(data["model"].item()) if "model" in data else None
        pretrained = str(data["pretrained"].item()) if "pretrained" in data else None
    if ids.shape != (N_WORDS,) or len(set(ids)) != N_WORDS or words.shape != ids.shape:
        raise ValueError("Expected all 1,854 unique THINGS concept IDs with readable names")
    if (
        things_ids.dtype.kind not in "iu"
        or not np.array_equal(np.sort(things_ids), np.arange(1, N_WORDS + 1))
        or not np.all(np.char.str_len(np.char.strip(words)))
    ):
        raise ValueError("Expected all original THINGS numeric IDs and nonempty words")
    if (
        emb.shape != (N_WORDS, 1024)
        or not np.isfinite(emb).all()
        or not np.allclose(np.linalg.norm(emb, axis=1), 1, rtol=0, atol=0.003)
    ):
        raise ValueError("Expected one unit-norm H14 text embedding per dictionary row")
    if model != "ViT-H-14":
        raise ValueError(f"Wrong dictionary model: {model}")
    if pretrained != "laion2b_s32b_b79k":
        raise ValueError(f"Wrong dictionary checkpoint: {pretrained}")
    return ids, words, things_ids, normalize(emb), {"model": model, "pretrained": pretrained}


def labels_for_rows(labels, rows, unique_ids, things_ids):
    positions = {int(thing): i for i, thing in enumerate(things_ids)}
    target = np.empty(len(rows), dtype=np.intp)
    for i, r in enumerate(rows):
        prefix, slug = labels[r].split("_", 1)
        which = positions.get(int(prefix))
        if which is None or unique_ids[which] != slug:
            raise ValueError(f"THINGS numeric ID and uniqueID do not both match {labels[r]}")
        target[i] = which
    return target


def ranks(scores, labels):
    # Stable dictionary order breaks equal scores, identically for all modes and exported words.
    ref = scores[np.arange(len(labels)), labels]
    ahead = (scores > ref[:, None]) | (
        (scores == ref[:, None]) & (np.arange(scores.shape[1]) < labels[:, None])
    )
    return ahead.sum(axis=1) + 1


def metrics(rank):
    return {
        "top1": float(np.mean(rank == 1)),
        "top5": float(np.mean(rank <= 5)),
        "mrr": float(np.mean(1 / rank)),
        "median_rank": float(np.median(rank)),
    }


def visual_targets(archive, rows, text, labels):
    """Use visual supervision only when the stored checkpoint name and T-only cross-modal check agree."""
    import torch

    path = archive / "data/native_training_v239/train_visual_embeddings.pt"
    info = {
        "path": str(path.relative_to(archive)),
        "expected_image_key": H14,
        "test": "True T photographs ranked against all 1854 uncentered text embeddings; require Top-1 >= 0.05 and Top-5 >= 0.10.",
    }
    if not path.is_file():
        return None, {**info, "enabled": False, "reason": "Missing archived image features"}
    bank = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if H14 not in bank or tuple(bank[H14].shape) != (16540, 1024):
        return None, {
            **info,
            "enabled": False,
            "reason": "Missing the checkpoint-specific raw visual projection",
        }
    v = bank[H14].numpy()[rows].astype(np.float32)
    if not np.isfinite(v).all() or not np.all(np.linalg.norm(v, axis=1) > 0):
        raise ValueError("Invalid raw H14 image embeddings in historical T")
    v = normalize(v)
    ranks_t = []
    for i in range(0, len(v), 256):
        scores = v[i : i + 256] @ text.T
        ranks_t.append(ranks(scores, labels[i : i + len(scores)]))
    audit = metrics(np.concatenate(ranks_t))
    passed = audit["top1"] >= 0.05 and audit["top5"] >= 0.10
    return v if passed else None, {**info, "enabled": passed, "image_to_text_in_T": audit}


def score_modes(train_eeg, query_eeg, train_words, word_bank, train_images):
    # Solve the identical EEG ridge for word supervision and optional paired H14 image supervision.
    mu = train_eeg.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = train_eeg.std(axis=0, dtype=np.float64).clip(1e-6).astype(np.float32)
    x = (train_eeg - mu) / sd
    targets = word_bank[train_words]
    text_mean = targets.mean(axis=0)
    gallery = normalize(word_bank - text_mean)
    targets = targets - text_mean
    if train_images is not None:
        targets = np.column_stack((targets, train_images - train_images.mean(axis=0)))
    gram = (x.T @ x).astype(np.float64)
    gram.flat[:: len(gram) + 1] += len(x) * RIDGE
    weight = np.linalg.solve(gram, (x.T @ targets).astype(np.float64)).astype(np.float32)
    pred = ((query_eeg - mu) / sd) @ weight
    text_scores = normalize(pred[:, : word_bank.shape[1]]) @ gallery.T
    result = {"text_ridge": text_scores}
    if train_images is not None:
        image_scores = normalize(pred[:, word_bank.shape[1] :]) @ gallery.T
        result.update(image_ridge=image_scores, image_text_average=(image_scores + text_scores) / 2)
    return result


def lock_mode(x, labels, text, image):
    shuffled = np.random.default_rng(SEED).permutation(np.unique(labels))
    all_ranks = {}
    folds = []
    for k, concepts in enumerate(np.array_split(shuffled, FOLDS)):
        held = np.isin(labels, concepts)
        scores = score_modes(
            x[~held], x[held], labels[~held], text, image[~held] if image is not None else None
        )
        record = {}
        for key, value in scores.items():
            rank = ranks(value, labels[held])
            record[key] = metrics(rank)
            all_ranks.setdefault(key, []).append(rank)
        folds.append(
            {"fold": k, "concepts": len(concepts), "photos": int(held.sum()), "metrics": record}
        )
    aggregate = {m: metrics(np.concatenate(parts)) for m, parts in all_ranks.items()}
    winner = max(
        aggregate, key=lambda m: (aggregate[m]["top1"], aggregate[m]["top5"], aggregate[m]["mrr"])
    )
    return winner, folds, aggregate


def run(args):
    ids, words, things_ids, bank, provenance = dictionary(args.dictionary)
    t, concepts, _, labels, channels, _ = read_registration(args.archive)
    if not np.array_equal(np.load(args.train_prep / "train_concept.npy"), concepts[t]):
        raise ValueError("Prepared historical training rows disagree with the archive")
    panels = {
        k: panel_rows(args.panels / f"{k}_layout.npz", t, concepts) for k in ("dev", "confirm")
    }
    if np.intersect1d(concepts[panels["dev"]], concepts[panels["confirm"]]).size:
        raise ValueError("Development and confirmation concepts overlap")
    split = json.loads(
        (args.archive / "data/recovery_20261108_v236/current_split_manifest.json").read_text()
    )
    if not np.isin(panels["confirm"], split["image_rows"]["V"]).all():
        raise ValueError("Confirmation must come from historical V")
    train_y = labels_for_rows(labels, t, ids, things_ids)
    image, axis_audit = visual_targets(args.archive, t, bank, train_y)
    train, query = read_signals(
        args.train_prep,
        args.archive / "data/recovery_20261108_v236/sub-03_train_mean4.pt",
        t,
        np.concatenate(list(panels.values())),
        channels,
    )
    x = multiscale_features(train)
    selected, folds, oof = lock_mode(x, train_y, bank, image)
    print(f"Locked full-vocabulary decoder from historical T: {selected}", flush=True)
    scores = score_modes(x, multiscale_features(query), train_y, bank, image)[selected]
    if scores.shape != (len(query), N_WORDS) or not np.isfinite(scores).all():
        raise ValueError("Invalid full-dictionary predictions")
    report = {
        "selected": selected,
        "subject": "sub-03",
        "physical_trials_per_mean": 4,
        "dictionary_sha256": hashlib.sha256(args.dictionary.read_bytes()).hexdigest(),
        "dictionary_provenance": provenance,
        "visual_axis_check": axis_audit,
        "candidates": N_WORDS,
        "chance_top1": 1 / N_WORDS,
        "chance_top5": 5 / N_WORDS,
        "training_photos": len(t),
        "training_classes": len(np.unique(train_y)),
        "training_row_sha256_i64": hashlib.sha256(np.asarray(t, dtype="<i8").tobytes()).hexdigest(),
        "ridge_per_row": RIDGE,
        "fold_seed": SEED,
        "inner_folds": folds,
        "out_of_fold": oof,
        "selection": "Highest out-of-concept full-vocabulary Top-1 on T, then Top-5, then MRR. Exact ties favor text_ridge.",
        "score_definition": "Cosine between the centered EEG-predicted embedding and each T-mean-centered text feature; image_text_average equally weights text- and image-trained scores.",
        "panels": {},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out) as temp:
        stage = Path(temp)
        pos = 0
        for name, rows in panels.items():
            panel = scores[pos : pos + len(rows)]
            pos += len(rows)
            top = np.argsort(-panel, axis=1, kind="stable")[:, :5]
            payload = {
                "rows": rows.tolist(),
                "selected": selected,
                "dictionary_size": N_WORDS,
                "predicted_words": words[top[:, 0]].tolist(),
                "top5_unique_ids": ids[top].tolist(),
                "top5_words": words[top].tolist(),
                "top5_things_ids": things_ids[top].tolist(),
                "top5_scores": np.take_along_axis(panel, top, axis=1).tolist(),
            }
            (stage / f"{name}_words.json").write_text(
                json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
            )
            if name in args.score_panels:
                actual = labels_for_rows(labels, rows, ids, things_ids)
                rank = ranks(panel, actual)
                report["panels"][name] = {
                    "photos": len(rows),
                    "metrics": metrics(rank),
                    "true_class_ranks": rank.tolist(),
                    "rows": rows.tolist(),
                }
        (stage / "object_words_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for f in sorted(stage.iterdir()):
            f.replace(args.out / f.name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "train-prep", "panels", "dictionary", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument(
        "--score-panels",
        nargs="*",
        choices=("dev", "confirm"),
        default=["dev", "confirm"],
        help="Compute accuracy only after locking and predicting; omit names for inference only.",
    )
    run(p.parse_args())
