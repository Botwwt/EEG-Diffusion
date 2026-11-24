"""Decode uncalibrated sub-03 R4 EEG from the full 1,854-name vocabulary.

python fit_visual_to_neural_objects.py --archive ROOT --dictionary dictionary.npz \
    --external external_anchors.npz --out run

The primary text-only rule and equal forward/reverse weights are fixed before
evaluation. THINGSplus photo anchors form a separate reference-assisted readout.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from decode_eeg_layout import EEG_SHA, digest, eeg_design, evaluation_rows, registration, save_json
from fit_object_words import dictionary, labels_for_rows, visual_targets
from fit_official_continuous import regress
from fit_photo_conditions import normalize

PRIMARY = "text_bidirectional"
RIDGE = 0.3
THINGSPLUS_SHA = "ed86912f6848ac5cf8b405a727f875295cc081cdd64271e80ae4d607b7d447e3"
H14_CHECKPOINT_SHA = "036e6e2bd49697511f4f8b8cb5ee465f93025f7a69a145eadeb9a881ace9b18d"


def load_anchors(path, ids):
    with np.load(path, allow_pickle=False) as z:
        words = z["unique_ids"].astype(str)
        features = z["embeddings"].astype(np.float32)
        eligible = z["eligible"]
        if (
            words.shape != (1854,)
            or len(set(words)) != 1854
            or set(words) != set(ids)
            or features.shape != (1854, 1024)
            or eligible.dtype != np.bool_
            or eligible.shape != (1854,)
            or not eligible.all()
            or not np.isfinite(features).all()
            or np.any(np.linalg.norm(features, axis=1) == 0)
        ):
            raise ValueError(
                "The full external comparison needs one eligible visual vector per THINGS concept"
            )
        ix = {w: i for i, w in enumerate(words)}
        meta = {k: str(z[k].item()) for k in ("source_sha256", "model_sha256")}
        if meta != {"source_sha256": THINGSPLUS_SHA, "model_sha256": H14_CHECKPOINT_SHA}:
            raise ValueError(
                "External gallery archive or LAION ViT-H/14 checkpoint differs from the registered release"
            )
    return normalize(features[[ix[w] for w in ids]]), meta


def standard(s):
    return (s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True).clip(1e-6)


def predict(train, train_word_ids, train_images, query, text, external):
    from fit_registered_objects import eeg_projection

    groups, labels = np.unique(train_word_ids, return_inverse=True)
    if not np.all(np.bincount(labels) == 10) or not np.all(
        train_word_ids.reshape(-1, 10) == train_word_ids.reshape(-1, 10)[:, :1]
    ):
        raise ValueError("Expected consecutive complete ten-photo training classes")
    centers = np.stack([train[labels == i].mean(0) for i in range(len(groups))])
    expected = {"text": regress(text[groups], centers, text, RIDGE)}
    if external is not None:
        expected["external_photo"] = regress(train_images, train, external, RIDGE)
    # Each forward model predicts the original 17-channel multiscale EEG
    # measurements. Learn the within-class whitening and 96 Fisher directions
    # on T only, then apply that *same* transform to queries and predicted EEG.
    parts = [train, query, *expected.values()]
    fish = eeg_projection(np.concatenate(parts), np.arange(len(train)))
    ft, fq, *predicted_eeg = np.split(fish, np.cumsum([len(x) for x in parts[:-1]]))
    neural_mean = ft.mean(0)
    q = normalize(fq - neural_mean)
    scores = {}
    for (family, _), fy in zip(expected.items(), predicted_eeg, strict=True):
        if family == "text":
            teacher, bank = text[train_word_ids], text
        else:
            teacher, bank = train_images, external
        # Reverse predicts H14 features from a query's EEG. The reverse
        # prototype centering uses only the training semantic feature mean.
        mean = teacher.mean(0)
        reverse = regress(ft, teacher, fq, RIDGE)
        a = q @ normalize(fy - neural_mean).T
        b = normalize(reverse - mean) @ normalize(bank - mean).T
        scores[family + "_to_neural"] = a
        scores["neural_to_" + family] = b
        scores[family + "_bidirectional"] = 0.5 * standard(a) + 0.5 * standard(b)
    if any(v.shape != (len(query), len(text)) or not np.isfinite(v).all() for v in scores.values()):
        raise ValueError("Invalid full-vocabulary scores")
    return scores


def csv_file(path, fields, rows):
    with path.open("w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(rows)


def digest_audit(manifest, records, train, panels, ids):
    if manifest is None or not manifest.is_file():
        return {
            "status": "No external file manifest supplied; visual near-duplicate audit previously covered only official R80."
        }
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(ids) or {r["unique_id"] for r in rows} != set(ids):
        raise ValueError("External file hashes and full vocabulary disagree")
    external = {r["sha256"] for r in rows}
    if not all(len(x) == 64 for x in external):
        raise ValueError("Invalid external photo hashes")
    scopes = {"all_archive_train_split": np.arange(len(records)), "historical_T": train, **panels}
    return dict(
        source_manifest_sha256=digest(manifest),
        method="Exact file SHA-256 only, evaluated after predictions. Re-encodings, altered crops and near-duplicates remain unresolved for R4.",
        exact_matching_photo_rows={
            k: sum(records[int(i)]["image_sha256"] in external for i in v)
            for k, v in scopes.items()
        },
    )


def rank_summary(r):
    return dict(
        n=len(r),
        top1_hits=int((r == 1).sum()),
        top5_hits=int((r <= 5).sum()),
        median_rank=float(np.median(r)),
        mrr=float(np.mean(1 / r)),
    )


def run(args):
    import torch

    records, concept, split, channels = registration(args.archive)
    ids, words, things, text, dict_meta = dictionary(args.dictionary)
    ids, words = np.array(ids), np.array(words)
    train = np.sort(split["T"])
    panels = evaluation_rows(concept, split, args.dev_starts, 20)
    if (
        len(train) != 13230
        or len(np.unique(concept[train])) != 1323
        or any(len(v) != 200 for v in panels.values())
        or set(concept[panels["dev"]]) & set(concept[panels["confirm"]])
        or any(set(concept[train]) & set(concept[v]) for v in panels.values())
    ):
        raise ValueError("Invalid held-concept T/dev/confirm registration")
    object_names = [r["concept_id"] for r in records]
    train_ids = labels_for_rows(object_names, train, ids, things)
    ext, ext_meta = (None, None) if not args.external else load_anchors(args.external, ids)
    images, axis = (
        (None, None) if ext is None else visual_targets(args.archive, train, text, train_ids)
    )
    if ext is not None and images is None:
        raise ValueError("The training H14 image-to-text coordinate audit failed")
    source = args.archive / "data/recovery_20261108_v236/sub-03_train_mean4.pt"
    if digest(source) != EEG_SHA:
        raise ValueError("Unexpected sub-03 mean-four EEG checksum")
    eeg = torch.load(source, mmap=True, weights_only=True, map_location="cpu").numpy()
    rows = np.concatenate(list(panels.values()))
    scores = predict(
        eeg_design(eeg, train, channels),
        train_ids,
        images,
        eeg_design(eeg, rows, channels),
        text,
        ext,
    )
    args.out.mkdir(exist_ok=True, parents=True)
    bounds = np.cumsum([0, *[len(v) for v in panels.values()]])
    for i, (panel, pr) in enumerate(panels.items()):
        ranked = np.argsort(-scores[PRIMARY][bounds[i] : bounds[i + 1]], axis=1, kind="stable")
        csv_file(
            args.out / f"{panel}_predictions.csv",
            ["array_row", "predicted_id", "predicted_word", "top5_ids"],
            (
                (int(row), ids[order[0]], words[order[0]], "|".join(ids[order[:5]]))
                for row, order in zip(pr, ranked, strict=True)
            ),
        )
    # The true IDs and the panel-restricted scoring set are first used here,
    # after both full-vocabulary primary prediction files have been written.
    summary, detail = {}, []
    for i, (panel, pr) in enumerate(panels.items()):
        truth = labels_for_rows(object_names, pr, ids, things)
        allowed = np.unique(truth)
        if len(allowed) != 20:
            raise ValueError("Expected twenty evaluation-only panel candidates")
        summary[panel] = {}
        for method, all_scores in scores.items():
            vals = all_scores[bounds[i] : bounds[i + 1]]
            order = np.argsort(-vals, axis=1, kind="stable")
            closed = allowed[np.argsort(-vals[:, allowed], axis=1, kind="stable")]
            rank = np.argmax(order == truth[:, None], axis=1) + 1
            rank20 = np.argmax(closed == truth[:, None], axis=1) + 1
            summary[panel][method] = dict(
                full1854=rank_summary(rank), restricted20_evaluation_only=rank_summary(rank20)
            )
            for j, row in enumerate(pr):
                detail.append(
                    (
                        panel,
                        int(row),
                        method,
                        ids[truth[j]],
                        words[truth[j]],
                        ids[order[j, 0]],
                        words[order[j, 0]],
                        "|".join(ids[order[j, :5]]),
                        int(rank[j]),
                        ids[closed[j, 0]],
                        int(rank20[j]),
                    )
                )
    csv_file(
        args.out / "scored.csv",
        [
            "panel",
            "array_row",
            "method",
            "truth_id",
            "truth_word",
            "full_predicted_id",
            "full_predicted_word",
            "full_top5_ids",
            "full_truth_rank",
            "restricted20_predicted_id",
            "restricted20_truth_rank",
        ],
        detail,
    )
    if axis is not None:
        axis = {k: v for k, v in axis.items() if k != "path"}
    manifest = args.external_manifest or (
        args.external.with_name("external_manifest.csv") if args.external else None
    )
    report = dict(
        subject="sub-03",
        repeats=4,
        train_photos=len(train),
        train_classes=len(np.unique(train_ids)),
        vocabulary=len(ids),
        primary=PRIMARY,
        rule="Prespecified before inspecting R4 query scores. Equal per-query standardized forward + reverse scores.",
        forward="Training photo H14 (external-assisted family) or class text (text family) to 17-channel multiscale EEG. Cosine after generalized Fisher within-class whitening.",
        reverse="Training EEG Fisher features to H14. Centered cosine to all 1854 text or external-photo prototypes.",
        ridge_per_training_example=RIDGE,
        fisher_dimensions=96,
        source_waveform_sha256=EEG_SHA,
        training_rows_sha256=hashlib.sha256(train.astype("<i8").tobytes()).hexdigest(),
        dictionary_sha256=digest(args.dictionary),
        dictionary_encoder=dict_meta,
        train_visual_axis_check=axis,
        external_source=ext_meta,
        external_bank_sha256=digest(args.external) if args.external else None,
        external_reference_scope="THINGSplus one photo per candidate. An existing pixel collision audit compared these anchors to the official 200 R80 photos; R4 panel near-duplicates have not been ruled out.",
        external_r4_exact_file_audit=digest_audit(manifest, records, train, panels, ids)
        if ext is not None
        else None,
        chance_full1854=1 / 1854,
        chance_restricted20=1 / 20,
        results=summary,
    )
    save_json(args.out / "summary.json", report)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "dictionary", "out"):
        p.add_argument("--" + arg, type=Path, required=True)
    for arg in ("external", "external-manifest", "dev-starts"):
        p.add_argument("--" + arg, type=Path)
    run(p.parse_args())
