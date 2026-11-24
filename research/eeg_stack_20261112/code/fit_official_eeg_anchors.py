"""Learn full-THINGS-vocabulary EEG prototypes from the official sub-03 train export.

Run official_object_readout.py --stage prepare first. This script uses the
same --base directory as that script's --out. --export-only prepares the
mean-four training embeddings, original row index, and independent alignment
checks. All ridge/metric choices use held-out official training concepts.
"""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from official_object_readout import (
    ASSETS,
    ATM,
    acquire,
    evaluate_readout,
    sha,
    tensor_storage,
    to_csv,
    unit,
)

TRAIN_EEG = (
    ATM + "emb_eeg/ATM_S_eeg_features_sub-03_train.pt",
    "c67a3ad7357cf460ac9578ce20be7bd12fc125cdd185be39f8494e13ea6c76d4",
)
ALPHAS = (0.1, 1.0, 10.0)
METRICS = ("cosine", "noise_whitened_cosine")
SEED = 20261125


def export(base):
    eeg_pt = acquire(base / "assets/eeg_train.pt", TRAIN_EEG)
    metadata = np.load(
        acquire(base / "assets/metadata.npy", ASSETS["metadata.npy"]), allow_pickle=True
    ).item()
    photo_stems = [Path(f).stem for f in metadata["train_img_files"]]
    folders = metadata["train_img_concepts"]
    ids = [c.split("_", 1)[1] for c in folders]
    if (
        len(photo_stems) != 16540
        or len(set(photo_stems)) != 16540
        or len(set(ids)) != 1654
        or ids != np.repeat(ids[::10], 10).tolist()
    ):
        raise ValueError("Unexpected official training image row layout")
    all_trials = tensor_storage(eeg_pt, 66160, text=False).reshape(16540, 4, 1024)
    means = all_trials.mean(1, dtype=np.float32)
    # Independent identity checks on training data only. The source generation
    # notebook exports ordered loaders and repeats each visual target four times.
    ids2 = np.arange(16540).reshape(-1, 10)
    other = np.roll(ids2, -1, axis=1).ravel()
    a = unit(all_trials[:, :2].mean(axis=1))
    b = unit(all_trials[:, 2:].mean(axis=1))
    paired_half = float(np.mean(np.einsum("ij,ij->i", a, b)))
    same_concept_half = float(np.mean(np.einsum("ij,ij->i", a, b[other])))
    train_visual = acquire(base / "assets/text_train.pt", ASSETS["text_train.pt"])
    with ZipFile(train_visual) as z:
        (img_path,) = [s for s in z.namelist() if s.endswith("/data/1")]
        imgs = unit(np.frombuffer(z.read(img_path), dtype="<f4").reshape(16540, 1024))
    positive = unit(means)
    paired_img = float(np.einsum("ij,ij->i", positive, imgs).mean())
    same_concept_img = float(np.einsum("ij,ij->i", positive, imgs[other]).mean())
    shuffled_img = float(
        np.einsum(
            "ij,ij->i", positive, imgs[np.random.default_rng(SEED).permutation(len(imgs))]
        ).mean()
    )
    if not (paired_img > shuffled_img and paired_half > same_concept_half):
        raise ValueError("Training export failed image alignment or EEG grouping sanity check")
    np.save(base / "atm_train_mean4.npy", means)
    to_csv(
        base / "atm_train_rows.csv",
        ["train_row", "stimulus_stem", "unique_id", "official_folder"],
        (
            dict(train_row=i, stimulus_stem=s, unique_id=ids[i], official_folder=folders[i])
            for i, s in enumerate(photo_stems)
        ),
    )
    (base / "atm_train_stems.txt").write_text("\n".join(photo_stems) + "\n", encoding="utf8")
    checks = {
        "rows": len(means),
        "input_shape": [66160, 1024],
        "photo_shape": list(means.shape),
        "mean": "Arithmetic mean of four encoded trial vectors before normalization. Not the encoder of the four-trial voltage mean.",
        "test": "Released sub-03 R80 is the encoder output of the mean voltage. Thus train/test noise and averaging stage differ.",
        "source_url": TRAIN_EEG[0],
        "input_sha256": TRAIN_EEG[1],
        "train_eeg_half_cosine_paired": paired_half,
        "train_eeg_half_cosine_next_same_concept_photo": same_concept_half,
        "train_eeg_image_cosine_paired": paired_img,
        "train_eeg_image_cosine_next_same_concept_photo": same_concept_img,
        "train_eeg_image_cosine_random_photo": shuffled_img,
        "output_sha256": sha(base / "atm_train_mean4.npy"),
        "stimulus_stems_sha256": sha(base / "atm_train_stems.txt"),
        "author_export_example": "https://github.com/dongyangli-del/EEG_Image_decode/blob/49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d/Generation/Generation_metrics_sub8.ipynb",
        "identity_scope": "Published subject-named artifacts, train order and feature compatibility audited. The same source model produces both splits in the author example. Per-subject encoder state hashes are unavailable.",
    }
    (base / "atm_train_provenance.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(checks), flush=True)
    return means, ids


def transform_and_score(train, train_t, candidates, queries, settings):
    """Map text to expected EEG direction. A single fit treats all dictionary entries alike."""
    n, _, dim = train.shape
    class_e = train.mean(axis=1)
    e_mean = class_e.mean(axis=0)
    t_mean = train_t.mean(axis=0)
    xt = unit(train_t - t_mean)
    all_t = unit(candidates - t_mean)
    gram = (xt.T @ xt).astype(np.float64)
    eig, vectors = np.linalg.eigh(gram)
    eig = eig.clip(0)
    left = all_t @ vectors.astype(np.float32)
    right = vectors.T.astype(np.float32) @ (xt.T @ (class_e - e_mean))
    centered_query = queries - e_mean
    if any(metric == METRICS[1] for _, metric in settings):
        errors = (train - class_e[:, None, :]).reshape(-1, dim)
        covariance = (errors.T @ errors) / (n * 9)
        ev, u = np.linalg.eigh(covariance.astype(np.float64))
        whiten = (u / np.sqrt(0.5 * ev.clip(0) + 0.5 * max(ev.mean(), 1e-12))).astype(np.float32)
    for alpha in sorted(set(a for a, _ in settings)):
        reg = alpha * eig.mean()
        anchors = (left / (eig + reg).astype(np.float32)) @ right
        for a, metric in settings:
            if a != alpha:
                continue
            if metric == "cosine":
                yield (a, metric), unit(centered_query) @ unit(anchors).T, (centered_query, anchors)
            else:
                qe, aa = centered_query @ whiten, anchors @ whiten
                yield (a, metric), unit(qe) @ unit(aa).T, (qe, aa)


def fit(base, out, train_eeg, labels):
    with np.load(base / "dictionary.npz") as f:
        ids, words, text, cat = (
            f["unique_ids"],
            f["words"],
            f["text_embeddings"],
            f["manual_categories"],
        )
    lookup = {s: i for i, s in enumerate(ids)}
    obj = np.array([lookup[s] for s in labels[::10]])
    xt = unit(train_eeg).reshape(1654, 10, 1024)
    settings = [(a, m) for a in ALPHAS for m in METRICS]
    totals = {s: np.zeros(3) for s in settings}
    fold = np.empty(len(obj), dtype=int)
    fold[np.random.default_rng(SEED).permutation(len(obj))] = np.arange(len(obj)) % 5
    for f in range(5):
        train, held = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        target = np.repeat(held, 10)
        for s, score, _ in transform_and_score(
            xt[train], text[obj[train]], text[obj], xt[held].reshape(-1, 1024), settings
        ):
            correct = score[np.arange(len(target)), target]
            rank = (score > correct[:, None]).sum(axis=1) + 1
            totals[s] += [(rank == 1).sum(), np.sum(1 / rank), (rank <= 5).sum()]
        print(f"Finished training-only validation fold {f + 1}/5", flush=True)
    best = max(settings, key=lambda s: (totals[s][0], totals[s][1], -settings.index(s)))
    metrics = [
        {
            "alpha": a,
            "metric": m,
            "top1": totals[a, m][0] / len(labels),
            "mrr": totals[a, m][1] / len(labels),
            "top5": totals[a, m][2] / len(labels),
        }
        for a, m in settings
    ]
    cfg = {
        "seed": SEED,
        "selection": "Highest pooled held-out-training-concept photo top1 in the full 1654 training vocabulary, tie by MRR then declared order.",
        "rows_per_concept": 10,
        "train_folds": 5,
        "selected": {"alpha": best[0], "metric": best[1]},
        "oof": metrics,
        "ridge": "Text-to-EEG centered prototypes. Alpha multiplies mean eigenvalue. Optional whitening uses 50% within-class residual covariance and 50% isotropic covariance.",
        "test_vocabulary": 1854,
        "test_features": "Author released sub-03 R80; labels first read by independent evaluator after predictions were saved.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_selection.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf8")
    queries = unit(np.load(base / "eeg_r80.npy"))
    _, score, (qe, aa) = next(transform_and_score(xt, text[obj], text, queries, [best]))
    rank = np.argsort(-score, axis=1, kind="stable")
    catsets = [set(v.strip() for v in c.split(",") if v.strip()) for c in cat]
    categories = sorted(set().union(*catsets))
    # Class-independent broad-category prototypes in the learned EEG geometry.
    normalized = unit(aa)
    broad_anchor = unit(
        np.stack([normalized[[k in group for group in catsets]].mean(axis=0) for k in categories])
    )
    broad = np.asarray(categories)[(unit(qe) @ broad_anchor.T).argmax(axis=1)]
    to_csv(
        out / "predictions.csv",
        ["eeg_row", "predicted_id", "predicted_word", "top5_ids", "broad_category"],
        (
            dict(
                eeg_row=i,
                predicted_id=ids[r[0]],
                predicted_word=words[r[0]],
                top5_ids="|".join(ids[r[:5]]),
                broad_category=broad[i],
            )
            for i, r in enumerate(rank)
        ),
    )
    np.savez(
        out / "readout.npz",
        ranked_ids=ids[rank],
        broad_predictions=broad,
        category_names=categories,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    means, labels = export(args.base)
    if not args.export_only:
        dest = args.out or args.base / "eeg-anchors"
        fit(args.base, dest, means, labels)
        evaluate_readout(dest, False, reference=args.base, broad_space="learned EEG")
