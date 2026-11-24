"""Infer the object words of official R80 EEG using training-photo visual anchors.

Run official_object_readout.py --stage prepare first. The optional --cohort10
downloads the nine other published ATM EEG outputs for the same 200 stimuli.
Only the official *training* image feature archive is opened.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from official_object_readout import (
    ASSETS,
    ATM,
    acquire,
    object_ids,
    sha,
    tensor_storage,
    to_csv,
    unit,
)

EEG_HASHES = (
    "64d6eff4f2bac98464e4f1f570817c53f0ff8f6ff6f533c4b10c6ffce4c1c0aa",
    "fa07bb080ff8324ab95f2679a5381ea584b59896612f1a04d1d6b376b130470f",
    ASSETS["eeg.pt"][1],
    "1878b71916e0ff2cec8639198e005f1646f90fcf5c0f86cf488dd472e2f715df",
    "f2cf2de016ed2ed6f960e4aa3bf0ed0c3c190656a85571a9a6f6b2357d6fcc72",
    "82f08912363f8fb5a96dcbb4f65869d0f6b7cc1d4ad0c61a2ef8c246758283b1",
    "855f2757e76dfc4a7d7f31884c86adea104bd274da9ad95e2b0268618bb672d2",
    "968512fc0584bfe630e12ffe7987382a95d32a3d36a5c70e297d27dc08da112c",
    "e12e620572fba4dbebea21e7946e4b3f0fdc23b72ba1ea46c245cb85ad19b244",
    "6544764addaae8b05b8aa2367bd8d3f3f45bc622ccc0da1eec01698dfdf5c57d",
)
ALPHAS = (0.25, 1.0)
GEOMETRIES = ("cosine", "center", "axis", "diag_axis")


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf8")


def train_data(base, ids, text):
    path = acquire(base / "assets/text_train.pt", ASSETS["text_train.pt"])
    meta_path = acquire(base / "assets/metadata.npy", ASSETS["metadata.npy"])
    groups = np.array(
        object_ids(np.load(meta_path, allow_pickle=True).item()["train_img_concepts"])
    )
    groups = groups.reshape(1654, 10)
    if not np.all(groups == groups[:, :1]) or len(set(groups[:, 0])) != 1654:
        raise ValueError("Training storage requires ten consecutive photos per concept")
    lookup = {word: i for i, word in enumerate(ids)}
    candidates = np.array([lookup[word] for word in groups[:, 0]])
    author_text = unit(tensor_storage(path, 1654, text=True))
    if not np.allclose(text[candidates], author_text, rtol=1e-5, atol=1e-6):
        raise ValueError("Training photo group order disagrees with the author's aligned text")
    # Pinned archive has row-major float32 img_features in storage 1.
    # Storage 0 above is its ordered training text. No test feature bundle is used.
    with ZipFile(path) as z:
        (entry,) = [i for i in z.infolist() if i.filename.endswith("/data/1")]
        if entry.file_size != 16540 * 1024 * 4:
            raise ValueError("Unexpected size for the training image tensor")
        photos = unit(np.frombuffer(z.read(entry), dtype="<f4").reshape(16540, 1024))
    return candidates, photos.reshape(1654, 10, 1024)


def kernels(text):
    cosine = text @ text.T
    return {"linear": cosine, "cubic_mix": (cosine + cosine**3) * 0.5}


def prototype_fits(kernel, fit, centers, alphas):
    """Centered kernel ridge for all words, with PRESS predictions for fitted words."""
    cross = kernel[:, fit].astype(np.float64)
    reference = cross[fit]
    cross -= cross.mean(axis=1, keepdims=True)
    cross -= reference.mean(axis=0)
    cross += reference.mean()
    eigen, vec = np.linalg.eigh(cross[fit])
    eigen = np.maximum(eigen, 0)
    coords = cross @ vec
    mean = centers.mean(axis=0, dtype=np.float64)
    y = centers - mean
    projected = vec.T @ y
    for alpha in alphas:
        pred = (coords / (eigen + alpha)) @ projected + mean
        leverage = (vec**2) @ (eigen / (eigen + alpha)) + 1 / len(fit)
        pred[fit] = (pred[fit] - leverage[:, None] * centers) / (1 - leverage[:, None])
        yield alpha, pred.astype(np.float32)


def fit_scales(photos):
    all_photos = photos.reshape(-1, photos.shape[-1])
    center = all_photos.mean(axis=0, dtype=np.float64).astype(np.float32)
    variance = all_photos.var(axis=0, dtype=np.float64).astype(np.float32)
    inv_std = (0.5 * variance + 0.5 * variance.mean()) ** -0.5
    return center, inv_std


def geometry(values, scales, mode):
    center, inv_std = scales
    if mode == "cosine":
        return unit(values)
    if mode == "center":
        return unit(values - center)
    if mode == "diag_axis":
        values = values * inv_std
        center = center * inv_std
    axis = center / np.linalg.norm(center)
    return unit(values - (values @ axis)[:, None] * axis)


def rank_summary(ranks):
    return {
        "n": len(ranks),
        "top1_hits": int(np.sum(ranks == 1)),
        "top5_hits": int(np.sum(ranks <= 5)),
        "top10_hits": int(np.sum(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
        "mrr": float(np.mean(1 / ranks)),
    }


def select(kernels_by_name, train, photos):
    # Every class is scored exactly once, with every one of its photos excluded
    # from training. All 1,854 candidates compete, including every unseen word.
    folds = np.array_split(np.random.default_rng(20261126).permutation(len(train)), 3)
    centers = photos.mean(axis=1)
    records = {}
    for i, val in enumerate(folds):
        print(f"Training-only visual validation fold {i + 1}/{len(folds)}", flush=True)
        keep = np.ones(len(train), dtype=bool)
        keep[val] = False
        scales = fit_scales(photos[keep])
        truth = np.repeat(train[val], photos.shape[1])
        measures = {g: geometry(photos[val].reshape(len(truth), -1), scales, g) for g in GEOMETRIES}
        for family, kernel in kernels_by_name.items():
            for alpha, anchors in prototype_fits(kernel, train[keep], centers[keep], ALPHAS):
                for mode, queries in measures.items():
                    scores = queries @ geometry(anchors, scales, mode).T
                    target = scores[np.arange(len(truth)), truth]
                    ranks = 1 + np.sum(scores > target[:, None], axis=1)
                    records.setdefault((family, alpha, mode), []).append(ranks)
    scores = {k: rank_summary(np.concatenate(v)) for k, v in records.items()}
    choice = max(scores, key=lambda k: (scores[k]["top1_hits"], scores[k]["mrr"]))
    return choice, scores


def released_eeg(base, cache, subject):
    fname = f"ATM_S_eeg_features_sub-{subject:02d}_test.pt"
    local = base / "assets/eeg.pt" if subject == 3 else cache / fname
    path = acquire(local, (ATM + "emb_eeg/" + fname, EEG_HASHES[subject - 1]))
    return unit(tensor_storage(path, 200, text=False))


def rank_and_write(out, query, ids, words, categories, anchors, scales, mode):
    out.mkdir(exist_ok=True, parents=True)
    query = geometry(unit(query), scales, mode)
    scores = query @ anchors.T
    ranking = np.argsort(-scores, axis=1, kind="stable")
    names = sorted(set().union(*categories))
    broad_prototypes = unit(
        np.stack([anchors[[c in row for row in categories]].mean(0) for c in names])
    )
    broad = np.array(names)[(query @ broad_prototypes.T).argmax(axis=1)]
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
            for i, r in enumerate(ranking)
        ),
    )
    np.savez_compressed(
        out / "readout.npz",
        ranked_dictionary_indices=ranking.astype(np.uint16),
        unique_ids=ids,
        broad_predictions=broad,
    )
    return ranking, broad


def paired_comparison(anchor, direct):
    rows = {}
    for k in (1, 5, 10):
        gained = int(np.sum((anchor <= k) & (direct > k)))
        lost = int(np.sum((anchor > k) & (direct <= k)))
        discordant = gained + lost
        tail = sum(comb(discordant, j) for j in range(min(gained, lost) + 1))
        rows[f"top{k}"] = {
            "anchor_only": gained,
            "direct_only": lost,
            "net_hits": gained - lost,
            "paired_mcnemar_two_sided_unadjusted_p": min(1.0, 2 * tail / 2**discordant),
        }
    return rows


def evaluate(out, base, ranking, broad, direct_order, ids, categories):
    meta = np.load(
        acquire(base / "assets/metadata.npy", ASSETS["metadata.npy"]), allow_pickle=True
    ).item()
    lookup = {s: i for i, s in enumerate(ids)}
    truth = np.array([lookup[s] for s in object_ids(meta["test_img_concepts"])])
    ranks = np.argmax(ranking == truth[:, None], axis=1) + 1
    direct_ranks = np.argmax(direct_order == truth[:, None], axis=1) + 1
    eligible = np.array([bool(categories[t]) for t in truth])
    hits = np.array([p in categories[t] for p, t in zip(broad, truth, strict=True)])
    summary = rank_summary(ranks)
    summary.update(
        vocabulary=len(ids),
        broad_correct=int(hits.sum()),
        broad_eligible=int(eligible.sum()),
        distinct_top1=int(len(np.unique(ranking[:, 0]))),
        uniform_top1_chance=1 / len(ids),
        direct_text=rank_summary(direct_ranks),
        paired_vs_direct=paired_comparison(ranks, direct_ranks),
    )
    to_csv(
        out / "scored.csv",
        ["eeg_row", "truth_id", "predicted_id", "truth_rank"],
        (
            dict(eeg_row=i, truth_id=ids[t], predicted_id=ids[ranking[i, 0]], truth_rank=ranks[i])
            for i, t in enumerate(truth)
        ),
    )
    dump(out / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("outputs/official-object-readout/sub-03"))
    p.add_argument("--out", type=Path, default=Path("outputs/official-visual-anchors"))
    p.add_argument("--cohort10", action="store_true")
    args = p.parse_args()
    base, out = args.base, args.out
    out.mkdir(parents=True, exist_ok=True)
    with np.load(base / "dictionary.npz", allow_pickle=False) as bank:
        ids, words, text = bank["unique_ids"], bank["words"], unit(bank["text_embeddings"])
        categories = [
            set(s.strip() for s in row.split(",") if s.strip()) for row in bank["manual_categories"]
        ]
        if (
            len(set(ids)) != 1854
            or text.shape != (1854, 1024)
            or str(bank["model"]) != "ViT-H-14"
            or str(bank["pretrained"]) != "laion2b_s32b_b79k"
        ):
            raise ValueError("Expected the complete released THINGS H14 dictionary")
    train, photos = train_data(base, ids, text)
    grams = kernels(text)
    (family, alpha, mode), results = select(grams, train, photos)
    provenance = {
        "protocol": "R80 THINGS-EEG2 official 200 unseen concepts; all 1,854 text candidates",
        "training_data": "16,540 H14 images across 1,654 training classes; ten unit photos per centroid",
        "selection": "Three seeded class-disjoint folds; all 16,540 held-out image pseudoqueries; max top1 then MRR",
        "fitted_candidates": "Analytic leave-one-class-out prediction for each fitted word; full prediction for unseen words",
        "selected": [family, alpha, mode],
        "fold_seed": 20261126,
        "all_training_scores": [
            dict(kernel=k[0], alpha=k[1], geometry=k[2], **v) for k, v in results.items()
        ],
        "train_visual_file": ASSETS["text_train.pt"],
        "dictionary_sha256": sha(base / "dictionary.npz"),
        "subject3_eeg_sha256": EEG_HASHES[2],
        "cohort10": "Unit-normalize each participant then equally average matching stimulus rows; separate multi-participant diagnostic",
        "published_encoder_caveat": "Original ATM authors monitored official-test retrieval over epochs; exact feature-export epoch is uncertain.",
    }
    dump(out / "selection.json", provenance)
    print("Frozen training-only choice:", provenance["selected"], flush=True)
    _, predicted = next(prototype_fits(grams[family], train, photos.mean(axis=1), [alpha]))
    scales = fit_scales(photos)
    anchors = geometry(predicted, scales, mode)
    conditions = {"sub-03": released_eeg(base, out / "assets", 3)}
    if args.cohort10:
        with ThreadPoolExecutor(max_workers=4) as pool:
            vectors = list(pool.map(lambda s: released_eeg(base, out / "assets", s), range(1, 11)))
        conditions["cohort10"] = unit(np.mean(vectors, axis=0))
    frozen = {
        k: rank_and_write(out / k, v, ids, words, categories, anchors, scales, mode)
        for k, v in conditions.items()
    }
    direct = {
        k: np.argsort(-(unit(v) @ text.T), axis=1, kind="stable") for k, v in conditions.items()
    }
    for name, (ranking, broad) in frozen.items():
        summary = evaluate(out / name, base, ranking, broad, direct[name], ids, categories)
        print(name, summary, flush=True)


if __name__ == "__main__":
    main()
