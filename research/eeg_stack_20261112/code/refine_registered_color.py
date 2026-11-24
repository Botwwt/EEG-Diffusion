"""Use a training-only RGB prediction to refine coarse color after registered-object generation."""

import argparse
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from decode_eeg_layout import eeg_design, fit_predict, read_rgb, registration, save_json
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

ARMS = (
    "category",
    "aligned",
    "aligned_jpeg",
    "color",
    "train_layout",
    "train_rgb",
    "word_swap",
    "shuffled",
)
KEYS = ("mean_rgb_mae", "rgb_rmse32", "luminance_layout_corr32")


def fit(args, records, labels, split, channels, rows):
    import torch

    t = np.asarray(split["T"])
    t = t[~np.isin(labels[t], np.unique(labels[rows]))]
    eeg = torch.load(
        args.archive / "data/recovery_20261108_v236/sub-03_train_mean4.pt",
        weights_only=True,
        mmap=True,
        map_location="cpu",
    ).numpy()
    imgs = args.archive / "assets/things_images_v254/training_images.zip"
    y = read_rgb(imgs, records, t, "generic training only")
    mean, pred = fit_predict(eeg_design(eeg, t, channels), y, eeg_design(eeg, rows, channels), True)
    np.savez(
        args.out / "color.npz",
        rows=rows,
        rgb=np.clip(pred, 0, 1),
        training_rows=t,
        train_rgb_mean=mean,
    )


def thumbnail(im):
    return np.asarray(im.resize((32, 32), Image.Resampling.BOX), dtype=np.float32) / 255.0


def color_adjust(im, predicted):
    observed = thumbnail(im)
    diff = gaussian_filter(predicted - observed, sigma=(2, 2, 0))
    y = (diff @ np.array([0.299, 0.587, 0.114], np.float32))[:, :, None]
    delta = (0.50 * (diff - y) + 0.30 * y).clip(-0.16, 0.16)
    smooth = np.stack(
        [
            np.asarray(
                Image.fromarray(d.astype(np.float32)).resize(im.size, Image.Resampling.BILINEAR)
            )
            for d in delta.transpose(2, 0, 1)
        ],
        axis=-1,
    )
    new = np.asarray(im, dtype=np.float32) / 255 + smooth
    return Image.fromarray(np.uint8(np.rint(np.clip(new, 0, 1) * 255)))


def metrics(arr, target):
    x, y = arr.reshape(-1, 3), target.reshape(-1, 3)
    weights = np.array([0.299, 0.587, 0.114])
    xl, yl = x @ weights, y @ weights
    xl, yl = xl - xl.mean(), yl - yl.mean()
    return [
        float(np.abs(x.mean(0) - y.mean(0)).mean()),
        float(np.sqrt(((x - y) ** 2).mean())),
        float(xl @ yl / max(np.linalg.norm(xl) * np.linalg.norm(yl), 1e-12)),
    ]


def swap_indices(folds, words):
    donor = np.arange(len(words))
    for fold in np.unique(folds):
        for word in np.unique(words):
            idx = np.flatnonzero((folds == fold) & (words == word))
            donor[idx] = np.roll(idx, 1)
    return donor


def paired_effect(delta, labels):
    classes = np.unique(labels)
    if not len(classes):
        return {}
    sums = np.stack([delta[labels == c].sum(0) for c in classes])
    counts = np.array([sum(labels == c) for c in classes])
    boot = np.random.default_rng(923763).integers(len(classes), size=(10000, len(classes)))
    replicate = sums[boot].sum(1) / counts[boot].sum(1)[:, None]
    return {
        k: dict(
            delta=float(delta[:, j].mean()),
            ci95=np.quantile(replicate[:, j], [0.025, 0.975]).tolist(),
        )
        for j, k in enumerate(KEYS)
    }


def assess(args, records, labels, rows, folds, words):
    archive = args.archive / "assets/things_images_v254/training_images.zip"
    with np.load(args.out / "color.npz", allow_pickle=False) as b:
        train = b["training_rows"]
        if not np.array_equal(rows, b["rows"]) or np.isin(labels[rows], labels[train]).any():
            raise ValueError("RGB decoder has an overlapping class or misaligned prediction")
        rgb = b["rgb"].transpose(0, 2, 3, 1)
        if "train_rgb_mean" in b:
            layout = b["train_rgb_mean"]
        else:
            # Older completed fits already store all allowed training indices.
            # Recover the same fitted intercept without re-fitting EEG.
            layout = (
                read_rgb(archive, records, train, "recover frozen training RGB mean")
                .mean(0, dtype=np.float64)
                .astype(np.float32)
            )
    layout = layout.transpose(1, 2, 0)
    flat = layout.mean((0, 1), keepdims=True)
    donor = swap_indices(folds, words)
    for arm in ("aligned_jpeg", "color", "train_layout", "train_rgb", "word_swap"):
        (args.images / arm).mkdir(exist_ok=True)
    for i in range(len(rows)):
        with Image.open(args.images / "aligned" / f"{i:03d}.jpg") as im:
            base = im.convert("RGB")
        base.save(args.images / "aligned_jpeg" / f"{i:03d}.jpg", quality=95)
        for arm, target in (
            ("color", rgb[i]),
            ("train_layout", layout),
            ("train_rgb", flat),
            ("word_swap", rgb[donor[i]]),
        ):
            color_adjust(base, target).save(args.images / arm / f"{i:03d}.jpg", quality=95)
    gt = read_rgb(
        args.archive / "assets/things_images_v254/training_images.zip",
        records,
        rows,
        "score frozen colors",
    ).transpose(0, 2, 3, 1)
    values = {}
    for arm in ARMS:
        z = []
        for i in range(len(rows)):
            with Image.open(args.images / arm / f"{i:03d}.jpg") as im:
                z.append(metrics(thumbnail(im.convert("RGB")), gt[i]))
        values[arm] = np.array(z)
    paired = {}
    selected = donor != np.arange(len(rows))
    same_truth = labels[rows] == labels[rows[donor]]
    for a, b in (
        ("aligned", "category"),
        ("aligned", "shuffled"),
        ("color", "aligned"),
        ("color", "category"),
        ("color", "aligned_jpeg"),
        ("color", "train_rgb"),
        ("color", "train_layout"),
        ("color", "word_swap"),
    ):
        eligible = selected if b == "word_swap" else np.ones(len(rows), bool)
        delta = values[a] - values[b]
        paired[a + " minus " + b] = paired_effect(delta[eligible], labels[rows[eligible]])
        if b == "word_swap":
            ix = selected & same_truth
            paired["color minus word_swap among same true category donors"] = paired_effect(
                delta[ix], labels[rows[ix]]
            )
    save_json(
        args.out / "color_results.json",
        dict(
            n=len(rows),
            rows=rows.tolist(),
            metrics={
                arm: dict(
                    zip(
                        KEYS,
                        v[selected if arm == "word_swap" else slice(None)].mean(0).tolist(),
                        strict=True,
                    )
                )
                for arm, v in values.items()
            },
            metric_rows={
                arm: int(selected.sum()) if arm == "word_swap" else len(rows) for arm in ARMS
            },
            swap_eligible=selected.tolist(),
            donor_original_rows=rows[donor].tolist(),
            n_swappable=int(selected.sum()),
            n_swappable_same_true_class=int((selected & same_truth).sum()),
            per_row={arm: v.tolist() for arm, v in values.items()},
            metric_order=list(KEYS),
            paired=paired,
        ),
    )


def private_gallery(args, records, rows, nouns, words):
    from render_official_words import crop512
    from score_official_objects import load_jpg

    choices = [
        min(np.flatnonzero(nouns == name), key=lambda i: rows[i]) for name in np.unique(nouns)
    ]
    archive = args.archive / "assets/things_images_v254/training_images.zip"
    with ZipFile(archive) as z:
        for page, subset in enumerate(np.array_split(choices, 2), 1):
            sheet = Image.new("RGB", (720, 6 * 248), "white")
            draw = ImageDraw.Draw(sheet)
            for j, i in enumerate(subset):
                record = records[rows[i]]
                data = z.read(record["image_member"])
                if hashlib.sha256(data).hexdigest() != record["image_sha256"]:
                    raise ValueError("Unexpected original photo in private gallery")
                with Image.open(io.BytesIO(data)) as im:
                    original = crop512(im.convert("RGB"))
                y = j * 248
                draw.text((4, y + 2), f"{nouns[i]}   EEG: {words[i]}   row {rows[i]}", fill="black")
                for k, (name, image) in enumerate(
                    (
                        ("Original", original),
                        ("Aligned", load_jpg(args.images / "aligned", i)),
                        ("Color", load_jpg(args.images / "color", i)),
                    )
                ):
                    draw.text((k * 240 + 4, y + 18), name, fill="black")
                    sheet.paste(
                        image.resize((208, 208), Image.Resampling.LANCZOS), (k * 240 + 4, y + 34)
                    )
            sheet.save(args.out / f"private_color_gallery_{page}.jpg", quality=94)


def score_objects(args, records, rows, words, folds):
    import torch
    from eval_instance_starts import load_visual_scorer, sha
    from score_official_objects import (
        image_features,
        load_dictionary,
        load_jpg,
        load_tokenizer,
        text_bank,
        topk_credit,
    )

    ids, unique, inverse, _, _ = load_dictionary(args.dictionary)
    nouns = np.array([records[r]["concept_id"].split("_", 1)[1] for r in rows])
    truth = np.array([np.flatnonzero(ids == name).item() for name in nouns])
    subset = np.unique(truth)
    if len(subset) != 12:
        raise ValueError("Expected 12 registered concepts")
    row = np.arange(len(rows))
    mask = swap_indices(folds, words) != row
    scores = {}
    with torch.inference_mode():
        model, digest = load_visual_scorer(args.archive)
        tokenizer = load_tokenizer(args.archive / "assets/native_generator_v243/turbo/tokenizer")
        bank = text_bank(model, tokenizer, unique, 32)
        for arm in ARMS:
            folder = args.images / arm
            embedded = image_features(
                model,
                lambda i, directory=folder: load_jpg(directory, i),
                len(rows),
                args.batch_size,
            )
            logits = (embedded @ bank.T)[:, inverse]
            scores[arm] = (
                topk_credit(logits, 1)[row, truth],
                topk_credit(logits[:, subset], 1)[row, np.searchsorted(subset, truth)],
            )
    results = {}
    for arm, vectors in scores.items():
        use = mask if arm == "word_swap" else np.ones(len(row), bool)
        entry = {"n": int(use.sum())}
        for key, correct, base in zip(
            ("full1854", "closed12"), vectors, scores["aligned"], strict=True
        ):
            v, b = correct[use], base[use]
            retained = np.minimum(v, b).sum()
            entry[key] = {
                "top1_credit": float(v.sum()),
                "aligned_top1_on_same_rows": float(b.sum()),
                "retained_aligned_credit": float(retained),
                "retention_fraction": float(retained / b.sum()) if b.sum() else None,
                "lost_credit": float(b.sum() - retained),
                "gained_credit": float(v.sum() - retained),
                "per_row_credit": correct.tolist(),
            }
        results[arm] = entry
    save_json(
        args.out / "color_objects.json",
        dict(
            rows=rows.tolist(),
            swap_eligible=mask.tolist(),
            results=results,
            clip_sha256=digest,
            dictionary_sha256=sha(args.dictionary),
            conditions_sha256=sha(args.conditions),
            retention_reference="Archived aligned JPEGs rescored with identical preprocessing",
        ),
    )
    private_gallery(args, records, rows, nouns, words)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "conditions", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--images", type=Path)
    p.add_argument("--dictionary", type=Path)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--stage", choices=("fit", "assess", "score-objects"), required=True)
    a = p.parse_args()
    records, labels, split, channels = registration(a.archive)
    with np.load(a.conditions, allow_pickle=False) as data:
        rows, folds, words = [data[k] for k in ("rows", "folds", "words")]
    if len(np.unique(rows)) != 120 or any(x.shape != (120,) for x in (rows, folds, words)):
        p.error("Expected 120 aligned original row IDs, folds and EEG-predicted words")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "fit":
        fit(a, records, labels, split, channels, rows)
    elif not a.images:
        p.error("--images is required to assess generated photographs")
    elif a.stage == "assess":
        assess(a, records, labels, rows, folds, words)
    elif a.dictionary:
        score_objects(a, records, rows, words, folds)
    else:
        p.error("--dictionary is required for score-objects")
