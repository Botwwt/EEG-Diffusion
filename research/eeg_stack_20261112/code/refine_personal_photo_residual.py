"""Refine frozen personal renders using a generic within-object EEG-to-RGB regression."""

import argparse
import hashlib
import io
from pathlib import Path

import numpy as np
from decode_eeg_layout import CHANNELS, digest, load_json, read_rgb, save_json
from PIL import Image
from refine_registered_color import KEYS, metrics, paired_effect, swap_indices, thumbnail
from score_personal_photo_layout import jpeg, original_records
from score_personal_photo_layout import render as registered_render

GRID = 4
RIDGE = 0.3
STRENGTH = 0.35
HOLDOUT_SEED = 260432
ARMS = ("frozen", "frozen_reencoded", "zero", "corrected", "swapped")
ZIP = "assets/things_images_v254/training_images.zip"


def packet(path, archive, subject, count, channels):
    with np.load(path, allow_pickle=False) as d:
        rows, names, raw, times = (d[k] for k in ("rows", "channels", "mean_eeg", "times"))
        ids, stems = d["unique_ids"].tolist(), d["stimulus_stems"].tolist()
        if (
            str(d["subject"]) != subject
            or int(d["repeats"]) != 4
            or rows.dtype.kind not in "iu"
            or rows.shape != (count,)
            or not np.all(np.diff(rows) > 0)
            or not np.array_equal(
                rows.reshape(-1, 10) % 10, np.tile(np.arange(10), (count // 10, 1))
            )
            or not np.all(rows.reshape(-1, 10) // 10 == rows[::10, None] // 10)
            or len(set(names)) != channels
            or raw.shape != (count, channels, 250)
            or raw.dtype != np.float32
            or not np.allclose(times, np.arange(250) / 250, atol=1e-7, rtol=0)
        ):
            raise ValueError(f"Unexpected subject, EEG axes, repeats or row IDs in {path}")
        indices = [names.tolist().index(name) for name in CHANNELS]
        z = raw[:, indices, 15:115].reshape(count, 17, 20, 5).mean(-1, dtype=np.float64)
    records, manifest_hash = original_records(archive, rows.tolist(), ids)
    if [Path(records[r]["stimulus_id"]).stem for r in rows] != stems or not np.isfinite(z).all():
        raise ValueError("EEG/photo identities or finite EEG design check failed")
    return rows, z.reshape(count, -1), records, manifest_hash


def coarse(images):
    # NCHW 32x32 verified crops -> NHWC 4x4.
    n = len(images)
    return (
        images.transpose(0, 2, 3, 1).reshape(n, GRID, 32 // GRID, GRID, 32 // GRID, 3).mean((2, 4))
    )


def regress(x, y):
    # Axes are concept, photo and feature. Both modalities lose their concept means.
    x = (x - x.mean(1, keepdims=True)).reshape(-1, x.shape[-1])
    y = (y - y.mean(1, keepdims=True)).reshape(len(x), -1)
    sd = x.std(0).clip(1e-12)
    z = x / sd
    gram = z.T @ z
    gram.flat[:: len(sd) + 1] += RIDGE * len(z)
    return sd, np.linalg.solve(gram, z.T @ y)


def residual_model(x, y, concepts):
    held = np.sort(np.random.default_rng(HOLDOUT_SEED).choice(len(concepts), 16, replace=False))
    train = np.setdiff1d(np.arange(len(concepts)), held)
    sd, weight = regress(x[train], y[train])
    preds, observed = [], []
    for fold in range(5):
        test = np.arange(10) // 2 == fold
        dx = x[held][:, test] - x[held][:, ~test].mean(1, keepdims=True)
        dy = y[held][:, test] - y[held][:, ~test].mean(1, keepdims=True)
        preds.append(((dx / sd) @ weight).reshape(-1))
        observed.append(dy.ravel())
    p, a = np.concatenate(preds), np.concatenate(observed)
    product, energy = float(p @ a), float(p @ p)
    gain = max(0.0, product / energy) if energy > 0 else 0.0
    sd, weight = regress(x, y)
    return (
        sd,
        weight,
        gain,
        dict(
            held_concepts=concepts[held].tolist(),
            selection_seed=HOLDOUT_SEED,
            held_photos=160,
            calibration_dot_product=product,
            calibration_prediction_energy=energy,
            calibration_target_energy=float(a @ a),
            holdout_zero_rmse=float(np.sqrt(np.mean(a**2))),
            holdout_scaled_rmse=float(np.sqrt(np.mean((gain * p - a) ** 2))),
        ),
    )


def fit(args):
    path = args.inputs / "generic" / f"{args.subject}.npz"
    rows, design, records, manifest_hash = packet(path, args.archive, args.subject, 1280, 17)
    with np.load(args.inputs / f"{args.subject}.npz", allow_pickle=False) as d:
        excluded = np.unique(d["rows"] // 10)
    eligible = np.setdiff1d(np.arange(1654), excluded)
    expected = np.sort(np.random.default_rng(260431).choice(eligible, 128, replace=False))
    concepts = rows[::10] // 10
    if len(excluded) != 24 or not np.array_equal(concepts, expected):
        raise ValueError("Generic training rows differ from the registered disjoint 128 concepts")
    y = coarse(read_rgb(args.archive / ZIP, records, rows, "generic training")).astype(np.float64)
    sd, weight, gain, validation = residual_model(
        design.reshape(128, 10, -1), y.reshape(128, 10, -1), concepts
    )
    args.model.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.model.with_suffix(".partial.npz")
    np.savez(
        tmp,
        sd=sd,
        weight=weight,
        gain=gain,
        subject=args.subject,
        rows=rows,
        grid=GRID,
        ridge=RIDGE,
        manifest_sha256=manifest_hash,
    )
    tmp.replace(args.model)
    save_json(
        args.model.with_suffix(".json"),
        dict(
            subject=args.subject,
            model_sha256=digest(args.model),
            packet_sha256=digest(path),
            manifest_sha256=manifest_hash,
            grid=GRID,
            ridge=RIDGE,
            gain=gain,
            **validation,
            eeg="17 posterior channels, R4, 60 <= t < 460 ms, nonoverlapping 20 ms bins",
            gain_method="Nonnegative least squares on 16 held generic concepts, 8 supports vs 2 queries; final ridge refit on all 128",
            training_photo_hashes=[records[r]["image_sha256"] for r in rows],
        ),
    )
    print(f"{args.subject}: fitted 1280 photos, fixed gain {gain:.8g}", flush=True)


def layouts(args, records, rows, x, by_row, words, refs, sd, weight, gain):
    refrows, reffolds, reftruth = (np.asarray(refs[k]) for k in ("rows", "folds", "truth"))
    vocab = np.asarray(refs["words"])
    support = np.asarray(refs["calibration_rows_by_fold_and_word"])
    if (
        len(vocab) != 12
        or len(set(vocab)) != 12
        or support.shape != (5, 12, 8)
        or len(refrows) != 120
        or set(rows) != set(refrows)
        or set(reffolds) != set(range(5))
        or len(reffolds) != 120
        or reftruth.shape != (120,)
        or set(reftruth) != set(range(12))
    ):
        raise ValueError("Unexpected registered fold, word or row axes")
    refpos = {r: i for i, r in enumerate(refrows)}
    folds = np.array([reffolds[refpos[r]] for r in rows])
    base = np.empty((len(rows), GRID, GRID, 3))
    residual = np.empty_like(base)
    for f, groups in enumerate(support):
        query = rows[folds == f]
        if len(query) != 24 or set(groups.ravel()) != set(rows) - set(query):
            raise ValueError("Held-out query overlaps calibration or a support row is missing")
        for j, group in enumerate(groups):
            if any(
                reftruth[refpos[r]] != j or records[r]["concept_id"].split("_", 1)[1] != vocab[j]
                for r in group
            ):
                raise ValueError("Calibration support class differs from the manifest")
        active = [j for j, w in enumerate(vocab) if ((folds == f) & (words == w)).any()]
        support_rows = groups[active].ravel()
        photos = coarse(read_rgb(args.archive / ZIP, records, support_rows, f"support fold {f}"))
        means = photos.reshape(len(active), 8, GRID, GRID, 3).mean(1)
        for j, ymean in zip(active, means, strict=True):
            idx = np.flatnonzero((folds == f) & (words == vocab[j]))
            center = x[[by_row[r] for r in groups[j]]].mean(0)
            dx = x[[by_row[r] for r in rows[idx]]] - center
            base[idx] = ymean
            residual[idx] = (gain * ((dx / sd) @ weight)).reshape(-1, GRID, GRID, 3)
    return folds, base, residual


def adjusted(im, grid):
    full = np.asarray(im, dtype=np.float32) / 255.0
    observed = full.reshape(GRID, 512 // GRID, GRID, 512 // GRID, 3).mean((1, 3))
    delta = (STRENGTH * (grid - observed)).astype(np.float32)
    smooth = np.stack(
        [
            np.asarray(Image.fromarray(d).resize(im.size, Image.Resampling.BILINEAR))
            for d in delta.transpose(2, 0, 1)
        ],
        axis=-1,
    )
    return Image.fromarray(np.rint((full + smooth).clip(0, 1) * 255).astype(np.uint8))


def encoded(im):
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=95)
    return out.getvalue()


def scores(values, labels, mask):
    classes = np.unique(labels[mask])
    effects = {}
    for test, ref in (
        ("corrected", "zero"),
        ("swapped", "zero"),
        ("corrected", "swapped"),
        ("zero", "frozen_reencoded"),
        ("corrected", "frozen_reencoded"),
    ):
        delta = (values[test] - values[ref])[mask]
        effects[f"{test}_minus_{ref}"] = (
            paired_effect(delta, labels[mask])
            if len(classes) >= 2
            else {
                k: {"delta": float(delta[:, j].mean()) if len(delta) else None, "ci95": None}
                for j, k in enumerate(KEYS)
            }
        )
    return dict(
        n=int(mask.sum()),
        n_true_classes=len(classes),
        means={
            a: {k: float(v[mask, j].mean()) if mask.any() else None for j, k in enumerate(KEYS)}
            for a, v in values.items()
        },
        paired=effects,
    )


def render(args):
    if args.out.resolve() == args.render.resolve():
        raise ValueError("Separate the refinement output from the frozen render path")
    data, receipt = registered_render(args.render, ["aligned"])
    rows = np.array(list(data))
    labels = np.array([data[r][1] for r in rows])
    words = np.array([data[r][2] for r in rows])
    path = args.inputs / f"{args.subject}.npz"
    eegrows, x, records, manifest_hash = packet(path, args.archive, args.subject, 240, 63)
    by_row = {r: i for i, r in enumerate(eegrows)}
    if any(
        r not in by_row or records[r]["concept_id"].split("_", 1)[1] != label
        for r, label in zip(rows, labels, strict=True)
    ):
        raise ValueError("Rendered image or EEG rows differ from the photo manifest")
    spec = args.inputs / args.panel / "ablation.json"
    refs = load_json(spec)
    with np.load(args.model, allow_pickle=False) as d:
        if (
            str(d["subject"]) != args.subject
            or str(d["manifest_sha256"]) != manifest_hash
            or int(d["grid"]) != GRID
            or float(d["ridge"]) != RIDGE
            or np.intersect1d(eegrows // 10, d["rows"] // 10).size
        ):
            raise ValueError("Generic model, target subject or source manifest mismatch")
        gain = float(d["gain"])
        folds, base, residual = layouts(
            args, records, rows, x, by_row, words, refs, d["sd"], d["weight"], gain
        )
    donor = swap_indices(folds, words)
    for arm in ARMS:
        (args.out / "images" / arm).mkdir(parents=True, exist_ok=True)
    hashes = {a: [] for a in ARMS}
    thumbs = {a: [] for a in ARMS}
    for i, r in enumerate(rows):
        src = args.render / "images" / "aligned" / f"{data[r][0]:03d}.jpg"
        im, src_hash = jpeg(src)
        pics = dict(
            frozen=src.read_bytes(),
            frozen_reencoded=encoded(im),
            zero=encoded(adjusted(im, base[i])),
            corrected=encoded(adjusted(im, base[i] + residual[i])),
            swapped=encoded(adjusted(im, base[i] + residual[donor[i]])),
        )
        for arm, blob in pics.items():
            h = hashlib.sha256(blob).hexdigest()
            if arm == "frozen" and h != src_hash:
                raise ValueError("Frozen JPEG changed during read")
            (args.out / "images" / arm / f"{i:03d}.jpg").write_bytes(blob)
            hashes[arm].append(h)
            with Image.open(io.BytesIO(blob)) as view:
                thumbs[arm].append(thumbnail(view.convert("RGB")))
        if (i + 1) % 24 == 0:
            print(args.panel, args.subject, "wrote", i + 1, "of", len(rows), flush=True)
    settings = dict(
        panel=args.panel,
        subject=args.subject,
        rows=rows.tolist(),
        true_labels=labels.tolist(),
        predicted_words=words.tolist(),
        folds=folds.tolist(),
        donor_rows=rows[donor].tolist(),
        source_condition_sha256=receipt["condition_sha256"],
        source_results_sha256=receipt["results_sha256"],
        source_jpeg_sha256=hashes["frozen"],
        jpeg_sha256=hashes,
        model_sha256=digest(args.model),
        target_packet_sha256=digest(path),
        manifest_sha256=manifest_hash,
        ablation_sha256=digest(spec),
        script_sha256=digest(Path(__file__)),
        jpeg_quality=95,
        grid=GRID,
        gain=gain,
        strength=STRENGTH,
        donor_rule="Within each predicted word and fold, rotate render indices by one; singleton maps to self",
    )
    save_json(args.out / "settings.json", settings)
    # Evaluation starts after all rendered JPEGs and their hashes have been frozen.
    truth = read_rgb(args.archive / ZIP, records, rows, "score queries").transpose(0, 2, 3, 1)
    values = {
        a: np.array([metrics(pic, t) for pic, t in zip(thumbs[a], truth, strict=True)])
        for a in ARMS
    }
    swappable = donor != np.arange(len(rows))
    same = swappable & (labels[donor] == labels)
    right = labels == words
    masks = dict(
        all=np.ones(len(rows), bool),
        predicted_object_correct=right,
        swappable=swappable,
        same_truth_swappable=same,
        correct_and_same_truth_swappable=right & same,
    )
    report = dict(
        subject=args.subject,
        panel=args.panel,
        settings_sha256=digest(args.out / "settings.json"),
        gain=gain,
        strength=STRENGTH,
        metric_keys=list(KEYS),
        original_sha256=[records[r]["image_sha256"] for r in rows],
        subsets={k: scores(values, labels, mask) for k, mask in masks.items()},
        per_photo={a: v.tolist() for a, v in values.items()},
        confidence_intervals="10,000 draws resampling represented true objects with replacement; photo-weighted mean delta; fewer than two objects has no CI",
    )
    save_json(args.out / "report.json", report)
    print("RGB report", args.out / "report.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True, dest="command")
    for name in ("fit", "render"):
        sub = commands.add_parser(name)
        for key in ("archive", "inputs", "model"):
            sub.add_argument("--" + key, type=Path, required=True)
        sub.add_argument("--subject", choices=["sub-04", "sub-05"], required=True)
        if name == "render":
            sub.add_argument("--panel", choices=["registered", "replicate"], required=True)
            sub.add_argument(
                "--render",
                type=Path,
                required=True,
                help="Frozen personal render folder containing results.json",
            )
            sub.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (fit if args.command == "fit" else render)(args)


if __name__ == "__main__":
    main()
