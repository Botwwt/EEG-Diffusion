"""Score frozen R4 photo reconstructions on a named 12-object, five-fold panel."""

import argparse
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from refine_registered_color import metrics, thumbnail
from render_official_words import crop512
from render_within_photo_ip import check_donor_geometry
from score_reconstruction_jpegs import registered
from score_within_photo_dino import (
    ARMS,
    FILES,
    REV,
    encoder,
    originals,
    read,
    reference,
    save,
    score_case,
    sha,
    summarize,
)

SEED = 260509
SCORING = {
    "dino_intra10": ("shots", "intra10"),
    "dino_strict_pair": ("pairs", "assignment"),
    "dino_independent_two": ("pairs", "independent_two"),
    "rgb_rmse32": ("shots", "rgb_rmse32"),
    "luma_pearson32": ("shots", "luma_pearson32"),
    "rgb_intra10": ("shots", "rgb_intra10"),
    "luma_intra10": ("shots", "luma_intra10"),
    "correct_word_rgb_rmse32": ("correct_word", "rgb_rmse32"),
    "correct_word_luma_pearson32": ("correct_word", "luma_pearson32"),
}


def validate_packet(args, subject, reg, settings):
    packet = args.conditions / subject / args.panel / "conditions_512.npz"
    with np.load(packet, allow_pickle=False) as data:
        words = data["predicted_words"]
        if words.shape != (120,) or not np.isin(words, reg["vocab"]).all():
            raise ValueError("Unknown or missing predicted words")
        columns = {w: i for i, w in enumerate(reg["vocab"])}
        allowed = reg["calibration"][reg["folds"], [columns[w] for w in words]]
        if (
            settings["subject"] != subject
            or settings["panel"] != args.panel
            or str(data["registration_sha256"]) != reg["sha"]
            or sha(packet) != settings["packet_sha256"]
            or not np.array_equal(settings["support_rows"], allowed)
            or not np.array_equal(data["support_rows"], allowed)
            or set(settings["jpeg_sha256"]) != set(ARMS)
        ):
            raise ValueError("Frozen EEG conditions or supports differ from registration")
        if args.evaluations:
            word_file = args.evaluations / subject / args.panel / "word_predictions.json"
            prediction = read(word_file)
            if (
                sha(word_file) != str(data["word_prediction_sha256"])
                or prediction["subject"] != subject
                or not np.array_equal(prediction["rows"], reg["rows"])
                or not np.array_equal(prediction["folds"], reg["folds"])
                or not np.array_equal(prediction["words"], reg["vocab"])
                or not np.array_equal(reg["vocab"][prediction["predicted"]], words)
            ):
                raise ValueError(
                    "Condition packet and independently saved object predictions differ"
                )
        own, prior, wrong = [data[k] for k in ARMS]
        row_map = {row: i for i, row in enumerate(reg["rows"])}
        donor = np.array([row_map[row] for row in data["donor_rows"]])
        if any(
            x.shape != (120, 1024) or not np.isfinite(x).all() for x in (own, prior, wrong)
        ) or not np.array_equal(prior, prior[donor]):
            raise ValueError("Wrong-EEG conditions do not follow the frozen donor map")
        angles = data["recipient_angle_radians"] if "recipient_angle_radians" in data else None
        if angles is not None:
            design_file = args.conditions.parent / "design.json"
            if (
                read(design_file).get("wrong_direction_mode") != "recipient_native_angle"
                or sha(design_file) != settings["design_sha256"]
            ):
                raise ValueError("Generated images use a different donor-angle design")
        check_donor_geometry(dict(own=own, uniform=prior, wrong=wrong), donor, angles)


def reference_pixels(table, bank, rows, checks, gallery_rows):
    out = []
    selected = {}
    for row, digest in zip(rows, checks, strict=True):
        rec = table[row]
        member = f"training_images/{rec['official_folder']}/{rec['stimulus_stem']}.jpg"
        raw = bank.read(member)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("DINO and pixel reference bytes differ")
        with Image.open(io.BytesIO(raw)) as im:
            image = crop512(im.convert("RGB"))
            out.append(thumbnail(image))
            if row in gallery_rows:
                selected[int(row)] = image
    return np.stack(out), selected


def rank_credit(values, correct, higher):
    best = max(values) if higher else min(values)
    return float(values[correct] == best) / np.count_nonzero(values == best)


def add_pixel_scores(shots, settings, folder, ref):
    labels = np.array([s["object"] for s in shots])
    groups = {k: np.flatnonzero(labels == k) for k in np.unique(labels)}
    for i, rec in enumerate(shots):
        candidates = groups[rec["object"]]
        j = np.flatnonzero(candidates == i).item()
        for field in ("rgb_rmse32", "luma_pearson32", "rgb_intra10", "luma_intra10"):
            rec[field] = {}
        for arm in ARMS:
            path = folder / "images" / arm / f"{i:03d}.jpg"
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != settings["jpeg_sha256"][arm][i]:
                raise ValueError(f"Changed generated JPEG: {path}")
            with Image.open(io.BytesIO(raw)) as im:
                rgb = thumbnail(im)
            scores = np.asarray([metrics(rgb, ref[t])[1:] for t in candidates])
            rec["rgb_rmse32"][arm] = float(scores[j, 0])
            rec["luma_pearson32"][arm] = float(scores[j, 1])
            rec["rgb_intra10"][arm] = rank_credit(scores[:, 0], j, higher=False)
            rec["luma_intra10"][arm] = rank_credit(scores[:, 1], j, higher=True)


def gallery(folder, dest, shots, original_photos=None, eeg_label="EEG"):
    labels = sorted({s["object"] for s in shots})
    chosen = [min((s for s in shots if s["object"] == w), key=lambda s: s["row"]) for w in labels]
    ordinal = {s["row"]: i for i, s in enumerate(shots)}
    width, height, tile = (344 if original_photos else 178), 201, 160
    sheet = Image.new("RGB", (width * 4, height * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for j, s in enumerate(chosen):
        left, top = j % 4 * width + 5, j // 4 * height + 4
        if original_photos:
            im = original_photos[s["row"]]
            sheet.paste(im.resize((tile, tile), Image.Resampling.LANCZOS), (left, top))
            draw.text((left, top + tile + 1), "Original", fill="black")
            draw.text((left, top + tile + 16), s["object"].replace("_", " "), fill="black")
            left += tile + 7
        with Image.open(folder / "images/own" / f"{ordinal[s['row']]:03d}.jpg") as im:
            sheet.paste(im.resize((tile, tile), Image.Resampling.LANCZOS), (left, top))
        draw.text((left, top + tile + 1), "Viewed: " + s["object"].replace("_", " "), fill="black")
        draw.text(
            (left, top + tile + 16), eeg_label + ": " + s["word"].replace("_", " "), fill="black"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, "JPEG", quality=94)
    return {"rows": [s["row"] for s in chosen], "sha256": sha(dest)}


def participant_summary(records):
    out = {}
    for metric, (group, field) in SCORING.items():
        data = records[group]
        if not data:
            out[metric] = {"n": 0, "n_objects": 0, "arms": {}, "own_minus": {}}
            continue
        stats = summarize(data, field)
        out[metric] = {
            "n": len(data),
            "n_objects": len({r["object"] for r in data}),
            "arms": {
                arm: {
                    "mean": stats[arm]["rate"],
                    "ci95": stats[arm]["rate_ci95"],
                    **(
                        {"credit": stats[arm]["correct_credit"]}
                        if "rmse" not in metric and "pearson" not in metric
                        else {}
                    ),
                }
                for arm in ARMS
            },
            "own_minus": {
                arm: {
                    "delta": stats["own_minus_" + arm]["delta_rate"],
                    "ci95": stats["own_minus_" + arm]["ci95"],
                }
                for arm in ("uniform", "wrong")
            },
        }
    return out


def pooled_summary(by_subject, labels):
    # Resample both shared stimulus classes and participants, keeping comparisons paired.
    n_subjects, n_classes = len(by_subject), len(labels)
    rng = np.random.default_rng(SEED)
    person = rng.integers(n_subjects, size=(10000, n_subjects, 1))
    category = rng.integers(n_classes, size=(10000, 1, n_classes))
    indices = {label: i for i, label in enumerate(labels)}
    out = {}
    for metric, (group, field) in SCORING.items():
        n = np.zeros((n_subjects, n_classes), int)
        sums = np.zeros((*n.shape, len(ARMS)))
        for s, records in enumerate(by_subject):
            for r in records[group]:
                c = indices[r["object"]]
                n[s, c] += 1
                sums[s, c] += [r[field][arm] for arm in ARMS]
        if not n.sum():
            out[metric] = {"n": 0, "arms": {}, "own_minus": {}}
            continue
        totals = sums.sum(axis=(0, 1))
        den = n[person, category].sum(axis=(1, 2))
        valid = den > 0
        samples = sums[person, category].sum(axis=(1, 2))[valid] / den[valid, None]
        item = {"n": int(n.sum()), "arms": {}, "own_minus": {}}
        for j, arm in enumerate(ARMS):
            item["arms"][arm] = {"mean": float(totals[j] / n.sum())}
            if "rmse" not in metric and "pearson" not in metric:
                item["arms"][arm]["credit"] = float(totals[j])
            if j:
                item["own_minus"][arm] = {
                    "delta": float((totals[0] - totals[j]) / n.sum()),
                    "ci95": np.quantile(samples[:, 0] - samples[:, j], [0.025, 0.975]).tolist(),
                }
        out[metric] = item
    return out


def run(args):
    import torch
    import transformers

    if len(set(args.subjects)) != len(args.subjects) or args.batch_size < 1:
        raise ValueError("Specify each subject once and a positive batch size")
    is_cohort = "cohort-3" in args.subjects
    if is_cohort:
        design = read(args.conditions.parent / "design.json")
        if (
            args.subjects != ["cohort-3"]
            or sorted(design.get("contributing_subjects", [])) != ["sub-01", "sub-04", "sub-05"]
            or design.get("physical_trials_per_photo") != 12
        ):
            raise ValueError(
                "Cohort scores require one registered three-person, twelve-observation output"
            )
    eeg_label = "3-person EEG" if is_cohort else "EEG"
    if args.private_gallery and not args.out.resolve().is_relative_to(Path("outputs").resolve()):
        raise ValueError("Private original-versus-render comparisons must be under outputs/")
    reg = registered(args.references, args.panel)
    if args.panel_lock:
        lock = read(args.panel_lock)
        if (
            lock["registration_sha256"] != reg["sha"]
            or lock["participants"] != args.subjects
            or lock["words"] != reg["vocab"].tolist()
        ):
            raise ValueError("Supplied panel or participant set differs from its frozen design")
    table = reference(args)
    encode = encoder(args)
    with zipfile.ZipFile(args.originals) as bank:
        ref = originals(args, encode, table, bank, args.panel)
        first_rows = (
            {min(reg["rows"][reg["labels"] == w]) for w in reg["vocab"]}
            if args.private_gallery
            else set()
        )
        pixels, selected_originals = reference_pixels(table, bank, reg["rows"], ref[2], first_rows)
    by_subject, cases = [], {}
    for subject in args.subjects:
        folder = args.renders / args.panel / subject
        settings = read(folder / "settings.json")
        validate_packet(args, subject, reg, settings)
        shots, pairs, hashes = score_case(folder, encode, ref, args.conditions)
        add_pixel_scores(shots, settings, folder, pixels)
        records = {
            "shots": shots,
            "pairs": pairs,
            "correct_word": [s for s in shots if s["object"] == s["word"]],
        }
        by_subject.append(records)
        save(
            args.out / args.panel / subject / "per_query.json",
            dict(images=shots, pairs=pairs, **hashes),
        )
        cases[subject] = {
            "provenance": hashes,
            "metrics": participant_summary(records),
            "gallery": gallery(
                folder,
                args.gallery_dir / f"photo_angle_{subject.replace('-', '')}_{args.panel}.jpg",
                shots,
                eeg_label=eeg_label,
            ),
        }
        if args.private_gallery:
            gallery(
                folder,
                args.out / args.panel / subject / "private_comparison.jpg",
                shots,
                selected_originals,
                eeg_label=eeg_label,
            )
        print(subject, len(shots), "queries,", len(pairs), "eligible pairs", flush=True)
    report = dict(
        panel=args.panel,
        subjects=list(args.subjects),
        registration_sha256=reg["sha"],
        panel_lock_sha256=sha(args.panel_lock) if args.panel_lock else None,
        unique_stimuli=len(reg["rows"]),
        generated_jpegs=120 * len(ARMS) * len(cases),
        dino=dict(
            model="facebook/dinov2-small",
            revision=REV,
            sha256=FILES,
            batch_size=args.batch_size,
            torch=torch.__version__,
            transformers=transformers.__version__,
            device=args.device,
        ),
        inputs=dict(
            original_zip_sha256=sha(args.originals),
            row_manifest_sha256=sha(args.manifest),
            scorer_sha256=sha(Path(__file__)),
            dino_scorer_sha256=sha(Path(__file__).with_name("score_within_photo_dino.py")),
        ),
        methods=dict(
            dino="Final normalized CLS cosine. Ten-photo identification includes every query. Strict two-photo assignment requires both decoded words correct in the same true class and fold. Each pair counts once. Independent two-choice credit is averaged across the two decisions per pair.",
            pixels="Archived 512-pixel center crop then BOX resize to 32x32 RGB in [0,1]. RMSE measures RGB error and Pearson compares spatial luminance. Pixel retrieval ranks all ten original photos of the same true class. Lower RMSE and higher correlation are better.",
            uncertainty="Participant intervals resample true classes with all eligible images/pairs, as in the earlier DINO scorer. Pooled differences resample participants and the shared object classes independently with replacement, 10,000 paired draws, seed 260509. The same stimuli are repeated across people.",
            gallery="Lowest original row per registered object, own-EEG generated JPEG only. Includes incorrect predictions.",
        ),
        cases=cases,
        pooled=pooled_summary(by_subject, sorted(reg["vocab"])),
    )
    if is_cohort:
        report["acquisition"] = dict(
            contributing_subjects=design["contributing_subjects"],
            physical_trials_per_photo=12,
            output_unit="One joint reconstruction of one stimulus from three people per arm",
        )
        report["methods"]["uncertainty"] = (
            "Each image has one three-person prediction. Paired intervals resample the twelve "
            "shared stimulus classes. They describe uncertainty over objects in this cohort."
        )
    save(args.release or args.out / "summary.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", default="fresh")
    p.add_argument("--subjects", nargs="+", default=["sub-04", "sub-05", "sub-01"])
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    base = Path("outputs/fresh-photo")
    for key, default in {
        "references": base / "references",
        "renders": base / "renders",
        "conditions": base / "conditions",
        "out": base / "pixel-scores",
        "gallery-dir": base / "gallery",
        "model": Path("outputs/photo-structure/dino/facebook-dinov2-small"),
        "originals": Path("outputs/photo-structure/training_images.zip"),
        "manifest": Path("outputs/official-object-readout/sub-03/atm_train_rows.csv"),
    }.items():
        p.add_argument("--" + key, type=Path, default=default)
    p.add_argument("--release", type=Path, help="Public aggregate JSON destination")
    p.add_argument(
        "--private-gallery",
        action="store_true",
        help="Write a local first-photo original and own-EEG comparison under outputs/",
    )
    p.add_argument(
        "--panel-lock", type=Path, help="Verify the registered panel and full participant set"
    )
    p.add_argument(
        "--evaluations", type=Path, help="Verify each independent EEG word-prediction file"
    )
    run(p.parse_args())
