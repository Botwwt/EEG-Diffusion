"""Independently score generated official-test images for a single participant's R4 and R80."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from official_object_readout import ASSETS, ORIGINALS, object_ids
from PIL import Image, ImageDraw
from refine_registered_color import metrics, thumbnail
from render_official_repeat_images import DICT_SHA
from render_official_words import crop512
from score_within_photo_dino import FILES, REV, encoder, read, sha

ARMS = ("r4", "r80", "wrong", "r4_visual", "r80_visual", "wrong_visual")
VISUAL = ARMS[3:]
PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
ROWS = np.arange(200)
GALLERY_ROWS = tuple(range(0, 200, 20))
MANIFEST_SHA = "74773a1ef288b8d1c974a256a6b1c39940949c36842fca1302dc76e641943089"
BOOTSTRAP_SEED = 260548


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decoded(raw, square):
    with Image.open(io.BytesIO(raw)) as im:
        require(im.format == "JPEG", "Expected a JPEG")
        if square:
            require(im.size == (512, 512) and im.mode == "RGB", "Generated image format changed")
            return im.copy()
        return crop512(im.convert("RGB"))


def references(root):
    archive = root / "evaluation/test_images.zip"
    manifest = root / "evaluation/originals_manifest.csv"
    meta_path = root / "assets/metadata.npy"
    require(
        sha(archive) == ORIGINALS[1]
        and sha(manifest) == MANIFEST_SHA
        and sha(meta_path) == ASSETS["metadata.npy"][1],
        "Official reference inputs changed",
    )
    # This pinned source uses NumPy's historical object-dictionary format.
    meta = np.load(meta_path, allow_pickle=True).item()
    words = object_ids(meta["test_img_concepts"])
    require(
        len(set(words)) == 200 and len(set(meta["test_img_files"])) == 200,
        "Expected unique test objects/photos",
    )
    with manifest.open(encoding="utf8", newline="") as f:
        records = list(csv.DictReader(f))
    require([int(r["eeg_row"]) for r in records] == ROWS.tolist(), "Reference rows out of order")
    result = []
    with ZipFile(archive) as z:
        for r, word, folder, file in zip(
            records, words, meta["test_img_concepts"], meta["test_img_files"], strict=True
        ):
            name = f"test_images/{folder}/{file}"
            require(
                r["unique_id"] == word and r["archive_member"] == name,
                "EEG row and official photo differ",
            )
            raw = z.read(name)
            require(
                hashlib.sha256(raw).hexdigest() == r["image_sha256"],
                "Original JPEG identity changed",
            )
            result.append(decoded(raw, False))
    return result


def gallery_metrics(generated, originals):
    require(
        generated.shape == originals.shape == (200, 384)
        and np.isfinite(generated).all()
        and np.isfinite(originals).all(),
        "Invalid DINO final CLS matrix",
    )
    cosine = generated.astype(np.float64) @ originals.astype(np.float64).T
    actual = cosine.diagonal()
    better = (cosine > actual[:, None]).sum(1)
    ties = (cosine == actual[:, None]).sum(1)
    require((ties >= 1).all(), "Missing reference diagonal")
    rank = 1 + better + (ties - 1) / 2
    return {
        "dino_top1": (better == 0) / ties,
        "dino_top5": np.clip((5 - better) / ties, 0, 1),
        "dino_mean_rank": rank,
        "dino_correct_cosine": actual,
    }


def ref_features(a, encode, originals):
    cache = a.out / "official_original_cls.npz"
    # Byte identity, processor parameters and encoder implementation bind the cache to this evaluation.
    provenance = {
        "zip": ORIGINALS[1],
        "manifest": MANIFEST_SHA,
        "metadata": ASSETS["metadata.npy"][1],
        **FILES,
        "device": a.device,
        "batch_size": str(a.batch_size),
        "encoder": sha(Path(__file__).with_name("score_within_photo_dino.py")),
        "scoring_source": sha(Path(__file__)),
    }
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            if (
                all(k in z and str(z[k]) == v for k, v in provenance.items())
                and str(z["feature_sha256"]) == hashlib.sha256(z["cls"].tobytes()).hexdigest()
            ):
                return z["cls"]
    features = encode(originals)
    require(
        gallery_metrics(features, features)["dino_top1"].sum() == 200,
        "Original DINO self matches failed",
    )
    np.savez(
        cache,
        cls=features,
        feature_sha256=hashlib.sha256(features.tobytes()).hexdigest(),
        **provenance,
    )
    return features


def verify_render(render, source, design_path, arms):
    settings_path = render / "settings.json"
    packet = render.parent / "render_packet.npz"
    s = read(settings_path)
    receipt = read(packet.with_suffix(".json"))
    design = read(design_path)
    dictionary_path = source / "dictionary.csv"
    require(sha(dictionary_path) == DICT_SHA, "Prompt dictionary differs")
    with dictionary_path.open(encoding="utf-8-sig", newline="") as f:
        dictionary = {r["unique_id"]: r["word"] for r in csv.DictReader(f)}
    native = arms == ARMS
    participant_dir = render.parent if native else render.parent.parent
    if native:
        require(
            receipt["subject"] == "sub-04"
            and receipt["physical_repeats"] == [4, 80]
            and receipt["condition_sha256"] == sha(packet)
            and receipt["inference_receipt_sha256"]
            == sha(participant_dir.parent / "inference_receipt.json"),
            "The native packet does not match this participant's frozen predictions",
        )
    else:
        require(
            arms == VISUAL
            and sha(packet) == sha(participant_dir / "global_angle_packet.npz")
            and sha(packet) == receipt["packet_sha256"]
            and sha(participant_dir / "render_packet.npz") == receipt["original_packet_sha256"]
            and receipt["design_sha256"] == sha(PUBLIC / "official_repeat_global_angle_design.json")
            and sha(packet) != receipt["original_packet_sha256"],
            "Global-angle condition identity differs",
        )
    require(
        s["rows"] == ROWS.tolist()
        and np.array_equal(s["seeds"], 923761 + ROWS)
        and s["packet_sha256"] == sha(packet)
        and s["design_sha256"] == sha(design_path)
        and design["packet_sha256"] == s["packet_sha256"]
        and tuple(design["arms"]) == arms
        and set(s["jpeg_sha256"]) == set(arms),
        "Render order, arms, random seeds or condition packet differs",
    )
    with np.load(packet, allow_pickle=False) as z:
        require(np.array_equal(z["rows"], ROWS), "Condition packet row order differs")
        if "seeds" in z:
            require(np.array_equal(z["seeds"], s["seeds"]), "Packet and settings seeds differ")
        donor = z["donor"]
        require(
            np.issubdtype(donor.dtype, np.integer)
            and np.array_equal(donor, np.roll(ROWS, 67))
            and float(z["radius"]) == 21.945598602294922,
            "Incorrect wrong-EEG donor permutation",
        )
        for budget in ("r4", "r80"):
            values = z[budget]
            if not native:
                require(
                    hashlib.sha256(values.tobytes()).hexdigest()
                    == receipt["directions"][budget]["output_unit_direction_sha256"],
                    "Calibrated angle features differ from frozen prediction",
                )
            with (participant_dir / "1012" / f"predictions_{budget.upper()}.csv").open(
                encoding="utf8", newline=""
            ) as f:
                predictions = list(csv.DictReader(f))
            require(
                values.shape == (200, 1024)
                and np.isfinite(values).all()
                and np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-6)
                and [int(r["eeg_row"]) for r in predictions] == ROWS.tolist()
                and np.array_equal(z[budget + "_words"], [r["predicted_id"] for r in predictions]),
                "Continuous condition or EEG word does not match registered source",
            )
        for arm in arms:
            ids = donor if arm.startswith("wrong") else ROWS
            budget = "r4" if arm.startswith("r4") else "r80"
            labels = z[budget + "_words"][ids]
            prompts = (
                ["A photograph."] * len(ids)
                if arm.endswith("_visual")
                else ["A photo of " + dictionary[w] + "." for w in labels]
            )
            require(
                np.array_equal(s["eeg_rows"][arm], ids) and s["prompts"][arm] == prompts,
                "Rendered donor rows or actual prompts differ from the registered arm",
            )
    candidates = [p for p in (render, render / "images") if all((p / arm).is_dir() for arm in arms)]
    require(len(candidates) == 1, "Expected one complete JPEG collection")
    files = {f"{i:03d}.jpg" for i in ROWS}
    for arm in arms:
        require(
            len(s["jpeg_sha256"][arm]) == 200
            and {p.name for p in (candidates[0] / arm).glob("*.jpg")} == files,
            f"Missing or extra JPEG in {arm}",
        )
    return s, candidates[0], donor


def generated(root, arm, digests):
    out = []
    for i, checksum in enumerate(digests):
        raw = (root / arm / f"{i:03d}.jpg").read_bytes()
        require(
            hashlib.sha256(raw).hexdigest() == checksum,
            f"Rendered JPEG checksum differs for {arm} row {i}",
        )
        out.append(decoded(raw, True))
    return out


def fixed_gallery(out, panel):
    tile, header = 176, 28
    groups = [
        ("words", ("r4", "r80", "wrong")),
        ("visual", ("r4_visual", "r80_visual", "wrong_visual")),
        ("angle_comparison", ("previous_r80_visual", "r80_visual", "wrong_visual")),
    ]
    for name, arms in groups:
        if not all(key in panel for key in arms):
            continue
        sheet = Image.new("RGB", (4 * tile, header + len(GALLERY_ROWS) * (tile + 20)), "white")
        ink = ImageDraw.Draw(sheet)
        labels = (
            ("Original", "Original-angle R80", "Calibrated-angle R80", "Calibrated wrong R80")
            if name == "angle_comparison"
            else ("Original", "R4", "R80", "Wrong R80 EEG")
        )
        for col, label in enumerate(labels):
            ink.text((col * tile + 6, 7), label, fill="black")
        for j, row in enumerate(GALLERY_ROWS):
            h = header + j * (tile + 20)
            ink.text((5, h + 2), f"Official row {row:03d}", fill="black")
            for col, key in enumerate(("original", *arms)):
                sheet.paste(
                    panel[key][row].resize((tile, tile), Image.Resampling.LANCZOS),
                    (col * tile, h + 20),
                )
        sheet.save(out / f"gallery_{name}.jpg", quality=92, subsampling=0)


def summary(raw, path, receipts, previous):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(200, size=(10000, 200))

    def interval(v):
        return np.quantile(v[idx].mean(1), [0.025, 0.975]).tolist()

    observed = {}
    for arm, metrics_by_row in raw.items():
        observed[arm] = {
            k: dict(mean=float(v.mean()), ci95=interval(v)) for k, v in metrics_by_row.items()
        }
        for k in ("dino_top1", "dino_top5"):
            observed[arm][k]["correct_credit"] = float(metrics_by_row[k].sum())
    effects = {}
    for a, b in (
        ("r80", "r4"),
        ("r80", "wrong"),
        ("r4", "wrong"),
        ("r80_visual", "r4_visual"),
        ("r80_visual", "wrong_visual"),
        ("r4_visual", "wrong_visual"),
    ):
        if a not in raw or b not in raw:
            continue
        effects[a + "_minus_" + b] = {}
        for k in raw[a]:
            diff = raw[a][k] - raw[b][k]
            effects[a + "_minus_" + b][k] = dict(
                delta=float(diff.mean()), paired_ci95=interval(diff)
            )
    cross = {}
    if previous:
        for arm in VISUAL:
            cross[arm] = {}
            for k, v in raw[arm].items():
                delta = v - previous[arm][k]
                cross[arm][k] = dict(delta=float(delta.mean()), paired_ci95=interval(delta))
        for control in ("wrong_visual", "r4_visual"):
            diffs = {}
            for k in raw["r80_visual"]:
                delta = (raw["r80_visual"][k] - raw[control][k]) - (
                    previous["r80_visual"][k] - previous[control][k]
                )
                diffs[k] = dict(delta=float(delta.mean()), paired_ci95=interval(delta))
            cross["change_in_r80_visual_minus_" + control] = diffs
    report = dict(
        subject="sub-04",
        unique_queries=200,
        unique_test_objects=200,
        images_per_arm=200,
        generated_jpegs=200 * len(raw),
        gallery_size=200,
        nominal_top1_chance=0.005,
        nominal_expected_mean_rank=100.5,
        interpretation="One test photograph per object. DINO rank among 200 different objects combines category and photo correspondence. Pixel metrics compare the actual center-cropped RGBs.",
        comparisons="R80 minus its comparator in original metric units. Lower mean rank and RGB RMSE are better; higher top-k, cosine and luminance correlation are better.",
        uncertainty="Percentile bootstrap of the 200 paired stimulus rows, 10,000 draws, one participant, seed 260548; gallery and fitted model held fixed. Intervals are marginal across the contrasts.",
        methods={
            "dino": "facebook/dinov2-small normalized final CLS cosine",
            "revision": REV,
            "pixels": "512-square center crop; BOX-downsample RGB to 32 square in [0,1], RMSE and spatial Pearson luminance with 0.299, 0.587, 0.114 weights.",
            "ties": "Average ranks and expected top-k credit under uniform ordering within exact ties.",
        },
        observed=observed,
        paired=effects,
        global_angle_minus_native_angle=cross,
        receipts=receipts,
    )
    path.write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf8"))


def previous_angle_results(render, source, donor, panels):
    participant_dir = render.parent.parent
    native = participant_dir / "render"
    old_settings, old_root, old_donor = verify_render(
        native, source, PUBLIC / "official_repeat_images_design.json", ARMS
    )
    scores = participant_dir / "independent_images"
    old_summary = read(scores / "summary.json")
    old = old_summary["receipts"]
    raw = scores / "per_row.npz"
    require(
        np.array_equal(donor, old_donor)
        and sha(raw) == old["per_row_scores_sha256"]
        and sha(native / "settings.json") == old["settings_sha256"]
        and old["dino_files_sha256"] == FILES
        and old["original_archive_sha256"] == ORIGINALS[1]
        and old["rows_manifest_sha256"] == MANIFEST_SHA,
        "Reference image/feature/score packet differs between angle settings",
    )
    with np.load(raw, allow_pickle=False) as z:
        require(
            np.array_equal(z["rows"], ROWS) and np.array_equal(z["donor_rows"], donor),
            "Unpaired native scores",
        )
        previous = {
            arm: {k: z[arm + "_" + k] for k in old_summary["observed"][arm]} for arm in VISUAL
        }
    key = "r80_visual"
    panels["previous_r80_visual"] = {}
    for i in GALLERY_ROWS:
        jpg = (old_root / key / f"{i:03d}.jpg").read_bytes()
        require(
            hashlib.sha256(jpg).hexdigest() == old_settings["jpeg_sha256"][key][i],
            "Old R80 comparison JPEG changed",
        )
        panels["previous_r80_visual"][i] = decoded(jpg, True)
    return previous, sha(scores / "summary.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("reference", "score", "angle"))
    p.add_argument("--source", type=Path, default=Path("outputs/official-object-readout/sub-03"))
    p.add_argument("--render", type=Path)
    p.add_argument(
        "--model", type=Path, default=Path("outputs/photo-structure/dino/facebook-dinov2-small")
    )
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--publish", type=Path)
    p.add_argument("--design", type=Path)
    a = p.parse_args()
    angle = a.stage == "angle"
    case = Path("outputs/official-repeat-object/sub-04")
    if angle:
        case = case / "global-angle"
    a.render = a.render or case / "render"
    a.out = a.out or case / "independent_images"
    a.design = a.design or PUBLIC / (
        "official_global_angle_images_design.json"
        if angle
        else "official_repeat_images_design.json"
    )
    arms = VISUAL if angle else ARMS
    a.out.mkdir(parents=True, exist_ok=True)
    original = references(a.source)
    encode = encoder(a)
    bank = ref_features(a, encode, original)
    require(
        gallery_metrics(bank, bank)["dino_top1"].sum() == 200,
        "Original identity sanity check failed",
    )
    if a.stage == "reference":
        print(
            "Verified 200 unique official photographs, row order, hashes and DINO self-retrieval",
            flush=True,
        )
        return
    settings, root, donor = verify_render(a.render, a.source, a.design, arms)
    panels = {"original": {i: original[i] for i in GALLERY_ROWS}}
    original_rgb = [thumbnail(im) for im in original]
    scores = {}
    for arm in arms:
        images = generated(root, arm, settings["jpeg_sha256"][arm])
        scores[arm] = gallery_metrics(encode(images), bank)
        pixel = np.array(
            [metrics(thumbnail(im), rgb)[1:] for im, rgb in zip(images, original_rgb, strict=True)]
        )
        scores[arm]["rgb_rmse32"], scores[arm]["luma_pearson32"] = pixel.T
        panels[arm] = {i: images[i] for i in GALLERY_ROWS}
        print(
            arm,
            int(scores[arm]["dino_top1"].sum()),
            "of 200; mean rank",
            scores[arm]["dino_mean_rank"].mean(),
            flush=True,
        )
    previous, previous_receipt = (
        previous_angle_results(a.render, a.source, donor, panels) if angle else ({}, None)
    )
    fixed_gallery(a.out, panels)
    raw = a.out / "per_row.npz"
    np.savez(
        raw,
        rows=ROWS,
        donor_rows=donor,
        **{f"{arm}_{k}": v for arm, d in scores.items() for k, v in d.items()},
    )
    receipts = dict(
        original_archive_sha256=ORIGINALS[1],
        rows_manifest_sha256=MANIFEST_SHA,
        official_metadata_sha256=ASSETS["metadata.npy"][1],
        dino_files_sha256=FILES,
        settings_sha256=sha(a.render / "settings.json"),
        render_design_sha256=sha(a.design),
        packet_sha256=settings["packet_sha256"],
        scorer_sha256=sha(Path(__file__)),
        per_row_scores_sha256=sha(raw),
    )
    output = a.out / "summary.json"
    if previous_receipt:
        receipts["native_score_summary_sha256"] = previous_receipt
    summary(scores, output, receipts, previous)
    if a.publish:
        a.publish.parent.mkdir(parents=True, exist_ok=True)
        a.publish.write_bytes(output.read_bytes())


if __name__ == "__main__":
    main()
