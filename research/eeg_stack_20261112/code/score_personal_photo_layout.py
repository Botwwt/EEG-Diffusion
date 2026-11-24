"""Score two existing registered R4 renders for one participant and one panel on CPU.

Example: python score_personal_photo_layout.py --archive PKU --baseline OLD_RENDER
--personal NEW_RENDER --out panel_layout.json. The render folders contain
results.json and images/<arm>/000.jpg through 119.jpg. Only the report is written.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from decode_eeg_layout import read_rgb, save_json
from PIL import Image
from refine_registered_color import KEYS, metrics, paired_effect, thumbnail

ARMS = {
    "baseline_aligned": ("baseline", "aligned"),
    "personal_aligned": ("personal", "aligned"),
    "personal_category": ("personal", "category"),
    "personal_shuffled": ("personal", "shuffled"),
}
METRICS = KEYS[1:]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def render(path, arms):
    raw = (path / "results.json").read_bytes()
    result = json.loads(raw)
    rows = result["rows"]
    labels = result["labels"]
    words = result["predicted_words"]
    if (
        len(rows) != 120
        or any(type(r) is not int or r < 0 for r in rows)
        or len(set(rows)) != 120
        or len(labels) != 120
        or len(words) != 120
        or len(set(labels)) != 12
        or set(words) - set(labels)
        or any(labels.count(w) != 10 for w in set(labels))
    ):
        raise ValueError(f"Invalid registered 12-object, 120-photo results in {path}")
    expected = {f"{i:03d}.jpg" for i in range(len(rows))}
    for arm in arms:
        if (
            result["results"][arm]["n"] != len(rows)
            or {p.name for p in (path / "images" / arm).glob("*.jpg")} != expected
        ):
            raise ValueError(f"{path}/{arm} lacks the complete registered JPEG set")
    return (
        {
            r: (i, label, word)
            for i, (r, label, word) in enumerate(zip(rows, labels, words, strict=True))
        },
        {
            "render_folder": path.name,
            "results_sha256": digest(raw),
            "condition_sha256": result["condition_sha"],
        },
    )


def original_records(archive, rows, labels):
    manifest = archive / "data/full_T_recovery_20261108_v238/image_manifest.csv"
    keep = set(rows)
    records = {}
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset_split"] != "train":
                continue
            row = int(r["array_row"])
            if row in keep:
                if row in records:
                    raise ValueError(f"Duplicate training manifest row {row}")
                records[row] = r
    if (
        set(records) != keep
        or len({r["image_member"] for r in records.values()}) != len(rows)
        or [records[row]["concept_id"].split("_", 1)[1] for row in rows] != labels
    ):
        raise ValueError("Render row IDs and true labels disagree with the original photo manifest")
    return records, digest(manifest.read_bytes())


def jpeg(path):
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as opened:
        if opened.format != "JPEG" or opened.size != (512, 512):
            raise ValueError(f"Expected a 512-square render JPEG at {path}")
        image = opened.convert("RGB")
    return image, digest(raw)


def summary(values, truth, mask):
    labels = truth[mask]
    counts = {k: int(np.sum(labels == k)) for k in np.unique(truth)}
    out = dict(
        n=int(mask.sum()),
        n_represented_classes=len(set(labels)),
        n_by_true_class=counts,
        mean={},
        personal_aligned_minus={},
    )
    for name, arr in values.items():
        out["mean"][name] = {
            k: float(arr[mask, j].mean()) if labels.size else None
            for j, k in enumerate(KEYS)
            if k in METRICS
        }
        if name != "personal_aligned":
            delta = (values["personal_aligned"] - arr)[mask]
            if len(set(labels)) >= 2:
                stat = paired_effect(delta, labels)
                out["personal_aligned_minus"][name] = {k: stat[k] for k in METRICS}
            else:
                out["personal_aligned_minus"][name] = {
                    k: dict(delta=float(delta[:, j].mean()) if labels.size else None, ci95=None)
                    for j, k in enumerate(KEYS)
                    if k in METRICS
                }
    return out


def differences(base, personal):
    diff = np.asarray(base, dtype=np.int16) - np.asarray(personal, dtype=np.int16)
    absolute = np.abs(diff)
    return dict(
        rgb_identical=not bool(absolute.any()),
        mean_absolute_8bit=float(absolute.mean()),
        root_mean_square_8bit=float(np.sqrt(np.square(diff, dtype=np.float64).mean())),
        max_absolute_8bit=int(absolute.max()),
        fraction_channel_values_changed=float((absolute != 0).mean()),
    )


def summarize_jpegs(records):
    if not records:
        return {"n": 0, "exact_jpeg_bytes": 0, "exact_decoded_rgb": 0}
    metrics = {}
    for name in (
        "mean_absolute_8bit",
        "root_mean_square_8bit",
        "max_absolute_8bit",
        "fraction_channel_values_changed",
    ):
        vals = np.asarray([r[name] for r in records])
        metrics[name] = dict(
            mean=float(vals.mean()),
            median=float(np.median(vals)),
            p95=float(np.quantile(vals, 0.95)),
            maximum=float(vals.max()),
        )
    return dict(
        n=len(records),
        exact_jpeg_bytes=sum(r["bytes_identical"] for r in records),
        exact_decoded_rgb=sum(r["rgb_identical"] for r in records),
        image_mean_absolute_error_at_most_one_8bit_level=sum(
            r["mean_absolute_8bit"] <= 1 for r in records
        ),
        difference_by_photo=metrics,
    )


def run(args):
    if args.out.suffix.lower() != ".json" or args.out.resolve() in {
        (folder / "results.json").resolve() for folder in (args.baseline, args.personal)
    }:
        raise ValueError("Use a separate JSON output path to preserve both render records")
    base, base_meta = render(args.baseline, ["aligned"])
    personal, personal_meta = render(args.personal, ["aligned", "category", "shuffled"])
    rows = sorted(base)
    if set(base) != set(personal) or any(base[r][1] != personal[r][1] for r in rows):
        raise ValueError(
            "The two renders must describe the same 120 original photos and object labels"
        )
    truth = np.asarray([base[r][1] for r in rows])
    records, manifest_sha = original_records(args.archive, rows, truth.tolist())
    original = read_rgb(
        args.archive / "assets/things_images_v254/training_images.zip",
        records,
        rows,
        "layout scoring",
    ).transpose(0, 2, 3, 1)
    arms = {k: [] for k in ARMS}
    examples, unchanged = [], []
    for i, row in enumerate(rows):
        sources = {
            "baseline": (args.baseline, base[row][0]),
            "personal": (args.personal, personal[row][0]),
        }
        decoded, hashes = {}, {}
        for name, (which, arm) in ARMS.items():
            folder, j = sources[which]
            im, hashes[name] = jpeg(folder / "images" / arm / f"{j:03d}.jpg")
            arms[name].append(metrics(thumbnail(im), original[i]))
            if arm == "aligned":
                decoded[which] = im
        record = dict(
            row=row,
            truth=truth[i],
            baseline_word=base[row][2],
            personal_word=personal[row][2],
            original_sha256=records[row]["image_sha256"],
            jpeg_sha256=hashes,
        )
        if base[row][2] == personal[row][2]:
            record["aligned_jpeg_comparison"] = dict(
                bytes_identical=hashes["baseline_aligned"] == hashes["personal_aligned"],
                **differences(decoded["baseline"], decoded["personal"]),
            )
            unchanged.append(record["aligned_jpeg_comparison"])
        examples.append(record)
    values = {k: np.asarray(v) for k, v in arms.items()}
    right = np.array([personal[r][2] == personal[r][1] for r in rows])
    old_right = np.array([base[r][2] == base[r][1] for r in rows])
    subsets = {
        "all": np.ones(len(rows), bool),
        "personal_word_correct": right,
        "both_words_correct": right & old_right,
    }
    for i, rec in enumerate(examples):
        rec["metrics"] = {
            k: {m: float(v[i, j]) for j, m in enumerate(KEYS) if m in METRICS}
            for k, v in values.items()
        }
    report = dict(
        n=120,
        classes=sorted(set(truth)),
        baseline=base_meta,
        personal=personal_meta,
        manifest_sha256=manifest_sha,
        source_sha256=digest(Path(__file__).read_bytes()),
        prediction_correct={"baseline": int(old_right.sum()), "personal": int(right.sum())},
        subsets={k: summary(values, truth, mask) for k, mask in subsets.items()},
        unchanged_word_aligned_jpegs=summarize_jpegs(unchanged),
        unchanged_word_nonidentical_bytes=summarize_jpegs(
            [r for r in unchanged if not r["bytes_identical"]]
        ),
        metrics={
            "rgb_rmse32": "RMSE across 32 x 32 RGB values in [0,1], lower is better",
            "luminance_layout_corr32": "Spatial Pearson across 1,024 luminance samples, higher is better; historical near-zero denominator guard",
        },
        preprocessing="Original JPEG SHA verified, short edge resized bilinearly to 512, center crop, then 16x16 pixel means. Generated JPEG decoded and BOX-resized to 32x32; matches refine_registered_color.py.",
        uncertainty="Per-photo paired deltas. Percentile intervals resample represented true object classes 10,000 times, retaining all eligible photos per class and weighting by photo count. Empty classes contribute no rows. Fewer than two represented classes gives no interval. Fixed participant, folds and gallery; unadjusted descriptive subsets.",
        scope="The caller selects two renders of the same participant and panel. Correctness refers to the EEG-predicted 12-way object word. Layout and color are measured independently on the saved JPEGs.",
        per_photo=examples,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.out, report)
    print(
        "Scored 120 photos; personal word correct",
        int(right.sum()),
        "; both correct",
        int((right & old_right).sum()),
        "; unchanged-word JPEG bytes identical",
        report["unchanged_word_aligned_jpegs"]["exact_jpeg_bytes"],
        "/",
        len(unchanged),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "baseline", "personal", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    run(p.parse_args())
