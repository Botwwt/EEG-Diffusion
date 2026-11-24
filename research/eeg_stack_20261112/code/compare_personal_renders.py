"""Show the first archived photo of every object, with matched baseline and personal renders."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from decode_eeg_layout import registration
from PIL import Image, ImageDraw, ImageOps


def run(args):
    def load(path):
        return json.loads((path / "results.json").read_text())

    old, new = load(args.baseline), load(args.personal)
    if any(old[k] != new[k] for k in ("rows", "labels", "clip_sha")):
        raise ValueError("Input galleries refer to different photos or evaluators")
    rows, labels = np.asarray(new["rows"]), np.asarray(new["labels"])
    if len(set(rows)) != 120 or len(set(labels)) != 12:
        raise ValueError("Expected the original balanced twelve-object panel")
    groups = [np.where(labels == w)[0] for w in sorted(set(labels))]
    if any(len(g) != 10 for g in groups):
        raise ValueError("Expected ten photographs per object")
    indices = [g[rows[g].argmin()] for g in groups]
    args.out.mkdir(parents=True, exist_ok=True)
    tile, gap = 144, 8
    public = Image.new("RGB", (4 * (tile + gap), 3 * (tile + 44)), "white")
    public_text = ImageDraw.Draw(public)
    for j, i in enumerate(indices):
        x, y = (j % 4) * (tile + gap), (j // 4) * (tile + 44)
        with Image.open(args.personal / "images/aligned" / f"{i:03d}.jpg") as im:
            public.paste(ImageOps.fit(im, (tile, tile)), (x, y))
        public_text.text((x + 2, y + tile + 1), "Viewed: " + labels[i], fill="black")
        public_text.text((x + 2, y + tile + 17), "EEG: " + new["predicted_words"][i], fill="black")
    public.save(args.out / "generated.jpg", quality=94)
    if args.archive is None:
        return
    records, _, _, _ = registration(args.archive)
    col = 3 * (tile + gap) + gap
    private = Image.new("RGB", (2 * col, 6 * (tile + 42)), "white")
    text = ImageDraw.Draw(private)
    with zipfile.ZipFile(args.archive / "assets/things_images_v254/training_images.zip") as bank:
        for j, i in enumerate(indices):
            x, y = (j % 2) * col, (j // 2) * (tile + 42)
            row = records[rows[i]]
            orig = bank.read(row["image_member"])
            if hashlib.sha256(orig).hexdigest() != row["image_sha256"]:
                raise ValueError("Stimulus file differs from manifest")
            for k, (role, word, source) in enumerate(
                (
                    ("Viewed", labels[i], io.BytesIO(orig)),
                    (
                        "Frozen",
                        old["predicted_words"][i],
                        args.baseline / "images/aligned" / f"{i:03d}.jpg",
                    ),
                    (
                        "Personal",
                        new["predicted_words"][i],
                        args.personal / "images/aligned" / f"{i:03d}.jpg",
                    ),
                )
            ):
                left = x + k * (tile + gap)
                text.text((left, y + 2), role + ": " + word, fill="black")
                with Image.open(source) as im:
                    private.paste(ImageOps.fit(im.convert("RGB"), (tile, tile)), (left, y + 20))
    private.save(args.out / "private_comparison.jpg", quality=94)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "personal", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument(
        "--archive", type=Path, help="Include a private comparison with original stimuli"
    )
    run(p.parse_args())
