"""Lay out the first registered stimulus for each object with matched EEG reconstructions."""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from render_official_words import crop512


def run(args):
    panel = json.loads(args.registration.read_bytes())
    columns = args.case or [
        ("Personal R4", args.support, "baseline"),
        ("Support diffusion", args.support, "matched"),
        ("EEG geometry", args.warp, "query"),
    ]
    roots = [Path(root) for _, root, _ in columns]
    manifests = [json.loads((f / "settings.json").read_bytes()) for f in roots]
    rows, labels = np.asarray(panel["rows"]), np.asarray(panel["words"])[panel["truth"]]
    if any(m["rows"] != rows.tolist() for m in manifests) or any(
        m["predicted_words"] != manifests[0]["predicted_words"] for m in manifests
    ):
        raise ValueError("Gallery runs must share the registered queries and EEG words")
    which = [min(np.flatnonzero(labels == c), key=lambda i: rows[i]) for c in sorted(set(labels))]
    tile = 148
    public = Image.new("RGB", (len(columns) * (tile + 5), len(which) * (tile + 21) + 23), "white")
    draws = ImageDraw.Draw(public)
    for j, (name, _, _) in enumerate(columns):
        draws.text((j * (tile + 5) + 2, 3), name, fill="black")
    source = None
    if args.images:
        if not args.manifest or not args.private:
            raise ValueError("Original comparisons require --manifest and --private")
        with args.manifest.open(encoding="utf8", newline="") as f:
            table = list(csv.DictReader(f))
        source = Image.new("RGB", (public.width + tile + 5, public.height), "white")
        private = ImageDraw.Draw(source)
        private.text((2, 3), "Viewed original", fill="black")
    for j, i in enumerate(which):
        y = 23 + j * (tile + 21)
        draws.text(
            (2, y), f"Viewed {labels[i]}  |  EEG {manifests[0]['predicted_words'][i]}", fill="black"
        )
        for c, (_, _, arm) in enumerate(columns):
            blob = (roots[c] / "images" / arm / f"{i:03d}.jpg").read_bytes()
            if hashlib.sha256(blob).hexdigest() != manifests[c]["jpeg_sha256"][arm][i]:
                raise ValueError("Generated image differs from its score run")
            with Image.open(io.BytesIO(blob)) as im:
                public.paste(
                    im.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS),
                    (c * (tile + 5), y + 16),
                )
        if source is not None:
            rec = table[rows[i]]
            if int(rec["train_row"]) != rows[i] or rec["unique_id"] != labels[i]:
                raise ValueError("Original JPEG registration mismatch")
            with zipfile.ZipFile(args.images) as bank:
                key = f"training_images/{rec['official_folder']}/{rec['stimulus_stem']}.jpg"
                with Image.open(io.BytesIO(bank.read(key))) as stim:
                    source.paste(
                        crop512(stim.convert("RGB")).resize((tile, tile), Image.Resampling.LANCZOS),
                        (0, y + 16),
                    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    public.save(args.out, quality=93)
    if source is not None:
        source.paste(public, (tile + 5, 0))
        args.private.parent.mkdir(parents=True, exist_ok=True)
        source.save(args.private, quality=93)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("registration", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    for name in ("images", "manifest", "private", "support", "warp"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--case", action="append", nargs=3, metavar=("LABEL", "RUN_DIR", "ARM"))
    args = p.parse_args()
    if (args.case and (args.support or args.warp)) or (
        not args.case and not (args.support and args.warp)
    ):
        p.error("Supply either --support and --warp, or a --case for each comparison column")
    run(args)
