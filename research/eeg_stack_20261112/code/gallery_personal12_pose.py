"""Publish the four generated pose edits for the preselected first photo of each object."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from score_personal12_pose import ARMS, DESIGN, LOCK, PEOPLE, check, read, unpack
from score_reconstruction_jpegs import registered
from score_within_photo_dino import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--renders", type=Path, default=Path("outputs/personal-pose/renders"))
    p.add_argument(
        "--references",
        type=Path,
        default=Path("outputs/object-head-confirmation/evaluation/references"),
    )
    p.add_argument("--out", type=Path, default=Path("research/eeg_stack_20261112/gallery"))
    a = p.parse_args()
    d = read(DESIGN)
    check(sha(DESIGN) == LOCK, "Changed gallery registration")
    seal = read(a.renders / "blind_images.json")
    a.out.mkdir(parents=True, exist_ok=True)
    public = {}
    for sub in PEOPLE:
        folder = a.renders / "personal12" / sub
        cfg = read(folder / "settings.json")
        check(sha(folder / "settings.json") == seal["cases"][sub], "Changed render settings")
        reg = registered(a.references / sub, "personal12")
        check(
            reg["sha"] == d["registration_sha256"][sub]
            and np.array_equal(reg["rows"], cfg["rows"]),
            "Wrong registered rows",
        )
        tile = 150
        width = 4 * tile + 12
        height = tile + 35
        sheet = Image.new("RGB", (2 * width, 6 * height), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (word, row) in enumerate(sorted(d["gallery_rows"][sub].items())):
            i = np.flatnonzero(reg["rows"] == row).item()
            check(row == reg["rows"][reg["labels"] == word].min(), "Different gallery selection")
            left = index % 2 * width + 3
            top = index // 2 * height
            draw.text(
                (left, top),
                f"Viewed: {word} | EEG: {cfg['predicted_words'][i]} | row {row}",
                fill="black",
            )
            for j, arm in enumerate(ARMS):
                fp = folder / "images" / arm / f"{i:03}.jpg"
                im = unpack(fp.read_bytes(), cfg["jpeg_sha256"][arm][i], True)
                sheet.paste(
                    im.resize((tile, tile), Image.Resampling.LANCZOS), (left + j * tile, top + 14)
                )
                draw.text(
                    (left + j * tile, top + tile + 15),
                    {
                        "neutral": "Unedited",
                        "prior": "8-support mean",
                        "own": "Query EEG",
                        "wrong": "Exchanged EEG",
                    }[arm],
                    fill="black",
                )
        path = a.out / f"pose_personal12_{sub}.jpg"
        sheet.save(path, "JPEG", quality=90, optimize=True)
        public[sub] = sha(path)
    print(json.dumps(public, indent=2))


if __name__ == "__main__":
    main()
