"""Lay out thirty frozen generated JPEGs from ten prespecified official test rows."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

STUDY = Path(__file__).resolve().parents[1]
ROWS = list(range(0, 200, 20))
COLS = (
    ("native", "r80_visual", "Native angle"),
    ("calibrated", "r80_visual", "Calibrated angle"),
    ("calibrated", "wrong_visual", "Wrong EEG"),
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_inputs(case, report):
    folder = case / "render"
    raw = (folder / "settings.json").read_bytes()
    if digest(raw) != report["receipts"]["settings_sha256"]:
        raise ValueError("The scored renderer receipt and gallery source differ")
    settings = json.loads(raw)
    if settings["rows"] != list(range(200)) or settings["seeds"] != list(range(923761, 923961)):
        raise ValueError("Unexpected official rows or recipient seeds")
    return folder, settings


def make(root, manifest, dest):
    public = STUDY / "results/object_fidelity"
    native = json.loads((public / "official_repeat_images.json").read_bytes())
    calibrated = json.loads((public / "official_global_angle_images.json").read_bytes())
    if any(
        d["receipts"]["rows_manifest_sha256"] != digest(manifest.read_bytes())
        for d in (native, calibrated)
    ):
        raise ValueError("Reference category manifest differs from the scored release")
    with manifest.open(encoding="utf8", newline="") as f:
        labels = list(csv.DictReader(f))
    if [int(x["eeg_row"]) for x in labels] != list(range(200)) or len(
        {r["unique_id"] for r in labels}
    ) != 200:
        raise ValueError("Incomplete official stimulus labels")
    sources = {
        "native": checked_inputs(root, native),
        "calibrated": checked_inputs(root / "global-angle", calibrated),
    }
    tile, gap, margin, gutter, heading, label_height = 152, 4, 16, 22, 50, 27
    group_width = tile * 3 + gap * 2
    width = 2 * group_width + gutter + 2 * margin
    row_height = tile + label_height + 5
    img = Image.new("RGB", (width, heading + 5 * row_height + margin), "#faf9f6")
    draw = ImageDraw.Draw(img)
    title_font = ImageFont.load_default(size=20)
    header_font = ImageFont.load_default(size=15)
    label_font = ImageFont.load_default(size=17)
    draw.text(
        (margin, 6),
        'Sub-04   /   80 presentations   /   Fixed prompt "A photograph."',
        fill="#15232a",
        font=title_font,
    )
    for index, row in enumerate(ROWS):
        group, slot = divmod(index, 5)
        x = margin + group * (group_width + gutter)
        y = heading + slot * row_height
        label = labels[row]["unique_id"].replace("_", " ")
        draw.text((x, y + 2), f"{row:03d}   {label}", fill="#15232a", font=label_font)
        for col, (variant, arm, heading_text) in enumerate(COLS):
            cx = x + col * (tile + gap)
            if slot == 0:
                draw.text((cx, 31), heading_text, fill="#46575e", font=header_font)
            folder, s = sources[variant]
            jpg = (folder / arm / f"{row:03d}.jpg").read_bytes()
            donor = (row - 67) % 200 if arm == "wrong_visual" else row
            if (
                digest(jpg) != s["jpeg_sha256"][arm][row]
                or s["eeg_rows"][arm][row] != donor
                or s["prompts"][arm][row] != "A photograph."
            ):
                raise ValueError(
                    f"Source JPEG, prompt or donor differs for {variant}, {arm}, row {row}"
                )
            with Image.open(io.BytesIO(jpg)) as source:
                if source.format != "JPEG" or source.mode != "RGB" or source.size != (512, 512):
                    raise ValueError("Expected scored 512-square RGB JPEG")
                img.paste(
                    source.resize((tile, tile), Image.Resampling.LANCZOS), (cx, y + label_height)
                )
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, quality=88, subsampling=0, optimize=True)
    print(
        f"{len(ROWS)} fixed rows, {len(ROWS) * len(COLS)} verified generated images, {width} x {img.height}; SHA256 {digest(dest.read_bytes())}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/official-repeat-object/sub-04"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/official-object-readout/sub-03/evaluation/originals_manifest.csv"),
    )
    parser.add_argument("--out", type=Path, default=STUDY / "gallery/official_repeat_sub04_r80.jpg")
    args = parser.parse_args()
    make(args.root, args.manifest, args.out)
