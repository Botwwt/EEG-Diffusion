"""Create the fixed common-renderer panels from registered RGB previews."""
import argparse
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile
import zipfile

from PIL import Image, ImageDraw, ImageFont, ImageOps

from collect_results import BUDGETS, METHODS, blob


SUBJECT = "sub-08"
HEADERS = ("Original", "Brain-HiVE", "ATM", "NICE", "EEGNet", "Ridge", "Ours")
TILE = 152
GAP = 12


def tile(image):
    with Image.open(BytesIO(image)) as source:
        return ImageOps.contain(source.convert("RGB"), (TILE, TILE), Image.Resampling.LANCZOS)


def display(packet, manifest, originals, output, font):
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))[:6]
    if [int(row["row0"]) for row in rows] != list(range(6)):
        raise ValueError("Expected the ordered official gallery")
    output.mkdir(parents=True, exist_ok=True)
    title = ImageFont.truetype(font, 23)
    heading = ImageFont.truetype(font, 16)
    label = ImageFont.truetype(font, 13)
    with tarfile.open(packet) as generated, zipfile.ZipFile(originals) as viewed:
        for budget in BUDGETS:
            width = GAP + len(HEADERS) * (TILE + GAP)
            height = 74 + len(rows) * (TILE + 25)
            panel = Image.new("RGB", (width, height), "#f3f5f6")
            draw = ImageDraw.Draw(panel)
            repeats = budget.removeprefix("official-").upper()
            draw.text((GAP, 13), f"Shared renderer · {repeats} · Participant 08", fill="#20232a", font=title)
            background = (255, 255, 255)
            for column, name in enumerate(HEADERS):
                x = GAP + column * (TILE + GAP)
                draw.text((x + TILE // 2, 54), name, font=heading, anchor="mm", fill="#263245")
            receipts = [
                json.loads(blob(generated, f"images/{method}/{budget}/{SUBJECT}/render.json"))
                for method in METHODS
            ]
            for row, reference in enumerate(rows):
                original = viewed.read(reference["archive_member"])
                if sha256(original).hexdigest() != reference["image_sha256"]:
                    raise ValueError("Original image differs from its registry")
                images = [original]
                for method, receipt in zip(METHODS, receipts, strict=True):
                    rgb = blob(generated, f"images/{method}/{budget}/{SUBJECT}/{row:05d}.png")
                    if sha256(rgb).hexdigest() != receipt["pixels_sha256"][row]:
                        raise ValueError(f"Previews differ for {budget}/{method}/{row}")
                    images.append(rgb)
                y = 73 + row * (TILE + 25)
                for col, image in enumerate(images):
                    x = GAP + col * (TILE + GAP)
                    scaled = tile(image)
                    draw.rectangle((x, y, x + TILE - 1, y + TILE - 1), fill=background)
                    panel.paste(scaled, (x + (TILE - scaled.width) // 2, y + (TILE - scaled.height) // 2))
                draw.text((GAP, y + TILE + 2), reference["display_name"].capitalize(), fill="#344047", font=label)
            panel.save(output / f"common-{repeats.lower()}-fixed-panel.png", optimize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=Path(__file__).with_name("common-canonical-scores.tar"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--originals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", default="DejaVuSans.ttf")
    args = parser.parse_args()
    display(args.packet, args.manifest, args.originals, args.output, args.font)
