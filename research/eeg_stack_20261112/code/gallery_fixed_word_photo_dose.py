"""Show the lowest official row of every registered class at each EEG acquisition budget."""

import argparse
import csv
from pathlib import Path
from zipfile import ZipFile

from fit_fixed_word_photo_dose import GROUPS, PANELS, PHOTO, REF, ROOT, SOURCE
from PIL import Image, ImageDraw, ImageFont
from register_fresh_photo_panel import freeze, read
from render_within_photo_ip import check, sha
from score_fixed_word_photo_dose import jpeg, registry
from score_within_photo_dino import reference

GALLERY = Path(__file__).resolve().parents[1] / "gallery"


def sheet(groups, captions, path):
    tile, gap, pad = 128, 3, 12
    card_width = tile * len(captions) + gap * (len(captions) - 1)
    card_height = tile + 43
    canvas = Image.new("RGB", (2 * card_width + 3 * pad, 6 * card_height + 7 * pad), "#faf9f6")
    draw = ImageDraw.Draw(canvas)
    bold = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    for n, (truth, word, pictures) in enumerate(groups):
        x = pad + (n % 2) * (card_width + pad)
        y = pad + (n // 2) * (card_height + pad)
        draw.text(
            (x, y),
            f"Viewed: {truth}   |   EEG: {word}",
            fill="#17232f" if truth == word else "#902f27",
            font=bold,
        )
        for i, (name, im) in enumerate(zip(captions, pictures, strict=True)):
            xx = x + i * (tile + gap)
            canvas.paste(im.resize((tile, tile), Image.Resampling.LANCZOS), (xx, y + 23))
            draw.text((xx, y + tile + 26), name, fill="#40464e", font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, "JPEG", quality=90, subsampling=0, optimize=True)


def run(a):
    sources = {g: a.root / "render_inputs" for g in GROUPS}
    names = ["1 observer", "3 observers", "9 observers"]
    suffix = ""
    if a.extra_root:
        sources["cohort-9-extra"] = a.extra_root
        names.append("9, guided")
        suffix = "_guided"
    table = reference(a)
    with (SOURCE / "dictionary.csv").open(newline="", encoding="utf8") as fh:
        labels = {r["unique_id"]: r["word"] for r in csv.DictReader(fh)}
    with ZipFile(a.originals) as z:
        for panel in PANELS:
            cases = {
                k: registry(k, panel, root, a.root / "render_inputs") for k, root in sources.items()
            }
            reg = read(REF / (panel + "_ablation.json"))
            base = cases["single-1"][1]
            selected = [
                min((i for i, c in enumerate(reg["truth"]) if c == k), key=lambda i: reg["rows"][i])
                for k in range(len(reg["words"]))
            ]
            private, public, entries = [], [], []
            for i in selected:
                row = reg["rows"][i]
                truth = reg["words"][reg["truth"][i]]
                word = base["predicted_words"][i]
                info = table[row]
                check(info["unique_id"] == truth, "Incorrect reference image row")
                member = f"training_images/{info['official_folder']}/{info['stimulus_stem']}.jpg"
                pics = []
                hashes = {}
                for k, (folder, settings) in cases.items():
                    file = folder / "images/own" / f"{i:03}.jpg"
                    check(
                        settings["rows"][i] == row
                        and settings["predicted_words"][i] == word
                        and sha(file) == settings["jpeg_sha256"]["own"][i],
                        "Frozen gallery image changed",
                    )
                    hashes[k] = sha(file)
                    pics.append(jpeg(file.read_bytes(), True))
                public.append((labels[truth], labels[word], pics))
                private.append((labels[truth], labels[word], [jpeg(z.read(member), False), *pics]))
                entries.append(
                    dict(
                        ordinal=i,
                        row=row,
                        truth=truth,
                        predicted_word=word,
                        generated_sha256=hashes,
                    )
                )
            pub = a.gallery / ("fixed_word_dose_" + panel + suffix + ".jpg")
            priv = a.root / "gallery" / (panel + suffix + "_originals.jpg")
            sheet(public, names, pub)
            sheet(private, ["Original", *names], priv)
            freeze(
                a.root / "gallery" / (panel + suffix + ".json"),
                dict(selected=entries, public_sha256=sha(pub), private_sha256=sha(priv)),
            )
            print(panel, "frozen rows", [r["row"] for r in entries], "public", sha(pub))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in dict(
        root=ROOT,
        gallery=GALLERY,
        manifest=SOURCE / "atm_train_rows.csv",
        originals=PHOTO / "training_images.zip",
    ).items():
        p.add_argument("--" + k, type=Path, default=v)
    p.add_argument("--extra-root", type=Path)
    run(p.parse_args())
