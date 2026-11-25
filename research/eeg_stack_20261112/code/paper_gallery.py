"""Make a private, fixed-sample contact sheet from a verified benchmark export."""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def font(size, bold=False):
    for path in (Path("C:/Windows/Fonts") / ("msyhbd.ttc" if bold else "msyh.ttc"),
                 Path("/usr/share/fonts/opentype/noto") / "NotoSansCJK-Regular.ttc"):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def build(verified, originals, output, subject, track, panel):
    coverage = json.loads((verified / "coverage.json").read_bytes())
    table = "train" if panel == "fresh" else "test"
    with (originals / f"{table}_photos.csv").open(encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))
    plan = json.loads((verified / "data/prep/paper_split.json").read_bytes())
    rows = plan["fresh_rows" if panel == "fresh" else "official_rows"]
    budget = "fresh-r4" if panel == "fresh" else "official-r4"
    arms = [("Brain-HiVE", "personal", "native-paper"), ("M1a + D2", track, "d2")]
    paths = [verified / f"images/{group}/{arm}/{budget}/{subject}" for _, group, arm in arms]
    pred = []
    for p in paths:
        with np.load(p / "observations.npz", allow_pickle=False) as s:
            assert s["row"].tolist() == rows
            pred.append(s["generated_top_word"].tolist())
    selection = coverage["preview_rows"][panel]
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(originals / ("training_images.zip" if panel == "fresh" else "test_images.zip")) as z:
        for page in range(2):
            picture = Image.new("RGB", (904, 1358), "#faf9f6")
            d = ImageDraw.Draw(picture)
            d.text((20, 14), "四次脑电能还原哪张照片？", font=font(27, True), fill="#172521")
            d.text((20, 59), f"{subject}   ·   {'官方陌生物体' if panel == 'official' else '留出物体的不同照片'}   ·   固定样本 {page*8+1}–{page*8+8} / 16",
                   font=font(17), fill="#42554e")
            d.text((20, 91), "生成图下方是 CLIP 在完整词典中的第一预测。原图在生成后才用于评分。", font=font(15), fill="#46514c")
            for j, index in enumerate(selection[page * 8:(page + 1) * 8]):
                x = 130 + (j % 4) * 190
                top = 140 + (j // 4) * 603
                item = records[rows[index]]
                assert int(item["row0"]) == rows[index]
                d.text((x, top), item["display_name"], font=font(16, True), fill="#223a34")
                with Image.open(io.BytesIO(z.read(item["archive_member"]))) as source:
                    imgs = [ImageOps.fit(source.convert("RGB"), (166, 166), method=Image.Resampling.BILINEAR)]
                for p in paths:
                    with Image.open(p / "preview" / f"{index:05d}.jpg") as saved:
                        imgs.append(saved.convert("RGB").resize((166, 166), Image.Resampling.LANCZOS))
                for k, (name, im) in enumerate(zip(("原图", arms[0][0], arms[1][0]), imgs)):
                    y = top + 32 + 187*k
                    if j % 4 == 0:
                        d.text((17, y + 68), name, font=font(15, True), fill="#33463d")
                    picture.paste(im, (x, y))
                    if k:
                        d.text((x, y + 165), str(pred[k-1][index]).replace("_", " "), font=font(13), fill="#3f4842")
            picture.save(output / f"{track}-{subject}-{budget}-{page+1}.jpg", quality=91)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("verified", type=Path)
    p.add_argument("originals", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--subject", default="sub-01")
    p.add_argument("--track", choices=("personal", "shared"), default="personal")
    p.add_argument("--panel", choices=("fresh", "official"), default="official")
    a = p.parse_args()
    build(a.verified, a.originals, a.output, a.subject, a.track, a.panel)
