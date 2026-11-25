"""Build a private, self-contained gallery of every official R4 test reconstruction for one person."""
import argparse
import base64
import csv
import hashlib
import html
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def jpg(data, expected):
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Changed image in gallery source")
    with Image.open(io.BytesIO(data)) as im:
        small = ImageOps.fit(im.convert("RGB"), (224, 224), method=Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=86, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def generate(root, archive, subject, track):
    baseline = root / f"images/personal/native-paper/official-r4/{subject}"
    method = root / f"images/{track}/d2/official-r4/{subject}"
    sources = []
    for folder in (baseline, method):
        receipt = json.loads((folder / "scores.json").read_bytes())
        render = (folder / "render.json").read_bytes()
        obs = (folder / "observations.npz").read_bytes()
        if (hashlib.sha256(obs).hexdigest() != receipt["observations_sha256"]
                or hashlib.sha256(render).hexdigest() != receipt["render_sha256"]):
            raise ValueError("Invalid observation provenance")
        meta = json.loads(render)
        with np.load(io.BytesIO(obs), allow_pickle=False) as scores:
            labels = scores["concept"].tolist()
            words = scores["generated_top_word"].tolist()
            if scores["row"].tolist() != meta["row"] or meta["row"] != list(range(200)):
                raise ValueError("Wrong official stimulus order")
        sources.append((folder, meta, labels, words))
    if any(sources[0][1][k] != sources[1][1][k] for k in ("seeds", "repeats", "prompt", "draws", "unet_calls_per_draw")):
        raise ValueError("Mismatched sampling settings")
    off = archive / "runs/visual_fidelity_20261125_000818/official"
    with (off / "originals_manifest.csv").open() as f:
        records = list(csv.DictReader(f))
    rows = []
    with zipfile.ZipFile(off / "test_images.zip") as z:
        for i, record in enumerate(records):
            name = record["unique_id"]
            if int(record["eeg_row"]) != i or any(s[2][i] != name for s in sources):
                raise ValueError("EEG or photograph identity mismatch")
            urls = [jpg(z.read(record["archive_member"]), record["image_sha256"])]
            urls.extend(jpg((p / f"{i:05d}.png").read_bytes(), m["pixels_sha256"][i]) for p, m, _, _ in sources)
            captions = [name, sources[0][3][i], sources[1][3][i]]
            cards = [f'<figure><img loading="lazy" alt="{html.escape(caption)}" src="{url}"><figcaption>{html.escape(caption)}</figcaption></figure>'
                     for caption, url in zip(captions, urls)]
            rows.append(f'<section data-name="{html.escape(name)}"><h2>#{i+1:03d}　{html.escape(name)}</h2><div>{"".join(cards)}</div></section>')
    if len(rows) != 200:
        raise ValueError("Incomplete gallery")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>官方 200 张 EEG 重建对照</title><style>
*{box-sizing:border-box}body{max-width:890px;margin:auto;padding:0 14px 35px;background:#f8f8f5;color:#162b24;font:16px/1.55 'Source Sans 3','Microsoft YaHei',sans-serif}
h1{margin:24px 0 4px;font-size:clamp(23px,4vw,34px)}p{margin:4px 0 14px}nav{background:#f8f8f5;position:sticky;top:0;padding:8px 0;border-bottom:2px solid #39725d}
input{padding:7px 10px;width:min(240px,60vw);border:1px solid #71887c;border-radius:4px;font:inherit}nav div,section div{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}
nav div{margin-top:7px;font-weight:bold}h2{font-size:17px;margin:14px 0 4px}figure{margin:0}img{width:100%;max-width:270px;display:block}figcaption{font-size:14px}
section{padding-bottom:12px;border-bottom:1px solid #ccd5cf}label,output{font-size:14px}</style>
<h1>官方 200 张 EEG 重建对照</h1><p>SUBJECT · TRAINING · 每张四次脑电，空提示，相同种子，单次成图。按数据集原顺序展示全部照片。重建图下方是 CLIP 在 1,854 个物体名中的第一预测。</p>
<nav><label>查找原物体 <input id="q" type="search" autocomplete="off"></label> <output id="n">200 / 200</output><div><span>原始照片</span><span>Brain‑HiVE（论文投影）</span><span>M1a + D2</span></div></nav>"""
    page = page.replace("SUBJECT", subject).replace("TRAINING", "两组均单人训练" if track == "personal" else "Brain‑HiVE 单人训练，M1a + D2 十人共享训练")
    page += "".join(rows) + """<script>
const filter=document.querySelector('#q'),counter=document.querySelector('#n'),rows=[...document.querySelectorAll('section')];
filter.addEventListener('input',()=>{const t=filter.value.trim().toLowerCase();let n=0;for(const r of rows){r.hidden=!r.dataset.name.toLowerCase().includes(t);if(!r.hidden)n++;}counter.textContent=n+' / '+rows.length;});
</script></html>"""
    out = root / f"gallery-{track}-{subject}-official-r4.html"
    tmp = out.with_suffix(".partial")
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(out)
    print(out.name, len(rows), hashlib.sha256(out.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("subject", choices=[f"sub-{i:02d}" for i in range(1, 11)])
    p.add_argument("track", choices=("personal", "shared"))
    a = p.parse_args()
    generate(Path(os.environ["RAD_ROOT"]), Path(os.environ["ARCHIVE"]), a.subject, a.track)
