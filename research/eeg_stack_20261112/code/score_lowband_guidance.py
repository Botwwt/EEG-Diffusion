"""Score four sealed personal12 lowband-guidance JPEG arms and replay the original native metrics."""

import argparse
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fit_spatial_photo import ZIP_SHA, table
from lowband_guidance import ARMS, DESIGN, LOCK, PANEL, SUB
from PIL import Image, ImageDraw
from refine_registered_color import metrics, thumbnail
from register_fresh_photo_panel import freeze, read, sha
from render_official_words import crop512
from render_within_photo_ip import check
from score_official_objects import topk_credit
from score_reconstruction_jpegs import CLIP_SHA, registered
from score_within_photo_dino import FILES, credit, encoder


def summarize(values, objects, universe, report_credit):
    values = np.asarray(values, float)
    objects = np.asarray(objects)
    labels = sorted(set(universe))
    sums = np.stack([values[objects == c].sum(0) for c in labels])
    counts = np.array([sum(objects == c) for c in labels])
    draw = np.random.default_rng(260570).integers(len(labels), size=(10000, len(labels)))
    n = counts[draw].sum(1)
    use = n > 0
    boot = sums[draw[use]].sum(1) / n[use, None]
    avg = values.mean(0)
    effects = {}
    for k in ("native", "prior", "wrong"):
        j = ARMS.index(k)
        own = ARMS.index("own")
        effects[k] = dict(
            delta=float(avg[own] - avg[j]),
            ci95=np.quantile(boot[:, own] - boot[:, j], [0.025, 0.975]).tolist(),
        )
    result = dict(
        n=len(values),
        objects=int(np.sum(counts > 0)),
        means=dict(zip(ARMS, avg.tolist(), strict=True)),
        own_minus=effects,
    )
    if report_credit:
        result["credit"] = dict(zip(ARMS, values.sum(0).tolist(), strict=True))
    return result


def sheets(photos, reg, words, out):
    choose = sorted([int(min(rs)) for k in range(12) if len(rs := reg["rows"][reg["truth"] == k])])
    for filename, columns in (("generated.jpg", ARMS), ("private.jpg", ("original", *ARMS))):
        tile = 136
        width = len(columns) * tile + 10
        height = tile + 35
        sheet = Image.new("RGB", (2 * width, 6 * height), "white")
        draw = ImageDraw.Draw(sheet)
        for s, row in enumerate(choose):
            i = np.flatnonzero(reg["rows"] == row).item()
            x = s % 2 * width
            y = s // 2 * height
            draw.text(
                (x, y), f"Viewed: {reg['labels'][i]} | EEG: {words[i]} | row {row}", fill="black"
            )
            for j, k in enumerate(columns):
                sheet.paste(
                    photos[k][i].resize((tile, tile), Image.Resampling.LANCZOS),
                    (x + j * tile, y + 14),
                )
                draw.text((x + j * tile, y + tile + 14), k, fill="black")
        sheet.save(out / filename, "JPEG", quality=94)
    return dict(
        rows=choose,
        generated_sha256=sha(out / "generated.jpg"),
        private_sha256=sha(out / "private.jpg"),
    )


def run(a):
    spec = read(DESIGN)
    check(sha(DESIGN) == LOCK, "Design changed")
    root = a.root / SUB
    out = root / "renders"
    cfg = read(out / "settings.json")
    met = read(root / "inputs.json")
    check(
        cfg["design_sha256"] == LOCK == met["design_sha256"]
        and cfg["input_sha256"] == met["input_sha256"] == sha(root / "inputs.npz")
        and cfg["source_settings_sha256"] == spec["sources"]["native_settings"]
        and cfg["step"] == 499
        and cfg["gain"] == 0.35
        and cfg["subject"] == SUB
        and cfg["native_parity"] == 120
        and set(cfg["jpeg_sha256"]) == set(ARMS),
        "Wrong rendered experiment",
    )
    native = a.data / "trial-object-replication" / SUB
    orig_cfg = native / "render_inputs/renders" / PANEL / SUB / "settings.json"
    check(sha(orig_cfg) == cfg["source_settings_sha256"], "Changed original render")
    orig = read(orig_cfg)
    reg = registered(a.data / "object-head-confirmation/evaluation/references" / SUB, PANEL)
    check(reg["sha"] == spec["sources"]["registration"], "Wrong reference registration")
    rows, fold, labels = reg["rows"], reg["folds"], reg["labels"]
    with np.load(root / "inputs.npz", allow_pickle=False) as z:
        words = z["words"]
        donor = z["donors"]
        support = reg["calibration"][fold, [reg["vocab"].tolist().index(w) for w in words]]
        for key, value in (
            ("rows", rows),
            ("folds", fold),
            ("supports", support),
            ("donors", orig["donor_rows"]),
            ("words", orig["predicted_words"]),
        ):
            check(np.array_equal(z[key], value), f"Changed inference field {key}")
    check(
        np.array_equal(cfg["seeds"], rows + 923761)
        and cfg["seeds"] == orig["seeds"]
        and cfg["jpeg_sha256"]["native"]
        == met["expected_native_jpeg_sha256"]
        == orig["jpeg_sha256"]["own"],
        "JPEG/seed parity failed",
    )
    a.manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    tab = table(a)
    bank = a.data / "photo-structure/training_images.zip"
    check(sha(bank) == ZIP_SHA, "Wrong original photograph release")
    refcache = (
        a.data / "object-head-confirmation/evaluation/pixels" / SUB / "original_cls/personal12.npz"
    )
    with np.load(refcache, allow_pickle=False) as z:
        dig = z["jpeg_sha256"]
        ref = z["cls"].astype(float)
        check(
            str(z["registration_sha"]) == reg["sha"]
            and str(z["model_sha"]) == FILES["model.safetensors"]
            and np.array_equal(z["rows"], rows)
            and hashlib.sha256(z["cls"].tobytes()).hexdigest() == str(z["feature_sha"]),
            "DINO gallery mismatch",
        )
    photos = {"original": []}
    with ZipFile(bank) as archive:
        for row, w, h in zip(rows, labels, dig, strict=True):
            r = tab[row]
            check(r["unique_id"] == w and int(r["train_row"]) == row, "Original class/row differs")
            raw = archive.read(f"training_images/{r['official_folder']}/{r['stimulus_stem']}.jpg")
            check(hashlib.sha256(raw).hexdigest() == h, "Reference SHA differs")
            with Image.open(io.BytesIO(raw)) as im:
                photos["original"].append(crop512(im.convert("RGB")))
    encoded = encoder(a)
    features = encoded(photos["original"]).astype(float)
    check(
        np.allclose(features, ref, rtol=0, atol=2e-6),
        "Fresh DINO evaluation disagrees with prior originals",
    )
    p = native / "evaluation/clip" / (SUB + ".json")
    historical_clip = read(p)
    c = historical_clip["cases"][PANEL + "/" + SUB]
    check(
        c["settings_sha256"] == sha(orig_cfg)
        and historical_clip["clip_sha256"] == CLIP_SHA
        and np.array_equal(c["rows"], rows)
        and np.array_equal(c["truth"], labels)
        and np.array_equal(c["predicted_words"], words)
        and np.array_equal(c["folds"], fold),
        "Historical original names unaligned",
    )
    history = read(native / "evaluation/pixels" / SUB / PANEL / SUB / "per_query.json")
    rawclip = out / "clip_logits.npz"
    with np.load(rawclip, allow_pickle=False) as z:
        check(
            str(z["settings_sha256"]) == sha(out / "settings.json")
            and str(z["model_sha256"]) == CLIP_SHA,
            "Wrong logits",
        )
        vocab = z["ids"].tolist()
        check(len(set(vocab)) == len(vocab) == 1854, "Wrong vocabulary")
        subids = np.array([vocab.index(v) for v in reg["vocab"]])
        targets = subids[reg["truth"]]
        logits = {arm: z[arm] for arm in ARMS}
    pairs = np.array(
        [np.flatnonzero((labels == w) & (fold == f)) for w in reg["vocab"] for f in range(5)]
    )
    pair = pairs[np.all(words[pairs] == labels[pairs], axis=1)]
    left, right = pair.T
    n = len(rows)
    idx = np.arange(n)
    shot, pairvals = {}, {}
    for arm in ARMS:
        pics = []
        for i, h in enumerate(cfg["jpeg_sha256"][arm]):
            f = out / arm / f"{i:03}.jpg"
            check(sha(f) == h, "JPEG SHA mismatch")
            with Image.open(f) as im:
                check(im.format == "JPEG" and im.size == (512, 512), "Invalid JPEG")
                pics.append(im.convert("RGB"))
        check(len(pics) == 120, "Incomplete generated JPEG arm")
        photos[arm] = pics
        sim = encoded(pics).astype(float) @ features.T
        m1 = sim[left, left] - sim[left, right]
        m2 = sim[right, right] - sim[right, left]
        pix = np.array(
            [
                metrics(thumbnail(im), thumbnail(ref))
                for im, ref in zip(pics, photos["original"], strict=True)
            ]
        )
        gallery = np.array([np.flatnonzero(labels == w) for w in labels])
        g = sim[idx[:, None], gallery]
        pairvals[arm] = dict(
            strict_joint=np.array([credit(x) for x in m1 + m2]),
            strict_independent_two=np.array(
                [(credit(x) + credit(y)) / 2 for x, y in zip(m1, m2, strict=True)]
            ),
        )
        check(
            logits[arm].shape == (120, 1854) and np.isfinite(logits[arm]).all(),
            "Invalid word scores",
        )
        shot[arm] = dict(
            dino_intra10=topk_credit(g, 1)[idx, np.argmax(gallery == idx[:, None], axis=1)],
            target_cosine=sim.diagonal(),
            rgb_rmse32=pix[:, 1],
            luma_pearson32=pix[:, 2],
            full1854=topk_credit(logits[arm], 1)[idx, targets],
            closed12=topk_credit(logits[arm][:, subids], 1)[idx, reg["truth"]],
        )
    check(
        np.array_equal(
            shot["native"]["dino_intra10"], [r["intra10"]["own"] for r in history["images"]]
        )
        and all(
            np.array_equal(shot["native"][k], c["per_row"][k]["own"])
            for k in ("full1854", "closed12")
        ),
        "Native classifier replay failed",
    )
    raw = {}
    metrics_out = {}
    for arrays, cluster in ((shot, labels), (pairvals, labels[left])):
        for k in arrays["own"]:
            raw[k] = np.stack([arrays[arm][k] for arm in ARMS], axis=1)
            metrics_out[k] = summarize(
                raw[k], cluster, labels, k not in ("rgb_rmse32", "luma_pearson32", "target_cosine")
            )
    e = metrics_out["dino_intra10"]
    names = metrics_out["full1854"]["credit"]
    gate = (
        e["own_minus"]["native"]["delta"] > 0
        and e["own_minus"]["prior"]["delta"] > 0
        and e["own_minus"]["wrong"]["ci95"][0] > 0
        and names["own"] >= names["native"]
    )
    rawf = root / "metrics.npz"
    np.savez(
        rawf,
        rows=rows,
        folds=fold,
        truth=labels,
        words=words,
        donors=donor,
        pair_rows=rows[pair],
        arms=ARMS,
        **raw,
    )
    gall = sheets(photos, reg, words, root)
    result = dict(
        subject=SUB,
        panel=PANEL,
        design_sha256=LOCK,
        driver_sha256=sha(Path(__file__)),
        renderer_sha256=cfg["renderer_sha256"],
        native_jpeg_matches=120,
        settings_sha256=sha(out / "settings.json"),
        input_sha256=met["input_sha256"],
        row_metrics_sha256=sha(rawf),
        clip_sha256=CLIP_SHA,
        clip_logits_sha256=sha(rawclip),
        dino_sha256=FILES["model.safetensors"],
        gallery=gall,
        advance=gate,
        metrics=metrics_out,
    )
    freeze(a.release or root / "evaluation.json", result)
    print("DINO", e["credit"], "full1854", names, "advance", gate)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--root", type=Path, default=Path("outputs/lowband-guidance"))
    p.add_argument(
        "--model", type=Path, default=Path("outputs/photo-structure/dino/facebook-dinov2-small")
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--release", type=Path)
    run(p.parse_args())
