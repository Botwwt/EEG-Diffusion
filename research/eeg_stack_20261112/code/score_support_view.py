"""Evaluate the sealed support-view reconstructions against the viewed photographs."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fit_spatial_photo import original, table
from PIL import Image, ImageDraw
from refine_registered_color import metrics, thumbnail
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import check
from score_official_objects import topk_credit
from score_reconstruction_jpegs import CLIP_SHA, registered
from score_within_photo_dino import FILES, ZIP_SHA, encoder
from support_view import ARMS, DESIGN


def pictures(folder, expected):
    photos = []
    for i, h in enumerate(expected):
        f = folder / f"{i:03d}.jpg"
        check(sha(f) == h, "Frozen JPEG changed")
        with Image.open(f) as pic:
            check(pic.size == (512, 512), "Image size differs")
            photos.append(pic.convert("RGB"))
    return photos


def summary(values, groups):
    labels, idx = np.unique(groups, return_inverse=True)
    rows = np.stack(list(values.values()), axis=-1)
    sums = np.zeros((len(labels), len(values)))
    np.add.at(sums, idx, rows)
    draws = np.random.default_rng(260569).integers(len(labels), size=(10000, len(labels)))
    boot = sums[draws].sum(1) / np.bincount(idx)[draws].sum(1)[:, None]
    own = list(values).index("own")
    effects = {}
    for i, k in enumerate(values):
        if k != "own":
            effects[k] = dict(
                delta=float(np.mean(rows[:, own] - rows[:, i])),
                ci95=np.quantile(boot[:, own] - boot[:, i], [0.025, 0.975]).tolist(),
            )
    return dict(
        n=len(groups),
        objects=len(labels),
        means={k: float(v.mean()) for k, v in values.items()},
        own_minus=effects,
    )


def sheets(reg, words, photos, out):
    fixed = [
        np.flatnonzero(reg["truth"] == k)[np.argmin(reg["rows"][reg["truth"] == k])]
        for k in range(12)
    ]
    for public, columns in ((False, ("original", "native", *ARMS)), (True, ("native", *ARMS))):
        tile = 106
        sheet = Image.new("RGB", (2 * (len(columns) * tile + 10), 6 * (tile + 38)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, q in enumerate(fixed):
            x = (i % 2) * (len(columns) * tile + 10)
            y = (i // 2) * (tile + 38)
            draw.text(
                (x, y), f"{reg['labels'][q]} | EEG: {words[q]} | {reg['rows'][q]}", fill="black"
            )
            for j, k in enumerate(columns):
                sheet.paste(
                    photos[k][q].resize((tile, tile), Image.Resampling.LANCZOS),
                    (x + j * tile, y + 15),
                )
                draw.text((x + j * tile, y + tile + 16), k, fill="black")
        p = out / ("generated.jpg" if public else "private.jpg")
        sheet.save(p, quality=94)


def run(a):
    p = a.root / a.subject
    out = p / "renders"
    cfg = read(out / "settings.json")
    met = read(p / "inputs.json")
    check(
        sha(DESIGN) == cfg["design_sha256"] == met["design_sha256"]
        and cfg["subject"] == met["subject"] == a.subject
        and cfg["renderer_sha256"] == sha(Path(__file__).with_name("support_view.py"))
        and cfg["input_sha256"] == sha(p / "inputs.npz") == met["sha256"],
        "Unknown input or renderer output",
    )
    reg = registered(
        a.data / "object-head-confirmation/evaluation/references" / a.subject, "personal12"
    )
    native = (
        a.data / "trial-object-encoder/personal12"
        if a.subject == "sub-04"
        else a.data / "trial-object-replication" / a.subject
    )
    folder = native / "render_inputs/renders/personal12" / a.subject
    native_cfg = read(folder / "settings.json")
    rows, folds, truth, labels = (reg[k] for k in ("rows", "folds", "truth", "labels"))
    with np.load(p / "inputs.npz", allow_pickle=False) as z:
        words = z["words"]
        supports = z["supports"]
        selected = {k: z["selected_" + k] for k in ("medoid", "own", "wrong")}
        for ids in selected.values():
            check(
                np.all((ids[:, None] == supports).sum(1) == 1), "View is outside the eight supports"
            )
        check(
            reg["sha"] == met["registration_sha256"]
            and np.array_equal(rows, z["rows"])
            and np.array_equal(folds, z["folds"])
            and np.array_equal(cfg["seeds"], rows + 923761)
            and np.array_equal(cfg["seeds"], native_cfg["seeds"])
            and np.array_equal(words, native_cfg["predicted_words"])
            and np.array_equal(rows, native_cfg["rows"])
            and np.array_equal(supports, native_cfg["support_rows"])
            and met["source_packet_sha256"] == native_cfg["packet_sha256"],
            "Comparison has different rows, words, supports or seeds",
        )
    a.manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    records = table(a)
    archive = a.data / "photo-structure/training_images.zip"
    check(sha(archive) == ZIP_SHA, "Wrong reference archive")
    with ZipFile(archive) as zipfile:
        photos = {"original": [original(zipfile, records, int(r)) for r in rows]}
    photos["native"] = pictures(folder / "images/own", native_cfg["jpeg_sha256"]["own"])
    photos.update({arm: pictures(out / arm, cfg["jpeg_sha256"][arm]) for arm in ARMS})
    native_clip = read(native / "evaluation/clip" / (a.subject + ".json"))
    old = native_clip["cases"]["personal12/" + a.subject]
    check(
        native_clip["clip_sha256"] == CLIP_SHA
        and old["settings_sha256"] == sha(folder / "settings.json")
        and all(
            np.array_equal(old[k], v)
            for k, v in (
                ("rows", rows),
                ("folds", folds),
                ("truth", labels),
                ("predicted_words", words),
                ("donor_rows", native_cfg["donor_rows"]),
            )
        ),
        "Wrong native naming results",
    )
    per_arm = {
        "native": {s: np.asarray(old["per_row"][s]["own"]) for s in ("full1854", "closed12")}
    }
    with np.load(out / "clip_logits.npz", allow_pickle=False) as z:
        check(
            str(z["settings_sha256"]) == sha(out / "settings.json")
            and str(z["model_sha256"]) == CLIP_SHA,
            "Different CLIP case",
        )
        names = z["ids"].tolist()
        subset = np.array([names.index(w) for w in reg["vocab"]])
        for arm in ARMS:
            scores = z[arm]
            per_arm[arm] = dict(
                full1854=topk_credit(scores, 1)[np.arange(120), subset[truth]],
                closed12=topk_credit(scores[:, subset], 1)[np.arange(120), truth],
            )
    enc = encoder(a)
    ref = enc(photos["original"]).astype(float)
    pos = {r: i for i, r in enumerate(rows)}
    supidx = np.array([[pos[r] for r in ss] for ss in supports])
    pairs = np.array(
        [np.flatnonzero((truth == k) & (folds == f)) for k in range(12) for f in range(5)]
    )
    both = pairs[(words[pairs] == labels[pairs]).all(1)]
    left, right = both.T
    pmetrics = {}
    selected_similarity = {}
    ix = np.arange(120)
    for arm, stats in per_arm.items():
        scores = enc(photos[arm]).astype(float) @ ref.T
        stats["dino_ten"] = np.asarray(
            [
                topk_credit(scores[i : i + 1, truth == truth[i]], 1)[
                    0, np.flatnonzero(ix[truth == truth[i]] == i).item()
                ]
                for i in ix
            ]
        )
        stats["original_cosine"] = scores.diagonal()
        if arm in selected:
            actual = scores[ix, [pos[r] for r in selected[arm]]]
            selected_similarity[arm] = dict(
                mean_selected_support_cosine=float(actual.mean()),
                selected_support_beats_original=int((actual > stats["original_cosine"]).sum()),
            )
        stats["max_support_cosine"] = scores[ix[:, None], supidx].max(1)
        stats["support_beats_original"] = (
            stats["max_support_cosine"] > stats["original_cosine"]
        ).astype(float)
        pix = np.array(
            [
                metrics(thumbnail(x), thumbnail(y))
                for x, y in zip(photos[arm], photos["original"], strict=True)
            ]
        )
        stats["rgb_rmse32"], stats["luma_pearson32"] = pix[:, 1], pix[:, 2]
        d1 = scores[left, left] - scores[left, right]
        d2 = scores[right, right] - scores[right, left]

        def credit(m):
            return (np.sign(m) + 1) / 2

        pmetrics[arm] = {"joint": credit(d1 + d2), "independent": (credit(d1) + credit(d2)) / 2}
    previous = read(
        native / "evaluation/pixels" / a.subject / "personal12" / a.subject / "per_query.json"
    )
    check(
        np.array_equal(
            per_arm["native"]["dino_ten"], [v["intra10"]["own"] for v in previous["images"]]
        ),
        "Native DINO replay changed",
    )
    allmetrics = {
        k: summary({arm: d[k] for arm, d in per_arm.items()}, truth) for k in per_arm["own"]
    }
    pairmetrics = {
        k: summary({arm: d[k] for arm, d in pmetrics.items()}, truth[left])
        for k in ("joint", "independent")
    }
    counts = {
        k: {arm: float(d[k].sum()) for arm, d in per_arm.items()}
        for k in ("full1854", "closed12", "dino_ten")
    }
    same = allmetrics["dino_ten"]["own_minus"]
    result = dict(
        design_sha256=sha(DESIGN),
        score_source_sha256=sha(Path(__file__)),
        subject=a.subject,
        settings_sha256=sha(out / "settings.json"),
        input_receipt_sha256=sha(p / "inputs.json"),
        input_sha256=met["sha256"],
        clip_logits_sha256=sha(out / "clip_logits.npz"),
        native_settings_sha256=sha(folder / "settings.json"),
        dino_sha256=FILES["model.safetensors"],
        clip_sha256=CLIP_SHA,
        selection_changes={
            k: met[k] for k in ("distinct_matched_vs_medoid", "distinct_matched_vs_wrong")
        },
        counts=counts,
        metrics=allmetrics,
        strict_pairs=pairmetrics,
        attraction_to_selected_support=selected_similarity,
        advance=all(same[k]["delta"] > 0 for k in ("native", "mean", "medoid"))
        and same["wrong"]["ci95"][0] > 0
        and counts["full1854"]["own"] >= counts["full1854"]["native"],
    )
    freeze(a.release or p / "evaluation.json", result)
    sheets(reg, words, photos, p)
    print(counts, "advance", result["advance"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", default="sub-04", choices=("sub-04", "sub-05", "sub-01"))
    p.add_argument("--root", type=Path, default=Path("outputs/support-view"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument(
        "--model", type=Path, default=Path("outputs/photo-structure/dino/facebook-dinov2-small")
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--release", type=Path)
    run(p.parse_args())
