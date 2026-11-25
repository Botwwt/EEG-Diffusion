"""Score saved patch-transport JPEGs against the actual ten photographs of each object."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fit_patch_transport import BLIND, SUBJECT, layout, registered_case
from fit_shared_photo_eeg import read, require, write
from fit_spatial_photo import original
from PIL import Image, ImageDraw
from readout_external_images import MODEL_SHA, h14_batch, h14_pixels, load_h14
from refine_registered_color import metrics, thumbnail
from score_within_photo_dino import FILES, encoder, reference, sha
from test_cohort_session_matched import NAMES

ARMS = ("base", *BLIND, "oracle")


def retrieval(sim, reg):
    rank, top, pairs, cats = [], [], [], []
    for i, label in enumerate(reg["labels"]):
        local = sim[i, reg["labels"] == label]
        val = sim[i, i]
        rank.append(1 + (local > val).sum() + 0.5 * ((local == val).sum() - 1))
        top.append(float(val == local.max()) / (local == local.max()).sum())
    for c in range(12):
        for fold in range(5):
            ix = np.flatnonzero((reg["truth"] == c) & (reg["folds"] == fold))
            require(len(ix) == 2, "Expected two fixed same-object views")
            i, j = ix
            margin = float(sim[i, i] + sim[j, j] - sim[i, j] - sim[j, i])
            pairs.append(float(margin > 0) + 0.5 * (margin == 0))
            cats.append(c)
    return {"rank": rank, "intra10": top, "assignment": pairs}, np.array(cats)


def describe(arms, categories, count):
    values = np.asarray([arms[arm] for arm in ARMS], dtype=float)
    require(values.shape == (len(ARMS), len(categories)), "Unpaired image metrics")
    by = np.stack([values[:, categories == c].mean(1) for c in range(12)], axis=1)
    draw = np.random.default_rng(260536).integers(12, size=(10000, 12))
    means = by[:, draw].mean(2)

    def difference(arm, ref):
        i, j = ARMS.index(arm), ARMS.index(ref)
        return dict(
            delta=float((values[i] - values[j]).mean()),
            ci95=np.quantile(means[i] - means[j], [0.025, 0.975]).tolist(),
        )

    return dict(
        n=len(categories),
        values=dict(zip(ARMS, (values.sum(1) if count else values.mean(1)).tolist(), strict=True)),
        own_minus={arm: difference("own", arm) for arm in ("neutral", "prior", "wrong")},
        oracle_minus_neutral=difference("oracle", "neutral"),
    )


def score(a):
    layout(a)
    table = reference(a)
    dino = encoder(a)
    hmodel, mean, std = load_h14(a.h14, a.device)

    def embed(images):
        feats = [
            h14_batch(
                hmodel, mean, std, [h14_pixels(im) for im in images[off : off + a.batch_size]]
            )
            for off in range(0, len(images), a.batch_size)
        ]
        return dict(h14=np.concatenate(feats), dino=dino(images))

    out = dict(
        design_sha256=sha(a.plan),
        dino_sha256=FILES["model.safetensors"],
        h14_sha256=MODEL_SHA,
        fit=read(a.fit_dir / "fit.json"),
        blind_lock_sha256=sha(a.out / "blind_lock.json"),
        panels={},
    )
    for panel in NAMES:
        reg, settings, base = registered_case(a, panel)
        dest = a.out / panel
        blind = read(dest / "blind.json")
        oracle = read(dest / "oracle.json")
        require(
            read(a.out / "blind_lock.json")[panel] == sha(dest / "blind.json")
            and all(
                m["rows"] == reg["rows"].tolist()
                and m["plan"] == sha(a.plan)
                and m["fit"] == out["fit"]["fit"]
                and m["settings"] == sha(base / "settings.json")
                for m in (blind, oracle)
            ),
            "Different photo identities or prior render",
        )
        hashes = {"base": settings["jpeg_sha256"]["own"], **blind["jpeg"], **oracle["jpeg"]}
        with ZipFile(a.originals) as archive:
            originals = [original(archive, table, int(row)) for row in reg["rows"]]
        bank = embed(originals)
        thumbnails = [thumbnail(im) for im in originals]
        chosen = [
            np.flatnonzero(reg["rows"] == min(reg["rows"][reg["labels"] == word])).item()
            for word in sorted(reg["vocab"])
        ]
        show = {"original": [originals[i].resize((120, 120)) for i in chosen]}
        observations = {}
        cats = {}
        for arm in ARMS:
            folder = base / "images/own" if arm == "base" else dest / arm
            imgs = []
            for i, digest in enumerate(hashes[arm]):
                f = folder / f"{i:03d}.jpg"
                require(sha(f) == digest, "Generated image bytes differ")
                with Image.open(f) as im:
                    imgs.append(im.convert("RGB"))
            require(len(imgs) == 120, "Incomplete JPEG set")
            if arm in ("base", "own", "oracle"):
                show[arm] = [imgs[i].resize((120, 120)) for i in chosen]
            for backend, feats in embed(imgs).items():
                results, paircats = retrieval(feats @ bank[backend].T, reg)
                for key, vec in results.items():
                    name = backend + "_" + key
                    cats[name] = paircats if key == "assignment" else reg["truth"]
                    observations.setdefault(name, {})[arm] = vec
            pixels = np.array(
                [metrics(thumbnail(im), ref) for im, ref in zip(imgs, thumbnails, strict=True)]
            )
            for col, name in ((1, "rgb_rmse32"), (2, "luma_pearson32")):
                observations.setdefault(name, {})[arm] = pixels[:, col].tolist()
                cats[name] = reg["truth"]
        # The unmodified source must reproduce the already published DINO decisions for every photo.
        previous = read(
            Path(__file__).resolve().parents[1]
            / "results/object_fidelity/cohort_session_images.json"
        )
        receipt = previous["panels"][panel]
        saved = a.root / "pixel-scores" / panel / SUBJECT / panel / SUBJECT / "per_query.json"
        require(
            sha(saved) == receipt["inputs"][SUBJECT]["pixel_rows_sha256"],
            "Earlier DINO rows changed",
        )
        old = read(saved)["images"]
        require(
            [r["row"] for r in old] == reg["rows"].tolist()
            and [r["intra10"]["own"] for r in old] == observations["dino_intra10"]["base"],
            "Earlier DINO baseline differs",
        )
        write(dest / "scores.json", observations)
        summary = {
            k: describe(v, cats[k], k.endswith(("intra10", "assignment")))
            for k, v in observations.items()
        }
        width, height = 500, 152
        gallery = Image.new("RGB", (width * 2, height * 6), "white")
        draw = ImageDraw.Draw(gallery)
        for j, i in enumerate(chosen):
            x, y = j % 2 * width, j // 2 * height
            draw.text(
                (x + 2, y + 1),
                f"{reg['labels'][i]} | EEG: {settings['predicted_words'][i]}",
                fill="black",
            )
            for k, arm in enumerate(("original", "base", "own", "oracle")):
                gallery.paste(show[arm][j], (x + 2 + k * 124, y + 15))
                draw.text((x + 2 + k * 124, y + 136), arm, fill="black")
        gallery.save(dest / "comparison.jpg", quality=93)
        out["panels"][panel] = dict(
            metrics=summary,
            blind_sha256=sha(dest / "blind.json"),
            oracle_sha256=sha(dest / "oracle.json"),
            row_scores_sha256=sha(dest / "scores.json"),
            gallery_sha256=sha(dest / "comparison.jpg"),
            fixed_rows=reg["rows"][chosen].tolist(),
        )
        print(panel, summary["h14_intra10"], flush=True)
    out["promotion"] = all(
        out["panels"][p]["metrics"]["h14_intra10"]["own_minus"][c]["ci95"][0] > 0
        for p in NAMES
        for c in ("neutral", "wrong")
    )
    write(a.out / "summary.json", out)
    if a.release:
        write(a.release, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, path in dict(
        root="outputs/session-matched-cohort",
        out="outputs/patch-transport",
        model="outputs/photo-structure/dino/facebook-dinov2-small",
        h14="outputs/external-object-anchors/assets",
        manifest="outputs/official-object-readout/sub-03/atm_train_rows.csv",
        originals="outputs/photo-structure/training_images.zip",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(path))
    p.add_argument("--release", type=Path)
    p.add_argument("--limited", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    score(p.parse_args())
