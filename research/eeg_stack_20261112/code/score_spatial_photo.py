"""Evaluate frozen spatial-transfer JPEGs on the original photographs, after image generation."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fit_spatial_photo import ARMS, LUMA, PLAN_SHA, original, save, small, table
from PIL import Image
from render_within_photo_ip import check, read, sha
from scipy.ndimage import uniform_filter
from score_within_photo_dino import FILES, ZIP_SHA, encoder
from score_within_photo_ip import estimate


def cosine(a, b):
    a = a.ravel().astype(float)
    b = b.ravel().astype(float)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def pixels(a, b):
    x = a @ LUMA
    y = b @ LUMA
    mx = uniform_filter(x, size=7)
    my = uniform_filter(y, size=7)
    vx = uniform_filter(x * x, size=7) - mx * mx
    vy = uniform_filter(y * y, size=7) - my * my
    cov = uniform_filter(x * y, size=7) - mx * my
    s = (
        (2 * mx * my + 0.01**2)
        * (2 * cov + 0.03**2)
        / ((mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2))
    )
    return dict(
        gray_ssim=float(s[3:-3, 3:-3].mean()),
        rgb_rmse=float(np.sqrt(np.mean((a - b) ** 2))),
        luma_correlation=cosine(x - x.mean(), y - y.mean()),
        gradient_cosine=cosine(np.asarray(np.gradient(x)), np.asarray(np.gradient(y))),
    )


def subset(values, labels, use):
    v = {k: np.asarray(x)[use] for k, x in values.items()}
    labels = labels[use]
    stats = {}
    for a, b in (
        ("own", "prior"),
        ("own", "wrong"),
        ("own", "neutral"),
        ("own", "old_own"),
        ("old_own", "old_prior"),
        ("prior", "neutral"),
        ("neutral", "baseline"),
    ):
        m = estimate(v[a] - v[b], labels)
        stats[a + "_minus_" + b] = dict(mean=m["mean"], ci95=m["ci95"])
    return dict(
        n=len(labels),
        objects=len(set(labels)),
        means={a: float(z.mean()) if len(z) else None for a, z in v.items()},
        comparisons=stats,
    )


def analyze_pic(im, model):
    if model is None:
        return small(im)
    return im.copy()


def run(a):
    t = table(a)
    check(sha(a.images) == ZIP_SHA, "Wrong stimulus archive")
    model = encoder(a) if a.backend == "dino" else None
    cases = {}
    collected = {}
    ref = {}
    # First locate every frozen render and verify that every arm exists before opening the original reference archive.
    for sub in a.subjects:
        for panel in a.panels:
            dest = a.root / "renders" / panel / sub
            marker = dest / "settings.json"
            cfg = read(marker)
            check(
                cfg["subject"] == sub
                and cfg["panel"] == panel
                and cfg["design_sha256"] == PLAN_SHA
                and set(cfg["jpeg_sha256"]) == set(ARMS),
                "Render receipts differ",
            )
            for arm in ARMS:
                for i, h in enumerate(cfg["jpeg_sha256"][arm]):
                    check(sha(dest / "images" / arm / f"{i:03d}.jpg") == h, "Frozen JPEG differs")
            cases[sub, panel] = (dest, cfg, sha(marker))
    with ZipFile(a.images) as ar:
        for (sub, panel), (dest, cfg, _) in cases.items():
            rows = np.asarray(cfg["rows"])
            folds = np.asarray(cfg["folds"])
            words = np.asarray(cfg["predicted_words"])
            labels = np.asarray([t[r]["unique_id"] for r in rows])
            if panel not in ref:
                pure = [original(ar, t, r) for r in rows]
                ref[panel] = (rows, model(pure) if model else [small(im) for im in pure])
            rr, target = ref[panel]
            check(np.array_equal(rr, rows), "Panel photos differ between participants")
            pos = {r: i for i, r in enumerate(rows)}
            donors = np.asarray([pos[r] for r in cfg["donor_rows"]])
            ii = np.arange(len(rows))
            groups = dict(
                all=np.ones(len(rows), bool),
                correct=words == labels,
                correct_strict_donor=(words == labels)
                & (labels == labels[donors])
                & (words == words[donors])
                & (folds == folds[donors])
                & (ii != donors),
            )
            pairs = []
            for w in sorted(set(labels)):
                for f in sorted(set(folds)):
                    inds = np.flatnonzero((labels == w) & (folds == f))
                    check(len(inds) == 2, "Expected two photos per panel object and fold")
                    if all(words[inds] == w):
                        pairs.append(inds)
            i, j = np.asarray(pairs, dtype=int).reshape(-1, 2).T
            pg = dict(
                correct_pair=np.ones(len(i), bool),
                correct_reciprocal=(donors[i] == j) & (donors[j] == i),
            )
            mv = {}
            pv = {}
            for arm in ARMS:
                out = []
                for lo in range(0, len(rows), a.batch_size):
                    block = []
                    for ix in range(lo, min(lo + a.batch_size, len(rows))):
                        with Image.open(dest / "images" / arm / f"{ix:03d}.jpg") as im:
                            check(im.size == (512, 512) and im.mode == "RGB", "Wrong JPEG shape")
                            block.append(analyze_pic(im, model))
                    out.extend(model(block) if model else block)
                if model:
                    s = np.asarray(out, dtype=np.float64) @ np.asarray(target, dtype=np.float64).T
                    gallery = np.where(labels[:, None] == labels[None, :], s, -np.inf)
                    best = gallery.max(1)
                    ties = np.isclose(gallery, best[:, None], rtol=0, atol=1e-12)
                    mv.setdefault("intra10", {})[arm] = ties[ii, ii] / ties.sum(1)
                    m = (s[i, i] + s[j, j] - s[i, j] - s[j, i]) / 2
                    pv.setdefault("assignment", {})[arm] = (m > 1e-12) + 0.5 * (abs(m) <= 1e-12)
                    pv.setdefault("cosine_margin", {})[arm] = m
                else:
                    data = [pixels(x, y) for x, y in zip(out, target, strict=True)]
                    for metric in data[0]:
                        mv.setdefault(metric, {})[arm] = np.array([rec[metric] for rec in data])

            def items(metrics, groups, labels):
                return {
                    k: {m: (v, labels, mask) for m, v in metrics.items()}
                    for k, mask in groups.items()
                }

            item = items(mv, groups, labels)
            if pv:
                item.update(items(pv, pg, labels[i]))
            collected[sub, panel] = item
            print(a.backend, sub, panel, flush=True)

    def combined(items):
        return {
            group: {
                m: subset(
                    {
                        arm: np.concatenate([item[group][m][0][arm] for item in items])
                        for arm in ARMS
                    },
                    np.concatenate([item[group][m][1] for item in items]),
                    np.concatenate([item[group][m][2] for item in items]),
                )
                for m in items[0][group]
            }
            for group in items[0]
        }

    summary = {
        s: dict(
            panels={
                p: dict(settings_sha256=cases[s, p][2], metrics=combined([collected[s, p]]))
                for p in a.panels
            },
            pooled=combined([collected[s, p] for p in a.panels]),
        )
        for s in a.subjects
    }
    save(
        a.out,
        dict(
            backend=a.backend,
            model_sha256=FILES["model.safetensors"] if model else None,
            design_sha256=PLAN_SHA,
            original_zip_sha256=ZIP_SHA,
            uncertainty="Percentile 95% intervals resample true-object clusters within participant (10000 draws, seed 260509). All deltas are first-minus-second, lower RGB RMSE is better.",
            subjects=summary,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in dict(
        root="outputs/spatial-photo",
        images="outputs/photo-structure/training_images.zip",
        manifest="outputs/official-object-readout/sub-03/atm_train_rows.csv",
        model="outputs/photo-structure/dino/facebook-dinov2-small",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    p.add_argument("--backend", choices=("pixels", "dino"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--subjects", nargs="+", default=["sub-04", "sub-05", "sub-01"])
    p.add_argument("--panels", nargs="+", default=["registered", "replicate"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())
