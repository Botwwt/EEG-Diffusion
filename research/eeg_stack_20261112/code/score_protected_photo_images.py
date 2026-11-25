"""Pair actual mix13 JPEGs against the ten photographs of each registered object."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fit_spatial_photo import original, table
from refine_registered_color import metrics, thumbnail
from register_fresh_photo_panel import freeze, read, sha
from render_protected_photo import ARMS, LOCK, PANEL, SUB, verified
from render_within_photo_ip import check
from run_prospective_readout import frozen_npz
from score_canonical_neighbor_images import gallery
from score_official_objects import topk_credit
from score_reconstruction_jpegs import CLIP_SHA, registered
from score_support_view import pictures
from score_within_photo_dino import FILES, ZIP_SHA, encoder

BASE = Path(__file__).resolve().parents[1]
PUB = BASE / "results/object_fidelity"
METRICS = ("dino_ten", "full1854", "closed12", "dino_cosine", "rgb_rmse32", "luma32")


def summarize(data, true_ids, mask, seed):
    ids = true_ids[mask]
    objects = np.unique(ids)
    check(len(ids) > 0, "Empty evaluation subset")
    totals = np.stack([data[mask][ids == o].sum(0) for o in objects])
    n = np.array([(ids == o).sum() for o in objects])
    draw = np.random.default_rng(seed).integers(len(objects), size=(10000, len(objects)))
    results = {}
    for m, key in enumerate(METRICS):
        v = totals[:, :, m]
        means = v.sum(0) / len(ids)
        b = v[draw].sum(1) / n[draw].sum(1)[:, None]
        one = dict(mean=dict(zip(ARMS, means.tolist(), strict=True)), own_minus={})
        if m < 3:
            one["top1_credit"] = dict(zip(ARMS, v.sum(0).tolist(), strict=True))
        for other in ("native", "wrong"):
            i = ARMS.index("own")
            j = ARMS.index(other)
            delta = data[mask, i, m] - data[mask, j, m]
            pair = dict(
                difference=float(means[i] - means[j]),
                ci95=np.quantile(b[:, i] - b[:, j], [0.025, 0.975]).tolist(),
            )
            if m < 3:
                pair.update(
                    credit_gained=float(delta.clip(0).sum()),
                    credit_lost=float((-delta).clip(0).sum()),
                )
            one["own_minus"][other] = pair
        results[key] = one
    return dict(n=len(ids), objects=len(objects), **results)


def run(a):
    root = a.out / "inputs"
    folder, cfg = verified(root)
    plan = read(root / "protocol.json")
    spec = plan["source_sha256"]
    historic = PUB / "fresh_mix_images.json"
    words_design = PUB / "fresh_mix_design.json"
    check(
        sha(historic) == spec["historical_images"] and sha(words_design) == spec["word_design"],
        "Historical results or registration differ",
    )
    old = read(historic)["panels"][PANEL]
    w = read(words_design)
    r = registered(a.data / "fresh-mix/references", PANEL)
    rows, labels, truth, folds = (r[key] for key in ("rows", "labels", "truth", "folds"))
    words = np.array(cfg["predicted_words"])
    donors = np.array(cfg["donor_rows"])
    selector = {v: i for i, v in enumerate(rows)}
    d = np.array([selector[x] for x in donors])
    word_ix = np.array([r["vocab"].tolist().index(v) for v in words])
    check(
        r["sha"] == old["registration_sha256"] == w["panels"][PANEL]["registration_sha256"]
        and np.array_equal(rows, cfg["rows"])
        and np.array_equal(folds, cfg["folds"])
        and np.array_equal(cfg["support_rows"], r["calibration"][folds, word_ix]),
        "Historical legal supports/rows differ",
    )
    a.manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    archive = a.data / "photo-structure/training_images.zip"
    check(
        sha(a.manifest) == w["sources"]["original_photo_rows"] and sha(archive) == ZIP_SHA,
        "Incorrect original photographs",
    )
    tbl = table(a)
    check(
        all(tbl[row]["unique_id"] == label for row, label in zip(rows, labels, strict=True)),
        "Wrong original photograph labels",
    )
    with ZipFile(archive) as f:
        images = {"original": [original(f, tbl, int(row)) for row in rows]}
    encode = encoder(a)
    target = encode(images["original"]).astype(float)
    target_rgb = [thumbnail(im) for im in images["original"]]
    score = np.empty((120, 3, len(METRICS)))
    naming = folder / "naming.npz"
    with np.load(naming, allow_pickle=False) as z:
        names = z["names"].tolist()
        check(
            len(set(names)) == len(names) == 1854
            and str(z["model_sha256"]) == CLIP_SHA
            and str(z["settings_sha256"]) == sha(folder / "settings.json"),
            "Wrong CLIP model, rows or dictionary",
        )
        ix = np.array([names.index(k) for k in r["vocab"]])
        for i, arm in enumerate(ARMS):
            pictures_arm = pictures(folder / "images" / arm, cfg["jpeg_sha256"][arm])
            images[arm] = pictures_arm
            sim = encode(pictures_arm).astype(float) @ target.T
            for c in np.unique(truth):
                s = np.flatnonzero(truth == c)
                check(len(s) == 10, "Wrong gallery size")
                score[s, i, 0] = topk_credit(sim[np.ix_(s, s)], 1).diagonal()
            logits = z[arm]
            check(
                logits.shape == (120, 1854) and np.isfinite(logits).all(),
                "Invalid vocabulary logits",
            )
            score[:, i, 1] = topk_credit(logits, 1)[np.arange(120), ix[truth]]
            score[:, i, 2] = topk_credit(logits[:, ix], 1)[np.arange(120), truth]
            score[:, i, 3] = sim.diagonal()
            score[:, i, 4:] = [
                metrics(thumbnail(im), t)[1:]
                for im, t in zip(pictures_arm, target_rgb, strict=True)
            ]
    check(np.isfinite(score).all(), "Nonfinite actual JPEG endpoint")
    hist = a.data / "fresh-mix/evaluation/mix13/per_row.npz"
    check(
        sha(hist) == old["row_scores_sha256"] == spec["native_scores"],
        "Archived native per-row scores differ",
    )
    with np.load(hist, allow_pickle=False) as z:
        oldarm = z["methods"].tolist().index("learned")
        oldmetrics = z["metrics"].tolist()
        strict = z["strict"]
        check(
            np.array_equal(z["rows"], rows)
            and np.array_equal(
                strict, np.isin(rows // 10, w["panels"][PANEL]["primary_object_ids"])
            )
            and strict.sum() == 110
            and all(
                np.array_equal(score[:, 0, m], z["scores"][:, oldarm, oldmetrics.index(key)])
                for m, key in enumerate(METRICS[:3])
            ),
            "Native DINO or naming per-row replay failed",
        )
    nonself = rows != donors
    same = nonself & (labels == labels[d])
    masks = dict(
        all=np.ones(120, bool),
        strict=strict,
        nonself=nonself,
        nonself_same_object=same,
        nonself_same_object_and_correct_word=same & (labels == words),
    )
    pops = {k: summarize(score, truth, mask, plan["seed"]) for k, mask in masks.items()}
    allrows = pops["all"]
    gate = (
        all(allrows["dino_ten"]["own_minus"][k]["ci95"][0] > 0 for k in ("native", "wrong"))
        and allrows["full1854"]["own_minus"]["native"]["difference"] >= 0
    )
    fp = a.out / "per_row.npz"
    frozen_npz(
        fp,
        dict(
            rows=rows,
            truth=truth,
            donor_rows=donors,
            arms=np.array(ARMS),
            metrics=np.array(METRICS),
            scores=score,
            **masks,
        ),
    )
    gallery(a.out / "private.jpg", ("original", *ARMS), r, words, images)
    gallery(a.out / "generated.jpg", ARMS, r, words, images)
    result = dict(
        design_sha256=LOCK,
        subject=SUB,
        panel=PANEL,
        ref_sha256=r["sha"],
        packet_sha256=cfg["packet_sha256"],
        scorer_sha256=sha(Path(__file__)),
        settings_sha256=sha(folder / "settings.json"),
        logits_sha256=sha(naming),
        dino_sha256=FILES["model.safetensors"],
        clip_sha256=CLIP_SHA,
        photos_zip_sha256=ZIP_SHA,
        native_jpeg_replays=120,
        native_score_replays_per_metric=120,
        per_row_sha256=sha(fp),
        fixed_generated_gallery_sha256=sha(a.out / "generated.jpg"),
        uncertainty="Paired percentile intervals sample entire true objects, with every eligible row retained. Differences in DINO10, full1854 and closed12 are fractions; RGB RMSE uses unit-intensity pixels.",
        populations=pops,
        advance=bool(gate),
    )
    freeze(a.release, result)
    print(
        "all120",
        {k: allrows[k]["top1_credit"] for k in METRICS[:3]},
        allrows["dino_ten"]["own_minus"],
        "advance",
        gate,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("outputs/protected-photo/images"))
    p.add_argument(
        "--model", type=Path, default=Path("outputs/photo-structure/dino/facebook-dinov2-small")
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--release", type=Path, default=PUB / "protected_photo_images.json")
    run(p.parse_args())
