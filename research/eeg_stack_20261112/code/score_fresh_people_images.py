"""Independent naming and photograph scores for two separately fitted single-person EEG models."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fit_spatial_photo import original, table
from PIL import Image, ImageDraw
from register_fresh_photo_panel import freeze, read, sha
from render_fresh_people_images import (
    IMAGE_LOCK,
    METHODS,
    PANELS,
    PUB,
    SUBJECTS,
    case,
    folder,
    gate,
    verified,
)
from render_personal_query import DICTIONARY
from render_within_photo_ip import check
from run_prospective_readout import frozen_npz
from score_fresh_mix_images import METRICS
from score_official_objects import topk_credit
from score_reconstruction_jpegs import CLIP_SHA, registered
from score_support_view import pictures
from score_within_photo_dino import FILES, ZIP_SHA, encoder


def sheet(path, sub, panel, pics, reg, words, private):
    arms = ("original", *METHODS) if private else METHODS
    n = 122
    w = n * len(arms) + 12
    h = n + 58
    im = Image.new("RGB", (2 * w, 6 * h + 24), "white")
    draw = ImageDraw.Draw(im)
    draw.text((4, 4), f"{sub} | {panel} | lowest original row per object", fill="black")
    for k in range(12):
        idx = np.flatnonzero(reg["truth"] == k)
        r = idx[np.argmin(reg["rows"][idx])]
        x = 4 + (k % 2) * w
        y = 24 + (k // 2) * h
        draw.text((x, y), f"True: {reg['labels'][r]}   Row {reg['rows'][r]}", fill="black")
        for j, m in enumerate(arms):
            u = x + n * j
            im.paste(pics[m][r].resize((n, n), Image.Resampling.LANCZOS), (u, y + 15))
            draw.text((u, y + n + 18), m, fill="black")
            if m in words:
                draw.text((u, y + n + 32), str(words[m][r]), fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", quality=90, subsampling=0)


def measure(a, sub, panel, locked, word_score, enc, bank, tab):
    cohort = a.study / sub
    root = cohort / "render_inputs"
    meta = locked["panels"][sub][panel]
    ref = registered(cohort / "references", panel)
    rows, folds, labels = ref["rows"], ref["folds"], ref["truth"]
    obj = rows // 10
    check(
        ref["sha"] == meta["registration_sha256"]
        and np.array_equal(obj, np.array(meta["object_ids"])[labels])
        and all(tab[int(r)]["unique_id"] == w for r, w in zip(rows, ref["labels"], strict=True)),
        "Reference registration or original stimulus changed",
    )
    data = case(root, sub, panel)
    cfg, repeat = verified(root, sub, panel, data)
    wordpath = cohort / "decisions" / (panel + ".npz")
    check(sha(wordpath) == word_score["decisions"][sub][panel], "Decision digest differs")
    with np.load(wordpath, allow_pickle=False) as z:
        check(
            np.array_equal(z["rows"], rows)
            and np.array_equal(z["words"], ref["vocab"])
            and np.array_equal(z["folds"], folds)
            and all(np.array_equal(z["words"][z[m]], data[m][2]) for m in METHODS),
            "Word readout no longer matches the sealed classifier",
        )
    clip = root / "scores" / panel / "clip_logits.npz"
    out = cohort / "evaluation/images" / panel
    scores = np.zeros((120, 2, 3), float)
    pics = {}
    words = {m: data[m][2] for m in METHODS}
    with np.load(clip, allow_pickle=False) as z:
        names = z["names"].tolist()
        check(
            str(z["model_sha256"]) == CLIP_SHA
            and str(z["dictionary_sha256"]) == DICTIONARY
            and np.array_equal(rows, z["rows"])
            and len(names) == len(set(names)) == 1854,
            "Naming model or dictionary changed",
        )
        select = np.array([names.index(w) for w in ref["vocab"]])
        for j, m in enumerate(METHODS):
            base = folder(root, sub, panel, m)
            idx = np.array([ref["vocab"].tolist().index(w) for w in words[m]])
            check(
                str(z[m + "_settings_sha256"]) == sha(base / "settings.json")
                and np.array_equal(ref["calibration"][folds, idx], data[m][4]),
                "Generated scores or eight supports changed",
            )
            logits = z[m]
            check(logits.shape == (120, 1854) and np.isfinite(logits).all(), "Missing rows")
            scores[:, j, 0] = topk_credit(logits, 1)[np.arange(120), select[labels]]
            scores[:, j, 1] = topk_credit(logits[:, select], 1)[np.arange(120), labels]
            pics[m] = pictures(base / "images/own", cfg[m]["jpeg_sha256"]["own"])
    pics["original"] = [original(bank, tab, int(r)) for r in rows]
    real = enc(pics["original"]).astype(float)
    for j, m in enumerate(METHODS):
        sim = enc(pics[m]).astype(float) @ real.T
        for k in range(12):
            ids = np.flatnonzero(labels == k)
            check(len(ids) == 10, "Incomplete within-object gallery")
            scores[ids, j, 2] = topk_credit(sim[np.ix_(ids, ids)], 1).diagonal()
    equal = words["legacy"] == words["learned"]
    check(
        np.array_equal(scores[equal, 0], scores[equal, 1]) and np.isfinite(scores).all(),
        "Same JPEG got different credit",
    )
    masks = np.isin(obj, meta["primary_object_ids"])
    file = out / "per_row.npz"
    frozen_npz(
        file,
        dict(
            rows=rows,
            object_ids=obj,
            primary=masks,
            methods=np.array(METHODS),
            metrics=np.array(METRICS),
            scores=scores,
        ),
    )
    generated = a.gallery / f"fresh_people_{sub}_{panel}.jpg"
    sheet(out / "private.jpg", sub, panel, pics, ref, words, True)
    sheet(generated, sub, panel, pics, ref, words, False)
    receipt = dict(
        registration_sha256=ref["sha"],
        decision_sha256=sha(wordpath),
        clip_logits_sha256=sha(clip),
        packet_sha256={m: data[m][-1] for m in METHODS},
        settings_sha256={m: sha(folder(root, sub, panel, m) / "settings.json") for m in METHODS},
        per_row_sha256=sha(file),
        gallery_sha256=sha(generated),
        independent_redraw=repeat,
    )
    return dict(rows=rows, objects=obj, primary=masks, scores=scores), receipt


def counts(obs):
    x = np.asarray(obs)
    d = x[:, 1] - x[:, 0]
    return {
        m: dict(
            legacy_hits=float(x[:, 0, j].sum()),
            learned_hits=float(x[:, 1, j].sum()),
            gained_credit=float(np.maximum(d[:, j], 0).sum()),
            lost_credit=float(np.maximum(-d[:, j], 0).sum()),
            delta_pp=float(100 * d[:, j].mean()),
        )
        for j, m in enumerate(METRICS)
    }


def cluster(obs, ids, seed):
    # Scores are row-paired. All participants who saw the same object travel in the same bootstrap cluster.
    groups, index = np.unique(ids, return_inverse=True)
    widths = np.bincount(index)
    diff = obs[:, 1] - obs[:, 0]
    sums = np.zeros((len(groups), len(METRICS)))
    np.add.at(sums, index, diff)
    draws = np.random.default_rng(seed).integers(len(groups), size=(10000, len(groups)))
    ratios = 100 * sums[draws].sum(1) / widths[draws].sum(1)[:, None]
    m = counts(obs)
    for i, k in enumerate(METRICS):
        m[k]["ci95_pp"] = np.quantile(ratios[:, i], [0.025, 0.975]).tolist()
    return dict(
        n=len(ids),
        unique_objects=len(groups),
        object_observations={str(n): int(np.sum(widths == n)) for n in sorted(set(widths))},
        metrics=m,
    )


def summarize(cases, primary, seed):
    ppl = {}
    all_scores = []
    all_objs = []
    all_rows = []
    for sub, c in cases.items():
        here = []
        objs = []
        keys = []
        menus = {}
        for p, v in c.items():
            use = v["primary"] if primary else np.ones(len(v["rows"]), bool)
            x = v["scores"][use]
            o = v["objects"][use]
            rows = v["rows"][use]
            here.append(x)
            objs.append(o)
            keys.append(rows)
            menus[p] = dict(n=len(x), metrics=counts(x))
        one = np.concatenate(here)
        oi = np.concatenate(objs)
        ri = np.concatenate(keys)
        check(
            len(ri) == len(set(ri)) and np.array_equal(oi, ri // 10),
            "Rows repeated within participant",
        )
        ppl[sub] = cluster(one, oi, seed) | dict(menus=menus)
        all_scores.append(one)
        all_objs.append(oi)
        all_rows.append(ri)
    obj = np.concatenate(all_objs)
    pooled = cluster(np.concatenate(all_scores), obj, seed)
    ns = len(set(np.concatenate(all_rows)))
    shared = len(set(all_rows[0]) & set(all_rows[1]))
    check(
        (pooled["n"], pooled["unique_objects"], ns, shared)
        == ((700, 35, 350, 350) if primary else (960, 60, 600, 360)),
        "Wrong repeated-participant cluster population",
    )
    return pooled | dict(unique_photos=ns, photos_observed_by_both=shared, by_subject=ppl)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, default=Path("outputs/fresh-people"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument(
        "--model", type=Path, default=Path("outputs/photo-structure/dino/facebook-dinov2-small")
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gallery", type=Path, default=PUB.parents[1] / "gallery")
    p.add_argument("--out", type=Path, default=PUB / "fresh_people_images.json")
    a = p.parse_args()
    r = gate(PUB)
    lock = read(PUB / "fresh_people_images_design.json")
    wordreg = PUB / "fresh_people_design.json"
    check(sha(wordreg) == lock["word_design_sha256"], "Word registration changed")
    spec = read(wordreg)
    a.manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    expected = read(PUB / "fresh_mix_design.json")["sources"]["original_photo_rows"]
    z = a.data / "photo-structure/training_images.zip"
    check(sha(a.manifest) == expected and sha(z) == ZIP_SHA, "Wrong original stimuli")
    tab = table(a)
    enc = encoder(a)
    cases = {s: {} for s in SUBJECTS}
    receipts = {s: {} for s in SUBJECTS}
    with ZipFile(z) as bank:
        for s in SUBJECTS:
            for m in PANELS:
                cases[s][m], receipts[s][m] = measure(a, s, m, spec, r, enc, bank, tab)
                print(s, m, "verified", flush=True)
    cohorts = {k: summarize(cases, k == "primary", lock["seed"]) for k in ("primary", "all")}
    main = cohorts["primary"]
    replication = main["metrics"]["full1854"]["ci95_pp"][0] > 0 and all(
        v["metrics"]["full1854"]["delta_pp"] >= 0 for v in main["by_subject"].values()
    )
    freeze(
        a.out,
        dict(
            image_registration_sha256=IMAGE_LOCK,
            word_registration_sha256=sha(wordreg),
            word_result_sha256=lock["word_results_sha256"],
            driver_sha256=sha(Path(__file__)),
            dino_sha256=FILES["model.safetensors"],
            clip_sha256=CLIP_SHA,
            originals_sha256=ZIP_SHA,
            seed=lock["seed"],
            populations=cohorts,
            replicated_primary=bool(replication),
            cases=receipts,
        ),
    )
    print(main["metrics"], "replicated", replication, flush=True)


if __name__ == "__main__":
    main()
