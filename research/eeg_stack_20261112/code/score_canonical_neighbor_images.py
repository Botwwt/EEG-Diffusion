"""Score the sealed canonical-neighbor JPEGs with CLIP-L/14 and DINOv2-S/14."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fit_spatial_photo import original, table
from PIL import Image, ImageDraw
from refine_registered_color import metrics, thumbnail
from register_fresh_photo_panel import freeze, read, sha
from render_personal_query import DICTIONARY, TURBO
from render_within_photo_ip import ADAPTER_SHA, check
from run_prospective_readout import frozen_npz
from score_official_objects import load_dictionary, load_tokenizer, text_bank, topk_credit, unit
from score_reconstruction_jpegs import CLIP_SHA, raw_features, registered
from score_support_view import pictures
from score_within_photo_dino import FILES, ZIP_SHA, encoder

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
ARMS = ("native", "own", "wrong", "zero")
PROTOCOL = PUBLIC / "canonical_neighbor_personal12_design.json"


def source(a):
    folder = a.root / "renders/personal12" / a.subject
    cfg = read(folder / "settings.json")
    check(
        cfg["subject"] == a.subject
        and cfg["panel"] == "personal12"
        and cfg["adapter_sha256"] == ADAPTER_SHA
        and cfg["turbo_sha256"] == TURBO
        and cfg["dictionary_sha256"] == DICTIONARY
        and np.array_equal(cfg["seeds"], 923761 + np.array(cfg["rows"]))
        and len(cfg["rows"]) == 120
        and cfg["design_sha256"] == sha(a.root / "design.json")
        and cfg["radius_sha256"] == sha(a.root / "radius.json"),
        "Unexpected reconstruction protocol",
    )
    check(
        set(cfg["jpeg_sha256"]) == set(ARMS)
        and cfg["jpeg_sha256"]["zero"] == cfg["jpeg_sha256"]["native"],
        "Unchanged condition must reproduce native",
    )
    return folder, cfg


def clip(a):
    from eval_instance_starts import load_visual_scorer

    folder, cfg = source(a)
    model, h = load_visual_scorer(a.archive.resolve())
    check(h == CLIP_SHA and sha(a.dictionary) == DICTIONARY, "Wrong naming assets")
    names, common, inverse, _, _ = load_dictionary(a.dictionary)
    tok = load_tokenizer(a.archive / "assets/native_generator_v243/turbo/tokenizer")
    text = text_bank(model, tok, common, 16)
    arrays = {}
    for arm in ARMS:
        pics = pictures(folder / "images" / arm, cfg["jpeg_sha256"][arm])
        arrays[arm] = (
            arrays["native"]
            if arm == "zero"
            else (unit(raw_features(model, pics.__getitem__, 120, 16)) @ text.T)[:, inverse]
        )
    frozen_npz(
        folder / "clip_logits.npz",
        dict(
            names=np.asarray(names),
            model_sha256=h,
            settings_sha256=sha(folder / "settings.json"),
            **arrays,
        ),
    )


def aggregate(vals, ids, seed):
    objects, labels = np.unique(ids, return_inverse=True)
    x = np.stack([vals[k] for k in ARMS], axis=1)
    totals = np.zeros((len(objects), len(ARMS)))
    np.add.at(totals, labels, x)
    draws = np.random.default_rng(seed).integers(len(objects), size=(10000, len(objects)))
    b = totals[draws].sum(1) / np.bincount(labels)[draws].sum(1)[:, None]
    j = ARMS.index("own")
    return dict(
        n=len(x),
        objects=len(objects),
        means={k: float(x[:, i].mean()) for i, k in enumerate(ARMS)},
        own_minus={
            k: dict(
                delta=float(np.mean(x[:, j] - x[:, i])),
                ci95=np.quantile(b[:, j] - b[:, i], [0.025, 0.975]).tolist(),
            )
            for i, k in enumerate(ARMS)
            if k != "own"
        },
    )


def gallery(p, kinds, reg, words, data):
    first = [
        np.flatnonzero(reg["truth"] == k)[np.argmin(reg["rows"][reg["truth"] == k])]
        for k in range(12)
    ]
    width = 112
    sheet = Image.new("RGB", (2 * (len(kinds) * width + 12), 6 * (width + 36)), "white")
    draw = ImageDraw.Draw(sheet)
    for j, q in enumerate(first):
        x = (j % 2) * (len(kinds) * width + 12)
        y = (j // 2) * (width + 36)
        draw.text((x, y), f"{reg['labels'][q]} | EEG: {words[q]} | {reg['rows'][q]}", fill="black")
        for i, k in enumerate(kinds):
            sheet.paste(
                data[k][q].resize((width, width), Image.Resampling.LANCZOS), (x + i * width, y + 15)
            )
            draw.text((x + i * width, y + width + 16), k, fill="black")
    sheet.save(p, quality=94)


def score(a):
    folder, cfg = source(a)
    spec = read(PROTOCOL)
    check(
        a.subject in spec["subjects"] and spec["comparison_arms"] == list(ARMS),
        "Unknown study population",
    )
    seed = spec["seed"]
    reg = registered(
        a.data / "object-head-confirmation/evaluation/references" / a.subject, "personal12"
    )
    rows, truth, folds, labels = (reg[k] for k in ("rows", "truth", "folds", "labels"))
    words = np.array(cfg["predicted_words"])
    native = (
        a.data / "trial-object-encoder/personal12"
        if a.subject == "sub-04"
        else a.data / "trial-object-replication" / a.subject
    )
    nativefile = native / "render_inputs/renders/personal12" / a.subject / "settings.json"
    old = read(nativefile)
    check(
        all(
            old[k] == cfg[k]
            for k in (
                "rows",
                "folds",
                "predicted_words",
                "donor_rows",
                "support_rows",
                "seeds",
                "radius",
                "turbo_sha256",
                "adapter_sha256",
                "dictionary_sha256",
            )
        )
        and old["jpeg_sha256"]["own"] == cfg["jpeg_sha256"]["native"]
        and np.array_equal(rows, cfg["rows"])
        and np.array_equal(folds, cfg["folds"]),
        "Native per-row comparison failed",
    )
    menu = reg["vocab"].tolist()
    check(
        np.array_equal(
            cfg["support_rows"], reg["calibration"][folds, [menu.index(w) for w in words]]
        ),
        "Unregistered supports",
    )
    imgs = {k: pictures(folder / "images" / k, cfg["jpeg_sha256"][k]) for k in ARMS}
    a.manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    records = table(a)
    archive = a.data / "photo-structure/training_images.zip"
    check(
        sha(archive) == ZIP_SHA
        and all(
            records[int(r)]["unique_id"] == label for r, label in zip(rows, labels, strict=True)
        ),
        "Reference identities changed",
    )
    with ZipFile(archive) as bank:
        imgs["original"] = [original(bank, records, int(r)) for r in rows]
    per = {k: {} for k in ARMS}
    with np.load(folder / "clip_logits.npz", allow_pickle=False) as z:
        check(
            str(z["settings_sha256"]) == sha(folder / "settings.json")
            and str(z["model_sha256"]) == CLIP_SHA,
            "CLIP observations changed",
        )
        names = z["names"].tolist()
        select = np.array([names.index(w) for w in menu])
        for k in ARMS:
            s = z[k]
            per[k]["full1854"] = topk_credit(s, 1)[np.arange(120), select[truth]]
            per[k]["closed12"] = topk_credit(s[:, select], 1)[np.arange(120), truth]
            per[k]["predicted_word12"] = topk_credit(s[:, select], 1)[
                np.arange(120), [menu.index(w) for w in words]
            ]
    enc = encoder(a)
    r = enc(imgs["original"]).astype(float)
    rgb = [thumbnail(pic) for pic in imgs["original"]]
    pairs = np.array(
        [np.flatnonzero((truth == k) & (folds == f)) for k in range(12) for f in range(5)]
    )
    left, right = pairs[(words[pairs] == labels[pairs]).all(1)].T
    pair = {}
    baseline = enc(imgs["native"]).astype(float) @ r.T
    for k in ARMS:
        c = baseline if k in ("native", "zero") else enc(imgs[k]).astype(float) @ r.T
        hit = np.empty(120)
        for obj in range(12):
            idx = np.flatnonzero(truth == obj)
            hit[idx] = topk_credit(c[np.ix_(idx, idx)], 1).diagonal()
        per[k]["dino_ten"] = hit
        per[k]["dino_original_cosine"] = c.diagonal()
        d1, d2 = c[left, left] - c[left, right], c[right, right] - c[right, left]

        def credit(delta):
            return (delta > 1e-12).astype(float) + 0.5 * (np.abs(delta) <= 1e-12)

        pair[k] = {"joint": credit(d1 + d2), "individual": (credit(d1) + credit(d2)) / 2}
        measurements = np.array(
            [metrics(thumbnail(pic), ref) for pic, ref in zip(imgs[k], rgb, strict=True)]
        )
        per[k]["rgb_rmse32"], per[k]["luma_pearson32"] = measurements[:, 1], measurements[:, 2]
    oldc = read(native / "evaluation/clip" / (a.subject + ".json"))["cases"][
        "personal12/" + a.subject
    ]
    oldd = read(
        native / "evaluation/pixels" / a.subject / "personal12" / a.subject / "per_query.json"
    )
    check(
        oldc["settings_sha256"] == oldd["settings_sha256"] == sha(nativefile)
        and all(
            np.array_equal(per["native"][k], oldc["per_row"][k]["own"])
            for k in ("full1854", "closed12")
        )
        and np.array_equal(
            per["native"]["dino_ten"], [p["intra10"]["own"] for p in oldd["images"]]
        ),
        "Native scorer differs",
    )
    whole = {m: aggregate({k: x[m] for k, x in per.items()}, truth, seed) for m in per["own"]}
    matched = {
        m: aggregate({k: x[m] for k, x in pair.items()}, truth[left], seed) for m in pair["own"]
    }
    counts = {
        m: {k: float(d[m].sum()) for k, d in per.items()}
        for m in ("dino_ten", "full1854", "closed12")
    }
    lookup = {r: i for i, r in enumerate(rows)}
    donor = np.array([lookup[r] for r in cfg["donor_rows"]])
    check(
        np.array_equal(folds, folds[donor]) and np.array_equal(words, words[donor]),
        "Unexpected donor",
    )
    nonself = (rows != rows[donor]) & (truth == truth[donor])
    subset = aggregate({k: x["dino_ten"][nonself] for k, x in per.items()}, truth[nonself], seed)
    result = dict(
        subject=a.subject,
        counts=counts,
        all=whole,
        strict=matched,
        bootstrap_seed=seed,
        study_design_sha256=sha(PROTOCOL),
        nonself_same_object_word_and_fold=subset,
        advance=whole["dino_ten"]["own_minus"]["native"]["ci95"][0] > 0
        and subset["own_minus"]["wrong"]["delta"] > 0
        and counts["full1854"]["own"] >= counts["full1854"]["native"],
        settings_sha256=sha(folder / "settings.json"),
        clip_sha256=CLIP_SHA,
        scorer_sha256=sha(Path(__file__)),
        registration_sha256=reg["sha"],
        dino_sha256=FILES["model.safetensors"],
        native_settings_sha256=sha(nativefile),
        clip_logits_sha256=sha(folder / "clip_logits.npz"),
        identical_matched_native=sum(
            x == y
            for x, y in zip(cfg["jpeg_sha256"]["own"], cfg["jpeg_sha256"]["native"], strict=True)
        ),
    )
    freeze(a.release or folder / "evaluation.json", result)
    gallery(folder / "private.jpg", ("original", "native", "own", "wrong"), reg, words, imgs)
    gallery(folder / "generated.jpg", ("native", "own", "wrong"), reg, words, imgs)
    print(counts, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("clip", "score"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--subject", default="sub-04", choices=("sub-04", "sub-05", "sub-01"))
    p.add_argument("--archive", type=Path)
    p.add_argument("--dictionary", type=Path)
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--model", type=Path, default=Path("outputs/photo-structure/dino/facebook-dinov2-small")
    )
    p.add_argument("--release", type=Path)
    a = p.parse_args()
    {"clip": clip, "score": score}[a.stage](a)
