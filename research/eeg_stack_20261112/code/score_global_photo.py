"""Measure photograph identity on real EEG images and on separately labeled visual interventions."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fit_spatial_photo import original, table
from refine_registered_color import metrics, thumbnail
from register_fresh_photo_panel import freeze, read, sha
from render_global_photo import ARMS, LOCK, PANEL, SUB, verify
from render_within_photo_ip import check
from run_prospective_readout import frozen_npz
from score_canonical_neighbor_images import gallery
from score_official_objects import topk_credit
from score_reconstruction_jpegs import CLIP_SHA, registered
from score_support_view import pictures
from score_within_photo_dino import FILES, ZIP_SHA, encoder

BASE = Path(__file__).resolve().parents[1]


def summarize(scores, truth, mask, seed):
    ids = truth[mask]
    groups = np.unique(ids)
    denom = np.array([(ids == c).sum() for c in groups])
    draws = np.random.default_rng(seed).integers(len(groups), size=(10000, len(groups)))
    means = {}
    draws_by_arm = {}
    for k, x in scores.items():
        totals = np.array([x[mask][ids == c].sum() for c in groups])
        means[k] = float(totals.sum() / len(ids))
        draws_by_arm[k] = totals[draws].sum(1) / denom[draws].sum(1)
    pairs = [("matched", k) for k in ("native", "prior", "wrong")] + [
        ("reference", "foil"),
        ("reference", "native"),
    ]
    diffs = {
        a + "_minus_" + b: dict(
            delta=means[a] - means[b],
            ci95=np.quantile(draws_by_arm[a] - draws_by_arm[b], [0.025, 0.975]).tolist(),
        )
        for a, b in pairs
    }
    return dict(n=len(ids), objects=len(groups), mean=means, paired=diffs)


def run(a):
    real = verify(a.root / "eeg", "eeg")
    oracle = verify(a.root / "oracle", "oracle")
    rows, folds, words, donors, support = real[:5]
    check(
        all(np.array_equal(x, y) for x, y in zip(real[:5], oracle[:5], strict=True)),
        "Diagnostic task differs from the matched EEG task",
    )
    reg = registered(a.data / "object-head-confirmation/evaluation/references" / SUB, PANEL)
    labels, truth = reg["labels"], reg["truth"]
    check(
        np.array_equal(reg["rows"], rows) and np.array_equal(reg["folds"], folds),
        "Gallery registration mismatch",
    )
    a.manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    tab = table(a)
    stimuli = a.data / "photo-structure/training_images.zip"
    check(
        sha(stimuli) == ZIP_SHA
        and all(tab[r]["unique_id"] == lab for r, lab in zip(rows, labels, strict=True)),
        "Original stimuli mismatch",
    )
    with ZipFile(stimuli) as z:
        pictures_by_arm = {"original": [original(z, tab, int(r)) for r in rows]}
    enc = encoder(a)
    target = enc(pictures_by_arm["original"]).astype(float)
    rgb = [thumbnail(im) for im in pictures_by_arm["original"]]
    scores = {}
    receipts = {}
    for stage, names in ARMS.items():
        root = a.root / stage
        folder = root / "renders" / PANEL / SUB
        p = folder / "settings.json"
        cfg = read(p)
        plan = read(root / "design.json")
        check(
            cfg["packet_sha256"] == plan["packet_sha256"]
            and cfg["design_sha256"] == sha(root / "design.json")
            and cfg["radius_sha256"] == plan["radius_sha256"]
            and all(cfg[k] == v for k, v in plan["generator"].items())
            and all(
                np.array_equal(cfg[k], x)
                for k, x in (
                    ("seeds", rows + 923761),
                    ("rows", rows),
                    ("folds", folds),
                    ("predicted_words", words),
                    ("support_rows", support),
                    ("donor_rows", donors),
                )
            )
            and cfg["batch_size"] == 8
            and set(cfg["jpeg_sha256"]) == set(names),
            "Returned images do not share the registered intervention",
        )
        if stage == "eeg":
            check(
                cfg["jpeg_sha256"]["native"] == plan["native_hashes"], "Native JPEG replay failed"
            )
        file = folder / "naming.npz"
        receipts[stage] = dict(
            settings_sha256=sha(p), packet_sha256=cfg["packet_sha256"], logits_sha256=sha(file)
        )
        with np.load(file, allow_pickle=False) as z:
            check(
                str(z["model_sha256"]) == CLIP_SHA and str(z["settings_sha256"]) == sha(p),
                "Naming model mismatch",
            )
            vocab = z["names"].tolist()
            ix = np.array([vocab.index(w) for w in reg["vocab"]])
            for k in names:
                ims = pictures(folder / "images" / k, cfg["jpeg_sha256"][k])
                pictures_by_arm[k] = ims
                sim = enc(ims).astype(float) @ target.T
                hit = np.empty(120)
                for c in np.unique(truth):
                    sel = np.flatnonzero(truth == c)
                    check(len(sel) == 10, "Incomplete ten-photo gallery")
                    hit[sel] = topk_credit(sim[np.ix_(sel, sel)], 1).diagonal()
                pixels = np.array(
                    [metrics(thumbnail(im), t) for im, t in zip(ims, rgb, strict=True)]
                )
                full = topk_credit(z[k], 1)[np.arange(120), ix[truth]]
                closed = topk_credit(z[k][:, ix], 1)[np.arange(120), truth]
                scores[k] = dict(
                    dino_ten=hit,
                    full1854=full,
                    closed12=closed,
                    dino_cosine=sim.diagonal(),
                    rgb_rmse32=pixels[:, 1],
                    luma32=pixels[:, 2],
                )
    base = a.data / "trial-object-encoder/personal12/evaluation"
    oldc = read(base / "clip/sub-04.json")["cases"]["personal12/sub-04"]["per_row"]
    oldd = read(base / "pixels/sub-04/personal12/sub-04/per_query.json")
    check(
        all(np.array_equal(oldc[k]["own"], scores["native"][k]) for k in ("closed12", "full1854"))
        and np.array_equal(
            [r["intra10"]["own"] for r in oldd["images"]], scores["native"]["dino_ten"]
        ),
        "Published native per-row replay failed",
    )
    path = a.root / "scores.npz"
    frozen_npz(
        path,
        {"rows": rows, "truth": truth}
        | {arm + "_" + m: col for arm, v in scores.items() for m, col in v.items()},
    )
    seeds = read(a.root / "eeg/protocol.json")["seed"]
    word_correct = words == labels
    masks = {
        "all": np.ones(120, dtype=bool),
        "correct_word": word_correct,
        "incorrect_word": ~word_correct,
    }
    summary = {
        k: {
            m: summarize({arm: v[m] for arm, v in scores.items()}, truth, mask, seeds)
            for m in scores["native"]
        }
        for k, mask in masks.items()
    }
    main = summary["all"]
    diff = main["dino_ten"]["paired"]
    advance = (
        all(diff["matched_minus_" + k]["ci95"][0] > 0 for k in ("native", "prior"))
        and diff["matched_minus_wrong"]["delta"] > 0
        and main["full1854"]["paired"]["matched_minus_native"]["delta"] >= 0
    )
    result = dict(
        protocol_sha256=LOCK,
        scorer_sha256=sha(Path(__file__)),
        dino_sha256=FILES["model.safetensors"],
        naming_model_sha256=CLIP_SHA,
        receipts=receipts,
        per_row_sha256=sha(path),
        populations=summary,
        totals={
            m: {k: float(v[m].sum()) for k, v in scores.items()}
            for m in ("dino_ten", "full1854", "closed12")
        },
        advance=bool(advance),
    )
    freeze(a.release, result)
    gallery(
        a.root / "private.jpg",
        ("original", "native", "matched", "reference", "foil"),
        reg,
        words,
        pictures_by_arm,
    )
    gallery(
        a.root / "eeg_generated.jpg",
        ("native", "matched", "prior", "wrong"),
        reg,
        words,
        pictures_by_arm,
    )
    gallery(a.root / "reference_diagnostic.jpg", ("reference", "foil"), reg, words, pictures_by_arm)
    print(result["totals"], "advance", result["advance"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/global-photo"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument(
        "--model", type=Path, default=Path("outputs/photo-structure/dino/facebook-dinov2-small")
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--release", type=Path, default=BASE / "results/object_fidelity/global_photo_images.json"
    )
    run(p.parse_args())
