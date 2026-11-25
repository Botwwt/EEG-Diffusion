"""Evaluate the independently fitted color correction on each participant's mixed EEG renders."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from fit_spatial_photo import LUMA, PLAN_SHA, original, render, small, table
from PIL import Image
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import check
from score_reconstruction_jpegs import registered
from score_spatial_photo import pixels
from score_within_photo_dino import ZIP_SHA

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
DESIGN = PUBLIC / "personal12_color_design.json"
ARMS = ("baseline", "neutral", "prior", "own", "wrong")


def roots(a, subject):
    fit = (
        a.data / "trial-object-encoder/personal12"
        if subject == "sub-04"
        else a.data / "trial-object-replication" / subject
    )
    inputs = fit / "render_inputs"
    models = a.data / "spatial-photo"
    src = inputs / "renders/personal12" / subject
    model = models / "models" / f"{subject}.npz"
    check(sha(model) == read(DESIGN)["spatial_model_sha256"][subject], "Changed color estimator")
    return inputs, models, src, model


def spatial_render(a, subject):
    inputs, models, src, _ = roots(a, subject)
    render(
        SimpleNamespace(
            out=models,
            subject=subject,
            panel="personal12",
            base=src,
            packet=inputs / "conditions" / subject / "personal12/conditions_512.npz",
            eeg=a.data / "object-head-confirmation/eeg" / (subject + ".npz"),
            confirmation=None,
            images=a.images,
            manifest=a.manifest,
        )
    )


def chroma(x):
    cells = x.reshape(8, 8, 8, 8, 3).mean((1, 3))
    y = cells @ LUMA
    return cells[..., [0, 2]] - y[..., None]


def rows(a, subject, tab, archive):
    inputs, models, src, model = roots(a, subject)
    dest = models / "renders/personal12" / subject
    cfg = read(dest / "settings.json")
    ref = registered(
        a.data / "object-head-confirmation/evaluation/references" / subject, "personal12"
    )
    check(
        cfg["source_settings_sha256"] == sha(src / "settings.json")
        and cfg["packet_sha256"]
        == sha(inputs / "conditions" / subject / "personal12/conditions_512.npz")
        and cfg["subject_model_sha256"] == sha(model)
        and cfg["eeg_sha256"] == sha(a.data / "object-head-confirmation/eeg" / (subject + ".npz"))
        and cfg["original_archive_sha256"] == ZIP_SHA
        and cfg["design_sha256"] == PLAN_SHA
        and np.array_equal(cfg["rows"], ref["rows"])
        and np.array_equal(cfg["folds"], ref["folds"]),
        "EEG, image or fold receipt mismatch",
    )
    index = {r: i for i, r in enumerate(cfg["rows"])}
    donor = [index[r] for r in cfg["donor_rows"]]
    labels = ref["labels"]
    words = np.array(cfg["predicted_words"])
    check(
        np.array_equal(words, words[donor]) and np.array_equal(ref["folds"], ref["folds"][donor]),
        "Wrong controls",
    )
    strict = (np.arange(len(labels)) != donor) & (words == labels) & (labels == labels[donor])
    values = {}
    for i, row in enumerate(ref["rows"]):
        gt = small(original(archive, tab, int(row)))
        target = chroma(gt)
        for k in ARMS:
            path = dest / "images" / k / f"{i:03d}.jpg"
            check(sha(path) == cfg["jpeg_sha256"][k][i], "Changed color-study JPEG")
            with Image.open(path) as im:
                check(im.size == (512, 512), "Wrong dimensions")
                rgb = small(im.convert("RGB"))
            scores = pixels(rgb, gt)
            scores["chroma_rmse"] = float(np.sqrt(np.mean((chroma(rgb) - target) ** 2)))
            for metric in ("chroma_rmse", "rgb_rmse", "gray_ssim"):
                values.setdefault(metric, {}).setdefault(k, []).append(scores[metric])
    return labels, strict, values, sha(dest / "settings.json")


def summarize(reports, use_strict):
    labels = np.concatenate([r[0] for r in reports])
    mask = (
        np.concatenate([r[1] for r in reports]) if use_strict else np.ones(len(labels), dtype=bool)
    )
    classes, ix = np.unique(labels, return_inverse=True)
    counts = np.bincount(ix[mask], minlength=len(classes))
    draw = np.random.default_rng(260556).integers(len(classes), size=(10000, len(classes)))
    denom = counts[draw].sum(1)
    usable = denom > 0
    metrics = {}
    for metric in reports[0][2]:
        vals = {k: np.concatenate([r[2][metric][k] for r in reports])[mask] for k in ARMS}
        delta = {}
        for control in ("neutral", "prior", "wrong"):
            d = vals["own"] - vals[control]
            sums = np.bincount(ix[mask], weights=d, minlength=len(classes))
            boot = sums[draw[usable]].sum(1) / denom[usable]
            delta[control] = dict(
                mean=float(d.mean()), ci95=np.quantile(boot, [0.025, 0.975]).tolist()
            )
        metrics[metric] = dict(means={k: float(v.mean()) for k, v in vals.items()}, own_minus=delta)
    return dict(n=int(mask.sum()), contributing_objects=int((counts > 0).sum()), metrics=metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("render", "score"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument(
        "--subjects", nargs="+", choices=read(DESIGN)["subjects"], default=read(DESIGN)["subjects"]
    )
    p.add_argument("--release", type=Path, default=PUBLIC / "personal12_color.json")
    a = p.parse_args()
    a.images = a.data / "photo-structure/training_images.zip"
    a.manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    if a.stage == "render":
        for s in a.subjects:
            spatial_render(a, s)
    else:
        check(
            sha(a.images) == ZIP_SHA and a.subjects == read(DESIGN)["subjects"],
            "Wrong cohort or originals",
        )
        tab = table(a)
        with ZipFile(a.images) as archive:
            studies = {s: rows(a, s, tab, archive) for s in a.subjects}
        stats = {
            name: {k: summarize(v, strict) for k, strict in (("all", False), ("strict", True))}
            for name, v in {
                **{s: [r] for s, r in studies.items()},
                "pooled": list(studies.values()),
            }.items()
        }
        comparisons = stats["pooled"]["strict"]["metrics"]["chroma_rmse"]["own_minus"]
        passed = all(v["ci95"][1] < 0 for v in comparisons.values())
        freeze(
            a.release,
            dict(
                design_sha256=sha(DESIGN),
                scorer_sha256=sha(Path(__file__)),
                source_zip_sha256=ZIP_SHA,
                settings_sha256={s: r[3] for s, r in studies.items()},
                results=stats,
                color_gate=passed,
            ),
        )
        print("Strict color:", comparisons, "gate:", passed)
