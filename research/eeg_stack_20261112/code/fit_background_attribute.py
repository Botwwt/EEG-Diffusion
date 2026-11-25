"""Fit and evaluate a within-object EEG readout for a preregistered background attribute."""

import argparse
import csv
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from measure_background_attribute import DESIGN, LOCK, check
from PIL import Image, ImageDraw
from register_fresh_photo_panel import freeze, read, sha
from render_official_words import crop512
from run_prospective_readout import frozen_npz
from scipy.linalg import solve

METHODS = ("prior", "eeg", "swapped_eeg", "photometry_eeg")
SUPPORT = np.array([[j for j in range(10) if j // 2 != i // 2] for i in range(10)])


def physical(a, spec, ids):
    receipt = a.data / "trial-object-inputs/sub-04/generic512.json"
    mont = a.data / "object-transfer/halves/sub-04.npz"
    check(
        sha(receipt) == spec["inputs"]["eeg"] and sha(mont) == spec["inputs"]["montage"],
        "Changed EEG inputs",
    )
    raw = read(receipt)
    with np.load(mont, allow_pickle=False) as z:
        names = z["channels"].tolist()
    pick = [raw["channel_names"].index(s) for s in names]
    check(len(pick) == 17 and raw["subject"] == "sub-04", "Wrong EEG montage")
    features = []
    for i in ids:
        src = receipt.parent / f"{i:04d}.npy"
        block = raw["blocks"][str(i)]
        check(
            sha(src) == block["output_sha256"] and int(block["official_row_start"]) == 10 * i,
            "EEG block mismatch",
        )
        x = np.load(src, allow_pickle=False)
        check(x.shape == (10, 4, 63, 125) and x.dtype == np.float32, "Wrong physical trial shape")
        x = np.take(x, pick, axis=2)[:, :, :, 15:115].astype(float)
        features.append(x.mean(1).reshape(10, 17, 20, 5).mean(-1).reshape(10, 340))
    return np.stack(features)


def annotation(a, i):
    file = a.out / "annotations" / f"{i:04d}.npz"
    with np.load(file, allow_pickle=False) as z:
        check(
            str(z["design"]) == LOCK
            and str(z["source_sha256"])
            == sha(Path(__file__).with_name("measure_background_attribute.py"))
            and np.array_equal(z["rows"], 10 * i + np.arange(10)),
            "Wrong photo annotation or row",
        )
        return {k: z[k] for k in ("labels", "appearance", "background", "jpeg_sha256")}


def within(z):
    return (z - z.mean(1, keepdims=True)).reshape(-1, z.shape[-1])


def ridge(x, y, penalty):
    return solve(x.T @ x + np.eye(x.shape[1]) * len(x) * penalty, x.T @ y, assume_a="pos")


def relative(z):
    return z - z[:, SUPPORT].mean(axis=2)


def scalefit(inc, target):
    return float(np.clip(np.sum(inc * target) / max(1e-12, np.sum(inc**2)), 0, 2))


def fit(a, d):
    fit_ids = d["fit_classes"]
    cal_ids = d["calibration_classes"]
    x = within(physical(a, d, fit_ids))
    records = [annotation(a, c) for c in fit_ids]
    phot = within(np.stack([z["appearance"] for z in records]))
    tgt = within(np.stack([z["labels"] for z in records])[:, :, None])
    xsd = x.std(0).clip(1e-6)
    psd = phot.std(0).clip(1e-6)
    wx = ridge(x / xsd, np.c_[tgt, phot / psd], 0.3)
    nuisance = ridge(phot / psd, tgt, 0.05)
    heads = np.column_stack((wx[:, 0], (wx[:, 1:] @ nuisance).ravel())) / xsd[:, None]
    qc = relative(physical(a, d, cal_ids)) @ heads
    yc = relative(np.stack([annotation(a, c)["labels"] for c in cal_ids])[:, :, None]).squeeze(-1)
    scales = [scalefit(qc[:, :, j], yc) for j in range(2)]
    frozen_npz(
        a.out / "model.npz",
        dict(heads=heads, scales=scales, design=LOCK, train_ids=fit_ids, calibration_ids=cal_ids),
    )
    print("Fit", len(fit_ids), "objects; calibrated", len(cal_ids), "objects", scales, flush=True)


def predict(a, d):
    ids = np.array(d["evaluation_classes"])
    with np.load(a.out / "model.npz", allow_pickle=False) as z:
        check(
            str(z["design"]) == LOCK
            and not np.isin(ids, np.r_[z["train_ids"], z["calibration_ids"]]).any(),
            "Overlapping evaluation objects",
        )
        heads, scales = z["heads"], z["scales"]
    inc = (relative(physical(a, d, ids)) @ heads) * scales
    predictions = []
    for n, i in enumerate(ids):
        values = annotation(a, i)["labels"]
        # Same object's two held photos share its legal eight photo support labels and EEG mean.
        p = np.array([values[s].mean() for s in SUPPORT])
        predictions.append(
            np.clip(
                np.stack(
                    (p, p + inc[n, :, 0], p + inc[n, np.arange(10) ^ 1, 0], p + inc[n, :, 1]),
                    axis=1,
                ),
                0.01,
                0.99,
            )
        )
    probs = np.stack(predictions)
    frozen_npz(
        a.out / "blind_predictions.npz",
        dict(
            rows=10 * ids[:, None] + np.arange(10),
            support_ordinals=SUPPORT,
            methods=METHODS,
            probabilities=probs,
            model_sha256=sha(a.out / "model.npz"),
            design=LOCK,
        ),
    )
    print("Sealed predictions for", probs.shape[:2], flush=True)


def policy(p):
    return np.where(
        (p[..., 1] >= 0.75) & (p[..., 1] - p[..., 0] >= 0.10),
        1,
        np.where((p[..., 1] <= 0.25) & (p[..., 1] - p[..., 0] <= -0.10), -1, 0),
    )


def examples(a, rows, truth, p, hashes):
    with (a.data / "official-object-readout/sub-03/atm_train_rows.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        tab = list(csv.DictReader(f))
    act = policy(p.ravel().reshape(-1, len(METHODS)))
    labels = truth.ravel()
    ids = rows.ravel()
    probs = p.reshape(-1, len(METHODS))
    order = np.argsort(ids)
    groups = {
        "clean_light": labels == 1,
        "varied": labels == 0,
        "unknown": labels == 0.5,
        "would_prompt_light": act == 1,
        "would_prompt_texture": act == -1,
    }
    selected = [(k, j) for k, mask in groups.items() for j in order[mask[order]][:2]]
    sheet = Image.new("RGB", (2 * 270, ((len(selected) + 1) // 2) * 195), "white")
    pencil = ImageDraw.Draw(sheet)
    listing = []
    with ZipFile(a.data / "photo-structure/training_images.zip") as z:
        for n, (group, j) in enumerate(selected):
            row = int(ids[j])
            r = tab[row]
            blob = z.read(f"training_images/{r['official_folder']}/{r['stimulus_stem']}.jpg")
            check(hashlib.sha256(blob).hexdigest() == hashes.ravel()[j], "Example JPEG mismatch")
            with Image.open(io.BytesIO(blob)) as im:
                pic = crop512(im.convert("RGB")).resize((145, 145), Image.Resampling.LANCZOS)
            x, y = n % 2 * 270, n // 2 * 195
            sheet.paste(pic, (x, y))
            pencil.text((x, y + 146), f"{group}: {r['unique_id']} #{row}", fill="black")
            pencil.text(
                (x, y + 162),
                f"measured {labels[j]:.1f}  prior {probs[j, 0]:.2f}  EEG {probs[j, 1]:.2f}",
                fill="black",
            )
            listing.append(
                dict(
                    group=group,
                    row=row,
                    word=r["unique_id"],
                    truth=float(labels[j]),
                    prior=float(probs[j, 0]),
                    eeg=float(probs[j, 1]),
                )
            )
    sheet.save(a.out / "private_examples.jpg", quality=90)
    return listing


def score(a, d):
    ids = np.array(d["evaluation_classes"])
    source = a.out / "blind_predictions.npz"
    with np.load(source, allow_pickle=False) as z:
        rows, p = z["rows"], z["probabilities"]
        check(
            str(z["design"]) == LOCK
            and str(z["model_sha256"]) == sha(a.out / "model.npz")
            and np.array_equal(rows, 10 * ids[:, None] + np.arange(10)),
            "Prediction receipt mismatch",
        )
    rec = [annotation(a, c) for c in ids]
    y = np.stack([r["labels"] for r in rec])
    known = y != 0.5
    draws = np.random.default_rng(260701).integers(len(ids), size=(10000, len(ids)))

    def estimate(value, mask):
        sums = np.where(mask, value, 0).sum(1)
        ns = mask.sum(1)
        den = ns[draws].sum(1)
        good = den > 0
        if not mask.any():
            return dict(n=0, mean=None, ci95=None)
        estimates = sums[draws].sum(1)[good] / den[good]
        return dict(
            n=int(ns.sum()),
            objects=int(np.count_nonzero(ns)),
            mean=float(sums.sum() / ns.sum()),
            ci95=np.quantile(estimates, [0.025, 0.975]).tolist(),
        )

    err = (p - y[:, :, None]) ** 2
    brier = {k: estimate(err[:, :, j], known) for j, k in enumerate(METHODS)}
    benefit = {
        k: estimate(err[:, :, j] - err[:, :, 1], known) for j, k in enumerate(METHODS) if j != 1
    }
    probdiff = p[:, ::2] - p[:, 1::2]
    yd = y[:, ::2] - y[:, 1::2]
    opposite = np.abs(yd) == 1
    match = yd[:, :, None] * probdiff
    pairs = {
        k: estimate((match[:, :, j] > 0) + 0.5 * (match[:, :, j] == 0), opposite)
        for j, k in enumerate(METHODS)
    }
    acc = {}
    for j, k in enumerate(METHODS):
        margin = (2 * y - 1) * (p[:, :, j] - 0.5)
        hit = (margin > 0) + 0.5 * (margin == 0)
        acc[k] = dict(
            confident=estimate(hit, known),
            light=estimate(hit, y == 1),
            textured=estimate(hit, y == 0),
        )
        acc[k]["balanced_accuracy"] = float(
            (acc[k]["light"]["mean"] + acc[k]["textured"]["mean"]) / 2
        )
    act = policy(p)
    select = act != 0
    chosen = {}
    for key, mask in (("all", select), ("light", act == 1), ("texture", act == -1)):
        chosen[key] = dict(
            attempts=int(mask.sum()),
            uncertain=int((mask & ~known).sum()),
            precision=estimate(np.where(act == 1, y == 1, y == 0), mask & known),
        )
    ep = pairs["eeg"]
    pol = chosen["all"]["precision"]
    passed = (
        all(v["ci95"][0] > 0 for v in benefit.values())
        and ep["n"] >= 20
        and ep["ci95"][0] > 0.5
        and pol["n"] >= 20
        and pol["mean"] >= 0.70
    )
    fractions = {}
    for tag in ("fit_classes", "calibration_classes", "evaluation_classes"):
        arrays = rec if tag == "evaluation_classes" else [annotation(a, c) for c in d[tag]]
        ya = np.stack([v["labels"] for v in arrays])
        invalid = np.stack([v["background"][:, 0] < 0.15 for v in arrays])
        fractions[tag] = dict(
            n=int(ya.size),
            clean_light=int((ya == 1).sum()),
            textured=int((ya == 0).sum()),
            uncertain=int((ya == 0.5).sum()),
            insufficient_background=int(invalid.sum()),
        )
    examples_used = examples(a, rows, y, p, np.stack([v["jpeg_sha256"] for v in rec]))
    result = dict(
        subject="sub-04",
        n=960,
        objects=96,
        registered_design_sha256=LOCK,
        annotation_driver_sha256=sha(Path(__file__).with_name("measure_background_attribute.py")),
        fit_driver_sha256=sha(Path(__file__)),
        model_sha256=sha(a.out / "model.npz"),
        predictions_sha256=sha(source),
        counts=fractions,
        brier=brier,
        paired_brier_gain=benefit,
        accuracy=acc,
        opposite_pairs=pairs,
        all_coded_mse={
            k: estimate(err[:, :, j], np.ones_like(known)) for j, k in enumerate(METHODS)
        },
        actions=chosen,
        gate_passed=bool(passed),
        examples=examples_used,
    )
    freeze(a.out / "summary.json", result)
    if a.release:
        freeze(a.release, result)
    print("brier gain", benefit, "pairs", ep, "actions", chosen, "passed", passed, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["fit", "predict", "score"])
    ap.add_argument("--data", type=Path, default=Path("outputs"))
    ap.add_argument("--out", type=Path, default=Path("outputs/eeg-background-attribute"))
    ap.add_argument("--release", type=Path)
    args = ap.parse_args()
    check(sha(DESIGN) == LOCK, "Changed background protocol")
    {"fit": fit, "predict": predict, "score": score}[args.stage](args, read(DESIGN))
