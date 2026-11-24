"""Estimate a generated photograph's horizontal orientation from posterior EEG asymmetry."""

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from decode_eeg_layout import eeg_design
from fit_spatial_photo import moments, original, other_eight, small, table, waves
from PIL import Image, ImageOps
from refine_registered_color import metrics, thumbnail
from render_within_photo_ip import check, packet, read, save, sha
from scipy.linalg import solve
from score_reconstruction_jpegs import registered
from score_within_photo_dino import ZIP_SHA

SUBS = ("sub-04", "sub-05", "sub-01")
ARMS = ("neutral", "prior", "own", "wrong")
# P7/P8, P5/P6, P3/P4, P1/P2, PO7/PO8, PO3/PO4, and O1/O2 in the archived order.
LEFT, RIGHT = [0, 1, 2, 3, 9, 10, 14], [8, 7, 6, 5, 13, 12, 16]


def asymmetry(x):
    bins = x[..., 425:833].reshape(*x.shape[:-1], 17, 24)[..., :12]
    return (bins[..., LEFT, :] - bins[..., RIGHT, :]).reshape(*x.shape[:-1], 84)


def odd(image):
    coarse = moments(small(image)).reshape(4, 2, 4, 2, 4).mean((1, 3))
    return (coarse[:, :2] - coarse[:, 3:1:-1]).reshape(-1).astype(np.float64)


def centered(x):
    return (x - x.mean(1, keepdims=True)).reshape(-1, x.shape[-1])


def fit(x, y):
    dx, dy = centered(x), centered(y)
    sx = dx.std(0).clip(1e-8)
    dx /= sx
    sy = dy.std(0).clip(1e-8)
    gram = dx.T @ dx
    gram.flat[:: len(sx) + 1] += len(dx)  # Fixed ridge of one after standardization.
    return solve(gram, dx.T @ dy, assume_a="pos") / sx[:, None], sy


def train(a):
    check(sha(a.images) == ZIP_SHA, "Calibration image archive changed")
    tab = table(a)
    output = {}
    classes = np.asarray(read(a.expanded / "features/sub-04.json")["training_concepts"])
    check(len(set(classes)) == 512, "Expected 512 distinct generic concepts")
    with ZipFile(a.images) as archive:
        targets = np.stack(
            [odd(original(archive, tab, r)) for c in classes for r in 10 * c + np.arange(10)]
        ).reshape(512, 10, 32)
    for sub in SUBS:
        folder = a.expanded / "features"
        meta = read(folder / (sub + ".json"))
        check(
            meta["subject"] == sub and np.array_equal(classes, meta["training_concepts"]),
            "Calibration subjects differ",
        )
        h = hashlib.sha256()
        blocks = []
        for c in classes:
            with np.load(folder / sub / f"{c:04d}.npz", allow_pickle=False) as z:
                x = z["features"]
                check(
                    x.shape == (10, 1003)
                    and int(z["start"]) == 10 * c
                    and str(z["raw_sha"]) == meta["raw_block_sha256"][str(c)],
                    "Generic EEG source mismatch",
                )
                h.update(x.tobytes())
                blocks.append(asymmetry(x))
        check(h.hexdigest() == meta["train_design_sha256"], "Generic EEG buffer changed")
        x = np.stack(blocks).astype(float)
        prior = other_eight(targets)
        delta = x - other_eight(x)
        outcomes = {"prior": [], "own": [], "wrong": []}
        for ev in np.array_split(np.random.default_rng(260511).permutation(len(x)), 4):
            tr = np.setdiff1d(np.arange(len(x)), ev)
            w, sy = fit(x[tr], targets[tr])
            resid = delta[ev] @ w
            for arm, p in [
                ("prior", prior[ev]),
                ("own", prior[ev] + resid),
                ("wrong", prior[ev] + resid[:, np.arange(10) ^ 1]),
            ]:
                similarity = (targets[ev] * p / sy**2).sum(-1)
                outcomes[arm].extend(((similarity > 0) + 0.5 * (similarity == 0)).mean(1))
        w, sy = fit(x, targets)
        a.out.mkdir(parents=True, exist_ok=True)
        path = a.out / (sub + ".npz")
        np.savez(path, classes=classes, w=w, sy=sy, source=h.hexdigest())
        output[sub] = dict(
            calibration_features_sha256=h.hexdigest(),
            model_sha256=sha(path),
            orientation_oof=stats(outcomes),
        )
    save(
        a.out / "training.json",
        dict(
            descriptor="32 left-minus-right cells of luminance, blue/red chroma and log contrast; seven posterior homologous differences, 40-280 ms; ridge=1. Four whole-class folds, eight other support photos. Diagnostic chooses original vs mirror of a real calibration photo.",
            generic_classes=512,
            images_per_person=5120,
            image_zip_sha256=ZIP_SHA,
            code_sha256=sha(Path(__file__)),
            subjects=output,
        ),
    )


def stats(arms, labels=None):
    own = np.asarray(arms["own"], float)
    if labels is None:
        labels = np.arange(len(own))
    _, ids, n = np.unique(labels, return_inverse=True, return_counts=True)
    draw = np.random.default_rng(260509).integers(len(n), size=(10000, len(n)))

    def stat(x):
        sums = np.bincount(ids, weights=x)
        return dict(
            mean=float(np.mean(x)),
            ci95=np.quantile(sums[draw].sum(1) / n[draw].sum(1), [0.025, 0.975]).tolist(),
        )

    return dict(
        n=len(own),
        arms={k: stat(x) for k, x in arms.items()},
        own_minus={k: stat(own - x) for k, x in arms.items() if k != "own"},
    )


def predict(a, sub):
    path = a.out / (sub + ".npz")
    receipt = read(a.out / "training.json")
    check(sha(path) == receipt["subjects"][sub]["model_sha256"], "Frozen orientation model changed")
    rows, folds, words, donors, supports, _, _, cond = packet(a.input, sub, a.panel)
    source = a.input / "renders" / a.panel / sub
    cfg = read(source / "settings.json")
    eeg = a.eeg / (sub + ".npz")
    with (
        np.load(a.input / "conditions" / sub / a.panel / "conditions_512.npz") as z,
        np.load(path) as m,
    ):
        check(
            not np.intersect1d(m["classes"], rows // 10).size
            and str(z["target_eeg_sha256"]) == sha(eeg)
            and cond == cfg["packet_sha256"],
            "Input provenance changed",
        )
        w, sy = m["w"], m["sy"]
    ids, wave = waves(eeg, sub)
    features = asymmetry(eeg_design(wave, np.arange(len(ids)), np.arange(17))).astype(float)
    ix = {r: i for i, r in enumerate(ids)}
    pos = {r: i for i, r in enumerate(rows)}
    residual = (
        features[[ix[r] for r in rows]] - features[[[ix[r] for r in ss] for ss in supports]].mean(1)
    ) @ w
    choices = {k: [] for k in ARMS}
    tab = table(a)
    cache = {}
    with ZipFile(a.images) as archive:
        for i, (r, ss) in enumerate(zip(rows, supports, strict=True)):
            check(
                all(
                    s != r and folds[pos[s]] != folds[i] and tab[s]["unique_id"] == words[i]
                    for s in ss
                ),
                "Support row mismatch",
            )
            for s in ss:
                if s not in cache:
                    cache[s] = odd(original(archive, tab, s))
            prior = np.stack([cache[s] for s in ss]).mean(0)
            src = source / "images" / "own" / f"{i:03d}.jpg"
            check(sha(src) == cfg["jpeg_sha256"]["own"][i], "Generated source changed")
            with Image.open(src) as im:
                base = odd(im.convert("RGB")) / sy**2
            choices["neutral"].append(False)
            for k, v in [
                ("prior", prior),
                ("own", prior + residual[i]),
                ("wrong", prior + residual[pos[donors[i]]]),
            ]:
                choices[k].append(bool(base @ v < 0))
    save(
        a.out / a.panel / sub / "choices.json",
        dict(
            rows=rows.tolist(),
            flip=choices,
            base_settings_sha256=sha(source / "settings.json"),
            condition_sha256=cond,
            model_sha256=sha(path),
            eeg_sha256=sha(eeg),
        ),
    )


def score(a):
    check(sha(a.images) == ZIP_SHA, "Original image archive changed")
    reg = registered(a.references, a.panel)
    tab = table(a)
    with ZipFile(a.images) as archive:
        refs = [original(archive, tab, r) for r in reg["rows"]]
    report = {}
    for sub in SUBS:
        directory = a.out / a.panel / sub
        source = a.input / "renders" / a.panel / sub
        decisions = read(directory / "choices.json")
        cfg = read(source / "settings.json")
        check(
            np.array_equal(reg["rows"], decisions["rows"])
            and decisions["base_settings_sha256"] == sha(source / "settings.json"),
            "Decision order or generated outputs changed",
        )
        with np.load(a.out / (sub + ".npz")) as m:
            sy = m["sy"]
        values = {
            k: {arm: [] for arm in ARMS} for k in ("orientation", "rgb_rmse32", "luma_pearson32")
        }
        for i, ref in enumerate(refs):
            src = source / "images" / "own" / f"{i:03d}.jpg"
            check(sha(src) == cfg["jpeg_sha256"]["own"][i], "Base JPEG changed")
            with Image.open(src) as im:
                base = im.convert("RGB")
            mirrored = ImageOps.mirror(base)
            margin = (odd(ref) * odd(base) / sy**2).sum()
            thumb = thumbnail(ref)
            raw = [metrics(thumb, thumbnail(v)) for v in (base, mirrored)]
            for arm in ARMS:
                flip = decisions["flip"][arm][i]
                signed = -margin if flip else margin
                values["orientation"][arm].append(float(signed > 0) + 0.5 * (signed == 0))
                for k, index in [("rgb_rmse32", 1), ("luma_pearson32", 2)]:
                    values[k][arm].append(raw[flip][index])
        correct = reg["labels"] == np.array(cfg["predicted_words"])
        report[sub] = dict(
            decision_sha256=sha(directory / "choices.json"),
            flips={k: sum(v) for k, v in decisions["flip"].items()},
            groups={
                g: {
                    k: stats(
                        {arm: np.asarray(v)[mask] for arm, v in draws.items()}, reg["truth"][mask]
                    )
                    for k, draws in values.items()
                }
                for g, mask in [("all", np.ones(len(correct), bool)), ("correct_word", correct)]
            },
        )
    save(
        a.report,
        dict(
            panel=a.panel,
            registration_sha256=reg["sha"],
            training=read(a.out / "training.json"),
            metric="Orientation gives credit when the frozen choice (original vs horizontal reflection of the same generated image) is closer to the viewed photo in the standardized odd descriptor. Reflecting a photo preserves its noun and can reverse printed text. Object-cluster paired bootstrap 95% intervals; RGB RMSE lower is better.",
            subjects=report,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("train", "predict", "score"))
    for key, default in dict(
        out="outputs/photo-orientation",
        expanded="outputs/photo-structure/expanded",
        images="outputs/photo-structure/training_images.zip",
        manifest="outputs/official-object-readout/sub-03/atm_train_rows.csv",
        input="outputs/fresh-photo",
        eeg="outputs/fresh-photo/eeg",
        references="outputs/fresh-photo/references",
        report="outputs/photo-orientation/fresh.json",
    ).items():
        p.add_argument("--" + key, type=Path, default=Path(default))
    p.add_argument("--panel", default="fresh")
    p.add_argument("--subjects", nargs="+", choices=SUBS, default=SUBS)
    a = p.parse_args()
    if a.stage == "train":
        train(a)
    elif a.stage == "predict":
        check(sha(a.images) == ZIP_SHA, "Support image archive changed")
        for subject in a.subjects:
            predict(a, subject)
    else:
        score(a)
