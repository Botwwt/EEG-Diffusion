"""Fit and render support-centered EEG predictions of local color and luminance contrast."""

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from decode_eeg_layout import CHANNELS, eeg_design
from PIL import Image
from refine_personal_photo_residual import adjusted, residual_model
from render_official_words import crop512
from render_within_photo_ip import check, read, sha
from scipy.fft import dctn, idctn
from scipy.linalg import cho_factor, cho_solve
from score_within_photo_dino import ROWS_SHA, ZIP_SHA

LUMA = np.array([0.299, 0.587, 0.114], np.float32)
ARMS = ("baseline", "neutral", "prior", "own", "wrong", "old_prior", "old_own")
PLAN_SHA = "402cc915e9facee0fb6de395458e8a77961258542b1e0d1448bcda5ea24797f6"


def save(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(".partial")
    t.write_bytes((json.dumps(data, indent=2, allow_nan=False) + "\n").encode())
    t.replace(p)


def table(a):
    check(sha(a.manifest) == ROWS_SHA, "Wrong THINGS image row manifest")
    with a.manifest.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def original(archive, tab, r):
    v = tab[r]
    check(int(v["train_row"]) == r, "Original stimulus row changed")
    with Image.open(
        io.BytesIO(archive.read(f"training_images/{v['official_folder']}/{v['stimulus_stem']}.jpg"))
    ) as im:
        return crop512(im.convert("RGB"))


def small(im, n=64):
    return np.asarray(im.resize((n, n), Image.Resampling.BOX), dtype=np.float32) / 255


def moments(rgb):
    luminance = rgb @ LUMA
    cells = (
        np.stack(
            (luminance, rgb[..., 2] - luminance, rgb[..., 0] - luminance, luminance**2), axis=-1
        )
        .reshape(8, 8, 8, 8, 4)
        .mean((1, 3))
    )
    cells[..., 3] = np.log(0.025 + np.sqrt(np.maximum(0, cells[..., 3] - cells[..., 0] ** 2)))
    return cells


def spatial(im):
    m = moments(small(im))
    b = np.asarray(im).reshape(32, 16, 32, 16, 3).mean((1, 3)).astype(np.float32) / 255
    return dctn(m, axes=(0, 1), norm="ortho"), b.reshape(4, 8, 4, 8, 3).mean((1, 3))


def cache(a):
    p = a.out / "generic_photos.npz"
    marker = p.with_suffix(".json")
    if p.is_file() and marker.is_file():
        record = read(marker)
        check(
            record
            == dict(
                cache_sha256=sha(p),
                image_zip_sha256=ZIP_SHA,
                manifest_sha256=ROWS_SHA,
                design_sha256=PLAN_SHA,
            ),
            "Generic spatial image cache or source changed",
        )
        with np.load(p, allow_pickle=False) as q:
            return q["classes"], q["dct"], q["old"]
    check(sha(a.images) == ZIP_SHA, "Wrong training image archive")
    meta = read(a.expanded / "features/sub-04.json")
    classes = np.asarray(meta["training_concepts"])
    check(len(set(classes)) == 512, "Wrong non-panel training classes")
    tab = table(a)
    coords = []
    coarse = []
    with ZipFile(a.images) as z:
        for c in classes:
            for r in c * 10 + np.arange(10):
                d, rgb = spatial(original(z, tab, r))
                coords.append(d)
                coarse.append(rgb)
    y = np.stack(coords).reshape(512, 10, 8, 8, 4)
    old = np.stack(coarse).reshape(512, 10, 4, 4, 3)
    partial = p.with_suffix(".partial.npz")
    np.savez(partial, classes=classes, dct=y, old=old)
    partial.replace(p)
    save(
        marker,
        dict(
            cache_sha256=sha(p),
            image_zip_sha256=ZIP_SHA,
            manifest_sha256=ROWS_SHA,
            design_sha256=PLAN_SHA,
        ),
    )
    return classes, y, old


def within(x):
    return (x - x.mean(axis=1, keepdims=True)).reshape(-1, x.shape[-1])


def ridge(x, y):
    x = within(x)
    y = within(y)
    sd = x.std(0).clip(1e-8)
    x = x / sd
    gram = x.T @ x
    gram.flat[:: len(sd) + 1] += 0.3 * len(x)
    return (
        cho_solve(cho_factor(gram, check_finite=False), x.T @ y, check_finite=False) / sd[:, None]
    )


def other_eight(x):
    paired = x.reshape(len(x), 5, 2, x.shape[-1])
    return np.repeat((x.sum(1)[:, None] - paired.sum(2)) / 8, 2, axis=1)


def old_eeg(wave):
    return (
        wave[:, :, 15:115]
        .reshape(len(wave), 17, 20, 5)
        .mean(-1, dtype=np.float64)
        .reshape(len(wave), -1)
    )


def fit(a):
    classes, target, old = cache(a)
    xroot = a.expanded / "features"
    meta = read(xroot / (a.subject + ".json"))
    check(
        np.array_equal(classes, meta["training_concepts"]), "Participants' generic classes differ"
    )
    blocks = []
    for c in classes:
        with np.load(xroot / a.subject / f"{c:04d}.npz", allow_pickle=False) as q:
            v = q["features"]
            check(
                int(q["start"]) == 10 * c
                and hashlib.sha256(v.tobytes()).hexdigest() == str(q["feature_sha"])
                and str(q["raw_sha"]) == meta["raw_block_sha256"][str(c)],
                "Wrong EEG training block",
            )
            blocks.append(v)
    x = np.stack(blocks)
    check(
        hashlib.sha256(x.tobytes()).hexdigest() == meta["train_design_sha256"],
        "Wrong training design",
    )
    x = x.astype(np.float64)
    y = target.reshape(512, 10, -1).astype(float)
    oof = np.empty_like(y)
    dx = x - other_eight(x)
    ymean = other_eight(y)
    dy = y - ymean
    for ev in np.array_split(np.random.default_rng(260519).permutation(512), 4):
        tr = np.setdiff1d(np.arange(512), ev)
        oof[ev] = dx[ev] @ ridge(x[tr], y[tr])
    b = np.digitize(np.indices((8, 8)).sum(0), [1, 3, 7]).ravel()
    yp = oof.reshape(-1, 64, 4)
    gt = dy.reshape(-1, 64, 4)
    gains = np.zeros((4, 4))
    for k in range(4):
        pv = yp[:, b == k]
        actual = gt[:, b == k]
        gains[k] = ((pv * actual).sum((0, 1)) / (pv * pv).sum((0, 1)).clip(1e-15)).clip(0, 1.5)
    g = gains[b]
    pred = oof.reshape(512, 10, 64, 4) * g
    # Each generic query takes as surrogate image the first photo in the next pair.
    # That photo is one of the eight supports and always differs from both held-out photos.
    donor = y[:, ((np.arange(10) // 2 + 1) % 5) * 2].reshape(512, 10, 64, 4)
    ym = ymean.reshape(512, 10, 64, 4)
    actual = y.reshape(512, 10, 64, 4)
    step = ym + pred - donor
    alpha = (((actual - donor) * step).sum((0, 1, 2)) / (step * step).sum((0, 1, 2))).clip(0, 0.5)
    hist = a.inputs / "generic" / (a.subject + ".npz")
    with np.load(hist, allow_pickle=False) as z:
        oldclasses = z["rows"][::10] // 10
        check(
            list(z["channels"]) == CHANNELS
            and np.array_equal(oldclasses, meta["original_concepts"]),
            "Wrong nested legacy training",
        )
        hx = old_eeg(z["mean_eeg"]).reshape(128, 10, -1)
    lut = {c: i for i, c in enumerate(classes)}
    sd, hw, hgain, legacy = residual_model(
        hx, old[[lut[c] for c in oldclasses]].astype(float), oldclasses
    )
    path = a.out / "models" / (a.subject + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        classes=classes,
        w=ridge(x, y),
        gain=g.reshape(8, 8, 4),
        alpha=alpha,
        old_classes=oldclasses,
        old_w=hw / sd[:, None] * hgain,
        design_sha256=PLAN_SHA,
    )
    save(
        path.with_suffix(".json"),
        dict(
            subject=a.subject,
            sha256=sha(path),
            design=PLAN_SHA,
            generic_classes=512,
            generic_photo_queries=5120,
            band_gain=gains.tolist(),
            alpha=alpha.tolist(),
            oof_squared_error_prior=np.mean(gt**2, axis=(0, 1)).tolist(),
            oof_squared_error_own=np.mean((gt - yp * g) ** 2, axis=(0, 1)).tolist(),
            legacy128=legacy,
            legacy_gain=hgain,
            training_sha=meta["train_design_sha256"],
        ),
    )
    print("Fitted", a.subject, alpha, gains, flush=True)


def up(v, n=512):
    return np.asarray(
        Image.fromarray(v.astype(np.float32)).resize((n, n), Image.Resampling.BILINEAR)
    )


def transfer(rgb, observed, layout, alpha):
    step = (layout - observed) * alpha
    dl, db, dr, logs = (up(v) for v in step.transpose(2, 0, 1))
    lo = up(observed[..., 0])
    lum = rgb @ LUMA
    dl = dl + (np.exp(logs).clip(0.75, 1.33) - 1) * (lum - lo)
    change = np.stack((dl + dr, dl - (LUMA[0] * dr + LUMA[2] * db) / LUMA[1], dl + db), -1).clip(
        -0.2, 0.2
    )
    return Image.fromarray(np.rint((rgb + change).clip(0, 1) * 255).astype(np.uint8))


def waves(path, subject):
    with np.load(path, allow_pickle=False) as q:
        check(
            str(q["subject"]) == subject
            and np.allclose(q["times"], np.arange(250) / 250, atol=1e-7),
            "Wrong EEG time basis",
        )
        ids = [q["channels"].tolist().index(c) for c in CHANNELS]
        wave = q["mean_eeg"] if "mean_eeg" in q else q["half_eeg"].mean(0)
        return q["rows"], wave[:, ids, :]


def render(a):
    check(sha(a.images) == ZIP_SHA, "Unexpected calibration photograph archive")
    model = a.out / "models" / (a.subject + ".npz")
    meta = read(model.with_suffix(".json"))
    check(sha(model) == meta["sha256"], "Model changed")
    with np.load(model, allow_pickle=False) as q:
        W, gain, alpha, hW, classes = (q[k] for k in ("w", "gain", "alpha", "old_w", "classes"))
        check(str(q["design_sha256"]) == PLAN_SHA, "Wrong experimental design")
    base = a.base or Path("outputs/photo-structure/residual_angle/renders") / a.panel / a.subject
    packet = (
        a.packet
        or Path("outputs/photo-structure/residual_angle/conditions")
        / a.subject
        / a.panel
        / "conditions_512.npz"
    )
    source = read(base / "settings.json")
    if a.confirmation is not None:
        lock = read(a.confirmation)
        check(
            lock["method_sha256"] == PLAN_SHA
            and lock["generic_model_sha256"][a.subject] == sha(model)
            and lock["new_condition_sha256"][a.subject] == sha(packet)
            and lock["angle_design_sha256"] == source["design_sha256"],
            "Inputs disagree with the fresh-panel confirmation lock",
        )
    check(source["panel"] == a.panel, "Angle input is from another panel")
    target = a.eeg or a.inputs / "halves" / (a.subject + ".npz")
    ids, wave = waves(target, a.subject)
    with np.load(packet, allow_pickle=False) as q:
        rows, folds, words, support, donor = (
            q[k] for k in ("rows", "folds", "predicted_words", "support_rows", "donor_rows")
        )
        check(
            all(
                np.array_equal(q[k], source[k])
                for k in ("rows", "folds", "predicted_words", "support_rows", "donor_rows")
            )
            and sha(packet) == source["packet_sha256"]
            and str(q["target_eeg_sha256"]) == sha(target),
            "Mismatched angle generator, EEG or source rows",
        )
    check(
        source["subject"] == a.subject
        and len(set(rows)) == len(rows)
        and len(set(ids)) == len(ids)
        and set(rows) <= set(ids)
        and set(rows) == set(donor)
        and not np.intersect1d(rows // 10, classes).size,
        "Generic and panel classes overlap or wrong IDs",
    )
    pos = {r: i for i, r in enumerate(ids)}
    ri = [pos[r] for r in rows]
    si = np.asarray([[pos[r] for r in s] for s in support])
    rowpos = {r: i for i, r in enumerate(rows)}
    dj = np.asarray([rowpos[r] for r in donor])
    check(
        np.array_equal(words, words[dj])
        and np.array_equal(folds, folds[dj])
        and np.array_equal(support, support[dj]),
        "Wrong donor",
    )
    for i, s in enumerate(support):
        check(
            len(set(s)) == 8 and set(s) <= set(rows[folds != folds[i]]),
            "Query photo leaked into its support",
        )
    eeg = eeg_design(wave, np.arange(len(ids)), np.arange(17)).astype(float)
    oldx = old_eeg(wave)
    delta = ((eeg[ri] - eeg[si].mean(1)) @ W).reshape(-1, 8, 8, 4) * gain
    old_delta = ((oldx[ri] - oldx[si].mean(1)) @ hW).reshape(-1, 4, 4, 3)
    dst = a.out / "renders" / a.panel / a.subject
    dst.mkdir(parents=True, exist_ok=True)
    tab = table(a)
    feat = {}
    hs = {arm: [] for arm in ARMS}
    for arm in ARMS:
        (dst / "images" / arm).mkdir(parents=True, exist_ok=True)
    with ZipFile(a.images) as ar:
        for i, s in enumerate(support):
            for r in s:
                check(tab[r]["unique_id"] == words[i], "Wrong predicted-class calibration photos")
                if r not in feat:
                    feat[r] = spatial(original(ar, tab, r))
            dc = np.stack([feat[r][0] for r in s]).mean(0)
            old = np.stack([feat[r][1] for r in s]).mean(0)
            origin = base / "images" / "own" / f"{i:03d}.jpg"
            check(sha(origin) == source["jpeg_sha256"]["own"][i], "Angle source JPEG mismatch")
            with Image.open(origin) as pic:
                im = pic.convert("RGB")
            rgb = np.asarray(im, np.float32) / 255
            observed = moments(small(im))
            for arm in ARMS:
                out = dst / "images" / arm / f"{i:03d}.jpg"
                if arm == "baseline":
                    if not out.exists():
                        try:
                            os.link(origin, out)
                        except OSError:
                            shutil.copyfile(origin, out)
                    check(
                        sha(out) == source["jpeg_sha256"]["own"][i], "Baseline differs from source"
                    )
                else:
                    if arm == "neutral":
                        edit = im
                    elif arm in ("old_prior", "old_own"):
                        edit = adjusted(im, old + (old_delta[i] if arm == "old_own" else 0))
                    else:
                        layout = dc + (
                            delta[i] if arm == "own" else delta[dj[i]] if arm == "wrong" else 0
                        )
                        edit = transfer(
                            rgb, observed, idctn(layout, axes=(0, 1), norm="ortho"), alpha
                        )
                    tmp = out.with_suffix(".partial")
                    edit.save(tmp, "JPEG", quality=95, subsampling=0)
                    tmp.replace(out)
                hs[arm].append(sha(out))
    save(
        dst / "settings.json",
        dict(
            subject=a.subject,
            panel=a.panel,
            rows=rows.tolist(),
            folds=folds.tolist(),
            predicted_words=words.tolist(),
            donor_rows=donor.tolist(),
            support_rows=support.tolist(),
            jpeg_sha256=hs,
            subject_model_sha256=sha(model),
            design_sha256=PLAN_SHA,
            source_settings_sha256=sha(base / "settings.json"),
            packet_sha256=sha(packet),
            eeg_sha256=sha(target),
            original_archive_sha256=ZIP_SHA,
            confirmation_sha256=sha(a.confirmation) if a.confirmation else None,
        ),
    )
    print("Rendered", a.subject, a.panel, len(rows) * len(ARMS), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("cache", "fit", "render"))
    for k, v in dict(
        out="outputs/spatial-photo",
        images="outputs/photo-structure/training_images.zip",
        manifest="outputs/official-object-readout/sub-03/atm_train_rows.csv",
        inputs="outputs/object-transfer",
        expanded="outputs/photo-structure/expanded",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    for k in ("packet", "eeg", "base", "confirmation"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--subject", default="sub-04")
    p.add_argument("--panel", default="registered")
    a = p.parse_args()
    check(sha(a.out / "design.json") == PLAN_SHA, "Missing fixed method design")
    {"cache": cache, "fit": fit, "render": render}[a.stage](a)
