"""Learn luminance-conditioned chroma quantiles on generic EEG and score the saved panel JPEGs."""

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from fit_spatial_photo import LUMA, original, ridge, small, within
from fit_support_covariance import eeg_features
from PIL import Image, ImageDraw, ImageFont
from register_fresh_photo_panel import freeze, read
from render_within_photo_ip import check, packet, sha
from score_object_panel import score_object_panel
from score_official_objects import unit
from score_within_photo_dino import FILES, ZIP_SHA, encoder, reference
from test_cohort_session_matched import NAMES, PEOPLE, compact

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
PLAN = PUBLIC / "eeg_color_transfer_design.json"
ARMS = ("neutral", "prior", "own", "wrong")
MODES = ("sub-04", "cohort-3")
SEED = 260541


def components(rgb):
    y = rgb @ LUMA
    ci = np.stack((rgb[..., 2] - y, rgb[..., 0] - y), axis=-1)
    yi = np.rint(y * 255).clip(0, 255).astype(np.uint8)
    hist = np.bincount(yi.ravel(), minlength=256)
    ranks = (np.cumsum(hist) - 0.5 * hist) / yi.size
    return y, ci, ranks[yi]


def palette(rgb):
    _, c, r = components(rgb)
    quartile = np.minimum((4 * r).astype(int), 3)
    global_q = np.quantile(c.reshape(-1, 2), [0.15, 0.5, 0.85], axis=0).T
    out = []
    for b in range(4):
        values = c[quartile == b]
        out.append(np.quantile(values, [0.15, 0.5, 0.85], axis=0).T if len(values) else global_q)
    return np.asarray(out, dtype=np.float64)


def fit(a):
    split = read(PUBLIC / "shared_photo_eeg_design.json")
    train, test = np.asarray(split["train_classes"]), np.asarray(split["test_classes"])
    ids = np.sort(np.r_[train, test])
    check(len(set(ids)) == 512 and len(train) == 416 and len(test) == 96, "Generic split changed")
    table = reference(a)
    with ZipFile(a.originals) as z:
        y = np.array(
            [[palette(small(original(z, table, int(10 * c + j)))) for j in range(10)] for c in ids]
        ).reshape(512, 10, 24)
    train, test = np.searchsorted(ids, train), np.searchsorted(ids, test)
    info = {}
    for sub in PEOPLE:
        classes, x = eeg_features(a.expanded, sub)
        check(np.array_equal(ids, classes), "Historical training order changed")
        pred = within(x[test]) @ ridge(x[train], y[train])
        true = within(y[test])
        shrink = float(np.clip(np.sum(pred * true) / max(np.sum(pred**2), 1e-16), 0, 1))
        w = ridge(x, y)
        dest = a.out / "fits" / (sub + ".npz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dest, weights=w, shrink=shrink, classes=classes, design_sha256=sha(PLAN))
        info[sub] = dict(
            shrink=shrink,
            generic_objects=512,
            heldout_cosine=float(
                np.sum(pred * true) / (np.linalg.norm(pred) * np.linalg.norm(true))
            ),
            packet_sha256=sha(dest),
            input_eeg_sha256=sha(a.expanded / "features" / (sub + ".json")),
        )
    freeze(
        a.out / "fitted.json",
        dict(
            design_sha256=sha(PLAN),
            split_sha256=sha(PUBLIC / "shared_photo_eeg_design.json"),
            subjects=info,
            generic_targets_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
            original_zip_sha256=ZIP_SHA,
        ),
    )


def get_cases(a):
    design = read(a.panel_root / "design.json")
    frozen = read(PUBLIC / "cohort_session_render_inputs.json")
    original_root = a.panel_root / "render_inputs"
    check(
        sha(original_root / "design.json") == frozen["design_sha256"], "Wrong frozen render design"
    )
    for name in NAMES:
        panel = design["panels"][name]
        for mode in MODES:
            path = original_root / "renders" / name / mode
            raw = read(path / "settings.json")
            rows, folds, words, donor, support, _, _, p_sha = packet(original_root, mode, name)
            reg = panel["registration"]
            index = {w: i for i, w in enumerate(reg["words"])}
            legal = np.asarray(reg["calibration_rows_by_fold_and_word"])[
                folds, [index[w] for w in words]
            ]
            check(
                raw["packet_sha256"] == p_sha == frozen["packets_sha256"][mode + "/" + name]
                and raw["design_sha256"] == frozen["design_sha256"]
                and np.array_equal(rows, reg["rows"])
                and np.array_equal(folds, reg["folds"])
                and np.array_equal(support, legal)
                and np.array_equal(raw["support_rows"], support)
                and np.array_equal(raw["predicted_words"], words)
                and np.array_equal(raw["donor_rows"], donor)
                and np.array_equal(raw["seeds"], 923761 + rows),
                "Changed frozen words, render, original identity or allowed supports",
            )
            yield name, mode, panel, path, raw, rows, folds, words, donor, support


def infer(a):
    table = reference(a)
    meta = read(a.out / "fitted.json")
    check(meta["design_sha256"] == sha(PLAN), "Wrong color model lock")
    acq = SimpleNamespace(out=a.panel_root, double=a.double)
    for name, mode, panel, path, _, rows, folds, words, donor, support in get_cases(a):
        ids = np.asarray(panel["selected_concepts"])
        source = (ids[:, None] * 10 + np.arange(10)).ravel()
        order = np.argsort(source)
        rr = source[order]
        q = np.searchsorted(rr, rows)
        s = np.searchsorted(rr, support)
        check(
            np.array_equal(rr[q], rows) and np.array_equal(rr[s], support),
            "Unknown compact EEG identity",
        )
        deviations = []
        for sub in PEOPLE if mode == "cohort-3" else [mode]:
            x, receipt = compact(acq, sub, name, ids)
            w = a.out / "fits" / (sub + ".npz")
            check(
                sha(w) == meta["subjects"][sub]["packet_sha256"]
                and read(a.panel_root / name / "result.json")["target_feature_receipts"][sub]
                == receipt,
                "Color model or test EEG differs",
            )
            with np.load(w, allow_pickle=False) as model:
                check(
                    not np.intersect1d(ids, model["classes"]).size, "Panel occurs in color training"
                )
                pred = (x[order] @ model["weights"]) * float(model["shrink"])
            deviations.append(pred[q] - pred[s].mean(1))
        prior = np.empty((120, 4, 2, 3), float)
        with ZipFile(a.originals) as archive:
            for f in range(5):
                use = np.flatnonzero(folds == f)
                allowed = np.unique(support[use])
                check(
                    not np.intersect1d(allowed, rows[use]).size,
                    "Active query photo used for palette inference",
                )
                palettes = {
                    int(r): palette(small(original(archive, table, int(r)))) for r in allowed
                }
                prior[use] = np.array([[palettes[int(r)] for r in v] for v in support[use]]).mean(1)
        lookup = {r: i for i, r in enumerate(rows)}
        jj = np.array([lookup[r] for r in donor])
        check(
            np.array_equal(prior, prior[jj])
            and np.array_equal(words, words[jj])
            and np.array_equal(folds, folds[jj]),
            "Different prior for wrong control",
        )
        deviation = np.mean(deviations, axis=0).reshape(120, 4, 2, 3)
        dest = a.out / "conditions" / name / (mode + ".npz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            dest,
            rows=rows,
            folds=folds,
            words=words,
            donor_rows=donor,
            support_rows=support,
            prior=prior,
            own=prior + deviation,
            wrong=prior + deviation[jj],
            design_sha256=sha(PLAN),
            settings_sha256=sha(path / "settings.json"),
        )


def chroma_edit(rgb, y, c, r, current, target):
    # Interpolate smooth color corrections within and between luminance strata.
    grid = np.linspace(-1, 1, 1025)
    lut = np.empty((4, 2, len(grid)), np.float32)
    for band in range(4):
        for k in range(2):
            knots, inv = np.unique(current[band, k], return_inverse=True)
            offsets = np.sort(target[band, k]) - current[band, k]
            v = np.bincount(inv, weights=offsets) / np.bincount(inv)
            lut[band, k] = np.interp(grid, knots, v)
    light = np.clip(r * 4 - 0.5, 0, 3)
    lo = light.astype(np.intp)
    hi = np.minimum(lo + 1, 3)
    frac = light - lo
    changes = []
    for k in range(2):
        idx = np.rint((c[..., k] + 1) * 512).clip(0, 1024).astype(np.intp)
        changes.append(
            0.5 * np.clip((1 - frac) * lut[lo, k, idx] + frac * lut[hi, k, idx], -0.12, 0.12)
        )
    blue, red = changes
    delta = np.stack((red, -(LUMA[0] * red + LUMA[2] * blue) / LUMA[1], blue), axis=-1)
    bound = np.where(
        delta > 0,
        (1 - rgb) / np.maximum(delta, 1e-12),
        np.where(delta < 0, rgb / np.maximum(-delta, 1e-12), 1.0),
    )
    scale = np.minimum(1.0, bound.min(axis=-1, keepdims=True))
    corrected = (rgb + scale * delta).clip(0, 1)
    check(np.max(abs(corrected @ LUMA - y)) < 2e-6, "Chroma transform changed luminance")
    return Image.fromarray(np.rint(255 * corrected).astype(np.uint8))


def render(a):
    for name, mode, _, path, raw, rows, _, _, _, _ in get_cases(a):
        dest = a.out / "renders" / name / mode
        condition = a.out / "conditions" / name / (mode + ".npz")
        with np.load(condition, allow_pickle=False) as z:
            check(
                str(z["settings_sha256"]) == sha(path / "settings.json")
                and np.array_equal(rows, z["rows"]),
                "Changed source render",
            )
            target = {k: z[k] for k in ARMS[1:]}
        for arm in ARMS:
            (dest / "images" / arm).mkdir(parents=True, exist_ok=True)

        def draw(i, path=path, raw=raw, target=target, dest=dest):
            source = path / "images/own" / f"{i:03d}.jpg"
            check(sha(source) == raw["jpeg_sha256"]["own"][i], "Changed generated JPEG")
            with Image.open(source) as opened:
                orig = opened.convert("RGB")
            check(orig.size == (512, 512), "Wrong generated dimensions")
            rgb = np.asarray(orig).astype(np.float32) / 255
            y, c, r = components(rgb)
            current = palette(small(orig))
            keys = []
            for arm in ARMS:
                result = (
                    orig if arm == "neutral" else chroma_edit(rgb, y, c, r, current, target[arm][i])
                )
                filename = dest / "images" / arm / f"{i:03d}.jpg"
                result.save(filename, "JPEG", quality=95, subsampling=0)
                keys.append(sha(filename))
            return keys

        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            digests = list(pool.map(draw, range(len(rows))))
        freeze(
            dest / "settings.json",
            dict(
                design_sha256=sha(PLAN),
                input_sha256=sha(path / "settings.json"),
                condition_sha256=sha(condition),
                source_jpeg_sha256=raw["jpeg_sha256"]["own"],
                rows=rows.tolist(),
                jpeg_sha256={arm: [v[i] for v in digests] for i, arm in enumerate(ARMS)},
            ),
        )
        print("Saved JPEGs", name, mode, 4 * len(rows), flush=True)


def oklab_chroma(c):
    x = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lms = np.cbrt(
        x
        @ np.array(
            [
                [0.4122214708, 0.2119034982, 0.0883024619],
                [0.5363325363, 0.6806995451, 0.2817188376],
                [0.0514459929, 0.1073969566, 0.6299787005],
            ]
        )
    )
    return (
        lms
        @ np.array(
            [
                [1.9779984951, 0.0259040371],
                [-2.4285922050, 0.7827717662],
                [0.4505937099, -0.8086757660],
            ]
        )
        * 100
    )


def measure(pics, truth):
    c = np.stack([small(im, 32) for im in pics])
    t = np.stack([small(im, 32) for im in truth])
    dist = np.linalg.norm(oklab_chroma(c) - oklab_chroma(t), axis=-1).mean((1, 2))
    rmse = np.sqrt(np.square(c - t).mean((1, 2, 3)))
    y = (c @ LUMA).reshape(len(c), -1)
    z = (t @ LUMA).reshape(len(t), -1)
    y -= y.mean(1, keepdims=True)
    z -= z.mean(1, keepdims=True)
    corr = np.sum(y * z, 1) / np.maximum(
        np.linalg.norm(y, axis=1) * np.linalg.norm(z, axis=1), 1e-12
    )
    return dict(oklab_chroma_error=dist, rgb_rmse=rmse, luma_correlation=corr)


def stats(values, labels, mask):
    inds = np.flatnonzero(mask)
    if not len(inds):
        return dict(n=0)
    n = np.bincount(labels[inds], minlength=12)
    sums = np.zeros((12, len(ARMS)))
    np.add.at(sums, labels[inds], values[inds])
    draw = np.random.default_rng(SEED).integers(12, size=(10000, 12))
    den = n[draw].sum(1)
    keep = den > 0
    b = sums[draw].sum(1)[keep] / den[keep, None]
    mean = values[inds].mean(0)
    return dict(
        n=len(inds),
        represented_objects=int(np.sum(n > 0)),
        arm_mean=dict(zip(ARMS, mean.tolist(), strict=True)),
        own_minus={
            arm: dict(
                delta=float(mean[2] - mean[j]),
                ci95=np.quantile(b[:, 2] - b[:, j], [0.025, 0.975]).tolist(),
            )
            for j, arm in enumerate(ARMS)
            if j != 2
        },
    )


def assess(a):
    # Verify every settings file before opening even one evaluation photograph.
    cases = list(get_cases(a))
    for name, mode, _, _, _, rows, *_ in cases:
        marker = a.out / "renders" / name / mode / "settings.json"
        check(
            read(marker)["rows"] == rows.tolist() and read(marker)["design_sha256"] == sha(PLAN),
            "Missing frozen render",
        )
    table = reference(a)
    encode = encoder(a)
    results = {}
    hashes = {}
    for name, mode, panel, _, _, rows, folds, words, donors, _ in cases:
        root = a.out / "renders" / name / mode
        cfg = read(root / "settings.json")
        reg = panel["registration"]
        labels = np.asarray(reg["truth"])
        truthwords = np.asarray(reg["words"])[labels]
        check(
            all(table[row]["unique_id"] == w for row, w in zip(rows, truthwords, strict=True)),
            "Reference object mismatch",
        )
        with ZipFile(a.originals) as archive:
            truth = [original(archive, table, int(row)) for row in rows]
            true_checks = [
                hashlib.sha256(
                    archive.read(
                        f"training_images/{table[r]['official_folder']}/{table[r]['stimulus_stem']}.jpg"
                    )
                ).hexdigest()
                for r in rows
            ]
        refpath = a.panel_root / "pixel-scores" / name / mode / "original_cls" / (name + ".npz")
        with np.load(refpath, allow_pickle=False) as z:
            check(
                np.array_equal(z["rows"], rows)
                and np.array_equal(z["jpeg_sha256"], true_checks)
                and str(z["model_sha"]) == FILES["model.safetensors"]
                and str(z["feature_sha"]) == hashlib.sha256(z["cls"].tobytes()).hexdigest(),
                "Wrong independent DINO original embeddings",
            )
            ref = z["cls"].astype(float)
        donorpos = {r: i for i, r in enumerate(rows)}
        jj = np.array([donorpos[r] for r in donors])
        groups = dict(
            all=np.ones(len(rows), bool),
            correct_word=words == truthwords,
            strict_wrong=(jj != np.arange(len(rows)))
            & (labels == labels[jj])
            & (words == words[jj])
            & (folds == folds[jj]),
        )
        metrics = {}
        shown = {}
        for arm in ARMS:
            pics = []
            for i, sig in enumerate(cfg["jpeg_sha256"][arm]):
                pic = root / "images" / arm / f"{i:03d}.jpg"
                check(sha(pic) == sig, "Scored JPEG changed")
                with Image.open(pic) as im:
                    pics.append(im.convert("RGB"))
            m = measure(pics, truth)
            generated = encode(pics).astype(float)
            # The design specifies the original separately centered retrieval procedure.
            # The raw-cosine DINO result also permits comparison to the specific session-panel scorer.
            for key, cos in (
                ("dino_intra10", unit(generated - generated.mean(0)) @ unit(ref - ref.mean(0)).T),
                ("dino_raw_intra10", generated @ ref.T),
            ):
                m[key] = np.asarray(
                    [
                        e["intra10_top1"]
                        for e in score_object_panel(cos, rows, truthwords)["per_photo"]
                    ]
                )
            for k, v in m.items():
                metrics.setdefault(k, []).append(v)
            shown[arm] = [
                pics[min(np.flatnonzero(labels == i), key=lambda j: rows[j])].resize(
                    (90, 90), Image.Resampling.LANCZOS
                )
                for i in range(12)
            ]
        dest = a.out / "scores" / name / mode
        dest.mkdir(parents=True, exist_ok=True)
        sheet = Image.new("RGB", (940, 740), "white")
        drawing = ImageDraw.Draw(sheet)
        for i, w in enumerate(reg["words"]):
            j = min(np.flatnonzero(labels == i), key=lambda j: rows[j])
            x = 470 * (i % 2)
            y = 122 * (i // 2)
            drawing.text((x + 3, y), f"{w} / {rows[j]} / EEG: {words[j]}", fill="black")
            for k, (lab, img) in enumerate(
                [("original", truth[j].resize((90, 90), Image.Resampling.LANCZOS))]
                + [(k, shown[k][i]) for k in ARMS]
            ):
                sheet.paste(img, (x + 3 + 93 * k, y + 14))
                drawing.text((x + 3 + 93 * k, y + 104), lab, fill="black")
        sheet.save(dest / "comparison.jpg", quality=94)
        arrays = {k: np.stack(v, axis=1) for k, v in metrics.items()}
        np.savez(dest / "rows.npz", rows=rows, labels=labels, **arrays)
        results[name + "/" + mode] = {
            g: {k: stats(v, labels, mask) for k, v in arrays.items()} for g, mask in groups.items()
        }
        hashes[name + "/" + mode] = dict(
            render_settings_sha256=sha(root / "settings.json"),
            scored_rows_sha256=sha(dest / "rows.npz"),
            independent_dino_reference_sha256=sha(refpath),
        )
        print(
            "Scored",
            name,
            mode,
            results[name + "/" + mode]["all"]["oklab_chroma_error"],
            flush=True,
        )
    passed = all(
        results[n + "/sub-04"]["all"]["oklab_chroma_error"]["own_minus"][c]["ci95"][1] < 0
        for n in NAMES
        for c in ("neutral", "prior")
    ) and all(
        results[n + "/sub-04"]["strict_wrong"]["oklab_chroma_error"]["own_minus"]["wrong"]["ci95"][
            1
        ]
        < 0
        for n in NAMES
    )
    freeze(
        a.release,
        dict(
            design_sha256=sha(PLAN),
            training=read(a.out / "fitted.json"),
            dino_model_sha256=FILES["model.safetensors"],
            n_unique_photos=240,
            generated_source_jpegs=480,
            color_output_jpegs=1920,
            arms=list(ARMS),
            cases=results,
            hashes=hashes,
            advance=bool(passed),
        ),
    )


def public_gallery(a):
    panels = read(a.panel_root / "design.json")["panels"]
    out = Image.new("RGB", (1056, 1136), "white")
    draw = ImageDraw.Draw(out)
    heading, caption, title = (ImageFont.load_default(size=n) for n in (18, 13, 14))
    for j, name in enumerate(NAMES):
        reg = panels[name]["registration"]
        root = a.out / "renders" / name / "sub-04"
        settings = read(root / "settings.json")
        rows = np.array(reg["rows"])
        labels = np.asarray(reg["truth"])
        y0 = j * 568
        draw.text(
            (8, y0 + 4),
            (
                "Single participant sub-04 / same-session 12 choices"
                if j == 0
                else "Single participant sub-04 / supplied four-word menus"
            ),
            fill="#172e3a",
            font=heading,
        )
        for i, word in enumerate(reg["words"]):
            pos = min(np.flatnonzero(labels == i), key=lambda k: rows[k])
            x = (i % 4) * 264 + 8
            y = y0 + (i // 4) * 178 + 34
            draw.text(
                (x, y), f"{word.replace('_', ' ')} / row {rows[pos]}", fill="#172e3a", font=title
            )
            for col, (arm, label) in enumerate(
                (("neutral", "Generated source"), ("own", "EEG + supports"))
            ):
                path = root / "images" / arm / f"{pos:03d}.jpg"
                check(sha(path) == settings["jpeg_sha256"][arm][pos], "Changed source for gallery")
                with Image.open(path) as im:
                    out.paste(
                        im.resize((123, 123), Image.Resampling.LANCZOS), (x + col * 128, y + 20)
                    )
                draw.text((x + col * 128, y + 148), label, fill="#3e5460", font=caption)
    a.gallery.parent.mkdir(parents=True, exist_ok=True)
    out.save(a.gallery, quality=93, subsampling=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fit", "infer", "render", "score", "gallery"))
    for k, v in dict(
        out=Path("outputs/eeg-color-transfer"),
        expanded=Path("outputs/photo-structure/expanded"),
        double=Path("outputs/double-calibration"),
        panel_root=Path("outputs/session-matched-cohort"),
        manifest=Path("outputs/official-object-readout/sub-03/atm_train_rows.csv"),
        originals=Path("outputs/photo-structure/training_images.zip"),
        model=Path("outputs/photo-structure/dino/facebook-dinov2-small"),
        release=PUBLIC / "eeg_color_transfer.json",
        gallery=PUBLIC.parents[1] / "gallery/eeg_color_transfer_sub04.jpg",
    ).items():
        p.add_argument("--" + k.replace("_", "-"), type=Path, default=v)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a = p.parse_args()
    dict(fit=fit, infer=infer, render=render, score=assess, gallery=public_gallery)[a.stage](a)
