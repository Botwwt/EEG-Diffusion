"""Prepare, apply and independently score a frozen EEG outline edit of real generated JPEGs."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from fit_direct_contour import LOCK, median_prior, zero_mean
from fit_fixed_word_photo_dose import PEOPLE, study_eeg
from fit_repeat_photo_subspace import centered, ridge
from fit_support_contour import (
    HERE,
    OUT,
    PHOTO,
    PUBLIC,
    SOURCE,
    check,
    original,
    population,
    read,
    save,
    segmentation,
    sha,
    silhouette,
    split,
)
from measure_foreground_geometry import MODEL_SHA, ZIP_SHA
from PIL import Image
from refine_registered_color import metrics, thumbnail
from render_fixed_word_photo_dose import PANELS, packet
from scipy.ndimage import map_coordinates
from score_object_panel import score_object_panel
from score_within_photo_dino import FILES, encoder

IMAGE_LOCK = "770b30c32efa062b358929299c864c424ed7f6662a1ff61ed1570753b41980f8"
DOSE = OUT / "fixed-word-photo-dose/render_inputs"
DOSE_SHA = "e304cc422fa1fef0755643e2ff6a4a5abb335f9109899fc89b0c121e4d211f58"
ARMS = ("neutral", "prior", "own", "wrong")


def design():
    check(
        sha(PUBLIC / "direct_contour_image_design.json") == IMAGE_LOCK
        and sha(DOSE / "design.json") == DOSE_SHA,
        "Changed image experiment",
    )


def base(panel):
    return DOSE / "renders" / panel / "cohort-9"


def out(panel):
    d = HERE / "render" / panel
    d.mkdir(exist_ok=True, parents=True)
    return d


def conditioning():
    design()
    check(
        read(HERE / "direct_result.json")["advance"] is True
        and read(HERE / "direct_result.json")["design_sha256"] == LOCK,
        "Generic gate failed",
    )
    table = population(SimpleNamespace(source=SOURCE))
    classes, _, _ = split()
    with np.load(HERE / "contours.npz", allow_pickle=False) as bank:
        check(
            np.array_equal(bank["rows"], (classes[:, None] * 10 + np.arange(10)).ravel()),
            "Generic contour order changed",
        )
        y = bank["values"].reshape(128, 10, 24).astype(float)
    calibration = read(HERE / "direct_calibration.json")
    check(
        calibration["contours_sha256"] == sha(HERE / "contours.npz")
        and calibration["design_sha256"] == LOCK,
        "Generic transfer changed",
    )
    coefficient = calibration["coefficient"]
    projections = []
    shared_rows = None
    for subject in PEOPLE:
        c, _, train, rows, q, _ = study_eeg(subject, table)
        check(
            np.array_equal(c, classes)
            and (shared_rows is None or np.array_equal(shared_rows, rows)),
            "Unaligned cohort",
        )
        shared_rows = rows
        projections.append(ridge(train, centered(y), q))
    source = PHOTO / "training_images.zip"
    check(
        sha(source) == read(PHOTO / "training_images.zip.receipt.json")["sha256"] == ZIP_SHA,
        "Wrong support archive",
    )
    model = segmentation()
    for panel in PANELS:
        rows, folds, words, donors, support, _, _, packet_sha = packet(DOSE, "cohort-9", panel)
        d = out(panel)
        settings = read(base(panel) / "settings.json")
        check(
            settings["rows"] == rows.tolist()
            and settings["folds"] == folds.tolist()
            and settings["seeds"] == (923761 + rows).tolist()
            and settings["packet_sha256"] == packet_sha
            and settings["predicted_words"] == words.tolist()
            and settings["support_rows"] == support.tolist()
            and settings["donor_rows"] == donors.tolist(),
            "Source rendering and physical registration differ",
        )
        # Each fold has a separate support-only cache. The active query images cannot be opened in that fold.
        measured = np.empty((len(rows), 8, 24), np.float64)
        with ZipFile(source) as z:
            for f in np.unique(folds):
                idx = np.flatnonzero(folds == f)
                allowed = np.unique(support[idx])
                check(not np.intersect1d(allowed, rows[idx]).size, "Active query in visual support")
                partial = {int(r): silhouette(model, original(z, table, int(r))) for r in allowed}
                for i in idx:
                    check(
                        all(table[r]["unique_id"] == words[i] for r in support[i]),
                        "Incorrect support concept",
                    )
                    measured[i] = np.stack([partial[int(r)] for r in support[i]])
        prior = median_prior(measured)
        qi = np.searchsorted(shared_rows, rows)
        si = np.searchsorted(shared_rows, support)
        check(
            np.array_equal(shared_rows[qi], rows) and np.array_equal(shared_rows[si], support),
            "Missing EEG rows",
        )
        continuous = zero_mean(
            np.mean([z[qi] - np.median(z[si], axis=1) for z in projections], axis=0)
        )
        lookup = {r: k for k, r in enumerate(rows)}
        donor = np.array([lookup[v] for v in donors])
        check(
            np.array_equal(support, support[donor]) and np.array_equal(words, words[donor]),
            "Donor has different legal supports",
        )
        np.savez(
            d / "conditions.npz",
            rows=rows,
            folds=folds,
            words=words,
            donor_rows=donors,
            support=support,
            prior=prior,
            own=prior + coefficient * continuous,
            wrong=prior + coefficient * continuous[donor],
            packet_sha256=packet_sha,
            baseline_settings_sha256=sha(base(panel) / "settings.json"),
            image_design_sha256=IMAGE_LOCK,
        )
    save(
        "blind_conditions.json",
        dict(
            image_design_sha256=IMAGE_LOCK,
            conditions={p: sha(out(p) / "conditions.npz") for p in PANELS},
            source_generic_gate_sha256=sha(HERE / "direct_result.json"),
            calibration_sha256=sha(HERE / "direct_calibration.json"),
            coefficient=coefficient,
        ),
    )


def warp(im, before, after, frame):
    """Monotone inverse radial map in the source ellipse, with the exact image border fixed."""
    h, w = im.height, im.width
    cx, cy = frame[0] * w - 0.5, frame[1] * h - 0.5
    y, x = np.indices((h, w), dtype=np.float64)
    dx = x - cx
    dy = y - cy
    phi = np.mod(np.arctan2(dy / (frame[3] * h), dx / (frame[2] * w)), 2 * np.pi)
    # The profile coordinates refer to sector centers.
    step = 0.35 * np.interp(phi * 24 / (2 * np.pi) - 0.5, np.arange(24), after - before, period=24)
    fx = np.where(dx > 0, dx / (w - 1 - cx), -dx / cx)
    fy = np.where(dy > 0, dy / (h - 1 - cy), -dy / cy)
    frac = np.maximum(fx, fy).clip(0, 1)
    inverse = 1 / (np.exp(step) * (1 - frac) + frac)
    xy = np.stack((cy + dy * inverse, cx + dx * inverse))
    arr = np.asarray(im, dtype=np.float32)
    result = np.stack(
        [
            map_coordinates(arr[:, :, c], xy, order=1, mode="nearest", prefilter=False)
            for c in range(3)
        ],
        axis=2,
    )
    return Image.fromarray(np.rint(result).clip(0, 255).astype(np.uint8))


def edit():
    design()
    model = segmentation()
    receipts = read(HERE / "blind_conditions.json")
    check(receipts["image_design_sha256"] == IMAGE_LOCK, "Wrong blind conditions")
    for panel in PANELS:
        d = out(panel)
        p = d / "conditions.npz"
        s = read(base(panel) / "settings.json")
        check(sha(p) == receipts["conditions"][panel], "Changed blind prediction")
        hashes = {k: [] for k in ARMS}
        with np.load(p, allow_pickle=False) as z:
            check(
                str(z["baseline_settings_sha256"]) == sha(base(panel) / "settings.json")
                and str(z["image_design_sha256"]) == IMAGE_LOCK,
                "Wrong source JPEGs",
            )
            for i, expected in enumerate(s["jpeg_sha256"]["uniform"]):
                src = base(panel) / "images/uniform" / f"{i:03d}.jpg"
                check(sha(src) == expected, "Changed uniform reconstruction")
                with Image.open(src) as source:
                    check(
                        source.size == (512, 512)
                        and source.format == "JPEG"
                        and source.mode == "RGB",
                        "Wrong image input",
                    )
                    im = source.copy()
                profile, frame = silhouette(model, im, with_frame=True)
                for arm in ARMS:
                    image = im if arm == "neutral" else warp(im, profile, z[arm][i], frame)
                    path = d / "images" / arm / f"{i:03d}.jpg"
                    path.parent.mkdir(exist_ok=True, parents=True)
                    temp = path.with_suffix(".partial")
                    image.save(temp, "JPEG", quality=95, subsampling=0)
                    temp.replace(path)
                    hashes[arm].append(sha(path))
        meta = dict(
            image_design_sha256=IMAGE_LOCK,
            condition_sha256=sha(p),
            uniform_source_sha256=s["jpeg_sha256"]["uniform"],
            jpeg_sha256=hashes,
            rows=s["rows"],
            folds=s["folds"],
            predicted_words=s["predicted_words"],
            support_rows=s["support_rows"],
            donor_rows=s["donor_rows"],
            seeds=s["seeds"],
        )
        save(f"render/{panel}/settings.json", meta)
    save(
        "blind_images.json",
        dict(
            image_design_sha256=IMAGE_LOCK,
            conditions_sha256=sha(HERE / "blind_conditions.json"),
            panels={p: sha(out(p) / "settings.json") for p in PANELS},
        ),
    )


def score(publish):
    design()
    blind = read(HERE / "blind_images.json")
    check(
        blind["image_design_sha256"] == IMAGE_LOCK
        and blind["conditions_sha256"] == sha(HERE / "blind_conditions.json"),
        "Images have not been frozen",
    )
    table = population(SimpleNamespace(source=SOURCE))
    model = segmentation()
    encode = encoder(
        SimpleNamespace(model=PHOTO / "dino/facebook-dinov2-small", device="cuda", batch_size=16)
    )
    cases = {}
    by_panel = {}
    input_hashes = {}
    eligibility = {}
    archive_path = PHOTO / "training_images.zip"
    check(sha(archive_path) == ZIP_SHA, "Reference archive differs from the registered release")
    with ZipFile(archive_path) as archive:
        for panel in PANELS:
            d = out(panel)
            s = read(d / "settings.json")
            check(sha(d / "settings.json") == blind["panels"][panel], "Changed image freeze")
            rows = np.array(s["rows"])
            concept = rows // 10
            donors = np.array(s["donor_rows"])
            eligibility[panel] = dict(
                total=len(rows),
                exchanged=int(np.sum(rows != donors)),
                exchanged_and_same_true_class=int(
                    np.sum((rows != donors) & (concept == donors // 10))
                ),
            )
            true = [original(archive, table, int(r)) for r in rows]
            truth_shape = np.stack([silhouette(model, x) for x in true])
            pixels = [thumbnail(x) for x in true]
            real = encode(true).astype(np.float64)
            check(
                np.allclose(np.linalg.norm(real, axis=1), 1, atol=1e-5)
                and np.all(np.argmax(real @ real.T, axis=1) == np.arange(120)),
                "DINO identities differ",
            )
            cases[panel] = {}
            by_panel[panel] = {}
            for arm in ARMS:
                paths = [d / "images" / arm / f"{i:03d}.jpg" for i in range(120)]
                check(
                    [sha(p) for p in paths] == s["jpeg_sha256"][arm], "Reopened JPEG was modified"
                )
                ims = []
                for p in paths:
                    with Image.open(p) as im:
                        ims.append(im.convert("RGB"))
                shapes = np.stack([silhouette(model, x) for x in ims])
                sim = encode(ims).astype(np.float64) @ real.T
                info = score_object_panel(sim, rows, concept)
                phot = info["per_photo"]
                rgb = np.array(
                    [metrics(thumbnail(x), t)[1:] for x, t in zip(ims, pixels, strict=True)]
                )
                values = dict(
                    shape_l1=np.abs(shapes - truth_shape).mean(1),
                    dino_within10=np.array([r["intra10_top1"] for r in phot]),
                    dino_photo120=np.array([r["photo_top1"] for r in phot]),
                    dino_object12=np.array([r["object_top1"] for r in phot]),
                    rgb_rmse32=rgb[:, 0],
                    luma_correlation32=rgb[:, 1],
                )
                labels = np.unique(concept)
                cases[panel][arm] = {
                    k: np.array([v[concept == c].mean() for c in labels]) for k, v in values.items()
                }
                by_panel[panel][arm] = {
                    k: float(v.sum() if k.startswith("dino_") else v.mean())
                    for k, v in values.items()
                }
                np.savez(d / f"{arm}_scores.npz", rows=rows, **values)
            input_hashes[panel] = s["condition_sha256"]
    rng = np.random.default_rng(260556)
    draw = {p: rng.integers(12, size=(10000, 12)) for p in PANELS}
    paired = {}
    overall = {}
    for arm in ARMS:
        overall[arm] = {
            key: float(np.concatenate([cases[p][arm][key] for p in PANELS]).mean())
            for key in cases[PANELS[0]][arm]
        }
        if arm == "own":
            continue
        paired[arm] = {}
        for key in overall[arm]:
            low = key in ("shape_l1", "rgb_rmse32")
            delta = {
                p: (-1 if low else 1) * (cases[p]["own"][key] - cases[p][arm][key]) for p in PANELS
            }
            samples = np.mean([d[draw[p]].mean(1) for p, d in delta.items()], axis=0)
            paired[arm][key] = dict(
                favorable_delta=float(np.mean(list(delta.values()))),
                ci95=np.quantile(samples, [0.025, 0.975]).tolist(),
            )
    passed = all(
        paired[k]["shape_l1"]["ci95"][0] > 0 and paired[k]["dino_within10"]["favorable_delta"] >= 0
        for k in ("prior", "wrong")
    )
    result = dict(
        image_design_sha256=IMAGE_LOCK,
        blind_sha256=sha(HERE / "blind_images.json"),
        jpeg_manifest_sha256=blind["panels"],
        scorer_sha256=sha(Path(__file__)),
        reference_zip_sha256=ZIP_SHA,
        foreground_model_sha256=MODEL_SHA,
        dino_model_sha256=FILES["model.safetensors"],
        condition_sha256=input_hashes,
        donor_eligibility=eligibility,
        panels=list(PANELS),
        n_photos=240,
        photos_per_object=10,
        objects=24,
        jpegs=960,
        observed_people=9,
        physical_trials_per_photo=36,
        measures="Within each panel DINO entries give correct credit out of 120; aggregate DINO entries give rates. Shape is mean absolute normalized log radius. Positive paired deltas favor matched EEG on every outcome.",
        per_panel=by_panel,
        overall=overall,
        paired=paired,
        passes_image_gate=bool(passed),
    )
    save("image_result.json", result)
    if publish:
        (PUBLIC / "direct_contour_images.json").write_bytes(
            (HERE / "image_result.json").read_bytes()
        )
    print("Independent JPEG scores", overall, "passed", passed, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("conditions", "edit", "score"))
    p.add_argument("--publish", action="store_true")
    a = p.parse_args()
    if a.stage == "conditions":
        conditioning()
    elif a.stage == "edit":
        edit()
    else:
        score(a.publish)
