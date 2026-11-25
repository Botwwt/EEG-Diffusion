"""Estimate EEG-predictable spatial differences across four overlapping H/14 image regions."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from fit_repeat_photo_subspace import centered, ridge
from fit_spatial_photo import original, table
from fit_within_photo_embedding import images, training
from PIL import Image
from readout_external_images import MODEL_SHA, h14_batch, load_h14
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import check
from score_within_photo_dino import ZIP_SHA

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
DESIGN = PUBLIC / "regional_h14_design.json"
SITES = ((0, 0), (160, 0), (0, 160), (160, 160))


def split(a):
    meta = read(a.data / "trial-object-encoder/generic/receipt.json")
    rng = np.random.default_rng(260558)
    return [
        np.sort(rng.choice(meta[key], n, replace=False))
        for key, n in (("train_classes", 128), ("test_classes", 32))
    ]


def crop_features(a, ids):
    out = a.root / "regions"
    out.mkdir(parents=True, exist_ok=True)
    check(sha(a.images) == ZIP_SHA, "Unexpected original photo archive")
    tab = table(a)
    model, mean, std = load_h14(a.data / "external-object-anchors/assets", a.device)
    files = {}
    with ZipFile(a.images) as archive:
        for i, cid in enumerate(ids):
            path = out / f"{cid:04d}.npz"
            if not path.is_file():
                views = []
                for row in int(cid) * 10 + np.arange(10):
                    frame = original(archive, tab, int(row))
                    for x, y in SITES:
                        crop = frame.crop((x, y, x + 352, y + 352)).resize(
                            (224, 224), Image.Resampling.BICUBIC
                        )
                        views.append(np.asarray(crop))
                vectors = (
                    np.concatenate(
                        [
                            h14_batch(model, mean, std, views[j : j + 8])
                            for j in range(0, len(views), 8)
                        ]
                    )
                    .reshape(10, 4, 1024)
                    .astype(np.float32)
                )
                tmp = path.with_suffix(".partial.npz")
                np.savez(
                    tmp,
                    concept=cid,
                    values=vectors,
                    design_sha256=sha(DESIGN),
                    model_sha256=MODEL_SHA,
                )
                tmp.replace(path)
            with np.load(path, allow_pickle=False) as z:
                check(
                    int(z["concept"]) == cid
                    and str(z["design_sha256"]) == sha(DESIGN)
                    and str(z["model_sha256"]) == MODEL_SHA
                    and z["values"].shape == (10, 4, 1024)
                    and np.allclose(np.linalg.norm(z["values"], axis=-1), 1, atol=1e-5),
                    "Invalid crop cache",
                )
            files[str(cid)] = sha(path)
            if (i + 1) % 16 == 0:
                print(i + 1, "/", len(ids), flush=True)
    return files


def load_fields(a, ids, sig):
    values = []
    for c in ids:
        p = a.root / "regions" / f"{c:04d}.npz"
        check(sha(p) == sig[str(c)], "Regional feature cache changed")
        with np.load(p, allow_pickle=False) as z:
            values.append(z["values"])
    y = np.array(values, dtype=float)
    return (y - y.mean(2, keepdims=True)).reshape(len(ids), 10, 4096)


def relative_to_support(x):
    paired = x.reshape(len(x), 5, 2, -1)
    prior = (x.sum(1)[:, None] - paired.sum(2)) / 8
    return x - np.repeat(prior, 2, axis=1)


def rotated(z):
    return z.reshape(*z.shape[:-1], 4, 1024)[..., ::-1, :].reshape(z.shape)


def generic(a, fit_ids, test_ids):
    meta = read(a.root / "generic_regions.json")
    check(
        meta["design_sha256"] == sha(DESIGN) and not np.intersect1d(fit_ids, test_ids).size,
        "Changed development design",
    )
    ids = np.r_[fit_ids, test_ids]
    y = load_fields(a, ids, meta["files"])
    c, x, _ = training(
        SimpleNamespace(expanded=a.data / "photo-structure/expanded"),
        "sub-04",
        512,
        images(a.data / "official-object-readout/sub-03"),
    )
    ix = {v: i for i, v in enumerate(c)}
    x = x[[ix[k] for k in ids]]
    n = len(fit_ids)
    raw = ridge(x[:n], centered(y[:n]), x[n:])
    target = relative_to_support(y[n:])
    matched = relative_to_support(raw)
    wrong = matched.reshape(len(test_ids), 5, 2, -1)[:, :, ::-1].reshape(matched.shape)
    pred = {
        "matched": matched,
        "mean": np.zeros_like(matched),
        "rotated": rotated(matched),
        "exchanged": wrong,
    }
    err = {k: np.square(z - target).sum(-1).mean(-1) / 4 for k, z in pred.items()}
    draws = np.random.default_rng(read(DESIGN)["bootstrap_seed"]).integers(
        len(test_ids), size=(10000, len(test_ids))
    )
    benefit = {}
    for k in ("mean", "rotated", "exchanged"):
        d = err[k] - err["matched"]
        benefit[k] = dict(
            mean=float(d.mean()), ci95=np.quantile(d[draws].mean(1), [0.025, 0.975]).tolist()
        )
    pairs = {}
    for k, z in pred.items():
        differences = z[:, ::2] - z[:, 1::2]
        margins = np.sum(differences * (target[:, ::2] - target[:, 1::2]), axis=-1)
        pairs[k] = float(((margins > 1e-12) + 0.5 * (abs(margins) <= 1e-12)).sum())
    freeze(
        a.release,
        dict(
            design_sha256=sha(DESIGN),
            source_sha256=sha(Path(__file__)),
            feature_receipt_sha256=sha(a.root / "generic_regions.json"),
            training_objects=fit_ids.tolist(),
            heldout_objects=test_ids.tolist(),
            n_photos=len(test_ids) * 10,
            n_pairs=len(test_ids) * 5,
            mean_squared_field_error={k: float(v.mean()) for k, v in err.items()},
            control_error_minus_matched=benefit,
            pair_credit=pairs,
            advance=all(benefit[k]["ci95"][0] > 0 for k in ("mean", "rotated")),
        ),
    )
    print(read(a.release)["advance"], benefit, pairs, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("extract", "generic"))
    p.add_argument("--root", type=Path, default=Path("outputs/regional-h14"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--release", type=Path, default=PUBLIC / "regional_h14.json")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    a.images = a.data / "photo-structure/training_images.zip"
    a.manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    train, test = split(a)
    if a.stage == "extract":
        files = crop_features(a, np.r_[train, test])
        freeze(
            a.root / "generic_regions.json",
            dict(
                files=files,
                design_sha256=sha(DESIGN),
                original_zip_sha256=ZIP_SHA,
                model_sha256=MODEL_SHA,
            ),
        )
    else:
        generic(a, train, test)
