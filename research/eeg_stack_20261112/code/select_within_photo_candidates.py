"""Select from a frozen bank of eight generated images per predicted object and calibration fold."""

import argparse
import hashlib
import io
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from readout_external_images import MODEL_REV, MODEL_SHA, h14_batch, h14_pixels, load_h14

ARMS = ("own", "uniform", "wrong", "fixed0")
PANELS = ("registered", "replicate")
SEED = 740291


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, obj):
    tmp = path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(path)


def unit(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if not np.isfinite(norm).all():
        raise ValueError("Nonfinite H/14 vector")
    # A zero residual has zero cosine against all candidates; ties use the first index.
    return v / np.maximum(norm, 1e-12)


def bank_metadata(args, panel):
    dirname = args.bank / "renders" / panel
    manifest = dirname / "settings.json"
    gen = read(manifest)
    path = args.bank / (panel + ".npz")
    with np.load(path, allow_pickle=False) as z:
        words, prior, support = [z[k] for k in ("words", "uniform", "support_rows")]
    if (
        gen["panel"] != panel
        or gen["condition_sha256"] != sha(path)
        or gen["design_sha256"] != sha(args.bank / "design.json")
        or gen["seeds"] != list(range(SEED, SEED + 8))
        or len(gen["records"]) != 480
        or len(words) != 12
        or len(set(words)) != 12
        or prior.shape != (5, 12, 1024)
        or support.shape != (5, 12, 8)
        or not np.allclose(np.linalg.norm(prior, axis=2), 1, atol=5e-5)
        or any(np.unique(s).size != 96 for s in support)
    ):
        raise ValueError(f"Incomplete or inconsistent candidate bank for {panel}")
    lookup = {}
    for r in gen["records"]:
        f, w, k = r["fold"], r["word"], r["candidate"]
        if (
            type(f) is not int
            or type(k) is not int
            or f not in range(5)
            or k not in range(8)
            or w not in words
            or r["seed"] != SEED + k
            or (f, w, k) in lookup
            or r["path"] != f"images/f{f}/{w}/{k:02d}.jpg"
        ):
            raise ValueError("Invalid bank manifest record")
        lookup[f, w, k] = r
    ordered = [lookup[f, w, k] for f in range(5) for w in words for k in range(8)]
    return (
        dirname,
        ordered,
        words,
        prior.astype(np.float64),
        support,
        dict(
            bank_manifest_sha256=sha(manifest),
            bank_prior_sha256=sha(path),
            model_sha256=MODEL_SHA,
            extraction_source_sha256=sha(Path(__file__).with_name("readout_external_images.py")),
        ),
    )


def candidate_features(args, panel, folder, records, provenance, extractor):
    path = args.out / "features" / (panel + ".npz")
    hashes = [r["sha256"] for r in records]
    key = json.dumps(
        {
            **provenance,
            "device": args.device,
            "batch_size": args.batch_size,
            "model_revision": MODEL_REV,
            "order": "fold, bank word ordinal, candidate",
        },
        sort_keys=True,
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if (
                str(z["key"]) != key
                or not np.array_equal(z["jpeg_sha256"], hashes)
                or z["unit_h14"].shape != (5, 12, 8, 1024)
                or hashlib.sha256(z["unit_h14"].tobytes()).hexdigest() != str(z["feature_sha256"])
            ):
                raise ValueError(
                    f"Existing H/14 cache has a different bank or extraction protocol at {path}"
                )
            for r in records:
                if sha(folder / r["path"]) != r["sha256"]:
                    raise ValueError("JPEG bytes changed after feature extraction")
            return z["unit_h14"].astype(np.float64), sha(path)
    if not extractor:
        extractor.extend(load_h14(args.assets, args.device))
    blocks = []
    for start in range(0, len(records), args.batch_size):
        pixels = []
        for r in records[start : start + args.batch_size]:
            data = (folder / r["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != r["sha256"]:
                raise ValueError("Candidate JPEG hash differs from bank manifest")
            with Image.open(io.BytesIO(data)) as im:
                if im.format != "JPEG" or im.mode != "RGB" or im.size != (512, 512):
                    raise ValueError("Expected one generated RGB 512-square JPEG")
                pixels.append(h14_pixels(im))
        blocks.append(h14_batch(*extractor, pixels))
    matrix = np.concatenate(blocks).reshape(5, 12, 8, 1024)
    if not np.allclose(np.linalg.norm(matrix, axis=-1), 1, atol=1e-5):
        raise ValueError("Candidate encoder did not produce finite unit H/14 vectors")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npz")
    np.savez(
        tmp,
        unit_h14=matrix,
        jpeg_sha256=hashes,
        key=key,
        feature_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
    )
    tmp.replace(path)
    return matrix.astype(np.float64), sha(path)


def place(source, target, digest):
    if target.exists():
        if sha(target) != digest:
            raise ValueError(f"Existing output does not match the chosen candidate: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        tmp = target.with_suffix(".partial.jpg")
        shutil.copyfile(source, tmp)
        if sha(tmp) != digest:
            raise ValueError("Copied image checksum mismatch") from None
        tmp.replace(target)


def select(args, subject, panel, bank, features, feature_sha):
    folder, records, bank_words, centers, cal, provenance = bank
    path = args.conditions / subject / panel / "conditions_512.npz"
    with np.load(path, allow_pickle=False) as z:
        rows, folds, words, donors, support = [
            z[k] for k in ("rows", "folds", "predicted_words", "donor_rows", "support_rows")
        ]
        eeg = {k: z[k].astype(np.float64) for k in ("own", "wrong", "uniform")}
    if (
        not (rows.shape == folds.shape == words.shape == donors.shape == (120,))
        or set(rows) != set(donors)
        or len(set(rows)) != 120
    ):
        raise ValueError("Invalid EEG row/word/fold/donor registration")
    if set(words) - set(bank_words) or set(folds) != set(range(5)) or support.shape != (120, 8):
        raise ValueError("Unknown predicted object or fold")
    cols = np.array([bank_words.tolist().index(w) for w in words])
    ref = unit(centers[folds, cols])
    if (
        any(
            v.shape != (120, 1024) or not np.allclose(np.linalg.norm(v, axis=1), 1, atol=5e-5)
            for v in eeg.values()
        )
        or not np.allclose(ref, eeg["uniform"], atol=2e-6, rtol=0)
        or not np.array_equal(np.sort(support, axis=1), np.sort(cal[folds, cols], axis=1))
    ):
        raise ValueError("EEG and image bank do not share the same eight-support prior")
    for f in range(5):
        if np.intersect1d(cal[f], rows[folds == f]).size or set(cal[f].ravel()) != set(
            rows[folds != f]
        ):
            raise ValueError("Invalid calibration split")
    lookup = {r: i for i, r in enumerate(rows)}
    ix = np.array([lookup[r] for r in donors])
    if not (
        np.array_equal(words, words[ix])
        and np.array_equal(folds, folds[ix])
        and np.array_equal(eeg["wrong"], eeg["own"][ix])
    ):
        raise ValueError("Wrong-EEG packet differs from the registered donor")
    # Center on the unit direction of the shared eight-support mean, matching the
    # upstream single-query residual metric. No target label or original enters ranking.
    candidates = unit(features[folds, cols])
    directions = unit(candidates - ref[:, None])
    scores = {
        k: np.einsum("nkd,nd->nk", directions, unit(unit(eeg[k]) - ref)) for k in ("own", "wrong")
    }
    scores["uniform"] = np.einsum("nkd,nd->nk", candidates, ref)
    chosen = {k: x.argmax(1) for k, x in scores.items()}
    chosen["fixed0"] = np.zeros(len(rows), dtype=int)
    if not np.array_equal(chosen["wrong"], chosen["own"][ix]):
        raise ValueError(
            "A wrong-EEG selection differs from its donor's selection in the same library"
        )
    dest = args.out / "renders" / panel / subject
    dest.mkdir(exist_ok=True, parents=True)
    hashes = {}
    for arm in ARMS:
        output = dest / "images" / arm
        output.mkdir(exist_ok=True, parents=True)
        hashes[arm] = []
        for i, (f, c, k) in enumerate(zip(folds, cols, chosen[arm], strict=True)):
            record = records[(f * 12 + c) * 8 + k]
            place(folder / record["path"], output / f"{i:03d}.jpg", record["sha256"])
            hashes[arm].append(record["sha256"])
    detail = dest / "selection.json"
    save(
        detail,
        dict(
            indices={k: v.tolist() for k, v in chosen.items()},
            scores={k: v.tolist() for k, v in scores.items()},
            score_columns=list(range(8)),
            tie_break="smallest candidate index",
            own_residual_norm=np.linalg.norm(unit(eeg["own"]) - ref, axis=1).tolist(),
            selected_bank_paths={
                k: [
                    records[(f * 12 + c) * 8 + j]["path"]
                    for f, c, j in zip(folds, cols, chosen[k], strict=True)
                ]
                for k in ARMS
            },
        ),
    )
    save(
        dest / "settings.json",
        dict(
            subject=subject,
            panel=panel,
            arms=list(ARMS),
            rows=rows.tolist(),
            folds=folds.tolist(),
            predicted_words=words.tolist(),
            donor_rows=donors.tolist(),
            support_rows=support.tolist(),
            packet_sha256=sha(path),
            candidate_feature_sha256=feature_sha,
            selection_sha256=sha(detail),
            **provenance,
            model_revision=MODEL_REV,
            selector_sha256=sha(Path(__file__)),
            jpeg_sha256=hashes,
            candidate_indices={k: v.tolist() for k, v in chosen.items()},
            candidate_seeds={k: (SEED + v).tolist() for k, v in chosen.items()},
            registration="Eight shared synthetic candidates per panel, fold and predicted object. Fixed0 selects index zero. Uniform uses uncentered cosine to the unit support prior. Own and wrong use centered cosine with their respective frozen EEG directions. One original candidate JPEG is copied or hard-linked without modification.",
        ),
    )
    print("Frozen selections", panel, subject, len(rows), "queries", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=Path("outputs/photo-structure/h14_bank"))
    p.add_argument("--conditions", type=Path, default=Path("outputs/photo-structure/within_photo"))
    p.add_argument("--out", type=Path, default=Path("outputs/photo-structure/h14_bank/selected"))
    p.add_argument("--assets", type=Path, default=Path("outputs/external-object-anchors/assets"))
    p.add_argument("--subjects", nargs="+", default=["sub-04", "sub-05", "sub-01"])
    p.add_argument("--panels", nargs="+", choices=PANELS, default=list(PANELS))
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    if a.batch_size <= 0:
        p.error("Positive batch size required")
    extractor = []
    for panel in a.panels:
        bank = bank_metadata(a, panel)
        features, signature = candidate_features(a, panel, bank[0], bank[1], bank[-1], extractor)
        for subject in a.subjects:
            select(a, subject, panel, bank, features, signature)


if __name__ == "__main__":
    main()
