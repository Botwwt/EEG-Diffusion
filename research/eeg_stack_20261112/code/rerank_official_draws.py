"""Freeze one of four real generations per official sub-03 R80 EEG.

Run from the repository root with the external-anchor Torch/Transformers environment.
No reference photos, evaluation labels, filenames describing objects, or gallery
embeddings enter this script. All methods see the same four ordered candidates.
"""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from fit_eeg_image_likelihood import ImageLikelihood
from PIL import Image
from readout_external_images import (
    MODEL_REV,
    MODEL_SHA,
    dump,
    h14_batch,
    h14_pixels,
    load_h14,
    sha,
    write_csv,
)

SOURCES = {
    "draw1_continuous": "continuous/predicted",
    "draw1_word_fused": "candidates/official_continuous/images/word_fused",
    "draw2_continuous": "candidates/official_second_draw/images/predicted",
    "draw2_word_fused": "candidates/official_second_draw/images/word_fused",
}
N = 200
EEG_SHA = "15e96ed9d6626662b85c96dee41ebf1fdaf2d0e66b643b84e52955107a64137f"
CONTROL_SEED = 20261128


def inputs(root):
    filenames = [f"{i:03}.jpg" for i in range(N)]
    columns = []
    for folder in SOURCES.values():
        base = root / folder
        if sorted(p.name for p in base.glob("*.jpg")) != filenames:
            raise ValueError(f"{folder} must contain exactly rows 000 to 199")
        columns.append([base / n for n in filenames])
    paths = np.array(columns, dtype=object).T
    hashes = np.array([[sha(p) for p in row] for row in paths])
    return paths, hashes


def crop(p):
    with Image.open(p) as opened:
        return h14_pixels(opened)


def encode(paths, hashes, args):
    import torch
    import transformers

    cache = args.out / "candidate_h14.npz"
    fp = "float16" if args.device.startswith("cuda") else "float32"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            if (
                np.array_equal(z["source_sha256"], hashes)
                and str(z["model_sha256"]) == MODEL_SHA
                and str(z["precision"]) == fp
                and z["embeddings"].shape == (N, 4, 1024)
            ):
                cached = z["embeddings"]
                if np.isfinite(cached).all() and np.allclose(
                    np.linalg.norm(cached, axis=-1), 1, atol=1e-5
                ):
                    return cached
    model, mean, std = load_h14(args.assets, args.device)
    flat = paths.ravel()
    parts = []
    for lo in range(0, len(flat), args.batch_size):
        pixels = [crop(p) for p in flat[lo : lo + args.batch_size]]
        parts.append(h14_batch(model, mean, std, pixels))
        if lo // args.batch_size % 10 == 0:
            print(f"Encoded {lo + len(pixels)}/{len(flat)} generated candidates", flush=True)
    embedded = np.concatenate(parts).reshape(N, len(SOURCES), 1024)
    temp = cache.with_suffix(".partial.npz")
    np.savez(
        temp,
        embeddings=embedded,
        source_sha256=hashes,
        model_sha256=MODEL_SHA,
        precision=fp,
        source_names=list(SOURCES),
        rows=np.arange(N),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
    )
    temp.replace(cache)
    return embedded


def freeze(eeg, features, scorer):
    rows = np.arange(len(eeg))
    rng = np.random.default_rng(CONTROL_SEED)
    wrong = rng.permutation(rows)
    while np.any(wrong == rows):
        wrong = rng.permutation(rows)
    scores = {
        m: scorer.score(eeg, features, m) for m in ("selected", "gaussian_global", "gaussian_image")
    }
    scores["wrong_eeg"] = scorer.score(eeg[wrong], features, "selected")
    choices = {"first_draw": np.zeros(len(eeg), dtype=int)}
    for mode, values in scores.items():
        if values.shape != features.shape[:2] or not np.isfinite(values).all():
            raise ValueError(f"Invalid scores for {mode}")
        choices[mode] = values.argmax(axis=1)
    return choices, scores, wrong


def export(out, paths, hashes, choices, scores, wrong):
    names = list(SOURCES)
    row_records = []
    candidate_records = []
    for row in range(len(paths)):
        rec = {"row": row, "wrong_eeg_row": int(wrong[row])}
        for arm, indices in choices.items():
            rec[arm] = names[indices[row]]
            rec[arm + "_sha256"] = hashes[row, indices[row]]
        row_records.append(rec)
        for j, source in enumerate(names):
            item = {"row": row, "candidate": source, "sha256": hashes[row, j]}
            item.update({name: float(values[row, j]) for name, values in scores.items()})
            candidate_records.append(item)
    write_csv(out / "choices.csv", row_records)
    write_csv(out / "candidate_scores.csv", candidate_records)
    archive = out / "images.tar"
    temp = archive.with_suffix(".partial.tar")
    with tarfile.open(temp, "w", format=tarfile.USTAR_FORMAT) as tar:
        for mode, indices in choices.items():
            dest = out / "images" / mode
            dest.mkdir(parents=True, exist_ok=True)
            for row, j in enumerate(indices):
                raw = paths[row, j].read_bytes()
                if hashlib.sha256(raw).hexdigest() != hashes[row, j]:
                    raise ValueError("Candidate changed after extraction")
                file = dest / f"{row:03}.jpg"
                file.write_bytes(raw)
                member = tarfile.TarInfo(file.relative_to(out).as_posix())
                member.size = len(raw)
                tar.addfile(member, io.BytesIO(raw))
        for name in ("choices.csv", "candidate_scores.csv"):
            raw = (out / name).read_bytes()
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            tar.addfile(member, io.BytesIO(raw))
    temp.replace(archive)
    return sha(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("outputs/object-fidelity"))
    parser.add_argument(
        "--readout", type=Path, default=Path("outputs/official-object-readout/sub-03")
    )
    parser.add_argument(
        "--assets", type=Path, default=Path("outputs/external-object-anchors/assets")
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/object-fidelity/reranked"))
    parser.add_argument("--batch-size", default=12, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("positive --batch-size required")
    args.out.mkdir(parents=True, exist_ok=True)
    paths, hashes = inputs(args.source_root)
    eeg_file = args.readout / "eeg_r80.npy"
    model_file = args.readout / "image-likelihood/likelihood.npz"
    if sha(eeg_file) != EEG_SHA:
        raise ValueError("Expected the released sub-03 R80 EEG rows in original order")
    eeg = np.load(eeg_file, allow_pickle=False)
    scorer = ImageLikelihood.load(model_file)
    features = encode(paths, hashes, args)
    choices, scores, wrong = freeze(eeg, features, scorer)
    tar_sha = export(args.out, paths, hashes, choices, scores, wrong)
    modes = list(choices)
    report = {
        "rows": len(paths),
        "ordered_candidate_sources": SOURCES,
        "checkpoint_revision": MODEL_REV,
        "checkpoint_sha256": MODEL_SHA,
        "eeg_sha256": EEG_SHA,
        "likelihood_sha256": sha(model_file),
        "selection": "Maximum of four frozen scores independently per EEG row. First source wins ties.",
        "selected_gamma": float(scorer.values["gamma"]),
        "selected_mix": float(scorer.values["mix"]),
        "arms": {
            "first_draw": "Always use draw1_continuous",
            "selected": "Train-concept cross-validation selected mixture and variance",
            "gaussian_global": "Gaussian likelihood with global per-axis residual variance",
            "gaussian_image": "Gaussian likelihood with image-dependent predicted residual variance",
            "wrong_eeg": "Identical candidates and selected scorer, deranged EEG rows",
        },
        "wrong_eeg_derangement_seed": CONTROL_SEED,
        "counts_by_source": {
            m: dict(zip(SOURCES, np.bincount(v, minlength=4).tolist(), strict=True))
            for m, v in choices.items()
        },
        "differing_rows": {
            a + "__" + b: int(np.sum(choices[a] != choices[b]))
            for i, a in enumerate(modes)
            for b in modes[i + 1 :]
        },
        "images_tar_sha256": tar_sha,
    }
    dump(args.out / "report.json", report)
    print(json.dumps(report["counts_by_source"], indent=2), flush=True)


if __name__ == "__main__":
    main()
