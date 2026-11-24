"""CPU transfer of the registered 8-shot decoder. Generic heads are fitted only on sub-03.

python fit_registered_transfer.py --archive ROOT --panel-dirs RUN/registered RUN/replicate \
    --subjects-dir PACKETS --out RUN/transfer [--subjects sub-03 sub-04]

Each sub-XX.npz supplies rows, mean_eeg (rows x 63 x 250), channels, times,
subject and repeats. Source-waveform and historical prediction checks precede
any new-participant inference. The conditions are compatible with render_registered_objects.py.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from decode_eeg_layout import EEG_SHA, eeg_design, registration
from fit_photo_conditions import axis_and_radius, normalize, restore_cone

VISUAL_KEY = "CLIP-ViT-H-14-laion2B-s32B-b79K"
METHODS = ("raw_eeg", "projected_eeg", "visual_h14", "fused")
SOURCE = "sub-03"
SEED = 260425


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        if path.suffix == ".json":
            tmp.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
        else:
            np.savez(tmp, **value)
    os.replace(tmp.name, path)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def registered_panel(path, labels, concept_words):
    ev = read(path / "evaluation.json")
    replay = read(path / "ablation.json")
    words = np.array(ev["words"])
    rows, truth, folds = (np.array(ev[k], dtype=np.int64) for k in ("rows", "truth", "folds"))
    support = np.array(replay["calibration_rows_by_fold_and_word"], dtype=np.int64)
    if (
        len(set(words)) != 12
        or len(set(rows)) != 120
        or set(folds) != set(range(5))
        or rows.shape != truth.shape
        or rows.shape != folds.shape
        or support.shape != (5, 12, 8)
        or ev["mix"] != 0.25
        or ev["trial_count"] != 4
        or ev["n_calibration_photos_per_object"] != 8
        or any(ev[k] != replay[k] for k in ("rows", "truth", "folds", "words"))
    ):
        raise ValueError(f"Invalid registered fold specification in {path}")
    classes = np.array([concept_words.index(w) for w in words])
    if not np.array_equal(labels[rows], classes[truth]):
        raise ValueError("Registered stimulus identity differs from the archive")
    for fold, cal in enumerate(support):
        test = rows[folds == fold]
        if set(cal.ravel()) != set(rows) - set(test) or len(test) != 24:
            raise ValueError("Calibration and query photos must partition each panel")
        if not np.all(labels[cal] == classes[:, None]):
            raise ValueError("Calibration photos disagree with registered words")
    generic = np.setdiff1d(np.unique(labels), classes)
    validation = np.random.default_rng(SEED).choice(generic, 120, replace=False)
    train = np.flatnonzero(np.isin(labels, np.setdiff1d(generic, validation)))
    if len(train) != ev["generic_train_photos"] or np.intersect1d(train, rows).size:
        raise ValueError("Historical generic training split changed")
    return dict(
        name=path.name,
        ev=ev,
        ablation=replay,
        rows=rows,
        truth=truth,
        folds=folds,
        words=words,
        support=support,
        train=train,
        receipt={
            "evaluation": sha(path / "evaluation.json"),
            "ablation": sha(path / "ablation.json"),
        },
    )


def packet(path, required, names, ref_times=None):
    with np.load(path, allow_pickle=False) as data:
        subject = str(data["subject"].item())
        if path.stem != subject or int(data["repeats"].item()) != 4:
            raise ValueError(f"Incorrect participant or repeat count in {path.name}")
        rows = data["rows"]
        channels = data["channels"].tolist()
        times = data["times"].astype(np.float64)
        x = data["mean_eeg"]
        if (
            rows.ndim != 1
            or not np.issubdtype(rows.dtype, np.integer)
            or len(set(rows)) != len(rows)
            or x.dtype != np.float32
            or x.shape != (len(rows), 63, 250)
            or times.shape != (250,)
            or len(channels) != 63
            or len(set(channels)) != 63
            or set(channels) != set(names)
        ):
            raise ValueError(
                f"Invalid row, channel, time or float32 waveform layout in {path.name}"
            )
        if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
            raise ValueError(f"Invalid sample times in {path.name}")
        if ref_times is not None and not np.allclose(times, ref_times, rtol=0, atol=1e-7):
            raise ValueError(f"Sample times disagree with sub-03 in {path.name}")
        lookup = {int(row): i for i, row in enumerate(rows)}
        if not all(int(r) in lookup for r in required):
            raise ValueError(f"Missing registered waveform rows in {path.name}")
        x = x[[lookup[int(r)] for r in required]][:, [channels.index(n) for n in names], :]
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite waveforms in {path.name}")
    return np.ascontiguousarray(x), times


def verify_waveforms(source, ref):
    diff = np.subtract(source, ref, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(ref, dtype=np.float64))))
    atol = 16 * np.finfo(np.float32).eps * rms
    maximum = float(np.abs(diff).max())
    if rms == 0 or not np.allclose(source, ref, atol=atol, rtol=5e-6):
        raise ValueError(
            f"sub-03 waveform mismatch: max absolute difference {maximum:g}, reference RMS {rms:g}"
        )
    return dict(
        compared_shape=list(source.shape),
        exact_equal=bool(np.array_equal(source, ref)),
        max_abs=maximum,
        rms_error=float(np.sqrt(np.mean(diff**2))),
        reference_rms=rms,
        rtol=5e-6,
        atol=float(atol),
    )


def ridge(x, y, key):
    # Match fit_official_continuous.regress, including the float32 Gram reduction.
    mu, sd, intercept = x.mean(0), x.std(0).clip(1e-5), y.mean(0)
    x = (x - mu) / sd
    gram = (x.T @ x).astype(np.float64)
    gram.flat[:: len(gram) + 1] += 0.3 * len(x)
    w = np.linalg.solve(gram, x.T @ (y - intercept)).astype(np.float32)
    return {key + "_mu": mu, key + "_sd": sd, key + "_bias": intercept, key + "_w": w}


def apply_ridge(m, x, k):
    return ((x - m[k + "_mu"]) / m[k + "_sd"]) @ m[k + "_w"] + m[k + "_bias"]


def fit_source(spec, eeg, posterior, raw_h, raw_f, t, lat):
    from scipy.linalg import eigh

    train = spec["train"]
    x = eeg_design(eeg, train, posterior)
    h, f = normalize(raw_h[train].astype(np.float32)), raw_f[train].astype(np.float32)
    model = ridge(x, np.column_stack((h, f)), "visual")
    # Historical eeg_projection relative to within-concept variation.
    mu, sd = x.mean(0), x.std(0).clip(1e-4)
    xc = ((x - mu) / sd).reshape(-1, 10, x.shape[1])
    centers = xc.mean(1)
    residual = (xc - centers[:, None]).reshape(-1, x.shape[1])
    within = residual.T @ residual / len(residual)
    within.flat[:: len(within) + 1] += 0.25
    between = centers.T @ centers / len(centers)
    vals, vecs = eigh(between, within, subset_by_index=(len(mu) - 96, len(mu) - 1))
    gain = np.sqrt(np.maximum(vals - 0.1, 0) / np.maximum(vals + 1.0, 1.0))
    axis, alpha, radius = axis_and_radius(f)
    model.update(
        proj_mu=mu,
        proj_sd=sd,
        proj_vecs=vecs,
        proj_gain=gain,
        center_h=h.mean(0),
        global_f=f.mean(0),
        axis=axis,
        alpha=alpha,
        radius=radius,
    )
    # These fold-specific centers are the only panel visual targets available at inference.
    cal = spec["support"]
    model["cal_h"] = (
        normalize(raw_h[cal].astype(np.float32).reshape(-1, 1024)).reshape(5, 12, 8, 1024).mean(2)
    )
    model["cal_f"] = raw_f[cal].astype(np.float32).mean(2)
    keep = np.flatnonzero(np.isin(t, train))
    if len(keep) == 0 or not np.array_equal(train[np.searchsorted(train, t[keep])], t[keep]):
        raise ValueError("Generic latent fitting rows disagree with source registration")
    coarse = np.empty((len(keep), 256), dtype=np.float32)
    for i in range(0, len(keep), 64):
        ix = keep[i : i + 64]
        coarse[i : i + len(ix)] = (
            lat[ix].astype(np.float32).reshape(-1, 4, 8, 8, 8, 8).mean((3, 5)).reshape(-1, 256)
        )
    model.update(ridge(x[np.searchsorted(train, t[keep])], coarse, "latent"))
    return model


def per_query_standard(scores):
    return (scores - scores.mean(1, keepdims=True)) / scores.std(1, keepdims=True).clip(1e-6)


def infer(spec, m, eeg, posterior, required):
    rows = spec["rows"]
    loc = np.searchsorted(required, rows)
    x = eeg_design(eeg, loc, posterior)
    raw = eeg[loc][:, posterior].reshape(len(rows), -1)
    projected = ((x - m["proj_mu"]) / m["proj_sd"]) @ m["proj_vecs"] * m["proj_gain"]
    hp, fp = np.split(apply_ridge(m, x, "visual"), 2, axis=1)
    starts = apply_ridge(m, x, "latent").reshape(-1, 4, 8, 8)
    local = {int(r): i for i, r in enumerate(rows)}
    predictions = {k: np.empty(len(rows), dtype=np.int64) for k in METHODS}
    condition, category = np.empty_like(fp), np.empty_like(fp)
    for fold, support in enumerate(spec["support"]):
        q = np.flatnonzero(spec["folds"] == fold)
        cal = np.array([[local[int(r)] for r in group] for group in support])
        raw_scores = normalize(raw[q]) @ normalize(raw[cal].mean(1)).T
        xp = per_query_standard(normalize(projected[q]) @ normalize(projected[cal].mean(1)).T)
        vh = per_query_standard(
            normalize(hp[q] - m["center_h"]) @ normalize(m["cal_h"][fold] - m["center_h"]).T
        )
        for k, mat in zip(METHODS, (raw_scores, xp, vh, 0.75 * xp + 0.25 * vh), strict=True):
            predictions[k][q] = mat.argmax(1)
        picked = predictions["fused"][q]
        center = m["cal_f"][fold, picked]
        residual = fp[q] - fp[cal].mean(1)[picked]
        cone = (m["global_f"], m["axis"], m["alpha"], m["radius"])
        condition[q] = restore_cone(center + residual, *cone)
        category[q] = restore_cone(center, *cone)
    values = dict(
        rows=rows,
        words=spec["words"][predictions["fused"]],
        folds=spec["folds"],
        condition=condition,
        class_only=category,
        start=starts,
    )
    if any(not np.isfinite(values[k]).all() for k in ("condition", "class_only", "start")):
        raise ValueError("Nonfinite registered condition")
    return predictions, values


def torch_views(root):
    import torch

    def array(rel):
        return torch.load(root / rel, map_location="cpu", mmap=True, weights_only=True)

    return (
        array("data/recovery_20261108_v236/sub-03_train_mean4.pt").numpy(),
        array("data/native_training_v239/train_visual_embeddings.pt")[VISUAL_KEY].numpy(),
        array("runs/HIVE_native_sub03_5096542/deterministic_F_all16540.pt").numpy(),
        np.load(
            root / "data/native_T_latents_v256/native_VAE_T_modes.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
    )


def load_cached_model(cache, signature, fit):
    path, receipt = cache / "generic_model.npz", cache / "generic_model.json"
    if receipt.is_file() and path.is_file():
        tag = read(receipt)
        if tag["signature"] == signature and tag["sha256"] == sha(path):
            with np.load(path, allow_pickle=False) as x:
                return {k: x[k] for k in x.files}
    model = fit()
    save(path, model)
    save(receipt, dict(signature=signature, sha256=sha(path)))
    return model


def run(args):
    root, dest = args.archive, args.out
    records, labels, split, posterior = registration(root)
    names = read(root / "data/recovery_20261108_v236/sub-03_layout.json")["channels"]
    concepts = [r["concept_id"].split("_", 1)[1] for r in records[::10]]
    panels = [registered_panel(p, labels, concepts) for p in args.panel_dirs]
    if (
        len(panels) != 2
        or len({p["name"] for p in panels}) != 2
        or set(panels[0]["words"]) & set(panels[1]["words"])
    ):
        raise ValueError(
            "Supply both original disjoint 12-object panels with distinct directory names"
        )
    required = np.unique(np.concatenate([p["rows"] for p in panels]))
    if len(required) != 240 or {p["ev"]["exact_hits"] for p in panels} != {85, 92}:
        raise ValueError("Expected the original 85- and 92-hit registrations and 240 unique photos")
    subjects = args.subjects or sorted(p.stem for p in args.subjects_dir.glob("sub-*.npz"))
    if not subjects or any(not re.fullmatch(r"sub-(0[1-9]|10)", s) for s in subjects):
        raise ValueError("Invalid or missing sub-XX.npz participant packets")
    keys = (
        "data/recovery_20261108_v236/sub-03_train_mean4.pt",
        "data/native_training_v239/train_visual_embeddings.pt",
        "runs/HIVE_native_sub03_5096542/deterministic_F_all16540.pt",
        "data/native_T_latents_v256/native_VAE_T_modes.npy",
        "data/recovery_20261108_v236/current_split_manifest.json",
        "data/recovery_20261108_v236/sub-03_layout.json",
        "data/full_T_recovery_20261108_v238/image_manifest.csv",
        "data/native_training_v239/train_image_ids.json",
    )
    files = {k: sha(root / k) for k in keys}
    if files[keys[0]] != EEG_SHA:
        raise ValueError("Unexpected archived sub-03 mean-four waveform digest")
    code = {
        f: sha(Path(__file__).with_name(f))
        for f in (Path(__file__).name, "decode_eeg_layout.py", "fit_photo_conditions.py")
    }
    archived, h, f, lat = torch_views(root)
    source_path = args.subjects_dir / (SOURCE + ".npz")
    src_eeg, times = packet(source_path, required, names)
    wave = verify_waveforms(src_eeg, archived[required])
    source_sha = sha(source_path)
    register = {}
    for spec in panels:
        print(f"Checking source model for {spec['name']}", flush=True)
        sig = digest(dict(archive=files, code=code, panel=spec["receipt"]))
        model = load_cached_model(
            dest / "cache" / spec["name"],
            sig,
            lambda spec=spec: fit_source(spec, archived, posterior, h, f, split["T"], lat),
        )
        predictions, values = infer(spec, model, src_eeg, posterior, required)
        for method, pred in predictions.items():
            if not np.array_equal(pred, spec["ablation"]["results"][method]["predicted"]):
                raise ValueError(f"{spec['name']} source replay disagrees on {method} predictions")
        if not np.array_equal(predictions["fused"], spec["ev"]["predicted"]):
            raise ValueError("Source fused predictions differ from registered photo order")
        register[spec["name"]] = model, sig, predictions, values
        print(
            f"{spec['name']}: reproduced {spec['ev']['exact_hits']}/120 before transfer", flush=True
        )
    save(
        dest / "source_verification.json",
        dict(
            subject=SOURCE,
            archive_sha256=files,
            code_sha256=code,
            source_packet_sha256=source_sha,
            waveform=wave,
            verified_panels={p["name"]: p["ev"]["exact_hits"] for p in panels},
        ),
    )
    for subject in dict.fromkeys([SOURCE, *subjects]):
        path = args.subjects_dir / (subject + ".npz")
        subject_sha = source_sha if subject == SOURCE else sha(path)
        target_eeg = None
        for spec in panels:
            model, sig, src_pred, src_values = register[spec["name"]]
            out = dest / spec["name"] / subject
            cond, ev = out / "conditions.npz", out / "evaluation.json"
            tag = digest([sig, source_sha, subject_sha, subject])
            if ev.exists() and cond.exists():
                old = read(ev)
                if old.get("signature") == tag and old["conditions_sha256"] == sha(cond):
                    continue
            if subject == SOURCE:
                preds, values = src_pred, src_values
            else:
                if target_eeg is None:
                    target_eeg, _ = packet(path, required, names, times)
                preds, values = infer(spec, model, target_eeg, posterior, required)
            results = {
                k: dict(predicted=v.tolist(), n=len(v), hits=int((v == spec["truth"]).sum()))
                for k, v in preds.items()
            }
            save(cond, values)
            save(
                ev,
                dict(
                    subject=subject,
                    source=SOURCE,
                    signature=tag,
                    model_signature=sig,
                    packet_sha256=subject_sha,
                    conditions_sha256=sha(cond),
                    words=spec["words"].tolist(),
                    rows=spec["rows"].tolist(),
                    folds=spec["folds"].tolist(),
                    truth=spec["truth"].tolist(),
                    predicted=preds["fused"].tolist(),
                    results=results,
                    exact_hits=results["fused"]["hits"],
                    total=120,
                    mix=0.25,
                    n_calibration_photos_per_object=8,
                    generic_train_photos=len(spec["train"]),
                    trial_count=4,
                ),
            )
            print(subject, spec["name"], {k: v["hits"] for k, v in results.items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--panel-dirs", type=Path, nargs=2, required=True)
    p.add_argument("--subjects-dir", type=Path, required=True)
    p.add_argument(
        "--subjects", nargs="+", help="Defaults to all available packets; sub-03 is always verified"
    )
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())
