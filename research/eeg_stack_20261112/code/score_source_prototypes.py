"""Score registered 12-way queries with sub-03 calibration EEG for every participant.

The source prototypes use eight different photos per class. Only the query waveform
comes from the new participant. Both protocols use the same registered source visual
reference bank. Run fit_registered_transfer.py before this CPU-only comparison.
"""

import argparse
import re
from pathlib import Path

import numpy as np
from decode_eeg_layout import eeg_design, registration
from fit_photo_conditions import normalize
from fit_registered_transfer import (
    SOURCE,
    apply_ridge,
    digest,
    packet,
    per_query_standard,
    read,
    registered_panel,
    save,
    sha,
)


def transform(x, m):
    return ((x - m["proj_mu"]) / m["proj_sd"]) @ m["proj_vecs"] * m["proj_gain"]


def source_centroids(panel, m, source_eeg, required, channels):
    rows = panel["rows"]
    local = {int(r): i for i, r in enumerate(rows)}
    design = eeg_design(source_eeg, np.searchsorted(required, rows), channels)
    projected = transform(design, m)
    centers = []
    for fold, cal in enumerate(panel["support"]):
        if np.intersect1d(rows[panel["folds"] == fold], cal).size:
            raise ValueError("A query photo entered its own source EEG prototypes")
        pos = np.array([[local[int(r)] for r in c] for c in cal])
        centers.append(normalize(projected[pos].mean(1)))
    return np.stack(centers)


def score(panel, m, centers, eeg, required, channels):
    # Target-side information is one row's waveform. There is no target-side
    # support lookup, covariance, normalizer or other multi-row estimate here.
    x = eeg_design(eeg, np.searchsorted(required, panel["rows"]), channels)
    qx = normalize(transform(x, m))
    qh = normalize(apply_ridge(m, x, "visual")[:, :1024] - m["center_h"])
    preds = {k: np.empty(len(x), dtype=np.int64) for k in ("projected_eeg", "fused")}
    for fold in range(len(centers)):
        q = np.flatnonzero(panel["folds"] == fold)
        xp = per_query_standard(qx[q] @ centers[fold].T)
        hp = per_query_standard(qh[q] @ normalize(m["cal_h"][fold] - m["center_h"]).T)
        preds["projected_eeg"][q] = xp.argmax(1)
        preds["fused"][q] = (0.75 * xp + 0.25 * hp).argmax(1)
    return preds


def comparison(calibrated, source_only, truth):
    a, b = calibrated == truth, source_only == truth
    diff = a.astype(float) - b
    classes = np.unique(truth)
    groups = np.array([diff[truth == c].mean() for c in classes])
    draws = np.random.default_rng(260427).integers(len(groups), size=(10000, len(groups)))
    return dict(
        n=len(truth),
        target_calibrated_hits=int(a.sum()),
        source_prototype_hits=int(b.sum()),
        target_only=int(np.sum(a & ~b)),
        source_only=int(np.sum(b & ~a)),
        target_minus_source_fraction=float(diff.mean()),
        paired_concept_ci95=np.quantile(groups[draws].mean(1), [0.025, 0.975]).tolist(),
    )


def run(args):
    if args.out.resolve() == args.transfer.resolve():
        raise ValueError("Use a separate output directory to preserve the eight-shot comparison")
    verification = read(args.transfer / "source_verification.json")
    root = args.archive
    rec, labels, _, channels = registration(root)
    layout_path = root / "data/recovery_20261108_v236/sub-03_layout.json"
    if (
        sha(layout_path)
        != verification["archive_sha256"]["data/recovery_20261108_v236/sub-03_layout.json"]
    ):
        raise ValueError("Archive channel layout differs from the gated transfer fit")
    names = read(layout_path)["channels"]
    classes = [r["concept_id"].split("_", 1)[1] for r in rec[::10]]
    panels = [registered_panel(p, labels, classes) for p in args.panel_dirs]
    required = np.unique(np.concatenate([p["rows"] for p in panels]))
    if (
        len(required) != 240
        or verification["verified_panels"] != {p["name"]: p["ev"]["exact_hits"] for p in panels}
        or set(verification["verified_panels"].values()) != {85, 92}
    ):
        raise ValueError("Both historically verified, disjoint source registrations are required")
    source_path = args.subjects_dir / (SOURCE + ".npz")
    if sha(source_path) != verification["source_packet_sha256"]:
        raise ValueError("Source packet differs from the archive waveform verification")
    source, times = packet(source_path, required, names)
    models = {}
    for p in panels:
        cache = args.transfer / "cache" / p["name"]
        receipt = read(cache / "generic_model.json")
        sig = digest(
            dict(
                archive=verification["archive_sha256"],
                code=verification["code_sha256"],
                panel=p["receipt"],
            )
        )
        if receipt["signature"] != sig or receipt["sha256"] != sha(cache / "generic_model.npz"):
            raise ValueError("Incompatible generic model or source registration")
        with np.load(cache / "generic_model.npz", allow_pickle=False) as pack:
            m = {
                k: pack[k]
                for k in (
                    "proj_mu",
                    "proj_sd",
                    "proj_vecs",
                    "proj_gain",
                    "visual_mu",
                    "visual_sd",
                    "visual_w",
                    "visual_bias",
                    "center_h",
                    "cal_h",
                )
            }
        centers = source_centroids(p, m, source, required, channels)
        replay = score(p, m, centers, source, required, channels)
        for k, pred in replay.items():
            if not np.array_equal(pred, p["ablation"]["results"][k]["predicted"]):
                raise ValueError(f"Source prototype replay disagrees for {p['name']}/{k}")
        models[p["name"]] = m, centers, sig
        print(
            f"Verified source prototypes for {p['name']}: {p['ev']['exact_hits']}/120", flush=True
        )
    subjects = args.subjects or sorted(path.stem for path in args.subjects_dir.glob("sub-*.npz"))
    for s in dict.fromkeys([SOURCE, *subjects]):
        if not re.fullmatch(r"sub-(0[1-9]|10)", s):
            raise ValueError("Invalid participant identifier")
        path = args.subjects_dir / (s + ".npz")
        target = source if s == SOURCE else packet(path, required, names, times)[0]
        packet_sha = sha(path)
        for p in panels:
            m, centers, sig = models[p["name"]]
            preds = score(p, m, centers, target, required, channels)
            truth = p["truth"]
            result = dict(
                subject=s,
                source=SOURCE,
                role="source_replay" if s == SOURCE else "held_out_participant",
                target_labeled_eeg_calibration_rows_per_fold=96 if s == SOURCE else 0,
                source_eeg_calibration_photos_per_class=8,
                visual_reference_photos_per_class=8,
                source_support_disjoint_from_query_photos=True,
                transfer_model_signature=sig,
                target_packet_sha256=packet_sha,
                words=p["ev"]["words"],
                rows=p["ev"]["rows"],
                folds=p["ev"]["folds"],
                truth=truth.tolist(),
                results={
                    k: dict(predicted=v.tolist(), n=len(v), hits=int((v == truth).sum()))
                    for k, v in preds.items()
                },
            )
            fit_result = args.transfer / p["name"] / s / "evaluation.json"
            if fit_result.is_file():
                other = read(fit_result)
                if (
                    other["model_signature"] != sig
                    or other["packet_sha256"] != packet_sha
                    or any(other[k] != result[k] for k in ("truth", "rows", "folds", "words"))
                ):
                    raise ValueError(
                        "Target-calibrated comparison does not use the same data and frozen model"
                    )
                result["eight_target_calibration_minus_source_prototypes"] = {
                    k: comparison(np.array(other["results"][k]["predicted"]), v, truth)
                    for k, v in preds.items()
                }
            save(args.out / p["name"] / s / "evaluation.json", result)
            print(s, p["name"], {k: int((v == truth).sum()) for k, v in preds.items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("archive", "subjects-dir", "transfer", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--panel-dirs", nargs=2, type=Path, required=True)
    p.add_argument("--subjects", nargs="+")
    run(p.parse_args())
