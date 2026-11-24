"""Verify and aggregate native renders for the full-head and expanded-calibration readouts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PANELS = ("registered", "replicate")
CHECKPOINT = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
METHODS = {"full_head": "full_head", "expanded_r4": "expanded"}


def read(path):
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def verified_case(pred_path, cond_path, rgb_path, method):
    pred, pred_sha = read(pred_path)
    cond, cond_sha = read(cond_path)
    rgb, rgb_sha = read(rgb_path)
    require(
        cond["source_evaluation_sha256"] == pred_sha
        and cond["conditions_sha256"] == rgb["condition_sha"]
        and rgb["clip_sha"] == CHECKPOINT
        and cond["source_method"] == method,
        f"Prediction, generator or scorer receipt mismatch at {rgb_path}",
    )
    require(
        all(pred[k] == cond[k] for k in ("subject", "rows", "words", "folds", "truth"))
        and pred["rows"] == rgb["rows"]
        and len(set(pred["rows"])) == 120,
        "Condition and scorer disagree on the registered trials",
    )
    n, decisions = 120, pred["results"][method]
    words, truth, predicted = map(
        np.asarray, (pred["words"], pred["truth"], decisions["predicted"])
    )
    require(
        words.size == 12
        and predicted.shape == truth.shape == (n,)
        and np.array_equal(np.bincount(truth, minlength=12), np.full(12, 10))
        and ((predicted >= 0) & (predicted < 12)).all()
        and np.array_equal(words[predicted], rgb["predicted_words"])
        and np.array_equal(words[truth], rgb["labels"])
        and decisions["n"] == n == rgb["results"]["aligned"]["n"],
        "Renderer used different object labels from the EEG readout",
    )
    aligned = rgb["results"]["aligned"]
    scores = {"eeg_object": (truth == predicted).astype(float)}
    totals = {
        "eeg_object": decisions["hits"],
        "within_class_photo": aligned["original_photo"]["intra10_top1"],
    }
    for key, field in (("generated_full1854", "full"), ("generated_closed12", "closed")):
        scores[key] = np.asarray(aligned["per_photo_" + field])
        totals[key] = aligned[field + "_top1"]
    if "per_photo_intra10" in aligned:
        scores["within_class_photo"] = np.asarray(aligned["per_photo_intra10"])
    for key, vector in scores.items():
        require(
            vector.shape == (n,)
            and ((0 <= vector) & (vector <= 1)).all()
            and np.isclose(vector.sum(), totals[key], rtol=0, atol=1e-9),
            f"Invalid per-query credits for {key}",
        )
    hashes = dict(prediction=pred_sha, conditions_receipt=cond_sha, render=rgb_sha)
    return pred, rgb, totals, scores, hashes


def paired(old, new, truth):
    delta = new - old
    means = np.array([delta[truth == i].mean() for i in range(12)])
    draws = np.random.default_rng(260508).integers(12, size=(10000, 12))
    return {
        "credit_change": float(delta.sum()),
        "rate_change": float(delta.mean()),
        "rate_change_by_object": means.tolist(),
        "class_bootstrap_ci95": np.quantile(means[draws].mean(1), [0.025, 0.975]).tolist(),
    }


def run(a):
    cases = {}
    for sub in ("sub-04", "sub-05", "sub-01"):
        cases[sub] = {}
        for panel in PANELS:
            rel = Path(panel) / sub
            rgb_root = a.transfer / (
                "confirm_render/personal"
                if sub == "sub-01"
                else "personal_render/personal_conditions"
            )
            pref, rgb, old, ref_scores, hashes = verified_case(
                a.transfer / "personal" / rel / "evaluation.json",
                a.transfer / "personal_conditions" / rel / "evaluation.json",
                rgb_root / rel / "results.json",
                "personal_r4",
            )
            entry = {"personal_r4": {"credit": old, "sha256": hashes}}
            for method, root in METHODS.items():
                if sub == "sub-01" and method == "full_head":
                    continue
                p, r, totals, scores, hashes = verified_case(
                    a.runs / root / rel / "evaluation.json",
                    a.runs / f"{method}_conditions" / rel / "evaluation.json",
                    a.runs / f"{method}_render" / rel / "results.json",
                    method,
                )
                require(
                    all(pref[k] == p[k] for k in ("rows", "words", "folds", "truth", "subject"))
                    and all(
                        rgb["results"]["original"][k] == r["results"]["original"][k]
                        for k in ("per_photo_full", "per_photo_closed", "original_photo")
                    ),
                    "Different photographs, folds or reference features in paired runs",
                )
                if method == "expanded_r4":
                    audit = p["session_audit"]
                    require(
                        p["reference_128_sha256"] == entry["personal_r4"]["sha256"]["prediction"],
                        "128-class reference changed",
                    )
                else:
                    require(
                        p["results"]["personal_r4"] == pref["results"]["personal_r4"],
                        "Full-head baseline replay differs",
                    )
                    audit = {name: p["session_control"][name] for name in ("personal_r4", method)}
                entry[method] = dict(
                    credit=totals,
                    paired_vs_personal_r4={
                        k: paired(ref_scores[k], scores[k], np.asarray(p["truth"]))
                        for k in ref_scores
                        if k in scores
                    },
                    session_control=audit,
                    sha256=hashes,
                )
            cases[sub][panel] = entry
    report = dict(
        panels=list(PANELS),
        roles={
            "sub-04": "development",
            "sub-05": "validation",
            "sub-01": "separate expanded-arm confirmation",
        },
        photos_per_case=120,
        unique_photos=240,
        full_vocabulary_size=1854,
        within_class_gallery_size=10,
        class_bootstrap="10,000 paired draws of 12 object classes per participant and panel; rates are fractions.",
        clip_sha256=CHECKPOINT,
        cases=cases,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("outputs/photo-structure"))
    p.add_argument("--transfer", type=Path, default=Path("outputs/object-transfer"))
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())
