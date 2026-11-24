"""Publish aggregate RGB and CLIP-L/14 results for four frozen photo-residual evaluations."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from refine_personal_photo_residual import ARMS, KEYS, scores
from score_official_objects import RESIDUAL_ARMS, personal_delta

SUBJECTS = ("sub-04", "sub-05")
PANELS = ("registered", "replicate")
NAMING = ("closed12", "full1854")
CONTRASTS = (("corrected", "zero"), ("zero", "frozen_reencoded"), ("corrected", "frozen_reencoded"))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def same_numeric(actual, expected):
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            same_numeric(v, expected[k]) for k, v in actual.items()
        )
    if actual is None or expected is None:
        return actual is expected
    return np.allclose(actual, expected, rtol=1e-9, atol=1e-12)


def case(cfg, report, model, naming, table):
    rows, words, truth, fold, donors = (
        np.asarray(cfg[k])
        for k in ("rows", "predicted_words", "true_labels", "folds", "donor_rows")
    )
    pos = {row: i for i, row in enumerate(rows)}
    require(
        len(rows) == len(pos) == 120 and set(donors) == set(rows) and len(set(truth)) == 12,
        "Wrong row or object set",
    )
    donor = np.array([pos[r] for r in donors])
    swapped = donors != rows
    same = swapped & (truth[donor] == truth)
    right = truth == words
    require(
        np.array_equal(words[donor], words)
        and np.array_equal(fold[donor], fold)
        and set(fold) == set(range(5)),
        "Donor crosses word/fold",
    )
    require(
        all(np.sum((truth == name) & (fold == f)) == 2 for name in set(truth) for f in range(5)),
        "Unbalanced test fold",
    )
    require(
        all(
            bool(swapped[i]) == (np.sum((words == w) & (fold == f)) > 1)
            for i, (w, f) in enumerate(zip(words, fold, strict=True))
        ),
        "Singleton donor mismatch",
    )
    require(
        cfg["model_sha256"] == model["model_sha256"]
        and cfg["manifest_sha256"] == model["manifest_sha256"]
        and cfg["gain"] == model["gain"] == report["gain"]
        and cfg["strength"] == report["strength"]
        and naming["source_results_sha256"] == cfg["source_results_sha256"]
        and naming["source_condition_sha256"] == cfg["source_condition_sha256"],
        "Input, gain or source render mismatch",
    )
    require(
        report["metric_keys"] == list(KEYS)
        and set(report["per_photo"]) == set(ARMS)
        and set(cfg["jpeg_sha256"]) == set(ARMS)
        and cfg["source_jpeg_sha256"] == cfg["jpeg_sha256"]["frozen"]
        and len(report["original_sha256"]) == 120
        and all(len(v) == 120 for v in cfg["jpeg_sha256"].values()),
        "Missing photo checksums, arm or metric",
    )
    require(
        all(
            cfg["jpeg_sha256"]["corrected"][i] == cfg["jpeg_sha256"]["swapped"][i]
            for i in np.flatnonzero(~swapped)
        ),
        "Self-donor changed the JPEG",
    )
    values = {k: np.asarray(v, float) for k, v in report["per_photo"].items()}
    require(
        all(v.shape == (120, len(KEYS)) and np.isfinite(v).all() for v in values.values()),
        "Wrong RGB score shape",
    )
    require(len(table) == 120, "Wrong CLIP CSV denominator")
    for i, r in enumerate(table):
        require(
            (
                int(r["index"]),
                int(r["row"]),
                r["truth"],
                r["predicted_word"],
                int(r["fold"]),
                int(r["donor_row"]),
                r["eligible_swap"].lower(),
            )
            == (i, rows[i], truth[i], words[i], fold[i], donors[i], str(bool(swapped[i])).lower()),
            "CSV and frozen settings differ",
        )
    credits = {}
    for name, arm in RESIDUAL_ARMS.items():
        mask = swapped if arm == "swapped" else np.ones(120, bool)
        require(naming["methods"][name]["n"] == int(mask.sum()), "CLIP arm denominator mismatch")
        credits[arm] = {}
        for task in NAMING:
            a = np.array([float(r[f"{name}_{task}_credit"]) for r in table])
            require(
                np.all((0 <= a) & (a <= 1))
                and np.isclose(a[mask].sum(), naming["methods"][name][task]["top1_credit"]),
                "Invalid CLIP credit totals",
            )
            credits[arm][task] = a
    masks = {
        "all": np.ones(120, bool),
        "predicted_object_correct": right,
        "swappable": swapped,
        "same_truth_swappable": same,
        "correct_and_same_truth_swappable": same & right,
    }
    compact = {}
    for key, mask in masks.items():
        source = report["subsets"][key]
        require(
            same_numeric(scores(values, truth, mask), source),
            f"Original RGB row-level replay differs in {key}",
        )
        if key not in ("all", "same_truth_swappable", "predicted_object_correct"):
            continue
        contrasts = {
            "all": CONTRASTS,
            "same_truth_swappable": (("corrected", "swapped"),),
            "predicted_object_correct": (("corrected", "zero"),),
        }[key]
        arms = sorted({arm for pair in contrasts for arm in pair})
        clip = {arm: {t: float(credits[arm][t][mask].sum()) for t in NAMING} for arm in arms}
        effects = {}
        for a, b in contrasts:
            label = f"{a}_minus_{b}"
            measures = {}
            for task in NAMING:
                d = personal_delta(credits[a][task][mask], credits[b][task][mask], truth[mask])
                measures[task] = {
                    k: d[k]
                    for k in ("n", "delta_credit", "delta_rate", "delta_rate_ci95")
                    if k in d
                }
            effects[label] = dict(rgb=source["paired"][label], naming=measures)
        compact[key] = dict(
            n=source["n"],
            n_true_classes=source["n_true_classes"],
            rgb_means={arm: source["means"][arm] for arm in arms},
            naming_top1_credit=clip,
            paired=effects,
        )
    return dict(
        n=120,
        subsets=compact,
        subgroup_counts={k: int(v.sum()) for k, v in masks.items()},
        model_sha256=cfg["model_sha256"],
        target_packet_sha256=cfg["target_packet_sha256"],
        source_condition_sha256=cfg["source_condition_sha256"],
        frozen_render_results_sha256=cfg["source_results_sha256"],
        strength=cfg["strength"],
        jpeg_quality=cfg["jpeg_quality"],
    )


def summarize(root):
    hashes = {}

    def read(name):
        raw = (root / name).read_bytes()
        hashes[name] = sha(raw)
        return json.loads(raw)

    naming = read("residual_naming/naming.json")
    require(
        naming["arm_directories"] == RESIDUAL_ARMS and naming["model"] == "OpenAI CLIP-L/14",
        "Incompatible L/14 scorer",
    )
    csvpath = "residual_naming/per_row.csv"
    hashes[csvpath] = sha((root / csvpath).read_bytes())
    with (root / csvpath).open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    require(
        len(records) == 480
        and set(naming["cases"]) == {f"{p}/{s}" for p in PANELS for s in SUBJECTS},
        "Missing CLIP case",
    )
    models, panels = {}, {}
    alignment = {}
    for sub in SUBJECTS:
        model = read(f"photo_residual/models/{sub}.json")
        require(
            model["subject"] == sub
            and model["held_photos"] == 160
            and len(model["training_photo_hashes"]) == 1280,
            "Wrong model provenance",
        )
        models[sub] = {
            k: model[k]
            for k in (
                "model_sha256",
                "packet_sha256",
                "manifest_sha256",
                "grid",
                "ridge",
                "gain",
                "held_photos",
                "holdout_zero_rmse",
                "holdout_scaled_rmse",
            )
        }
        panels[sub] = {}
        for panel in PANELS:
            prefix = f"photo_residual/{panel}/{sub}/"
            cfg, report = read(prefix + "settings.json"), read(prefix + "report.json")
            named = naming["cases"][f"{panel}/{sub}"]
            require(
                hashes[prefix + "settings.json"]
                == report["settings_sha256"]
                == named["settings_sha256"]
                and cfg["panel"] == report["panel"] == panel
                and cfg["subject"] == report["subject"] == sub,
                "Case identity mismatch",
            )
            signature = (cfg["rows"], cfg["true_labels"], cfg["folds"])
            require(
                alignment.setdefault(panel, signature) == signature,
                "Registration across participants differs",
            )
            table = sorted(
                (r for r in records if (r["panel"], r["subject"]) == (panel, sub)),
                key=lambda r: int(r["index"]),
            )
            panels[sub][panel] = case(cfg, report, model, named, table)
            prior_path = f"personal_render/personal_conditions/{panel}/{sub}/results.json"
            if (root / prior_path).exists():
                prior = read(prior_path)
                require(
                    hashes[prior_path] == cfg["source_results_sha256"]
                    and prior["rows"] == cfg["rows"]
                    and prior["labels"] == cfg["true_labels"]
                    and prior["predicted_words"] == cfg["predicted_words"],
                    "Original render scores differ from frozen case",
                )
                arm = prior["results"]["aligned"]
                require(arm["n"] == 120, "Incomplete original render naming")
                panels[sub][panel]["original_render_naming"] = dict(
                    n=arm["n"],
                    full1854=arm["full_top1"],
                    closed12=arm["closed_top1"],
                    checkpoint_sha256=prior["clip_sha"],
                )
    require(
        set(alignment["registered"][0]).isdisjoint(alignment["replicate"][0]), "Overlapping panels"
    )
    all_cases = [case_result for subject in panels.values() for case_result in subject.values()]

    def rgb_intervals(group, comparison):
        return [c["subsets"][group]["paired"][comparison]["rgb"] for c in all_cases]

    def spans_zero(interval):
        return interval[0] <= 0 <= interval[1]

    all_rmse = rgb_intervals("all", "corrected_minus_zero")
    correct_rmse = rgb_intervals("predicted_object_correct", "corrected_minus_zero")
    swapped_rgb = rgb_intervals("same_truth_swappable", "corrected_minus_swapped")
    return dict(
        participants={"sub-04": "development", "sub-05": "single-participant confirmation"},
        protocol="One frozen personal render per query (K=1), 12 registered objects with 10 held-out photographs each. Both subjects saw the same photographs.",
        arms={
            "frozen_reencoded": "Frozen aligned JPEG resaved at matching JPEG settings",
            "zero": "Move the low-frequency colors toward the predicted object's mean from eight calibration photos",
            "corrected": "Add the EEG-derived within-object RGB residual to the same calibration mean",
            "swapped": "Use another query's residual within the same fold and predicted word. The reported within-object subset also requires identical true object and distinct donor.",
        },
        metrics="At 32x32, lower mean_rgb_mae and rgb_rmse32 and higher luminance_layout_corr32 indicate closer agreement with the original. Naming uses tie-split Top-1 credit for the true object.",
        uncertainty="Paired 95% percentile intervals across 10,000 resamples of represented true objects, weighting by eligible photo count. These are descriptive intervals within each subject and panel.",
        descriptive_case_counts=dict(
            total_subject_panel_combinations=len(all_cases),
            corrected_vs_zero_all_photos_rgb_rmse_ci_below_zero=sum(
                x["rgb_rmse32"]["ci95"][1] < 0 for x in all_rmse
            ),
            corrected_vs_zero_correct_word_rgb_rmse_ci_includes_zero=sum(
                spans_zero(x["rgb_rmse32"]["ci95"]) for x in correct_rmse
            ),
            corrected_vs_same_truth_word_fold_donor_all_three_color_intervals_include_zero=sum(
                all(spans_zero(x[k]["ci95"]) for k in KEYS) for x in swapped_rgb
            ),
        ),
        gain_calibration="The 16 held generic concepts choose the scalar gain. Their scaled RMSE is an in-calibration estimate; final ridge uses all 128 generic concepts.",
        l14=dict(
            model=naming["model"],
            checkpoint_sha256=naming["checkpoint_sha256"],
            dictionary_sha256=naming["dictionary_sha256"],
            templates=naming["templates"],
        ),
        models=models,
        subjects=panels,
        source_sha256=hashes,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    published = summarize(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".partial")
    tmp.write_text(
        json.dumps(published, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n"
    )
    tmp.replace(args.out)
