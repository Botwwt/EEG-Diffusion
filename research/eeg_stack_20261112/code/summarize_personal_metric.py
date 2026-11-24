"""Summarize paired registered-object recognition and generated-image scores for sub-04/05."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_registered_sessions import audit_panel

SUBJECTS = {"sub-04": "development", "sub-05": "confirmation"}
PANELS = ("registered", "replicate")
COMPARISONS = {"personal_r4": "r4_fused", "cross_photo_session": "crossed_fused"}
ARM_NAMES = ("category", "aligned", "shuffled", "original")
SEED = 260432


def read(path, root, hashes):
    raw = path.read_bytes()
    hashes[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def paired(a, b, truth):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not (a.shape == b.shape == truth.shape and np.isfinite([a, b]).all()):
        raise ValueError("Invalid aligned scores")
    delta = a - b
    by_class = np.array([delta[truth == k].mean() for k in np.unique(truth)])
    boot = np.random.default_rng(SEED).integers(len(by_class), size=(10000, len(by_class)))
    return dict(
        n=len(a),
        personal_credit=float(a.sum()),
        baseline_credit=float(b.sum()),
        improved_rows=int((delta > 0).sum()),
        worsened_rows=int((delta < 0).sum()),
        equal_rows=int((delta == 0).sum()),
        delta_credit=float(delta.sum()),
        delta_accuracy=float(delta.mean()),
        class_bootstrap_ci95=np.quantile(by_class[boot].mean(1), [0.025, 0.975]).tolist(),
        delta_accuracy_by_word=by_class.tolist(),
    )


def recognition(root, panel, sub, sessions, hashes):
    inputs = [
        read(root / branch / panel / sub / "evaluation.json", root, hashes)
        for branch in ("personal", "cross_session", "transfer")
    ]
    personal, cross, transfer = inputs
    control = read(root / "personal_mechanism" / panel / sub / "evaluation.json", root, hashes)
    if any(
        control[k] != personal[k]
        for k in (
            "rows",
            "truth",
            "folds",
            "words",
            "subject",
            "generic_sha256",
            "target_sha256",
            "generic_classes",
        )
    ) or any(control["results"][k] != personal["results"][k] for k in COMPARISONS):
        raise ValueError("Mechanism replay changed a frozen input or original prediction")
    if any(
        d[k] != personal[k]
        for d in inputs[1:]
        for k in ("words", "rows", "folds", "truth", "subject")
    ):
        raise ValueError("Classification rows, objects, participant or folds differ")
    if (
        personal["subject"] != sub
        or personal["target_sha256"] != cross["half_packet_sha256"]
        or cross["full_packet_sha256"] != transfer["packet_sha256"]
        or cross["model_signature"] != transfer["model_signature"]
        or cross["session_plan_sha256"] != hashes["sessions.json"]
        or cross["results"]["r4_fused"]["predicted"] != transfer["results"]["fused"]["predicted"]
    ):
        raise ValueError("Classification inputs differ from the frozen comparison")
    if np.intersect1d(np.asarray(personal["rows"]) // 10, personal["generic_classes"]).size:
        raise ValueError("A registered object occurs in generic training")
    if len(set(personal["generic_classes"])) != 128:
        raise ValueError("Expected 128 distinct generic training concepts")
    truth = np.asarray(personal["truth"])
    for name in ("class_means", "same_photo_across_sessions"):
        result = control["results"][name]
        preds = np.asarray(result["predicted"])
        if (
            preds.shape != truth.shape
            or not np.isin(preds, np.arange(12)).all()
            or result["hits"] != int((preds == truth).sum())
            or result["n"] != len(truth)
        ):
            raise ValueError(f"Invalid mechanism control predictions for {name}")
    record = {
        **personal,
        "results": {
            **personal["results"],
            **{k: cross["results"][k] for k in COMPARISONS.values()},
        },
    }
    audit = audit_panel(sessions["subjects"][sub], record)
    pair = np.asarray(sessions["subjects"][sub]["session_pairs"])[
        np.asarray(personal["rows"]) // 10
    ]
    _, group = np.unique(pair, axis=0, return_inverse=True)
    concept_pair = np.array([np.unique(group[truth == k]).item() for k in range(12)])
    counts = np.bincount(concept_pair)
    decisions = {}
    for candidate, base in COMPARISONS.items():
        sets = [np.asarray(record["results"][name]["predicted"]) for name in (candidate, base)]
        hits = [p == truth for p in sets]
        expected = [(concept_pair[p] == group) / counts[group] for p in sets]
        comparison = paired(*hits, truth)
        comparison.update(
            compared_to=base,
            session_conditional={
                name: {
                    key: audit["methods"][name][key]
                    for key in (
                        "correct",
                        "conditional_expected_hits",
                        "accuracy_minus_conditional_expectation",
                        "permutation_null_hits_ci95",
                        "permutation_one_sided_p",
                        "exact_permutation_count",
                    )
                }
                for name in (candidate, base)
            },
            paired_excess_over_conditional_chance=paired(
                hits[0] - expected[0], hits[1] - expected[1], truth
            ),
        )
        decisions[candidate] = comparison
    return (
        record,
        transfer,
        dict(
            role=SUBJECTS[sub],
            words=personal["words"],
            generic_train_concepts=128,
            generic_sha256=personal["generic_sha256"],
            session_halves_sha256=personal["target_sha256"],
            session_only_oracle_expected_hits=audit["pair_only_oracle_expected_hits"],
            comparisons=decisions,
            post_hoc_covariance_controls={
                "comparison": "Candidate is cross_photo_session; same training rows, 60-460 ms EEG features, rank, whitening and cross-session support/query scoring in all three arms.",
                "cross_photo_session_minus": {
                    k: paired(
                        np.asarray(personal["results"]["cross_photo_session"]["predicted"])
                        == truth,
                        np.asarray(control["results"][k]["predicted"]) == truth,
                        truth,
                    )
                    for k in ("class_means", "same_photo_across_sessions")
                },
            },
        ),
    )


def compact_layout(root, panel, sub, renders, paths, hashes):
    path = root / "personal_gallery" / panel / sub / "photo_layout.json"
    layout = read(path, root, hashes)
    for name, render, source in zip(("personal", "baseline"), renders, paths, strict=True):
        if (
            layout[name]["results_sha256"] != hashes[source.relative_to(root).as_posix()]
            or layout[name]["condition_sha256"] != render["condition_sha"]
        ):
            raise ValueError(f"Layout metrics came from different {name} images")
    new, old = renders
    indexed = {r["row"]: r for r in layout["per_photo"]}
    if len(indexed) != layout["n"] or sorted(indexed) != sorted(new["rows"]):
        raise ValueError("Layout report and renders contain different stimuli")
    rows = [indexed[r] for r in new["rows"]]
    truth, labels, baseline = (
        np.asarray([r[k] for r in rows]) for k in ("truth", "personal_word", "baseline_word")
    )
    if (
        truth.tolist() != new["labels"]
        or labels.tolist() != new["predicted_words"]
        or baseline.tolist() != old["predicted_words"]
    ):
        raise ValueError("Layout subgroup identity disagrees with frozen render labels")
    same = labels == baseline
    n_exact = sum(
        r["jpeg_sha256"]["baseline_aligned"] == r["jpeg_sha256"]["personal_aligned"]
        for i, r in enumerate(rows)
        if same[i]
    )
    unchanged = layout["unchanged_word_aligned_jpegs"]
    if (
        unchanged["n"] != int(same.sum())
        or unchanged["exact_jpeg_bytes"] != n_exact
        or not n_exact <= unchanged["exact_decoded_rgb"] <= int(same.sum())
    ):
        raise ValueError("Unchanged-word JPEG counts differ from archived byte digests")
    subgroups = {}
    keys = ("rgb_rmse32", "luminance_layout_corr32")
    names = ("baseline_aligned", "personal_aligned", "personal_category", "personal_shuffled")
    values = {a: np.array([[r["metrics"][a][k] for k in keys] for r in rows]) for a in names}
    for name, mask in (
        ("all", np.ones(len(rows), bool)),
        ("personal_word_correct", labels == truth),
        ("both_words_correct", (labels == truth) & (baseline == truth)),
    ):
        reported = layout["subsets"][name]
        if (
            int(mask.sum()) != reported["n"]
            or len(np.unique(truth[mask])) != reported["n_represented_classes"]
        ):
            raise ValueError("Subgroup sizes differ from their frozen word decisions")
        for arm in names:
            if not np.allclose(
                values[arm][mask].mean(0),
                [reported["mean"][arm][k] for k in keys],
                atol=1e-12,
                rtol=0,
            ):
                raise ValueError("Subgroup means differ from the row scores")
        delta = (values["personal_aligned"] - values["baseline_aligned"])[mask]
        groups = [truth[mask] == k for k in np.unique(truth[mask])]
        counts = np.array([g.sum() for g in groups])
        sums = np.stack([delta[g].sum(0) for g in groups])
        draws = np.random.default_rng(923763).integers(len(groups), size=(10000, len(groups)))
        ci = np.quantile(sums[draws].sum(1) / counts[draws].sum(1)[:, None], [0.025, 0.975], axis=0)
        contrast = reported["personal_aligned_minus"]["baseline_aligned"]
        for i, k in enumerate(keys):
            if not np.allclose(
                [delta[:, i].mean(), *ci[:, i]],
                [contrast[k]["delta"], *contrast[k]["ci95"]],
                atol=1e-12,
                rtol=0,
            ):
                raise ValueError("Layout paired interval differs from the source report")
        subgroups[name] = dict(
            n=reported["n"],
            n_represented_classes=len(groups),
            means=reported["mean"],
            personal_minus_transfer_aligned=contrast,
        )
    return dict(
        photo_layout_sha256=hashes[path.relative_to(root).as_posix()],
        photo_manifest_sha256=layout["manifest_sha256"],
        producing_scorer_sha256=layout["source_sha256"],
        subgroups=subgroups,
        unchanged_word_jpegs={
            k: unchanged[k] for k in ("n", "exact_jpeg_bytes", "exact_decoded_rgb")
        },
    )


def render_eval(root, panel, sub, record, transfer, hashes):
    truth = np.asarray(record["truth"])
    words = np.asarray(record["words"])
    folders = root / "personal_render"
    personal_path = folders / "personal_conditions" / panel / sub / "results.json"
    base_path = folders / "transfer" / panel / sub / "results.json"
    if sub == "sub-04" and not base_path.exists():
        base_path = root / f"sub04_{panel}_render.json"
    a, b = [read(path, root, hashes) for path in (personal_path, base_path)]
    if (
        a["clip_sha"] != b["clip_sha"]
        or b["condition_sha"] != transfer["conditions_sha256"]
        or any(d["rows"] != record["rows"] or d["labels"] != words[truth].tolist() for d in (a, b))
        or b["predicted_words"] != words[transfer["results"]["fused"]["predicted"]].tolist()
    ):
        raise ValueError(
            "Render scorer, source condition, rows, labels or baseline predictions differ"
        )
    matches = [
        m
        for m in COMPARISONS
        if a["predicted_words"] == words[record["results"][m]["predicted"]].tolist()
    ]
    if len(matches) != 1:
        raise ValueError("Rendered personal labels do not uniquely match a frozen method")
    if a["results"]["original"] != b["results"]["original"]:
        raise ValueError("Rendered comparisons use different original reference scores")
    result = {}
    for arm in ARM_NAMES:
        x, y = a["results"][arm], b["results"][arm]
        if x["n"] != 120 or y["n"] != 120:
            raise ValueError("Expected 120 paired generated outputs")
        obj = {}
        for key in ("full", "closed"):
            va, vb = (np.asarray(z["per_photo_" + key]) for z in (x, y))
            if (
                not np.isclose(va.sum(), x[key + "_top1"])
                or not np.isclose(vb.sum(), y[key + "_top1"])
                or np.any(va < 0)
                or np.any(va > 1)
                or np.any(vb < 0)
                or np.any(vb > 1)
            ):
                raise ValueError("Saved semantic credits disagree with row sums")
            obj[key + "_top1"] = paired(va, vb, truth)
        obj["reference_gallery_hits"] = {
            k: dict(
                personal=float(v),
                baseline=float(y["original_photo"][k]),
                difference=float(v - y["original_photo"][k]),
            )
            for k, v in x["original_photo"].items()
        }
        result[arm] = obj
    return dict(
        personal_word_method=matches[0],
        baseline_word_method="transfer/fused",
        personal_condition_sha256=a["condition_sha"],
        transfer_condition_sha256=b["condition_sha"],
        scorer_sha256=a["clip_sha"],
        low_frequency_layout=compact_layout(
            root, panel, sub, (a, b), (personal_path, base_path), hashes
        ),
        arms=result,
    )


def summarize(root):
    hashes = {}
    sessions = read(root / "sessions.json", root, hashes)
    subjects = {}
    for sub, role in SUBJECTS.items():
        reports = {}
        query_classes = []
        for panel in PANELS:
            rec, trans, out = recognition(root, panel, sub, sessions, hashes)
            out["generation"] = render_eval(root, panel, sub, rec, trans, hashes)
            reports[panel] = out
            query_classes.extend(np.unique(np.asarray(rec["rows"]) // 10))
            if panel == PANELS[0]:
                generic = rec["generic_classes"]
            elif generic != rec["generic_classes"]:
                raise ValueError("Personal training set differs between panels")
        if len(set(query_classes)) != 24 or np.intersect1d(generic, query_classes).size:
            raise ValueError("Training set or panel exclusion failed")
        subjects[sub] = dict(role=role, panels=reports)
    return dict(
        design="128 other concepts (1,280 images) per participant; eight labeled calibration photos per registered class and 120 four-repeat queries per panel.",
        interpretation="Sub-04 development and sub-05 single-participant confirmation are reported separately.",
        interval="Paired 95% percentile bootstrap over the twelve objects per participant and panel, keeping ten photos together. Session-pair expectations are fixed at their observed panel values. Intervals describe variability over these objects.",
        bootstrap_seed=SEED,
        bootstrap_draws=10000,
        conditional_null="Whole-category permutations within each participant's scheduled two-session group; positive excess means more hits than this conditional expectation.",
        render_scoring="CLIP-L/14. Closed is 12 prompt categories, full is the 1,854-name vocabulary. Original photo retrieval exports contain aggregate hits only.",
        layout_scoring="32 x 32 RGB RMSE (lower is better) and spatial luminance correlation (higher is better) against original RGB images. Three descriptive subgroups use exact EEG word predictions. Paired intervals resample represented true object classes 10,000 times, retain eligible photos, and weight by photo count.",
        source_files_sha256=hashes,
        code_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "fit_personal_session_metric.py",
                "audit_registered_sessions.py",
                "summarize_personal_metric.py",
            )
        },
        subjects=subjects,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/object-transfer"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("research/eeg_stack_20261112/results/object_fidelity/personal_metric.json"),
    )
    args = parser.parse_args()
    result = summarize(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".partial")
    tmp.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n"
    )
    tmp.replace(args.out)
