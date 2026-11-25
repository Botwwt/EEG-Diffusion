"""Attach hash-bound official Inception results on each comparison's available participants."""

from pathlib import Path

import numpy as np
import paper_aggregate as agg
import paper_inception_delta as scorer


def read_export(root, index, primary_index):
    evidence, scores = {}, {}
    for rel, expected in sorted(index.items()):
        folder = (root / rel).resolve()
        agg.require(folder.is_relative_to(root.resolve()) and rel in primary_index, f"Unregistered Inception case {rel}")
        summary, data = scorer.read_sidecar(folder)
        receipts = {"scores_sha256": scorer.file_sha(folder / f"{scorer.STEM}.json"),
                    "observations_sha256": summary["observations_sha256"], "n": summary["n"]}
        agg.require(all(expected[k] == value for k, value in receipts.items())
                    and primary_index[rel]["scores_sha256"] == summary["source_sha256"]["scores.json"]
                    and primary_index[rel]["observations_sha256"] == summary["source_sha256"]["observations.npz"],
                    f"Inception sidecar differs from the committed export at {rel}")
        evidence[rel] = {**receipts, "panel": summary["panel"], "scorer_sha256": summary["scorer_sha256"],
                         "source_scores_sha256": summary["source_sha256"]["scores.json"],
                         "method": summary["method"], "gallery_sha256": summary["gallery_sha256"]}
        order = np.argsort(data["row"])
        scores[rel] = (data["row"][order], data["concept"][order], data[scorer.METRIC][order, None])
    return evidence, scores


def add_stats(record, left, right, scores, panel, draws, seed):
    raw_people = list(left) if right is None else [p for p in left if p in right]
    people = [p for p in raw_people if left[p] in scores and (right is None or right[p] in scores)]
    n = agg.PANELS[panel][0]
    details = {"participants": people, "missing_among_primary": [p for p in raw_people if p not in people],
               "scored_images" if right is None else "scored_image_pairs": len(people) * n}
    calibration = "per_participant" not in record
    if calibration:
        details.update(participants=[], missing_among_primary=[], generator_only=True, scored_galleries=len(people))
    record["inception_official"] = details
    if not people:
        return
    rows, labels, _ = scores[left[people[0]]]
    for arm in ([left] if right is None else [left, right]):
        for p in people:
            row, concept, _ = scores[arm[p]]
            agg.require(len(row) == n and np.array_equal(row, rows) and np.array_equal(concept, labels),
                        f"Inception stimuli or concepts differ between participants or arms for {panel}")
    a = np.stack([scores[left[p]][2] for p in people])
    b = np.stack([scores[right[p]][2] for p in people]) if right is not None else None
    # The aggregate's two-way statistic already has the strict n-1 credit denominator.
    metric = "inception_2way"
    measurements, conditional = agg.describe(a, b, people, labels, panel, [metric], draws, seed)
    for stat in [measurements[metric], *(conditional[p][metric] for p in people)]:
        for key in ("credit", "credit_a", "credit_b"):
            if key in stat:
                stat[key] = round(stat[key])
    record["metrics"][scorer.METRIC] = measurements[metric]
    if not calibration:
        for p in people:
            record["per_participant"][p][scorer.METRIC] = conditional[p][metric]
    else:  # No EEG participant for the generator calibrations.
        stat = record["metrics"][scorer.METRIC]
        stat.pop("crossed_ci95")
        stat["stimulus_ci95"] = conditional["gallery"][metric]["stimulus_ci95"]


def add_matrix(matrix, scores, panel, draws, seed):
    methods = {}
    for method, arm in matrix["arms"].items():
        methods[method] = {p: arm["sources"][p]["folder"] for p in arm["participants"]}
        add_stats(arm, methods[method], None, scores, panel, draws, seed)
    for pair in matrix["comparisons"]:
        a, b = pair["arm_a"], pair["arm_b"]
        left = {p: methods[a][p] for p in pair["participants"]}
        right = {p: methods[b][p] for p in pair["participants"]}
        add_stats(pair, left, right, scores, panel, draws, seed)


def attach(root, index, primary_index, matrices, extra, draws, seed, object_studies=None):
    evidence, scores = read_export(Path(root), index, primary_index)
    for matrix in matrices.values():
        add_matrix(matrix, scores, matrix["panel"], draws, seed)
    if extra is not None:
        for record in extra["calibration"].values():
            add_stats(record, {"gallery": record["source"]["folder"]}, None, scores, "dev", draws, seed)
        add_matrix(extra["class_name_supervision"], scores, "dev", draws, seed)
    object_folders = set()
    if object_studies is not None:
        additional = [object_studies["development"], *object_studies["components"].values(), *object_studies["confirmation"].values()]
        additional.extend(study["development"] for study in object_studies.get("historical_followups", {}).values())
        for matrix in additional:
            add_matrix(matrix, scores, matrix["panel"], draws, seed)
            object_folders.update(arm["sources"][p]["folder"] for arm in matrix["arms"].values()
                                  for p in arm["inception_official"]["participants"])
    formal_folders = {arm["sources"][p]["folder"] for matrix in matrices.values() for arm in matrix["arms"].values()
                      for p in arm["inception_official"]["participants"]}
    formal = sum(len(arm["inception_official"]["participants"]) for matrix in matrices.values() for arm in matrix["arms"].values())
    auxiliary = {rel for rel in scores if rel.startswith(("calibration/", "exploratory/"))} | (object_folders - formal_folders)
    return {"method": scorer.METHOD, "weights_sha256": scorer.WEIGHTS_SHA256, "gallery_sha256": scorer.GALLERY,
            "metric": scorer.METRIC, "historical_metric": "inception_2way",
            "historical_input": "ImageNet-normalized RGB with transform_input=False.",
            "official_input": "PIL short-edge bilinear resize 512, center crop 512, tensor bilinear resize 342, ImageNet normalize. "
                              "the official pretrained Inception-v3 factory also applies its internal transform_input=True.",
            "identification": "Pearson over pooled features. For each generated query, count gallery distractors strictly below its "
                              "paired original and divide by gallery size minus one. Ties receive zero credit.",
            "factory_graph_check": {"slurm_job": 5343371, "test_inputs": 2, "input_shape": [3, 342, 342],
                                    "eager_factory_fx_max_abs": 0.0, "corrected_factory_fx_max_abs": 0.0,
                                    "legacy_factory_fx_max_abs": 2.942490577697754, "legacy_factory_max_relative_to_peak": .629187227566065,
                                    "recompute": "paper_inception_delta.py --verify-graph --device cpu --weights <official checkpoint>"},
            "formal_scored_cases": formal, "auxiliary_scored_cases": len(auxiliary), "object_study_scored_unique_cases": len(object_folders),
            "object_study_additional_cases_outside_formal": len(object_folders - formal_folders),
            "exported_scored_cases": len(scores), "sources": evidence}


def table_cell(arm):
    coverage = arm["inception_official"]["participants"]
    number = len(coverage)
    estimate = f"{arm['metrics'][scorer.METRIC]['mean']:.4f}" if number else "—"
    return f"{estimate}（{number}/{len(arm['participants'])}）"
