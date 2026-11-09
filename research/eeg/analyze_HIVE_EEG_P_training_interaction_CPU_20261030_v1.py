"""Terminal-only supplementary interaction of four fixed training policies.

This is a classical factorial contrast, not a new model or an efficacy gate.
The four independently fitted policies are not checkpoint grafts. No inference
about biological causality or metric-independent synergy is made.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = ("ordinary", "EEG_only_correct", "P_only_correct", "joint_correct")
KEYS = ("target_cosine", "DINO2AFC", "gallery_top1",
        "within_concept10_top1", "within_concept10_two_afc")
COEFFICIENTS = {
    "EEG_minus_ordinary": (-1, 1, 0, 0),
    "P_minus_ordinary": (-1, 0, 1, 0),
    "joint_minus_ordinary": (-1, 0, 0, 1),
    "joint_minus_EEG": (0, -1, 0, 1),
    "joint_minus_P": (0, 0, -1, 1),
    "training_policy_interaction": (1, -1, -1, 1),
}
NUMERICAL_ZERO = 1e-12
STUDY_SHA = "e83c3d0a19751bbdc72c3d93ea4484057294a469137c9dc6925d1e289f0a8c7e"


def read(path):
    return json.loads(Path(path).read_bytes())


def reference(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())


def analyze(values, queries):
    if set(values) != set(METHODS) or len(queries) not in (80, 400):
        raise ValueError("Require the four fixed policies and complete80/400 queries")
    n = len(queries)
    ids = [q["query_id"] for q in queries]
    if len(set(ids)) != n or len({q["subject_id"] for q in queries}) != 1:
        raise ValueError("Require unique identities and the registered single subject")
    groups = [[i for i, q in enumerate(queries) if q["concept_id"] == c]
              for c in sorted({q["concept_id"] for q in queries})]
    if any(len(group) != 10 for group in groups):
        raise ValueError("Require all ten stimuli of each concept; never drop a query")
    banks = [np.asarray(values[method], dtype=np.float64) for method in METHODS]
    if any(bank.shape != (n, len(KEYS)) or not np.isfinite(bank).all() for bank in banks):
        raise ValueError("Invalid full-denominator metric arrays")
    for bank in banks:
        if np.any(np.abs(bank[:, 0]) > 1 + NUMERICAL_ZERO):
            raise ValueError("Cosine out of range")
        if np.any(bank[:, 1:] < 0) or np.any(bank[:, 1:] > 1):
            raise ValueError("Rank score out of range")
    raw = {name: sum(coefficient * bank for coefficient, bank in zip(weights, banks))
           for name, weights in COEFFICIENTS.items()}
    # Draw concept clusters jointly for every method, contrast and metric.
    # Independent per-arm resampling would lose the paired covariance.
    draws = np.random.default_rng(1729).integers(0, len(groups), (1000, len(groups)))
    statistics = {}
    for name, delta in raw.items():
        concept_means = np.stack([delta[group].mean(axis=0) for group in groups])
        bootstrap = concept_means[draws].mean(axis=1)
        intervals = np.quantile(bootstrap, (.025, .975), axis=0)
        means = delta.mean(axis=0)
        statistics[name] = {
            key: dict(difference=float(means[j]),
                      descriptive_concept_bootstrap95=intervals[:, j].tolist())
            for j, key in enumerate(KEYS)
        }
    flags = {}
    for key in KEYS:
        joint = statistics["joint_minus_ordinary"][key]
        against_single = [statistics[name][key] for name in ("joint_minus_EEG", "joint_minus_P")]
        interaction = statistics["training_policy_interaction"][key]
        flags[key] = dict(
            joint_directionally_above_ordinary=joint["descriptive_concept_bootstrap95"][0] > NUMERICAL_ZERO,
            joint_directionally_above_both_single_updates=all(
                field["descriptive_concept_bootstrap95"][0] > NUMERICAL_ZERO for field in against_single),
            additive_scale_positive_interaction=interaction["descriptive_concept_bootstrap95"][0] > NUMERICAL_ZERO,
            additive_scale_negative_interaction=interaction["descriptive_concept_bootstrap95"][1] < -NUMERICAL_ZERO,
        )
    return dict(query_count=n, concepts=len(groups), training_seed=1729,
                bootstrap_draws=1000, coefficients={k:list(v) for k, v in COEFFICIENTS.items()},
                method_order=list(METHODS), statistics=statistics, descriptive_flags=flags), raw


def terminal_gate(directory):
    directory = Path(directory)
    manifest = directory / "received_manifest.json"
    if not manifest.is_file():
        return dict(status="WAIT_FOR_INDEPENDENTLY_AUDITED_TERMINAL_ARCHIVE",
                    candidate_outcomes_read=0, new_GPU_jobs=0)
    value = read(manifest)
    if value.get("status") != "COMPLETE_CORE_TERMINAL_EVIDENCE_READY" or value.get("exact_main_live") is not False:
        raise ValueError("Incomplete or live outcome archives cannot be analyzed")
    audit_path = directory / "independent_local_statistics_and_provenance.json"
    if not audit_path.is_file():
        return dict(status="WAIT_FOR_ORIGINAL_INDEPENDENT_LOCAL_AUDIT",
                    candidate_outcomes_read=0, new_GPU_jobs=0)
    return None


def run(directory, output):
    directory = Path(directory).resolve()
    waiting = terminal_gate(directory)
    if waiting is not None:
        return waiting
    registration = read(Path(__file__).resolve().parent /
                        "configs/HIVE_EEG_P_training_interaction_analysis_20261030_v1.json")
    if registration["study_sha256"] != STUDY_SHA:
        raise ValueError("Supplementary analysis registration mismatch")
    for source in registration["pinned_local_sources"]:
        actual = reference(source["path"])
        if actual["bytes"] != source["bytes"] or actual["sha256"] != source["sha256"]:
            raise ValueError("Registered analysis source changed")
    # Re-run the unchanged full local audit; a loose PASS string is insufficient.
    import audit_HIVE_paired_semantic_terminal_local_20261030_v1 as original
    if reference(directory / "protocol/study.json")["sha256"] != STUDY_SHA:
        raise ValueError("The experiment is not the fixed five-arm study")
    checked = original.audit(directory)
    manifest = read(directory / "received_manifest.json")
    stage = manifest["stage"]
    if stage not in (80, 400):
        raise ValueError("Not a registered terminal stage")
    queries = read(directory / "reference/panel.json")["Dcheck"][:stage]
    raw_path = directory / f"actual/images/all_query_values{stage}.npz"
    with np.load(raw_path, allow_pickle=False) as archive:
        values = {method: np.stack([archive[method + "__" + key] for key in KEYS], axis=1)
                  for method in METHODS}
    result, raw = analyze(values, queries)
    result.update(status="COMPLETE_SUPPLEMENTARY_TRAINING_POLICY_INTERACTION",
                  UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  supplementary_registration=reference(Path(__file__).resolve().parent /
                      "configs/HIVE_EEG_P_training_interaction_analysis_20261030_v1.json"),
                  experiment=reference(directory / "protocol/study.json"),
                  source_raw_values=reference(raw_path),
                  received_manifest=reference(directory / "received_manifest.json"),
                  original_local_audit=reference(directory / "independent_local_statistics_and_provenance.json"),
                  original_terminal_decision=checked["stages"][-1]["decision"],
                  original_decision_not_modified=True,
                  analysis_only_no_new_eligibility_gate=True,
                  single_subject_exposed_development_panel=True,
                  no_checkpoint_grafts_no_biological_causality=True,
                  metric_scale_dependent_not_monotone_invariant=True,
                  new_models=0, new_images=0, new_GPU_jobs=0, novelty_established=False, SOTA=False)
    output = Path(output).resolve()
    if output == directory or output.is_relative_to(directory):
        raise ValueError("Supplementary output must not modify the received evidence archive")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "all_paired_policy_contrasts.npz",
                        **{name + "__" + key: delta[:, j]
                           for name, delta in raw.items() for j, key in enumerate(KEYS)})
    (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.directory, arguments.output), allow_nan=False))
