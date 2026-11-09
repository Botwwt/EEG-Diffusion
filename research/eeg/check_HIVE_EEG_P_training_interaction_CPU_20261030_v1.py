"""Check paired interaction inference with counterexamples and historical data.

Historical arm aliases test arithmetic only; they do not estimate this study's
EEG/P interaction. No current candidate image scores are read.
"""
import copy
import json
import tempfile
from pathlib import Path

import numpy as np

import analyze_HIVE_EEG_P_training_interaction_CPU_20261030_v1 as I
import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S

L = Path(__file__).resolve().parent


def rejected(call):
    try:
        call()
    except (ValueError, KeyError, AssertionError):
        return
    raise AssertionError("Invalid data were accepted")


def main():
    checks = []
    engineering = L / "audit/HIVE_paired_semantic_batch2_complete_20261030_v3/actual/actual_batch2.npz"
    assignment_errors = []
    with np.load(engineering, allow_pickle=False) as source:
        for name in source.files:
            if not name.endswith("_features"):
                continue
            g = source[name].astype(np.float64)
            v = source[name[:-len("_features")] + "_targets"].astype(np.float64)
            g /= np.linalg.norm(g, axis=1, keepdims=True)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            score = g @ v.T
            correct = score[0, 0] + score[1, 1]
            wrong = score[0, 1] + score[1, 0]
            margin = (g[0] - g[1]) @ (v[0] - v[1])
            assignment_errors.append(abs(margin - (correct - wrong)))
            # Stable two-assignment normalized negative log likelihood.
            nll = np.logaddexp(correct / .07, wrong / .07) - correct / .07
            assignment_errors.append(abs(.07 * nll - .07 * np.logaddexp(0, -margin / .07)))
    assert len(assignment_errors) == 20 and max(assignment_errors) < 2e-12
    checks.append("ten real engineering pairs obey exact two-assignment margin/logistic identity; not a calibrated neural posterior")
    q = [dict(query_id=f"fixture_{i}", subject_id="fixture_sub-01",
              concept_id=f"concept_{i//10:02d}") for i in range(80)]
    # Independent endpoint scores share large cluster variation. Additive
    # cancellation must precede bootstrap; independent arm draws would fail.
    base = np.repeat(np.linspace(.15, .55, 8), 10)[:, None] * np.ones((1, 5))
    additive = dict(zip(I.METHODS, (base, base + .04, base + .03, base + .07)))
    report, raw = I.analyze(additive, q)
    assert np.max(np.abs(raw["training_policy_interaction"])) < 1e-15
    assert all(max(map(abs, field["descriptive_concept_bootstrap95"])) < 1e-15
               for field in report["statistics"]["training_policy_interaction"].values())
    assert not report["descriptive_flags"]["DINO2AFC"]["additive_scale_positive_interaction"]
    checks.append("additive policies cancel shared cluster variation before paired bootstrap")

    diminishing = dict(zip(I.METHODS, (base, base + .04, base + .03, base + .05)))
    report, _ = I.analyze(diminishing, q)
    flag = report["descriptive_flags"]["DINO2AFC"]
    assert flag["joint_directionally_above_both_single_updates"]
    assert flag["additive_scale_negative_interaction"]
    checks.append("joint beats both single updates yet has diminishing additive returns")

    bad = dict(zip(I.METHODS, (base, base - .04, base - .04, base - .02)))
    report, _ = I.analyze(bad, q)
    flag = report["descriptive_flags"]["DINO2AFC"]
    assert flag["additive_scale_positive_interaction"]
    assert not flag["joint_directionally_above_ordinary"]
    checks.append("positive interaction is not a successful method when all changes are harmful")

    # Within-cluster +/− query changes cancel exactly. Query-independent
    # bootstrap would incorrectly create apparent cluster-level uncertainty.
    changes = np.tile(np.array([.02, -.02] * 5), 8)[:, None] * np.ones((1, 5))
    balanced = dict(zip(I.METHODS, (base, base, base, base + changes)))
    report, _ = I.analyze(balanced, q)
    assert max(abs(x) for x in report["statistics"]["training_policy_interaction"]["DINO2AFC"]["descriptive_concept_bootstrap95"]) < 1e-15
    checks.append("whole-concept resampling preserves within-concept balanced cancellation")

    invalid = copy.deepcopy(q); invalid[1]["query_id"] = invalid[0]["query_id"]
    rejected(lambda: I.analyze(additive, invalid))
    invalid = copy.deepcopy(q); invalid[0]["subject_id"] = "different_subject"
    rejected(lambda: I.analyze(additive, invalid))
    invalid = copy.deepcopy(q); invalid[0]["concept_id"] = "partial_concept"
    rejected(lambda: I.analyze(additive, invalid))
    rejected(lambda: I.analyze({k:v for k, v in additive.items() if k != "P_only_correct"}, q))
    invalid_values = copy.deepcopy(additive); invalid_values["ordinary"][0, 1] = np.nan
    rejected(lambda: I.analyze(invalid_values, q))
    invalid_values = copy.deepcopy(additive); invalid_values["ordinary"][0, 1] = 1.01
    rejected(lambda: I.analyze(invalid_values, q))
    checks.append("duplicate, mixed-subject, partial-concept, missing-policy, nonfinite and invalid scores rejected")

    with tempfile.TemporaryDirectory(prefix="EEG_P_terminal_gate_") as folder:
        p = Path(folder)
        wait = I.run(p, p / "unused")
        assert wait["candidate_outcomes_read"] == 0 and not (p / "unused").exists()
        (p / "received_manifest.json").write_text(json.dumps(dict(
            status="COMPLETE_CORE_TERMINAL_EVIDENCE_READY", exact_main_live=True)))
        rejected(lambda: I.run(p, p / "unused"))
        (p / "received_manifest.json").write_text(json.dumps(dict(
            status="COMPLETE_CORE_TERMINAL_EVIDENCE_READY", exact_main_live=False)))
        wait = I.run(p, p / "unused")
        assert wait["status"] == "WAIT_FOR_ORIGINAL_INDEPENDENT_LOCAL_AUDIT" and wait["candidate_outcomes_read"] == 0
    checks.append("absent terminal or original independent audit returns WAIT without metrics; live archives rejected")

    historical = L / "audit/HIVE_common_continuation_complete_20261030_v1"
    queries = I.read(historical / "reference/panel.json")["Dcheck"]
    aliases = dict(ordinary="ordinary", EEG_only_correct="relative_only",
                   P_only_correct="AP_only", joint_correct="joint_same")
    errors = []
    checked_cells = 0
    for n in (80, 400):
        with np.load(historical / f"actual/images/all_query_values{n}.npz", allow_pickle=False) as source:
            values = {method:np.stack([source[alias + "__" + key] for key in I.KEYS], axis=1)
                      for method, alias in aliases.items()}
        report, actual = I.analyze(values, queries[:n])
        # A second implementation uses an explicit four-term sum, then
        # weighted cluster counts (not the new reader's indexed means).
        for name, coefficients in I.COEFFICIENTS.items():
            delta = np.einsum("m,mnk->nk", np.asarray(coefficients, dtype=np.float64),
                              np.stack([values[method] for method in I.METHODS]))
            groups = [[i for i, row in enumerate(queries[:n]) if row["concept_id"] == c]
                      for c in sorted({row["concept_id"] for row in queries[:n]})]
            means = np.stack([delta[group].mean(0) for group in groups])
            rng = np.random.default_rng(1729)
            counts = np.stack([np.bincount(rng.integers(0, len(groups), len(groups)),
                                          minlength=len(groups)) for _ in range(1000)]) / len(groups)
            interval = np.quantile(counts @ means, (.025, .975), axis=0)
            errors.append(float(np.max(np.abs(actual[name] - delta))))
            for j, key in enumerate(I.KEYS):
                field = report["statistics"][name][key]
                errors.append(abs(field["difference"] - float(delta[:, j].mean())))
                errors.extend(np.abs(np.array(field["descriptive_concept_bootstrap95"]) - interval[:, j]).tolist())
                checked_cells += n
        # Re-check the five ordinary pair differences through the original
        # production comparator, whose full study gate is unchanged.
        for name in ("EEG_minus_ordinary", "P_minus_ordinary", "joint_minus_ordinary", "joint_minus_EEG", "joint_minus_P"):
            expected = S.compare({key:actual[name][:, j] for j, key in enumerate(I.KEYS)},
                                 {key:np.zeros(n) for key in I.KEYS}, queries[:n])
            for key in I.KEYS:
                errors.append(abs(report["statistics"][name][key]["difference"] - expected[key]["difference"]))
    assert max(errors) < 2e-12
    checks.append("historical aliases80/400 all six contrasts match independent weighted-cluster arithmetic")
    result = dict(status="PASS_SUPPLEMENTARY_INTERACTION_CPU_CHECKS", checks=checks,
                  real_engineering_assignment_pairs=10,
                  maximum_assignment_identity_error=max(assignment_errors),
                  actual_engineering_source=I.reference(engineering),
                  historical_alias_metric_contrast_cells=checked_cells,
                  maximum_historical_error=max(errors),
                  historical_aliases_not_actual_EEG_P_evidence=True,
                  actual_new_candidate_results_read=False,
                  actual_terminal_analysis_NOT_executed=True,
                  actual_original_full_archive_integration_NOT_executed=True,
                  new_models=0, new_images=0, new_GPU_jobs=0)
    out = L / "audit/HIVE_EEG_P_training_interaction_preparation_20261030_v1"
    out.mkdir(exist_ok=True)
    (out / "actual_CPU_checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
