# Experiment-to-code index

`archive/` contains historical source, not a portable training command. Files are byte-identical to the source paths listed in [`source_manifest.json`](../provenance/source_manifest.json). Original absolute paths, guarded asset hashes and server-only checkpoint handling are intentionally visible. Read the [input contracts](../docs/REPRODUCIBILITY.md) before attempting a port.

| Study | Sources under `archive/` | Evidence / status |
|---|---|---|
| Strong visual representation | `visual/eeg_native_single_visual_reference_v1.py` | Mean80 retrieval improvement; actual aggregates in `results/retrieval.json` |
| I01 condition-space supervision | `visual/eeg_brainhive_post_adapter_reference_v1.py`, `visual/eeg_brainhive_condition_pullback_*` | Actual image comparisons; negative |
| I02 conditional prior / v | `visual/eeg_native_conditional_prior_*`, `visual/eeg_native_velocity_prior_*` | Actual images; v repairs epsilon, no confirmed advantage over direct query |
| I03 cross-fitting | `visual/eeg_crossfit_*` | Actual images; uncertain increment |
| I04 hidden mean/scale | `visual/eeg_norm_stats_*` | Actual images; expanded check negative |
| I05 image evidence gradient | `visual/eeg_direct_evidence_*` | Actual images; negative |
| I06 repeat reliability | Historical component results summarized in notes; VAR source is not included in this focused release | No established component advantage |
| I07 precision score-PDE | `semantic/prepare_precision_score.py`, `semantic/run_precision_score.py`, `semantic/evaluate_precision_score.py`, `semantic/precision_score_math.py` | Semantic vectors only; negative |
| I08 repeat composition / mean–noise separation | `semantic/run_semantic_fiber.py`, `semantic/audit_repeat_score_composition.py`, `semantic/decompose_guidance_mean_cov.py`, `semantic/mean_margin_prediction.py` | Vector diagnostics and metric trade-offs |
| I09 residual correction | `semantic/residual_direction_transport.py` | Semantic vectors; ranking gains with other losses worsening |
| I10 SORT | `semantic/sort_endpoint.py`, `semantic/sort_math.py`, `semantic/diagnose_sort_vacuity.py` | Calibrated constraints vacuous; negative |
| R4 baseline matching / trace | `semantic/r4_prior_match.py`, `semantic/native_prior_trace.py` | Semantic vectors only; baseline correction |
| Untested Gaussian reference candidate | `semantic/gaussian_reference.py`, `semantic/validate_reference.py` | Classical identities / synthetic checks; no established image gain |

`support/` supplies historical helper modules imported by some scripts. Its VAR-related training entry point is retained as an import dependency, not as an additional current research direction. The upstream ATM classes, raw data, feature packets and checkpoints are external. A complete successful end-to-end execution is not implied by source availability.

