# Release validation — v0.2, 2026-10-18

## Numerical core

Local validation: Windows, Python 3.11.9, NumPy 2.4.6, pytest 9.1.1, Ruff 0.17.0. These are review-tool versions, not the historical GPU environment.

- **32 CPU tests passed.** Tests cover metric target/tie semantics, paired resampling, duplicate physical trial evidence, shared-session covariance, analytic conjugate cases, invalid parameters, immutable moment buffers, stable high-precision conditioning, CLI errors and integrity checks.
- The refactored Gaussian reference agrees with the frozen historical implementation in five coupled multivariate cases at `atol=1e-10`, `rtol=1e-11`.
- Local statement/branch-inclusive coverage: **85%**. Coverage measures exercised code paths, not scientific validity.
- Ruff checks and formatting pass for active `src/`, `scripts/` and `tests/`. Frozen historical files are excluded from style rewrites.
- **72 frozen source/assets** and **3 aggregate evidence files** matched their SHA-256 manifests. The two former portable source copies were refactored/removed from the active package; their originals remain frozen and their history is recorded separately.

## Report and figure exports

- HTML and source-data CSV match the released aggregates.
- Two statistical figures export as editable-text SVG and vector PDF, with embedded Arial TrueType fonts in the local PDF export. Inspection confirmed no raster data marks in either PDF.
- Figure text is 7 pt, with 8-pt bold panel labels at the recorded 174-mm width. Data, renderer, builder and output hashes are recorded.
- Chrome QA at 1440 × 1050 and 390 × 844: no JavaScript errors, broken images or page-level horizontal overflow. All three reading tabs, keyboard navigation, image zoom/Escape and command copying were exercised. Print CSS exposes all papers. Desktop, mobile, figure and print previews were visually inspected.

## Scope

This validation does not retrain an encoder/prior, extract image features, regenerate historical pictures, recompute old intervals from raw features, re-audit original preprocessing, or confirm a new method advantage. Existing low-resolution generated-image crops remain archival discussion material.

The v0.1 release had 11 tests and 74 unchanged source/asset entries; that record is retained in Git history. v0.2 separates frozen-source preservation from active numerical refactors. CI independently checks Linux/Python 3.11 and 3.12, and Windows/Python 3.11.
