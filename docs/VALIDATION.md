# Release validation — 2026-10-18

Local validation used Windows, Python 3.11.9, NumPy 2.4.6 and pytest 9.1.1. The lightweight package installed in editable mode. These are release-tool versions, not a reconstructed GPU environment.

- **11 CPU tests passed:** metric target/tie semantics, paired query resampling, duplicate trial evidence, correlated session noise, source hashes and summary consistency.
- **74 curated source/assets matched SHA-256:** original training/diagnostic files and selected assets are byte-identical to the recorded local archive.
- **Static report rebuilt:** `python scripts/build_report.py --check` matched HTML, both SVG figures and CSV data.
- **Source syntax:** portable and historical Python sources compiled successfully; historical GPU entry points were not executed.
- **Browser QA:** Chrome at 1440 × 1050 and 390 × 844; no JavaScript errors, broken source images or page-level horizontal overflow. Image enlargement, Escape/close, keyboard access and the fixed-example disclosure were exercised. Screenshots were visually inspected.

This validation does not retrain an encoder/prior, extract image features, recompute historical bootstrap intervals, re-audit the original EEG preprocessing, or confirm a new method advantage. GitHub Actions repeats the CPU checks independently.

