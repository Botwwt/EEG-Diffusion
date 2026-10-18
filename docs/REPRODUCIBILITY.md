# Reproduction scope

## Three distinct levels

1. **Runnable now:** install the small package, verify bundled file hashes and aggregate arithmetic, test metric definitions and Gaussian identities, rebuild the static report. No neural training occurs.
2. **Historical source inspection:** `experiments/archive/` preserves executed code and selected configurations byte for byte. The source manifest records its original location and SHA-256. This level exposes model updates and selection logic; it does not supply all original inputs.
3. **Full experiment replay:** requires the original licensed EEG/stimulus release, exact preprocessing and splits, visual feature caches, upstream model revisions, pretrained generator/evaluator weights, and artifact manifests. Those assets are not included. An old `/proc/.../fd/...` checkpoint descriptor is not a downloadable checkpoint.

The quick start validates level 1. It must not be reported as a fresh replay of levels 2–3.

## Inputs and information budget

The task is EEG-only visual reconstruction: inference receives EEG recorded while viewing an image. It does not receive that image, its caption, its true class, or a test-gallery lookup. Gallery metrics are evaluated after generation. The oracle column in the report uses the true image embedding and is excluded from deployable results.

| Contract | Required for a full replay |
|---|---|
| Data | THINGS-EEG2 and stimulus access under their original terms; participant/session/trial and stimulus identity mapping |
| Processing | Exact montage, time window, scaling/whitening source, repeat counts, author array ordering and physical trial IDs |
| Splits | Image/concept assignments and the record of previous development exposure; preprocessing fitted within the declared training boundary |
| Encoder | Exact visual layer/width, EEG architecture, joint visual head, seeds, update sequence and checkpoint selection |
| Prior / generator | Pinned upstream code, objective/parameterization, scheduler, CFG, clipping, repeat-count matching, generator and adapter weights |
| Evaluation | Frozen image set, one output per declared query/seed, evaluator weights/version, distractor gallery and resampling unit |

The historical source assumes some author EEG–image array ordering. Independence of upstream multivariate noise normalization was not fully resolved in the archived retrieval study. Do not imply that hash checks establish those upstream assumptions.

## Original upstream code

- ATM: [EEG_Image_decode](https://github.com/dongyangli-del/EEG_Image_decode). The archived native main-path experiments pin `49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d`; `paper_main_runtime.py` checks the revision and selects unchanged author definitions. Other visual-reference scripts check source-file hashes. Do not silently replace them with a current branch.
- Image generation uses separately obtained Diffusers / Stable Diffusion XL or SDXL-Turbo and adapters. Follow their model terms.
- Independent image evaluations use separately obtained DINOv3 or the evaluator named in the specific experiment. Do not combine their metrics.
- VAR was used only in the repeat-reliability component comparison. It is autoregressive, not a diffusion experiment.

## Environments

The portable package declares its own small dependency range in `pyproject.toml`; CI exercises Python 3.11 and 3.12. This range is **not** the historical GPU environment.

Several semantic pilots explicitly assert PyTorch **2.5.0 / CUDA 12.4**. Historical visual scripts depend on Linux `fcntl`, sealed memory files, fixed asset hashes and `/root/...` directories; exact requirements differ by phase. Their imports are listed in `provenance/import_inventory.json`. No universal training lockfile is fabricated for these different environments.

Before porting a GPU study, reconstruct its input manifests and dependency versions, then run the original numerical checks. Ported versions should be saved separately, with a diff and numerical equivalence checks; do not overwrite the frozen source.

## What the verification command checks

`eeg-diffusion verify` checks bundled source/asset SHA-256 values, consistency between the retrieval summary and aggregate extract, and two displayed arithmetic differences. The v/R4 values are explicitly marked as transcribed rounded report values. Historical intervals are retained; this package does not claim to recompute them without per-query features.

`tests/` additionally checks paired resampling, target exclusion/tie behavior, duplicate physical trial handling, correlated-repeat uncertainty, stable high-precision conditioning and agreement with the frozen historical kernel. Those tests validate numerical components, not image quality.

## Active-code refactor, v0.2

The numerical core was separated from byte-identical source preservation. `provenance/refactors.json` records the former copied files. Historical originals remain in `experiments/archive/semantic/`; the active Gaussian kernel now uses typed interfaces, explicit shape and covariance validation, read-only owned moments and Cholesky precision solves. Regression tests compare both implementations on coupled multivariate cases, in addition to independent analytic cases.

`provenance/evidence_manifest.json` separately protects the released aggregate JSON files. Figure hashes, renderer version, export dimensions and data hashes appear in `provenance/figure_manifest.json`. The HTML and CSV are deterministic standard-library builds. Matplotlib exports editable SVG and embedded-font PDF; fresh rendering may vary with the renderer/font environment, which is recorded rather than hidden.

See [API.md](API.md) for the runnable demo, expected output and numerical assumptions. See [FIGURE_AND_CODE_STANDARDS.md](FIGURE_AND_CODE_STANDARDS.md) for the reporting guidance used and the remaining full-study requirements.

