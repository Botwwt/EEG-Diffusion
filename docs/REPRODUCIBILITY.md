# Reproduction scope

## What can be reproduced

1. **CPU checks.** Install the small package, run `python -m eeg_diffusion verify` for the October aggregates and hashes, and run `python -m eeg_diffusion verify-stack` for the November Top-1 table. `python scripts/verify_research_snapshot.py` checks the November 9 snapshot and parses Python files in the experiment archives.
2. **Source and results inspection.** `experiments/archive/` preserves the October code and selected configurations byte for byte. `research/eeg/` holds the earlier Brain-HIVE and ATM work. `research/eeg_stack_20261112/` contains the November benchmark source and aggregate logs. See the [current results and statistical scope](CURRENT_RESEARCH.md).
3. **GPU replay.** Training and image generation require the external EEG and stimuli, exact split and preprocessing, checkpoint revisions, and study-specific caches under the declared runtime. The instructions below describe those dependencies.

## Inputs and information budget

The single-participant reconstruction arms take EEG grouped by stimulus repetitions. The scoring scripts use the reference image gallery to evaluate generated images. Oracle and multi-participant diagnostic runs have their own input protocols and are reported separately.

| Contract | Required for a full replay |
|---|---|
| Data | THINGS-EEG2 and stimulus access under their original terms; participant/session/trial and stimulus identity mapping |
| Processing | Exact montage, time window, scaling/whitening source, repeat counts, author array ordering and physical trial IDs |
| Splits | Image/concept assignments and the record of previous development exposure; preprocessing fitted within the declared training boundary |
| Encoder | Exact visual layer/width, EEG architecture, joint visual head, seeds, update sequence and checkpoint selection |
| Prior / generator | Pinned upstream code, objective/parameterization, scheduler, CFG, clipping, repeat-count matching, generator and adapter weights |
| Evaluation | Frozen image set, one output per declared query/seed, evaluator weights/version, distractor gallery and resampling unit |

The historical source assumes some author EEG–image array ordering. Independence of upstream multivariate noise normalization was not fully resolved in the archived retrieval study. Do not imply that hash checks establish those upstream assumptions.

## November GPU replay

Use `RAD_ROOT` for the Linux GPU working directory expected by `code/common.py`. Follow the archive's environment and asset instructions for THINGS-EEG2, the Brain-HIVE source and Fusion Prior, SDXL-Turbo and the evaluator weights. Preserve the exact 17 EEG channels, train/test identities, alignment between subjects and the provenance of the upstream whitened arrays. [Runtime layout and external assets](../research/eeg_stack_20261112/EXPERIMENT_HISTORY.md)

Run `python -m eeg_diffusion check-o13-inputs --rad-root /path/to/rad` once the workspace has been prepared. It checks required path entries and the shapes of the EEG, visual-target and O10 arrays that O13 consumes. A successful check establishes those prerequisites. Run the archived environment and asset checks for model loading, CUDA availability and dataset ordering.

| Phase | Prerequisites and outputs |
| --- | --- |
| Data and visual targets | `prep.py` produces the EEG arrays, aligned image IDs, visual embeddings and fused targets. The archived pipeline also builds the 512-pixel latent preparation and the U-Net K/V metric. |
| O10 | `o10_factorized.py` produces `F_sem_train.pt` and the conditions and starting latents consumed by O12–O14. |
| O12 and O13 | With the O10 inputs prepared, `o12_splitrep_medoid_bigg.py` and `o13_xbrain_shared_trunk.py` run training, condition export and evaluation. |
| O14 | `o14_triarch_xbrain_bigg.py` also reads the O13 adapted predictions and uses the OpenCLIP-bigG text targets where configured. |

The O12–O14 entry points take `--phase all`. Their default full runs use the ten-subject dataset. Use the predefined five model seeds when replaying the O13/O14 training and adaptation stages together. Outputs live under `RAD_ROOT/runs`. Use fresh or independently verified caches to reproduce a specific run. [Entry points and reported protocol](../research/eeg_stack_20261112/O12_O13_O14_REPORT.md) · [Archived source](../research/eeg_stack_20261112/code/)

O13/O14 training includes the training partition of the evaluated participants. Split-half test-time adaptation uses the final 200 groups of test EEG for each participant, pairing grouped repeats and using the other groups in the loss. The condition construction also uses statistics across test EEG-derived outputs. `0tta` indicates zero adaptation updates. A prospective replay using frozen training statistics or a separate calibration partition requires its own evaluation. [O13 adaptation and condition export](../research/eeg_stack_20261112/code/o13_xbrain_shared_trunk.py) · [O14 adaptation and condition export](../research/eeg_stack_20261112/code/o14_triarch_xbrain_bigg.py)

The scoring and generator roles of CLIP are distinct. The runner encodes a fixed empty prompt using both SDXL text encoders. The bigG arms alter its second encoder's embeddings with EEG predictions. The OpenAI ViT-L/14 image encoder scores the generated images, and the 16-NFE arm uses Brain-HIVE features for candidate selection. [Generator](../research/eeg_stack_20261112/code/o11_tri_stream_bigg.py) · [Candidate selection](../research/eeg_stack_20261112/code/o14_triarch_xbrain_bigg.py) · [Evaluator](../research/eeg_stack_20261112/code/metrics.py)

The public snapshot provides aggregate tables, selected reconstructions and the core executors. Replaying the full study also requires the original caches or recipes that rebuild them, exact asset hashes, per-stimulus records and the full noise-audit and development scripts. [November results archive](../research/eeg_stack_20261112/results/)

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

