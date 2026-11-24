# EEG-Diffusion

Reconstructing viewed images from EEG with Brain-HIVE and SDXL-Turbo. This research repository publishes the experiment source, reconstruction examples and tools for checking the reported results.

[Reconstruction gallery](research/eeg_stack_20261112/gallery/GALLERY.md) · [Trial-Prec & DINOv2 report (2026-11-24)](research/eeg_stack_20261112/TRIAL_PREC_DINOV2_REPORT.md) · [DRaFT-Intra4 & z0 diagnostics (2026-11-24)](research/eeg_stack_20261112/DRAFT_INTRA4_REPORT.md) · [SwapDiff report (2026-11-23)](research/eeg_stack_20261112/SWAPDIFF_REPORT.md) · [O12–O14 report](research/eeg_stack_20261112/O12_O13_O14_REPORT.md) · [Reproduction guide](docs/REPRODUCIBILITY.md) · [研究进展](docs/CURRENT_RESEARCH.md)

[![checks](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml/badge.svg)](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml)
[![Code license: MIT](https://img.shields.io/badge/code-MIT-35677c.svg)](LICENSE)

## SwapDiff-EEG: Within-Concept 2×2 EEG-Swap Denoising Contrast (`R = 4`, All-10 & Held-Out-6)

Demoting shared-trunk `O13/O14` to the baseline, we backpropagate a doubly-balanced within-concept **2×2 EEG-swap denoising contrast** $d(A, E_A) + d(B, E_B) < d(A, E_B) + d(B, E_A)$ directly through the frozen 4-step SDXL-Turbo UNet (`t ∈ {999, 749, 499, 249}`) into the EEG encoder and a zero-initialized low-rank IP conditioning residual (`4 × 2048`). See the full standalone report and all 10-subject / untouched Held-Out-6 paired statistics in [`research/eeg_stack_20261112/SWAPDIFF_REPORT.md`](research/eeg_stack_20261112/SWAPDIFF_REPORT.md).

- Across all 10 subjects (`22,500` same-concept photo pairs on sealed unseen `fold 0` concepts, **`R = 4` real EEG**, `4-NFE nsel = 1`), **`Arm3b_RepRawSwap_CondLyr`** beats the **100% architecture- and parameter-matched linear-`L_KV` representation control `Arm1b_RepIntra_CondLyr`** on frozen-UNet denoising swap (**`+9.36%pp`**, `83.88% vs 74.52%`, `10/10`, `p < 0.0001`; untouched Held-Out-6 `6/6`, `p = 0.0001`), fixed-`z0`/fixed-seed controlled swap (**`CLIP +1.57%pp`** `p = 0.0027`, **`Alex5 Intra10 +3.20%pp`** `9/10 p = 0.0019`, **`Alex5 Swap +2.96%pp`** `10/10 p < 0.0001`, **`PixCorr Swap +2.76%pp`** `10/10 p = 0.0003`; Held-Out-6 `6/6` on Alex5 & PixCorr), and free-EEG generation (**`CLIP Swap +1.23%pp`** `8/10 p = 0.0140`, **`Alex5 Intra10 +1.28%pp`** `9/10 p = 0.0351`, **`SSIM +0.020`** `10/10 p < 0.0001`, inter-concept `g500 = 3.54%`).
- **Follow-up `z0 × cond` `2×2` Diagnostics & 4-Step `DRaFT-Intra4` ([`DRAFT_INTRA4_REPORT.md`](research/eeg_stack_20261112/DRAFT_INTRA4_REPORT.md))**: Weakening `z0` anchoring (`s625 → s375`) transfers control to `cond` and lifts All-10 `SwapDiff_Arm3b` to **`18.86% CLIP Intra10`** (`9/10` vs `Arm1b`), **`20.24% CLIP dc10`** (`+1.78%pp`, `p = 0.0279`), and **`67.44% CLIP Swap`** (`+1.40%pp`, `p = 0.0020`; Held-Out-6 `6/6`, `p = 0.0132`). End-to-end 4-step within-concept 4-way competition (`D2_Compete4`) beats uncontrasted similarity (`D1_SimOnly`) on both Held-Out-6 and All-10 (`Inception Swap +1.48%pp p = 0.0019`, `Alex5 Swap +1.62%pp p = 0.0018`, `SSIM +0.0098` over `SwapDiff p < 0.0001`), and unfreezing the shared EEG trunk + PCA `Δz0` (`D4`) reaches **`28.28% Alex5 Intra10`** (`10/10` vs `Arm1b`, `p = 0.0007`) and `3.67%` Held-Out `g500`, while remaining statistically tied with single-step `SwapDiff_Arm3b @ s375` on `OpenAI CLIP ViT-L/14 Intra10 Top-1` (`18.76%–18.82% vs 18.86%`).
- **Breaking the `R = 4` SNR Bottleneck: Dual-Stream Trial-Precision Aggregation (`M1`) & Non-CLIP `DINOv2` Geometric DRaFT (`M2`) ([`TRIAL_PREC_DINOV2_REPORT.md`](research/eeg_stack_20261112/TRIAL_PREC_DINOV2_REPORT.md))**: Replacing naive 4-trial voltage averaging with zero-parameter channel-wise inverse-RMS ERP + latency-jitter-invariant per-trial feature fusion + precision `z0_ridge` (`M1a`) lifts untouched Held-Out-6 `F-rep` (3,270 unseen images) by **`+1.73%pp` (`38.87% → 40.60%`, `6/6`, `p = 0.0002`)** and pushes **`M1a + D2_Compete4 @ s375`** to an all-time high **`19.64%` All-10 / `19.73%` Held-Out-6 `OpenAI CLIP ViT-L/14 Intra10 Top-1`** (`+2.07%pp` HO-6 `6/6 p = 0.0195` vs `Naive Arm1b`; **`+0.86%pp` All-10 `p = 0.0292` vs 100%-symmetric `M1a + Arm1b`**, `29.00% Alex5 Intra10 p = 0.0389`, `3.80% g500`, `0.3664 SSIM`). Non-CLIP `DINOv2` CLS + `4×4` spatial patch scoring (`M2a`) further achieves the highest `Inception Swap` (`63.66%` All-10) and `Alex5 Swap` (`80.03%` All-10 / `80.42%` HO-6) when paired with `M1a`, while confirming that `R = 4` input SNR (`M1`) rather than the visual scorer (`M2`) was the primary CLIP ceiling.

<details>
<summary>View Held-Out 2×2 EEG-Swap comparison gallery (R = 4 EEG, fixed z0 & seed)</summary>

![Held-Out 2x2 EEG Swap Gallery](research/eeg_stack_20261112/gallery/swap_gallery_heldout.jpg)

</details>

## Overview

An EEG encoder maps repeated responses to visual conditions. The O13 experiment adds a training loss that aligns EEG from different participants viewing the **same image** while retaining EEG-to-image supervision. The pretrained Brain-HIVE interface and SDXL-Turbo turn these conditions and EEG-derived starting latents into reconstructions. The cross-participant variants use one participant's EEG to generate each image.

## Results

The November 2026 benchmark uses [THINGS-EEG2](https://doi.org/10.1016/j.neuroimage.2022.119754), with 10 participants, 200 test stimuli and 80 EEG repeats per stimulus. The runs below retain four independently sampled reconstructions per participant and stimulus. Each reconstruction uses one candidate and four U-Net function evaluations.

| Configuration | Participants in training | CLIP 200-way Top-1 ↑ | CLIP 2AFC ↑ | SSIM ↑ |
| --- | --- | ---: | ---: | ---: |
| Brain-HIVE reproduction | One per model | 11.76% | 0.7986 | 0.3120 |
| O10 | One per model | 16.20% | 0.8347 | 0.4217 |
| O12 three-view control | One per model | 16.54% | 0.8343 | 0.4214 |
| O13 matched control, cross-participant loss disabled | All 10 | 17.62% | 0.8386 | 0.4234 |
| O13 with same-image cross-participant loss | All 10 | 18.31% | 0.8387 | 0.4237 |
| O14 without the EEG-to-text branch | All 10 | **18.46%** | 0.8387 | 0.4251 |

CLIP Top-1 ranks the original stimulus among 200 reference images using each reconstruction as the query. The O13 loss ablation gains **0.69 percentage points** (participant-paired 95% CI −0.10 to +1.48). The evaluated participants contribute training data, and adapted runs use each participant's grouped test EEG. Candidate selection at 16 U-Net evaluations and the 10-participant fusion diagnostic have separate budgets and are listed in the [full result table](docs/CURRENT_RESEARCH.md#实验结果).

<details>
<summary>View sequential reconstruction examples</summary>

![The first twelve stimuli for sub-10, with their O12–O14 reconstructions and the ten-participant diagnostic in the final column](research/eeg_stack_20261112/gallery/gallery_o14_unbiased_seq_sub10.jpg)

</details>

## Quick start

Use Python 3.11 or newer. Set up a virtual environment from the repository root.

```bash
git clone https://github.com/Botwwt/EEG-Diffusion.git
cd EEG-Diffusion
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. Then run the CPU checks.

```bash
python -m pip install -e '.[dev]'
python -m eeg_diffusion reference-demo
python -m eeg_diffusion verify
python -m eeg_diffusion verify-stack
python -m pytest -q
```

The demo uses synthetic observations. `verify` checks the October release and its manifests. `verify-stack` reconstructs hit counts from the November log and computes paired intervals over the ten participants. These commands run on a CPU with the files in this repository. See [installation notes](docs/INSTALLATION.md) for setup and report rebuilding.

## GPU reproduction

GPU replay uses the original dataset, model weights and prepared caches in a separate working directory (`RAD_ROOT`). The [asset and environment guide](docs/REPRODUCIBILITY.md#november-gpu-replay) maps each stage to the archived executors.

| Stage | Entry point |
| --- | --- |
| Prepare EEG and visual targets | [`prep.py`](research/eeg_stack_20261112/code/prep.py) |
| Build the O10 prerequisites | [`kv_metric.py`](research/eeg_stack_20261112/code/kv_metric.py) and [`o10_factorized.py`](research/eeg_stack_20261112/code/o10_factorized.py) |
| Replay O12–O14 | [GPU commands and dependencies](research/eeg_stack_20261112/README.md#gpu-replay) |

Check a prepared O13 workspace before launching it.

```bash
python -m eeg_diffusion check-o13-inputs --rad-root /path/to/rad
```

## Repository guide

| Path | Contents |
| --- | --- |
| [`src/eeg_diffusion/`](src/eeg_diffusion/) and [`tests/`](tests/) | Numerical utilities, evidence audits and CPU tests |
| [`research/eeg_stack_20261112/`](research/eeg_stack_20261112/) | November GPU runs, evidence and gallery |
| [`research/eeg/`](research/eeg/) | Earlier Brain-HIVE and ATM work, including the ordinary adapter and its controls |
| [`experiments/`](experiments/) and [`results/`](results/) | October experiment source and released aggregates |
| [`scripts/`](scripts/) and [`provenance/`](provenance/) | Report builders, archive integrity and recovery tools |

The [October report](https://botwwt.github.io/EEG-Diffusion/) records the early experiments. The [research plan](docs/RESEARCH_PLAN.md) tracks the proposed confirmation study.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite the software and the exact revision you used. See the [MIT license](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md) for code and asset terms.
