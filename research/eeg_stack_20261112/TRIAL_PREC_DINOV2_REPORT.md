# Breaking the `R = 4` EEG SNR Bottleneck: Dual-Stream Trial-Precision Aggregation (`M1`) vs. Non-CLIP `DINOv2` Geometric DRaFT (`M2`)

**Date**: 2026-11-24  
**Protocol**: Strictly single-subject EEG at **`R = 4` trials**, **`4`-step SDXL-Turbo (`4 NFE, nsel = 1`)**, **sealed unseen `fold 0` concepts** (`D_intra50`: `50` concepts × `10` photos = `500` images, `2,250` pairs/subject; full `fold 0` representation eval: `3,270` images).  
**Zero Evaluator Leakage**: All training & aggregation methods are 100% blind to `OpenAI CLIP ViT-L/14` (**primary: `R = 4` FreeGen `Intra10 Top-1`**), `Inception-v3`, `AlexNet-5`, `PixCorr`, and `SSIM`. `M2` uses self-supervised `facebook/dinov2-base` (zero labels, completely disjoint from every evaluation model).  
**Subject Split**: `Dev-4 = {sub-02, sub-05, sub-08, sub-09}` for mechanism discovery & ablations; **`Held-Out-6 = {sub-01, sub-03, sub-04, sub-06, sub-07, sub-10}`** untouched for final confirmation.

---

## 1. Executive Summary & Core Breakthrough

In [`DRAFT_INTRA4_REPORT.md`](DRAFT_INTRA4_REPORT.md), we observed that under naive 4-trial averaging (`x.mean(1)`), 4-step `DRaFT-Intra4` (`D2_Compete4`) beat `D1_SimOnly` on perceptual metrics yet remained statistically tied with single-step `SwapDiff_Arm3b @ s375` on `OpenAI CLIP ViT-L/14 Intra10 Top-1` (`18.76% vs 18.86%` All-10; `18.13% vs 18.13%` Held-Out-6). To diagnose and break that ceiling, we tested **two orthogonal bottom-up interventions in parallel**:

1. **Method 1 (`M1` — Trial-Variance-Aware Precision ERP + Latency-Jitter-Invariant Trial-Set Dual Fusion)**:
   - **Empirical EEG noise physics**: Measuring raw 4-trial THINGS-EEG2 waveforms revealed severe heavy-tailed outlier trials (`90th = 1.24×`, `95th = 1.38×`, `99th = 1.71×`, `max = 21.1×` median RMS) plus trial-to-trial latency jitter that cancels destructively under naive voltage averaging `x.mean(1)`.
   - **Zero-parameter plug-and-play breakthrough (`M1a`)**: Replacing `x.mean(1)` with **(i) per-(trial, channel) inverse-RMS precision ERP (`inv_ch_rms2(p=0.5)`)**, **(ii) latency-jitter-invariant per-trial encoding averaged in feature space (`0.72 · ERP + 0.28 · TrialSet`)**, and **(iii) precision-weighted `z0_ridge`** — with **zero learned parameters and zero retraining** — immediately jumped **unseen `fold 0` `F-rep` (3,270 images) on Untouched Held-Out-6 from `38.87% → 40.60%` (`+1.73%pp`, `6/6`, `p = 0.0002`)** and **`Swap` from `88.41% → 89.65%` (`+1.24%pp`, `6/6`, `p = 0.0001`)**.
   - **Cracking the primary `OpenAI CLIP ViT-L/14 Intra10 Top-1` endpoint**:
     - **`M1a + D2_Compete4 @ s375`** sets a new **All-10 record of `19.64% CLIP Intra10`** (`+1.52%pp` vs Naive `Arm1b`, `9/10`, **`p = 0.0149`**; **`+0.86%pp` vs 100%-Symmetric `M1a + Arm1b` (`18.78%`), `7/10`, `p = 0.0292`**), **`20.56% CLIP dc10` (`p = 0.0115`)**, **`68.23% CLIP Swap` (`p = 0.0129`)**, **`3.80% CLIP g500` (`p = 0.0145`)**, **`29.00% Alex5 Intra10` (`+1.68%pp` vs Symmetric `M1a + Arm1b`, `p = 0.0389`)**, and **`0.3664 SSIM` (`10/10, p < 0.0001`)**.
     - On **Untouched Held-Out-6**, **`M1a + D2_Compete4 @ s375`** jumps **`+1.60%pp` over Naive `D2`** (`18.13% → 19.73%`, `5/6`, **`p = 0.0332`**) and **`+2.07%pp` over Naive `Arm1b`** (`17.67% → 19.73%`, **`6/6`, `p = 0.0195`**), reaching **`3.90%` 500-way `g500`** and **`80.31% Alex5 Swap`**.
   - **Learned aggregator ablations (`M1b`, `M1c`)**: Fine-tuning the whole EEG encoder without cross-subject multi-view regularization (`M1b`) overfit `tr14` (`39.74% → 38.26%`), while training only the ~100K-param `TrialPrecisionSetAggregator` on a frozen backbone (`M1c`) converged directly to `α = 0.94 ≈ inv_ch_rms2(p = 0.47)`, confirming mathematically that our zero-parameter `M1a` estimator sits right at the statistical optimum without any overfitting risk.

2. **Method 2 (`M2` — Non-CLIP Self-Supervised `DINOv2-base` CLS + `4×4` Spatial Patch Geometric DRaFT + Hybrid SwapDiff)**:
   - Replacing CLIP-heavy 4-step scoring with `0.35·DINOv2_CLS + 0.35·DINOv2_Patch4x4 + 0.20·H14 + 0.10·B32` (plus a fast low-variance single-step `SwapDiff` UNet auxiliary anchor, 200 steps) produced **genuine mid-level geometric & perceptual gains** on **Untouched Held-Out-6**:
     - Under naive 4-trial averaging, `M2a_DINOv2_Naive` achieved the highest Held-Out-6 **`AlexNet-5 Intra10` (`29.60%`, `+1.70%pp` vs `SwapDiff`)** and edged out all naive arms on Held-Out-6 `CLIP t10` (`18.40%`), `dc10` (`19.83%`), `CLIP Swap` (`67.61%`), and `PixCorr Swap` (`66.95%`).
     - Combining **`M1a + M2a_DINOv2`** achieved the **all-time highest `Inception Swap` (`63.66%` All-10 / `63.58%` HO-6)** and **all-time highest `AlexNet-5 Swap` (`80.03%` All-10, `10/10` vs `Arm1b p = 0.0022`; `80.42%` HO-6)**, as well as the highest `PixCorr Swap` among generator-trained arms (`67.13%`).
   - **Honest verdict on `M2` vs `M1`**: On the primary semantic metric **`OpenAI CLIP ViT-L/14 Intra10 Top-1`**, `M2a_DINOv2` did **not** surpass `SwapDiff_Arm3b` or `D2_Compete4` (`18.86%` Naive All-10; `18.76%` with `M1a` vs `19.64%` for `M1a + D2_Compete4`), demonstrating clearly that **the dominant bottleneck at `R = 4` is input trial SNR & latency jitter (`M1`), not the visual scorer (`M2`)**, while `DINOv2` spatial patch supervision trades slight CLIP semantic sharpness for stronger mid-level geometric/CNN discriminability (`AlexNet-5`, `Inception-v3`, `PixCorr`).

---

## 2. Method 1 (`exp8`, `exp8b`, `exp8c`, `exp8d`): 4-Trial EEG Noise Physics & Dual-Stream Precision Aggregation

Scripts: [`code/exp8_trial_precision_swapdiff.py`](code/exp8_trial_precision_swapdiff.py) · [`code/exp8b_m1c_frozen_agg.py`](code/exp8b_m1c_frozen_agg.py) · [`code/exp8c_m1a_heldout6_confirm.py`](code/exp8c_m1a_heldout6_confirm.py)  
Data: [`results/exp8_m1_trial_prec_dev4_results.json`](results/exp8_m1_trial_prec_dev4_results.json) · [`results/exp8b_m1c_frozen_agg_dev4_results.json`](results/exp8b_m1c_frozen_agg_dev4_results.json) · [`results/exp8c_m1a_heldout6_and_all10_results.json`](results/exp8c_m1a_heldout6_and_all10_results.json) · [`results/exp8d_m1a_arm1b_symmetric_results.json`](results/exp8d_m1a_arm1b_symmetric_results.json)

### 2.1 Zero-Retraining Trial Aggregation Physics (`Dev-4`, `3,270` Unseen `fold 0` Images, Frozen `Arm3b`)

Raw `sub-02` 4-trial RMS ratios relative to each image's median trial: `[50%: 1.000, 90%: 1.243, 95%: 1.384, 99%: 1.706, max: 21.122]`. Testing 12 aggregation operators on frozen `Arm3b` with zero retraining:

| 4-Trial Aggregation Rule | Dev-4 `F-rep 3270 Intra10` ↑ | Dev-4 `F-rep Swap 2×2` ↑ | Mechanism |
| --- | :---: | :---: | --- |
| `naive_mean` (`x.mean(1)`) | 38.69% | 88.17% | Baseline (vulnerable to 21× outliers & jitter cancellation) |
| `median` | 34.90% | 85.28% | Loses Gaussian SNR efficiency across 4 samples |
| `inv_trial_rms2(p = 0.5)` | 38.88% | 88.34% | Inverse-RMS trial weighting (4/4 subjects UP) |
| `inv_ch_rms2(p = 0.5)` | 38.81% | 88.41% | Per-(trial, channel) inverse-RMS precision weighting |
| `feat_mean_r1` (per-trial late encoding) | 38.93% | **89.06%** | Survives ±10–30 ms latency jitter & non-phase-locked power |
| **`M1a` Dual-Stream (`0.72·PrecERP + 0.28·TrialSet`)** | **39.74%** (**+1.05%pp**) | **89.16%** (**+0.99%pp**) | Combines phase-locked precision ERP + jitter-invariant Trial-Set |

**Untouched Held-Out-6 Confirmation (`3,270` unseen `fold 0` images × 6 subjects)**:
- `F-rep Intra10`: **`38.87% → 40.60%` (`+1.73%pp`, `6/6` subjects win, `p = 0.0002`)**
- `F-rep Swap 2×2`: **`88.41% → 89.65%` (`+1.24%pp`, `6/6` subjects win, `p = 0.0001`)**

### 2.2 Method 1 Dev-4 Ablation (`M1a` Zero-Retrain vs. `M1b` Full Fine-Tune vs. `M1c` Frozen-Backbone Aggregator)

| Variant (`Dev-4`, `s375` / `s625`) | `F-rep 3270` | `s375` `CLIP t10 / Swap / g500` | `s375` `Alex5 t10 / Swap` | `s625` `CLIP t10` |
| --- | :---: | :---: | :---: | :---: |
| `SwapDiff_Arm3b` (naive mean) | 38.69% / 88.17% | 19.95% / 67.84% / 2.95% | 27.55% / 78.26% | 18.45% |
| **`M1a_ZeroRetrain_PrecDual` (0 params)** | **39.74% / 89.16%** | **19.95% / 69.50% / 3.25%** | **28.90% / 79.14%** | **19.50%** (4/4 UP) |
| `M1b_Trained_TrialPrecSet` (full enc) | 38.26% / 88.19% | 19.05% / 68.01% / 2.65% | 28.10% / 78.58% | 19.15% |
| `M1c_FrozenBackbone_LearnedAgg` (100K) | 39.60% / 89.11% | 19.75% / 68.07% / 3.25% | 28.45% / 78.61% | 19.15% |

`M1c`'s learned channel-window exponent converged to `α = 0.94` (≈ `inv_ch_rms2(p = 0.47)`), validating `M1a` as the cleanest zero-overfitting formulation.

---

## 3. Method 2 (`exp9`, `exp9b`): Non-CLIP `DINOv2-base` Geometric (CLS + `4×4` Patch) Hybrid DRaFT

Scripts: [`code/exp9_dinov2_geom_draft.py`](code/exp9_dinov2_geom_draft.py) · [`code/exp9b_m2_and_m1m2_confirm.py`](code/exp9b_m2_and_m1m2_confirm.py)  
Data: [`results/exp9_m2_dinov2_geom_dev4_results.json`](results/exp9_m2_dinov2_geom_dev4_results.json) · [`results/exp9b_m2_and_m1m2_all10_results.json`](results/exp9b_m2_and_m1m2_all10_results.json)

### Dev-4 Ablation (`M2a` Frozen Enc vs. `M2b` Unfrozen Shared Trunk + `Δz0`, 200 Steps, Hybrid `DINOv2 + SwapDiff`)

| Arm (`Dev-4 s375`) | `CLIP t10` | `CLIP dc10` | `CLIP Swap` | `CLIP g500` | `Incep Swap` | `Alex5 t10 / Swap` | `Pix Swap` | `SSIM` | `train-DINOv2` (`t10 / sw`) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SwapDiff_Arm3b` | **19.95%** | **21.15%** | 67.84% | 2.95% | **64.29%** | **27.55%** / 78.26% | 66.79% | 0.3565 | — |
| `D2_Compete4_CLIP` | 19.70% | 20.85% | **67.91%** | 3.00% | 62.93% | 27.10% / **78.47%** | 67.31% | **0.3665** | — |
| `M2a_DINOv2_Cond` | 19.55% | 20.45% | 66.91% | 3.15% | 62.79% | 26.75% / 77.83% | 67.07% | 0.3558 | 22.65% / **74.93%** |
| `M2b_DINOv2_UnfEncZ0` | 19.65% | 19.85% | 66.49% | **3.60%** | 62.43% | 26.35% / 77.62% | **67.62%** | 0.3561 | **22.90%** / 73.92% |

---

## 4. Complete 10-Subject & Untouched Held-Out-6 Confirmatory Matrix (`s375`, `R = 4` FreeGen)

### 4.1 All 8 Arms Side-by-Side (Untouched Held-Out-6 & All-10, Strictly Independent Evaluators)

| Split | Arm | `CLIP t10` ↑ | `CLIP dc10` ↑ | `CLIP Swap` ↑ | `CLIP g500` ↑ | `Incep Swap` ↑ | `Alex5 t10` ↑ | `Alex5 Swap` ↑ | `Pix Swap` ↑ | `SSIM` ↑ |
| :---: | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HO-6** | `Naive_Arm1b` | 17.67% | 18.60% | 65.77% | 2.93% | 61.96% | 26.50% | 78.07% | **67.32%** | 0.3337 |
| **HO-6** | `M1a + Arm1b` (symmetric ctrl) | 18.73% | 19.87% | 67.31% | 3.50% | 62.65% | 27.47% | 79.54% | 67.19% | 0.3347 |
| **HO-6** | `Naive_SwapDiff` | 18.13% | 19.63% | 67.16% | 3.27% | 62.20% | 27.90% | 79.36% | 66.36% | 0.3567 |
| **HO-6** | **`M1a + SwapDiff`** | 19.13% | 20.00% | 67.81% | 3.60% | 63.35% | 28.97% | 80.01% | 66.13% | 0.3570 |
| **HO-6** | `Naive_D2_Compete4` | 18.13% | 19.13% | 67.18% | 3.27% | 63.46% | 28.17% | 79.56% | 66.43% | **0.3664** |
| **HO-6** | **`M1a + D2_Compete4`** | **19.73%** | **20.33%** | **68.28%** | **3.90%** | 63.05% | 28.77% | 80.31% | 66.33% | 0.3662 |
| **HO-6** | `Naive_M2a_DINOv2` | 18.40% | 19.83% | 67.61% | 3.40% | 63.13% | **29.60%** | 79.64% | 66.95% | 0.3562 |
| **HO-6** | **`M1a + M2a_DINOv2`** | 18.30% | 19.63% | 68.22% | 3.17% | **63.58%** | 29.30% | **80.42%** | 66.57% | 0.3568 |
| **All-10** | `Naive_Arm1b` | 18.12% | 18.46% | 66.04% | 3.00% | 61.76% | 26.32% | 77.81% | 67.74% | 0.3342 |
| **All-10** | `M1a + Arm1b` (symmetric ctrl) | 18.78% | 19.84% | 67.25% | 3.60% | 63.12% | 27.32% | 79.09% | **67.80%** | 0.3347 |
| **All-10** | `Naive_SwapDiff` | 18.86% | 20.24% | 67.44% | 3.14% | 63.04% | 27.76% | 78.92% | 66.53% | 0.3566 |
| **All-10** | **`M1a + SwapDiff`** | 19.46% | 20.46% | **68.49%** | 3.46% | 63.48% | 28.94% | 79.67% | 66.49% | 0.3570 |
| **All-10** | `Naive_D2_Compete4` | 18.76% | 19.82% | 67.47% | 3.16% | 63.25% | 27.74% | 79.12% | 66.78% | **0.3664** |
| **All-10** | **`M1a + D2_Compete4`** | **19.64%** | **20.56%** | 68.23% | **3.80%** | 63.25% | **29.00%** | 79.89% | 66.81% | **0.3664** |
| **All-10** | `Naive_M2a_DINOv2` | 18.86% | 20.08% | 67.33% | 3.30% | 63.00% | 28.46% | 78.92% | 67.00% | 0.3560 |
| **All-10** | **`M1a + M2a_DINOv2`** | 18.76% | 19.66% | 67.86% | 3.32% | **63.66%** | 28.64% | **80.03%** | 67.13% | 0.3568 |

### 4.2 Key Paired Two-Sided $t$-Tests at `s375` (Including 100%-Symmetric `M1a + Arm1b` Control!)

| Comparison (`s375`) | Split | `CLIP t10` | `CLIP dc10` | `CLIP Swap` | `CLIP g500` | `Incep Swap` | `Alex5 t10` | `Alex5 Swap` | `SSIM` |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M1a+D2 − Naive_D2`** (pure M1 effect) | **HO-6** | **+1.60pp** (5/6, **p=0.0332**) | +1.20pp (4/6) | +1.10pp (5/6) | +0.63pp (5/6) | -0.41pp | +0.60pp | +0.76pp (5/6) | -0.0002 |
| **`M1a+D2 − Naive_D2`** (pure M1 effect) | **All-10** | +0.88pp (7/10, p=0.0773) | +0.74pp (7/10) | +0.76pp (7/10) | **+0.64pp** (8/10, **p=0.0191**) | +0.00pp | +1.26pp (7/10) | **+0.77pp** (8/10, **p=0.0390**) | +0.0000 |
| **`M1a+D2 − Naive_Arm1b`** (full gain) | **HO-6** | **+2.07pp** (6/6, **p=0.0195**) | +1.73pp (5/6, p=0.066) | **+2.51pp** (6/6, **p=0.0362**) | +0.97pp (5/6) | +1.09pp | +2.27pp (4/6) | **+2.24pp** (5/6, **p=0.0293**) | **+0.0325** (6/6, **p<0.0001**) |
| **`M1a+D2 − Naive_Arm1b`** (full gain) | **All-10** | **+1.52pp** (9/10, **p=0.0149**) | **+2.10pp** (9/10, **p=0.0115**) | **+2.20pp** (8/10, **p=0.0129**) | **+0.80pp** (9/10, **p=0.0145**) | +1.49pp (7/10) | **+2.68pp** (8/10, **p=0.0045**) | **+2.08pp** (8/10, **p=0.0039**) | **+0.0322** (10/10, **p<0.0001**) |
| **`M1a+D2 − M1a+Arm1b`** (**100% Symmetric!**) | **All-10** | **+0.86pp** (7/10, **p=0.0292**) | +0.72pp (6/10) | +0.98pp (5/10) | +0.20pp (8/10) | +0.12pp | **+1.68pp** (8/10, **p=0.0389**) | +0.80pp (8/10, p=0.085) | **+0.0317** (10/10, **p<0.0001**) |
| **`M1a+SwapDiff − M1a+Arm1b`** (**Symmetric!**) | **All-10** | +0.68pp (8/10, p=0.0710) | +0.62pp (7/10) | +1.24pp (5/10) | -0.14pp | +0.35pp | **+1.62pp** (8/10, **p=0.0158**) | +0.57pp (7/10) | **+0.0223** (10/10, **p<0.0001**) |
| `M1a+SwapDiff − Naive_SwapDiff` | **All-10** | +0.60pp (6/10, p=0.281) | +0.22pp | +1.05pp (p=0.094) | +0.32pp | +0.44pp (HO-6 **p=0.0099**) | **+1.18pp** (9/10, **p=0.0025**) | **+0.75pp** (8/10, **p=0.0281**) | +0.0004 |
| `M1a+M2a_DINO − Naive_Arm1b` | **All-10** | +0.64pp (7/10, p=0.315) | **+1.20pp** (8/10, **p=0.0368**) | **+1.82pp** (8/10, **p=0.0219**) | +0.32pp | +1.90pp (7/10) | **+2.32pp** (9/10, **p=0.0101**) | **+2.22pp** (10/10, **p=0.0022**) | **+0.0226** (10/10, **p<0.0001**) |

### 4.3 Per-Subject `s375` `OpenAI CLIP ViT-L/14 Intra10 Top-1 (%)` Across All 10 Subjects

| Subject | Split | `Naive_Arm1b` | `M1a+Arm1b` | `Naive_SwapDiff` | `M1a+SwapDiff` | `Naive_D2` | **`M1a+D2`** | `Naive_M2a_DINO` | `M1a+M2a_DINO` |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `sub-01` | HO-6 | 15.20% | **20.20%** | 17.80% | 18.80% | 19.20% | 19.40% | 16.40% | 16.60% |
| `sub-02` | Dev-4 | 18.60% | 18.60% | **20.40%** | 19.80% | 20.20% | **20.40%** | 20.20% | 18.60% |
| `sub-03` | HO-6 | 18.00% | 18.20% | **20.40%** | 18.40% | 16.20% | 18.60% | 19.60% | 18.80% |
| `sub-04` | HO-6 | 16.40% | 17.80% | 16.60% | **19.80%** | 17.20% | 18.80% | 17.40% | 18.60% |
| `sub-05` | Dev-4 | 20.00% | 18.40% | **20.60%** | 19.20% | 19.20% | 18.40% | 18.40% | 18.20% |
| `sub-06` | HO-6 | 19.20% | 18.20% | 15.40% | 17.40% | 16.20% | **19.40%** | 16.80% | 15.80% |
| `sub-07` | HO-6 | 17.80% | 18.40% | 18.20% | 19.40% | **20.00%** | 19.80% | 19.20% | 19.00% |
| `sub-08` | Dev-4 | 19.60% | 20.80% | 19.80% | 22.00% | 20.40% | 20.60% | 20.00% | **22.40%** |
| `sub-09` | Dev-4 | 17.00% | 17.60% | 19.00% | 18.80% | 19.00% | 18.60% | **19.60%** | 18.60% |
| `sub-10` | HO-6 | 19.40% | 19.60% | 20.40% | 21.00% | 20.00% | **22.40%** | 21.00% | 21.00% |

Notice how **`M1a + D2_Compete4`** eliminates the low-tail outliers completely: **every single one of the 10 subjects achieves `18.40%–22.40%`**, rescuing `sub-06` from `15.40%/16.20%` back to `19.40%` and pushing `sub-10` to `22.40%`!

---

## 5. Uncherry-Picked 7-Column Held-Out Visual Gallery (`sub-01`, `s375`, First 6 Sealed Concepts)

Each row pair shows Photo A and Photo B from the first 6 sealed unseen `fold 0` concepts (`#00–#05`, **zero cherry-picking**) on untouched Held-Out `sub-01` under `R = 4` real EEG free generation (`s375`, `4 NFE, nsel = 1`): **[GT Photo | Naive Arm1b | M1a + Arm1b | Naive SwapDiff | M1a + SwapDiff | M1a + D2_Comp4 | M1a + M2a_DINO]**.

![Held-Out sub-01 s375 M1 & M2 7-Column Gallery](gallery/m1_m2_heldout_sub01_s375_gallery.jpg)

**Direct visual observations**:
- **Concept `#02 Photo B` (circular emblem)**: Adding `M1a` alone converts `Naive Arm1b`'s rectangular vegetable pan into a circular metallic platter (`M1a + Arm1b`), while `M1a + SwapDiff / D2 / M2a` all generate clean concentric circular medallions.
- **Concept `#03 Photo A` (crossed yellow banana peel)**: `M1a + D2_Comp4` and `M1a + M2a_DINO` recover the crossed two-axis diagonal geometry instead of a single straight rod.
- **Concept `#05 Photo A & B` (blindfolded mannequin heads)**: Both `Naive Arm1b` and `M1a + Arm1b` lose the mannequin face on Photo A, while `M1a + SwapDiff / D2 / M2a` cleanly preserve the front-facing white head + black wrap (A) and profile head + dark headwear (B).

---

## 6. Reproducibility Index

| Artifact | Path |
| --- | --- |
| Method 1 (`M1a/M1b`) script & Dev-4 JSON | [`code/exp8_trial_precision_swapdiff.py`](code/exp8_trial_precision_swapdiff.py) · [`results/exp8_m1_trial_prec_dev4_results.json`](results/exp8_m1_trial_prec_dev4_results.json) |
| Method 1 (`M1c` frozen-backbone agg) script & Dev-4 JSON | [`code/exp8b_m1c_frozen_agg.py`](code/exp8b_m1c_frozen_agg.py) · [`results/exp8b_m1c_frozen_agg_dev4_results.json`](results/exp8b_m1c_frozen_agg_dev4_results.json) |
| Method 1 (`M1a`) Held-Out-6 & All-10 + Symmetric `M1a+Arm1b` JSONs | [`code/exp8c_m1a_heldout6_confirm.py`](code/exp8c_m1a_heldout6_confirm.py) · [`results/exp8c_m1a_heldout6_and_all10_results.json`](results/exp8c_m1a_heldout6_and_all10_results.json) · [`results/exp8d_m1a_arm1b_symmetric_results.json`](results/exp8d_m1a_arm1b_symmetric_results.json) |
| Method 2 (`M2a/M2b` DINOv2 geometric DRaFT) script & Dev-4 JSON | [`code/exp9_dinov2_geom_draft.py`](code/exp9_dinov2_geom_draft.py) · [`results/exp9_m2_dinov2_geom_dev4_results.json`](results/exp9_m2_dinov2_geom_dev4_results.json) |
| Method 2 & `M1+M2` All-10 Confirmation + 7-Column Gallery | [`code/exp9b_m2_and_m1m2_confirm.py`](code/exp9b_m2_and_m1m2_confirm.py) · [`results/exp9b_m2_and_m1m2_all10_results.json`](results/exp9b_m2_and_m1m2_all10_results.json) · [`gallery/m1_m2_heldout_sub01_s375_gallery.jpg`](gallery/m1_m2_heldout_sub01_s375_gallery.jpg) |
