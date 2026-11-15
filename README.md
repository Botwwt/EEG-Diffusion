# EEG-Diffusion

### Visual EEG-to-Image Reconstruction

[Research report](https://botwwt.github.io/EEG-Diffusion/) · [Research plan](docs/RESEARCH_PLAN.md) · [Experiments](docs/EXPERIMENTS.md) · [Installation](docs/INSTALLATION.md)

[![checks](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml/badge.svg)](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml)
[![Code license: MIT](https://img.shields.io/badge/code-MIT-35677c.svg)](LICENSE)

## Current research state — 2026-11-09

Start with [the current method and evidence](docs/CURRENT_RESEARCH.md). The repository now includes the [HIVE/ATM source, protocol and report archive](research/eeg/README.md) and its [file provenance](provenance/research_snapshot_20261109.json).

The full ten-subject R80 comparison supports ordinary over native Brain-HIVE (DINO2AFC 0.872769 → 0.891198; SSIM 0.328023 → 0.353494). Additional semantic benefit over stronger F-only/mean-direction controls remains unconfirmed. A second independent method, end-to-end acceleration and SOTA are not established. New HPC encoder fits are baseline reconstruction, not new image gains.

Model weights, private EEG, large feature arrays and machine credentials are excluded from Git. See the [verified storage and recovery record](docs/STORAGE.md). The October proposal below is historical; consult the dated current-state page before interpreting it as an active or completed experiment.

## Stacked results on the ordinary head — 2026-11-15

**Original-components pipeline (headline).** A prior-art audit found that the joint encoder, best-of-N selection,
output shrinkage and condition amplification of the stack below come from same-field papers (MindEye2 / ENIGMA,
MindEye, MindEye2, Perceptogram). Leaving them out and keeping only components that are ours or transferred from other
fields — native 512 px 5-window RSVP QR-PCA start latents with retinotopic-layout-disambiguated kNN, inverse-variance
aggregation of the 80 test repeats (robust statistics), fractional img2img strength by re-injection (Differential
Diffusion) with a step-2/3 coarse-band tether and spatial-contrast-restored Tweedie coupling, and 6-window 42-encoder
split-half contrastive test-time adaptation with conical optimal transport onto the training target cone —
on top of the per-subject Brain-HIVE encoder and the ordinary head (one per subject), one draw per image, no post-hoc blur:

| setting | PixCorr | SSIM | Alex2 | Alex5 | Incep | CLIP |
|---|---|---|---|---|---|---|
| Brain-HIVE, reproduced | 0.1828 | 0.3120 | 0.8266 | 0.8959 | 0.7440 | 0.7986 |
| + ordinary head (per subject) | 0.2016 | 0.3360 | 0.8299 | 0.9017 | 0.7551 | 0.8107 |
| O3: + original components | 0.2636 | 0.4073 | 0.8727 | 0.9152 | 0.7581 | 0.8118 |
| O4: + test-time-adapted start image | 0.2687 | 0.4090 | 0.8753 | 0.9158 | 0.7573 | 0.8121 |
| O5: + seed-ensemble condition with repeat bagging | 0.2692 | 0.4108 | 0.8780 | 0.9176 | 0.7584 | 0.8139 |
| O6: + multi-window encoder ensemble for the condition | 0.2711 | 0.4116 | 0.8796 | 0.9188 | 0.7616 | 0.8142 |
| **O7: + 6-win TTA+cone + 5-win 512px start + tether + Tweedie-CR** | **0.3074** | **0.4259** | **0.8971** | **0.9271** | **0.7812** | **0.8219** |

O7 improves on the reproduced baseline in 10/10 subjects on every metric, beats O6 across the board (PixCorr +0.0363,
SSIM +0.0143, Alex2 +0.0176, Alex5 +0.0083, Incep +0.0196, CLIP +0.0077; 10/10, 9/10, 10/10, 10/10, 10/10, 8/10 subjects),
and ranks 1 / 2 / 2 / 1 (tie) / 1 / 2 against published 10-subject EEG-only results with a single draw and zero
cross-subject data. Reliability-driven strength maps, full-step ILVR, low-pass re-injection and un-restored Tweedie
averaging were tested with controls and reported alongside the positive components in the folder README.

**Full stack including the borrowed components (credited):** [`research/eeg_stack_20261112/`](research/eeg_stack_20261112/README.md) stacks a multi-window
multi-seed joint ten-subject semantic encoder ensemble, an 8-encoder 4-orientation steerable SDXL-VAE contour `BrainMLP`,
native 512 px 5-window RSVP QR-PCA + contour-disambiguated start latents, and best-of-N candidate selection (using only
Brain-HIVE's own `FEmbedder`, steerable VAE contours and start-latent gradients; zero evaluation-network weights, zero
post-hoc blur in B⁴-1 and B⁴) on top of the shared ordinary head. THINGS-EEG test set, R80, 10 subjects, K = 4 images
per query, Brain-HIVE metrics (SwAV not measured):

| setting | PixCorr | SSIM | Alex2 | Alex5 | Incep | CLIP |
|---|---|---|---|---|---|---|
| Brain-HIVE, reproduced | 0.1828 | 0.3120 | 0.8266 | 0.8959 | 0.7440 | 0.7986 |
| + ordinary head | 0.2016 | 0.3360 | 0.8299 | 0.9017 | 0.7551 | 0.8107 |
| A″: stack, best of 8, shrinkage | 0.2782 | 0.4043 | 0.8823 | 0.9255 | 0.7837 | 0.8251 |
| B″: A″ + condition amplification ×2 | 0.2710 | 0.3995 | 0.8859 | 0.9269 | 0.7863 | 0.8266 |
| B‴: B″ with best of 16, shrinkage 24 px | 0.2748 | 0.4007 | 0.8861 | 0.9270 | 0.7882 | 0.8278 |
| **B⁴-1: full stack, single draw (`nsel = 1`), no blur** | 0.3232 | 0.4289 | 0.9073 | 0.9343 | 0.7938 | 0.8309 |
| **B⁴: B⁴-1 + best of 16, no blur** | **0.3319** | **0.4319** | **0.9130** | **0.9377** | 0.8004 | 0.8384 |
| **B⁴#sv(0.20,0.20,0.18@16): lighter spatial/grad weights, no blur** | 0.3309 | 0.4314 | 0.9125 | **0.9381** | **0.8019** | **0.8394** |

A″, B″, B‴, B⁴-1 and B⁴ improve on the reproduced baseline for every metric in every subject; B⁴#sv(0.20,0.20,0.18@16)
also improves on B‴ for every metric in all 10/10 subjects. Best of 8 / 16 uses 8× / 16× the sampling budget of a single
draw (whereas B⁴-1 uses a single draw and ranks 1 / 1 / 2 / 1 / 1 / 1). Against published 10-subject EEG-only results
(inclusion rule and sources in [`results/leaderboard.md`](research/eeg_stack_20261112/results/leaderboard.md)), **B⁴** and **B⁴#sv(0.20,0.20,0.18@16)** rank
**1 / 1 / 1 / 1 / 1 / 1** (first on all six metrics: PixCorr, SSIM, Alex2, Alex5, Inception and CLIP). Every arm, the
per-subject numbers (`results/table10.txt`, `results/o7_b4_stack.txt`), the selection rules and the negative results are in
the folder's README; external methods are quoted there from their papers (not re-run here, metric implementations differ).

## Overview

Visual EEG-to-image reconstruction combines visual evidence decoded from brain activity with a pretrained image diffusion model. We study the connection between EEG–image representation alignment and the denoising responses that determine generated content. Our experiments examine intermediate visual supervision, conditional priors and image-level guidance.

## Historical proposal: denoising-response distillation

The next experiment learns to predict the denoising correction associated with the observed image. For a state along an EEG-conditioned sampling trajectory, the training teacher compares image and EEG conditions in the same frozen denoiser:

$$
r_t^*=D(x_t,t,c^I)-D(x_t,t,c),
\qquad
D_{\mathrm{EEG}}=D(x_t,t,c)+R_\theta(x_t,t,c,b).
$$

Here, $c$ is the base EEG condition, $b$ is a temporal representation of the same EEG, and $c^I$ is the training image condition. The teacher supplies response targets during training; inference uses the EEG-conditioned response predictor.

The EEG encoder and image generator remain fixed. Training matches the teacher response with a weighted mean-squared error; evaluation compares the generated images with direct condition regression and conventional adapter training. See the [experiment design](docs/RESEARCH_PLAN.md).

## Experimental setting

Experiments use paired visual stimuli and EEG from THINGS-EEG. The new study uses R4 as its primary repeat protocol and a four-step SDXL-Turbo backbone. Comparisons include direct regression, a standard conditional prior, a conventional adapter and a capacity-matched condition-only network. Each query and seed produces one image; image content, perceptual quality and computational cost are evaluated together.

The [research report](https://botwwt.github.io/EEG-Diffusion/) presents original paper figures and result tables, our representation-alignment experiments and generated-image comparisons. Experiment-specific settings and results are listed in the [experiment record](docs/EXPERIMENTS.md).

## Installation

```bash
git clone https://github.com/Botwwt/EEG-Diffusion.git
cd EEG-Diffusion
python -m pip install -e '.[dev,report]'
python -m pytest -q
```

## Repository structure

```text
src/eeg_diffusion/       Numerical methods and evaluation utilities
experiments/            Experiment index and research scripts
results/                Aggregate results and figure source data
scripts/                Report and figure builders
docs/                   Research report and documentation
tests/                  Numerical and regression tests
provenance/             Source and figure metadata
```

## Related work

- [NVOL: EEG-based Visual Retrieval and Reconstruction](https://arxiv.org/abs/2609.02582) — intermediate visual representations and hierarchical generation.
- [BrainHIVE: Learning Brain Representation with Hierarchical Visual Embeddings](https://arxiv.org/abs/2602.07495) — visual fusion and pretrained generation interfaces.
- [On Distillation of Guided Diffusion Models](https://arxiv.org/abs/2210.03142), CVPR 2023 — denoising-response supervision and progressive distillation.
- [DEFT](https://arxiv.org/abs/2406.01781) — learning a small conditional correction with a frozen diffusion prior.

## Citation and license

Software citation metadata is provided in [CITATION.cff](CITATION.cff). Code is released under the [MIT License](LICENSE); third-party materials are attributed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
