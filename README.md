# EEG-Diffusion

### Visual EEG-to-Image Reconstruction

[Research report](https://botwwt.github.io/EEG-Diffusion/) · [Research plan](docs/RESEARCH_PLAN.md) · [Experiments](docs/EXPERIMENTS.md) · [Installation](docs/INSTALLATION.md)

[![checks](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml/badge.svg)](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml)
[![Code license: MIT](https://img.shields.io/badge/code-MIT-35677c.svg)](LICENSE)

## Current research state — 2026-11-09

Start with [the current method and evidence](docs/CURRENT_RESEARCH.md). The repository now includes the [HIVE/ATM source, protocol and report archive](research/eeg/README.md) and its [file provenance](provenance/research_snapshot_20261109.json).

The full ten-subject R80 comparison supports ordinary over native Brain-HIVE (DINO2AFC 0.872769 → 0.891198; SSIM 0.328023 → 0.353494). Additional semantic benefit over stronger F-only/mean-direction controls remains unconfirmed. A second independent method, end-to-end acceleration and SOTA are not established. New HPC encoder fits are baseline reconstruction, not new image gains.

Model weights, private EEG, large feature arrays and machine credentials are excluded from Git. See the [verified storage and recovery record](docs/STORAGE.md). The October proposal below is historical; consult the dated current-state page before interpreting it as an active or completed experiment.

## Stacked results on the ordinary head — 2026-11-17

**Original-components pipeline (headline).** A prior-art audit found that the joint encoder, best-of-N selection,
output shrinkage and condition amplification of the stack below come from same-field papers (MindEye2 / ENIGMA,
MindEye, MindEye2, Perceptogram). Leaving all four out — and also keeping the `K = 4` evaluated samples per query strictly
independent — **O8** extends **O7a** with a single-subject `FlexBrainEncoder` multi-scale temporal-pyramid family
(`train_flex.py`) combined on the training set via 5-fold concept-disjoint out-of-fold simplex ensembling
(`oof_blend_select.py`, `alpha_o8 = 0.50`) and a paired per-subject `ord_o8_flex` ordinary head on the same frozen
**O7a** native 512 px 5-window (`48–248`, `40–300`, `60–360`, `40–500`, `0–1000 ms` post-stimulus at `250 Hz`;
`200 ms` RSVP SOA) QR-PCA start latents, inverse-variance 80-repeat aggregation, fractional Differential-Diffusion
re-injection (`smap c0.625,0.5,1.0`) with step-2/3 `8 × 8` low-band tether, split-half contrastive TTA, and pure
training-cone restoration (`w = 1.00`, zero directional amplification, `nsel = 1`, no post-hoc blur):

| setting | PixCorr | SSIM | Alex2 | Alex5 | Incep | CLIP |
|---|---|---|---|---|---|---|
| Brain-HIVE, reproduced | 0.1828 | 0.3120 | 0.8266 | 0.8959 | 0.7440 | 0.7986 |
| + ordinary head (per subject) | 0.2016 | 0.3360 | 0.8299 | 0.9017 | 0.7551 | 0.8107 |
| O3: + original components | 0.2636 | 0.4073 | 0.8727 | 0.9152 | 0.7581 | 0.8118 |
| O4: + test-time-adapted start image | 0.2687 | 0.4090 | 0.8753 | 0.9158 | 0.7573 | 0.8121 |
| O5: + seed-ensemble condition with repeat bagging | 0.2692 | 0.4108 | 0.8780 | 0.9176 | 0.7584 | 0.8139 |
| O6: + multi-window encoder ensemble for the condition | 0.2711 | 0.4116 | 0.8796 | 0.9188 | 0.7616 | 0.8142 |
| O7a: + 6-win TTA + pure cone (`w = 1.00`) + 5-win 512px start + tether (indep K = 4) | 0.3036 | 0.4256 | 0.8953 | 0.9257 | 0.7827 | 0.8224 |
| O7: O7a + directional cone amp (`w = 1.50`) + K = 4 Tweedie coupling | 0.3074 | 0.4259 | 0.8971 | 0.9271 | 0.7812 | 0.8219 |
| **O8: O7a + single-subject `FlexBrainEncoder` + OOF blend (`w = 1.00`, indep K = 4)** | 0.3025 | 0.4252 | **0.8970** | **0.9281** | **0.7891** | **0.8299** |
| **O8b: O8 + K = 4 Tweedie coupling (`w = 1.00`, no cone amp)** | **0.3064** | **0.4266** | **0.8977** | **0.9280** | **0.7886** | **0.8300** |

**O8** improves on O6 across 10/10 subjects on PixCorr, Alex2, Alex5, Incep, CLIP, EffNet (`0.8434`, `−0.0200`) and 200-way
CLIP top-1 (`0.1302 → 0.1581`) and 8/10 on SSIM; over **O7a**, **O8** lifts **Alex5 `+0.0024` (`9/10`, above Perceptogram
`0.927`), Incep `+0.0065` (`7/10`), CLIP `+0.0075` (`9/10`, matching CogCapPro `0.830`), EffNet `−0.0061` (`10/10`)** and
CLIP top-1 `+0.0065` (`0.1516 → 0.1581`), at `−0.0011` PixCorr and `−0.0004` SSIM, using strictly single-subject data, `nsel = 1`,
and independent `K = 4` draws. Over O6 on the 6 untouched held-out subjects `{1, 3, 4, 6, 7, 10}`, **O8** gains
`+0.0294 / +0.0111 / +0.0177 / +0.0094 / +0.0283 / +0.0128` (`6/6` on all metrics except SSIM `4/6`; vs
`+0.0344 / +0.0174 / +0.0170 / +0.0093 / +0.0263 / +0.0200`, `4/4` on `{2, 5, 8, 9}`). **O8b** adds only `--tmf` K = 4
Tweedie coupling (`w = 1.00` kept), reaching `SSIM 0.4266` (above ENIGMA `0.4264`) and `CLIP 0.8300`. An `R = 4`-gated
start-latent ablation (`O8#w5start`) lost `PixCorr −0.0069` and `Alex2 −0.0090` (`results/o8_b5_stack.txt`) and was not adopted.

**Full stack including the borrowed components (credited):** [`research/eeg_stack_20261112/`](research/eeg_stack_20261112/README.md) stacks multi-window
multi-seed joint ten-subject semantic ensembles (`BrainMLP` in B⁴; adding `train_ms_flex.py` `FlexBrainEncoder` +
5-fold OOF simplex blending `w_b5 = (0.65, 0.10, 0.25)` + `ord_b5_flex` in B⁵), an 8-encoder 4-orientation (`0°/45°/90°/135°`)
coarse `16 × 16` SDXL-VAE contour `BrainMLP`, directional cone amplification (`w = 1.75`), native 512 px 5-window QR-PCA +
contour-disambiguated start latents, and best-of-16 candidate selection (scored using only Brain-HIVE's own `FEmbedder`,
EEG-predicted VAE contours and start-latent gradients; zero evaluation-network weights, zero post-hoc blur in B⁴-1a..B⁵) on
top of the shared ordinary head. Both strictly independent `K = 4` draws (`--tmf` off: **B⁴-1a**, **B⁴a**, **B⁵-1a**, **B⁵a**)
and step-1/2 `K = 4` Tweedie-coupled draws (`--tmf` on: **B⁴-1**, **B⁴**, **B⁵-1**, **B⁵**) are reported on the THINGS-EEG test
set (R80, 10 subjects, `K = 4` images per query, Brain-HIVE metrics; SwAV not measured):

| setting | PixCorr | SSIM | Alex2 | Alex5 | Incep | CLIP |
|---|---|---|---|---|---|---|
| Brain-HIVE, reproduced | 0.1828 | 0.3120 | 0.8266 | 0.8959 | 0.7440 | 0.7986 |
| + ordinary head | 0.2016 | 0.3360 | 0.8299 | 0.9017 | 0.7551 | 0.8107 |
| A″: stack, best of 8, shrinkage | 0.2782 | 0.4043 | 0.8823 | 0.9255 | 0.7837 | 0.8251 |
| B″: A″ + condition amplification ×2 | 0.2710 | 0.3995 | 0.8859 | 0.9269 | 0.7863 | 0.8266 |
| B‴: B″ with best of 16, shrinkage 24 px | 0.2748 | 0.4007 | 0.8861 | 0.9270 | 0.7882 | 0.8278 |
| B⁴-1a: full stack, `nsel = 1`, independent K = 4 draws, no blur | 0.3184 | 0.4273 | 0.9063 | 0.9340 | 0.7937 | 0.8313 |
| B⁴-1: B⁴-1a + K = 4 Tweedie coupling (`nsel = 1`), no blur | 0.3232 | 0.4289 | 0.9073 | 0.9343 | 0.7938 | 0.8309 |
| **B⁴a: B⁴-1a + best of 16, independent K = 4 draws, no blur** | 0.3280 | 0.4305 | **0.9118** | **0.9377** | 0.8000 | 0.8380 |
| **B⁴: B⁴a + K = 4 Tweedie coupling (best of 16), no blur** | **0.3319** | **0.4319** | **0.9130** | **0.9377** | 0.8004 | 0.8384 |
| B⁵-1a: Flex OOF-simplex stack, `nsel = 1`, indep K = 4 draws, no blur | 0.3117 | 0.4259 | 0.9048 | 0.9334 | 0.7966 | 0.8399 |
| B⁵-1: B⁵-1a + K = 4 Tweedie coupling (`nsel = 1`), no blur | 0.3162 | 0.4274 | 0.9056 | 0.9334 | 0.7965 | 0.8397 |
| **B⁵a: B⁵-1a + best of 16, independent K = 4 draws, no blur** | 0.3211 | 0.4294 | 0.9092 | 0.9375 | **0.8029** | **0.8450** |
| **B⁵: B⁵a + K = 4 Tweedie coupling (best of 16), no blur** | 0.3247 | 0.4306 | **0.9101** | **0.9377** | 0.8017 | **0.8449** |

**B⁵a** (`CLIP 0.8450`, `Incep 0.8029`, `EffNet 0.8363`, `CLIP top-1 0.1862`) and **B⁵** (`CLIP 0.8449`, `Alex2 0.9101`,
`EffNet 0.8370`, `CLIP top-1 0.1884`) lift semantic CLIP similarity by `+0.0070` (`10/10` subjects over B⁴a) and `+0.0065`
(`9/10` over B⁴) while trading off `−0.0069 / −0.0072` PixCorr and `−0.0011 / −0.0013` SSIM relative to B⁴a/B⁴ (and **B⁵-1a** at
`nsel = 1` with independent `K = 4` draws already reaches `CLIP 0.8399`, `+0.0086` over B⁴-1a in `10/10` subjects). Against
published 10-subject EEG-only results ([`results/leaderboard.md`](research/eeg_stack_20261112/results/leaderboard.md)), B⁴a, B⁴
and **B⁵** (`Alex2 0.9101` vs Perceptogram `0.910` `+0.0001`) exceed all six published maxima simultaneously (with different
metric implementations across papers). Every arm, the per-subject numbers (`results/table10.txt`, `results/o7_b4_stack.txt`,
`results/o8_b5_stack.txt`), the selection rules and the negative results are in the folder's README.

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
