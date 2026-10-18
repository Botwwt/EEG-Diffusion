# EEG-Diffusion

### Visual EEG-to-Image Reconstruction

[Research report](https://botwwt.github.io/EEG-Diffusion/) · [Research plan](docs/RESEARCH_PLAN.md) · [Experiments](docs/EXPERIMENTS.md) · [Installation](docs/INSTALLATION.md)

[![checks](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml/badge.svg)](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml)
[![Code license: MIT](https://img.shields.io/badge/code-MIT-35677c.svg)](LICENSE)

## Overview

Visual EEG-to-image reconstruction combines visual evidence decoded from brain activity with a pretrained image diffusion model. We study the connection between EEG–image representation alignment and the denoising responses that determine generated content. Our experiments examine intermediate visual supervision, conditional priors and image-level guidance.

## Research direction: denoising-response distillation

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
