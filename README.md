# EEG-Diffusion

[![checks](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml/badge.svg)](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-287d75.svg)](LICENSE)

Visual EEG-to-image reconstruction: code, development results, and a focused research report.

We study how visual evidence decoded from EEG survives the transition to an image diffusion model. The clearest positive result is currently **visual representation alignment**, not a validated new diffusion sampler. This is a research archive with small runnable numerical tools; it is not a pretrained end-to-end reconstruction product.

## Code

Python 3.11+; the quick start requires no GPU, EEG, pretrained weights, or downloads beyond Python dependencies.

```bash
git clone https://github.com/Botwwt/EEG-Diffusion.git
cd EEG-Diffusion
python -m pip install -e '.[test]'
eeg-diffusion verify
eeg-diffusion reference-demo
python -m pytest -q
python scripts/build_report.py --check
```

| Location | What is available |
|---|---|
| [`src/eeg_diffusion/`](src/eeg_diffusion) | CPU metrics, paired query bootstrap, classical Gaussian reference, integrity checks |
| [`experiments/archive/`](experiments/archive) | Byte-identical historical training, sampling and diagnostic sources; **server-dependent** |
| [`results/`](results) | Selected aggregate evidence and explicit evaluation protocols |
| [`provenance/`](provenance) | Source hashes, origins and import inventory |

The archive includes intermediate-layer alignment, conditional priors, cross-fitting, direct image gradients, precision score-PDE, residual correction and SORT. See the [experiment-to-code index](experiments/README.md). Original upstream model classes are obtained separately and retain their licenses. The GPU experiments require external data, checkpoints and the original asset contracts; they are **not** reproduced by the quick start. [Reproduction scope →](docs/REPRODUCIBILITY.md)

## Research report

**[Read the HTML report →](https://botwwt.github.io/EEG-Diffusion/)** · [Open its source](docs/index.html) · [Experiment notes](docs/EXPERIMENTS.md)

The compact Chinese report contains three paper readings with **original English screenshots**, highlighted phrases, our paired results, fixed generated examples and the proposed next experiment. All paper figures are attributed. There is no slide deck requirement.

To view locally, open `docs/index.html`. To regenerate its charts and text from the released aggregates:

```bash
python scripts/build_report.py
```

## What the evidence says

| Experiment | Observed result | Scope |
|---|---|---|
| Intermediate CLIP layer + joint head | **81% / 83%** retrieval Top-1; final-layer joint head 35.5% / 33.5% | One subject, mean of 80 repeats, 200 candidates, two fixed training seeds |
| Correct vs shuffled EEG, actual SDXL outputs | DINOv3 2AFC difference **+0.3519**, archived paired-image interval [0.2569, 0.4513] | 32 fixed development queries; one subject and training seed; 200-gallery Top-1 only **1/32** |
| Standard v-parameterized prior | Repairs the tested epsilon configuration; advantage over the direct EEG query unconfirmed | Separate 32-concept image experiment; a borrowed baseline tool |

These are different systems/protocols, not a single performance trajectory. Retrieval accuracy is not image reconstruction accuracy. Reused development stimuli, limited subjects and fixed training seeds do not establish independent confirmation or SOTA. More complex mechanisms have not yet produced a stable original image-generation advantage. [Definitions and negative results →](docs/EXPERIMENTS.md)

## Next question

Can an independently checked model of EEG visual-observation error improve image content under a fixed generator and sampling budget? We plan to separate measurement error from visual mapping error and denoising uncertainty. This is **a proposal, not a completed method**. Classical Gaussian posterior formulas and TMPD are references, not our new theory.

## Reuse and citation

Original code: [MIT](LICENSE). Paper screenshots, external models and stimuli have separate rights; see [third-party notices](THIRD_PARTY_NOTICES.md). Raw EEG, stimulus photographs, participant records, weights and operational credentials are not distributed.

Please cite the upstream methods used in your experiment as well as the exact release/commit of this archive. [`CITATION.cff`](CITATION.cff) describes this software archive, not an accepted research paper. Contributions should include a minimal example and identify the evaluation unit; see [CONTRIBUTING.md](CONTRIBUTING.md).

