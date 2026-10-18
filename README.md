# EEG-Diffusion

**Visual EEG-to-image reconstruction — numerical tools, experiments and evidence.**

[Research report](https://botwwt.github.io/EEG-Diffusion/) · [Numerical API](docs/API.md) · [Experiment index](experiments/README.md) · [Reproduction scope](docs/REPRODUCIBILITY.md)

[![checks](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml/badge.svg)](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml)
[![Code license: MIT](https://img.shields.io/badge/code-MIT-35677c.svg)](LICENSE)

We study how visual evidence decoded from EEG can guide image diffusion. This repository provides a tested CPU numerical core, frozen historical experiment sources, aggregate results and a bilingual research report. The clearest positive evidence is currently representation alignment; a stable original diffusion mechanism advantage on actual images has not been established.

## Run the code

Python **3.11+**, Linux or Windows. Use a virtual environment. The core requires NumPy; no GPU, EEG data or model weights are needed for this demo.

```bash
git clone https://github.com/Botwwt/EEG-Diffusion.git
cd EEG-Diffusion
python -m pip install -e '.[dev,report]'

eeg-diffusion reference-demo
eeg-diffusion verify
python -m pytest -q
```

The synthetic demo returns a posterior mean of `[0.4]`, covariance `[[0.5]]`, and two unique trials, up to floating-point precision. `verify` checks frozen-source and evidence hashes plus displayed arithmetic. Neither command retrains an encoder or regenerates historical images. [Installation, tested versions and measured demo times](docs/INSTALLATION.md)

```python
from eeg_diffusion.metrics import image_two_way, retrieval_top1

scores = [[0.8, 0.9, 0.1]]
target = [0]
print(retrieval_top1(scores, target))  # [0.]
print(image_two_way(scores, target))   # [0.5]
```

Pairwise identification and gallery Top-1 measure different things. Shapes, assumptions, tie handling, correlated-repeat conditioning and bootstrap units are documented in the [API reference](docs/API.md).

## Read the report

### [Visual EEG-to-image reconstruction ↗](https://botwwt.github.io/EEG-Diffusion/)

Three primary-paper readings use original English screenshots and highlighted passages. The report then presents paired results, fixed generated examples, unsuccessful mechanisms and the next research question. Statistical plots are available as editable SVG and vector PDF, with source data and rendering metadata.

```bash
python scripts/build_report.py          # HTML + source-data CSV
python scripts/build_figures.py         # SVG + PDF, requires [report]
python scripts/build_report.py --check
python scripts/build_figures.py --check
```

Open `docs/index.html` locally or use GitHub Pages. Figure captions distinguish seed points, aggregate means and archived intervals. [Figure and software reporting choices](docs/FIGURE_AND_CODE_STANDARDS.md)

## Evidence at a glance

| Study | Observed result | Evaluation boundary |
|---|---|---|
| Layer-12 joint visual alignment | **81% / 83%** Top-1; final-layer joint head **35.5% / 33.5%** | One subject, mean80, 200 candidates, two fixed training seeds; retrieval |
| Correct versus shuffled EEG | Image 2AFC difference **+0.3519**, archived interval **[0.2569, 0.4513]** | 32 fixed development queries, one trained model; 200-gallery Top-1 **1/32** |
| Standard v prior | Repairs the tested epsilon configuration; advantage over the direct query unconfirmed | Separate actual-image experiment; a borrowed baseline tool |

These are separate systems and protocols. Retrieval is not image reconstruction accuracy. Reused development stimuli and limited subjects do not establish independent confirmation or SOTA. The [experiment record](docs/EXPERIMENTS.md) includes negative results and limits.

## Repository map

```text
src/eeg_diffusion/       Typed numerical core and CLI
tests/                  Analytic, regression and integrity tests
scripts/                Reproducible HTML and figure builders
docs/                   Report, API and reproduction documentation
results/                Aggregate evidence and source-data CSV
provenance/             Hashes, figure metadata and refactor history
experiments/archive/    Frozen executed sources; server-dependent
```

The archive covers visual alignment, conditional priors, cross-fitting, image gradients, precision score-PDE, repeat composition, residual correction and SORT. It retains original paths and asset contracts. Complete GPU replay requires separately obtained data, weights, caches and the original environment; it is not delivered by the quick start. [Inputs and dependencies](docs/REPRODUCIBILITY.md)

## Reproducibility and reuse

- **Validation:** [local results](docs/VALIDATION.md); CI checks Python 3.11/3.12 on Linux and Python 3.11 on Windows. Ruff applies to active code; historical executed files remain unchanged.
- **Data availability:** selected aggregates, figure data and fixed output crops are included. Participant EEG, stimulus photographs and pretrained weights are external.
- **Code availability:** original code is [MIT licensed](LICENSE). Upstream models and paper screenshots retain their own rights; see [third-party notices](THIRD_PARTY_NOTICES.md).
- **Versioning:** cite the exact commit or software version using [CITATION.cff](CITATION.cff), and cite the upstream methods used. This archive is not an accepted method paper and has no minted DOI.

Contributions should include a minimal example and the correct statistical unit. See [CONTRIBUTING.md](CONTRIBUTING.md).
