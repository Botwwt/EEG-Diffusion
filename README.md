# EEG-Diffusion

EEG-to-image reconstruction from one participant's physical EEG presentations. We measure whether each generated image depicts the viewed object and how closely it preserves the source photograph.

[ICME 2027 manuscript and experiment package](research/icme2027/README.md) · [中文实验总结](docs/OBJECT_LOG_REVIEW.md) · [Training and inference](research/eeg_stack_20261112/PAPER_BENCHMARK_PROTOCOL.md) · [Public inputs](research/eeg_stack_20261112/public_inputs/README.md)

[![checks](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml/badge.svg)](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml)

## Official THINGS-EEG2 reconstruction

The full common-renderer experiment covers ten participants, 200 unseen concepts and two physical repeat budgets. Each method produces one image per query from EEG. All six methods use the same generator, predicted layout, empty prompt and seed schedule. The evaluator uses the same official gallery and complete 1,854-entry vocabulary for every method.

Each row summarizes 2,000 reconstructions. Naming is Top-1 over 1,854 names, and photo retrieval is Top-1 over 200 originals. Bold marks the best point estimate for each metric at a given repeat budget.

### R4 · Four EEG presentations

| Method | CLIP name ↑ | SigLIP2 name ↑ | CLIP photo ↑ | DINOv2 photo ↑ | PixCorr ↑ | SSIM ↑ | AlexNet(2) ↑ | AlexNet(5) ↑ | Inception ↑ | CLIP two-way ↑ | EffNet ↓ | SwAV ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Brain-HiVE | 0.35% | 0.50% | 3.05% | 4.45% | 0.1647 | 0.3719 | 0.7518 | 0.7684 | 0.6105 | 0.6495 | 0.9459 | 0.6356 |
| ATM | 0.40% | 0.36% | 3.20% | 3.85% | 0.1608 | 0.3592 | 0.7397 | 0.7564 | 0.6009 | 0.6336 | 0.9515 | 0.6438 |
| NICE | 0.45% | 0.36% | 3.10% | 3.20% | 0.1580 | 0.3613 | 0.7220 | 0.7459 | 0.5985 | 0.6274 | 0.9537 | 0.6439 |
| EEGNet | 0.20% | 0.45% | 2.90% | 3.90% | 0.1580 | 0.3529 | 0.7250 | 0.7486 | 0.5910 | 0.6311 | 0.9532 | 0.6443 |
| Ridge | 0.30% | 0.15% | 1.25% | 1.55% | 0.1719 | 0.3998 | 0.6644 | 0.6676 | 0.5524 | 0.5722 | 0.9494 | 0.6186 |
| EEG-Diffusion (ours) | **1.00%** | **0.92%** | **6.40%** | **8.20%** | **0.1883** | **0.4182** | **0.7911** | **0.8201** | **0.6715** | **0.7073** | **0.9089** | **0.5830** |

### R80 · Eighty EEG presentations

| Method | CLIP name ↑ | SigLIP2 name ↑ | CLIP photo ↑ | DINOv2 photo ↑ | PixCorr ↑ | SSIM ↑ | AlexNet(2) ↑ | AlexNet(5) ↑ | Inception ↑ | CLIP two-way ↑ | EffNet ↓ | SwAV ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Brain-HiVE | 0.95% | 1.32% | 6.45% | 7.55% | 0.1862 | 0.3617 | 0.8009 | 0.8441 | 0.6719 | 0.7284 | 0.9176 | 0.5952 |
| ATM | 0.50% | 0.56% | 4.85% | 5.55% | 0.1761 | 0.3520 | 0.7812 | 0.8233 | 0.6418 | 0.6916 | 0.9331 | 0.6147 |
| NICE | 0.80% | 0.46% | 4.25% | 5.20% | 0.1671 | 0.3488 | 0.7686 | 0.8165 | 0.6420 | 0.6843 | 0.9352 | 0.6108 |
| EEGNet | 0.40% | 0.46% | 3.90% | 3.95% | 0.1745 | 0.3549 | 0.7575 | 0.7996 | 0.6140 | 0.6709 | 0.9435 | 0.6152 |
| Ridge | 0.35% | 0.35% | 1.30% | 1.85% | 0.1817 | 0.3968 | 0.6616 | 0.6717 | 0.5623 | 0.5771 | 0.9482 | 0.6146 |
| EEG-Diffusion (ours) | **2.40%** | **2.08%** | **11.45%** | **16.25%** | **0.2384** | **0.4267** | **0.8769** | **0.9158** | **0.7782** | **0.7990** | **0.8540** | **0.5272** |

The paired gains in DINOv2 exact-photograph retrieval are 3.75 percentage points [1.05, 6.70] at R4 and 8.70 [5.25, 12.30] at R80. The SigLIP2 naming intervals include zero. The separate native-system comparison includes ATM, CognitionCapturer, Brain-HiVE and ENIGMA. Our method reaches 0.4267 SSIM, and ENIGMA leads in both exact-photograph rankers.

[Detailed scores and DINOv2 appendix](research/icme2027/experiments/full-metric-benchmark/RESULTS.md) · [Native-system table](research/icme2027/NATIVE.md) · [R80 fixed samples](research/icme2027/experiments/full-metric-benchmark/figures/common-r80-fixed-panel.png)

## Method

The image condition combines a repeat-weighted EEG waveform with features from individual presentations. A separate EEG readout predicts the coarse starting layout. During training, the model draws four images from EEG for four photographs of the same concept using shared diffusion noise. It learns a bidirectional assignment between the generated images and their individual sources. The loss differentiates through all four denoising steps.

The two layout controls and their scores are available in the manuscript package. The compact personal decoder and the full set of historical protocols are available in the archive.

[Method and hyperparameters](research/icme2027/paper/source/manuscript.tex) · [Training controls](research/icme2027/README.md#结果) · [Compact single-query baseline](research/eeg_stack_20261112/code/paper_object_infer.py)

## Reproduce

[The ICME 2027 package](research/icme2027/README.md#复核与复现) includes the 24,000-image common-renderer score packets, model and stimulus bindings, and scripts for verification and table reconstruction. [GPU replay](research/eeg_stack_20261112/PAPER_BENCHMARK_PROTOCOL.md) and the [input guide](research/eeg_stack_20261112/public_inputs/README.md) describe the EEG, images, and pretrained weights needed for generation.

The historical paired benchmark covers 368 participant–method–budget combinations and 103,600 scored image outputs, including released ATM images. The archive stores individual observations and input hashes.

```bash
python -m pip install -e '.[dev]'
python research/eeg_stack_20261112/code/package_paper_scores.py --unpack research/eeg_stack_20261112/results/paper_benchmark_scores.zip outputs/paper-benchmark/verified
python research/eeg_stack_20261112/code/build_paper_benchmark_report.py --root outputs/paper-benchmark/verified
python -m pytest -q
```

## Experiment archive

- [Historical experiments by research question](research/eeg_stack_20261112/README.md)
- [Paired participant-level benchmark](docs/PAPER_BENCHMARK_RESULTS.md) and [earlier experimental decisions](docs/OBJECT_LOG_REVIEW.md)
- [ObjEEG-Diff report](research/eeg_stack_20261112/OBJ_RECOG_BENCHMARK_REPORT.md) for test-time adaptation and candidate selection
- [Calibrated object task](docs/OBJECT_FIDELITY.md), [12-object replication](docs/TRIAL_OBJECT_PERSONAL12.md), and [galleries](research/eeg_stack_20261112/gallery/GALLERY.md)

[Citation](CITATION.cff) · [MIT license](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
