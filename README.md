# EEG-Diffusion

EEG-to-image reconstruction from one participant's physical EEG presentations. We measure whether each generated image depicts the viewed object and how closely it preserves the source photograph.

[ICME 2027 manuscript and experiment package](research/icme2027/README.md) · [中文实验总结](docs/OBJECT_LOG_REVIEW.md) · [Training and inference](research/eeg_stack_20261112/PAPER_BENCHMARK_PROTOCOL.md) · [Public inputs](research/eeg_stack_20261112/public_inputs/README.md)

[![checks](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml/badge.svg)](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml)

## Official THINGS-EEG2 reconstruction

The full common-renderer experiment covers ten participants, 200 unseen concepts and two physical repeat budgets. Each method produces one image per query from EEG. All six methods use the same generator, predicted layout, empty prompt and seed schedule. The evaluator uses the same official gallery and complete 1,854-entry vocabulary for every method.

| EEG presentations | DINOv2 exact-photo Top-1 | Best baseline | SigLIP2 object naming | Best baseline |
| --- | ---: | ---: | ---: | ---: |
| 4 | 8.20% | 4.45% | 0.925% | 0.500% |
| 80 | 16.25% | 7.55% | 2.075% | 1.325% |

The paired gains in DINOv2 exact-photograph retrieval are 3.75 percentage points [1.05, 6.70] at R4 and 8.70 [5.25, 12.30] at R80. The SigLIP2 naming intervals include zero. The separate native-system comparison includes ATM, CognitionCapturer, Brain-HiVE and ENIGMA. Our method reaches 0.4267 SSIM, and ENIGMA leads in both exact-photograph rankers.

[All 12 common-renderer metrics](research/icme2027/experiments/full-metric-benchmark/RESULTS.md) · [Native-system table](research/icme2027/NATIVE.md) · [R80 fixed samples](research/icme2027/experiments/full-metric-benchmark/figures/common-r80-fixed-panel.png)

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
