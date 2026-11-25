# EEG-Diffusion

Reconstructing a viewed photograph from one person's EEG. The primary outcome is whether the generated image depicts the right object. The second is whether it preserves the particular photograph among ten views of that object.

[中文实验总结](docs/OBJECT_LOG_REVIEW.md) · [Benchmark results](docs/PAPER_BENCHMARK_RESULTS.md) · [Training and inference](research/eeg_stack_20261112/PAPER_BENCHMARK_PROTOCOL.md) · [Public inputs](research/eeg_stack_20261112/public_inputs/README.md)

[![checks](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml/badge.svg)](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml)

## Results from EEG alone

The THINGS-EEG2 official gallery has 200 objects and ten participants. Each method draws one image from the same participant's EEG with an empty prompt, four denoising steps and a shared stimulus-specific seed. A frozen evaluator names the generated image from 1,854 nouns. The image retrieval column counts how often the exact original ranks first among all 200 photos. Ties share credit. These matched personal fits use the same 1,323 training concepts. [Per-person scores, complete image metrics and 95% paired intervals](docs/PAPER_BENCHMARK_RESULTS.md)

| Training and model | Physical presentations | Object name / 2,000 | Original photo / 2,000 | SSIM |
| --- | ---: | ---: | ---: | ---: |
| Personal Brain-HiVE, paper projection | 4 | 14 | 94 | 0.3122 |
| Personal M1a + D2 | 4 | 14 | 88 | 0.4306 |
| Personal Brain-HiVE, paper projection | 80 | 37 | 206 | 0.3273 |
| Personal M1a + D2 | 80 | 28 | 176 | 0.4359 |
| M1a + D2, ten-person training | 80 | 51 | 228 | 0.4267 |

The personal D2 model improves four-repeat SSIM by 0.1185 (95% interval 0.1081 to 0.1288). Both personal methods depict few identifiable objects. Shared training raises the R80 naming count by 14 over personal Brain-HiVE, with a paired interval of −1.15 to +2.45 percentage points. On a second gallery of 50 held-out objects with ten photos each, the personal models identify 899 and 884 of 5,000 exact within-object photos, respectively. The protocol documents prior feature-level exposure of part of this gallery. [Paired comparisons and gallery history](research/eeg_stack_20261112/PAPER_BENCHMARK_PROTOCOL.md)

## Models

**Compact single-query baseline.** Average the selected presentations, encode the result once with a participant-specific BrainMLP, apply the frozen Brain-HiVE fusion adapter and image projection, then condition SDXL-Turbo on four visual tokens. Each draw starts from noise. The program takes preprocessed EEG, a fitted encoder and a seed. Training-only experiments compare photo and category targets, vision layers, rendered-image supervision and calibration of the visual condition. [Single-query program](research/eeg_stack_20261112/code/paper_object_infer.py) · [Experimental decisions](docs/OBJECT_LOG_REVIEW.md)

**M1a + D2.** M1a combines a repeat-weighted evoked response with individually encoded EEG presentations. A separate model predicts low-frequency layout. D2 trains four reconstructed images to match their four source photographs within the same object category. During generation the query uses only EEG. The historical 19.64% result measures the ten-photo within-object task. [Historical R4 evaluator](research/eeg_stack_20261112/code/eval_m1a_d2.py) · [Matched loss and readout ablations](docs/PAPER_BENCHMARK_RESULTS.md)

## Reproduce the benchmark

The numerical release covers all 368 prespecified participant–method–budget combinations and 103,600 scored image outputs, including the external ATM images. It contains individual observations, stimulus IDs, score definitions, and hashes of inputs and generated outputs. Rebuild the tables on a CPU with Python 3.11 or newer.

```bash
python -m pip install -e '.[dev]'
python research/eeg_stack_20261112/code/package_paper_scores.py --unpack research/eeg_stack_20261112/results/paper_benchmark_scores.zip outputs/paper-benchmark/verified
python research/eeg_stack_20261112/code/build_paper_benchmark_report.py --root outputs/paper-benchmark/verified
python -m pytest -q
```

For GPU generation, stage the THINGS EEG and photograph archives, the pinned visual targets, and the frozen Brain-HiVE and SDXL weights. The data guide verifies the 17 posterior channels and the exact row ordering. The protocol includes commands for training, rendering, scoring and one-query inference. [Input preparation](research/eeg_stack_20261112/public_inputs/README.md) · [GPU protocol](research/eeg_stack_20261112/PAPER_BENCHMARK_PROTOCOL.md) · [Environment and third-party assets](docs/REPRODUCIBILITY.md)

To inspect a complete participant after rendering, run `python "$RAD_ROOT/code/paper_full_gallery.py" sub-01 personal`. It builds a local HTML gallery in stimulus order with all 200 original photos, Brain-HiVE reconstructions and D2 reconstructions side by side. [Gallery builder](research/eeg_stack_20261112/code/paper_full_gallery.py)

## Experiment archive

| Question | Evidence |
| --- | --- |
| What improved object identity, photo details or layout in the earlier experiments? | [Historical log review](docs/OBJECT_LOG_REVIEW.md) · [Scientific experiment registry](docs/EXPERIMENT_REGISTRY.md) |
| How does the matched reconstruction compare with published methods? | [Brain-HiVE reproduction and released ATM images](docs/PAPER_BENCHMARK_RESULTS.md) · [Other authors' reported numbers and figure-level checks](research/eeg_stack_20261112/results/paper_benchmark_external.md) |
| What happens when the candidate objects and eight labeled supports per object are supplied? | [Calibrated object task](docs/OBJECT_FIDELITY.md) · [Session-controlled 12-object replication](docs/TRIAL_OBJECT_PERSONAL12.md) · [Two additional participants](docs/FRESH_PEOPLE_IMAGES.md) · [Galleries](research/eeg_stack_20261112/gallery/GALLERY.md) |

[Citation](CITATION.cff) · [MIT license](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
