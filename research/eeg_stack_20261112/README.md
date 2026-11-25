# November EEG reconstruction experiments

This directory contains the November Brain-HiVE / SDXL-Turbo studies. The EEG-only benchmark scores the object depicted in a single image first, then its match to the viewed photograph. M1a + D2 at s375 was selected on the historical within-object R4 task. In the ten-person four-repeat comparison, personal Brain-HiVE and personal D2 tied at 14/2,000 exact-name credits. The single-query program accepts a personal EEG array, an encoder checkpoint and a seed. [Benchmark and training protocol](PAPER_BENCHMARK_PROTOCOL.md) · [Results](../../docs/PAPER_BENCHMARK_RESULTS.md) · [Model history](../../docs/OBJECT_LOG_REVIEW.md) · [Single-query program](code/paper_object_infer.py)

| What to read | Where to look |
| --- | --- |
| **Trial-Precision Dual-Stream (`M1`) & `DINOv2` Geometric DRaFT (`M2`) (2026-11-24)**: breaks `R = 4` SNR ceiling (`19.64%` All-10 / `19.73%` HO-6 `CLIP Intra10`, `p = 0.0292` symmetric!) | **[Trial-Prec & DINOv2 report](TRIAL_PREC_DINOV2_REPORT.md)** · [M1 All-10 JSON](results/exp8c_m1a_heldout6_and_all10_results.json) · [M2 All-10 JSON](results/exp9b_m2_and_m1m2_all10_results.json) · [7-Col Gallery](gallery/m1_m2_heldout_sub01_s375_gallery.jpg) |
| **`z0 × cond` `2×2` Diagnostics & 4-Step `DRaFT-Intra4` (2026-11-24)**: anchoring (`s375`), 4-way full-chain competition, All-10 & Held-Out-6 | **[DRaFT-Intra4 report](DRAFT_INTRA4_REPORT.md)** · [All-10/HO-6 JSON](results/exp7_heldout6_and_all10_results.json) · [Held-Out gallery](gallery/draft_heldout_sub01_s375_gallery.jpg) |
| **SwapDiff-EEG (2026-11-23)**: within-concept `2×2` EEG-swap denoising contrast (`R = 4`, All-10 & Held-Out-6) | **[SwapDiff report](SWAPDIFF_REPORT.md)** · [10-sub JSON](results/swapdiff_10sub_results.json) · [Swap gallery](gallery/swap_gallery_heldout.jpg) |
| O12–O14 experiment design, controls and figures | [Experiment report](O12_O13_O14_REPORT.md) |
| Participant-level Top-1 and paired statistics | [Released result log](results/o12_o13_o14_info_ceiling_xbrain.txt) |
| Sequential examples, failure cases and condition labels | [Reconstruction gallery](gallery/GALLERY.md) |
| Other configurations and earlier results | [Experiment history](EXPERIMENT_HISTORY.md) and [results](results/) |
| Cluster-specific scripts and asset download lists | [Job archive](jobs/README.md) |

## GPU replay

Set `RAD_ROOT` to a workspace containing the original EEG and stimuli, model and evaluator weights, Brain-HIVE source, and prepared caches. The [reproduction guide](../../docs/REPRODUCIBILITY.md#november-gpu-replay) lists the stages. The [environment and preparation recipes](EXPERIMENT_HISTORY.md) document the original runtime.

```bash
export RAD_ROOT=/path/to/rad
python -m eeg_diffusion check-o13-inputs --rad-root "$RAD_ROOT"
```

With the O10 prerequisites available, run the appropriate executor from this directory in the archived GPU environment.

```bash
python code/o12_splitrep_medoid_bigg.py --phase all
python code/o13_xbrain_shared_trunk.py --phase all
python code/o14_triarch_xbrain_bigg.py --phase all
```

O12 and O13 use O10 conditions and starting latents. O14 also uses O13's adapted predictions. The old GPU executors assume the original THINGS-EEG2 R80 protocol, including the ordered stimulus groups and five predefined seeds used by O13/O14 adaptation.

From the repository root, `python -m eeg_diffusion verify-stack` recomputes Top-1 arithmetic from the released log without a GPU.
