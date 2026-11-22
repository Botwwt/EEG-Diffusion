# November EEG reconstruction experiments

This directory contains the Brain-HIVE / SDXL-Turbo experiments from November 12–23, 2026. See the [current results and limits](../../docs/CURRENT_RESEARCH.md) for the protocol and claims used on the project homepage.

| What to read | Where to look |
| --- | --- |
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
