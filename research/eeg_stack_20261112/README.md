# November EEG reconstruction experiments

This directory contains the November Brain-HiVE / SDXL-Turbo studies. The EEG-only benchmark scores the object depicted in a single image first, then its match to the viewed photograph. M1a + D2 at s375 was selected on the historical within-object R4 task. In the ten-person four-repeat comparison, personal Brain-HiVE and personal D2 tied at 14/2,000 exact-name credits. The shared model's single-query program accepts four or 80 actual EEG repeats from one participant, a registered checkpoint set and a seed. [Benchmark and training protocol](PAPER_BENCHMARK_PROTOCOL.md) · [Results](../../docs/PAPER_BENCHMARK_RESULTS.md) · [Model history](../../docs/OBJECT_LOG_REVIEW.md) · [Shared single-query program](code/paper_shared_infer.py)

| What to read | Where to look |
| --- | --- |
| ObjEEG-Diff and its 13-method object-recognizability benchmark. The report includes test-set adaptation and four-candidate selection experiments. | [ObjEEG-Diff report](OBJ_RECOG_BENCHMARK_REPORT.md) · [All-10 stats](results/objbench_all10_stats.json) |
| The full ten-participant R80 comparison, with four original-system baselines scored against the same photographs | [Unified reconstruction tables](../../docs/PAPER_UNIFORM_RECONSTRUCTION.md) |
| Open naming before generation, ground-truth token rank, and full-dimensional visual flow | [Readout and intervention results](TOKEN_GATE_ZERO_STUDY.md) |
| Which denoising steps carry EEG information through the image Value projection | [Fixed-input renders and EEG-EditBench tables](BACKGROUND_EDITBENCH_STUDY.md) |
| Estimated layout responses, a reversed-response control, and real repeats from different photographs of the same object | [Physical EEG readouts and matched images](LAYOUT_RESPONSE_STUDY.md) |
| Continuous, posterior mean, and posterior mode estimates from the same EEG readout | [Categorical and continuous readouts](CATEGORICAL_READOUT_STUDY.md) |
| Training-photo ERP prediction, counterfactual image scoring, and held-concept guidance | [Image-to-EEG scoring and controls](NEURAL_ENCODING_STUDY.md) |
| Combining D2 and O13 before or after image cross-attention softmax | [Late-attention controls and scores](O13_RESPONSE_MIX_STUDY.md) |
| Visual supervision at five depths of the frozen vision tower used by BRIDGE | [Matched readout study](BRIDGE_DEPTH_STUDY.md) |
| Replaying O13 visual composition with four EEG observations and frozen training statistics | [Study and replay](O13_COMPOSITION_STUDY.md) · [Paired scores and input receipts](results/o13_composition.json) |
| Replacing the image condition at one denoising step with a reference photograph | [Study and replay](D2_STAGE_REFERENCE_STUDY.md) · [Paired scores and input receipts](results/d2_stage_reference.json) |
| Training with cross-object EEG exchanges and evaluating open generation with a fixed layout | [Study and replay](D2_OBJECT_SWAP_STUDY.md) · [Paired scores and input receipts](results/d2_object_swap.json) |
| Measuring EEG condition spread at the generator and testing a matched correction to shared D2 | [Study and replay](CONDITION_CALIBRATION_STUDY.md) · [Scores and input receipts](results/condition_calibration.json) |
| CBraMod joint attention across measured repeats and an independent image naming replay of D2 | [Study and replay](CBRAMOD_REPEAT_STUDY.md) · [Paired scores and training receipts](results/cbramod_repeats.json) |
| Local repeat-spectrum readouts and matched EEG swaps for whole-photo reconstruction | [Study and replay](LOCAL_SPECTRA_STUDY.md) · [Image scores and input receipts](results/local_spectra.json) |
| Whole-photo inference with measured EEG consistency and an equal-norm visual direction diagnostic | [Study and replay](FUSED_POSTERIOR_STUDY.md) · [Image scores and input receipts](results/fused_posterior.json) |
| Matching EEG-derived local visual conditions to spatial attention | [Study and replay](SPATIAL_PHOTO_STUDY.md) · [Image scores and input receipts](results/spatial_photo.json) |
| I-JEPA masking errors as supervision for object information in EEG generation | [Study and replay](IJEPA_CONTEXT_STUDY.md) · [Image scores and input receipts](results/ijepa_context.json) |
| Channel weighting, latency alignment and the position of pooling over EEG repeats | [Study and replay](REPEAT_POOLING_STUDY.md) · [Image scores and physiological checks](results/repeat_pooling.json) |
| Replacing EEG image-token subspaces selected by behavior-derived attributes | [Study and replay](BEHAVIORAL_TOKEN_SUBSPACE_STUDY.md) · [Readout scores](results/behavioral_token_subspace.json) |
| Paired flow matching and conditional means for relative EEG image tokens | [Study and replay](PAIRED_TOKEN_FLOW_STUDY.md) · [Paired image scores](results/paired_token_flow.json) |
| Training the EEG encoder with exchanged neighboring stimulus responses | [Study and replay](RSVP_CONTEXT_TRAINING_STUDY.md) · [Paired image scores](results/rsvp_context_training_study.json) |
| Foreground-weighted visual training targets for object decoding | [Study and replay](FOREGROUND_TARGET_STUDY.md) · [Readout scores](results/foreground_target_study.json) |
| Repeat-noise calibration and training-event overlap | [Study and replay](REPEAT_NOISE_READOUT_STUDY.md) · [Paired scores](results/repeat_noise_readout_study.json) |
| Object-specific EEG training with background-matched photo pairs and blind image generation | [Study and replay](OBJECT_FOREGROUND_STUDY.md) · [Paired scores](results/object_foreground_study.json) |
| Continuous EEG semantic conditioning with a matched 80-repeat Brain-HiVE comparison | [Study and replay](REPEAT_SEMANTIC_BRIDGE_STUDY.md) · [Paired image scores](results/repeat_semantic_bridge.json) |
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
