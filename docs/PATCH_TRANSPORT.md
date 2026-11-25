# Can moving pixels recover the viewed photograph?

For sub-04, we tried moving parts of a frozen four-trial EEG reconstruction to match local visual features predicted from EEG. The basketball hoop, birdhouse and checkerboard in the fixed gallery made the drawback visible. Moving a small number of control points could bend a rigid object. We repeated the experiment with a bound on local stretching. Both versions missed the registered photo-identification threshold. [First design](../research/eeg_stack_20261112/results/object_fidelity/patch_transport_design.json) · [Stretch-limited follow-up](../research/eeg_stack_20261112/results/object_fidelity/patch_transport_limited_design.json)

Each test used 120 photos and twelve objects. All twelve objects shared one three-person recording-session combination in the stress panel. The other panel gave the object decoder an explicit four-word menu per query. Both panels had been evaluated before. Every editing arm started from the same saved JPEG, decoded word and eight permitted support photographs. The main scorer used LAION ViT-H/14 cosine to identify the stimulus among ten photographs of its true object. The generator uses H/14 conditioning. The editing objective uses DINOv2 local patches. [Source photographs, folds and acquisition protocol](COHORT_SESSION_CONTROLS.md)

| Panel and editor | Neutral JPEG re-encoding | Eight-support mean | Matched EEG | Exchanged EEG | True patch field |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stress, initial | 17 | 13 | 11 | 11 | 13 |
| Four-word menus, initial | 15 | 13 | 13 | 13 | 13 |
| Stress, stretch-limited | 17 | 15 | 16 | 18 | 18 |
| Four-word menus, stretch-limited | 15 | 15 | 14 | 15 | 15 |

The original source JPEGs before re-encoding scored 19/120 and 14/120. The original DINOv2 CLS results, 15/120 and 14/120, reproduced for every query. With the first editor, supplying the real target patch field reduced RGB RMSE by 0.0052 and 0.0057 relative to the neutral JPEG and raised luminance correlation by 0.023 and 0.036. The same positive control recovered 13/120 photos on each panel under H/14. It used the original photograph as input to measure the editing method's capacity. The matched EEG arms only used the eight permitted supports and the measured query EEG. [Initial measurements](../research/eeg_stack_20261112/results/object_fidelity/patch_transport.json) · [Follow-up measurements](../research/eeg_stack_20261112/results/object_fidelity/patch_transport_limited.json)

## Method and replay

A personal ridge regressed 1,003 posterior EEG values onto sixteen local DINOv2-S/14 descriptors. During fitting we removed each calibration object's average from EEG and visual features. A split of 416 other objects for ridge training and 96 objects for calibrating a single shrinkage coefficient gave a coefficient of 0.520. The squared-error reduction was 0.62% on the same 96 objects used to choose that coefficient. The final ridge used all 512 objects. For a query, its predicted object supplied the mean EEG and patch descriptors of eight support photos outside its fold. The participant's EEG deviation from their support mean predicted a visual deviation. The exchanged control substituted the EEG deviation of a query with the same predicted word and fold.

The editor resampled the pixels of the frozen reconstruction with a smooth two-dimensional displacement field. The first version bounded each knot displacement. The follow-up also bounded the displacement derivative norm by 0.25, limiting local changes in shape. Both used 32 Adam steps and a four-by-four grid of interior knots. Oracle renders ran after all blind outputs for both panels were saved and hashed. Uncertainty intervals resample the twelve objects together with their ten photos. [Training and editing](../research/eeg_stack_20261112/code/fit_patch_transport.py) · [JPEG, reference and rank checks](../research/eeg_stack_20261112/code/score_patch_transport.py)

Prepare the original [session-matched inputs and rendered images](COHORT_SESSION_CONTROLS.md#replay), generic posterior EEG, verified stimulus ZIP, DINOv2-S/14 and LAION H/14 weights. From the repository root, use an environment with PyTorch, Transformers, SciPy, NumPy and Pillow.

```sh
C=research/eeg_stack_20261112/code
python "$C/fit_patch_transport.py" fit
for mode in original limited; do
  flag=()
  if [ "$mode" = limited ]; then flag=(--limited); fi
  python "$C/fit_patch_transport.py" blind "${flag[@]}"
  python "$C/fit_patch_transport.py" oracle "${flag[@]}"
  python "$C/score_patch_transport.py" "${flag[@]}"
done
```

The local comparison sheets show the lowest official row for each of the twelve objects. Read across each group of four for the original, frozen generation, matched EEG edit and target-informed diagnostic. Both correct and incorrect decoded words are included. Files are under `outputs/patch-transport/{stress12,new_three4}/comparison.jpg` and the corresponding `limited/` directory.
