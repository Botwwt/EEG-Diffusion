# EEG-guided changes to the color of a generated image

For sub-04, adjusting the generated image using the predicted object's eight support photographs lowered average color error on both session-controlled panels. Adding the query EEG reduced it a little further. The comparison against a different EEG from the same object, decoded word and fold had an interval spanning zero on both panels. The result keeps color transfer as a diagnostic for the existing generator. [Method registered before scoring](../research/eeg_stack_20261112/results/object_fidelity/eeg_color_transfer_design.json) · [Results, intervals and image hashes](../research/eeg_stack_20261112/results/object_fidelity/eeg_color_transfer.json)

## Color error on the saved images

Each row uses 120 photos. All four versions start from the same generated image and use the same JPEG settings. The table measures mean distance between the chromatic components of actual generated and original pixels in OKLab at 32 × 32, multiplied by 100. Lower is better. The last column compares the query's EEG with the support-only edit, keeping every source image paired. Intervals resample 12 whole objects.

| Candidate menu | EEG acquisition | Neutral re-encode | Support mean | Own EEG + supports | Wrong EEG + supports | Own minus support mean, 95% interval |
|---|---|---:|---:|---:|---:|---:|
| Twelve words | sub-04, four trials | 5.816 | 5.436 | 5.394 | 5.439 | −0.041 [−0.069, −0.015] |
| Four supplied words | sub-04, four trials | 5.720 | 5.332 | 5.309 | 5.344 | −0.023 [−0.044, −0.0004] |
| Twelve words | Three people, twelve trials | 5.433 | 5.104 | 5.078 | 5.087 | −0.026 [−0.038, −0.013] |
| Four supplied words | Three people, twelve trials | 5.607 | 5.303 | 5.289 | 5.303 | −0.014 [−0.025, −0.003] |

The stricter control requires a distinct donor who saw a different photo of the same true object, with the same predicted word in the same fold. On these rows, own minus wrong was +0.022 [−0.014, +0.063] for sub-04 on the twelve-choice panel (20 rows) and −0.033 [−0.072, +0.011] on the four-choice panel (39 rows). The corresponding three-person differences were +0.007 [−0.030, +0.031] and −0.011 [−0.029, +0.009]. This control was required for promotion.

On the independent, panel-centered DINOv2-S/14 same-object ten-photo ranking, neutral → own for sub-04 was 15 → 15 and 15 → 16 out of 120. The three-person results were 18 → 19 and 16 → 13. The released table also includes the raw-cosine DINO score used by the original session-panel analysis. Every image in this experiment retains its original decoded word and spatial arrangement. These two panels had already been studied with other methods. [Session controls and acquisition budgets](COHORT_SESSION_CONTROLS.md)

## What changed in the image

The earlier spatial method adjusted an 8 × 8 grid of local brightness, color and contrast. This method estimates a color distribution separately for shadows, midtones and highlights. Each training photograph supplies 24 numbers by measuring three quantiles of two color differences in each of four luminance-rank bands. The input uses four presentations and the original 17 posterior electrodes. A fixed ridge regression removes the training object's mean from both EEG and target color. A single shrinkage factor per participant was fitted on 96 generic classes after training on the other 416. Final coefficients used all 512 generic objects. Shrinkage was 0.642 for sub-04, 0.595 for sub-05 and 0.539 for sub-01. [Earlier spatial correction](SPATIAL_PHOTO.md)

For each held-out query the model starts with the average palette of the predicted object's eight permitted support photos. It adds the predicted EEG deviation from their mean. In the three-person condition, the three personal deviations receive equal weight. The image transform interpolates between source and target color quantiles, applies half the bounded displacement and keeps BT.601 luminance constant before JPEG encoding. The evaluation opens the saved JPEGs, verifies their hashes, and compares them with the original photographs. The four acquisition-by-panel cases produced 1,920 JPEGs from 480 generated source images of 240 different target photographs.

The comparison sheet uses the lowest stimulus row in every object class for sub-04 and shows two generated images per item. Original-photo comparisons remain in the private score folders. [Fixed generated-image comparisons](../research/eeg_stack_20261112/gallery/eeg_color_transfer_sub04.jpg)

## Replay

Use the original stimulus ZIP and row manifest, expanded generic training features, both session-matched render roots and their cached independent DINO reference embeddings. The scoring environment needs the pinned DINOv2-S/14 checkpoint and compatible PyTorch and Transformers packages. Run from the repository root. [Implementation](../research/eeg_stack_20261112/code/fit_eeg_color_transfer.py)

```sh
C=research/eeg_stack_20261112/code
export OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3
for stage in fit infer render; do
  "$PY" "$C/fit_eeg_color_transfer.py" "$stage" || exit 1
done
"$GPU_PY" "$C/fit_eeg_color_transfer.py" score
"$PY" "$C/fit_eeg_color_transfer.py" gallery
```

The private outputs are under `outputs/eeg-color-transfer/`. Each `renders/PANEL/MODE/settings.json` records the source image, condition packet and output JPEG hashes. Each `scores/PANEL/MODE/` contains per-row measures and a fixed comparison with the original stimuli.
