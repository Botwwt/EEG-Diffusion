# EEG-guided spatial correction of one generated photograph

This experiment asks whether EEG identifies where a particular photograph was brighter, darker or differently colored after the object and a generated image are already available. The input image is the frozen one-draw, angle-calibrated SDXL/IP-Adapter reconstruction. All image variants use that same input, the same decoded object word, and the same eight calibration photographs for that word and fold. [Angle-calibrated generator](PHOTO_STRUCTURE.md#calibrating-the-residual-angle-before-generation) · [Spatial method lock](../research/eeg_stack_20261112/results/object_fidelity/spatial_photo_design.json)

## What the decoder learns

Each participant contributed 5,120 photos of 512 calibration objects, each with four EEG presentations. Training removed each object's mean from both modalities. The regression used 1,003 features from the 17 posterior electrodes and a fixed ridge penalty of 0.3 times the number of training photos. Four 8×8 fields represented mean luminance, blue minus luminance, red minus luminance, and log local luminance standard deviation. A signed two-dimensional discrete cosine transform represented each field. Four object-disjoint training folds estimated the reliability of each field in four frequency bands and the editing strength. To estimate editing strength on these training objects, one different, same-object calibration photo supplied a surrogate input image. Each of the four editing strengths reached the prespecified ceiling of 0.5 in all three participants. [Fit, support checks and image transform](../research/eeg_stack_20261112/code/fit_spatial_photo.py)

For a query, the decoder subtracts the mean EEG of the eight supports for its predicted object. Adding its predicted visual deviation to their spatial mean supplies the target image layout. The transform interpolates the local brightness and color changes across the input image and adjusts local luminance contrast. The local contrast multiplier is bounded between 0.75 and 1.33. The absolute change in each RGB channel is capped at 0.20 on the 0–1 scale.

The seven output arms are the byte-copied angle image, a neutral JPEG re-encoding, the eight-support spatial prior, that prior plus the query EEG, and the same prior plus a different query's EEG from the same predicted object and fold. Two additional images apply the earlier 128-object 4×4 RGB model to this same angle image, once with its class mean and once with its EEG prediction. All edited outputs use the same JPEG settings. The comparison with the earlier model changes the training budget, representation and image transform together. [Original 4×4 model](../research/eeg_stack_20261112/code/refine_personal_photo_residual.py)

## Results on the original 24 objects

Each participant saw the same 240 photographs. In sub-04, sub-05 and sub-01 respectively, 180, 147 and 143 queries received the correct object word. Restricting to correct-word queries with a different donor that also shared the true object, predicted word and fold left 124, 82 and 76 queries. The intervals below resample complete true objects within each participant. The strict donor comparison retains the recipient's own frozen angle image and changes only the added spatial residual.

| Participant | Correct-word luminance correlation, prior → own | Strict-donor grayscale SSIM, own minus wrong [95% interval] | DINO correct assignments, base / prior / own / wrong |
| --- | ---: | ---: | ---: |
| Sub-04 | 0.07156 → 0.07514 | +0.00085 [+0.00014, +0.00156] | 48 / 49 / 49 / 46 of 72 |
| Sub-05 | 0.06910 → 0.07193 | +0.00126 [+0.00016, +0.00260] | 30 / 32 / 32 / 30 of 51 |
| Sub-01 | 0.09394 → 0.09751 | +0.00112 [+0.00007, +0.00196] | 25 / 27 / 26 / 25 of 47 |

The paired interval for the luminance-correlation gain was above zero in each participant. Own versus prior RGB RMSE intervals on correctly named queries included zero in all three. DINOv2-S/14 assigned **107/170** original-photo pairs correctly with own EEG and **108/170** with the spatial prior. In the same-object ten-photo gallery, own EEG identified **79/720** originals and the spatial prior identified **78/720**. The earlier 4×4 EEG correction applied to the same angle images identified **82/720**. The fresh lock's narrative recorded the old own-EEG total as 78. The score file it fingerprints contains 30, 24 and 25 hits, totaling 79. [Pixel and DINO scoring](../research/eeg_stack_20261112/code/score_spatial_photo.py)

## Fresh-panel confirmation

The confirmation kept these three participants and introduced 12 different objects. The lock records all three fitted model hashes, all three fresh angle-condition hashes and both original-panel score-file hashes. It fixed the two pixel comparisons above and the DINO assignment and ten-photo outcomes before the fresh angle images were available locally. This tests generalization across objects and photographs for the same participants. [Timestamped confirmation lock](../research/eeg_stack_20261112/results/object_fidelity/spatial_photo_confirmation_design.json)

| Participant | Correct-word queries | Luminance correlation, own minus prior [95% interval] | Strict donor queries | Grayscale SSIM, own minus wrong [95% interval] | DINO assignments, base / prior / own / wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sub-04 | 85 | +0.00149 [−0.00037, +0.00318] | 55 | +0.001315 [+0.000004, +0.003255] | 20 / 20 / 20 / 20 of 34 |
| Sub-05 | 60 | +0.00281 [+0.00004, +0.00583] | 21 | +0.003478 [+0.000076, +0.006068] | 6 / 6 / 6 / 6 of 15 |
| Sub-01 | 64 | +0.00041 [−0.00129, +0.00229] | 37 | +0.001317 [+0.000325, +0.002583] | 13 / 13 / 13 / 13 of 23 |

On the new objects, the small grayscale SSIM advantage over an incorrect EEG residual repeated in all three participants. The strict donor subsets represented 11, 7 and 8 objects, respectively. Only sub-05's interval for luminance correlation over the spatial prior excluded zero. In all 360 queries, DINO identified the exact photo among ten of its object on 44 angle baselines, 45 spatial priors, 46 own-EEG outputs, 46 wrong-EEG outputs and 46 older 4×4 outputs. Each of the first four arms received **39/72** two-photo assignment credits. The added spatial signal therefore improved a low-frequency image measure without establishing better identification of the viewed photo. [Compact scores, class-paired intervals and provenance](../research/eeg_stack_20261112/results/object_fidelity/spatial_photo.json)

## Replay

Start with the 512-object calibration files, the posterior EEG packets, the registered angle render and the public training-image ZIP from the [photograph reconstruction replay](PHOTO_STRUCTURE.md#replay). Run from the repository root with Python 3.11+, NumPy, SciPy and Pillow. DINO scoring uses a CUDA or CPU environment with PyTorch, Transformers and the pinned DINOv2-small assets. The old panels must each have a frozen source directory containing `settings.json` and `images/own/000.jpg` through `119.jpg`. The [fresh-panel replay](FRESH_PHOTO_REPLAY.md) prepares its 12-object registration, EEG packets and angle images in the same format. Spatial fitting and editing require no diffusion model or GPU.

```sh
C=research/eeg_stack_20261112/code
R=research/eeg_stack_20261112/results/object_fidelity
P=outputs/spatial-photo
F=outputs/fresh-photo
mkdir -p "$P"
cp "$R/spatial_photo_design.json" "$P/design.json"
cp "$R/spatial_photo_confirmation_design.json" "$P/fresh_lock.json"
"$PY" "$C/fit_spatial_photo.py" cache --out "$P"
for s in sub-04 sub-05 sub-01; do
  "$PY" "$C/fit_spatial_photo.py" fit --subject "$s" --out "$P"
  for p in registered replicate; do
    "$PY" "$C/fit_spatial_photo.py" render --subject "$s" --panel "$p" --out "$P"
  done
  "$PY" "$C/fit_spatial_photo.py" render --subject "$s" --panel fresh --out "$P" \
    --eeg "$F/eeg/$s.npz" --packet "$F/conditions/$s/fresh/conditions_512.npz" \
    --base "$F/renders/fresh/$s"
done
for b in pixels dino; do
  runtime="$PY"
  if [ "$b" = dino ]; then runtime="$DINO_PY"; fi
  "$runtime" "$C/score_spatial_photo.py" --root "$P" --backend "$b" --out "$P/$b.json"
  "$runtime" "$C/score_spatial_photo.py" --root "$P" --backend "$b" --panels fresh --out "$P/fresh_$b.json"
done
```

The four score files report per-panel and per-participant results. Pass `--device cpu` to score DINO without CUDA. For byte-level replay with the original frozen model and input files, also pass `--confirmation "$P/fresh_lock.json"` when rendering the fresh panel. That option checks the registered hashes before writing images. The archive audit command is `"$PY" "$C/summarize_spatial_photo.py" --root "$P" --out "$P/spatial_photo.json"`. Keep the raw EEG, stimuli and any contact sheets containing originals in the private workspace.
