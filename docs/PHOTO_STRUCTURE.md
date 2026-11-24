# Photograph structure and personal calibration

Can a saved reconstruction identify the particular photograph a participant saw after EEG has already supplied the object name? Two experiments tested sub-04 and sub-05 using the personal 128-object readout. Both kept each query's predicted word fixed. Each participant saw the same 120 distinct photographs in each of two disjoint 12-object panels, giving 240 stimulus photos and 480 participant–photo queries. The support-guided generator and foreground warp used the same frozen personal JPEG baseline. [Per-participant measurements and input hashes](../research/eeg_stack_20261112/results/object_fidelity/photo_structure.json)

## What changed in the images

**Support photos.** A linear model learned associations between within-object EEG and image differences on the participant's 128 outside calibration objects. It used 17 posterior electrodes from 60 to 460 ms, public H/14 image features, grayscale layout and oriented edges. Four object-disjoint training folds set the mixture of uniform, EEG-to-image and cross-session EEG affinities. Each query then weighted only the eight authorized photos of its predicted object. Their pooled VAE latent supplied 80% of the starting latent and the existing EEG start supplied 20%. The generator ran the last two steps of the archived four-step Turbo schedule. Uniform, permuted-weight and other-query EEG controls used the same rendering recipe. [Selector](../research/eeg_stack_20261112/code/fit_support_structure.py) · [Generator](../research/eeg_stack_20261112/code/render_support_structure.py)

**Foreground geometry.** A frozen U2NetP saliency mask supplied the horizontal and vertical centroid and the log spread on each axis. Direct regression and a 48-dimensional projection predicted within-object deviations from 40–520 ms posterior EEG. The projection learned from corresponding generic training photos in seven other participants. Each test query used its own four EEG trials and the mean geometry of the predicted object's eight calibration photos. A bounded translation and axis scaling transformed the existing generated JPEG. The saved arms include the original JPEG, a neutral re-encoding, class-prior warp, own-EEG warp and other-query warp. [EEG model](../research/eeg_stack_20261112/code/fit_eeg_foreground_geometry.py) · [Image transform](../research/eeg_stack_20261112/code/warp_foreground_geometry.py)

The original query image entered scoring only after that query's selection or geometry estimate was frozen. The same photo can be a legal support in a different fold. The support model put most of the weight on the uniform component. The two EEG-dependent components together received 14.4% on sub-04 and 15.1% on sub-05.

## Which original photograph was viewed?

CLIP-L/14 scored the saved JPEGs. For the strict two-photo test, both held-out photographs must share the true object, fold and predicted word. The evaluator compared the sum of cosine similarities for the correct assignments with the sum for the exchanged assignments. One pair supplies one observation. A tie contributes half credit. Chance is 50%. [Frozen JPEG evaluator](../research/eeg_stack_20261112/code/score_reconstruction_jpegs.py)

| Panel and participant | Eligible pairs | Uniform support → EEG-weighted support | Class-prior warp → own-EEG warp |
| --- | ---: | ---: | ---: |
| Registered, sub-04 | 23 | 16 → 14 | 16 → 15 |
| Registered, sub-05 | 21 | 11 → 10 | 11 → 12 |
| Replication, sub-04 | 28 | 17 → 17 | 12 → 15 |
| Replication, sub-05 | 25 | 14 → 13 | 11 → 10 |
| Descriptive total | 97 | **58 → 54** | **50 → 52** |

Across the 480 queries, 1,854-way naming credit was 235.5 → 237 for the support comparison and 230 → 229.5 for the warp comparison. Identification among ten originals of the true object was 46 → 43 and 53 → 54, respectively. The unchanged JPEG baseline scored 230.5/480 on naming, 51/480 on same-object identification and 54/97 on two-photo assignment. Separate within-participant, within-panel confidence intervals for both two-photo improvements included zero. The interval resamples true objects, keeping their photographs and pairs together.

Support-guided rendering improved mean grayscale SSIM over the original JPEG by 0.051–0.062 across the four cases. Its paired SSIM differences from uniform weights and from verified other-query weights included zero in all four. Geometry error uses the same saliency estimator that supplies the warp targets. Query geometry improved on a neutral re-encoding in all four cases on that measure. Its extra improvement over class geometry had a 95% interval below zero only on sub-05 replication. Direct grayscale SSIM differences between the own-EEG and class-prior warps included zero in all four. The strict other-query comparisons used 32/25/42/35 eligible rows in table order and 9/4/14/10 pairs for which both donors were exchanged. [Support pixel scoring](../research/eeg_stack_20261112/code/score_support_structure.py) · [Foreground and pixel scoring](../research/eeg_stack_20261112/code/score_foreground_warp.py)

## Personal calibration size and scalp coverage

This separate experiment permitted changes to the predicted object word. The 512-object candidate retained the 128-object posterior readout's 17 channels, four-trial averaging, 96 Fisher directions, eight target supports and four-step generator. It added 384 objects drawn from the remaining non-panel classes with seed 260508. This adds 3,840 distinct photographs and 15,360 physical EEG presentations **per person**, for totals of 5,120 photos and 20,480 presentations. The equal-data full-scalp candidate reused the original 1,280 photos (5,120 presentations). It combined the posterior readout with a 63-channel four-repeat view and a 63-channel cross-session view over 40–600 ms. Sub-04 generic-object cross-validation fixed the three coefficients at 1, 0.25 and 0.25 before target evaluation. Prior outcomes for the 128-object reference on these two people were known. [Frozen design](../research/eeg_stack_20261112/results/object_fidelity/expanded_calibration_design.json)

| EEG correct among twelve, out of 120 | 128 outside objects, posterior | 128 outside objects, full-scalp combination | 512 outside objects, posterior |
| --- | ---: | ---: | ---: |
| Registered, sub-04 | 68 | 73 | 86 |
| Registered, sub-05 | 60 | 66 | 75 |
| Replication, sub-04 | 72 | 76 | 94 |
| Replication, sub-05 | 67 | 69 | 72 |
| Descriptive total | **267/480** | **284/480** | **327/480** |

The expanded posterior readout was then locked for an additional participant. Sub-01's EEG object scores rose from 41 to 67 on the registered panel and from 65 to 76 on replication. The next table reports 1,854-way naming credit for generated RGB before JPEG compression and top-one identification among the ten source photos of the true object. Each cell gives **naming / same-object photo credit**, each out of 120. The three participants saw the same 240 distinct photographs. [Per-case data, class-paired intervals and session controls](../research/eeg_stack_20261112/results/object_fidelity/readout_scale.json) · [Additional-participant lock](../research/eeg_stack_20261112/results/object_fidelity/expanded_confirmation_design.json)

| Panel and participant | 128 posterior | 128 full-scalp combination | 512 posterior |
| --- | ---: | ---: | ---: |
| Registered, sub-04 | 53 / 14 | 59 / 11 | 72 / 11 |
| Replication, sub-04 | 67 / 16 | 70 / 19 | 85 / 20 |
| Registered, sub-05 | 46 / 12 | 51 / 9 | 62 / 9 |
| Replication, sub-05 | 62.5 / 12 | 65.5 / 8 | 67 / 8 |
| Registered, sub-01 confirmation | 35 / 13 | — | 53 / 11 |
| Replication, sub-01 confirmation | 58 / 14 | — | 67.5 / 12 |

On sub-01 alone the expanded arm added 27.5 naming credits across 240 queries while same-object photo identification went from 27 to 23. Over all three participants the corresponding totals were **321.5 → 406.5/720** and **81 → 71/720**. The 128-class full-scalp alternative was tested on sub-04/05 only. Public galleries show the generated photo for the lowest original row of each object, including misses. [Sub-01 registered](../research/eeg_stack_20261112/gallery/expanded_sub01_registered.jpg) · [Sub-01 replication](../research/eeg_stack_20261112/gallery/expanded_sub01_replicate.jpg) · [Sub-05 registered](../research/eeg_stack_20261112/gallery/expanded_sub05_registered.jpg) · [Sub-05 replication](../research/eeg_stack_20261112/gallery/expanded_sub05_replicate.jpg)

## Photo-specific H/14 embeddings

With the expanded object word fixed, a class-centered EEG ridge prediction added a continuous visual residual to the H/14 mean of the eight authorized support photos. An eligible pair shares the true object and fold, and both predicted words must be correct. Joint assignment gives both EEG predictions and both candidate photo embeddings to the evaluator and enforces a one-to-one match. With the common support mean alone, the assignments tie and split credit. The single-query check scores each EEG against those same two alternatives independently. Both endpoints have chance 50%. [Embedding-level outcomes](../research/eeg_stack_20261112/results/object_fidelity/within_photo_embedding.json) · [Additional-participant design](../research/eeg_stack_20261112/results/object_fidelity/within_photo_confirmation_design.json)

The object word in both fits comes from the 512-object readout. The table varies how many outside objects trained the EEG-to-H/14 residual. Ridge and visual whitening were selected once on four object-disjoint folds of the first 128 sub-04 training objects. The same ridge value (2.0) and zero whitening were used in every fit.

| Participant | Joint assignments, 128-object residual | Joint assignments, 512-object residual | Independent choices, 512-object residual |
| --- | ---: | ---: | ---: |
| Sub-04 | 51/72 | 54/72 | 76/144 |
| Sub-05 | 33/51 | 37/51 | 51/102 |
| Sub-01 confirmation | 34/47 | 39/47 | 48/94 |

The joint result measures relative correspondence between two EEG responses and two known candidates in H/14 space. Using uncentered embeddings, the independent choices on these same examples sit near chance. After seeing those results, a diagnostic subtracted the normalized eight-support prototype from the EEG prediction and each visual candidate before measuring their residual cosine. Each query still chose independently between two photos of its correctly predicted object. Scores rose to **98/144** on sub-04, **64/102** on sub-05 and **70/94** on sub-01. The corresponding wrong-EEG scores were 50, 35 and 25. Candidates enter only at scoring. Sub-01 was a confirmation for the original uncentered embedding analysis. The centered comparison is post hoc for this release. [Residual-cosine report and uncertainty](../research/eeg_stack_20261112/results/object_fidelity/photo_angle.json)

The SDXL/IP-Adapter experiment generated **1,440 JPEGs**, one per arm for each of 480 participant–photo queries on sub-04/05. A different encoder, OpenAI CLIP-L/14, compared the saved images to the source photos. The image-encoder magnitude was fixed at 21.9456 using 32 generic training photos. Each comparison shares the expanded model's predicted word and row-specific noise. [Locked image experiment](../research/eeg_stack_20261112/results/object_fidelity/within_photo_render_design.json) · [JPEG counts, photo margins and paired intervals](../research/eeg_stack_20261112/results/object_fidelity/within_photo_render.json)

| Saved JPEGs, correctly named pairs | Eligible pairs | Uniform prior | Own EEG | Wrong EEG |
| --- | ---: | ---: | ---: | ---: |
| Registered, sub-04 | 33 | 17 | 16 | 13 |
| Replication, sub-04 | 39 | 23 | 26 | 26 |
| Registered, sub-05 | 27 | 14 | 16 | 13 |
| Replication, sub-05 | 24 | 13 | 17 | 13 |

The own-EEG images matched **75/123** pairs, versus **67/123** for the uniform prior. All four own-minus-uniform 95% intervals included zero. On reciprocal swaps of exactly those two queries, own/wrong assignment was 11/8 of 21, 20/22 of 31, 11/8 of 17 and 9/7 of 14 in table order. These four paired intervals also included zero. The 123 pairs here come from the expanded object readout and require two correct words. The earlier support/warp experiment admitted 97 pairs under the original 128-object readout and required a shared predicted word.

A second visual scorer, DINOv2-small, used its frozen final-layer CLS token on the **same JPEGs**. Own EEG scored below the uniform prior on the two-photo assignment for both participants. Across both panels, the own-minus-uniform difference was −6.9 percentage points on sub-04 (95% interval −18.3 to +4.3) and −15.7 points on sub-05 (−34.0 to +2.1). The ten-photo test includes every query, even when its predicted word was wrong. [DINO outcomes, preprocessing and hashes](../research/eeg_stack_20261112/results/object_fidelity/within_photo_dino.json)

| DINO score across both panels | Eligible pairs or queries | Own EEG | Uniform prior | Wrong EEG |
| --- | ---: | ---: | ---: | ---: |
| Two-photo assignment, sub-04 | 72 pairs | 34 | 39 | 29 |
| Two-photo assignment, sub-05 | 51 pairs | 19 | 27 | 23 |
| Same-object ten-photo identification, sub-04 | 240 queries | 13 | 17 | 24 |
| Same-object ten-photo identification, sub-05 | 240 queries | 23 | 23 | 20 |

### Calibrating the residual angle before generation

The original H/14 prediction rotated only 0.165 radians from the class prototype at the median on the sub-04 generic training set. The matching real-photo median was 0.582 radians. A single 3.5244× gain and a 0.789-radian ceiling, estimated on those 128 outside-panel objects, determined the new angles for all queries. Word, row noise, adapter magnitude and four denoising steps stayed fixed. All 480 uniform-prior JPEGs matched their original files byte for byte. [Calibration design and estimates](../research/eeg_stack_20261112/results/object_fidelity/photo_angle.json)

| Scorer and task | Participant | Pairs or queries | Own EEG | Uniform prior | Wrong EEG |
| --- | --- | ---: | ---: | ---: | ---: |
| CLIP-L/14, two-photo assignment | Sub-04 | 72 | 51 | 40 | 30 |
| CLIP-L/14, two-photo assignment | Sub-05 | 51 | 33 | 27 | 22 |
| DINOv2, two-photo assignment | Sub-04 | 72 | 48 | 39 | 26 |
| DINOv2, two-photo assignment | Sub-05 | 51 | 30 | 27 | 18 |
| DINOv2, same-object ten-photo identification | Sub-04 | 240 | 30 | 17 | 16 |
| DINOv2, same-object ten-photo identification | Sub-05 | 240 | 23 | 23 | 19 |

On DINOv2's two-photo assignment, the participant-level paired intervals for own versus wrong EEG were +16.4 to +44.4 percentage points on sub-04 and +4.3 to +44.3 on sub-05. Both intervals for own versus the uniform prior included zero. The four per-panel CLIP-L/14 intervals for own versus uniform also included zero. The ten-photo endpoint gave a gain over uniform on sub-04 and equal credit on sub-05. [CLIP-L/14 per-panel results](../research/eeg_stack_20261112/results/object_fidelity/photo_angle_render.json) · [DINOv2 results and paired intervals](../research/eeg_stack_20261112/results/object_fidelity/photo_angle_dino.json)

Sub-01 was designated for a separate angle-calibrated image confirmation after the sub-04/05 image results were known. The team already knew this person's object-decoding, earlier rendering and embedding results. The confirmation fixed both condition-packet hashes, the training-derived gain, adapter radius and row seeds before producing the angle-calibrated images. [Additional-participant image protocol](../research/eeg_stack_20261112/results/object_fidelity/photo_angle_confirmation_design.json)

On sub-01, DINOv2 scored **25/47** strict pairs for own EEG, **24/47** for the prior and **17/47** for wrong EEG. The own-minus-prior interval was −15.6 to +22.7 percentage points, and the own-minus-wrong interval was −8.5 to +41.9. Independent choices between the two originals scored 53, 44 and 42 out of 94 queries. Ten-photo identification across all 240 queries gave 25, 19 and 18. Across all three participants the own-EEG images had 103/170 correct joint assignments and 78/720 ten-photo identifications. Under DINOv2, both primary paired contrasts on the additional person remained uncertain. [Three-person DINOv2 evaluation](../research/eeg_stack_20261112/results/object_fidelity/photo_angle_dino.json)

CLIP-L/14 assigned **31/47** strict pairs on sub-01 using own EEG, **22/47** using the shared prior and **12/47** using wrong EEG. The own-minus-wrong interval was positive in each panel. The own-minus-prior intervals reached or crossed zero. On the ten-photo test the scores were 21, 25 and 24 of 240. Full-vocabulary naming gave 129.5, 128.5 and 132.5 credits. [Per-panel paired intervals and scoring inputs](../research/eeg_stack_20261112/results/object_fidelity/photo_angle_render.json)

| Direct angle-calibrated JPEGs, all three participants | Own EEG | Uniform prior | Wrong EEG |
| --- | ---: | ---: | ---: |
| CLIP-L/14 naming among 1,854, all queries | **433/720** | 432/720 | 436/720 |
| CLIP-L/14 two-photo assignment, both object words correct | **115/170** | 89/170 | 64/170 |
| DINOv2 two-photo assignment, same eligible pairs | **103/170** | 90/170 | 61/170 |
| CLIP-L/14 same-object ten-photo identification, all queries | **65/720** | 59/720 | 62/720 |
| DINOv2 same-object ten-photo identification, all queries | **78/720** | 59/720 | 53/720 |

These are descriptive totals from **2,160 JPEGs** (three arms × 720 participant–photo queries). The two-photo assignment requires correct decoded objects for both queries. Comparing a single generated image to ten originals includes every query and has 10% chance accuracy. The Brain-HIVE 128→512 object-readout comparison above scored RGB before JPEG compression. Here the independent models reopened the saved JPEGs, so 406.5/720 from the original 512-object pipeline and 433/720 from this one are descriptive results under different scoring inputs.

The generated-only galleries use the lowest original row in each object class. [Sub-01 registered](../research/eeg_stack_20261112/gallery/photo_angle_sub01_registered.jpg) · [Sub-01 replication](../research/eeg_stack_20261112/gallery/photo_angle_sub01_replicate.jpg) · [Sub-05 registered](../research/eeg_stack_20261112/gallery/photo_angle_sub05_registered.jpg) · [Sub-05 replication](../research/eeg_stack_20261112/gallery/photo_angle_sub05_replicate.jpg)

### Selecting among eight generated images

A separate experiment gave every arm eight candidates. For each panel, fold and registered object, the generator saw the object's word and the pooled visual features of its eight calibration photos. Eight fixed seeds produced eight JPEGs per cell. The two panels contain 5 folds × 12 objects × 8 images each, for **960 generated JPEGs** shared across participants. The frozen 512-object readout supplies the query's word. Its EEG residual ranks those eight candidates by support-centered H/14 cosine. The controls use another EEG residual from the same fold and predicted word, proximity to the calibration-photo mean, or the first fixed seed. The selected JPEG is copied unchanged. The held-out original enters the independent scorers after selection. [Experiment design](../research/eeg_stack_20261112/results/object_fidelity/generated_bank_design.json) · [Frozen generator inputs, seeds and weights](../research/eeg_stack_20261112/results/object_fidelity/generated_bank_inputs.json)

| DINOv2 on selected JPEGs | Sub-04 | Sub-05 | Sub-01 |
| --- | ---: | ---: | ---: |
| Two-photo assignment, own / prior / wrong EEG | 31 / 36 / 37.5 of 72 | 29.5 / 25.5 / 23.5 of 51 | 25 / 23.5 / 23.5 of 47 |
| Ten-photo identification, own / prior / wrong EEG | 19 / 18 / 22 of 240 | 22 / 21 / 19 of 240 | 19 / 18 / 26 of 240 |

Two correctly named queries in a pair draw from the same eight JPEGs. Both the prior selector and the fixed first-seed control choose exactly the same JPEG twice, giving half credit on every such pair. The own-EEG selector scored **85.5/170** across the three participants, versus **85/170** for the prior. Each participant's paired 95% interval for own versus prior and for own versus wrong included zero. [DINOv2 bank scores, fixed-seed control and intervals](../research/eeg_stack_20261112/results/object_fidelity/generated_bank_dino.json)

| CLIP-L/14 on selected JPEGs | Sub-04 | Sub-05 | Sub-01 |
| --- | ---: | ---: | ---: |
| Two-photo assignment, own / prior / wrong EEG | 33 / 36 / 42.5 of 72 | 24.5 / 25.5 / 29.5 of 51 | 33 / 23.5 / 14.5 of 47 |
| Ten-photo identification, own / prior | 22 / 18 of 240 | 15 / 18 of 240 | 22 / 26 of 240 |
| Naming among 1,854, own / prior | 167 / 171 of 240 | 136 / 138.5 of 240 | 126.5 / 128.5 of 240 |

The CLIP two-photo gain on sub-01 was +20.2 percentage points over the prior (95% interval +8.8 to +32.6). The other two participants scored below the prior, and DINOv2's result on sub-01 was 25 versus 23.5 of 47. The eight-candidate experiment remains a diagnostic. The selected photo-level route generates one image from the angle-calibrated condition. [CLIP bank scores and per-object intervals](../research/eeg_stack_20261112/results/object_fidelity/generated_bank_clip.json)

## Replay

Run the shell blocks from the repository root. The [registered R4 replay](OBJECT_FIDELITY_REPLAY.md#1-registered-r4) prepares the public physical EEG packets, schedule and range receipts, the fitted transfer model, two registered panels and personal conditions and renders. Its [within-object color replay](OBJECT_FIDELITY_REPLAY.md#within-object-color-residuals) produces the original-photo hash reports. Use Python 3.11+ with NumPy, SciPy, Pillow and ONNX Runtime for CPU work. Image generation and CLIP scoring require the custodian-provided historical PKU archive and its compatible CUDA Python. Keep stimuli, EEG, weights and original-versus-render comparisons in the local replay workspace.

In the layout below, `I` holds public `sub-XX.npz`, `generic/`, `halves/`, `ranges/`, and `sessions.json`. Original 128-class fits are under `I/personal/PANEL/SUBJECT/evaluation.json`, and original-photo receipts under `I/photo_residual/PANEL/sub-04/report.json`. `OLD` contains `PANEL/SUBJECT/results.json` and `images/aligned/`. `COND` contains the corresponding conditions. `REF` contains both `PANEL_ablation.json` files. Set `PANELS` to the parent of the two full `PANEL/{evaluation,ablation}.json` registrations and `TRANSFER` to the fitted transfer directory. The geometry scorer also reads baseline results at `I/personal_render/personal_conditions/PANEL/SUBJECT/results.json`. In the previous replay these inputs come from `W/panels`, `W/participant-fit`, `P/fits`, `P/conditions`, `P/render` and the seven-person `Q` outputs.

```sh
set -eu
C=research/eeg_stack_20261112/code
I=outputs/object-transfer
BASE=outputs/official-object-readout/sub-03
REF=outputs/object-fidelity
S=outputs/photo-structure
OLD="$I/personal_render/personal_conditions"
COND="$I/personal_conditions"
A="$PKU_ARCHIVE"
mkdir -p "$S"
```

Obtain the training photos from the pinned public ZIP and save it as `$S/training_images.zip`. Save the U2NetP ONNX weights as `$S/u2netp.onnx`. The extractor checks their SHA-256 values, beginning `b91edf1a` and `309c8469`. The ATM visual bank under `$BASE/assets/text_train.pt` contains one H/14 image vector per public training row. The geometry model also needs the same 128 generic calibration classes for sub-01, 02, 06, 07, 08, 09 and 10, prepared for the seven-participant study. [Pinned photo archive](https://huggingface.co/datasets/gasparyanartur/things-eeg2/tree/0e2ad986a26d90b6192cd69daeac32ff069c2460/imgs) · [U2NetP release weights](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx) · [Author H/14 bank](https://huggingface.co/datasets/LidongYang/EEG_Image_decode/blob/dbe34bb2407164f70883c661c76664f0d596e522/ViT-H-14_features_train.pt)

### Support and foreground

```sh
"$PY" "$C/fit_support_structure.py" extract-structure --subject sub-04 --inputs "$I" --source "$BASE" \
  --photo-zip "$S/training_images.zip" --out "$S/support_structure_features.npz"
for s in sub-04 sub-05; do
  "$PY" "$C/fit_support_structure.py" fit --subject "$s" --inputs "$I" --references "$REF" \
    --source "$BASE" --structure "$S/support_structure_features.npz" --out "$I/support_structure_joint"
done
for s in sub-04 sub-05; do
  "$PY" "$C/fit_support_structure.py" score --subject "$s" --inputs "$I" --references "$REF" \
    --source "$BASE" --structure "$S/support_structure_features.npz" --out "$I/support_structure_joint"
  for p in registered replicate; do
    "$GPU" "$C/render_support_structure.py" --archive "$A" --dictionary "$BASE/dictionary.csv" \
      --case "$COND/$p/$s/conditions.npz" "$I/support_structure_joint/$p/$s/selection.npz" \
      "$REF/${p}_ablation.json" "$OLD/$p/$s" "$S/support_render/$p/$s"
    "$PY" "$C/score_support_structure.py" --panel "$p" --subject "$s" --renders "$S/support_render/$p/$s" \
      --train-zip "$S/training_images.zip" --train-rows "$BASE/atm_train_rows.csv" \
      --registration "$REF/${p}_ablation.json" --selection "$I/support_structure_joint/$p/$s/selection.npz" \
      --photo-report "$I/photo_residual/$p/sub-04/report.json" --baseline-results "$OLD/$p/$s/results.json" \
      --out "$S/support_render/$p/$s/structure.json"
  done
done
"$PY" "$C/measure_foreground_geometry.py" --model "$S/u2netp.onnx" --images "$S/training_images.zip" \
  --manifest "$BASE/atm_train_rows.csv" --generic "$I/generic/sub-04.npz" --target "$I/sub-04.npz" \
  --out "$S/foreground_geometry.npz"
"$PY" "$C/fit_eeg_foreground_geometry.py" --inputs "$I" --references "$REF" --geometry "$S/foreground_geometry.npz" \
  --out "$S/geometry_readout"
"$PY" "$C/warp_foreground_geometry.py" --geometry "$S/geometry_readout" --baselines "$OLD" --results "$OLD" \
  --registrations "$I/personal" --model "$S/u2netp.onnx" --out "$S/geometry_warp"
"$PY" "$C/score_foreground_warp.py" --renders "$S/geometry_warp" --geometry "$S/foreground_geometry.npz" \
  --predictions "$S/geometry_readout" --model "$S/u2netp.onnx" --inputs "$I" --references "$REF" \
  --train-zip "$S/training_images.zip" --train-rows "$BASE/atm_train_rows.csv" --out "$S/geometry_warp_evaluation.json"
"$GPU" "$C/score_reconstruction_jpegs.py" --archive "$A" --dictionary "$BASE/dictionary.csv" --reference-dir "$REF" \
  --renders "$S/support_render" --renders "$S/geometry_warp" --geometry-estimates "$S/geometry_readout" \
  --geometry-features "$S/foreground_geometry.npz" --out "$S/structure_clip_report.json"
"$PY" "$C/summarize_photo_structure.py" --clip "$S/structure_clip_report.json" --geometry "$S/geometry_warp_evaluation.json" \
  --support "$S/support_render" --reference-dir "$REF" --out "$S/photo_structure.json"
```

### Two alternative object readouts

The 512-object fit fetches the additional public EEG byte ranges. Keep the original 128-object range cache and its receipts for all three participants. The full-scalp fit reconstructs all 63 channels from those same verified bytes. The two fits produce `PANEL/SUBJECT/evaluation.json`. Use the original transfer conditions and four-step generator for the RGB comparison. [Calibration expansion](../research/eeg_stack_20261112/code/prepare_expanded_personal.py) · [Full-scalp fitting and selection](../research/eeg_stack_20261112/code/fit_full_head_objects.py)

```sh
"$PY" "$C/prepare_expanded_personal.py" --inputs "$I" --references "$REF" --train-rows "$BASE/atm_train_rows.csv" \
  --subjects sub-04 sub-05 --out "$S/expanded"
"$PY" "$C/prepare_expanded_personal.py" --inputs "$I" --references "$REF" --train-rows "$BASE/atm_train_rows.csv" \
  --subjects sub-01 --out "$S/expanded"
"$PY" "$C/fit_full_head_objects.py" select --inputs "$I" --out "$S/full_head"
"$PY" "$C/fit_full_head_objects.py" evaluate --inputs "$I" --out "$S/full_head" --references "$REF" --subjects sub-04 sub-05
"$PY" "$C/fit_full_head_objects.py" summarize --inputs "$I" --out "$S/full_head"
for method in expanded_r4 full_head; do
  fits="$S/full_head"; people="sub-04 sub-05"
  if [ "$method" = expanded_r4 ]; then fits="$S/expanded"; people="sub-04 sub-05 sub-01"; fi
  "$GPU" "$C/condition_registered_predictions.py" --archive "$A" --transfer "$TRANSFER" --subjects-dir "$I" \
    --panel-dirs "$PANELS/registered" "$PANELS/replicate" --prediction-runs "$fits" --prediction-method "$method" \
    --subjects $people --out "$S/${method}_conditions"
  for p in registered replicate; do
    for s in $people; do
      "$GPU" "$C/render_registered_objects.py" --archive "$A" --dictionary "$BASE/dictionary.csv" --aligned-only \
        --case "$S/${method}_conditions/$p/$s/conditions.npz" "$S/${method}_render/$p/$s"
    done
  done
done
"$PY" "$C/summarize_readout_scale.py" --runs "$S" --transfer "$I" --out "$S/readout_scale.json"
```

The scale summarizer also reads the original 128-class `I/confirm_render/personal/PANEL/sub-01/results.json`. To reproduce the H/14 embedding experiment, run these CPU stages after the 512-object fit. The inference stage writes frozen embedding packets before the scoring stage reads query-image features.

```sh
embedding () {
  "$PY" "$C/fit_within_photo_embedding.py" "$@" --source "$BASE" --expanded "$S/expanded" \
    --inputs "$I" --references "$REF" --out "$S/within_photo"
}
embedding select
cp research/eeg_stack_20261112/results/object_fidelity/within_photo_confirmation_design.json \
  "$S/within_photo/sub-01_confirmation_design.json"
for s in sub-04 sub-05 sub-01; do
  embedding infer --subject "$s"
  embedding score --subject "$s"
done
embedding release --release-file "$S/within_photo_embedding.json"
```

To reproduce the image-domain test, stage the published design and frozen conditioning packets and restore the H/14 radius from the 32 preselected generic photos. Set `ADAPTER` to the base SDXL ViT-H adapter. Set `H14` to the pinned H/14 model directory and `H14_PY` to the compatible CUDA Python with PyTorch and Transformers, as in the external-reference replay. The script checks the encoder digest. All three arms use the same row seed and word. [Tencent adapter](https://huggingface.co/h94/IP-Adapter/blob/018e402774aeeddd60609b4ecdb7e298259dc729/sdxl_models/ip-adapter_sdxl_vit-h.safetensors) · [H/14 assets](OBJECT_FIDELITY_REPLAY.md)

```sh
H="$S/h14_residual_render"
mkdir -p "$H/conditions"
cp research/eeg_stack_20261112/results/object_fidelity/within_photo_render_design.json "$H/design.json"
cp "$C/render_within_photo_ip.py" "$C/score_within_photo_ip.py" "$H/"
for s in sub-04 sub-05; do
  for p in registered replicate; do
    mkdir -p "$H/conditions/$s/$p"
    cp "$S/within_photo/$s/$p/conditions_512.npz" "$H/conditions/$s/$p/"
  done
done
"$H14_PY" "$C/prepare_within_photo_ip_radius.py" --design "$H/design.json" --packets "$S/within_photo" \
  --assets "$H14" --source "$BASE" --train-zip "$S/training_images.zip" --out "$H/radius.json"
for s in sub-04 sub-05; do
  "$GPU" "$C/render_within_photo_ip.py" --root "$H" --archive "$A" --dictionary "$BASE/dictionary.csv" \
    --adapter "$ADAPTER" --subject "$s"
done
"$GPU" "$C/score_within_photo_ip.py" --root "$H" --archive "$A" --dictionary "$BASE/dictionary.csv" \
  --reference-dir "$REF" --subjects sub-04 sub-05 --out "$H/clip-sub-04-sub-05.json"
"$PY" "$C/summarize_within_photo_ip.py" --root "$H" --references "$REF" --scores "$H/clip-sub-04-sub-05.json" \
  --out "$S/within_photo_render.json"
```

For an independent visual readout, place the three files named in the DINO report from model revision `ed25f3a31f01632728cabb09d1542f84ab7b0056` in `DINO`. The evaluator verifies their hashes, the original-photo ZIP and the row manifest. It writes the aggregate to `$S/dino/scoring/summary.json`. [Frozen scorer](../research/eeg_stack_20261112/code/score_within_photo_dino.py)

```sh
"$H14_PY" "$C/score_within_photo_dino.py" --model "$DINO" --originals "$S/training_images.zip" \
  --manifest "$BASE/atm_train_rows.csv" --references "$REF" --conditions "$S/within_photo" \
  --renders "$H/renders" --subjects sub-04 sub-05 --out "$S/dino/scoring"
```

For the centered single-query check, the first command estimates the natural angular scale on generic calibration objects and the second scores the unchanged embedding packets against the two held-out alternatives. The scale experiment's gain was 3.5244 (generic observed median angle 0.5822 radians, predicted median 0.1652). The centered retrieval score above uses the original unamplified predictions.

```sh
for stage in calibrate score; do
  "$PY" "$C/calibrate_photo_angles.py" "$stage" --source "$BASE" --references "$REF" \
    --expanded "$S/expanded" --within "$S/within_photo" --base "$H" --out "$S/residual_angle"
done
```

Package the calibrated vectors with the original per-row seeds and adapter radius. These two participants were the first image experiment at this angle. DINO reads the amplified condition packets for this run.

```sh
G="$S/residual_angle"
"$PY" "$C/calibrate_photo_angles.py" package --source "$BASE" --references "$REF" \
  --expanded "$S/expanded" --within "$S/within_photo" --base "$H" --out "$G"
cp "$C/render_within_photo_ip.py" "$C/score_within_photo_ip.py" "$G/"
for s in sub-04 sub-05; do
  "$GPU" "$C/render_within_photo_ip.py" --root "$G" --archive "$A" --dictionary "$BASE/dictionary.csv" \
    --adapter "$ADAPTER" --subject "$s"
done
"$GPU" "$C/score_within_photo_ip.py" --root "$G" --archive "$A" --dictionary "$BASE/dictionary.csv" \
  --reference-dir "$REF" --subjects sub-04 sub-05 --out "$G/clip-sub-04-sub-05.json"
"$PY" "$C/summarize_within_photo_ip.py" --root "$G" --references "$REF" --subjects sub-04 sub-05 --scores "$G/clip-sub-04-sub-05.json" \
  --out "$S/photo_angle_render.json"
"$H14_PY" "$C/score_within_photo_dino.py" --model "$DINO" --originals "$S/training_images.zip" \
  --manifest "$BASE/atm_train_rows.csv" --references "$REF" --conditions "$G/conditions" \
  --renders "$G/renders" --subjects sub-04 sub-05 --out "$S/dino/angle_scoring"
```

To replay the additional participant, keep the packaged gain, condition packets and generator settings. The last command evaluates DINOv2 on all three people using the same originals and saved images.

```sh
"$GPU" "$C/render_within_photo_ip.py" --root "$G" --archive "$A" --dictionary "$BASE/dictionary.csv" \
  --adapter "$ADAPTER" --subject sub-01
"$GPU" "$C/score_within_photo_ip.py" --root "$G" --archive "$A" --dictionary "$BASE/dictionary.csv" \
  --reference-dir "$REF" --subjects sub-01 --out "$G/clip-sub-01.json"
"$PY" "$C/summarize_within_photo_ip.py" --root "$G" --references "$REF" \
  --scores "$G/clip-sub-04-sub-05.json" "$G/clip-sub-01.json" --out "$S/photo_angle_render.json"
"$H14_PY" "$C/score_within_photo_dino.py" --model "$DINO" --originals "$S/training_images.zip" \
  --manifest "$BASE/atm_train_rows.csv" --references "$REF" --conditions "$G/conditions" \
  --renders "$G/renders" --subjects sub-04 sub-05 sub-01 --out "$S/dino/angle_scoring_three"
```

For the separate eight-candidate experiment, use the original unit-gain EEG packets and the exact machine-readable design. H/14 ranks the bank after generation. The two visual scorers each encode the 960 bank JPEGs once. Use the offline archive Python for CLIP and the modern Transformers environment for H/14 and DINO.

```sh
K="$S/h14_bank"
R=research/eeg_stack_20261112/results/object_fidelity
mkdir -p "$K"
cp "$R/generated_bank_inputs.json" "$K/design.json"
cp "$R/within_photo_render_radius.json" "$K/radius.json"
"$PY" "$C/render_within_photo_bank.py" prepare --root "$K" --source "$BASE" --references "$REF" \
  --conditions "$S/within_photo"
for p in registered replicate; do
  "$GPU" "$C/render_within_photo_bank.py" render --root "$K" --panel "$p" --archive "$A" --adapter "$ADAPTER"
done
"$H14_PY" "$C/select_within_photo_candidates.py" --bank "$K" --conditions "$S/within_photo" \
  --assets "$H14" --out "$K/selected" --subjects sub-04 sub-05 sub-01
"$GPU" "$C/score_generated_bank.py" encode --backend clip --features "$K/clip.npz" --bank "$K/renders" \
  --references "$REF" --archive "$A" --dictionary "$BASE/dictionary.csv"
"$H14_PY" "$C/score_generated_bank.py" encode --backend dino --features "$K/dino.npz" --bank "$K/renders" \
  --model "$DINO" --originals "$S/training_images.zip" --manifest "$BASE/atm_train_rows.csv" \
  --references "$REF" --out "$K/dino_reference_cache"
for b in clip dino; do
  "$PY" "$C/score_generated_bank.py" score --backend "$b" --features "$K/$b.npz" --selection "$K/selected/renders" \
    --references "$REF" --conditions "$S/within_photo" --output "$K/$b-summary.json"
done
```
