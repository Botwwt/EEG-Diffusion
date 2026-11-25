# Foreground pose on the personal 12-object panels

Editing toward the mean pose of eight support photos reduced foreground-axis error on the saved images. Adding an individual query's EEG pose passed neither of the two registered comparisons. The matched edit identified the viewed photo on 44/360 ten-way DINOv2 trials, compared with 45/360 for the support mean. [Evaluation registration](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_evaluation_design.json) · [Per-participant and pooled estimates](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_evaluation.json)

## Source images and controls

Each of the three participants contributed 120 queries. The personal menus overlap, giving 290 unique original photos of 29 objects. The source was the frozen mixed trial-encoder/Fisher reconstruction for every query. The EEG word, eight authorized support photos, four measured presentations and generation seed were fixed. An image editor produced four JPEGs per query. The unedited control re-encoded the source with the same JPEG settings. The support-mean edit used the eight allowed photos of the predicted object. The query and exchanged edits used the corresponding EEG pose. An exchanged donor came from the same predicted word and fold. Singleton groups kept the recipient's EEG on 43 rows. [Source classifier and image protocol](TRIAL_OBJECT_PERSONAL12.md) · [Editor and old-panel oracle](../research/eeg_stack_20261112/code/edit_personal_pose.py)

The evaluator reopened and verified the bytes of all 1,440 edited JPEGs. It applied the same pinned U2NetP foreground model to each edit and original. Each pixel contributed its continuous saliency weight to the foreground center and covariance. If that covariance is $C$, the two components of the foreground-axis vector are $(C_{xx}-C_{yy})/\operatorname{tr}(C)$ and $2C_{xy}/\operatorname{tr}(C)$. This vector measures elongation and orientation in the image plane. It tends toward zero for circular shapes and gives the same result after a 180° reversal. The primary error is the squared Euclidean distance between the two measured vectors. [Scoring code](../research/eeg_stack_20261112/code/score_personal12_pose.py)

## Results on the saved JPEGs

| Outcome | Unedited | Support mean | Query EEG | Exchanged EEG |
| --- | ---: | ---: | ---: | ---: |
| Foreground-axis vector error, all 360 ↓ | 0.29149 | 0.25994 | 0.25748 | 0.26271 |
| Foreground-axis vector error, 82 verified donor rows ↓ | 0.23894 | 0.22655 | 0.23731 | 0.23558 |
| Center error, fraction of image width ↓ | 0.14217 | 0.12475 | 0.12337 | 0.12478 |
| Absolute log-radius error ↓ | 0.22370 | 0.22142 | 0.22188 | 0.22599 |
| RGB error, 32 × 32 RMSE ↓ | 0.29354 | 0.28901 | 0.28901 | 0.28879 |
| Spatial luminance correlation ↑ | 0.05772 | 0.06336 | 0.06225 | 0.06441 |
| DINOv2 identification among ten original photos /360 | 41 | 45 | 44 | 42 |
| DINOv2 strict assignment of two originals /57 pairs | 38 | 36 | 34 | 35 |
| DINOv2 independently identifying one of two originals /114 queries | 60 | 59 | 58 | 57 |
| DINOv2 eight-support classifier, viewed object /360 | 187 | 187 | 187 | 187 |
| CLIP-L/14 generated-image naming, supplied 12 words /360 | 187 | 186 | 187 | 187 |
| CLIP-L/14 generated-image naming, full 1,854 words /360 | 166 | 153 | 151 | 151 |

The support mean reduced axis error by 0.03154 relative to the unedited control (paired 95% interval −0.05513 to −0.00887). Matching the EEG to the query changed axis error by −0.00246 relative to that mean (−0.01143 to +0.00623). There were 82 queries from 18 objects where the EEG word was correct and a different query of the same true object supplied the exchanged pose. On these rows matched minus exchanged error was +0.00173 (−0.03716 to +0.04309). Intervals use 10,000 paired draws of distinct object IDs, preserving shared objects across the three fixed people. Both primary comparisons required an upper bound below zero. The signed intervals characterize variation across objects within this experiment.

On the individual menus, query minus support-mean axis error was +0.00409 for sub-04, +0.01347 for sub-05 and −0.02494 for sub-01. The sub-01 interval was −0.05162 to −0.00210. The same participant had 17/120 ten-photo hits after the query edit and 19/120 after the support-mean edit. A lower geometric error alone did not establish better identification of the viewed photograph.

The original's anisotropy-vector length was below 0.20 on 104 of 360 participant–photo queries. The supplementary angle comparison required the original and all four edits to reach 0.20, leaving 135 queries. Their absolute axis errors were 32.08°, 33.67°, 33.32° and 33.56° in table order. The continuous-vector comparison above retained every query.

For DINOv2 ten-way identification the gallery contained all ten originals of the viewed object, including the two held out from the active fold. The strict comparison required both held-out words to be correct, yielding 22, 16 and 19 pairs for sub-04, sub-05 and sub-01. Independent two-choice scoring counted each query's own decision. The local object-preservation probe classified edits by the cosine to each of the twelve eight-support visual prototypes, with both held-out photos excluded. It retained the source image's object decision in all 360 queries across each of the four arms. These supplied-menu decisions do not measure full-vocabulary naming. For the original source JPEGs, ten-way DINOv2 gave 42/360. The matched-codec unedited control gave 41/360 after recompression.

An additional full-resolution check used all 262,144 pixels in each crop. The 512 × 512 RGB RMSE was 0.32140 for the unedited control, 0.31595 for the support mean, 0.31598 for the query edit and 0.31575 for exchanged EEG. The query and support-mean errors differed by +0.000034 (paired 95% interval −0.000351 to +0.000420). Full-resolution luminance correlations were 0.04994, 0.05465, 0.05366 and 0.05533. [Full-resolution pixel audit](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_pixel512.json)

The separate CLIP-L/14 naming test reopened the four sets of finalized JPEGs. Matching the EEG pose reduced full-vocabulary naming from 166 to 151 of 360 relative to the unedited control (−4.17 percentage points, paired 95% interval −6.57 to −1.84). The support mean also reduced naming, from 166 to 153 (−3.61 points, −5.83 to −1.67). Query minus support mean was −0.56 points (−2.00 to +0.57). The supplied-menu test gave the same 187/360 viewed-object hits for unedited and query-pose images. Object decisions can therefore remain stable among the twelve supplied words while changing among all 1,854 labels. [Independent naming results by participant and arm](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_names.json)

## Fixed galleries

The lowest original row of each object was selected before scoring. The public galleries contain four generated images per object, including incorrectly predicted words. The private replay adds the actual viewed photo and the original generated source. [Sub-04](../research/eeg_stack_20261112/gallery/pose_personal12_sub-04.jpg) · [Sub-05](../research/eeg_stack_20261112/gallery/pose_personal12_sub-05.jpg) · [Sub-01](../research/eeg_stack_20261112/gallery/pose_personal12_sub-01.jpg)

## Separate true-pose diagnostic

On 120 earlier registered-panel reconstructions, giving the editor the true image's pose changed DINOv2 ten-photo identification from 12 to 19 and kept 86/120 correct object decisions. That diagnostic reports the mean Euclidean axis-vector error (0.439 before, 0.315 after). The 360-query experiment above uses squared error and derives the pose from supports and EEG. [Original oracle registration and measurements](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_old_oracle.json)

## Replay

After the three edited-image folders and their common seal exist, run the evaluator in an environment with the pinned DINOv2-S/14 snapshot and ONNX Runtime. On the original Windows workstation the ONNX provider was DirectML. The local private image archive must match the THINGS training-image hash and published row index. The evaluator also checks the original row-ordered image hashes against the previous per-participant DINO reference cache.

~~~sh
C=research/eeg_stack_20261112/code
python "$C/score_personal12_pose.py" prepare
python "$C/score_personal12_pose.py" score
python "$C/gallery_personal12_pose.py"
~~~

The scoring stage writes private per-image measurements, full estimates and the three original-photo contact sheets under `outputs/personal-pose/evaluation`. To reproduce the public JSON, pass `--release research/eeg_stack_20261112/results/object_fidelity/personal12_pose_evaluation.json` to the scoring stage. The optional offline CLIP-L/14 naming stage needs the generated JPEGs, their settings, the personal menus and the archived model assets.

~~~sh
python "$C/score_personal12_pose.py" clip \
  --renders "$POSE/renders" --references "$PERSONAL_REFERENCES" \
  --archive "$ARCHIVE" --dictionary "$DICTIONARY" --out "$POSE/evaluation/clip"
~~~
