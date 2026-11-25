# Individual reconstruction on new, session-matched objects

Three participants each viewed 120 photographs from a personal menu of 12 objects. The frozen posterior-EEG classifier chose the viewed object on **161/360 participant–photo queries** (44.7%, object-bootstrap 95% interval 34.4–55.0%). Independent OpenAI CLIP-L/14 named the viewed object in the matched generated JPEG on **160/360** queries among the 12 supplied words and **145/360** among all 1,854 THINGS words. The menus overlap across people. The 360 queries represent **290 distinct original photographs and 29 distinct objects**. [Panel registration](../research/eeg_stack_20261112/results/object_fidelity/single_person_confirmation_panel.json) · [Image protocol](../research/eeg_stack_20261112/results/object_fidelity/single_person_confirmation_image_design.json) · [Evaluation design](../research/eeg_stack_20261112/results/object_fidelity/personal12_image_evaluation_design.json) · [Full results](../research/eeg_stack_20261112/results/object_fidelity/personal12_image_evaluation.json)

| Participant | EEG word /120 | Matched JPEG, 12 words /120 | Matched JPEG, 1,854 words /120 | Reference photo, 12 words /120 | Reference photo, 1,854 words /120 | DINO retrieval /120 |
|---|---:|---:|---:|---:|---:|---:|
| sub-04 | 53 | 52 | 45 | 119 | 100 | 7 |
| sub-05 | 56 | 56 | 55 | 120 | 100 | 14 |
| sub-01 | 52 | 52 | 45 | 118 | 86 | 22 |
| All participant queries | 161 | 160 | 145 | 357 | 286 | 43 |

Counting each original photograph just once, CLIP named **226/290** correctly among 1,854 words. The 12-word reference scores use each participant's own vocabulary. Uniform random word choice has probability 1/12 (8.33%) for the supplied menu or 1/1,854 (0.054%) for the full dictionary.

## What the image comparisons measure

The 17-channel readout was fitted on 512 other objects. Every target query contributed four physical EEG presentations from one person. All 12 candidates in that person's menu shared one recording-session pair. Five folds provided eight labeled calibration photos and their EEG per candidate. The predicted word selected the eight supports for the generator. Every query ran through the same four-step SDXL-Turbo and H/14 IP-Adapter pipeline. Matched EEG, the mean of the eight supports, and exchanged EEG used the same word and seed. Exchanged EEG came from a query with the same predicted word and fold; 55 one-member groups reused their own query. [Condition and generator receipts](../research/eeg_stack_20261112/results/object_fidelity/single_person_confirmation_render_inputs.json)

| Saved-JPEG endpoint | Matched | Support mean | Exchanged EEG | Matched minus support mean, 95% paired interval |
|---|---:|---:|---:|---:|
| CLIP, supplied 12 labels /360 | 160 | 161 | 160 | −0.28 pp [−0.91, 0.00] |
| CLIP, full 1,854 labels /360 | 145 | 150 | 146 | −1.39 pp [−3.33, +0.63] |
| DINOv2-S/14, original among ten /360 | 43 | 46 | 43 | −0.83 pp [−4.21, +2.43] |
| CLIP, original among ten /360 | 41 | 34 | 30 | +1.94 pp [−1.11, +5.00] |
| DINO, strict two-photo assignment /44 | 33 | 26 | 22 | +15.91 pp [−5.00, +43.75] |
| CLIP, strict two-photo assignment /44 | 28 | 24 | 24 | +9.09 pp [−10.64, +33.33] |
| 32×32 RGB RMSE, lower is better | 0.2940 | 0.2994 | 0.3019 | −0.00535 [−0.00858, −0.00202] |
| 32×32 spatial luminance correlation | 0.0497 | 0.0415 | 0.0414 | +0.00819 [−0.00705, +0.02391] |

The ten-photo gallery contains **all ten original photographs of the true object**, including its eight calibration photos and the two held-out photos. A correct prediction therefore puts all eight generator supports in that gallery. A wrong word leads the generator to another object's eight supports. The current query photograph is always held out from conditioning. The evaluator uses its original pixels to score every query, including word errors. Uniform ten-way choice would average 36/360. The matched DINO estimate was 11.9% [9.1%, 14.9%].

The strict task requires both predicted words in a true-object/fold pair to be correct. It asks which of the two assignments couples both generated JPEGs to their original photographs. Random assignment would average 22/44. Seventeen pairs from nine objects had the two photographs as reciprocal EEG donors. On that subset, DINO credited 11/17 matched, 10/17 support mean and 6/17 exchanged (matched minus exchanged +29.41 pp [8.33, 54.57]). The all-query ten-photo comparisons above give the broader measure of photograph identification.

Intervals use 10,000 paired draws of the 29 object identities. Each selected object retains all of its photographs, arms and participants who saw it. Point estimates weight participant queries equally. The uncertainty describes variation over objects for these three people. The matched arm reduced mean RGB error; its all-query DINO ten-photo accuracy showed no clear gain over either control. [Counts, paired intervals and hashes](../research/eeg_stack_20261112/results/object_fidelity/personal12_image_evaluation.json)

## Fixed image sheets

Each sheet shows the matched generated JPEG for the lowest original row of every object in one participant's registered menu, including word errors. Original-versus-generated sheets stay in the private evaluation outputs. [sub-04](../research/eeg_stack_20261112/gallery/photo_angle_sub04_personal12.jpg) · [sub-05](../research/eeg_stack_20261112/gallery/photo_angle_sub05_personal12.jpg) · [sub-01](../research/eeg_stack_20261112/gallery/photo_angle_sub01_personal12.jpg)

## Replay

From the repository root, set `C=research/eeg_stack_20261112/code` and `R=outputs/object-head-confirmation`. Rebuild the metadata-only registration, source and condition packets, then the separate evaluation references.

```sh
python "$C/register_single_person_confirmation.py"
python "$C/package_single_person_confirmation.py" all
python "$C/score_personal12_images.py" register
```

Use the frozen renderer in the archived generator environment for each participant. On the evaluator machine, `ARCHIVE` is the locally available PKU archive with the original photo manifest, ZIP, offline tokenizer and CLIP weights. `DICTIONARY` is the frozen 1,854-object dictionary. Keep each person's reference directory separate since the three vocabularies differ.

```sh
mkdir -p "$R/evaluation/clip"
for S in sub-04 sub-05 sub-01; do
  python "$C/score_within_photo_ip.py" --root "$R/render_inputs" --subjects "$S" --panels personal12 --reference-dir "$R/evaluation/references/$S" --archive "$ARCHIVE" --dictionary "$DICTIONARY" --out "$R/evaluation/clip/$S.json"
  python "$C/score_reference_names.py" --panel personal12 --references "$R/evaluation/references/$S" --archive "$ARCHIVE" --dictionary "$DICTIONARY" --out "$R/evaluation/clip/$S-refnames.json"
done
python "$C/score_personal12_images.py" pixels
python "$C/score_personal12_images.py" summarize
```

The pixel evaluator verifies the public THINGS ZIP and row mapping and uses offline DINOv2-S/14. The summary verifies the exact registrations, conditions, seeds, per-JPEG checks, independent scorer reports and gallery hashes. [Independent EEG encoder experiments](SINGLE_PERSON_OBJECT_ENCODERS.md)
