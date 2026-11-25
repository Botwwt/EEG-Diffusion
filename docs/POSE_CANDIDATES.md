# Can EEG choose a more faithful generated photo from eight candidates?

A selector that combined foreground pose and H/14 image features identified the original photograph in **69/720** queries. Exchanging EEG between queries of the same predicted object and fold gave **73/720**. The eight generated candidates contained a DINOv2 ten-way hit for **164/720** queries, as determined after the selections were frozen. The gap to that ceiling did not translate into a reliable improvement from the query EEG. [Design registered before selection](../research/eeg_stack_20261112/results/object_fidelity/pose_candidate_design.json) · [Results and paired intervals](../research/eeg_stack_20261112/results/object_fidelity/pose_candidate.json)

## How the choice was made

Three participants each contributed four measured EEG presentations for each of 240 photos, split between two 12-object panels. We reused all 960 existing four-step SDXL-Turbo JPEGs. For each panel, fold and predicted object, eight seeds produced eight candidates with the same word and eight-support image condition. Every method could return one of these same eight JPEGs. A different participant or query could therefore select the same JPEG. The earlier H/14 selection and reverse image-to-EEG encoding results on these panels were known. [Original bank and image hashes](../research/eeg_stack_20261112/results/object_fidelity/generated_bank_inputs.json) · [Previous reverse-encoding selector](EEG_ENCODING_BANK.md)

The motivation was that two candidates can match the same object word and differ in their position, extent, elongation or orientation. We measured five foreground coordinates on each candidate JPEG using U2NetP. Each participant's pretrained pose decoder estimated a change relative to the EEG average of the eight authorized supports for that query's predicted word. Its posterior and full-trial mappings had been fit on 512 other objects, with their gains calibrated on generic objects. The origin for matching the predicted shift was the mean pose of the eight **generated** candidates in the active cell. This allows for differences between the layout of natural and generated photos. The selector accessed the permitted support EEG, the existing H/14 scores and the synthetic JPEGs. [Frozen pose readout](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_readout.json) · [Image-pose measurement](PERSONAL12_POSE_EVALUATION.md)

We ranked the eight candidates by distance to the predicted pose, with each component scaled by the generic training variation. We also ranked them by the previously frozen H/14 residual cosine score. Equal weights combined the two ranks. Rank ties received the average rank and a final tie selected the lowest seed index. The mean control used zero EEG pose displacement and cosine to the eight-support H/14 prior. The wrong-EEG control exchanged both modalities within the predicted word and fold. All 720 selections were saved and a second run reproduced each selection packet byte for byte before scoring. [Selector and independent scoring](../research/eeg_stack_20261112/code/score_pose_candidates.py)

## Evidence from the selected generated images

| Choice from the eight | DINO ten-photo identification /720 | DINO two-photo assignment /170 | CLIP ten-photo identification /720 | CLIP full-vocabulary naming /720 |
| --- | ---: | ---: | ---: | ---: |
| First fixed seed | 63 | 85 | 66 | 426 |
| Original H/14 support mean | 57 | 85 | 62 | 438 |
| Original H/14 query EEG | 60 | 85.5 | 59 | 429.5 |
| Pose-only query EEG | 68 | 86.5 | 50 | 438 |
| Pose + H/14 support mean | 64 | 85 | 65 | 443 |
| Pose + H/14 query EEG | **69** | **83** | **61** | **431** |
| Pose + H/14 exchanged EEG | 73 | 89 | 59 | 431 |
| Evaluation-only DINO oracle | 164 | 154 | 75 | 426 |

Ten-way identification compares every selected JPEG with all ten original photos of the viewed object, including incorrect EEG words. Uniform guessing would average 72/720. The strict assignment uses the 170 pairs where the EEG correctly predicted the object for both held-out photos. An exact tie between two assignments gets half credit, which also applies when both queries choose the same JPEG. CLIP naming used all 1,854 labels. [Machine results by participant and panel](../research/eeg_stack_20261112/results/object_fidelity/pose_candidate.json)

The primary evidence required a positive lower 95% bound against each of three controls. On DINO ten-way identification the joint selector gained **1.25 percentage points** against H/14 alone (95% paired interval −0.56 to +2.92), gained **0.69** against the combined mean (−1.25 to +2.64), and lost **0.56** against exchanged EEG (−2.36 to +1.25). The intervals resampled the 24 distinct objects, keeping together every participant and photo for each sampled object. Against the combined mean, full-vocabulary naming declined by **1.67 points** (−3.47 to −0.14). The joint selector changed the original H/14 choice in 415/720 queries.

| Participant | Joint EEG | Exchanged EEG | Joint mean | H/14 EEG | DINO oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| sub-04, 240 queries | 24 | 26 | 25 | 19 | 57 |
| sub-05, 240 queries | 25 | 22 | 20 | 22 | 58 |
| sub-01, 240 queries | 20 | 25 | 19 | 19 | 49 |

For each query, the DINO oracle saw the original and the nine same-object rivals and selected the candidate with the largest correct-photo margin. We verified independently that this achieved the best available ten-way credit on every row. It found a hit on 99/470 correct-word and 65/250 wrong-word queries. A wrong-object reconstruction can resemble one of ten photos more than the rest. The oracle ceiling concerns this within-object ranking test. Its selected images reached 75/720 on the separate CLIP ten-way measure, using the unchanged DINO choices. The different figures show the dependence on the scoring representation.

## Generated examples and replay

Each sheet shows the lowest archive row for every object on sub-04's panel. The displayed candidate index runs from 0 to 7. The final column uses the true-photo DINO oracle. All displayed tiles are generated photos from the existing bank. [First panel](../research/eeg_stack_20261112/gallery/pose_candidate_sub04_registered.jpg) · [Second panel](../research/eeg_stack_20261112/gallery/pose_candidate_sub04_replicate.jpg)

Restore the generated bank with its pinned JPEGs, selection receipts and two evaluator caches. Restore the three personal pose models, full-trial encoder checkpoints and original registered four-repeat EEG range blocks with their receipts. The selection stage checks all range hashes, reproduces the two posterior trial means from the full 63-electrode streams, verifies that all 512 training objects are separate from the evaluation objects, and checks every active fold's eight supports. It uses the previous H/14 scores and replays the historical selections. The scoring stage verifies the sealed selection packets and the independent evaluator caches before reading target image features. [Bank preparation and evaluator caches](PHOTO_STRUCTURE.md#replay) · [Pose and physical inputs](../research/eeg_stack_20261112/code/fit_personal_pose.py)

~~~sh
C=research/eeg_stack_20261112/code
python "$C/score_pose_candidates.py" select
python "$C/score_pose_candidates.py" score \
  --release research/eeg_stack_20261112/results/object_fidelity/pose_candidate.json
python "$C/gallery_pose_candidates.py"
~~~

The selection command uses the repository's four-trial PyTorch environment with safetensors and ONNX Runtime. The original run used CUDA for the trial encoder and DirectML for U2NetP. `--provider CPUExecutionProvider` selects ONNX Runtime's CPU provider on other workstations. The scoring step uses the frozen DINO and CLIP features of the actual JPEGs and runs on CPU.
