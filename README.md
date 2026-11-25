# EEG-Diffusion

Reconstructing viewed photographs from EEG. The primary task is to draw the right object from four presentations of one photograph to one participant. We also test whether the image preserves details that distinguish that photograph from nine others of the same object.

[中文研究摘要](docs/CURRENT_RESEARCH.md) · [Experiment registry](docs/EXPERIMENT_REGISTRY.md) · [Reconstruction galleries](research/eeg_stack_20261112/gallery/GALLERY.md) · [Reproduction guide](docs/REPRODUCIBILITY.md)

[![checks](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml/badge.svg)](https://github.com/Botwwt/EEG-Diffusion/actions/workflows/checks.yml)

## Selected model for calibrated objects

**For a person with four EEG presentations and twelve calibrated object candidates, use the two EEG encoders trained on 1,024 objects.** The first encoder processes four separate 63-electrode trials. A posterior Fisher projection supplies a second score. The model combines their standardized scores with weights of 59.1% and 40.9%, fitted on training data from sub-04 and then fixed for sub-05 and sub-01. Each participant has their own two fitted encoders. The predicted word selects eight permitted support photos. The visual decoder combines their average H/14 image feature with a photo-specific estimate from EEG. A frozen IP-Adapter and four SDXL-Turbo steps produce one image. The query photo is used in evaluation. [First participant and inference](docs/FRESH_MIX.md) · [Replication in two more people](docs/FRESH_PEOPLE.md) · [Image results, fixed galleries and replay](docs/FRESH_PEOPLE_IMAGES.md)

### What improved on saved images

In the two-person replication, expanding training from 512 to 1,024 objects helped the generator draw the correct kind of object. Both people saw the same 350 primary photographs. Their recording sessions determined how the 35 objects were grouped into twelve-object menus. Each photo was held out from its eight support examples. The intervals below resample objects, keeping both people's observations together. [Original registration and paired results](docs/FRESH_PEOPLE_IMAGES.md)

| Endpoint on the two replication participants | Original / 700 | Expanded / 700 | Paired gain in percentage points (95% interval) |
| --- | ---: | ---: | ---: |
| EEG selects the correct object among twelve | 253 | **281** | +4.00 [1.29, 6.57] |
| Independent CLIP names the generated image among 1,854 objects | 218 | **243** | +3.57 [0.86, 6.14] |
| DINOv2 identifies the viewed photograph among ten of the same object | 67 | 72 | +0.71 [−1.00, 2.57] |

Full-vocabulary naming improved from 107 to 123.5 out of 350 in sub-05, and from 111 to 119.5 in sub-01. Tied first places split the credit. Exact-photo retrieval with the expanded model was 10.3%, close to the 10% chance rate. Across the primary and supplemental objects, all 631 pairs with unchanged words gave identical JPEG bytes in independent draws. [Two-person word results](docs/FRESH_PEOPLE.md) · [Generated-image comparison](docs/FRESH_PEOPLE_IMAGES.md)

### Earlier 512-object model on three session-matched menus

Each participant's twelve candidate objects shared a recording-session pair. The 360 participant–photo queries cover 290 distinct photographs of 29 objects. The models use the same diffusion seed for each query. On all 264 queries where the word stayed the same, the generated images matched byte for byte. [Panel registration and original Fisher results](docs/PERSONAL12_CONFIRMATION.md) · [Paired image scores and intervals](research/eeg_stack_20261112/results/object_fidelity/personal12_trial_replication_images.json)

| Endpoint on 360 queries | Fisher | Trial encoder + Fisher | Paired difference in percentage points (95% interval) |
| --- | ---: | ---: | ---: |
| Correct EEG object word among 12 | 161 | **187** | +7.22 [3.14, 11.43] |
| Generated JPEG named correctly among 12 words | 160 | **187** | +7.50 [3.53, 11.62] |
| Generated JPEG named correctly among 1,854 words | 145 | **166** | +5.83 [2.06, 9.71] |
| Viewed photo ranked first among ten photos of its object, DINOv2 | **43** | 42 | −0.28 [−2.65, 2.35] |

The first participant gained ten full-vocabulary image hits. The two separately registered replications gained eleven together (from 100 to 111 of 240 images), with +12 hits in sub-01 and −1 in sub-05. The 512-object fusion more often depicts the correct object than Fisher alone on these menus. Recognition of the particular photograph remains near one in ten. Intervals resample objects while keeping the observed people together. [First comparison and replication](docs/TRIAL_OBJECT_PERSONAL12.md) · [All twelve objects per person, side by side](research/eeg_stack_20261112/gallery/GALLERY.md#personal-twelve-object-menus-with-matching-recording-sessions)

### First evaluation of the expanded decoder in sub-04

On four new session-matched menus, the larger personal decoder improved the object shown in the saved JPEG. Both EEG branches trained on 1,024 other objects. Training-only calibration assigned 59.1% of the score to the trial encoder and 40.9% to Fisher. Each query still used four presentations and eight supports per candidate. The visual decoder and diffusion sampler used the same weights and query seeds. [Training and single-query inference](docs/FRESH_MIX.md)

| Endpoint on 350 photos of 35 held-out objects | Original 512-object model | Larger model | Paired change in percentage points (95% interval) |
| --- | ---: | ---: | ---: |
| Correct EEG word among 12 | 159 | 176 | +4.86 [1.14, 8.57] |
| Generated image named correctly among 1,854 words | 141 | 157.5 | +4.71 [1.29, 8.14] |
| Exact photo identified among ten of its object | 44 | 46 | +0.57 [−1.71, 2.86] |

The half credit comes from a tie in full-vocabulary naming. Across all 480 queries including the supplemental objects, the 365 with unchanged EEG words gave byte-identical JPEGs from the two models. The two additional participants above used the same mixture weights. [Evaluation on saved JPEGs and fixed galleries](docs/FRESH_MIX_IMAGES.md)

### Can the EEG place the object like the viewed photograph?

On the same 360 queries we held the generated image fixed and allowed an EEG readout to move, resize and rotate its foreground. The encoder predicted foreground orientation better than the eight-photo mean on 96 held-out development objects in the same dataset. On the final edited JPEGs, DINOv2 identified 44/360 exact photos with the query EEG, 45/360 with the eight-photo mean and 42/360 with exchanged EEG. Re-encoding the unedited source gave 41/360. Full-vocabulary naming fell from 166/360 in the matched-codec control to 151/360 after the query EEG edit. The support mean accounted for most of the measured layout improvement. The selected model above still uses the original diffusion output. [Edited images, controls and replay](docs/PERSONAL12_POSE_EVALUATION.md) · [EEG pose readout](docs/PERSONAL_POSE_READOUT.md) · [Four-region visual feature experiment](docs/REGIONAL_PHOTO_FEATURES.md)

### Results on an earlier shared 12-object panel

We locked the 12 objects, 120 photos, folds, model and primary photo-identification endpoint before inspecting their EEG or scoring images. The same three previously studied people viewed all 120 photos. Each row below includes every query, including incorrect EEG words. [Registration and per-participant image results](docs/FRESH_PHOTO_VALIDATION.md)

| Outcome on the 360 participant–photo queries | Own EEG | Eight-support visual mean | Wrong EEG from the same predicted word and fold |
| --- | ---: | ---: | ---: |
| Correct EEG word among 12 | 209/360 (58.1%) | Same words | Same words |
| Generated image named correctly among 1,854 labels, CLIP-L/14 | 135/360 (37.5%) | 141/360 | 142/360 |
| Viewed photo ranks first among ten of its object, DINOv2 | 44/360 (12.2%) | 51/360 | 43/360 |
| Viewed photo ranks first among ten of its object, CLIP-L/14 | 29/360 (8.1%) | 32/360 | 38/360 |

With naming restricted to the 12 registered words, CLIP recognized the generated objects in 208/360 queries. The same scorer assigned the correct name to 71/120 **original photographs** among 1,854 labels, and to 119/120 among twelve. For the primary DINOv2 photograph test, chance is 10%. Own EEG minus the eight-support mean was **−1.94 percentage points** (paired 95% interval −5.28 to +1.39). The model can often draw the correct kind of object. Identifying the exact viewed photograph remains an open problem. [Original-photo naming by object](research/eeg_stack_20261112/results/object_fidelity/fresh_reference_names.json) · [Photo-level scores and uncertainty](research/eeg_stack_20261112/results/object_fidelity/fresh_photo_dino_pixels.json)

On a fourth panel with **twelve different objects**, the same posterior readout decoded the correct word for **232/360** queries. CLIP named 232/360 generated images among those twelve words and **199/360 (55.3%)** among 1,854 labels. Same-object photo identification was **42/360** with DINOv2, compared with **31/360** for the mean and **38/360** for wrong EEG. A proposed 63-electrode object readout gave 228 correct words, 199 full-vocabulary image hits and 40 same-object photo hits. We retained the 17-electrode readout. The two panels use different objects and share participants. [Fourth-panel design, paired comparison and examples](docs/FULL_HEAD_512.md)

![One EEG-generated image for the lowest archive row of each of the twelve fresh objects in sub-01](research/eeg_stack_20261112/gallery/photo_angle_sub01_fresh.jpg)

Each tile shows the true object label and the word decoded from EEG. A single fixed rule selected the examples, including errors. The replay guide also builds private side-by-side comparisons with the original stimuli. [Sub-04 gallery](research/eeg_stack_20261112/gallery/photo_angle_sub04_fresh.jpg) · [Sub-05 gallery](research/eeg_stack_20261112/gallery/photo_angle_sub05_fresh.jpg) · [Image replay](docs/FRESH_PHOTO_VALIDATION.md#reproduce-the-saved-image-comparison)

## Single-person reconstruction without target-class EEG supports

On the official 200-photo test, sub-04 viewed each stimulus 80 times. The decoder can generate with a continuous EEG estimate and the same prompt, “A photograph.”, for every query. Scaling the estimated visual deviation from the training-image mean raised DINOv2 retrieval from **10/200 to 20/200** against the actual photographs. The paired gain was 5.0 percentage points (95% interval 1.0 to 9.0). Independent CLIP photo retrieval rose from **5/200 to 17/200**. Full 1,854-label naming gave **2/200** for both versions. With the EEG-decoded object word added to the prompt, angular calibration raised DINOv2 retrieval from **13/200 to 24/200** (paired gain 5.5 points, 95% interval 1.0 to 10.0). CLIP named **4/200** correctly at both angles. At the calibrated angle, the named arm retrieved four more photos than the generic prompt with DINOv2 (95% interval for the difference −2.5 to +6.5 points). Each test object has a single photograph, so this benchmark cannot resolve same-object photographic detail. [Physical EEG and calibration](docs/OFFICIAL_REPEAT_OBJECT.md) · [Generated images, naming and controls](docs/OFFICIAL_REPEAT_IMAGES.md)

We also compared two visual decoders of the **same 80 presentations**, holding the decoded word and seed fixed. Independent DINO retrieval gave 24/200 for the personal head, 29/200 for a head fitted to released ATM features and 32/200 for their equal blend. CLIP full-vocabulary naming on the generated images gave 4, 4 and 3 hits respectively. We retained the personal model for the object-first task. [Paired results and upstream training details](docs/DUAL_R80.md) · [Fixed gallery](research/eeg_stack_20261112/gallery/official_dual_r80.jpg)

## Other experiments

With the object word fixed, a separate study pooled four presentations from each of nine viewers of the same photograph. Across two previously studied panels, the generated object was named correctly on **180/240** queries within the panel vocabulary. Independent DINOv2 identified the viewed photo among ten of the same object on **30/240** ordinary pooled reconstructions. Doubling the difference between matched and support-mean diffusion predictions gave **37/240**, compared with **22/240** for exchanged EEG. The gain over the ordinary pooled reconstruction was 2.9 percentage points (95% interval −0.4 to +6.7), and RGB error increased. Each query uses 36 physical presentations. [Fixed-word acquisition study and image controls](docs/FIXED_WORD_PHOTO_DOSE.md)

| Question | Evidence | Results and code |
| --- | --- | --- |
| How much object recognition survives a recording-session control? | On twelve objects recorded in the same combination of sessions, CLIP named 55/120 single-person and 69/120 three-person generated images among the twelve labels. Full-vocabulary credits were 48 and 59.5. DINO identified 15/120 exact photos for each. Three people contribute twelve physical observations per picture. | [Session audit and matched-object experiments](docs/COHORT_SESSION_CONTROLS.md) · [Fixed examples](research/eeg_stack_20261112/gallery/GALLERY.md#session-matched-objects) |
| Where is the generator losing the identity of the photo? | On 120 third-panel images from sub-04, replacing only the EEG word by the true word gave 13 same-object photo hits versus 14 from EEG. Supplying the exact viewed-photo H/14 feature instead gave 74. The latter is a reference-input diagnostic. | [Condition substitutions, pixel scores and paired intervals](docs/PHOTO_CONDITION_DIAGNOSTIC.md) |
| Which EEG features distinguish photographs of the same object? | On 96 held-out generic objects, direct H/14 prediction gave 9,371/12,960 correct pair assignments. Cross-session reproducibility gave 9,569 and cross-participant supervision gave 9,589. All three people improved in feature space. | [Repeat recordings](docs/REPEAT_PHOTO_SUBSPACE.md) · [Other participants as training teachers](docs/SHARED_PHOTO_EEG.md) · [Generator-gradient experiment](docs/IP_DENOISER_GRADIENT.md) |
| Does personal calibration help draw the right object? | With 128 other calibration objects and seven additional people, Brain-HIVE image naming rose from 503/1,680 (29.9%) to 678.5/1,680 (40.4%). On the earlier panels in three people, expanding calibration from 128 to 512 objects raised naming from 321.5/720 to 406.5/720. | [Seven participants](docs/OBJECT_FIDELITY.md) · [Calibration-size comparison](docs/PHOTO_STRUCTURE.md#personal-calibration-size-and-scalp-coverage) |
| Does the H/14 condition carry information about the individual photo? | On two earlier 12-object panels, CLIP named 433/720 own-EEG images. When both EEG words were correct and two photos had to be assigned jointly, CLIP gave 115/170 for own EEG, 89/170 for the mean and 64/170 for wrong EEG. All-query ten-photo identification was 65/720 for CLIP and 78/720 for DINOv2. | [H/14 angle and eight-candidate controls](docs/PHOTO_STRUCTURE.md#calibrating-the-residual-angle-before-generation) |
| Can EEG recover color and composition? | Low-frequency color and local contrast improved a strict grayscale SSIM comparison against swapped EEG in three participants on fresh objects. All four spatial variants had 39/72 correct DINOv2 pair assignments. | [Spatial residuals](docs/SPATIAL_PHOTO.md) · [Foreground and support experiments](docs/PHOTO_STRUCTURE.md) · [Local visual prediction errors](docs/CONTEXT_RESIDUAL.md) |
| Which model leads the separate ten-participant R4 benchmark? | M1a + D2 at s375 identified 982/5,000 photographs (19.64%) within concept. All ten participants saw the same 500 photographs. The benchmark uses VAE-decoded references and a shared noise seed within each concept. | [Architecture, matched controls and evaluation protocol](docs/EXPERIMENT_REGISTRY.md) |
| What happens with unfamiliar objects or more repeats? | On a separate sub-03 R4 panel without target-class examples, 60/200 generated images were assigned to the correct one of 20 concepts. The separate official 80-repeat, 200-stimulus benchmark gave 18.46% for O14 across ten people. | [Unregistered-object R4 study](docs/OBJECT_FIDELITY.md) · [Official R80 study](research/eeg_stack_20261112/O12_O13_O14_REPORT.md) |

## Run the checks

The public numerical audits run on a CPU with Python 3.11 or newer. GPU reconstruction additionally requires the EEG and image archives, model weights and research environments listed in the replay guide. [Installation](docs/INSTALLATION.md) · [Model assets and GPU replay](docs/REPRODUCIBILITY.md)

```bash
python -m pip install -e '.[dev]'
python -m eeg_diffusion verify-r4
python -m eeg_diffusion verify-stack
python -m eeg_diffusion verify
python -m pytest -q
```

[Citation](CITATION.cff) · [MIT license](LICENSE) · [Third-party assets](THIRD_PARTY_NOTICES.md)
