# Do photographs of an object improve EEG object identification?

Neither of two development experiments met its recorded rule for advancing to a new panel or image generation. Giving a classifier eight photographs of each candidate tied the original EEG classifier at 377/720. With zero target-class calibration examples, a different method using an independent photograph per word gained five correct object identifications in 640 trials. Its paired interval included zero. Both experiments used four measured presentations per person and photograph. [Aggregate results and replay hashes](../research/eeg_stack_20261112/results/object_fidelity/object_prior_checks.json)

## Eight support photographs per candidate

For each of three participants, a ridge fitted on 512 generic classes mapped within-class CLIP H/14 photo differences into a 96-dimensional EEG Fisher space. A candidate's eight allowed photos then estimated the covariance around its measured eight-EEG mean. Each query retained the original Fisher score and received a fixed Gaussian likelihood correction. The shuffle moved complete sets of eight photographs among candidates in the same menu while preserving all EEG means. Both photos in the query fold were excluded from support. [Frozen design](../research/eeg_stack_20261112/results/object_fidelity/support_image_covariance_design.json)

| Participant | Original Fisher | Support covariance | Shuffled covariance |
|---|---:|---:|---:|
| sub-04 | 134 | 132 | 136 |
| sub-05 | 112 | 114 | 112 |
| sub-01 | 131 | 131 | 126 |
| Total | 377 | 377 | 374 |

The 720 decisions cover 240 distinct photos shared by three separately fitted participants. One panel offered 12 session-matched objects. The other supplied one of three fixed four-object menus. The candidate gained 0.00 percentage points against Fisher (95% interval −2.92 to +2.64) and +0.42 against the shuffle (−1.94 to +2.64). Only sub-05 gained over Fisher. Intervals resample whole objects within each panel and keep the three people together. Both panels had already been studied. All nine native decision arrays matched the historical calls, including the secondary twelve-way reading of the four-choice panel. [Panel registration and session controls](COHORT_SESSION_CONTROLS.md)

In a separate generic-class check, models trained on 384 classes and tested on 128. Repeated menus produced 34,560 decisions on 1,280 distinct photos across the three people. Original, candidate and shuffled scores were 21,031, 18,963 and 18,932. The candidate lost 5.98 points against Fisher (−6.80 to −5.16).

## External photographs with zero target-class supports

For sub-04, a kernel ridge fitted on 884 objects mapped one independent THINGSplus reference photograph per word into an estimated class-average visual embedding. A query's EEG assigned probabilities to all 1,854 words. Their weighted reference embedding adjusted its continuous H/14 direction while preserving its calibrated angle. Another 64 classes selected the temperature (0.846) and weight (0.25). A disjoint 64 previously studied classes with ten photos each supplied the evaluation. An audit compared all 1,854 external references with the 1,280 calibration and evaluation photographs and found zero flagged pairs under its byte, thumbnail and perceptual-hash rules. [Frozen design](../research/eeg_stack_20261112/results/object_fidelity/external_word_prior_design.json)

| Top-one hits out of 640 | Global angle | Matched prior + EEG | Wrong category prior | Wrong EEG within object |
|---|---:|---:|---:|---:|
| Object among 64 image centroids | 83 | 88 | 77 | 114 |
| Photo among ten of its object | 127 | 135 | 129 | 79 |
| Photo among all 640 | 28 | 29 | 20 | 15 |
| Name among 1,854 text embeddings | 0 | 0 | 0 | 1 |

Against the global-angle baseline, matched object identification gained +0.78 percentage points (95% interval −0.47 to +2.19). Against the wrong-category prior it gained +1.72 (−0.31 to +3.91). Both fell short of the positive lower bounds required for rendering. Within-object photo identification gained +1.25 points against the baseline (−0.16 to +2.81) and +8.75 against a different photo's EEG direction from the same object (+4.53 to +12.97). Intervals use 10,000 paired whole-object resamples. The latter comparison establishes sensitivity to the continuous EEG input on these validation photos.

The object gallery excludes the recipient's photo ordinal from all 64 centroids. When another same-object photo supplies the EEG direction, its own photo remains in its object's gallery centroid. The 114 hits in that control therefore cannot establish an object-identification gain. The EEG word classifier itself picked 11/640 correct names among all 1,854. The text row above scores the *downstream visual vectors*. These experiments score decisions and features. They produced no new image or official-200 result. [Existing global-angle experiment](OFFICIAL_REPEAT_IMAGES.md)

## Replay

From the repository root, use Python with NumPy, SciPy and Pillow and the existing EEG feature, training-image and reference caches. The covariance stage reads `outputs/photo-structure/expanded`, `outputs/double-calibration`, `outputs/session-matched-cohort` and the official training H/14 bank under `outputs/official-object-readout/sub-03`. The external-prior stages also use `outputs/official-repeat-object` (training and angle registrations), `outputs/external-object-anchors/sub-03` (independent reference bank and audit) and `outputs/photo-structure/training_images.zip`. The individual EEG in this experiment belongs to sub-04. The shared `sub-03` directory supplies stimulus features and the vocabulary.

```sh
C=research/eeg_stack_20261112/code
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python "$C/fit_support_image_covariance.py"
for stage in audit calibrate validate; do
  OPENBLAS_NUM_THREADS=5 OMP_NUM_THREADS=5 python "$C/fit_external_word_prior.py" "$stage" || exit 1
done
python "$C/summarize_object_priors.py" --strict-replay --out outputs/object-prior-replay/summary.json
```

Private outputs default to `outputs/object-prior-replay/{covariance,external}`. Both fitters accept `--out`. The summary records hashes and aggregate counts. On the retained CPU environment the covariance decisions, external EEG conditions, collision audit, calibration, and individual evaluation arrays matched their original bytes. Omit `--strict-replay` to report hashes from a different numerical environment.

## Related methods

ATM already used EEG-derived visual and optional caption conditions for diffusion. Brain‑HiVE aligned brain signals to fused visual embeddings and generated images with a frozen diffusion model and an empty text prompt. The two interventions here tested the extra information supplied by candidate-specific support covariance and an external visual reference bank. [ATM, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/ba5f1233efa77787ff9ec015877dbd1f-Paper-Conference.pdf) · [Brain‑HiVE, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c5506f154e5f2203a7dca2bee2faeced-Abstract-Conference.html)

I-JEPA predicts hidden-region embeddings from visible regions of the same image. Here the fitted quantities are within-object visual and EEG differences or an EEG-weighted direction relative to a calibrated mean. Subtracting a single image's error from a fixed mean would merely center and reverse its error score. [I-JEPA, CVPR 2023, §3](https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf)
