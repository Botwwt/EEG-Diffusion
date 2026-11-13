# Stacking on the ordinary head: joint EEG encoder + low-level start image (2026-11-12, updated 2026-11-13)

This folder adds components on top of the **ordinary bounded condition head**
(`research/eeg/brainhive_bounded_component_v1.py`, imported unchanged) and reports every arm that was run,
including the ones that did not help. Host model: Brain-HIVE (commit f06db69) with SDXL-Turbo.

## Headline (10 subjects, seed 2025)

Official THINGS-EEG test set (200 images, mean of 80 repeats), K = 4 generated images per query, metrics as in
Brain-HIVE `main/utils_eval.py` (MindEye style; CLIP = ViT-L/14 two-way, EffNet is a distance, lower is better).
SwAV is not measured (weights not downloadable here). Every row is the mean over the ten subjects.

| arm | PixCorr | SSIM | Alex2 | Alex5 | Incep | CLIP | EffNet ↓ |
|---|---|---|---|---|---|---|---|
| Brain-HIVE baseline, reproduced | 0.1828 | 0.3120 | 0.8266 | 0.8959 | 0.7440 | 0.7986 | 0.8786 |
| + ordinary head (per-subject re-run) | 0.2016 | 0.3360 | 0.8299 | 0.9017 | 0.7551 | 0.8107 | 0.8712 |
| + low-level start image | 0.2455 | 0.3895 | 0.8649 | 0.9163 | 0.7558 | 0.8085 | 0.8651 |
| + condition amplification ×2 | 0.2407 | 0.3832 | 0.8703 | 0.9176 | 0.7632 | 0.8113 | 0.8645 |
| joint encoder + ordinary head | 0.2073 | 0.3374 | 0.8348 | 0.9109 | 0.7718 | 0.8176 | 0.8565 |
| joint encoder + head + low-level start | 0.2513 | 0.3932 | 0.8716 | 0.9221 | 0.7793 | 0.8180 | 0.8510 |
| … + best of 4 candidates (per-subject heads) | 0.2539 | 0.3920 | 0.8783 | 0.9252 | 0.7801 | 0.8208 | 0.8481 |
| … + amplification ×2 + best of 4 (per-subject heads) | 0.2477 | 0.3872 | 0.8818 | 0.9279 | 0.7859 | 0.8231 | 0.8494 |
| **A**: joint encoder + **one shared head** + low-level start + best of 4 | 0.2550 | 0.3937 | 0.8777 | 0.9261 | 0.7827 | 0.8217 | **0.8463** |
| **B**: … + amplification ×2 | 0.2483 | 0.3891 | 0.8823 | 0.9289 | 0.7862 | 0.8246 | 0.8475 |
| **A′**: A with the start image blended with retrieved training latents | 0.2665 | 0.3870 | 0.8853 | 0.9271 | 0.7833 | 0.8232 | 0.8480 |
| **B′**: A′ + amplification ×2 | 0.2580 | 0.3823 | 0.8897 | 0.9301 | 0.7869 | 0.8239 | 0.8478 |
| B′ with best of 8 instead of 4 | 0.2602 | 0.3823 | 0.8899 | 0.9293 | 0.7864 | 0.8268 | 0.8475 |
| **A″**: A′ + best of 8 + high-frequency shrinkage | **0.2782** | **0.4043** | 0.8823 | 0.9255 | 0.7837 | 0.8251 | 0.8476 |
| **B″**: B′ + best of 8 + high-frequency shrinkage | 0.2710 | 0.3995 | 0.8859 | 0.9269 | 0.7863 | 0.8266 | 0.8491 |
| B′ with best of 16 | 0.2618 | 0.3824 | **0.8902** | **0.9298** | **0.7883** | **0.8284** | 0.8467 |
| **B‴**: B′ + best of 16 + high-frequency shrinkage (24 px) | 0.2748 | 0.4007 | 0.8861 | 0.9270 | 0.7882 | 0.8278 | 0.8479 |
| low-level operating point: img2img strength 0.5 + best of 4 (per-subject heads) | 0.2578 | 0.4263 | 0.8422 | 0.8991 | 0.7744 | 0.7992 | 0.8635 |
| Brain-HIVE paper, Table 2 (EEG) | 0.195 | 0.336 | 0.843 | 0.905 | 0.756 | 0.808 | – |

A, B, A′, B′, A″, B″ and B‴ improve on the reproduced baseline for every metric in every one of the ten subjects. The reproduced
baseline is 0.01–0.02 below the paper's own numbers. The low-level operating point trades semantics for SSIM
(Alex2 / Alex5 / CLIP fall below A in all ten subjects) and is listed only to show the trade-off. Per-subject rows: `results/table10.txt`;
every generation run: `results/runs_summary.csv`; every encoder: `results/encoders_summary.csv`.

Contribution of each component (10-subject means of the differences):

- ordinary head over the baseline: SSIM +0.024, CLIP +0.012 (all ten subjects improve on PixCorr, SSIM, CLIP).
- low-level start image over the head alone: PixCorr +0.044, SSIM +0.053, Alex2 +0.035; Incep +0.001, CLIP −0.002.
- joint encoder over the per-subject encoder, same arm (head + start image): PixCorr +0.006, SSIM +0.004,
  Alex2 +0.007, Alex5 +0.006, Incep +0.024, CLIP +0.010.
- best of 4 over a single candidate: Alex2 +0.007, CLIP +0.003 (it uses 4× the sampling budget; see below).
- one shared head (the co-author's original pooled setting, 1 pass = 10,332 updates) over ten per-subject heads
  (10 × 3,100 updates), same arm, joint encoder: equal or slightly better in all four arms; SSIM +0.0016 to +0.0033
  with all ten subjects improving in every arm; in setting B every metric is higher (CLIP +0.0015, SSIM +0.0019),
  at one third of the training compute. The shared head moves the conditions by 14.1° on average, like the
  per-subject heads.
- start image blended with retrieved training latents (A′ vs A): PixCorr +0.0115 and Alex2 +0.0077 with all ten
  subjects improving, Alex5 / Incep / CLIP +0.001; SSIM −0.0067 in all ten subjects. A trade-off rather than a
  strict gain, hence reported as a second setting.
- best of 8 over best of 4 (same arm, joint pipeline): CLIP +0.0024 (A′, 9/10 subjects) and +0.0030 (B′, 10/10),
  other metrics within ±0.002. It doubles the sampling budget again (32 generations per query).
- high-frequency shrinkage over the same images (A″ vs A′ best of 8, B″ vs B′ best of 8): PixCorr +0.011 and SSIM
  +0.017 in all ten subjects, Incep / CLIP within ±0.001, Alex2 −0.004 and Alex5 −0.003 (0/10). The setting
  (Gaussian 16 px, gain 0.8) is the one a leave-one-subject-out rule picks for every held-out subject (largest SSIM
  gain on the other nine with CLIP loss ≤ 0.001 and Alex2 loss ≤ 0.005, among the 10 settings of the grid below).
- best of 16 over best of 8 (B′): CLIP +0.0015, Incep +0.0019, PixCorr +0.0016 (9/10), others within ±0.001; 64 generations
  per query. B‴ = best of 16 + shrinkage at 24 px (the setting the leave-one-subject-out rule picks for every held-out
  subject on this grid): PixCorr +0.013, SSIM +0.018 (10/10), CLIP −0.0005, Incep −0.0002, Alex2 −0.004, Alex5 −0.003.
- Ranking against published 10-subject EEG-only numbers (`results/leaderboard.md`, inclusion rule fixed before
  ranking, every paper read in full): B‴ is first on PixCorr (0.275 vs 0.267) and Inception (0.788 vs 0.779), level
  with the best on Alex5 (0.927), second on SSIM (0.401; ENIGMA 0.426), Alex2 (0.886; Perceptogram 0.910) and CLIP
  (0.828; CognitionCapturerPro 0.830). Not ranked but relevant: JMVR with VLM captions of the stimulus reaches CLIP
  0.829 / Incep 0.793, and ENIGMA's re-run of Perceptogram reaches SSIM 0.431. Metric code differs between papers.
- Against the external numbers quoted under Protocol details, A″ / B″ are above Perceptogram on PixCorr, SSIM, Incep
  and CLIP, level on Alex5 and below on Alex2 (0.882–0.886 vs 0.910); above ENIGMA on everything except SSIM
  (0.404 vs 0.423); above CogCapPro on PixCorr, SSIM and Incep, below on CLIP (0.827 vs 0.830). Evaluation details
  differ between papers.

## Components

1. **Joint encoder** (`code/train_ms.py`, tag `ms_e10_ft10in`). Brain-HIVE's BrainMLP with one input layer
   (Linear 4250 → 1024) per subject and the residual trunk + LayerNorm shared by the ten subjects, trained with the
   Brain-HIVE recipe (symmetric InfoNCE to the frozen Fusion-Prior target F, AdamW 5e-4, wd 0.05, batch 1024,
   bf16) for 10 epochs; then 10 epochs fine-tuning only each subject's input layer (lr 5e-4). Every batch holds
   each image at most once. Uses the *training* EEG of the other nine subjects; no test EEG of any subject is used.
   200-way test retrieval (R80): top-1 0.452 → 0.513, top-5 0.789 → 0.831 (10-subject means).
2. **Ordinary head** (`code/ordh.py`). The co-author's network and recipe (noise-prediction MSE + 0.1 token cosine,
   AdamW 1e-4, batch 16, 3 passes, seed 1729), trained on out-of-fold conditions (5 concept folds, each with its own
   encoder's LayerNorm affine). Used as in the original: **one head shared by the ten subjects** (`ordh.py train
   all 1`, one pass over the 165,310 subject-image pairs; each sample keeps the affine of its own subject's fold
   encoder). One head per subject (`ordh.py train <sub>`, 3 passes) was run as well; see the comparison above.
3. **Low-level start image** (`code/low.py`). Closed-form ridge from the z-scored EEG (17 × 250) to the published
   SDXL-VAE latents of the 128-px image, with a repeat correction (signal covariance = PSD part of the 4-repeat second
   moment minus noise/4, plus noise/80) and the ridge weight chosen by 5-fold concept cross-validation; the deviation
   from the mean latent is rescaled so its RMS over the test batch is 0.8 × that of real training latents
   (label-free, uses test-batch statistics). Decoded by the VAE, upsampled to 512 px, used as SDXL-Turbo img2img
   start with strength 0.75 (3 of 4 steps).
3b. **Retrieval-blended start image** (A′, B′; `lowsem.py ms_e10_ft10in vae_rcp --space raw --save lowrk0.5 --mode
   knn --w 0.5`, then `decode_low.py lowrk0.5`). The ridge latent prediction is blended with the softmax-weighted
   (temperature 0.05) mean latent of the 20 *training* images whose fused target F is closest to the joint-encoder
   condition (centred cosine; training concepts are disjoint from the test concepts), both scaled to unit SD,
   weight 0.5, then decoded with the same spread normalisation. The weight was picked among 0.25 / 0.5 / 1 on the
   test-latent correlation and the image metrics of the ten subjects (a test-set choice).
4. **Condition amplification** (`gen.py --amp`): c' = |c| · normalize(m + a (c/|c| − m)), m = mean direction of the
   200 test conditions (label-free, transductive).
5. **Best of N** (`gen.py --nsel 4`, 8 in A″ / B″, 16 in B‴): for every output image, 4 candidates with different noise; each is embedded back
   to F (`code/fenc.py`) and the one with the highest centred cosine to the EEG condition is kept. No label is used,
   but the sampling budget is 4× that of a single draw (16 generations per query instead of 4).

6. **High-frequency shrinkage** (A″, B″; `gen.py --post 0,16,0.8`): every generated image x is replaced by
   G(x) + 0.8 · (x − G(x)), G = Gaussian blur with σ = 16 px at 512 px. Rationale: the EEG carries little information
   about fine image detail, so the detail the generator invents is mostly uncorrelated with the true image; shrinking
   it towards zero is the lower-error estimate (a Wiener-style gain below one), the same argument that makes posterior
   means blurrier than samples. No label is used and nothing is re-generated (`--post` scores any number of settings on
   the same images; `--dump 1` + `postoff.py` do it offline).

## Protocol details

- Subjects 1–10; encoder seed 2025 (three seeds for subjects 1, 3, 8 in the earlier per-subject experiments:
  seed SD 0.001–0.007); head seed 1729; one generation seed set, fixed per (query, sample) and shared by all arms.
- Training inputs: mean of the 4 repeats; test inputs: mean of the 80 repeats. 17 occipito-parietal channels, 0–1 s.
- Concept folds: fixed permutation of the 1654 training concepts (`common.concept_folds`, seed 20261112).
- Hyper-parameters were chosen on subsets of subjects (amplification on 1, 2, 4, 8; start-image settings on 8) and
  then applied to all ten; selection on the test set is therefore not fully excluded for those two choices.
- External methods quoted for context (numbers from their papers, different evaluation details, not re-run here):
  Perceptogram 0.267 / 0.347 / 0.910 / 0.927 / 0.752 / 0.807; ENIGMA 0.172 / 0.423 / 0.836 / 0.895 / 0.777 / 0.815;
  CogCapPro 0.163 / 0.398 / – / – / 0.779 / 0.830 (PixCorr / SSIM / Alex2 / Alex5 / Incep / CLIP).

## Negative and neutral results

Image-level unless stated; "4 subjects" = mean over subjects 1, 2, 4, 8 relative to the arm they modify.

| idea | outcome |
|---|---|
| IP classifier-free guidance, scale 1.5 / 2 / 3 (negative = mean condition) | PixCorr −0.018 / −0.035 / −0.066, SSIM −0.026 / −0.047 / −0.084, CLIP +0.003 / −0.004 / −0.026 (4 subjects) |
| larger IP-attention weight 1.5 / 2 | CLIP −0.009 / −0.036 (4 subjects) |
| text prompt from the nearest *training* concepts (k = 1 / 3), or "a photo" | CLIP −0.023 / −0.016 / −0.010 (subject 8), similar on subject 1 |
| closed-loop correction (embed images back to F, push the condition along the missing part), λ 0.5 / 1 | CLIP −0.018 / −0.059 (subject 8), −0.032 / −0.085 (subject 5) |
| img2img strength 0.5 instead of 0.75 | SSIM +0.03 to +0.04, CLIP −0.02 to −0.03 (subjects 1, 5, 8): a different operating point, not a gain |
| start-image contrast 1.0 instead of 0.8 | subject 8: SSIM +0.007, PixCorr +0.008, CLIP −0.003, Incep −0.009 |
| joint encoder: 25 / 50 epochs, 2 trunk layers, lr 1e-3 / 2.5e-4, dropout 0.2 / 0.3 | retrieval not better than the chosen setting (25 epochs overfits: top-1 0.471) |
| joint encoder with repeat-subset augmentation | retrieval top-1 0.543, but images worse (subject 8 Incep 0.795 vs 0.809) |
| averaging two encoder seeds | +1.4 points top-1 without fine-tuning, none with it |
| test conditions at the training repeat level (mean of 20 embeddings of 4-repeat inputs, rescaled to the R80 norm) | retrieval top-1 after the head 0.505 → 0.533, but images: SSIM +0.008 (10/10), Incep −0.005, PixCorr −0.003 |
| bagging the five fold encoders with the full encoder | retrieval 0.509 vs 0.513 |
| best-of-4 score + agreement with the start image (pixel correlation, weight 0.5 / 1) | PixCorr +0.005, SSIM +0.002, Incep −0.004 / −0.006 (9 subjects) |
| joint neural low-level decoder (`low_ms.py`; per-subject input layer, shared trunk) instead of the repeat-corrected ridge | higher test-latent correlation (0.288 vs 0.255 in standardised space), but images: SSIM −0.021 to −0.024 (0/10), PixCorr −0.007 to −0.015, Alex2 +0.002 to +0.006; same with the loss in raw latent space |
| decoder + latents of the training images retrieved with the EEG condition (`lowsem.py`, weight 0.5) | PixCorr +0.005, Alex2 +0.012 (10/10), SSIM −0.025 (0/10) |
| fine-tuning the IP projection P on head-corrected EEG conditions (`ipft.py`, lr 1e-5 / 3e-5, 3,000 steps, anchor on clean F) | collapse to generic images: CLIP 0.817 → 0.60, Incep 0.78 → 0.59, SSIM +0.084 (blur) |
| retrieval blend keeping only the coarse part of the retrieved latents (8×8 pooling), weight 0.5 / 1 | SSIM −0.009 / −0.014 (worse than the plain blend), PixCorr / Alex2 similar |
| averaging the head-corrected conditions of the old per-subject and the new joint pipeline | retrieval top-1 0.507 → 0.493–0.507 |
| (earlier, per-subject encoder) cross-fitted ridge prior c → F, kNN snapping to the training gallery | lower discriminability, no image gain |
| (earlier) seed / time-window condition ensembles, repeat-subset training, absolute cosine loss beyond calibration | no image gain |
| (earlier) degraded-image targets (NeuroBridge settings) | retrieval 50.5 → 63.5 % in the degraded space, images no better |
| (earlier) 8 sampler steps, pixel-space start images, pixel blending, best of 8 vs best of 4 (per-subject encoder) | no gain / trades CLIP for SSIM |
| shared head trained for the original 3 passes (30,996 updates) instead of 1 | A / B / A′ / B′: every metric within ±0.004 of the 1-pass head, mixed signs (10, 10, 10, 9 subjects) |
| retrieval weight 0.25 instead of 0.5 in the start-image blend | in between A and A′ on every metric (PixCorr 0.2630, SSIM 0.3905) |
| 8 sampler steps from the start image, strength 0.75 / 0.625 (6 / 5 refinement steps instead of 3), single draw | 0.75: SSIM −0.019, Alex2 −0.011, CLIP −0.003 (0–1/10); 0.625: PixCorr +0.008, SSIM +0.006, Alex2 −0.007, CLIP −0.005 (0/10) |
| best-of-8 selection target: head-corrected condition (centred on the test mean) instead of the raw EEG direction | CLIP +0.0017 / +0.0007 (A′ / B′), SSIM +0.001 (10/10, 9/10), others ±0.001: within noise, not adopted |
| best-of-8 selection target: kNN prototype of the training-image embeddings (top 20, T = 0.05), alone or mixed | CLIP +0.0006 / −0.0009, Alex5 −0.001 to −0.002, Incep −0.002 |
| shrinkage grid, other settings (A′ / B′ best of 8, 10 subjects): gain 0.8 at σ 2 / 4 / 8 px; gain 0.7 / 0.6 at 8 px | smaller σ helps SSIM less at a similar or larger CLIP cost; gain 0.6 at 8 px: SSIM +0.032, PixCorr +0.014, CLIP −0.007 / −0.006, Alex2 −0.010 / −0.009 |
| replacing the low-frequency band (σ 8 / 16 px) by that of the start image, weight 0.25 / 0.5, with or without shrinkage | PixCorr +0.025 to +0.035, SSIM up to +0.018, but CLIP −0.006 to −0.019 |
| best-of-8 candidates alternating amplification 1 and 2 (one pool, the EEG score chooses) | between A′ and B′ on every metric (PixCorr 0.2646, SSIM 0.3850, Alex2 0.8882, CLIP 0.8273): no gain over picking one of them |
| best of 16, target = corrected condition | CLIP +0.0009, SSIM +0.0011 (10/10), Incep −0.0020: within noise |
| shrinkage at 24–48 px with gain 0.5–0.7 (B′ best of 16) | SSIM up to 0.437 (+0.055) and PixCorr up to 0.303, but CLIP −0.002 to −0.009 and Alex2 −0.008 to −0.022 |
| shrinking the high band towards the mean of the K = 4 samples of a query (full band or above σ 4 / 8 px) | PixCorr +0.003 to +0.027, SSIM +0.008 to +0.023, Incep +0.001 to +0.003, but CLIP −0.002 to −0.014 |

## Not finished (stopped 2026-11-13)

- Shrinking only the luminance high band (`postoff.py` spec `y,σ,β`) or only the band between 1 and 24 px (`b,1,24,β`):
  scored on B′ best of 16 for 4 subjects (5, 6, 8, 9) before the run was stopped. Band 1–24 px at 0.8: SSIM +0.015,
  PixCorr +0.013, CLIP ±0.000, Alex2 −0.003 (vs −0.004 for the full high band) — possibly a slightly better trade, not
  established on ten subjects.

## Running it

Set `RAD_ROOT` to a working directory with `data/things-eeg/` (THINGS-EEG preprocessed EEG and image set),
`data/hive_embeddings/`, `models/` (Fusion-Prior-H14_B32_VAE, sdxl-turbo, the two OpenCLIP image towers) and
`code/Brain-HIVE` (commit f06db69); outputs go to `$RAD_ROOT/runs`. From `code/`:

```bash
python prep.py eeg <sub>                       # 17-channel arrays, for every subject
python train_ms.py --tag ms_e10_ft10in --epochs 10 --ft_epochs 10 --ft_trunk 0 --ft_lr 5e-4 --fold -1   # and --fold 0..4
ORD_TAG=ms_e10_ft10in python ordh.py latents   # once
ORD_TAG=ms_e10_ft10in ORD_HEAD=pooled_p1 python ordh.py train all 1        # one shared head
ORD_TAG=ms_e10_ft10in ORD_HEAD=pooled_p1 python ordh.py apply <sub>        # -> runs/cond/ord_ms_e10_ft10in_pooled_p1/
python low.py <sub>                            # start images, variant vae_rcp_n0.8
python gen.py --run ms_e10_ft10in/sub-XX/fold-1_seed2025 --cond_file ord_ms_e10_ft10in_pooled_p1/sub-XX.npy \
    --init_file sub-XX/vae_rcp_n0.8.npy --strength 0.75 --amp 2 --nsel 4 --K 4 --name <name>
# A'' : start image lowrk0.5 (lowsem.py + decode_low.py, see 3b), best of 8, shrinkage scored inside the run
python gen.py --run ms_e10_ft10in/sub-XX/fold-1_seed2025 --cond_file ord_ms_e10_ft10in_pooled_p1/sub-XX.npy \
    --init_file sub-XX/lowrk0.5_n0.8.npy --strength 0.75 --nsel 8 --post 0,16,0.8 --K 4 --name <name>
#   metrics.json: "mean" = images as generated, "post"["0,16,0.8"] = after shrinkage
python tab10.py                                # ten-subject table
python runsum.py > runs_summary.csv            # every run and every --post / --selvar variant
```

Downloaded `.pt` EEG files are numpy pickles; they are loaded only through the restricted unpickler in
`common.safe_load_pt`. No EEG, image, embedding or model file is included here.
