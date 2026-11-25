# Can EEG recover an object's outline within its predicted category?

On 240 photographs, adding a nine-person EEG contour estimate to the eight-support median changed every generated JPEG. It left all 240 independent DINO within-category identification outcomes unchanged. The median alone identified 19/240 exact photographs, as did matched and exchanged EEG. Matched EEG also missed the registered image-level outline criterion. [Image design](../research/eeg_stack_20261112/results/object_fidelity/direct_contour_image_design.json) · [Image results and paired intervals](../research/eeg_stack_20261112/results/object_fidelity/direct_contour_images.json)

## What the EEG predicted

We used U2NetP to segment each foreground, centered it, and scaled its horizontal and vertical coordinates by their respective standard deviations. In each of 24 angular sectors we measured the 95th percentile of the radius, smoothed the log radii, and removed their mean. Mean absolute difference between these profiles measures normalized outer shape. The profile discards location and axis scale and cannot fully represent holes or separated parts. Earlier experiments changed four foreground moments, mixed support latents, or optimized the movement of local image patches. [Foreground and support experiments](PHOTO_STRUCTURE.md) · [Local patch deformation](PATCH_TRANSPORT.md)

Each of 32 held-out generic categories supplied five pairs of query photographs. Each pair shared the other eight photographs of the correct category as supports. Nine people contributed four measured EEG trials each per photograph. We first used EEG to predict 16 components of a local DINO patch representation and select one of the eight supports. Next we used each person's 1,003 EEG features to predict the 24 contour values directly. We estimated a single shrinkage coefficient of 0.53719 from four class-wise cross-validation folds on the other 96 categories. The 32 evaluation categories had already informed the choice to try the direct method, so its results are developmental. [Patch selector design and results](../research/eeg_stack_20261112/results/object_fidelity/support_contour.json) · [Direct estimator design](../research/eeg_stack_20261112/results/object_fidelity/direct_contour_design.json) · [Direct estimator results](../research/eeg_stack_20261112/results/object_fidelity/direct_contour.json)

| Outline for the 320 held-out photos | Mean absolute log-radius error ↓ |
| --- | ---: |
| Most representative single support (medoid) | 0.14739 |
| Single support selected by predicted DINO patches | 0.19059 |
| Single support selected by the query photograph's actual DINO patches, scoring only | 0.19408 |
| Closest legal support to the actual query contour, scoring only | 0.11325 |
| Coordinate-wise median of the eight supports | 0.14183 |
| Median plus matched EEG contour estimate | **0.14101** |
| Median plus the other query's EEG estimate | 0.14202 |

The lower error of the best legal support shows there was useful shape variation among the eight examples. Even the true DINO patch representation selected poor shape matches under the chosen distance. Direct prediction reduced error by 0.00082 against the median (95% object-bootstrap interval 0.00010 to 0.00164) and by 0.00101 against exchanged EEG (0.00041 to 0.00167). Selecting the single closest support to the direct prediction raised error to 0.14798. The continuous prediction qualified for the image test. [Patch selector registration](../research/eeg_stack_20261112/results/object_fidelity/support_contour_design.json) · [Direct estimates and controls](../research/eeg_stack_20261112/results/object_fidelity/direct_contour.json)

## What reached the images

We refit the nine EEG maps on all 128 generic categories, kept shrinkage fixed, and used two previously studied 12-category panels. Each recipient kept the sub-04 predicted word, eight permitted support photos, seed, and original four-step uniform reconstruction. In each active fold the two held-out photographs of each category were excluded from support processing. The editor took the same source JPEG for all four arms, applied 35% of the specified change in log radius with a continuous map that fixes the image border, and encoded each output once. The neutral control received the same JPEG encoding. The recipient's own reference photo entered only at scoring. [Source reconstruction and cohort](FIXED_WORD_PHOTO_DOSE.md) · [Editor and scoring code](../research/eeg_stack_20261112/code/render_direct_contour.py)

| Actual JPEG scores across both panels | Neutral | Eight-support median | Matched EEG | Exchanged EEG |
| --- | ---: | ---: | ---: | ---: |
| Remeasured contour error ↓ | 0.16146 | 0.15527 | 0.15622 | 0.15536 |
| DINO exact photo among ten of the true category, hits / 240 | 17 | 19 | 19 | 19 |
| DINO exact photo among all 120 in its panel, hits / 240 | 12 | 13 | 13 | 13 |
| DINO category among twelve, hits / 240 | 180 | 180 | 180 | 180 |
| RGB error on 32×32 thumbnails ↓ | 0.299732 | 0.299527 | 0.299525 | 0.299527 |
| Luminance correlation on 32×32 thumbnails ↑ | 0.02760 | 0.02861 | 0.02867 | 0.02862 |

Matched EEG increased contour error by 0.00096 against the median (95% interval −0.00086 to +0.00299) and by 0.00086 against exchanged EEG (−0.00359 to +0.00493). Intervals resample twelve whole categories separately within each panel, pairing all arms across 10,000 draws. The matched and exchanged JPEGs differed for all 223 rows with an eligible donor. Those donors shared the recipient's predicted word and fold, and 131 also shared its true category. All three edited arms had identical per-row DINO correct/incorrect outcomes on each retrieval endpoint. The small contour gain before editing did not establish a gain in photograph fidelity. [Per-panel results, eligibility and checksums](../research/eeg_stack_20261112/results/object_fidelity/direct_contour_images.json)

## Reproduce

From the repository root, stage the nine-person EEG receipts and original ordinary cohort-9 renders using the fixed-word study. The contour scripts expect the official row CSV under `outputs/official-object-readout/sub-03`, the verified THINGS training photo ZIP, U2NetP, and DINOv2-S/14 under `outputs/photo-structure`, and the physical EEG under `outputs/object-transfer`. The measured run used Python 3.12, PyTorch CUDA for DINO and ONNX Runtime DirectML for U2NetP. The scripts check the model and archive digests. [Baseline replay](FIXED_WORD_PHOTO_DOSE.md#复现) · [Image assets and preprocessing](PHOTO_STRUCTURE.md) · [Patch selector](../research/eeg_stack_20261112/code/fit_support_contour.py) · [Direct estimator](../research/eeg_stack_20261112/code/fit_direct_contour.py)

```sh
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
C=research/eeg_stack_20261112/code
for step in patches contours select score; do
  python "$C/fit_support_contour.py" "$step" || exit 1
done
for step in calibrate predict score; do
  python "$C/fit_direct_contour.py" "$step" || exit 1
done
for step in conditions edit score; do
  python "$C/render_direct_contour.py" "$step" || exit 1
done
```

All private row scores and JPEGs go to `outputs/eeg-shape-recovery`. The last stage validates both sealed image manifests, reopens all 960 JPEGs, and measures them against the 240 registered reference photographs. Each script's `score --publish` writes its aggregate JSON to the research archive.
