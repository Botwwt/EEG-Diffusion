# Using one support photo to initialize a reconstruction

Can EEG pick a better starting view than the average of eight photos of the predicted object? We tested this question on the 120 previously studied queries from sub-04's personal 12-object panel. Each query uses four physical presentations and the frozen mixed trial-encoder/Fisher word. This is a support-image-assisted reconstruction: the generator receives coarse color and shape from other photographs of the selected object. [Registered experiment](../research/eeg_stack_20261112/results/object_fidelity/support_view_design.json) · [Source classifier and personal menu](TRIAL_OBJECT_PERSONAL12.md)

## Inputs available to the generator

For each of five folds, each candidate object supplied eight labeled photos and their EEG. The two query photos of that object, along with the other 22 queries in the active fold, were excluded from its supports. The selector compared the eight allowable H/14 image embeddings with the frozen EEG-derived H/14 direction. Every photo could participate as a support in a different fold. We independently rebuilt all 480 starting thumbnails with access to current-fold query photos prohibited. Every byte, selected support row and source-stimulus hash agreed with the prepared packet. [Preparation and renderer](../research/eeg_stack_20261112/code/support_view.py)

| Arm | Starting thumbnail |
| --- | --- |
| Mean | Pixelwise average of the eight permitted RGB thumbnails. |
| Medoid | The one support closest by cosine to their mean H/14 direction. |
| Matched | The support closest by cosine to the recipient's EEG-derived H/14 direction. |
| Exchanged | The support closest to another query's H/14 direction in the same predicted word and fold, retaining the recipient for a singleton. |
| Native comparison | The archived ordinary four-step text-and-H/14 reconstruction from random noise. |

Only the starting thumbnail changes across the four new arms. Every arm retains the recipient's matched continuous H/14 condition, the predicted word, image dimensions and original row seed. The renderer downsamples each support crop to 32 × 32 RGB, bilinearly upsamples it to 512 × 512 and encodes the VAE posterior mode. It uses the final two Euler ancestral updates of the four-step Turbo schedule. The native comparison used four updates and a noise start, so its difference also includes the change in sampling procedure. [Generator and frozen controls](../research/eeg_stack_20261112/code/support_view.py)

The matched choice differed from the medoid on 41/120 rows and from the exchanged choice on 48/120. Thirteen exchanged donors were the query itself. On a further 59 rows the donor was a different query that selected the same permissible support photo. All 120 rows contribute to the primary endpoint.

## Evaluation

DINOv2-S/14 compares each completed JPEG with all ten original photos of the viewed object. All queries count, including incorrect object words. When the predicted word is right, the ten-way gallery contains both held-out originals and all eight support photos. A two-way comparison uses only the two held-out originals, restricted to object/fold pairs with both predicted words correct. It reports the joint assignment of two JPEGs and the independent decisions of each query. There are 22 such pairs for sub-04. Uniform guessing would average 12/120 ten-way hits and 11/22 joint assignments. [Independent scoring](../research/eeg_stack_20261112/code/score_support_view.py)

The control for object identity uses OpenAI CLIP-L/14 with both the 12 supplied words and the full 1,854-word dictionary. The existing native images yielded 8/120 DINO ten-way hits, 65/120 correct supplied-menu names and 55/120 correct full-dictionary names. Paired intervals draw entire objects among the twelve, using 10,000 bootstrap replicates. Promotion requires more ten-way hits than native, mean and medoid, a positive lower 95% bound relative to exchanged EEG and at least as many full-dictionary hits as native. The new participant runs depend on that gate.

## Sub-04 results

| Generated image outcome | Native | Mean | Medoid | Matched | Exchanged |
| --- | ---: | ---: | ---: | ---: | ---: |
| Viewed photo first among ten, DINO /120 | 8 | 17 | 8 | 7 | 5 |
| Correct object among 1,854 names, CLIP /120 | 55 | 22 | 30 | 34 | 35 |
| Correct object among 12 supplied names, CLIP /120 | 65 | 63 | 65 | 65 | 65 |
| Strict two-photo assignment, DINO /22 | 16 | 10 | 12 | 14 | 11 |
| RGB RMSE at 32 × 32 | 0.289 | 0.250 | 0.301 | 0.298 | 0.302 |
| Spatial luminance correlation | 0.066 | 0.115 | 0.063 | 0.080 | 0.048 |

Initializing from the matched photograph scored 7/120 on the primary ten-way test. This was 10 fewer hits than the pixel mean (−8.33 percentage points, paired 95% interval −14.17 to −2.50), one fewer than native and two more than the exchanged view (+1.67 points, 0.00 to +4.17). On 109/120 rows the matched reconstruction was closer in DINO space to its selected support than to the viewed original. The corresponding counts were 112 for the medoid and 114 for exchanged supports. The intervention failed the registered gate. [Sub-04 metrics and intervals](../research/eeg_stack_20261112/results/object_fidelity/support_view_sub04.json) · [Generated reconstructions for the first row of every object](../research/eeg_stack_20261112/gallery/support_view_sub04.jpg)

A separate full-resolution check after the main analysis found that initialization from the matched 32 × 32 support thumbnail reduced RMS luminance edge strength from 0.03551 to 0.00589. The viewed originals averaged 0.04141. The support-mean arm had edge strength 0.00402. All three quantities were computed from adjacent pixels in the reopened 512 × 512 images. [Post-render edge measurements](../research/eeg_stack_20261112/results/object_fidelity/lowband_guidance_edges.json)

## Replay

Prepare the personal12 trial/Fisher conditions and original THINGS row mapping first. The rendering transfer contains the four starting-thumbnail arrays, selected support IDs, original row IDs, matched H/14 directions and their radius. Reference originals are required when independently scoring the completed JPEGs. The fixed public sheet contains only generated images for the lowest archive row of each of the twelve objects. The companion private sheet places the viewed original beside those same generated images. [Personal panel preparation and references](PERSONAL12_CONFIRMATION.md) · [Trial-encoder comparison](TRIAL_OBJECT_PERSONAL12.md)

~~~sh
C=research/eeg_stack_20261112/code
python "$C/support_view.py" prepare --subject sub-04 --root "$RUN"
python "$C/support_view.py" render --subject sub-04 --root "$RUN" \
  --archive "$ARCHIVE" --adapter "$ADAPTER" --dictionary "$DICTIONARY"
python "$C/support_view.py" clip --subject sub-04 --root "$RUN" \
  --archive "$ARCHIVE" --dictionary "$DICTIONARY"
python "$C/score_support_view.py" --subject sub-04 --root "$RUN"
~~~
