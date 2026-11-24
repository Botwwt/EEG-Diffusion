# Reconstruction on twelve additional objects

The three previously studied people viewed the same 120 photographs. Each query averages four physical EEG presentations. The personal readout, H/14 photo residual, angular gain, prompt, diffusion weights, step count and row-specific noise are frozen at the settings from [the previous experiment](PHOTO_STRUCTURE.md). The model uses eight other photos and EEG recordings per object in each fold. The [panel registration](../research/eeg_stack_20261112/results/object_fidelity/fresh_panel_design.json) records the sampling frame, explicit exclusions, random seed and folds.

## Endpoints fixed before scoring

- **Photograph identity.** For each of the 120 query outputs per participant, compare the DINOv2-S/14 final CLS embedding of the generated JPEG with all ten original images of its true object. Report top-1 credit, its 10% chance level, and paired differences against the eight-support mean and the wrong EEG from the same predicted word and fold. The true object is supplied only to this evaluation. Also report the separately centered CLIP-L/14 ten-photo endpoint from the previous release.
- **Object identity.** Report accuracy of the 12-way EEG readout and independent naming credit for the saved JPEG against all 1,854 THINGS labels. Include all misdecoded queries.
- **Within-object correspondence.** Report matching of the two held-out photos in each object-fold where both predicted words are correct, the number of eligible pairs, and independent decisions for the two queries. These secondary endpoints use a smaller gallery. The original stimulus never enters the diffusion input.
- **Appearance.** Compare saved JPEGs with the archived crops using 32×32 RGB error and centered luminance correlation. Show the lowest archive row for each object regardless of outcome.
- **Uncertainty.** Report each person separately and preserve photo pairing in all arm comparisons. Resample whole object classes for intervals. The three people share the same photos, so their 360 responses represent 120 distinct stimuli.

## Results

| Participant | Correct word from EEG /120 | Correct generated-image name among 1,854 /120 | DINO ten-photo own / mean / wrong, each of 120 | DINO two-photo own / mean / wrong |
| --- | ---: | ---: | ---: | ---: |
| sub-04 | 85 | 53 | 14 / 17 / 11 | 20 / 16 / 16 of 34 |
| sub-05 | 60 | 39 | 14 / 17 / 17 | 6 / 6 / 6 of 15 |
| sub-01 | 64 | 43 | 16 / 17 / 15 | 13 / 10 / 8 of 23 |

The primary ten-photo score was 44/360, versus 51/360 with the mean and 43/360 with wrong EEG. The paired gain over the mean was −1.94 percentage points (95% interval −5.28 to +1.39), and over wrong EEG it was +0.28 points (−4.44 to +5.00). For correctly decoded words, the true-object ten-photo gallery contains the eight support photos and the two held-out targets. For incorrectly decoded words, the eight supports come from a different object. Ten percent describes guessing an index uniformly. The mean and wrong arms measure the behavior of the actual image prior and a matched EEG swap. Both candidates in each strict two-photo comparison were held out. [DINO and pixel reports](../research/eeg_stack_20261112/results/object_fidelity/fresh_photo_dino_pixels.json)

Across all 72 pairs with both EEG words correct, DINOv2 assigned 39 pairs correctly with own EEG, 32 with the mean, and 30 with wrong EEG. The own-minus-mean interval spans −10.2 to +26.7 percentage points. CLIP-L/14 gave 45, 36 and 32 of 72, respectively. On the full ten-photo gallery, centered CLIP gave 29, 32 and 38 hits of 360. The extra queries or pairs contributed by another participant still concern the same 120 photographs. [Independent CLIP results](../research/eeg_stack_20261112/results/object_fidelity/fresh_photo_clip.json)

CLIP-L/14 named 135/360 own-EEG JPEGs correctly among all 1,854 concepts, and 208/360 with the vocabulary restricted to twelve. On the actual 120 original photographs the same rule had 71 correct among 1,854 and 119 among twelve. In the full vocabulary it correctly named ten of the ten original payphones and none of the ten locks. The reference score helps interpret differences across sets of objects. [Reference-image naming by class](../research/eeg_stack_20261112/results/object_fidelity/fresh_reference_names.json)

The mean 32×32 RGB error decreased from 0.29423 with the eight-support mean to 0.28772 with the query EEG (paired difference −0.00651, 95% interval −0.01186 to −0.00115). Restricting to the 209 correct EEG words gave a difference of −0.00481 (−0.01079 to +0.00216). Mean luminance correlation across all queries was 0.0596 for own EEG and 0.0704 for the prior. The class- and participant-resampled difference includes zero. The [EEG report](../research/eeg_stack_20261112/results/object_fidelity/fresh_panel_eeg.json) includes a separate within-session-pair label permutation. The four-trial queries and supports can share recording sessions.

Two experimental image corrections use other photos for fitting. One transfers low-frequency spatial color and contrast. The other tests whether left–right differences recorded over paired posterior electrodes can help select the horizontal orientation of a generated image.

## Horizontal orientation

Can EEG choose between the generated image and its exact mirror? We fitted a separate ridge regression per person, using 40–280 ms activity in seven homologous posterior electrode pairs. Training removed each object's mean, leaving left–right differences in luminance, two color channels and local contrast across a 4×4 grid. Inference makes one binary decision using the predicted object's eight supports and the query EEG. Early visual responses to spatially localized stimuli motivated the hypothesis. Natural photographs were the empirical test. [Visual-evoked potential source study](https://pubmed.ncbi.nlm.nih.gov/11835601/) · [Decoder and evaluation](../research/eeg_stack_20261112/code/fit_photo_orientation.py)

On 5,120 generic photographs per participant, four folds holding out entire objects gave 53.1%, 54.5% and 58.4% accuracy for choosing a real calibration photograph's orientation against its own mirror. The class prior gave 51.2% in each person. On the 360 **fresh generated images**, the better orientation under the prespecified left–right feature distance was chosen 187 times by keeping the image as drawn, 173 times using the support prior, 176 times using the query EEG, and 181 times using the matched donor. Each participant's paired interval for own EEG minus prior included zero. The renderer keeps its original horizontal orientation. The color and contrast correction has its own held-out pixel and DINO results. [Orientation scores](../research/eeg_stack_20261112/results/object_fidelity/fresh_photo_orientation.json) · [Spatial correction](SPATIAL_PHOTO.md)

## Reproduce the saved-image comparison

Prepare the three frozen personal EEG packets using the [source and folding instructions](FRESH_PHOTO_REPLAY.md). Each generation job needs one GPU and the pinned offline Turbo archive, IP-Adapter and THINGS dictionary. Keep the two Python driver copies beside the outputs so the independent digest audit can identify the executed sources.

```sh
C=research/eeg_stack_20261112/code
F=outputs/fresh-photo
mkdir -p "$F"
cp "$C/render_within_photo_ip.py" "$C/score_within_photo_ip.py" "$F/"
export PYTHONPATH="$C${PYTHONPATH:+:$PYTHONPATH}"
for s in sub-04 sub-05 sub-01; do
  "$GPU" "$F/render_within_photo_ip.py" --root "$F" --subject "$s" --archive "$ARCHIVE" \
    --adapter "$ADAPTER" --dictionary "$DICTIONARY" --batch-size 8
done
"$GPU" "$F/score_within_photo_ip.py" --root "$F" --reference-dir "$F/references" \
  --archive "$ARCHIVE" --dictionary "$DICTIONARY" --subjects sub-04 sub-05 sub-01 --out "$F/clip.json"
"$PY" "$C/summarize_within_photo_ip.py" --root "$F" --references "$F/references" \
  --scores "$F/clip.json" --out "$F/clip_summary.json"
"$DINO_PY" "$C/score_fresh_photo.py" --panel-lock "$F/panel_lock.json" \
  --evaluations "$F/evaluations" --private-gallery --release "$F/dino_pixels.json"
"$GPU" "$C/score_reference_names.py" --archive "$ARCHIVE" --references "$F/references" \
  --dictionary "$DICTIONARY" --out "$F/reference_names.json"
for stage in train predict score; do "$PY" "$C/fit_photo_orientation.py" "$stage"; done
```

The DINO scorer verifies the training-image ZIP and manifest, both independent of the input H/14 vectors. Its fixed gallery takes the lowest row for each object. The optional private comparison places the actual stimulus beside the generated image. The public panels show generated pixels and labels. Every arm uses the same diffusion seed for the same query. A wrong-EEG donor comes from the same predicted word and fold. A group of one keeps its own donor, so the report also identifies cases with a distinct donor.
