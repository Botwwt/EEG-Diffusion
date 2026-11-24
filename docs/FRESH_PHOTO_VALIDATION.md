# Reconstruction on twelve additional objects

The three previously studied people viewed the same 120 photographs. Each query averages four physical EEG presentations. The personal readout, H/14 photo residual, angular gain, prompt, diffusion weights, step count and row-specific noise are frozen at the settings from [the previous experiment](PHOTO_STRUCTURE.md). The model uses eight other photos and EEG recordings per object in each fold. The [panel registration](../research/eeg_stack_20261112/results/object_fidelity/fresh_panel_design.json) records the sampling frame, explicit exclusions, random seed and folds.

## Endpoints fixed before scoring

- **Photograph identity.** For each of the 120 query outputs per participant, compare the DINOv2-S/14 final CLS embedding of the generated JPEG with all ten original images of its true object. Report top-1 credit, its 10% chance level, and paired differences against the eight-support mean and the wrong EEG from the same predicted word and fold. The true object is supplied only to this evaluation. Also report the separately centered CLIP-L/14 ten-photo endpoint from the previous release.
- **Object identity.** Report accuracy of the 12-way EEG readout and independent naming credit for the saved JPEG against all 1,854 THINGS labels. Include all misdecoded queries.
- **Within-object correspondence.** Report matching of the two held-out photos in each object-fold where both predicted words are correct, the number of eligible pairs, and independent decisions for the two queries. These secondary endpoints use a smaller gallery. The original stimulus never enters the diffusion input.
- **Appearance.** Compare saved JPEGs with the archived crops using 32×32 RGB error and centered luminance correlation. Show the lowest archive row for each object regardless of outcome.
- **Uncertainty.** Report each person separately and preserve photo pairing in all arm comparisons. Resample whole object classes for intervals. The three people share the same photos, so their 360 responses represent 120 distinct stimuli.

Two experimental image corrections are being developed using other photos. One transfers low-frequency spatial color and contrast. The other tests whether left–right differences recorded over paired posterior electrodes can help select the horizontal orientation of a generated image. Any fitting or parameter choice uses the generic calibration objects. Their outcomes are reported separately from the fixed primary model.
