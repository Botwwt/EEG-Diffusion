# Predicting four local image features from EEG

We tested whether four presentations of a photograph to sub-04 distinguished its quadrants well enough to condition diffusion separately in each region. We cropped four overlapping squares from each stimulus, embedded each crop with the pinned LAION ViT-H/14 and subtracted the mean of the four embeddings. This removes appearance shared by the four regions of that photo. For a pair of held-out photos, eight other photos of the same object supplied the typical regional features. The EEG readout tried to predict what changed relative to those eight.

The model trained on ten photos apiece from 128 objects, using the same 1,003 posterior EEG features and ridge penalty as the global visual readout. On 320 photographs of 32 other objects, the full-amplitude EEG update increased average squared regional error from 0.07736 with the eight-photo prior to 0.08190. Exchanging EEG between the two held-out photos increased error further, to 0.08215. The update showed some sensitivity to the input, but its magnitude made the field less accurate. [Initial protocol](../research/eeg_stack_20261112/results/object_fidelity/regional_h14_design.json) · [Initial results](../research/eeg_stack_20261112/results/object_fidelity/regional_h14.json)

In a follow-up, four object-disjoint folds of the 128 training objects estimated one common gain of **0.03432**. We froze it and evaluated all 640 photographs from the 64 remaining objects of the historical 96-object split. These objects were new to this regional experiment and had figured in earlier experiments with other targets. Matched EEG scored 170/320 on two-photo assignments, exchanged EEG 150/320 and the mean 160/320 with half credit for ties. The calibrated field gave squared error 0.0838805 versus 0.0838840 for the support mean. The estimated reduction was 0.00000352 (95% object-bootstrap interval −0.00000109 to +0.00000812). It beat exchanging the four update vectors between diagonally opposite quadrants, but the comparison to the mean missed the registered threshold for running masked diffusion on personal12. Both sets of numbers measure predicted features. [Calibration protocol](../research/eeg_stack_20261112/results/object_fidelity/regional_h14_calibration_design.json) · [Independent regional evaluation](../research/eeg_stack_20261112/results/object_fidelity/regional_h14_calibration.json)

## Reproduce the regional study

Use the verified original THINGS training-photo ZIP and row manifest, the posterior EEG features for sub-04, the 512-object split and the matching H/14 weights. The extractor stores one file per object, verifies the model and stimulus archive, and checks the cached vectors and their hashes at scoring. The shared EEG loader also requires the official full-image H/14 tensor at `outputs/official-object-readout/sub-03/assets/text_train.pt`. It reads the 512-object full-image vectors alongside the EEG features, then discards those vectors. Only regional crop targets from the 128 training objects enter gain fitting. A positive gain preserves pair-assignment rankings. Its separate test measures the error amplitude. [Original photo archive and manifest](SPATIAL_PHOTO.md) · [Posterior EEG calibration](PHOTO_STRUCTURE.md) · [Generic object split](SINGLE_PERSON_OBJECT_ENCODERS.md)

```sh
C=research/eeg_stack_20261112/code
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python "$C/fit_regional_h14.py" extract
python "$C/fit_regional_h14.py" generic
python "$C/calibrate_regional_h14.py" fit
python "$C/calibrate_regional_h14.py" extract
python "$C/calibrate_regional_h14.py" evaluate
```
