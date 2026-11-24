# Which EEG differences between photographs survive a second recording?

For two photos of the same object, a difference that recurs when the participant views them in a different recording session is a candidate visual signal. We learned 64 directions that preserve such differences, then used those directions to predict deviations from the H/14 image class mean. The second session helps learn a filter on training photos. At test time the readout uses the same person's four physical presentations. Trial reproducibility has a substantial prior literature, including reliable spatial filters for steady-state visual stimuli. This experiment tests its application to **photo-specific visual conditioning in natural-image reconstruction**. [Reliable Components Analysis](https://pubmed.ncbi.nlm.nih.gov/25579449/) · [Task-related component analysis](https://pubmed.ncbi.nlm.nih.gov/28436836/)

## Feature test

The training split contained 416 generic object classes. The verification split contained 96 other generic classes, or 960 distinct photos. None of the 48 registered target objects appeared in either group. All three people saw the same images. For each verification object, two photos are held out and the other eight supply the common calibration examples. The readout predicts a visual direction for each held-out photo. The test asks whether the total cosine similarity is higher for the correct pairing or for exchanging the photographs. Each person contributes all 45 pairs per object. These overlapping pairs are correlated. The intervals below resample the 96 object identities jointly across the three people. [Registered design](../research/eeg_stack_20261112/results/object_fidelity/repeat_photo_subspace_design.json) · [Classwise scores and input receipts](../research/eeg_stack_20261112/results/object_fidelity/repeat_photo_subspace.json)

| Predicted visual condition | Correct assignments / 12,960 | Difference in favor of matched repeats, percentage points (95% interval) |
| --- | ---: | ---: |
| Selected 17-channel ridge readout | 9,371 (72.31%) | +1.53 (+1.03 to +2.03) |
| Posterior + full-scalp ridge using four-trial means | 9,214 (71.10%) | +2.74 (+2.01 to +3.46) |
| Posterior + same 64-direction procedure with late-session photo identities shifted within each training object | 9,272 (71.54%) | +2.29 (+1.72 to +2.89) |
| Posterior + 64 matched-repeat directions | **9,569 (73.83%)** | Reference |

Each training photo contributes two two-trial means from different recording sessions. Each mean supplies 63 electrodes × 14 temporal bins. We remove the training object mean and standardize the coordinates. A regularized generalized eigenproblem finds the directions with the largest early–late covariance relative to total variance. A ridge readout predicts the visual residual from those 64 features. It contributes half the residual at inference. The additional correct assignments were 93 for sub-01, 55 for sub-04 and 50 for sub-05. A related experiment used two other people's EEG as training supervision and used only the current person's EEG at inference. These are feature-space comparisons. The registered image experiment keeps the native word and angle fixed and evaluates actual JPEGs against the viewed photos. [Implementation](../research/eeg_stack_20261112/code/fit_repeat_photo_subspace.py) · [Cross-participant feature result](../research/eeg_stack_20261112/results/object_fidelity/shared_photo_eeg.json) · [Registered image comparison](../research/eeg_stack_20261112/results/object_fidelity/photo_direction_render_design.json)

After preparing the [full-scalp training blocks](FULL_HEAD_512.md) and the [H/14 visual data](FRESH_PHOTO_REPLAY.md), reproduce the feature audit with the scientific Python environment. The default reads the fixed generic split and checks the physical EEG feature receipts for all three participants.

```sh
python research/eeg_stack_20261112/code/fit_repeat_photo_subspace.py
```
