# Selecting generated photographs with an image-to-EEG encoder

An encoder fitted to each participant's four-repeat EEG selected among eight already generated photographs of their predicted object. DINOv2 identified the viewed photograph among ten originals of the same object on **56 of 720 queries**. Exchanging EEG between two queries in the same predicted object and fold gave 63 correct. The previous H14 residual selector gave 60. The registered gate stopped the method on these two development panels. [Protocol locked before the new selections](../research/eeg_stack_20261112/results/object_fidelity/eeg_encoding_bank_design.json) · [Independent DINO results](../research/eeg_stack_20261112/results/object_fidelity/eeg_encoding_bank.json)

| Selector | All queries / 720 | Predicted word correct / 470 | Distinct donor with same true object / 310 |
|---|---:|---:|---:|
| Forward image → EEG encoder | 56 | 26 | 25 |
| Same encoder with exchanged EEG | 63 | 32 | 28 |
| Same encoder with mean support EEG | 60 | 32 | 25 |
| Previously frozen H14 residual | 60 | 31 | 27 |
| Previously frozen H14 support mean | 57 | 32 | 24 |
| Fixed first candidate | 63 | 31 | 27 |

Compared with exchanged EEG, the encoding selector changed top-1 accuracy by −0.97 percentage points (95% paired interval −2.78 to +0.69). The difference was −0.56 points against mean support EEG (−2.50 to +1.39) and −0.56 against the frozen H14 residual (−2.36 to +1.11). Each comparison used the same eight JPEGs, word and fold. The new selection agreed with the old H14 residual on 48.47% of queries. The two panels share 240 distinct target photographs across three participants. Intervals resampled all 24 object identities with their queries and fixed participants together. These results do not establish an improvement in within-object image selection. [Per-participant counts and secondary cosine endpoints](../research/eeg_stack_20261112/results/object_fidelity/eeg_encoding_bank.json)

## Encoder and controls

For each participant we fitted a ridge from within-object H14 image differences to within-object EEG differences on 5,120 photographs of 512 other calibration objects. The EEG target contained 1,003 features from the existing four-repeat average on 17 posterior electrodes. We standardized the H14 and EEG features using their respective calibration residual standard deviations. The image-to-EEG ridge penalty was twice the number of calibration rows. For the prediction errors on the calibration data we formed a covariance matrix from equal parts the empirical covariance and its diagonal.

For a new query, each candidate's unit H14 vector was centered on the average of the eight allowed unit H14 supports. The encoder predicted the corresponding EEG difference. We compared that prediction to the difference between the current participant's recorded EEG and their mean EEG on those same eight supports. The candidate with the smallest squared error under the fixed inverse covariance was selected. The mean-EEG control sets the observed difference to zero. The exchanged control uses another registered query's EEG in the same word and fold. The previous selectors and fixed candidate zero were retained as separate controls. [Selection and evaluation script](../research/eeg_stack_20261112/code/score_eeg_encoding_bank.py)

The bank held 960 unique JPEGs for 120 object/fold cells across the two previously studied panels. Every JPEG hash and its frozen candidate H14 feature cache were verified. The new indices for all 720 participant queries were saved before reading the candidate-level DINO scores or original-image DINO features. All six methods used the same pinned DINO cache. Replaying the three historical comparator arms reproduced every participant/panel's previous image choices and DINO top-1 counts. [Original candidate-bank experiment](../research/eeg_stack_20261112/results/object_fidelity/generated_bank_dino.json)

## CPU replay

Prepare the original candidate bank, its candidate H14 and DINO caches, the three participants' posterior R4 generic features, and the two panels' frozen condition packets as in the [photograph reconstruction replay](PHOTO_STRUCTURE.md#replay). From the repository root run

```bash
C=research/eeg_stack_20261112/code
export OPENBLAS_NUM_THREADS=4
python "$C/score_eeg_encoding_bank.py" register
python "$C/score_eeg_encoding_bank.py" select
python "$C/score_eeg_encoding_bank.py" score
```

The new working directory is `outputs/eeg-encoding-bank`. Each `choices.npz` gives the row IDs, eight verified candidate JPEG hashes and each selector's index. The preceding bank supplies the corresponding original generated JPEGs. No new diffusion run is required.
