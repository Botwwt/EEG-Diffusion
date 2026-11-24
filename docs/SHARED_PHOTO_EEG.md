# Cross-participant supervision for photograph differences

When two photographs depict the same object, EEG responses from other participants can provide a second training target for the participant of interest. On 96 held-out generic objects, adding this supervision raised the fraction of photograph pairs assigned correctly in H14 space from 72.31% to 73.99%. The paired gain was 1.68 percentage points (95% object-bootstrap interval 1.12 to 2.24). The average cosine margin decreased, from 0.002584 to 0.001863. More assignments had the right sign while the average separation became smaller. [Registered design](../research/eeg_stack_20261112/results/object_fidelity/shared_photo_eeg_design.json) · [Verified results](../research/eeg_stack_20261112/results/object_fidelity/shared_photo_eeg.json)

| Calibration model | sub-01 / 4,320 | sub-04 / 4,320 | sub-05 / 4,320 | All pairs | Mean matched-minus-swapped cosine |
|---|---:|---:|---:|---:|---:|
| Direct EEG → H14 | 3,121 | 3,168 | 3,082 | 72.31% | 0.002584 |
| Cross-participant photograph supervision | 3,215 | 3,225 | 3,149 | 73.99% | 0.001863 |
| Same-object, wrong-photograph EEG teachers | 3,088 | 3,115 | 3,058 | 71.46% | 0.001490 |
| Equal-rank H14-only supervision | 3,142 | 3,173 | 3,073 | 72.44% | 0.002102 |

Against wrong-photograph EEG supervision, the gain was 2.53 points (95% interval 1.89 to 3.18). Against equal-rank H14-only supervision, it was 1.55 points (0.93 to 2.16). The signed difference in the mean cosine margin against direct EEG → H14 was −0.000721 (−0.000820 to −0.000627). All three participants improved on pair assignment. This met the registered criterion to evaluate a generated image. The measurement here uses two original H14 vectors as alternatives, a known object, eight support photographs and their measured EEG. It does not measure free image reconstruction, large-gallery retrieval, a new participant or independent image-encoder fidelity. [Paired comparisons](../research/eeg_stack_20261112/results/object_fidelity/shared_photo_eeg.json)

## Model and separation

All three participants contributed the same 512 generic concepts with ten original training photographs per concept. Each input is the person's four-trial average on the 17 posterior electrodes, represented by the established 1,003 features. We verified the original ten-row interval, physical block digest and feature digest per class and person. The four image-reconstruction panels together contain 48 other concepts. All 48 were excluded. Of the 227 generic concepts outside earlier H14 hyperparameter, angular-scale, image-radius and paired-UNet development/evaluation sets, seed 260526 selected 96 for evaluation. All coefficient fits used the other 416 objects. The held-out stimuli belong to the existing generic calibration pool and may have contributed to earlier models. These models were fit anew with whole-object separation for this comparison.

For each source participant, both teachers’ training responses were centered within object. Each teacher was scaled by their own training standard deviations and entered with equal weight. A reduced-rank ridge estimated the 64 directions in the source EEG that best predicted the other two people’s response to the same photograph. The input ridge penalty was 0.5 times the number of training rows. We scaled each resulting component to unit training variance and fit a ridge to the within-object H14 residuals. Both this ridge and the full-dimensional direct ridge used a penalty of twice the number of training rows. The final prediction averaged their outputs.

The shuffled control repeated the entire fit after a deterministic nonzero rotation of each teacher's ten EEG rows inside every training concept. The H14-only control selected 64 directions under visual supervision and applied the same ridge penalty and 50/50 mixture. All heads were written before held-out scoring. Inference for each participant read only that participant's query and support EEG. The other people's responses to that test photograph were unavailable to the predictor.

Each of the 45 distinct pairs per concept used the other eight photograph embeddings and that person's eight support EEGs. We independently predicted two unit H14 directions, compared both assignments, and credited a tie as 0.5. The 12,960 decisions are 4,320 distinct photograph pairs repeated for three people. The 95% intervals resampled 96 concepts 10,000 times, keeping the 45 overlapping pairs, three participants and four models together. This provides uncertainty across objects conditional on these participants.

## Local replay

From the repository root, prepare the [personal 512-class EEG feature stores](PHOTO_STRUCTURE.md#replay), [fresh exclusions](FRESH_PHOTO_REPLAY.md), published official training H14 features and row dictionary, and the public prospective-panel and paired-denoiser designs. Use Python 3.11+ with NumPy, SciPy and Pillow. The registration reads metadata. The second command reads the EEG and H14 calibration features, saves all fitted heads, then scores held-out objects. It writes private per-object margins, weight digests and the compact result to `outputs/shared-photo-eeg`.

```bash
C=research/eeg_stack_20261112/code
export OPENBLAS_NUM_THREADS=4
python "$C/fit_shared_photo_eeg.py" register
python "$C/fit_shared_photo_eeg.py" run
```

## Fourth-panel image experiment

After the generic gate passed, the same fixed model was refit on all 512 calibration classes. The fourth panel already had EEG object-word results when this follow-up was authorized. It is a development comparison on these 12 objects. Its conditions use only one participant's target EEG and preserve each row's posterior-only predicted word, eight supports, original IP-Adapter radius, seed and native conditioning angle.

For the matched condition we replaced the native tangent around the support mean with the new EEG-predicted tangent. For the wrong condition we transferred another query's tangent within the same predicted word and fold, using the recipient's original angular magnitude. A singleton receives its own direction. The support mean for the uniform arm was copied byte for byte. Both candidate arms reproduced the recipient's native conditioning angle within 0.000000007 radians on all 360 queries. The historical code and the refitted direct ridge each reproduced every original float32 native condition. [Application and packet receipts](../research/eeg_stack_20261112/results/object_fidelity/shared_photo_eeg_application.json)

Prepare the prospective panel's posterior words, recorded EEG and original directions before packaging. The `fit` stage records an application lock and writes the final generic heads. The `package` stage verifies their hashes and the native comparator, then writes `outputs/prospective-photo/refinements/shared/{design.json,radius.json,conditions}`. The renderer uses the explicit recipient-angle rule in the new design. The same-image seed and byte-identical uniform conditions make the old uniform images an exact reproducibility control. [Shared-response packager](../research/eeg_stack_20261112/code/package_shared_photo_eeg.py) · [Common recipient-angle validation](../research/eeg_stack_20261112/code/package_repeat_photo_subspace.py)

```bash
python "$C/package_shared_photo_eeg.py" fit
python "$C/package_shared_photo_eeg.py" package
```
