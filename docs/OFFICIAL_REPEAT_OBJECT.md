# Object decoding from four to eighty EEG presentations

Sub-04 named 6 of 200 official test objects from 1,854 names using eighty EEG presentations per photograph. Four presentations yielded zero exact names. The paired difference was +3.0 percentage points (95% stimulus-bootstrap interval +1.0 to +5.5). The decoder placed 19 correct names in its top five at eighty presentations and six at four. This experiment used one person, one photograph per official object and one fixed repeat permutation. [Aggregate results and source hashes](../research/eeg_stack_20261112/results/object_fidelity/official_repeat_object_evidence.json)

## Repeat budget and candidate gallery

Each entry counts correct responses among the same 200 photographs. The decoder ranked all 1,854 names before scoring. The 200-name column restricts the resulting scores to the true object vocabulary. The 200-photo column supplies the actual original-image features to the scorer and measures nearest-image retrieval in the learned canonical space. Within-object photographic fidelity needs multiple originals per object. The companion image study evaluates object naming, DINO retrieval and pixel agreement on generated JPEGs. [Image study and replay](OFFICIAL_REPEAT_IMAGES.md)

| EEG input | Full 1,854 names, top 1 | Full 1,854 names, top 5 | 200 names, top 1 | 200 originals, top 1 | Wrong EEG, 200 originals |
|---|---:|---:|---:|---:|---:|
| 4 presentations | 0 | 6 | 13 | 22 | 2 |
| 8 presentations | 1 | 12 | 20 | 37 | 2 |
| 16 presentations | 4 | 15 | 20 | 40 | 0 |
| 32 presentations | 5 | 16 | 21 | 55 | 0 |
| 80 presentations | 6 | 19 | 27 | 65 | 0 |
| Released sub-04 ATM, 80 presentations | 2 | 8 | 16 | 65 | — |

The full-dictionary wrong-EEG top-one count was zero at every budget. The R80 gain relative to the released ATM was +2.0 percentage points for exact naming (95% interval −0.5 to +5.0) and +5.5 for top-five naming (+1.0 to +10.0). Both identified 65/200 original images, with 28 gains and 28 losses on individual objects. R80 gained 43/200 image identifications over R4 (+21.5 points, +14.5 to +28.5). At R80 the smaller 512-object training set gave 7 exact names, 18 top-five names and 43 correct original images. All intervals resample the 200 paired object/stimulus identities, conditional on the participant, gallery and repeat order. [Machine-readable counts and paired intervals](../research/eeg_stack_20261112/results/object_fidelity/official_repeat_object_evidence.json)

The EEG feature extractor averaged disjoint blocks of four physical presentations. Averaging their features preserved the training scale of the band-power measurements. The five budgets used nested subsets from one permutation of the eighty physical repeats. The classifier learned correspondences between posterior EEG features and per-photograph H/14 image embeddings from 1,012 generic training categories. A separate 128-category development split used 884 categories for fitting and 1,280 photographs for ranking the candidate models. The selected model averaged standardized similarities from two references per vocabulary entry, a text-derived visual prototype and an external THINGSplus photograph, in 32 canonical dimensions. It scored 13/1,280 exact generic names, compared with 4 for direct ridge. The external reference audit checked all 1,854 × 200 comparisons against the official images. [Readout, split and controls](../research/eeg_stack_20261112/code/fit_official_repeat_objects.py)

## Which directions drive generation?

The IP-Adapter receives a unit vector in the original 1,024-dimensional H/14 image coordinates. A companion ridge maps the same EEG features to that vector. At R80 its average cosine to the correct original was 0.6504, compared with 0.6093 to a fixed mismatched original. Direct cosine with the 200 native H/14 embeddings selected four correct originals. The canonical decoder in the table above evaluates a different representation. On the 1,280 generic development photographs, the native direction had a median cosine of 0.9624 to the training image mean.

A separate experiment split those 128 generic categories into 64 for fitting angular corrections and 64 for evaluation. Each had ten distinct photographs. Every EEG query competed against the ten photographs of its own object, with one correct target per query. The angular validation classes were held out of the correction fit. These same classes had contributed to choosing the original semantic decoder rank.

| Condition | Correct within ten (640 queries) | Fraction beating a same-object alternative | Mean cosine to own original | Mean cosine to next photograph in class |
|---|---:|---:|---:|---:|
| Native ridge | 83 | 0.5431 | 0.6313 | 0.6232 |
| Rank-dependent correction | 88 | 0.5568 | 0.6296 | 0.6185 |
| Single global gain | 127 | 0.6332 | 0.4889 | 0.4661 |

The rank-dependent correction failed its registered requirement to exceed the global comparator. The global gain produced 44 additional within-ten identifications over native (+6.88 percentage points, 95% paired class-bootstrap interval +3.75 to +10.31). Its pairwise wins increased by 9.01 points (+7.34 to +10.71), while average absolute target cosine fell by 0.1424. This ranking result motivated the global generation follow-up. The 640 generic validation observations informed that choice. [Original angular comparison](../research/eeg_stack_20261112/code/calibrate_official_repeat_angles.py) · [Follow-up registration](../research/eeg_stack_20261112/results/object_fidelity/official_repeat_global_angle_design.json)

The global packet keeps each native direction's orientation around the unit training-image mean. It changes the angle from θ to min(3.2474376826288287 × θ, 1.061957199384292 radians). The gain is the ratio of the median true and median predicted angles on the 64 calibration objects. The cap is the 95th percentile of their true-image angles. Both packets use the same 200 official EEG rows, predicted words, donor assignment and image-embedding radius of 21.945598602294922. The receipt records the input and output direction hashes for R4 and R80 plus the unit training center. [Global packet receipt](../research/eeg_stack_20261112/results/object_fidelity/official_repeat_global_angle_packet.json)

## Replay

Run these commands from the repository root with the research Python environment. The physical extractor obtains and validates the revision-pinned public sub-04 test file (2.016 GB). Training uses the existing sub-04 512-object and extension feature caches, the official training row manifest and H/14 embeddings, the full text dictionary, and the external anchors with their collision audit. Evaluation additionally uses the official test originals, their H/14 features and the released sub-04 ATM embeddings. The original packet uses the saved generic H/14 radius. The tools expose their source paths through `--help`. [512-object feature preparation](../research/eeg_stack_20261112/code/prepare_expanded_personal.py) · [Extension preparation](../research/eeg_stack_20261112/code/prepare_double_calibration.py) · [Official asset preparation](../research/eeg_stack_20261112/code/official_object_readout.py)

```powershell
$env:OPENBLAS_NUM_THREADS = '5'
$env:OMP_NUM_THREADS = '5'
$code = 'research/eeg_stack_20261112/code'
python "$code/fit_official_repeat_objects.py" register
python "$code/prepare_official_repeat_eeg.py"
python "$code/fit_official_repeat_objects.py" predict
python "$code/fit_official_repeat_objects.py" package
python "$code/fit_official_repeat_objects.py" evaluate
python "$code/calibrate_official_repeat_angles.py" register
python "$code/calibrate_official_repeat_angles.py" validate
python "$code/export_official_global_angle.py" register
python "$code/export_official_global_angle.py" export
python "$code/summarize_official_repeat_objects.py"
```

The default output root is `outputs/official-repeat-object/`. The official per-row scores are in `scored_queries.csv`. The renderer accepts `sub-04/render_packet.npz` or `sub-04/global_angle_packet.npz`. Each packet contains `rows`, `r4`, `r80`, `r4_words`, `r80_words`, `donor` and `radius`. The summary replays the generic photo comparison and independently checks all 400 exported directions from the original matrix and center. The registered follow-up binds to the exact original fit bytes. Different BLAS builds can change floating-point output hashes. [Packet export](../research/eeg_stack_20261112/code/export_official_global_angle.py) · [Summary and independent formula check](../research/eeg_stack_20261112/code/summarize_official_repeat_objects.py)
