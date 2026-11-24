# Replay the fresh 12-object EEG panel

The registration fixes 120 photographs, five folds and the same three participants studied on the earlier panels. It draws 12 of 624 eligible concepts using seed 260510. The historical D and V partitions, the personal decoder's 512 training classes and the other enumerated calibration and evaluation sets account for 1,030 excluded concepts after deduplication. The sampled concepts belonged to historical T, which earlier sub-03 models used for generic training and out-of-fold comparisons. [Frozen sampling frame and model settings](../research/eeg_stack_20261112/results/object_fidelity/fresh_panel_design.json) · [Prespecified image endpoints](FRESH_PHOTO_VALIDATION.md)

## Prepare the inputs

Run from the repository root using Python 3.11+ with NumPy and SciPy. Reuse the historical replay assets. The registration reads protocol metadata only. Its output is sufficient to determine every query and its eight permitted support rows per object, independently of the recordings. [Earlier personal model and angle-calibration replay](PHOTO_STRUCTURE.md#replay)

| Local directory | Required inputs |
|---|---|
| `outputs/official-object-readout/sub-03/` | Original `atm_train_rows.csv`, `evaluation/originals_manifest.csv` with official test identities and `assets/text_train.pt` with the published H/14 training-image embeddings |
| `outputs/object-fidelity/` | Both original `*_ablation.json` panel registrations and `r4_layout.json` |
| `outputs/object-transfer/` | The three participants' public release receipts, `sessions.json`, the historical sub-04 halves packet and the cached sub-04 physical ranges for the two old panels |
| `outputs/photo-structure/` | `expanded/features/` with 512 feature blocks and a receipt per participant, the historical sub-04 registered `expanded/.../evaluation.json`, `within_photo/selection.json`, `residual_angle/{scale,radius,design}.json`, and the historical sub-04 registered `conditions_512.npz` from both `within_photo` and `residual_angle/conditions` |

Set `ARCHIVE` to the root of the historical archive and `PY` to the Python executable. Copy its original split manifest without reserializing it. New physical ranges come from the pinned public EEG release and are cached by participant and byte range. A complete fresh run reads 36 ten-photo blocks, totaling 181,440,000 waveform bytes. [Pinned public release](https://huggingface.co/datasets/LidongYang/EEG_Image_decode/tree/dbe34bb2407164f70883c661c76664f0d596e522/Preprocessed_data_250Hz)

```sh
C=research/eeg_stack_20261112/code
W=outputs/fresh-photo
mkdir -p "$W/source"
cp "$ARCHIVE/data/recovery_20261108_v236/current_split_manifest.json" "$W/source/"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
"$PY" "$C/register_fresh_photo_panel.py"
"$PY" "$C/fit_fresh_photo_panel.py" all
```

The first command writes the exact excluded IDs, the per-fold rows and the panel lock. The `all` command runs `replay`, `fetch`, `infer` and `verify` in order. Each stage also runs individually. On the original environment, replay recovered all 240 historical sub-04 waveforms bit for bit. On the old registered panel, all 120 predicted words matched, and the maximum absolute difference was zero for the unrotated prediction and each of the three final H/14 conditions. [Registration implementation](../research/eeg_stack_20261112/code/register_fresh_photo_panel.py) · [Source checks and frozen inference](../research/eeg_stack_20261112/code/fit_fresh_photo_panel.py)

## Follow a query through the files

All paths below are relative to `outputs/fresh-photo/`. The row key is the original zero-based public training row. Source waveforms and feature bundles sort by this key. The render packet and evaluations follow the fold order in the registration.

| File | Data and identity check |
|---|---|
| `references/fresh_ablation.json` | The 120 rows, 12 words, true word indices, folds and 5 × 12 × 8 allowed support rows. The registration digest appears in each condition packet. |
| `eeg/SUB.npz` and `eeg/SUB.json` | The physical source for the decoder. The packet stores rows, two 2-trial posterior waveforms, 17 channel names, 250 sample times and session IDs. The receipt identifies the pinned release, byte ranges and per-block hashes. |
| `target/SUB.npz` | A derived copy for other readouts. `source_sha256` identifies `eeg/SUB.npz`. `mean_eeg` equals its float32 `half_eeg.mean(axis=0)`. `features` is the original 120 × 1,003 feature design. |
| `evaluations/SUB/fresh/word_predictions.json` | Frozen zero-based indices of predicted words in the registered vocabulary. `evaluation.json` joins the predictions to the true words after the generation packets have been saved. |
| `conditions/SUB/fresh/conditions_512.npz` | The three unit H/14 directions, predicted word, eight support rows and donor row for each query. `target_eeg_sha256` identifies `eeg/SUB.npz`, and `word_prediction_sha256` identifies the word predictions. |
| `design.json` and `radius.json` | The render contract records the three condition digests and the frozen generic image radius. Pass the fresh root and panel `fresh` to the renderer. |

`verify` recomputes all 361,080 compact feature values from the three source packets, checks the independent registration, condition, donor and word identities, then recounts each participant's classifications from the saved word indices. It calculates participant-specific 95% intervals by resampling whole concepts and checks a secondary null that permutes object labels only within their acquisition-session pairs. [Aggregate EEG results and file hashes](../research/eeg_stack_20261112/results/object_fidelity/fresh_panel_eeg.json)
