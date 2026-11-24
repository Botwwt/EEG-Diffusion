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

## Two candidate EEG object readouts

Both candidates used the original personal 96-direction Fisher space. Sub-04 selected each method on 384 of the 512 external calibration objects, using four object-disjoint folds and 12-way episodes with eight supports and two held-out queries per class. A further 128 generic objects checked the frozen choice. Each of their 1,280 photos was evaluated in nine fixed sets of competing objects, giving 11,520 decisions per participant. The three target panels had already been evaluated during development. Every zero-weight prediction on those panels reproduced the original cosine readout.

| Candidate | Sub-04 selection, out of 34,560 | Generic check, baseline → candidate, out of 11,520 each (sub-04 / sub-05 / sub-01) | Three target panels, out of 1,080 | Fresh panel alone, out of 360 |
| --- | ---: | --- | ---: | ---: |
| Blend a visual prediction of the class EEG prototype with its measured EEG prototype (10% visual) | 21,930 → 22,249 | 7,506 → 7,560 / 6,557 → 6,642 / 6,968 → 7,105 | 679 → 680 | 209 → 212 |
| Estimate shared noise from the 96 labeled EEG supports (25% sample covariance, 75% isotropic prior), then mix 25% Mahalanobis score with 75% cosine | 21,930 → 22,151 | 7,506 → 7,553 / 6,557 → 6,611 / 6,968 → 7,054 | 679 → 675 | 209 → 210 |

The modest generic gains yielded one additional correct word for the visual prototype and four fewer for the covariance readout across the 1,080 target queries. The selected object decoder keeps the original cosine rule. These tests scored decoded words. [Visual-to-EEG prototype results](../research/eeg_stack_20261112/results/object_fidelity/visual_eeg_prototypes.json) · [Support covariance results](../research/eeg_stack_20261112/results/object_fidelity/support_covariance.json)

Run each `generic` stage before its `evaluate` stage. The former locks the sub-04 choice before checking the additional 128 objects. The latter verifies that lock and baseline predictions before scoring the three participants. The covariance script defaults to the generic class split in the released visual-prototype JSON. Its split checksum normalizes CRLF to LF. Local output directories are `outputs/visual-eeg-prototypes/` and `outputs/support-covariance/`.

```sh
for stage in generic evaluate; do "$PY" "$C/fit_visual_eeg_prototypes.py" "$stage"; done
for stage in generic evaluate; do "$PY" "$C/fit_support_covariance.py" "$stage"; done
```

## Fourth panel and full-scalp confirmation

The fourth panel draws 12 of the 612 remaining eligible objects with seed 260513. All 120 photos are new to the target panels. The same three people return. The full-scalp candidate qualified at exactly the registered threshold on the three development panels, gaining 2, 1 and 8 words for sub-04, sub-05 and sub-01. On the fourth panel, the posterior and full-scalp word counts were 81 and 75, 69 and 68, and 82 and 85, respectively. Across 360 decisions the full-scalp model lost four correct words. The two word vectors for each person were frozen before image generation. [Prospective registration](../research/eeg_stack_20261112/results/object_fidelity/prospective_panel_design.json) · [Word counts, waveform and render-packet hashes](../research/eeg_stack_20261112/results/object_fidelity/prospective_render_inputs.json)

The default invocation recreates only the registration. The separate `replay` stage reproduces the historical fresh-panel vectors, words, supports and donors for all three participants. For the complete fourth-panel run, prepare the two 512-object readout inputs as in the [full-scalp experiment](../research/eeg_stack_20261112/code/prepare_full_head_512.py) and retain the nine earlier participant-by-panel word receipts in `outputs/full-head-512/predictions/`. Then run the following commands from the repository root with the same `PY` and `C` as above.

```sh
W=outputs/prospective-photo
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
"$PY" "$C/run_prospective_readout.py"
for stage in replay fetch classify conditions; do
  "$PY" "$C/run_prospective_readout.py" "$stage" || exit 1
done
for method in expanded_r4 full_head_512; do
  root="$W/render_inputs/$method"
  for sub in sub-04 sub-05 sub-01; do
    "$PY" "$C/render_within_photo_ip.py" --root "$root" --panels prospective --subject "$sub" \
      --archive "$ARCHIVE" --dictionary "$DICTIONARY" --adapter "$ADAPTER" || exit 1
  done
done
```

`fetch` checks the physical source ranges against the pinned release. `classify` reuses the two frozen object readouts. `conditions` maps each predicted word to that fold’s eight calibration photos, uses the frozen EEG residual and angular scale, and seals the three directions for the existing renderer. Both branches use seed `923761 + original row` for every arm. The two render roots contain separate word predictions, packet hashes, designs and generated JPEGs. The registered query photos enter the image metrics after the rendering settings have been saved. [Stage implementation](../research/eeg_stack_20261112/code/run_prospective_readout.py) · [Independent DINO and pixel scorer](../research/eeg_stack_20261112/code/score_fresh_photo.py)

The gallery uses the lowest stimulus row in every object class for all three participants. Run `"$PY" "$C/gallery_prospective_readouts.py" plan` to verify the fixed twelve rows and `"$PY" "$C/gallery_prospective_readouts.py" build` after both JPEG sets are in place. The public comparisons contain the two generated images and decoded words. The corresponding original-photo comparisons stay under `outputs/prospective-photo/gallery/`. [Frozen gallery rows](../research/eeg_stack_20261112/results/object_fidelity/prospective_gallery_plan.json)

The public sheets show all twelve object classes for [sub-04](../research/eeg_stack_20261112/gallery/prospective_sub04.jpg), [sub-05](../research/eeg_stack_20261112/gallery/prospective_sub05.jpg) and [sub-01](../research/eeg_stack_20261112/gallery/prospective_sub01.jpg). The release records the SHA-256 of each source JPEG and each finished sheet. [Gallery verification record](../research/eeg_stack_20261112/results/object_fidelity/prospective_gallery.json)
