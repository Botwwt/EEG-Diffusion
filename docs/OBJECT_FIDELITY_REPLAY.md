# Replaying the object-fidelity experiments

| Task | Participant and query | Recognition problem |
| --- | --- | --- |
| Personal R4, primary | Seven additional participants, four repeats per query, 120 distinct photos in each of two panels | Learn from 128 other objects per participant plus eight calibration photos for each of twelve target objects. Compare the same generator and noise under frozen and personal EEG readouts. |
| Registered R4 | Sub-03, four EEG repeats per photo, 120 queries per panel | Choose among 12 registered objects. Eight other photos per object supply calibration in each fold. |
| Registered R4 transfer | Source sub-03 and new participants viewing the same 120 photos per panel | Fit the generic decoder on sub-03. Measure recognition for each new participant using the same eight labeled calibration photos per object and fold. |
| Unseen R4 | Sub-03, four repeats, 200 queries per panel | Decode previously unseen concepts from EEG. Distinguish the 20-class photo-prototype score, 10-way within-object identification and 1,854-word decoding. |
| Official R80 | Sub-03, released ATM embeddings of 80-repeat means, 200 queries | Predict from all 1,854 words or generate for one held-out photo per object. The 200-way original-photo gallery measures identification. |

Run the shell examples from the repository root. Set `PKU_ARCHIVE` to your copy of the historical PKU experiment archive. Set `PY` to a Python 3.11+ NumPy/SciPy/Pillow environment with PyTorch for fitted models. Set `GPU` to the archive-compatible CUDA Python with PyTorch, torchvision, Diffusers, Transformers and native PKU model dependencies. Use a fresh output directory per recipe.

```sh
C=research/eeg_stack_20261112/code
W=outputs/object-replay
BASE="$W/official/sub-03"
A="$PKU_ARCHIVE"
mkdir -p "$W"
"$PY" "$C/official_object_readout.py" --out "$BASE" --stage prepare
```

The last command downloads and verifies public ATM features and THINGS metadata. It creates `dictionary.csv`, `dictionary.npz`, `eeg_r80.npy` and `provenance.json`. Both R4 experiments reuse this dictionary. To replay the raw R4 models, obtain the PKU archive from its custodian. The public THINGS-EEG2 recordings and ATM embeddings do not supply its historical partitions, checkpoints, image registration and trained adapters. The required archive layout includes the following.

| Location relative to the archive root | Contents |
| --- | --- |
| `data/recovery_20261108_v236/`, `data/full_T_recovery_20261108_v238/` | Sub-03 mean-four waveforms, channel layout, T/D/V split and SHA-indexed original-image manifest |
| `data/native_training_v239/`, `data/native_T_latents_v256/` | Ordered training stimulus IDs, visual embeddings and registered T-only VAE modes |
| `runs/HIVE_native_sub03_5096542/`, `runs/HIVE_matching_ordinary_calibration_v264/` | Native sub-03 encoder, deterministic visual targets and ordinary control checkpoint |
| `assets/things_images_v254/training_images.zip`, `assets/native_generator_v243/`, `assets/evaluation_v272/`, `code/` | Original RGB stimuli, SDXL-Turbo and adapter assets, evaluator weights and receipts, v239/v253 native modules and v272 OpenAI CLIP evaluator source |

### 1. Registered R4

Each query photograph is held out once across five folds. The classifier may use other EEG and image exemplars of the same 12 objects. The second panel draws another 12 objects with seed 260426 and uses the frozen mixing weight 0.25.

```sh
registered () {
  dest="$W/panels/$1"
  shift
  "$PY" "$C/fit_registered_objects.py" --archive "$A" --out "$dest" "$@"
  "$GPU" "$C/render_registered_objects.py" --archive "$A" --conditions "$dest/conditions.npz" \
    --dictionary "$BASE/dictionary.csv" --out "$dest/render"
}
registered registered
registered replicate --replication
```

Inspect `evaluation.json` for the EEG word decisions and `render/results.json` for object scores from the generated images. Each `render/images/` contains 120 JPEGs per arm. To check the photo-specific component, rerun the render command with `--swap-check` in the same output directory. The swap operates within fold and predicted word. Report the eligible number of photos. For the coarse-color experiment, run the following on the panel of interest.

```sh
dest="$W/panels/registered"
color () {
  "$1" "$C/refine_registered_color.py" --archive "$A" --conditions "$dest/conditions.npz" \
    --images "$dest/render/images" --out "$dest/color" --dictionary "$BASE/dictionary.csv" --stage "$2"
}
color "$PY" fit
color "$PY" assess
color "$GPU" score-objects
```

#### Transfer to new participants

Each panel has its own source model. Generic sub-03 training uses 1,522 concepts (15,220 photos), excluding that panel's 12 registered objects and 120 development concepts. The two panels share a procedure and the frozen 75% projected-EEG / 25% visual-score mixture. The primary model includes ten of the second panel's concepts in generic training. The second model includes all twelve of the primary panel's concepts. Each source model therefore tests transfer on its own excluded concepts.

For each fold, the raw and projected EEG classifiers use the new participant's EEG for eight labeled calibration photos per object (96 photos). The fused classifier also uses H/14 embeddings of these calibration images. The visual-only classifier requires zero labeled calibration EEG from the new participant. It compares the source-trained EEG-to-H/14 prediction against the same twelve image prototypes. All four classifiers receive a closed vocabulary of registered objects. The generator also uses the new participant's calibration EEG to center its predicted continuous feature.

The authors applied prestimulus baseline correction, resampled to 250 Hz and saved the 250 poststimulus samples with a 300-element time vector. Use `source_times[50:]` for the time axis. Waveform indices 49 and 50 already represent 196 and 200 ms, and the last sample represents 996 ms. Convert the four physical trials to float32 before taking their arithmetic mean. The decoder selects 17 posterior channels by electrode name. The authors estimated session-specific multivariate noise normalization (MVNN) from each participant's official training partition, which includes the newly defined panel queries. Zero labeled target EEG for the visual branch describes the decoder stage on these released waveforms. A prospective first-recording experiment would also need an independently available preprocessing transform. [Author preprocessing](https://github.com/dongyangli-del/EEG_Image_decode/blob/49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d/EEG-preprocessing/preprocessing_utils.py#L74-L193) · [Author time-axis loader](https://github.com/dongyangli-del/EEG_Image_decode/blob/49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d/Retrieval/eegdatasets_leaveone.py#L156-L162)

The registered fits above assemble `$W/panels/{registered,replicate}/ablation.json` alongside each panel's `evaluation.json`. The ablation contains the source predictions for all four classifiers and all 5 × 12 × 8 calibration row IDs. For an older export missing this file, rerun the CPU fit in that panel directory with the same flags. The extractor reads 120.96 MB of waveform bytes per participant for the 240 selected photos. Its receipts record the pinned source revision, upstream full-file identifier, exact HTTP byte ranges and downloaded-block hashes. [Public EEG release](https://huggingface.co/datasets/LidongYang/EEG_Image_decode/tree/dbe34bb2407164f70883c661c76664f0d596e522/Preprocessed_data_250Hz)

If you already have the four flat PKU result exports, restore their panel layout with the following commands.

```sh
R=outputs/object-fidelity
for panel in registered replicate; do
  mkdir -p "$W/panels/$panel"
  cp "$R/${panel}_eeg.json" "$W/panels/$panel/evaluation.json"
  cp "$R/${panel}_ablation.json" "$W/panels/$panel/ablation.json"
done
```

Fetch the EEG packets, fit the source models and summarize participant-level results.

```sh
"$PY" "$C/fit_official_eeg_anchors.py" --base "$BASE" --export-only
"$PY" "$C/prepare_registered_subjects.py" --train-rows "$BASE/atm_train_rows.csv" \
  --registered "$W/panels/registered/evaluation.json" --replication "$W/panels/replicate/evaluation.json" \
  --out "$W/participant-eeg" --subjects 3 4 1 2 5 6 7 8 9 10
"$PY" "$C/prepare_registered_subjects.py" --train-rows "$BASE/atm_train_rows.csv" \
  --out "$W/participant-eeg" --subjects 1 2 3 4 5 6 7 8 9 10 --check-only "$W/participant-eeg/verification.json"
"$PY" "$C/fit_registered_transfer.py" --archive "$A" --subjects-dir "$W/participant-eeg" \
  --panel-dirs "$W/panels/registered" "$W/panels/replicate" --out "$W/participant-fit"
cp "$W/panels/registered/evaluation.json" "$W/panels/registered_eeg.json"
cp "$W/panels/replicate/evaluation.json" "$W/panels/replicate_eeg.json"
"$PY" "$C/summarize_registered_transfer.py" --runs "$W/participant-fit" --references "$W/panels" \
  --out "$W/participant-fit/summary.json"
```

Check `participant-fit/source_verification.json` for the comparison of extracted and native sub-03 waveforms, including exact equality and maximum absolute difference. The transfer code replays the four source predictions before evaluating another participant. Each panel's per-participant `evaluation.json` records predictions and row IDs. The summary excludes sub-03 from the transfer estimates and uses 10,000 crossed bootstrap samples of new participants and registered concepts, keeping a concept's ten photos together. Report 120 photos per person and panel alongside the number of new participants. The same photos repeat across participants. These scores measure 12-way EEG object recognition. Claims about the exact photograph require independent evaluation of generated images.

**Recording-session control.** Acquisition assigned all ten images of each object to the same two recording sessions within a participant. Thus a query and its same-object calibration examples share a session pair even though the photographs differ. The following evaluator reconstructs the photo-to-session mapping from the 40 archived presentation-order files and checks that every photo was assigned to exactly two sessions. Session identities enter after the classifier predictions are frozen. [Original stimulus-order code](https://github.com/gifale95/eeg_encoding/blob/101997d4f14c41b615bcaf8c0907740106d260c5/00_data_collection/train/defineStimuliOrder.m#L31-L50) · [Forty original session schedules](https://github.com/gifale95/eeg_encoding/tree/101997d4f14c41b615bcaf8c0907740106d260c5/00_data_collection/train/train_order)

```sh
"$PY" "$C/audit_registered_sessions.py" --runs "$W/participant-fit" --out "$W/session-audit"
"$PY" "$C/prepare_registered_subjects.py" --train-rows "$BASE/atm_train_rows.csv" \
  --registered "$W/panels/registered/evaluation.json" --replication "$W/panels/replicate/evaluation.json" \
  --out "$W/participant-eeg" --subjects 1 2 3 4 5 6 7 8 9 10 \
  --session-halves --sessions "$W/session-audit/sessions.json"
```

`sessions.json` lists the two sessions per concept and the source-file hashes. For each frozen classifier, `registered_session_audit.json` reports observed accuracy and the exact chance expectation obtained by permuting whole category identities within each session pair. If a classifier's predicted object belongs to the query's pair, chance credit is one divided by the number of registered objects in that pair. A prediction outside the pair receives zero. The session-only oracle reports the best expected performance available from the pair identity alone. The gap above conditional chance describes discrimination among objects after accounting for this grouping. Separating visual responses from other acquisition effects requires additional controls.

The second command reads the cached four trials and writes `participant-eeg/halves/sub-XX.npz`. The authors concatenated each photo's two trials from its earlier session and two from its later session before shuffling the four slots with seed 20200220. In the stored data, zero-based slots 1 and 3 come from the earlier session and slots 0 and 2 from the later session. Each packet records both two-trial means and the original session numbers for each photo in `session_ids`. The earlier session is 1 or 2 and the later session is 3 or 4, depending on participant and object. Use those absolute IDs when reporting separate-session calibration and query results. [Author trial merge and shuffle](https://github.com/dongyangli-del/EEG_Image_decode/blob/49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d/EEG-preprocessing/preprocessing_utils.py#L262-L299) · [Author shuffle seed](https://github.com/dongyangli-del/EEG_Image_decode/blob/49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d/EEG-preprocessing/preprocessing.py#L52)

**Source prototypes, cross-session matching and an ERP baseline.** Replay these CPU comparisons using the frozen transfer weights and the same five folds.

```sh
"$PY" "$C/score_source_prototypes.py" --archive "$A" --subjects-dir "$W/participant-eeg" \
  --panel-dirs "$W/panels/registered" "$W/panels/replicate" --transfer "$W/participant-fit" \
  --out "$W/source-prototypes"
"$PY" "$C/score_cross_session_objects.py" --archive "$A" --subjects-dir "$W/participant-eeg" \
  --panel-dirs "$W/panels/registered" "$W/panels/replicate" --transfer "$W/participant-fit" \
  --halves-dir "$W/participant-eeg/halves" --sessions "$W/session-audit/sessions.json" \
  --out "$W/cross-session"
cp "$W/panels/registered/ablation.json" "$W/panels/registered_ablation.json"
cp "$W/panels/replicate/ablation.json" "$W/panels/replicate_ablation.json"
"$PY" "$C/score_cross_session_erp.py" --halves "$W/participant-eeg/halves" \
  --references "$W/panels" --out "$W/erp"
for run in participant-fit source-prototypes cross-session erp; do
  "$PY" "$C/audit_registered_sessions.py" --runs "$W/$run" --out "$W/session-audit" \
    --report "$W/session-audit/$run.json"
done
```

| Comparison | What the target participant supplies | Interpretation |
| --- | --- | --- |
| Source EEG prototypes | A four-trial query. Sub-03 supplies the eight labeled calibration EEG exemplars per class. | Source and target share the same registered photo sets and visual reference bank. This measures transfer with zero labeled target EEG after the authors' preprocessing. |
| Cross-session projected EEG | Early and late two-trial averages for each query and each target calibration photo. | Match early queries to late support, reverse the direction, then average scores. Compare with matching early to early and late to late on the same four trials. In the fused variants, H/14 always uses the full four-trial query. All twelve classes remain candidates in every direction. |
| Cross-session ERP | The same target calibration and query halves, restricted to 17 electrodes and fifteen 20-ms bins from 60 to 360 ms (end exclusive). | The 255-feature classifier estimates a shrinkage discriminant using the fold's calibration data. Compare crossed, same-session, and mean-four predictions. This is a participant-calibrated EEG-only baseline with no external visual decoder. |

For single-direction outputs, both scorers name the query half first (`early_to_late` means early query, late support). The bidirectional metrics are `crossed_fused` for the frozen projected/visual model and `cross` for the ERP classifier. Each audit report retains the participant's frozen prediction-file SHA, conditional chance, the equal-weight average across occupied session pairs, and exact permutation probabilities. The new-participant summary reports a crossed participant-and-concept interval for the mean excess over conditional chance. Comparisons of image fidelity require separate rendering and scoring of the saved conditions.

#### Personal calibration on 128 additional objects

Each of sub-04 and sub-05 contributed **1,280 additional labeled calibration photos** with four EEG trials per photo. Both participants saw the same 128 objects. The extractor excludes the 24 registered objects, draws 128 of the remaining 1,630 with NumPy seed 260431, sorts the selected indices and keeps all ten photos of each. The five evaluation folds still use eight other labeled calibration photos for each of the 12 registered objects. These results therefore describe a personally calibrated decoder.

```sh
P="$W/personal"
cp "$W/session-audit/sessions.json" "$W/participant-eeg/sessions.json"
"$PY" "$C/prepare_generic_subjects.py" --root "$W/participant-eeg" --train-rows "$BASE/atm_train_rows.csv" \
  --registered "$W/panels/registered/evaluation.json" --replication "$W/panels/replicate/evaluation.json" \
  --subjects 4 5
for panel in registered replicate; do
  cp "$W/panels/$panel/ablation.json" "$W/panels/${panel}_ablation.json"
done
personal_fit () {
  "$PY" "$C/fit_personal_session_metric.py" --generic "$W/participant-eeg/generic" \
    --halves "$W/participant-eeg/halves" --references "$W/panels" --out "$P/fits" --subjects "$1"
}
personal_fit sub-04
```

Select the method by its combined number of correct 12-way EEG decisions on sub-04. The personal R4 projection scored **68 + 72 = 140/240** across the registered and replication panels. The cross-photo, cross-session projection scored **47 + 53 = 100/240**. Freeze that choice before running the sub-05 fit. Sub-05 fits its own projection from its own 1,280 additional photos.

```sh
method=personal_r4
printf '%s\n' "$method" > "$P/selected_method.txt"
personal_fit sub-05
```

On sub-05 the frozen choice scored **60/120** on the registered panel and **67/120** on the replication panel, for **127/240**. The cross-session alternative scored 22/120 and 45/120. Read the row-level decisions in `$P/fits/{registered,replicate}/sub-05/evaluation.json`. These counts measure EEG object recognition. To evaluate generated images, combine the selected decisions with each participant's existing R4 generator inputs.

```sh
"$PY" "$C/condition_registered_predictions.py" --archive "$A" --transfer "$W/participant-fit" \
  --subjects-dir "$W/participant-eeg" --panel-dirs "$W/panels/registered" "$W/panels/replicate" \
  --prediction-runs "$P/fits" --prediction-method "$(cat "$P/selected_method.txt")" \
  --subjects sub-04 sub-05 --out "$P/conditions"
for subject in sub-04 sub-05; do
  for panel in registered replicate; do
    "$GPU" "$C/render_registered_objects.py" --archive "$A" --dictionary "$BASE/dictionary.csv" \
      --conditions "$P/conditions/$panel/$subject/conditions.npz" --out "$P/render/$panel/$subject"
  done
done
```

When a predicted class changes, the conditioning script looks up its calibration prototype and recomputes the continuous condition for that class. It checks the original fold, waveform, model and starting-latent identities. Each render writes `results.json` and 120 JPEGs per arm under `images/{category,aligned,shuffled}`. Read `full_top1` for generated-image naming against all 1,854 words and `original_photo` for the original-photo gallery scores.

#### Locked confirmation on seven additional participants

The frozen comparison pairs the personal `personal_r4` readout with the sub-03 `fused` readout for sub-01, 02, 06, 07, 08, 09 and 10. Both use the same 12 registered objects, eight calibration photos per object per fold, generator and per-row seeds. The personal method also fits each participant's 1,280 photos from 128 other objects. The seven participants' frozen-source scores were already available when the design was fixed. Their personal predictions and renders were still unseen. Sub-03 supplied the source model, sub-04 selected the method and sub-05 provided an earlier check. [Frozen design and endpoint definitions](../research/eeg_stack_20261112/results/object_fidelity/personal_confirmation_design.json)

Run the following blocks in the same shell after the transfer, recording-session and sub-04/05 recipes above. Start with a fresh `Q`. Its `transfer` directory points to the fitted source models, evaluations and frozen generator conditions from the transfer recipe.

```sh
set -eu
Q="$W/personal-confirmation"
subjects="sub-01 sub-02 sub-06 sub-07 sub-08 sub-09 sub-10"
mkdir -p "$Q/reference-driver"
ln -s "$(cd "$W/participant-fit" && pwd)" "$Q/transfer"
cp "$W/session-audit/sessions.json" "$Q/sessions.json"
pin=e8c8870e331dc23b64a3cd4eb77788ed348930fa
git show "$pin:$C/render_registered_objects.py" > "$Q/reference-driver/render_registered_objects.py"
render_aligned () {
  "$GPU" "$C/render_registered_objects.py" --archive "$A" --dictionary "$BASE/dictionary.csv" \
    --aligned-only --batch-size 8 "$@"
}
```

First replay the four historical comparisons on the same GPU type and software stack. The formal A40 check matched all 480 JPEGs byte for byte, including both readouts on sub-04 registered and sub-05 replicate. The following loop uses prior three-arm renders under `OLD3/{frozen,personal}/PANEL/SUBJECT/`. With a fresh output root it rebuilds those references using the pinned driver. Each new `replay_check.json` compares all 120 aligned JPEG digests, per-photo naming credit, retrieval totals, condition and evaluator hashes.

```sh
OLD3="$Q/three-arm"
for pair in registered/sub-04 replicate/sub-05; do
  frozen="$Q/transfer/$pair/conditions.npz"
  personal="$P/conditions/$pair/conditions.npz"
  for arm in frozen personal; do
    ref="$OLD3/$arm/$pair"
    packet="$frozen"
    if [ "$arm" = personal ]; then packet="$personal"; fi
    if [ ! -f "$ref/results.json" ]; then
      PYTHONPATH="$C${PYTHONPATH:+:$PYTHONPATH}" "$GPU" "$Q/reference-driver/render_registered_objects.py" \
        --archive "$A" --dictionary "$BASE/dictionary.csv" --conditions "$packet" --out "$ref" --batch-size 8
    fi
  done
  render_aligned --case "$frozen" "$Q/parity/frozen/$pair" \
    --case "$personal" "$Q/parity/personal/$pair" \
    --verify-against "$OLD3/frozen/$pair" "$OLD3/personal/$pair"
done
```

Fetch the seven personal calibration sets from the same fixed draw and apply the selected readout. The fit also computes the original session control. The packaging command explicitly selects `personal_r4` for every participant and panel. For each predicted word it recomputes the continuous condition using that fold's calibration prototype and preserves the original R4 starting latent.

```sh
"$PY" "$C/prepare_generic_subjects.py" --root "$W/participant-eeg" --train-rows "$BASE/atm_train_rows.csv" \
  --registered "$W/panels/registered/evaluation.json" --replication "$W/panels/replicate/evaluation.json" \
  --subjects 1 2 6 7 8 9 10
"$PY" "$C/fit_personal_session_metric.py" --generic "$W/participant-eeg/generic" \
  --halves "$W/participant-eeg/halves" --references "$W/panels" --out "$Q/personal" --subjects $subjects
"$PY" "$C/condition_registered_predictions.py" --archive "$A" --transfer "$Q/transfer" \
  --subjects-dir "$W/participant-eeg" --panel-dirs "$W/panels/registered" "$W/panels/replicate" \
  --prediction-runs "$Q/personal" --prediction-method personal_r4 --subjects $subjects --out "$Q/personal_conditions"
```

One GPU process per participant loads the generator and independent OpenAI CLIP-L/14 once for all four cases. It renders one matched JPEG per query with four denoising steps and noise seed `923761 + array_row`. It opens the original photos for scoring after all four cases have generated their images. The loop runs participants sequentially.

```sh
for s in $subjects; do
  render_aligned \
    --case "$Q/transfer/registered/$s/conditions.npz" "$Q/confirm_render/frozen/registered/$s" \
    --case "$Q/personal_conditions/registered/$s/conditions.npz" "$Q/confirm_render/personal/registered/$s" \
    --case "$Q/transfer/replicate/$s/conditions.npz" "$Q/confirm_render/frozen/replicate/$s" \
    --case "$Q/personal_conditions/replicate/$s/conditions.npz" "$Q/confirm_render/personal/replicate/$s"
done
"$PY" "$C/summarize_personal_confirmation.py" --root "$Q" --out "$Q/personal_confirmation.json"
```

Each of the 28 case directories contains 120 files under `images/aligned/` and a `results.json` with per-query credit. The primary endpoint is top-1 naming credit among 1,854 concepts on the generator's RGB output before JPEG compression, with fractional credit for tied names. Each panel contributes 840 paired participant–photo queries from 120 distinct photos. The summary also reports 12-way EEG recognition, 12-way generated-image naming and identification among the ten original photos of the true object. Panel-specific 95% intervals use 10,000 paired bootstrap draws over participants and concepts, keeping each concept's ten photos together. The combined totals are descriptive. The supplementary EEG analysis compares each decoder with its recording-session-pair conditional expectation using the archived acquisition schedules. [Confirmation summary script](../research/eeg_stack_20261112/code/summarize_personal_confirmation.py)

### 2. Unseen R4

The fit uses 13,230 training photos in 1,323 concepts. Development uses 20 other concepts. Confirmation selects 20 disjoint concepts from historical V. Each concept supplies ten original photos. The archived EEG row indices in both layout bundles define every downstream comparison.

```sh
U="$W/unseen"
"$PY" "$C/prepare_mean4_probe.py" --archive "$A" --out "$U/prep"
"$PY" "$C/decode_eeg_layout.py" --archive "$A" --out "$U/layout"
"$PY" "$C/fit_photo_conditions.py" --archive "$A" --train-prep "$U/prep" \
  --panels "$U/layout" --out "$U/conditions"
"$PY" "$C/fit_object_words.py" --archive "$A" --train-prep "$U/prep" \
  --panels "$U/layout" --dictionary "$BASE/dictionary.npz" --out "$U/words"
unseen_render () {
  dest="$1"
  shift
  "$GPU" "$C/eval_visual_fidelity.py" --archive "$A" --layout "$U/layout" \
    --conditions "$U/conditions" --out "$U/$dest" "$@"
}
unseen_render standard
unseen_render word_render --words "$U/words"
unseen_render free_dev --panels dev --free-arms free free_stronger
unseen_render free_confirm --panels confirm --free-arms free_stronger free_stronger_shuffled
```

Check `layout_result.json`, `conditions_report.json` and `object_words_report.json` for training choices and per-panel metrics. Render directories contain the panel's `object_results.json`, per-arm score matrices and fixed examples. The 20-way object test averages similarities to nine other photos per candidate class after removing the query's ordinal from every class. A 10-way hit identifies the exact photo inside its true class. The image-oracle arm accesses held-out visual targets for diagnosis.

### 3. Official R80

**Public feature replay.** Run the stages below with the prepared dictionary. The prediction stage sees the EEG and full word dictionary. The evaluation stage indexes the original JPEGs against the official order by SHA. The published encoder was monitored on the official test set during training. Its exact export epoch is unverified.

```sh
"$PY" "$C/official_object_readout.py" --out "$BASE" --stage predict
"$PY" "$C/official_object_readout.py" --out "$BASE" --stage evaluate --originals
"$PY" "$C/fit_official_eeg_anchors.py" --base "$BASE"
"$PY" "$C/fit_official_visual_anchors.py" --base "$BASE" --out "$W/visual"
"$ATM_PY" "$C/reproduce_official_atm_prior.py" --base "$BASE" --out "$BASE/atm-prior" --evaluate
```

The first three word readouts write `predictions.csv` under `$BASE`, `$BASE/eeg-anchors` and `$W/visual/sub-03`. The two learned readouts select their settings using training concepts. A fresh CPU replay of the public text baseline reproduced the frozen prediction table, all true-label ranks and all scores. The complete ranking differed in 22 of 370,800 entries at positions 302 or lower. Use `ATM_PY` with PyTorch and `diffusers==0.30.0`. The prior script verifies the original code and checkpoint digests, checks its vectorized output against the authors' single-row implementation, and saves the 10-step/CFG 2 and 50-step/CFG 5 **feature vectors** before scoring. See `atm-prior/provenance.json` for tolerances and hashes.

**External reference images.** Set `H14` to a directory containing the pinned files below and `H14_PY` to a Python 3.11+ environment with PyTorch, Transformers, SciPy, Pillow and scikit-image. The reference run used PyTorch 2.9.1+cu128 and Transformers 4.49.0. The encoder compares one CC0 photo per dictionary word against the official JPEGs for duplicate exclusion. It then predicts from EEG by cosine similarity to the full eligible reference bank. Official ground-truth IDs enter scoring after predictions are saved.

- Download `images_THINGSplus-CC0.zip` from the [pinned THINGSplus CC0 mirror](https://huggingface.co/datasets/poojasingh11/THINGS-images/blob/fed41790a15a7a5d64c73564a74a1c9c6f7f720c/images_THINGSplus-CC0.zip).
- Download `model.safetensors`, `config.json` and `preprocessor_config.json` into the same directory from the [pinned LAION H/14 model](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/1c2b8495b28150b8a4922ee1c8edee224c284c0c).

```sh
E="$BASE/evaluation"
"$H14_PY" "$C/readout_external_images.py" --assets "$H14" --readout "$BASE" \
  --out "$W/external" --audit-released-image-space
```

Read `collision_audit.json`, `predictions.csv`, `scored.csv` and `summary.json` together. In the replay all 1,854 references passed 370,800 comparisons. EEG word retrieval gives 2/200 top-1 and 17/200 top-5. Exemplar SSIM measures retrieved pixels. The next step measures newly generated pixels.

**Single generated image per query.** This route requires the archive's offline Turbo bundle and its OpenAI CLIP ViT-L/14 evaluator. Each arm uses row-matched noise. Substitute another readout's frozen CSV to compare word decoders.

```sh
"$GPU" "$C/render_official_words.py" --archive "$A" --predictions "$W/external/predictions.csv" \
  --manifest "$E/originals_manifest.csv" --images "$E/test_images.zip" --dictionary "$BASE/dictionary.csv" \
  --out "$W/external-render"
official_score () {
  dest="$1"
  shift
  "$GPU" "$C/score_official_objects.py" --archive "$A" --dictionary "$BASE/dictionary.csv" \
    --images "$E/test_images.zip" --manifest "$E/originals_manifest.csv" \
    --score-photos --out "$dest" --render "$@"
}
official_score "$W/external-score" "$W/external-render"
```

The scorer counts generated-image object recognition against the 1,854-word dictionary, giving fractional credit for ties between identical readable names. It writes `summary.json`, `per_row.csv`, a blinded gallery and per-render `scores.json` for paired RGB, SSIM and CLIP photo scores. The external-word render obtained 2/200 generated object hits, 11/200 original-photo identifications and mean SSIM 0.17554. `--oracle` on the renderer adds a truth-word diagnostic in its own directory. The evaluator may inspect the official originals. The generator receives only its frozen predictions and noise in the EEG arms.

To generate directly from the public ATM prior's H/14 vector, set `ADAPTER` to the pinned base SDXL ViT-H adapter. This is a four-step Turbo adaptation of the released prior. It has its own raw-EEG, trained ridge, shuffled and mean-condition comparisons. [Pinned Tencent IP-Adapter weights](https://huggingface.co/h94/IP-Adapter/blob/018e402774aeeddd60609b4ecdb7e298259dc729/sdxl_models/ip-adapter_sdxl_vit-h.safetensors)

```sh
"$PY" "$C/render_official_ip_adapter.py" --prepare-base "$BASE" --conditions "$W/ip/conditions.npz"
"$GPU" "$C/render_official_ip_adapter.py" --archive "$A" --adapter "$ADAPTER" \
  --prior "$BASE/atm-prior/demo_10_cfg2/h14_prediction.npy" --conditions "$W/ip/conditions.npz" --out "$W/ip/render"
official_score "$W/ip/score" "$W/ip/render"
```

**Four-candidate EEG selection.** Generate two continuous-condition images and two images with the same condition plus a predicted word. This branch has a four-image budget per row, two seeds and two conditioning variants. It uses the visual-anchor readout's frozen words. The single-image word route above has its own inputs and compute budget.

```sh
K="$W/candidates4"
"$PY" "$C/fit_official_continuous.py" --archive "$A" --train "$BASE/atm_train_mean4.npy" \
  --train-names "$BASE/atm_train_stems.txt" --query "$BASE/eeg_r80.npy" --out "$K/conditions.npz"
draw () {
  "$GPU" "$C/render_official_continuous.py" --archive "$A" --conditions "$K/conditions.npz" \
    --words "$W/visual/sub-03/predictions.csv" --candidates-only --seed "$1" \
    --manifest "$E/originals_manifest.csv" --images "$E/test_images.zip" --out "$K/candidates/$2"
}
draw 531466 official_continuous
draw 531477 official_second_draw
mkdir -p "$K/continuous"
ln -s ../candidates/official_continuous/images/predicted "$K/continuous/predicted"
"$PY" "$C/fit_eeg_image_likelihood.py" --base "$BASE" --out "$BASE/image-likelihood"
"$H14_PY" "$C/rerank_official_draws.py" --source-root "$K" --readout "$BASE" \
  --assets "$H14" --out "$K/reranked"
official_score "$K/score" "$K/reranked"
```

The continuous fit checks the 16,540 training photo stems and uses the historical T rows for EEG-to-condition and 8×8 latent prediction. The candidate renderer accepts the evaluation paths for CLI compatibility and skips opening them in `--candidates-only` mode. The selector compares each row's four generated H/14 vectors with EEG, then records chosen source indices and JPEG hashes for the first draw, trained selector and controls. The final evaluator uses OpenAI L/14 and the original 200 JPEGs. Compare the selected images with the first draw and with selection by the wrong EEG row.

The release aggregator reads the named score exports listed in `generate()`, checks paired row identities where available, and records the SHA-256 of each input. With that export layout restored, run `"$PY" "$C/aggregate_object_fidelity.py" --inputs outputs --out "$W/summary.json"` to rebuild the cross-task table. The unrestricted public ATM stages above can be replayed independently of this full aggregate. [Aggregation source](../research/eeg_stack_20261112/code/aggregate_object_fidelity.py)

### Open-vocabulary EEG readout and photo-level comparisons

The R4 full-vocabulary EEG readout uses the same historical T photos and fixed dev/confirmation panels as the unregistered generator. It writes the unconditional 1,854-way prediction CSV before scoring against the 20 panel objects. Supply the optional THINGSplus bank to reproduce the separate external-reference analysis.

```sh
"$PY" "$C/fit_visual_to_neural_objects.py" --archive "$A" --dictionary "$BASE/dictionary.npz" \
  --external "$W/external/external_anchors.npz" --external-manifest "$W/external/external_manifest.csv" \
  --out "$W/open-neural"
```

For the personal render comparisons, point `OLD` and `NEW` at the matched frozen and personal render folders for the same participant and panel. The first command writes a generated-only gallery and a separate private original-versus-output comparison. The second scores every saved JPEG, with additional subsets for correct object predictions. Append `--controls` to the personal fit above and use a separate output directory to reproduce the post-hoc covariance comparisons.

```sh
"$PY" "$C/compare_personal_renders.py" --archive "$A" --baseline "$OLD" --personal "$NEW" --out "$W/gallery"
"$PY" "$C/score_personal_photo_layout.py" --archive "$A" --baseline "$OLD" --personal "$NEW" --out "$W/gallery/photo_layout.json"
```

### Within-object color residuals

For the within-object color experiment, `INPUTS` contains the mean EEG packets, `generic/` packets and the frozen `registered/ablation.json` and `replicate/ablation.json`. Set `PERSONAL` to the render root whose next component is the panel. Use a separate output root for the color branches. The CPU stage reads the subset of calibration photos allowed by each fold. Query photographs enter RGB scoring after JPEG generation has finished.

```sh
R="$W/photo_residual"
for s in sub-04 sub-05; do
  "$PY" "$C/refine_personal_photo_residual.py" fit --archive "$A" --inputs "$INPUTS" \
    --subject "$s" --model "$R/models/$s.npz"
  for p in registered replicate; do
    "$PY" "$C/refine_personal_photo_residual.py" render --archive "$A" --inputs "$INPUTS" \
      --subject "$s" --panel "$p" --model "$R/models/$s.npz" --render "$PERSONAL/$p/$s" --out "$R/$p/$s"
  done
done
"$GPU" "$C/score_official_objects.py" personal-residuals --archive "$A" --dictionary "$BASE/dictionary.csv" \
  --renders "$R" --out "$W/residual_naming" --device cuda
"$PY" "$C/summarize_personal_residual.py" --root "$W" --out "$W/personal_photo_residual.json"
```

### Registered R4 extensions and the additional sub-01 confirmation

The photo-structure replay starts from the 128-object personal readout, its paired JPEGs and its eight authorized supports per fold. It includes the original-photo hash check, support-guided generation, foreground warp and separate JPEG scoring. The calibration-size replay uses the same two 12-object panels and adds the 512-object posterior readout and the 128-object full-scalp readout. Sub-01 supplies an additional confirmation for the 512-object readout and the H/14 embedding experiment. The direct one-image H/14 route, independent DINOv2 scoring and separate eight-candidate selection budget each have execution commands. [Commands, inputs and output layout](PHOTO_STRUCTURE.md#replay)

### References

- Li et al., *Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion*. [ATM author code](https://github.com/dongyangli-del/EEG_Image_decode/tree/49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d) · [Pinned authors' feature release](https://huggingface.co/datasets/LidongYang/EEG_Image_decode/tree/dbe34bb2407164f70883c661c76664f0d596e522)
- Gifford et al., *A large and rich EEG dataset for modeling human visual object recognition*. [THINGS-EEG2 paper](https://doi.org/10.1016/j.neuroimage.2022.119754)
- Stoinski, Perkuhn and Hebart, *THINGSplus*. The authors supply one additional copyright-free exemplar per concept. [THINGSplus paper](https://doi.org/10.3758/s13428-023-02110-8)
- Peebles and Xie, *Scalable Diffusion Models with Transformers*. The DiT README is a presentation reference. The generation commands above run SDXL-Turbo. [DiT README](https://github.com/facebookresearch/DiT#readme)
