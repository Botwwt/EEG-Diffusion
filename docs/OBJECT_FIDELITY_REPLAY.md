# Replaying the object-fidelity experiments

| Task | Participant and query | Recognition problem |
| --- | --- | --- |
| Registered R4 | Sub-03, four EEG repeats per photo, 120 queries per panel | Choose among 12 registered objects. Eight other photos per object supply calibration in each fold. |
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
  dest="$W/registered/$1"
  shift
  "$PY" "$C/fit_registered_objects.py" --archive "$A" --out "$dest" "$@"
  "$GPU" "$C/render_registered_objects.py" --archive "$A" --conditions "$dest/conditions.npz" \
    --dictionary "$BASE/dictionary.csv" --out "$dest/render"
}
registered familiar
registered replication --replication
```

Inspect `evaluation.json` for the EEG word decisions and `render/results.json` for object scores from the generated images. Each `render/images/` contains 120 JPEGs per arm. To check the photo-specific component, rerun the render command with `--swap-check` in the same output directory. The swap operates within fold and predicted word. Report the eligible number of photos. For the coarse-color experiment, run the following on the panel of interest.

```sh
dest="$W/registered/familiar"
color () {
  "$1" "$C/refine_registered_color.py" --archive "$A" --conditions "$dest/conditions.npz" \
    --images "$dest/render/images" --out "$dest/color" --dictionary "$BASE/dictionary.csv" --stage "$2"
}
color "$PY" fit
color "$PY" assess
color "$GPU" score-objects
```

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

### Original sources

- Li et al., *Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion*. [ATM author code](https://github.com/dongyangli-del/EEG_Image_decode/tree/49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d) · [Pinned authors' feature release](https://huggingface.co/datasets/LidongYang/EEG_Image_decode/tree/dbe34bb2407164f70883c661c76664f0d596e522)
- Gifford et al., *A large and rich EEG dataset for modeling human visual object recognition*. [THINGS-EEG2 paper](https://doi.org/10.1016/j.neuroimage.2022.119754)
- Stoinski, Perkuhn and Hebart, *THINGSplus*. The authors supply one additional copyright-free exemplar per concept. [THINGSplus paper](https://doi.org/10.3758/s13428-023-02110-8)
- Peebles and Xie, *Scalable Diffusion Models with Transformers*. The DiT README is a presentation reference. The generation commands above run SDXL-Turbo. [DiT README](https://github.com/facebookresearch/DiT#readme)
