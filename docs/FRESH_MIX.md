# More calibration objects improved twelve-way EEG recognition on new objects

For sub-04, expanding both EEG classifiers from 512 to 1,024 training objects and fitting one score weight raised recognition from **159/350 (45.4%) to 176/350 (50.3%)**. The new classifier corrected 30 decisions and spoiled 13. The paired difference was **+4.86 percentage points**, with a 95% interval of **+1.14 to +8.57**. Both models used the same four physical presentations per photograph, twelve possible objects and eight support photographs per candidate. This comparison measures the word chosen from EEG. [Prospective design and object lists](../research/eeg_stack_20261112/results/object_fidelity/fresh_mix_design.json) · [Decision counts and paired intervals](../research/eeg_stack_20261112/results/object_fidelity/fresh_mix.json)

| Sub-04 recording sessions | Strict objects | Photographs | Original model correct | New model correct |
|---|---:|---:|---:|---:|
| 1 and 3 | 11 | 110 | 59 | 69 |
| 1 and 4 | 8 | 80 | 39 | 40 |
| 2 and 3 | 5 | 50 | 22 | 26 |
| 2 and 4 | 11 | 110 | 39 | 41 |
| **Combined** | **35** | **350** | **159** | **176** |

After excluding the full 1,024-class fit, previous target menus, historical development and validation objects, and the protected fifth panel, the conservative pool contained 35 classes. No session pair contained twelve. Each menu therefore included all its strict classes and enough prespecified additional classes to reach twelve. The 13 additional objects came from a historical sub-03 RGB-gain calibration. The classifier always ranked all twelve candidates. Including the supplemental objects as queries gave **230/480 versus 246/480**, a difference of **+3.33 points** [0.21, 6.67]. The primary estimate weights each of the 350 photographs equally. Its bootstrap resamples whole objects within each of the four fixed menus. These intervals describe one participant.

## What the training comparison identifies

The original model averaged standardized scores from a 512-object trial encoder and a 512-object Fisher projection equally. The new model fit both components on 1,024 objects. Four class-disjoint training folds provided 9,280 same-session calibration decisions from 928 objects. The earlier scale study’s 96 development objects could contribute to estimating the component models and were excluded from all queries, support sets and labels used to fit the mixture weight. The prespecified regularized likelihood assigned **0.59078** to the neural scores and **0.40922** to Fisher before anyone scored the new menus. [Earlier scale study](SCALE_TRIAL_OBJECT.md) · [Weight fitting](../research/eeg_stack_20261112/code/fit_fresh_mix.py)

On the 350 strict photographs, the expanded equal-weight mixture recognized 173, the expanded neural component alone recognized 182, and the expanded Fisher component alone recognized 154. The learned mixture’s difference from equal weighting was +0.86 points [−0.29, 2.00]. Its difference from neural alone was −1.71 points [−6.00, 2.57]. Exchanging each query’s four EEG presentations with another menu candidate’s photograph in the same fold reduced the learned model to 17/350. All 480 predictions and the visual conditions retain the preselected weight.

## Reproduction

Run from the repository root after preparing the original 512-object trial encoder, the verified [512 additional trial blocks](SCALE_TRIAL_OBJECT.md) and the archived posterior and H/14 features. The commands use the published EEG-release revision. The acquisition verifies two independent reads of each of the 48 raw object ranges and saves the full trial data plus the compact model inputs locally.

```sh
C=research/eeg_stack_20261112/code
python "$C/register_fresh_mix.py"
python "$C/prepare_fresh_mix.py" --workers 6
python "$C/fit_fresh_mix.py" all
python "$C/score_fresh_mix.py"
python "$C/package_fresh_mix.py"
```

Eight input packets cover two classifiers and four menus. Both use the original 512-object H/14 photo predictor, angular gain and visual radius. Each predicted word selects its eight allowed support photographs. On the 365/480 rows where the classifiers chose the same word, the support identities and the own and uniform H/14 vectors matched exactly. The wrong condition takes a donor from the resulting same-word, same-fold group. [EEG acquisition](../research/eeg_stack_20261112/code/prepare_fresh_mix.py) · [Scoring](../research/eeg_stack_20261112/code/score_fresh_mix.py) · [Blinded rendering inputs](../research/eeg_stack_20261112/code/package_fresh_mix.py)

For GPU rendering, extract `outputs/fresh-mix/cluster/fresh_mix_blind_inputs.tar` and use either `inputs/legacy` or `inputs/learned` as `ROOT`. The archive contains the condition vectors and predicted words; the registrations for scoring are in the separate `references` directory.

```sh
python "$C/render_within_photo_ip.py" --root "$ROOT" --subject sub-04 \
  --panels mix13 mix14 mix23 mix24 --batch-size 8 \
  --archive "$ARCHIVE" --adapter "$ADAPTER" --dictionary "$DICTIONARY"
```

## Predict one calibrated query

The portable sub-04 model takes four EEG presentations of one photo and eight labeled supports for each of twelve candidate objects. On all 480 registered photographs, independent single-query calls produced the same word, the same eight selected support IDs, and exactly the same 1,024-dimensional own and uniform H/14 vectors as batched inference. The largest absolute difference among standardized mixture scores was 0.000000784 on CUDA. [Single-query implementation](../research/eeg_stack_20261112/code/infer_fresh_mix.py) · [Four-menu parity report](../research/eeg_stack_20261112/results/object_fidelity/fresh_mix_single_query.json)

Export once after fitting the fresh model. The first command prepares the original 512-class visual projection used by both decoders. It can be skipped when that portable bundle is already present. The second command carries its visual projection into the new bundle, exports the expanded Fisher projection and copies the frozen 1,024-object neural checkpoint.

```sh
python "$C/infer_personal_object.py" prepare --subject sub-04 \
  --model-dir outputs/personal-object-inference/sub-04
python "$C/infer_fresh_mix.py" export
python "$C/infer_fresh_mix.py" replay --device cuda \
  --example outputs/fresh-mix/portable/query_example.npz
python "$C/infer_fresh_mix.py" predict --device cuda \
  --input outputs/fresh-mix/portable/query_example.npz \
  --out outputs/fresh-mix/portable/prediction.npz
```

For a new labeled support bank, supply the same nine NPZ fields. The scalar `subject` is `sub-04`. The query has a scalar `query_id` and `query_eeg[4,63,250]`. Provide unique `support_ids[12,8]`, `support_eeg[12,8,4,63,250]`, native ViT-H/14 image embeddings `support_h14[12,8,1024]`, and twelve unique THINGS `candidate_words`. Supply the 63 electrode names in `channels` and post-stimulus seconds in `times[250]`. The physical trials use the released MVNN-preprocessed input and presentation order. Query and support IDs must be disjoint. The prediction contains the selected word, two 1,024-dimensional unit directions, native radius and the twelve candidate scores. The saved result feeds the [one-query image renderer](../research/eeg_stack_20261112/code/render_personal_query.py).
