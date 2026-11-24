# A 512-object, full-scalp personal EEG readout

This comparison keeps the 512 generic training objects, four physical EEG presentations per photograph and eight labeled supports per object fixed. Each query is classified among twelve words. The full-scalp model adds early/late activity from all 63 electrodes to the posterior-only model. Its two weights, 0.25 for the full-scalp four-trial mean and 0.25 for the early/late score, came from the earlier 128-object generic experiment. The posterior score retains weight 1. The three panels here were previously studied, and all three people viewed the same 360 distinct photographs. [Frozen design](../research/eeg_stack_20261112/results/object_fidelity/full_head_512_design.json)

## EEG recognition

Each cell shows the posterior-only and combined 63-channel correct counts, in that order. Each panel contains 120 queries. Across the three earlier panels the full-head model gained 11 correct responses across 1,080 participant–photo observations. All three participant-specific intervals include zero. On the prospectively registered fourth panel the combined readout yielded four fewer correct responses in 360 observations. [Three-panel EEG results](../research/eeg_stack_20261112/results/object_fidelity/full_head_512_eeg.json) · [Fourth-panel registration](../research/eeg_stack_20261112/results/object_fidelity/prospective_panel_design.json)

| Participant | Original panel | Replication panel | Third panel | Earlier three combined, paired difference in percentage points (95% interval) | Prospective fourth panel |
|---|---:|---:|---:|---:|---:|
| sub-04 | 86 → 86 | 94 → 96 | 85 → 85 | 265 → 267, +0.6 [−3.3, +4.4] | 81 → 75 |
| sub-05 | 75 → 73 | 72 → 77 | 60 → 58 | 207 → 208, +0.3 [−3.9, +4.2] | 69 → 68 |
| sub-01 | 67 → 68 | 76 → 77 | 64 → 70 | 207 → 215, +2.2 [−3.3, +7.8] | 82 → 85 |

The two older model components address different outcomes. A word change can alter the prompt and which eight photographs provide the H/14 support. Conversely, equal predicted words preserve the original matched H/14 direction. In the first three panels, 225 of the 1,080 predictions changed and 855 matched the original words and H/14 input bytes. Naming and exact-photo correspondence of a generated JPEG require separate scoring.

## Generated images on the prospectively registered fourth panel

The full-scalp readout did not improve either preregistered image endpoint. Both readouts scored 199/360 on object naming against 1,854 labels. On the primary exact-photo test, the posterior readout identified 42/360 originals and the full-scalp readout identified 40/360. The three people viewed the same 120 photographs. Each row contributed three saved JPEGs per readout, giving 2,160 JPEGs in total. [Paired image results and input digests](../research/eeg_stack_20261112/results/object_fidelity/prospective_paired_images.json)

| Independent image test | Posterior own / uniform / wrong | Full-scalp own / uniform / wrong | Full-scalp minus posterior, own EEG (95% interval, percentage points) |
|---|---:|---:|---:|
| DINOv2 exact original among ten photos of the true object, all 360 queries | 42 / 31 / 38 | 40 / 28 / 31 | −0.6 [−3.6, +2.8] |
| CLIP object name among 1,854 labels, all 360 queries | 199 / 181 / 189 | 199 / 191 / 200 | 0.0 [−5.8, +5.8] |
| CLIP object name among the twelve panel labels, all 360 queries | 232 / 232 / 232 | 228 / 228 / 228 | −1.1 [−9.2, +5.8] |
| DINOv2 two-photo assignment, 64 common correct-word pairs | 36 / 29 / 35 | 36 / 29 / 32 | 0.0 [0.0, 0.0] |
| CLIP two-photo assignment, same 64 pairs | 38 / 34 / 27 | 38 / 34 / 27 | 0.0 [0.0, 0.0] |

The uniform arm uses the mean visual direction of the eight support photos for the predicted word. The wrong arm uses a donor EEG from the same predicted word and fold, retaining the original query's seed. On the twelve-label naming test, every JPEG's correct/incorrect outcome matched the EEG word's correct/incorrect outcome across both readouts and all three arms. The independent 1,854-label test gives the generator more ways to name the wrong object.

Applied to the 120 distinct original stimulus photographs, the same CLIP weights and dictionary named 83/120 originals correctly among 1,854 labels and 119/120 among twelve. Full-dictionary recognition ranged from 2/10 for `sifter` to 10/10 for four objects. These reference counts describe the naming evaluator on actual photographs. [Original-photo naming results](../research/eeg_stack_20261112/results/object_fidelity/prospective_reference_names.json)

The two readouts predicted the same word on 293 queries. All 293 matched JPEGs were byte-identical. The other 67 predictions comprised 23 corrections, 27 losses and 17 changes between two wrong words. Changing the EEG word recovered 21 correct full-dictionary image names and lost 21 others. Both readouts predicted both words correctly on 20, 16 and 28 pairs for sub-04, sub-05 and sub-01 respectively. The 64 common pairs therefore compare identical matched JPEGs. Wrong-EEG JPEGs can differ if other predictions change the donor pool. Intervals resample participants and shared true-object classes separately, retaining paired images and folds within each sampled cell.

## Physical EEG and model

Each concept occupies ten consecutive rows in the original ATM training file. A row contains four trials, 63 channels and 250 samples. Each participant's earlier 512-class fit recorded SHA-256 digests for the original 5,040,000-byte concept ranges. The extractor verifies every range against those digests and exactly reproduces its earlier posterior input. It stores only the new float32 features, two session means per picture and 14 consecutive 40-ms bins per channel over 40–600 ms. The original 128-class range files are reused, reducing new downloads per participant to 384 blocks, 1,935,360,000 bytes. Raw blocks for the other 384 classes pass through memory. [Physical feature extractor](../research/eeg_stack_20261112/code/prepare_full_head_512.py)

The three score branches reuse the earlier projection functions. Two 96-dimensional Fisher projections use the four-trial posterior and full-scalp inputs respectively. A 32-dimensional projection estimates early/late agreement between different training photographs of the same object. It removes time-bin offsets associated with the generic training sessions. The query's early features compare with late support means, and late query features compare with early supports. Every branch uses the same fold, two held-out photographs per object, and row-wise score normalization. Session labels, query photographs and true query names do not enter classification. [Historical model](../research/eeg_stack_20261112/code/fit_full_head_objects.py) · [512-class evaluator](../research/eeg_stack_20261112/code/fit_full_head_512.py)

## Replay

Start from the [512-class and angle-model data](PHOTO_STRUCTURE.md#replay) and the [third panel's physical EEG receipts](FRESH_PHOTO_REPLAY.md). Use the repository root as the working directory. Set `PY` to Python 3.11+ with NumPy, SciPy and Pillow. The development gate was fixed before computing the full-scalp target results. It requires a strictly positive net change in sub-04's correct predictions across the three panels before proceeding to sub-05 and sub-01.

```sh
C=research/eeg_stack_20261112/code
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
"$PY" "$C/prepare_full_head_512.py" design
"$PY" "$C/prepare_full_head_512.py" fetch --subject sub-04 --workers 5
"$PY" "$C/fit_full_head_512.py" --subject sub-04
if "$PY" -c 'import json,sys; sys.exit(not json.load(open("outputs/full-head-512/sub-04_gate.json"))["passed"])'; then
  for s in sub-05 sub-01; do
    "$PY" "$C/prepare_full_head_512.py" fetch --subject "$s" --workers 5
    "$PY" "$C/fit_full_head_512.py" --subject "$s"
  done
fi
"$PY" "$C/fit_full_head_512.py" --summarize
```

The per-query files are `outputs/full-head-512/predictions/PANEL/SUB/evaluation.json`. They include the original row IDs, supports' fold numbers, vocabulary, predictions and true-class index used at scoring. Before calculating changes, the code requires the 512-class posterior-only branch to reproduce every saved word for all three panels. On sub-04 it also verifies exact historical 128-class and target feature values for both full-scalp halves. The [aggregate EEG results](../research/eeg_stack_20261112/results/object_fidelity/full_head_512_eeg.json) give class-bootstrap intervals and the within-session-pair conditional expectation for each model.

For an independently registered later panel, supply a separate physical-data root. It must contain `references/PANEL_ablation.json`, the corresponding `eeg/SUB.npz` and range receipt `eeg/SUB.json`, and the 12 checksum-verified ten-photo physical blocks in `ranges/REV/SUB/`. Its classes must be disjoint from the 512 generic objects and all 36 earlier targets. This command uses the frozen original weights on both the full-scalp and posterior-only branches.

```sh
"$PY" "$C/fit_full_head_512.py" --subject sub-04 \
  --confirmation-root "$NEXT_PANEL_ROOT" --confirmation-panel prospective
```

To prepare the H/14 generation inputs for the first three panels, run `"$PY" "$C/package_full_head_512.py"`. The packaging stage replays the original posterior-word directions, supports and donors before recomputing them for the full-head words. It lists which matched directions and prompts remain byte-identical. This permits comparison against the earlier same-seed generated images. Actual image naming requires generating and scoring the candidate JPEGs with the independent image evaluator. [Condition packaging](../research/eeg_stack_20261112/code/package_full_head_512.py)

For the fourth panel, use the [frozen render-input manifest](../research/eeg_stack_20261112/results/object_fidelity/prospective_render_inputs.json). Save one CLIP report per readout and collect its three completed participant render directories. The DINO scorer checks row order, folds, original image identity, word predictions, support rows, model weights, noise seeds and JPEG hashes. It writes the per-query records to `outputs/prospective-photo/paired_scores`. [Paired scorer](../research/eeg_stack_20261112/code/score_prospective_readouts.py)

```sh
P=outputs/prospective-photo
for m in expanded_r4 full_head_512; do
  R="$P/render_inputs/$m"
  "$PY" "$C/score_within_photo_ip.py" --root "$R" --archive "$ARCHIVE" \
    --dictionary "$DICTIONARY" --reference-dir "$P/references" --panels prospective \
    --subjects sub-04 sub-05 sub-01 --out "$R/clip.json"
done
"$PY" "$C/score_prospective_readouts.py" all \
  --release research/eeg_stack_20261112/results/object_fidelity/prospective_paired_images.json
```

## Later H/14 direction experiments on the same photographs

Two generator experiments use the posterior readout's words on this already examined panel. One estimates a component of the EEG signal that recurs across physical presentations. The other estimates a component shared across participants. The methods passed their own held-out generic-object tests before their directions were used for image generation. The fourth panel provides development evidence for these two methods. The prospective full-scalp readout comparison above used the earlier frozen generator. [Repeat-based fit](../research/eeg_stack_20261112/code/fit_repeat_photo_subspace.py) · [Shared-response fit](../research/eeg_stack_20261112/code/fit_shared_photo_eeg.py)

Each candidate preserves the posterior word, the eight support photos, the magnitude of the recipient's original angular deviation and the same image seed. Its wrong arm rotates the recipient vector by that same angle in a donor's direction. The matched comparisons use the original posterior JPEG, its uniform JPEG and the candidate's wrong JPEG for each query. The evaluator also counts how many regenerated uniform JPEGs match the originals byte for byte. Confidence intervals for these development comparisons resample the twelve true object classes together for all three participants and all arms. They are conditional on the three observed participants.

| Test | Original posterior, own | Repeat-based, own / wrong | Shared-response, own / wrong | Uniform |
|---|---:|---:|---:|---:|
| DINOv2 exact original among ten, 360 queries | 42 | 40 / 33 | 37 / 31 | 31 |
| CLIP name among 1,854, 360 queries | 199 | 182 / 184 | 192 / 189 | 181 |
| DINOv2 two-photo assignment, 78 correct-word pairs | 46 | 47 / 39 | 47 / 39 | 37 |
| CLIP two-photo assignment, same 78 pairs | 46 | 49 / 37 | 51 / 39 | 42 |

Both candidates scored below the original posterior direction on the primary exact-photo test. The paired differences were −0.6 percentage points [−3.1, +2.2] for the repeat-based direction and −1.4 [−3.6, +0.6] for the shared-response direction. Both scored one additional correct DINOv2 assignment on the restricted two-photo task. All 720 uniform JPEGs matched the original uniform JPEGs byte for byte. [Paired development results](../research/eeg_stack_20261112/results/object_fidelity/prospective_generator_candidates.json)

After the original posterior DINO scores and all candidate JPEGs are available, save each candidate's independent CLIP report as `outputs/prospective-photo/refinements/METHOD/clip.json` using the same CLIP command and participant list above with `R="$P/refinements/$m"` for `m=repeat` and `m=shared`. Then run the following command to compute the paired DINO and CLIP comparisons on identical registered queries. [Scoring implementation](../research/eeg_stack_20261112/code/score_prospective_candidates.py)

```sh
"$PY" "$C/score_prospective_candidates.py" all
```
