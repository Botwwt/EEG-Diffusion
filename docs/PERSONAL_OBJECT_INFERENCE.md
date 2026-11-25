# One EEG query with a personal twelve-word calibration

The mixed trial/Fisher decoder takes four EEG presentations of one photograph and chooses a word from twelve supplied candidates. Each candidate needs eight distinct calibration photographs with known words, their EEG, and their image embeddings. The decoder returns the chosen word, a 1,024-dimensional unit H/14 direction, the eight-photo visual prior, and the image-embedding radius. One invocation requires neither the viewed photograph nor the true word. [Inference source](../research/eeg_stack_20261112/code/infer_personal_object.py)

## Prepare the personal model

From the repository root, use the same Python environment as the trial experiment, with NumPy, SciPy, PyTorch and safetensors. The `prepare` command reads the participant's frozen checkpoint and verified 512-category calibration cache and exports a portable folder containing `model.safetensors`, `heads.npz` and `receipt.json`. For sub-04 it reads `outputs/trial-object-encoder/target`; for sub-05 and sub-01 it reads `outputs/trial-object-replication/<subject>/fit`. The visual fit uses `outputs/photo-structure/expanded/features`, the released H/14 training features, and the registered H/14 angle and radius. [Sub-04 model and training source](../research/eeg_stack_20261112/code/package_trial_object_confirmation.py) · [Participant replication](../research/eeg_stack_20261112/code/replicate_trial_object_confirmation.py)

```sh
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
C=research/eeg_stack_20261112/code
M=outputs/personal-object-inference
python "$C/infer_personal_object.py" prepare --subject sub-04 --model-dir "$M/sub-04"
python "$C/infer_personal_object.py" predict --model-dir "$M/sub-04" \
  --input query.npz --out prediction.npz
```

Inference defaults to CPU. Supply exactly the following fields in an NPZ written with NumPy arrays. A transferred model folder and the query NPZ suffice for `predict`. Check the participant's recording montage and preprocessing before comparing new inputs with the reported data.

| Field | Shape | Contents |
| --- | --- | --- |
| `subject`, `query_id` | scalar each | Model participant and a distinct opaque identifier for the current photograph |
| `channels`, `times` | (63,), (250,) | Unique channel names and sample times in seconds from 0 to 0.996 at 250 Hz. The loader reorders named channels. |
| `query_eeg` | (4, 63, 250) | Four physically measured trials in the public THINGS-EEG2 preprocessed MVNN signal units and stored repetition order. Preserve the full post-onset second. |
| `candidate_words` | (12,) | Twelve unique Unicode word IDs in the intended score order |
| `support_ids` | (12, 8) | Unique calibration photo IDs aligned to the candidate words. All 96 must differ from the query ID. Use integer or Unicode IDs. |
| `support_eeg` | (12, 8, 4, 63, 250) | Same participant, montage, times, signal preprocessing and stored trial order as the query |
| `support_h14` | (12, 8, 1024) | OpenCLIP ViT-H/14 projected image vectors in the author's published visual coordinates, aligned to the support IDs. Each vector is normalized before averaging. |

The convolutional trial encoder uses the first 125 samples. Fisher and the visual readout use 17 posterior channels and the full second, with the early pair of stored slots (1, 3) and late pair (0, 2) averaged in float32. Each head compares query EEG with twelve support EEG prototypes. Equal weighting of their standardized score vectors gives the mixed decision. The selected word supplies its eight visual supports. The direct visual readout subtracts their average EEG prediction from the query prediction and applies the frozen angular calibration. The visual support pixels and labels of the held-out target never enter the computation. [Decision and conditioning protocol](../research/eeg_stack_20261112/results/object_fidelity/trial_object_personal12_design.json)

The output NPZ includes `word`, `mixed_scores`, `fisher_scores`, `trial_scores`, selected `support_ids`, `own`, `uniform`, and scalar `radius`. Both H/14 vectors are unit directions. Multiplying by `radius` supplies the positive image embedding expected by the archived adapter interface. Scores are standardized similarities. [Adapter rendering](../research/eeg_stack_20261112/code/render_within_photo_ip.py)

## Replay one independent query at a time

The replay harness constructs each input from four measured trials and the 96 allowed support photos, calls the same single-query decoder, and checks the result against the existing rendering packet. Across the three participants, all 360 words matched, and the maximum absolute difference in either stored float32 direction was zero on this run. The denominator counts agreement with the frozen model. Ground-truth recognition accuracy has a separate evaluation. [Single-query parity](../research/eeg_stack_20261112/results/object_fidelity/single_query_personal_parity.json) · [Sub-04 word and image accuracy](../research/eeg_stack_20261112/results/object_fidelity/personal12_trial_images.json)

```sh
for s in sub-04 sub-05 sub-01; do
  python "$C/infer_personal_object.py" prepare --subject "$s" --model-dir "$M/$s" || exit 1
  python "$C/infer_personal_object.py" replay --subject "$s" --model-dir "$M/$s" \
    --save-example "$M/${s}_query.npz" || exit 1
done
python "$C/infer_personal_object.py" audit --model-dir "$M" --out "$M/replay_summary.json"
python "$C/infer_personal_object.py" predict --model-dir "$M/sub-04" \
  --input "$M/sub-04_query.npz" --out "$M/sub-04_prediction.npz"
```

## Generate the image

On a GPU with the archived generator environment and weights, feed that prediction directly to the four-step renderer. Choose the seed before inspecting any reference. The example uses the seed for archive row 519. The output directory contains the reconstruction JPEG and a JSON file with the prediction, seed and asset hashes. [Generator assets](FRESH_PHOTO_REPLAY.md)

```sh
python "$C/render_personal_query.py" --input "$M/sub-04_prediction.npz" \
  --archive "$ARCHIVE" --adapter "$ADAPTER" --dictionary "$DICTIONARY" \
  --seed 924280 --out "$M/single_image"
```

In the one-query GPU replay, the predicted word was `bag` and the saved JPEG was 512 × 512. The input word, H/14 direction and seed matched the first row of the registered eight-at-a-time render. The one-at-a-time JPEG had a mean absolute RGB difference of 0.00391 on the 0–1 scale relative to that batched image. The 360-query parity reported above applies to the words and conditioning vectors.
