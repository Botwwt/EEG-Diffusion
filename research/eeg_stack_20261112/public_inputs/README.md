# Build the public EEG and visual inputs

These commands prepare the inputs used by the paired benchmark from the published THINGS-EEG2 recordings, original JPEGs and Brain-HiVE visual embeddings. The CPU verifier reopens the source archives and compares each retained float16 bit, photo identity, channel, repeat and session assignment. The original ten-participant check covered **821,600 trials and 3,491,800,000 samples**. The resulting 62-file checksum list had SHA-256 `d135aa9025945ba889c5b1b157167d2a2543e4c26e9f3099cd263083f709f6f0`. [EEG release](https://huggingface.co/datasets/Haitao999/things-eeg/tree/944f0d50e37e125eb6b7b98d14a41571c9ed0ef7/Preprocessed_data_250Hz_whiten) · [Official photographs](https://osf.io/y63gw/) · [Visual embeddings](https://huggingface.co/datasets/fakekungfu/Brain-HIVE_Visual_Embeddings/tree/4efe61ddfde96eea962c391ec01736d42d9b60ef/things-eeg)

## Prepare and verify

Run from the repository root in a Python 3.11 environment. The full source EEG occupies 26.16 GB and the 17-channel arrays occupy another 6.98 GB. The downloader also needs temporary space for the largest files being verified. The frozen gallery reference used CPU PyTorch 2.3.0 and PyArrow 18.1.0.

```sh
python -m pip install 'numpy==1.26.4' Pillow 'pyarrow==18.1.0' safetensors einops
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.3.0'

recipes=research/eeg_stack_20261112/public_inputs
export PAPER_INPUTS_ROOT="$PWD/outputs/paper-benchmark/inputs"
export RAD_ROOT="$PWD/outputs/paper-benchmark/replay"
export ARCHIVE="$PWD/outputs/paper-benchmark/archive"

python "$recipes/stage_public_inputs.py" --groups eeg images visual code fusion-encoder
python "$recipes/index_official_photos.py"
python "$recipes/prepare_native_eeg.py"
python "$recipes/verify_native_inputs.py"
python "$recipes/build_official_visual_reference.py"
python "$recipes/handoff_benchmark.py" --prep "$RAD_ROOT/data/prep" --training-targets "$ARCHIVE/data/native_training_v239"
```

| Input under `RAD_ROOT/data/prep` | Contents |
| --- | --- |
| `sub-XX_train.npy` and `sub-XX_test.npy` | Native float16 waveforms. Shapes are 16,540 × 4 × 17 × 250 and 200 × 80 × 17 × 250, respectively. |
| `sub-XX_{train,test}.json` and `sessions/*.npy` | Per-person source and output hashes, channel indices and one session ID for each physical repeat. |
| `eeg_{train,test}_ids.json` and `{train,test}_ids.json` | EEG and photographic row maps. Training EEG paths start with `train_images/` and ZIP paths start with `training_images/`. Their 16,540 filename stems agree in order. |
| `F_test.npy` and `paper_test_F_receipt.json` | A 200 × 1,024 float32 gallery for feature retrieval. The receipt binds the ordered photo IDs, input shard hashes, frozen vision encoder and actual output digest. |
| `paper_input_provenance/` | Public asset and source manifests, the original-format EEG checksum list and the full verification report. The checksum list uses paths relative to `PAPER_INPUTS_ROOT/prep17`. |

The original CPU gallery had SHA-256 `cccf550303de1d036f758b4bd08dd7fc8d44cb5510f828361d03b5388e003753`. The builder checks all three training tensors against their archived float32 byte hashes and retains them with their 16,540 ordered IDs. Different PyTorch versions can serialize identical tensors into different containers. [Asset hashes and source revisions](public_manifest.json) · [Brain-HiVE source file hashes](author_sources.json)

## Connect the photo and generator inputs

The RGB scorer reads the two official CSVs, test JPEG archive and training JPEG archive from the following paths. The photo index builds the original manifest in EEG row order and the 1,854-name dictionary in THINGS vocabulary order.

```sh
official="$ARCHIVE/runs/visual_fidelity_20261125_000818/official"
photos="$PAPER_INPUTS_ROOT/image_set"
mkdir -p "$official" "$ARCHIVE/assets/things_images_v254"
cp "$photos/originals_manifest.csv" "$photos/dictionary.csv" "$photos/test_images.zip" "$official/"
cp "$photos/training_images.zip" "$ARCHIVE/assets/things_images_v254/"
```

For generation, fetch the projection and attention adapter and put the three Fusion Prior components in the paths consumed by the preparation script. The separate SDXL-Turbo base and vision scorer weights are used in the generation and scoring stages. [Fusion Prior release](https://huggingface.co/fakekungfu/Fusion-Prior-H14_B32_VAE/tree/135a284109e71f5cde3d3e6d5dee6268a3c7987a) · [SDXL-Turbo base](https://huggingface.co/stabilityai/sdxl-turbo) · [Paired benchmark protocol](../PAPER_BENCHMARK_PROTOCOL.md)

```sh
python "$recipes/stage_public_inputs.py" --groups fusion-adapters
prior="$PAPER_INPUTS_ROOT/fusion-prior-source"
mkdir -p "$RAD_ROOT/code" "$ARCHIVE/assets/native_generator_v243/prior_H14" "$ARCHIVE/assets/native_teacher_v239"
cp -R "$PAPER_INPUTS_ROOT/Brain-HIVE" "$RAD_ROOT/code/"
cp -R "$prior/fusion_encoder" "$prior/proj" "$prior/attn_adapter" "$ARCHIVE/assets/native_generator_v243/prior_H14/"
cp "$prior/fusion_encoder/diffusion_pytorch_model.safetensors" "$ARCHIVE/assets/native_teacher_v239/fusion_encoder.safetensors"
```

| Frozen component | Published safetensors SHA-256 |
| --- | --- |
| Fusion encoder | `b345fca423c5fe43178999ae38255bc2297011ce85c1ed664f5d54a9843e6e96` |
| Projection | `8ef44b68e2692abf01e4e6e7717a316b9260369a32ea7b958d5dfe35bb8661b2` |
| Attention adapter | `75a023a59cd2ea6f0488e090cea765f6dfda4042cf038be234c286160849c968` |
