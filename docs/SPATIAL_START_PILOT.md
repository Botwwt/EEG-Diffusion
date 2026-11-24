# Personal EEG as a spatial diffusion start

This development experiment adds an EEG estimate of spatial layout to the previously fitted H/14 angle condition. The image generator receives a continuous 4 × 8 × 8 grid in SDXL VAE units. The 24-query image pilot covers the entire first fold of sub-04's original registered panel, with two photographs from each of twelve objects. The same participant and objects had appeared in earlier image studies. [Earlier H/14 comparison](PHOTO_STRUCTURE.md#calibrating-the-residual-angle-before-generation)

## What the pilot measured

The matched spatial EEG and zero-residual arms retrieved the correct original on the same four of 24 queries. Among seven same-object pairs with both words correct, the matched arm assigned two pairs correctly and the zero-residual arm assigned three. All 24 queries were generated and evaluated. [Aggregate counts, intervals and input digests](../research/eeg_stack_20261112/results/object_fidelity/spatial_start_pilot.json)

| Endpoint | Matched spatial EEG | Support mean | Wrong spatial EEG | Earlier four-step result |
|---|---:|---:|---:|---:|
| DINO top-1 among ten originals of the true object | 4/24 | 4/24 | 3/24 | 4/24 |
| DINO paired assignment, both predicted words correct | 2/7 | 3/7 | 2/7 | 6/7 |
| DINO paired assignment, all object pairs | 6/12 | 7/12 | 6/12 | 10/12 |
| Mean RGB RMSE at 32 × 32, lower is better | 0.2530 | 0.2528 | 0.2539 | 0.2867 |
| Mean spatial luminance correlation, higher is better | 0.1640 | 0.1768 | 0.1634 | 0.0443 |

The support-mean and matched runs both had lower pixel error than the earlier model, which also used a different denoising schedule. Adding spatial EEG to the identical two-step support start changed RGB RMSE by +0.00011 (95% whole-object bootstrap interval −0.00295 to +0.00278) and luminance correlation by −0.0128 (−0.0431 to +0.0155). The image results did not justify expanding this pilot.

On the generic calibration objects, ridge 8 yielded 1,440/2,560 correct paired latent assignments, 56.25% (95% whole-object interval 54.18% to 58.28%). Its pooled residual cosine was 0.0352. Relative to the eight-support prior, the cross-validated continuous-layout R² was −0.00084 at the native ridge output and −0.0220 after the registered generative gain of 2.321. The archived VAE modes were extracted at the correct original scale. The largest absolute difference on 24 independently encoded overlapping photographs was 0.00000167. [Generic and image results](../research/eeg_stack_20261112/results/object_fidelity/spatial_start_pilot.json)

## Model and controls

Each of 512 separate generic objects supplies ten photographed exemplars with their EEG features and VAE targets. The model removes each latent channel's spatial mean per photograph and subtracts the object's mean from both modalities during training. A ridge maps 1,003 posterior EEG features to the remaining 256 values. Four held-object folds on sub-04 select the ridge by mean squared prediction error. The registration also gives the rule for partially restoring the generic residual amplitude. The final fit uses all 512 objects. [Fitting code](../research/eeg_stack_20261112/code/fit_personal_spatial_latents.py)

For each query, the frozen personal Fisher classifier's predicted word selects eight support photographs and their EEG vectors. All three spatial starts contain the same mean of the eight native VAE grids. The matched start adds the query's predicted layout minus its eight support predictions. The zero start uses only the support mean. The wrong start adds the prediction from the original same-word, same-fold donor. All three receive the same query's H/14 angle vector, text and row seed. Two queries have singleton donor groups and are retained. Bilinear interpolation expands the start to 64 × 64. Img2img uses the last two steps of the original SDXL-Turbo schedule at timesteps 499 and 249. The historical comparator uses four steps from noise. [Renderer](../research/eeg_stack_20261112/code/render_personal_spatial_latents.py)

## Replay the 24-query pilot

Start from the [registered and personal model replay](PHOTO_STRUCTURE.md#replay) and [fresh registration](FRESH_PHOTO_REPLAY.md). Run from the repository root. Set `ARCHIVE` to the historical custodian archive, `ANGLE` to the previous angle experiment root, `GPU` to the historical PyTorch 2.3.0 CUDA environment, and `PY` to local Python 3.11+ with NumPy and SciPy. The fixed 4,150 generic training photos already in historical T use their pooled archived modes. The 970 other generic training photos and 50 earlier panel photos use the same native VAE with its original preprocessing. An additional 24 generic overlap photos check the two encoders numerically. [Pinned source encoder](../research/eeg/PKU_build_native_T_latents_GPU_20261109_v256.py)

```sh
C=research/eeg_stack_20261112/code
L=outputs/latent-spatial
T="$ARCHIVE/data/native_T_latents_v256"
mkdir -p "$L/source"
cp "$T/registration.json" "$T/completion.json" "$L/source/"
"$PY" - "$T" "$L/source/historical_t8.npz" <<'PYTHON'
import hashlib, json, sys
from pathlib import Path
import numpy as np
root, dest = map(Path, sys.argv[1:])
zfile = root / 'native_VAE_T_modes.npy'
with zfile.open('rb') as fp:
    assert hashlib.file_digest(fp, 'sha256').hexdigest() == 'b1198b3e07361e3b941f87e3b60146b1cc7957293e813db69f7b59cfa8469a2d'
rows = json.loads((root/'registration.json').read_bytes())['array_rows']
z = np.load(zfile, mmap_mode='r')
assert z.shape == (len(rows), 4, 64, 64) and z.dtype == np.float32
grids = [z[i:i+64].reshape(-1,4,8,8,8,8).mean((3,5)) for i in range(0,len(rows),64)]
np.savez(dest, rows=np.asarray(rows), latents=np.concatenate(grids))
PYTHON
"$PY" "$C/prepare_personal_spatial_latents.py" manifest
"$GPU" "$C/prepare_personal_spatial_latents.py" encode --archive "$ARCHIVE" --out "$L"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
"$PY" "$C/fit_personal_spatial_latents.py" all --subjects sub-04 --panels registered
"$GPU" "$C/render_personal_spatial_latents.py" --archive "$ARCHIVE" --h14-root "$ANGLE" \
  --adapter "$ADAPTER" --dictionary "$DICTIONARY" --subject sub-04 --panel registered \
  --spatial "$L/conditions/sub-04/registered/spatial.npz" --out "$L" --limit 24 --batch-size 8
"$DINO_PY" "$C/score_personal_spatial_pilot.py" --h14-root "$ANGLE" \
  --release research/eeg_stack_20261112/results/object_fidelity/spatial_start_pilot.json
```

`ADAPTER` is the original Tencent ViT-H SDXL image adapter and `DICTIONARY` is the registered THINGS 1,854-word CSV. `DINO_PY` uses PyTorch, Transformers, NumPy, SciPy and Pillow. The scorer verifies all 72 new JPEG digests and the 24 historical JPEGs before accessing original photographs. It scores DINOv2-S/14 CLS cosine against the full 120-photo original gallery, restricting within-object retrieval to the ten photos of the true object. It also records uncentered RGB error, spatial luminance correlation and two-photo matching. For the old four-step control, per-row ten-photo hits and all seven eligible paired assignments must reproduce the earlier independent DINO results. A fixed contact sheet uses original source row 96, the first registered query. Generated and original images stay in the local workspace. [Scoring code](../research/eeg_stack_20261112/code/score_personal_spatial_pilot.py)
