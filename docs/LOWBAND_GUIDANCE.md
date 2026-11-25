# Guiding one native denoising step with an EEG color layout

This test uses all 120 queries on sub-05's previously studied personal 12-object panel. We held the decoded word, continuous H/14 direction, seed, four diffusion steps and noise schedule fixed. The intervention adds a weak spatial correction at the third denoising step. The spatial readout supplies an 8 × 8 RGB anchor estimated from the mean of eight authorized support photos and the query's EEG residual. This experiment evaluates a new use of the existing readout on familiar evaluation data. [Frozen design and source hashes](../research/eeg_stack_20261112/results/object_fidelity/lowband_guidance_design.json) · [Spatial readout construction](SPATIAL_PHOTO.md)

The spatial ridge was fitted separately for sub-05 using 512 other objects. The first three decoded spatial channels describe luminance and two color differences. We convert those to RGB after the inverse discrete cosine transform. The generic model's per-band gains are retained. The source photos used to construct the anchor are always the predicted object's eight off-fold supports. The four EEG presentations from the query and its permitted supports determine the residual. The other two photos of each object in the active fold are excluded from all support calculations. [Input preparation and denoising code](../research/eeg_stack_20261112/code/lowband_guidance.py)

## Four arms

| Arm | Intervention |
| --- | --- |
| Native | Ordinary four-step text-to-image generation. Every one of its 120 JPEGs must reproduce the archived original bytes. |
| Prior | Correct the clean latent toward the average layout of the predicted object's eight supports. |
| Matched | Add the recipient's EEG-predicted spatial residual to the prior. |
| Exchanged | Add the spatial residual of the registered donor from the same predicted word and fold. On 108/120 rows the donor is a different query. |

At diffusion timestep 499, the native prediction for the noise-free latent is $x_0=x-\sigma\epsilon$. We compute spatial averages over nonoverlapping 8 × 8 blocks of its four 64 × 64 latent channels. The target uses the same averages of a VAE posterior mode, obtained by encoding a bilinear enlargement of the predicted RGB grid. We upsample their difference bilinearly and add 0.35 times this correction to the clean estimate by modifying the predicted noise supplied to the original Euler ancestral update. The noise generators, VAE decoder and updates at timesteps 999, 749 and 249 follow the original procedure. The native arm bypasses the correction function.

## Image outcomes

The primary outcome is independent DINOv2-S/14 identification of the viewed original among the ten real photos of its object on every query. The two held-out originals give a separate comparison on pairs with both words correct. CLIP-L/14 tests object naming among the twelve supplied names and the full 1,854-label dictionary. The evaluator reopens each 512-pixel JPEG, verifies its SHA, and measures 32 × 32 RGB error and luminance correlation against the original. Native DINO and naming credits must reproduce the earlier per-query scores. The development gate requires more ten-way hits than native and prior, a positive lower 95% paired object-bootstrap bound over exchanged EEG and at least the native full-vocabulary score. [Independent scoring](../research/eeg_stack_20261112/code/score_lowband_guidance.py)

## Sub-05 results

All 120 generated native JPEGs reproduced the original bytes. The matched correction gave 17/120 DINO ten-way hits, four more than native (paired gain +3.33 percentage points, 95% interval 0.00 to +7.50). Its advantage over exchanged EEG was two hits (+1.67 points, 0.00 to +4.17). Full-vocabulary naming fell by one hit. The correction therefore failed the preregistered promotion gate. [Per-arm estimates and intervals](../research/eeg_stack_20261112/results/object_fidelity/lowband_guidance_sub05.json) · [Generated examples from all twelve objects](../research/eeg_stack_20261112/gallery/lowband_guidance_sub05.jpg)

| Outcome | Native | Support prior | Matched EEG | Exchanged EEG |
| --- | ---: | ---: | ---: | ---: |
| DINO viewed photo among ten /120 | 13 | 16 | 17 | 15 |
| CLIP viewed object among 1,854 labels /120 | 54 | 53 | 53 | 54 |
| CLIP viewed object among 12 labels /120 | 54 | 54 | 54 | 54 |
| DINO joint held-out photo assignment /16 | 10 | 9 | 10 | 9 |
| RGB RMSE at 32 × 32 | 0.30202 | 0.28827 | 0.28773 | 0.28823 |
| Luminance correlation at 32 × 32 | 0.05318 | 0.06734 | 0.06894 | 0.06801 |

Most of the lower RGB error was already present with the support prior. Matched minus prior RMSE was −0.00053 (−0.00119 to +0.00002). The matched luminance correlation exceeded exchanged EEG by 0.00092 (+0.00005 to +0.00183), a small secondary result.

We separately checked full-resolution edge strength after seeing the primary scores. In the 512 × 512 images, RMS luminance change between adjacent pixels decreased from 0.03115 for native to 0.02164 for matched guidance. The originals averaged 0.03664. All three corrected arms had nearly the same edge strength. The mean cosine between native and matched gradient fields was 0.765. Thus a better coarse color comparison coexisted with weaker fine-scale contrast. This measurement can also reflect texture and noise. [Post-render edge audit and units](../research/eeg_stack_20261112/results/object_fidelity/lowband_guidance_edges.json)

## Replay

Restore the original sub-05 trial/Fisher conditions, foreground-color readout, THINGS image archive and four-repeat EEG packet. The preparation stage verifies the input and training hashes. Render and run independent CLIP in the offline archive environment used for the native sub-05 run. The DINO stage also needs the previously verified personal12 original-image cache. The original run uses the 120 row seeds and batch size eight. [Participant inputs](TRIAL_OBJECT_PERSONAL12.md) · [Batch job](../research/eeg_stack_20261112/jobs/run_lowband_guidance.sbatch)

~~~sh
C=research/eeg_stack_20261112/code
python "$C/lowband_guidance.py" prepare --root "$RUN"
python "$C/lowband_guidance.py" render --root "$RUN" \
  --archive "$ARCHIVE" --adapter "$ADAPTER" --dictionary "$DICTIONARY"
python "$C/lowband_guidance.py" clip --root "$RUN" \
  --archive "$ARCHIVE" --dictionary "$DICTIONARY"
python "$C/score_lowband_guidance.py" --root "$RUN"
~~~
