# Two visual decoders of the same 80 EEG observations

The personal decoder and the released ATM readout each retrieved 65 of the 200 official photographs in their respective matching spaces for sub-04. Each retrieved 28 stimuli missed by the other. This prompted an additional test using the same participant, photographs and eighty physical observations per photograph. The original 200 were previously studied. Each object has one photograph in the gallery. [Earlier measurement and acquisition protocol](OFFICIAL_REPEAT_OBJECT.md) · [Image protocol committed before rendering](../research/eeg_stack_20261112/results/object_fidelity/dual_r80_image_protocol.json)

## Decoders and prompts

The personal visual decoder predicts a ViT-H/14 image direction using 17 posterior EEG electrodes. The second decoder starts from the released 1,024-dimensional ATM EEG embedding. On 64 generic calibration objects, its median angular distance from the training-photo mean was 1.434 radians, compared with 0.887 for actual photo embeddings. The registered rule therefore fitted a ridge to native image directions and calibrated the angular gain. The equal-weight model averages the unit directions of the personal and ATM-derived heads and normalizes the result.

All three aligned image conditions use the same word estimated by the original personal EEG readout, image-feature radius, four SDXL-Turbo steps and noise seed for each row. In the exchanged control, the other row supplies both the word and the dual direction while the recipient keeps its seed. The ATM-derived arm uses this study's word and renderer. The original ATM authors trained their encoder on the upstream training set and monitored the official test set while training. The 64 validation objects were held out only from the downstream fit in this study. The exact epoch behind the released features could not be independently verified. [ATM model and original procedure](https://proceedings.neurips.cc/paper_files/paper/2024/file/ba5f1233efa77787ff9ec015877dbd1f-Paper-Conference.pdf) · [Revision-pinned author features](https://huggingface.co/datasets/LidongYang/EEG_Image_decode/tree/dbe34bb2407164f70883c661c76664f0d596e522/emb_eeg)

The released ATM training tensor contains 66,160 observations for 16,540 training photos. We averaged four consecutive entries per photo, consistent with the authors’ loader. In a row-order check on 640 generic photos, that arrangement beat another image of the same object in 97.3% of comparisons. Two alternative repeat-major arrangements gave 48.7% and 49.4%. The public release provides no trial-by-trial manifest for its feature export. [Feature counts, source hashes and evaluation](../research/eeg_stack_20261112/results/object_fidelity/dual_r80_feature_evidence.json)

## What reached the generator

On the 200 official photos, nearest-original retrieval in the native H/14 space of the generator gave 17 correct for the personal head, 33 for the adapted ATM head, 26 for their equal-weight average, and zero for the exchanged average. The earlier 65-hit results used a distinct retrieval-specific representation for each head. Separately, a fixed equal blend of the two word scores ranked the correct word first on 4/200 among 1,854 labels. The individual readouts scored 6/200 (personal) and 2/200 (ATM). Restricting the menu to the 200 true object words gave 31, 27 and 16 respectively. None of the new word scores replaced the fixed prompt in this image comparison.

## Generated images

Every entry uses all 200 original stimulus IDs in the same order. Full naming uses split credit across tied vocabulary entries. As a scorer check, CLIP gave 170.25/200 credits to the originals themselves. The 200-word column restricts the scoring vocabulary to the true object list; those names were unavailable to the image generator.

| R80 direction, identical words in the aligned arms | CLIP full 1,854-word naming | CLIP 200-word naming | CLIP original-photo retrieval | DINOv2 original-photo retrieval | RGB RMSE | Luminance correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous personal head | 4 | 14 | 18 | 24 | .2757 | .0299 |
| Adapted ATM head | 4 | 22 | 23 | 29 | .2712 | .0477 |
| Equal blend | 3 | 19 | 19 | 32 | .2721 | .0470 |
| Exchanged blend and word | 0 | 1 | 1 | 0 | .3027 | .0088 |

The equal blend fell one hit below each individual head on the primary full-vocabulary object test. Its difference from the previous personal head was −0.5 percentage points (paired 95% interval −2.5 to +1.0). DINO retrieval gained eight images over the personal head (+4.0 points, −1.0 to +9.0) and three over the adapted ATM head (+1.5 points, −3.5 to +6.5). RGB error fell by .00362 relative to the personal head (−.00677 to −.00036). These conditional intervals resample the 200 previously studied objects with the participant, noise seeds and image gallery fixed. The fusion failed the registered image-naming criterion. [Paired CLIP decisions](../research/eeg_stack_20261112/results/object_fidelity/official_dual_r80_clip_paired.json) · [DINO and pixel statistics](../research/eeg_stack_20261112/results/object_fidelity/official_dual_r80_images.json)

In the fixed gallery, row 100 retains the dense, colorful arrangement of the jelly beans. In row 060 the cupcake becomes a metal appliance, and in row 160 the seagull becomes a small mammal in all three aligned arms. Each row is a multiple of twenty in the original archive. [Fixed generated images](../research/eeg_stack_20261112/gallery/official_dual_r80.jpg)

## Replay

First reproduce the official EEG inputs and the previous calibrated named images for sub-04. The published instructions list the model weights, the evaluation image archive, the dictionary, the independent scorers and the Slurm environments. The ATM acquisition step verifies the two revision-pinned author archives before decoding their numeric storage. [Official EEG preparation](OFFICIAL_REPEAT_OBJECT.md#replay) · [Generator assets and reference comparison](OFFICIAL_REPEAT_IMAGES.md#replay) · [Fit specification](../research/eeg_stack_20261112/results/object_fidelity/dual_r80_fit_design.json)

From the repository root, run the following in the research environment.

```sh
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
C=research/eeg_stack_20261112/code
S=outputs/official-repeat-object/sub-04/dual-r80
for step in acquire register prepare score; do
  python "$C/fit_dual_r80.py" "$step"
done
python "$C/register_dual_r80_images.py"
python "$C/package_official_repeat_jobs.py" --series dual_r80 --out outputs/official-repeat-object/replay-bundles
```

Submit the resulting render and score bundles with `SERIES=dual_r80` using the existing official-image Slurm commands. Extract each of the three render archives under `$S/render` and the scorer's `clip.tar` under `$S`. The transport keys `r4`, `r80` and `wrong` correspond here to the ATM-derived head, the equal blend and the exchanged blend. All use 80 observations. The registration records this mapping and verifies the word, row identities and seed protocol against the previous R80 input. The scientific reports use the decoder names.

```sh
python "$C/render_official_repeat_images.py" join --packet "$S/render_packet.npz" \
  --design "$S/render_design.json" --dictionary outputs/official-object-readout/sub-03/dictionary.csv --out "$S/render"
python "$C/score_dual_r80_images.py" dino
python "$C/score_dual_r80_images.py" clip
```

The scorers verify all generated JPEGs and the previous row-level results. They write private row scores and the fixed every-twentieth-row original-image gallery under `$S/independent_scores`. The public results contain aggregate statistics and provenance hashes. [EEG fit and audit](../research/eeg_stack_20261112/code/fit_dual_r80.py) · [Matched independent scoring](../research/eeg_stack_20261112/code/score_dual_r80_images.py)
