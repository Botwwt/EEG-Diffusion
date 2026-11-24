# Where does the viewed photo's identity get lost?

On the third object panel, the selected reconstruction identified 14 of the 120 viewed photos for sub-04 in a DINOv2 gallery of ten photos of the same object. Replacing the predicted visual condition with the embedding of the actual viewed photo raised that count to 74. Replacing only the predicted object word gave 13. This diagnostic used a previously scored panel to locate the bottleneck. [Numerical results, stimulus registration and condition hashes](../research/eeg_stack_20261112/results/object_fidelity/photo_condition_diagnostic.json)

| Prompt supplied to SDXL-Turbo | ViT-H/14 visual condition | Viewed photo first of ten | Mean RGB error at 32×32 ↓ | Mean luminance correlation at 32×32 ↑ |
| --- | --- | ---: | ---: | ---: |
| EEG word | EEG estimate using eight calibration photos | 14/120 | 0.2862 | 0.0577 |
| True object word | Same EEG estimate | 13/120 | 0.2850 | 0.0613 |
| Same EEG word | Exact viewed-photo embedding | 74/120 | 0.2567 | 0.1410 |
| True object word | Exact viewed-photo embedding | 72/120 | 0.2568 | 0.1440 |

Only the first row is usable at inference. Each row uses the same photos, four-step generator, per-photo noise seed, projection radius and output size. The scorer hashes the saved JPEGs and replays the EEG-only ten-way result for every individual stimulus. Supplying the exact visual feature with the EEG word improved identification by 50.0 percentage points (paired 95% interval +38.3 to +61.7, resampling the twelve objects). Even among the 85 photos whose EEG word was already correct, the EEG estimate recovered 5 and the exact visual feature recovered 49. The experiment gives a reason to improve the photo-specific EEG readout. It cannot tell us how much of the missing visual information exists in four EEG trials, and the generator still missed 46 photos with the correct visual embedding. [Scorer and generation driver](../research/eeg_stack_20261112/code/diagnose_photo_conditioning.py)

## Reproduce the intervention

First run the [third-panel reconstruction](FRESH_PHOTO_REPLAY.md) and its [saved-JPEG evaluation](FRESH_PHOTO_VALIDATION.md#reproduce-the-saved-image-comparison). Use the EEG analysis environment to prepare the frozen input, a compute GPU with the pinned offline generator to render, and the DINOv2 environment to score. Paths default to the project output directories. Supply the generator archive, dictionary and adapter from the reconstruction guide.

```sh
C=research/eeg_stack_20261112/code
python "$C/diagnose_photo_conditioning.py" prepare
python "$C/diagnose_photo_conditioning.py" render --archive "$ARCHIVE" \
  --dictionary "$DICTIONARY" --adapter "$ADAPTER"
python "$C/diagnose_photo_conditioning.py" score --release outputs/photo-conditioning-diagnostic/report.json
```
