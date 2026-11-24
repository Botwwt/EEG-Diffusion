# Can EEG recover what a photograph’s context fails to predict?

For a flower viewed from above, its surrounding petals make much of its center predictable. An unusual dark patch at that location contributes a signed difference between the observed visual features and the features predicted from the surroundings. The question here is whether EEG identifies that photo-specific difference and whether predicting it explicitly improves reconstruction.

## Why use a local vector?

Subtracting a global mean error from each candidate leaves their order unchanged. In a balanced SwapDiff contrast, any additive difficulty term attached to each photograph also cancels. The existing evaluator already reports row/column-centered identification. A local vector preserves both the location and the direction of a deviation. I-JEPA predicts masked-region representations, and S-JEPA has visualized local prediction errors. These precedents motivate testing their connection to paired EEG. [I-JEPA](https://arxiv.org/abs/2301.08243) · [S-JEPA](https://arxiv.org/abs/2611.09302) · [Existing centered metrics](../research/eeg_stack_20261112/code/exp0_intra10_swap_gate.py)

## Pilot on real EEG

The legacy sub-03 archive contains a four-trial voltage mean per photo, 63 channels and stimulus identities. A checked join on original row and stimulus IDs aligned 13,230 training-set photos with SDXL VAE modes. The pilot selected the 17 posterior channels, used 10-sample temporal bins, and applied the fixed concept split to these available photos. It fitted on 1,059 concepts and evaluated on 264 different concepts with ten photos each. The earlier R4 generation benchmark uses a different set of photos, a different readout and a generated-image endpoint.

The visual predictor divides each VAE latent into a 4×4 grid and predicts each cell from its four neighbors using a ridge model with linear and tanh features. It sees training-fold photos during fitting. On evaluation photos, neighboring cells explain 72.51% of the variance relative to the training mean. This supplies a cheap visual-context proxy. A pretrained I-JEPA network would define a different target.

The raw EEG readout predicts the grid directly. The second readout predicts the signed residual. For comparison in raw space, we add that residual to the visual context predicted from the EEG-only raw readout. For comparison in residual space, we apply the same context subtraction to the raw readout. The within-concept shuffled-training control uses the same EEG and residuals with the photo identities deranged.

| Target space, ten-photo retrieval | Direct EEG readout | Explicit residual readout | Residual training targets shuffled within concept |
| --- | ---: | ---: | ---: |
| Raw VAE grid | 575/2,640 (21.78%) | 576/2,640 (21.82%) | 556/2,640 (21.06%) |
| Signed local residual | 413/2,640 (15.64%) | 423/2,640 (16.02%) | 289/2,640 (10.95%) |

In residual space, shuffling EEG at evaluation gives 10.23% for the explicitly trained readout. Its advantage over the direct model in raw space is 0.04 percentage points (95% interval −0.34 to +0.42). In residual space the corresponding advantage is 0.38 points (−0.08 to +0.83). Intervals resample the 264 paired concepts 10,000 times with seed 20261124. They describe uncertainty over this participant’s stimulus concepts. The difference from the shuffled-training control supports a photo-specific EEG signal. Only one extra correct raw-grid match gives little reason to change the generation model. [Scores by concept](../research/eeg_stack_20261112/results/context_residual_sub03_mean4.json)

### Reproduce the probe

On a compute node with the legacy native sub-03 archive, the preparation step requires NumPy and PyTorch. The probe requires NumPy. The preparation directory contains the selected EEG and an aligned view of the VAE archive. The two commands below reproduced the pilot on four Slurm CPU cores.

```bash
code=research/eeg_stack_20261112/code
python "$code/prepare_mean4_probe.py" --archive "$EEG_ARCHIVE" --out "$WORK/prep"
python "$code/probe_context_residual.py" --prep "$WORK/prep" --subjects 3 \
  --eeg-input mean4 --protocol archive_sub03_mean4_T --out "$WORK/probe.json"
```

A workspace with separate four-trial arrays can use `--eeg-input trials4`. This probe takes their voltage mean. The M1a + D2 generation evaluator uses a separate input path. [Preparation and row checks](../research/eeg_stack_20261112/code/prepare_mean4_probe.py) · [Probe implementation](../research/eeg_stack_20261112/code/probe_context_residual.py)

## Follow-up generation comparison

The follow-up changes the starting latent while holding the participant's Brain-HIVE conditioning tokens, generator, four steps, candidate count and noise fixed across arms. Train on all 1,323 concepts in the legacy T partition. Select 20 complete concepts from outside T with seed 20261126 and use all ten photos of each. These photos occur outside the CPU pilot. The participant and dataset have appeared in previous research.

Four EEG readouts supply the starting grids. The two controls are a direct visual-grid readout and the explicit context-residual readout from the first pilot. The other two emphasize the within-concept visual differences during fitting, using a fixed penalty toward the original ridge head. Fit one to raw targets and one to signed residual targets. Test EEG is used individually. Concept labels group training examples and define the evaluation gallery and shared noise.

The primary comparison is the within-concept residual head versus the original raw head on OpenAI CLIP ViT-L/14 ten-photo Top-1 of actual generated images. To attribute an advantage to context residuals, compare the within-concept residual head with the corresponding within-concept raw head. Use the original stimulus RGB as reference, report all four arms and paired concept intervals, and retain a pairwise swap metric as secondary evidence. This native sub-03 pipeline has a different EEG checkpoint and starting-latent resolution from M1a + D2.
