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

| Starting latent, fixed native conditioning | Intra10 hits | Intra10 Top-1 | Pairwise swap |
| --- | ---: | ---: | ---: |
| Direct raw-grid ridge | 27/200 | 13.50% | 61.56% |
| Context + global residual ridge | 27/200 | 13.50% | 61.78% |
| Within-concept correction on the raw grid | 27/200 | 13.50% | 61.00% |
| Context + within-concept correction on the residual | 28/200 | 14.00% | 62.67% |

The last arm corrected six previous misses and lost five previous hits. Relative to the direct model, the primary difference was **+0.50 percentage points** (95% concept-paired interval −2.50 to +3.50). Against the matched within-concept raw-grid correction it was **+0.50** (−3.00 to +4.00). Across the four preselected example photos, all arms made broadly similar objects, including generating a plate of food for a bagpipe. This experiment produced 800 reconstructions. Its uncertainty and narrow scope support retaining the existing default. The published JSON includes all 80 arm–concept scores and the 12 paired intervals. [Generation results](../research/eeg_stack_20261112/results/context_residual_sub03_generation.json)

With the aligned pilot data, run the first command on a CPU compute node and the second on one GPU. The renderer uses the legacy archive's fitted EEG encoder, pinned generator, CLIP weights and training image ZIP. It writes a contact sheet, per-photo ranks and per-concept scores.

```bash
python "$code/predict_instance_starts.py" --archive "$EEG_ARCHIVE" --prep "$WORK/prep" --out "$WORK/starts"
python "$code/eval_instance_starts.py" --archive "$EEG_ARCHIVE" \
  --starts "$WORK/starts/instance_starts.npz" --out "$WORK/render" --batch-size 8
```

[Readout fitting and held-out selection](../research/eeg_stack_20261112/code/predict_instance_starts.py) · [Generation and scoring](../research/eeg_stack_20261112/code/eval_instance_starts.py) · [Exact renderer used for the archived run](https://github.com/Botwwt/EEG-Diffusion/blob/2df65717d446e28e89057ac7c2ec8c0c88bd9486/research/eeg_stack_20261112/code/eval_instance_starts.py)

## Does photo-specific EEG agreement recover finer details?

A separate CPU experiment on the 264 evaluation concepts kept a common 16×16 VAE reference gallery. Each EEG prediction was split into a 4×4 low-frequency component and six oriented detail bands at 8×8 and 16×16. The candidate estimates whether to retain each band's detail from agreement between two EEG windows, 80–160 and 160–320 ms. The readouts, agreement thresholds and gain parameters used only the 1,059 training concepts. Four internal concept folds supplied out-of-fold calibration predictions. Earlier work in the repository had already studied repeat-based spatial reliability. [Earlier reliability controls](../research/eeg_stack_20261112/results/reliability_gain.txt)

| Detail estimator | Intra10 hits, same 16×16 gallery | Mean squared error |
| --- | ---: | ---: |
| 4×4 projection | 570/2,640 | 0.355882 |
| Direct 16×16 prediction | **581/2,640** | 0.357445 |
| One common gain | 572/2,640 | 0.355753 |
| Fixed gain by orientation and scale | 571/2,640 | 0.355717 |
| Gain from the two EEG windows for each photo | 571/2,640 | **0.355716** |
| Gate values drawn from the training distribution | 571/2,640 | 0.355718 |

The rule for advancing to image generation required higher identification and lower squared error than both direct prediction and fixed per-band gains, with a positive lower 95% bound for the identification gain over direct prediction. Photo-specific gating reduced squared error but lost ten hits against direct prediction, **−0.38 percentage points** (concept-paired 95% interval −0.87 to +0.11). It tied the static gate on identification. The GPU comparison above remains the only generation run for these ideas. [Complete evidence](../research/eeg_stack_20261112/results/spatial_detail_sub03_mean4.json) · [CPU screening code and decision rule](../research/eeg_stack_20261112/code/probe_spatial_detail.py)

```bash
python "$code/probe_spatial_detail.py" --prep "$WORK/prep" --out "$WORK/detail.json"
```
