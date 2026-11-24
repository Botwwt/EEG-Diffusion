# Paired EEG conditioning at a frozen denoiser

This sub-04 experiment tests whether the personal EEG–H14 readout points in directions that help SDXL-Turbo denoise the correct photograph. The fit changes the direction sent to the official ViT-H IP-Adapter. It preserves the EEG-predicted angle from the eight-photo support mean for each query. The known generic object name, supports, text conditioning, generator weights, diffusion timestep and Gaussian noise field are shared by the two photographs in each comparison. [Frozen design and photo-row registration](../research/eeg_stack_20261112/results/object_fidelity/ip_denoiser_gradient_design.json)

## Held-out result

The native EEG direction favored its own photograph over the exchanged photograph on 32 of 48 object pairs. The adapted head also favored 32. Its paired advantage over native was +0.000252 in noise MSE (95% interval −0.000748 to +0.001228). Against the reversed-sign control the difference was +0.000723 (−0.000345 to +0.001852). The adapted head increased absolute matched MSE by 0.001012 (+0.000026 to +0.001995). The prespecified gate stopped this candidate before target-panel generation. [Verified aggregate and bootstrap intervals](../research/eeg_stack_20261112/results/object_fidelity/ip_denoiser_gradient.json)

| Conditioning direction | Correct denoiser pairs / 48 | Mean swapped minus matched noise MSE | Matched noise MSE |
|---|---:|---:|---:|
| Frozen EEG–H14 | 32 | +0.000952 | 0.103011 |
| Gradient adaptation | 32 | +0.001203 | 0.104023 |
| Reversed-sign training control | 29 | +0.000480 | 0.103697 |
| Evaluation-only true image tangent | 47 | +0.003498 | 0.100251 |

The native EEG directions paired correctly with the original image embeddings in 33 of 48 cases, compared with 31 for the adaptation and 29 for the control. The native EEG denoising margin had a positive interval (+0.000368 to +0.001574), and the true-image control paired correctly on 47 of 48 cases. Thus the native denoiser already responds to some photograph-specific signal under these conditions. Exact noise prediction on a true image latent and a completed image sampled from noise are different endpoints. [Generic results](../research/eeg_stack_20261112/results/object_fidelity/ip_denoiser_gradient.json)

## What is fitted

Of the 512 generic calibration objects, 415 have all ten scaled 64 × 64 VAE modes in historical T. The design chooses 128 objects for the adaptation head and 48 for evaluation. Each object provides one pair and eight supports. The 48 evaluation objects occur in neither the original 128-object hyperparameter experiment nor the IP-Adapter radius calibration. They are excluded from every EEG–H14 coefficient fit. Four object-disjoint fits provide the adaptation examples. The test readout uses the other 464 generic classes. [Personal H14 calibration and model replay](PHOTO_STRUCTURE.md#replay)

For each training pair, the denoiser predicts noise twice at the same support-mean condition. Taking the negative gradient of the difference between the two mean squared noise errors gives a direction in which the first photograph's error decreases relative to the second's. The design projects that direction onto the H14 unit-sphere tangent. A rank-16 ridge maps the difference in the two EEG tangents to the unit teacher direction. At inference the learned tangent contributes weight 0.5 alongside the original unit tangent at weight 1. Each query can be evaluated independently with its object prior.

The original SwapDiff method differentiates the direct matched-versus-swapped loss. D2 optimizes four-step reconstruction with within-object competition among four images. Here the teacher differentiates at one shared conditioning point, and the held-out readout measures the actual nonlinear response at each query's predicted condition. A control reverses the teacher signs of 64 preselected training pairs. An evaluation-only image-embedding control replaces the tangent with the true photograph's H14 tangent while preserving its paired EEG angle. [SwapDiff objective](../research/eeg_stack_20261112/code/exp8_trial_precision_swapdiff.py#L388) · [Four-step D2 objective](../research/eeg_stack_20261112/code/exp6_draft_intra4.py#L9)

## Replay

Start from the existing 512-class sub-04 EEG feature receipts, official training H14 features and row dictionary, historical T VAE registration and completion manifest, within-photo model selection and calibrated angle/radius records. The preparation script's defaults point to their earlier replay locations. From the repository root run

```bash
C=research/eeg_stack_20261112/code
W=outputs/ip-denoiser-gradient
OPENBLAS_NUM_THREADS=4 python "$C/prepare_ip_denoiser_gradient.py"
python "$C/fit_ip_denoiser_gradient.py" --root "$W" --check-only
```

On the archive GPU host, use one A40 and four CPUs. The measured run took 127 seconds and reserved 8.3 GiB GPU memory at peak. Export `ARCHIVE` to the historical archive root. The wrapper uses the archive's offline environment with Torch 2.3.0 and Diffusers 0.36.0, native SDXL-Turbo and the official adapter in `official/ip-adapter_sdxl_vit-h.safetensors`. The driver verifies SHA-256 hashes for the VAE mode tensor, row registration, generator and IP-Adapter before model execution. [Slurm wrapper](../research/eeg_stack_20261112/jobs/run_ip_denoiser_gradient.sbatch)

```bash
# In a local staging directory, with W and C resolved to repository paths
mkdir -p ip_grad
cp "$W/design.json" "$W/inputs.json" "$W/generic_pairs.npz" "$C/fit_ip_denoiser_gradient.py" ip_grad/
tar -cf inputs.tar ip_grad
sha256sum inputs.tar
# Copy inputs.tar and the repository's current jobs/run_ip_denoiser_gradient.sbatch to the GPU submission directory.
# On the login node, verify the archive transfer against the digest printed above, then submit.
sbatch run_ip_denoiser_gradient.sbatch inputs.tar VERIFIED_ARCHIVE_SHA256
# After retrieving ip_grad/results.tar into the local experiment directory
tar -xf "$W/results.tar" -C "$W"
python "$C/summarize_ip_denoiser_gradient.py" --root "$W" --result "$W/out" \
  --out research/eeg_stack_20261112/results/object_fidelity/ip_denoiser_gradient.json
```

The primary contrast is swapped minus matched noise MSE, averaged across both photographs for each object. The two prespecified comparisons require positive lower 95% paired object-bootstrap bounds against both the native and sign-reversed heads, no increase in absolute matched MSE against native, and positive mean contrast for the image-tangent control. The equal-condition center has zero contrast by construction. All 48 objects contribute once. A separate descriptive calculation compares the same EEG conditions to the two true photographs in H14 space. These assays concern denoising at two timesteps on noised image modes. A generated-image claim would require independent scoring of complete diffusion samples.

The first submission encountered NUL bytes in the driver extracted on the shared filesystem. The input tar on the login node remained intact. The completed job copied the verified archive into node-local storage and extracted it there. The replay wrapper follows that procedure, checks the driver and design hashes and verifies the result archive as it is copied back. The recorded input archive SHA-256 was `da1cd2a4789fd688e7fbfd821e73e9fbaad6add6702308e299cb0e13c371b6f4`. Repacked archives can have different container hashes. The design and numeric input signatures for the measured run are recorded in the result.
