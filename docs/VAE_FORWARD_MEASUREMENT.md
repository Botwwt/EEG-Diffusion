# Which VAE changes can repeated EEG measure?

Three of sixteen candidate VAE directions predicted repeatable EEG responses for sub-04. A correction restricted to those directions improved a support-photo proposal, but direct EEG-to-VAE regression recovered more information about the viewed photograph. The prespecified comparison therefore stopped before diffusion rendering. [Registered design](../research/eeg_stack_20261112/results/object_fidelity/vae_forward_measurement_design.json) · [Measured outcomes](../research/eeg_stack_20261112/results/object_fidelity/vae_forward_measurement.json)

The experiment partitioned 512 generic objects into 320 for fitting, 96 for selecting measurable directions and 96 for evaluation. Each object contributed ten photographs with four physical EEG presentations apiece. Within-object changes in the 4 × 8 × 8 VAE latent predicted 17 posterior EEG channels, averaged in 20 ms bins from 60 to 460 ms. The early and late two-presentation means provided separate observations. Three directions passed the multiplicity-adjusted calibration criterion. On the evaluation objects, the retained projections explained 10.7% of early-half variance, 8.9% of late-half variance and 13.8% of four-repeat variance. With the correct object supplied, EEG identified its own image on 647 of 960 individual two-photo choices and 344 of 480 joint assignments.

For each held-out photo, the proposal was the medoid of its object's eight allowed support photos. The forward update compared the candidate's predicted EEG with the observed EEG and applied 35% of the estimated latent correction. Lower mean squared error indicates a closer match to the original photo's VAE latent.

| Medoid proposal or correction | Mean squared error |
| --- | ---: |
| Unchanged medoid | 0.393709 |
| Forward, matched EEG | 0.390829 |
| Forward, zero EEG | 0.393038 |
| Forward, exchanged EEG | 0.393172 |
| Direct inverse, matched EEG | 0.324555 |
| Direct inverse, zero EEG | 0.331617 |

The EEG-specific gain over each method's zero-EEG control was **0.002209 forward versus 0.007061 direct**. Their paired difference was −0.004853, with a 95% interval of [−0.005963, −0.003803] over the 96 objects. An oracle given the original latent could remove 22.56% of the medoid error in the retained directions. The weak EEG update removed 0.73%. These numbers describe clean support-photo proposals. An intermediate diffusion latent and actual generated JPEGs would require a separate test.

## Replay

Run the three stages from the repository root with the archived private EEG and VAE inputs listed in the design. The prediction stage seals candidate corrections before evaluation reads the held-out latent targets. [Implementation](../research/eeg_stack_20261112/code/fit_vae_forward_measurement.py)

```sh
C=research/eeg_stack_20261112/code
export OPENBLAS_NUM_THREADS=4
python "$C/fit_vae_forward_measurement.py" fit
python "$C/fit_vae_forward_measurement.py" predict
python "$C/fit_vae_forward_measurement.py" evaluate
```
