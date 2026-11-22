# Numerical reference API

The public package provides classical reference calculations and evaluation utilities. It does not load an EEG encoder or run image diffusion. Historical methods remain in `experiments/archive/`.

## Gaussian observations

The model is

\[
y\sim\mathcal N(m,P),\qquad z_r=Ay+b_{s(r)}+\eta_r,
\]

with independent session effects \(b_s\sim\mathcal N(0,B)\) and unique-trial errors \(\eta_r\sim\mathcal N(0,R)\). The stacked observation covariance has blocks \(\delta_{ij}R+\mathbf1_{s(i)=s(j)}B\). Correlated observations are conditioned jointly.

```python
import numpy as np
from eeg_diffusion.gaussian_reference import posterior_reference

reference = posterior_reference(
    observations=np.array([[0.8], [0.8]]),
    trial_ids=["trial-a", "trial-b"],
    session_ids=["session-1", "session-1"],
    operator=np.eye(1),
    prior_mean=np.zeros(1),
    prior_covariance=np.eye(1),
    trial_noise=np.eye(1),
    session_noise=0.5 * np.eye(1),
)
print(reference.mean)        # [0.4]
print(reference.covariance)  # [[0.5]]
```

| Argument | Shape | Constraint |
|---|---|---|
| `observations` | `(R, m)` | Finite real array; `(0, m)` is allowed |
| `trial_ids`, `session_ids` | Length `R` | Hashable identifiers; trial IDs identify physical measurements |
| `operator` | `(m, d)` | Statistical feature observation operator |
| `prior_mean` | `(d,)` | Finite vector |
| `prior_covariance` | `(d, d)` | Symmetric positive definite |
| `trial_noise` | `(m, m)` | Symmetric positive definite |
| `session_noise` | `(m, m)` | Symmetric positive semidefinite; zero allowed |

Copied physical trials do not increase precision. Inconsistent duplicates raise `ValueError`. Shared-session observations remain correlated. Returned moments own read-only buffers. Cholesky precision solves avoid explicitly subtracting nearly equal covariance matrices. The dense observation covariance requires O((R m)²) memory; this is intended for small reference problems.

All parameters must be fitted within the declared training boundary. A numerically valid covariance is not evidence that the EEG observation model is calibrated.

## Forward and reverse reference identities

`forward_reference(clean, reference, alpha, noise)` evaluates

\[
x_t=m+\alpha(x_0-m)+\sqrt{1-\alpha^2}L\epsilon,\quad LL^T=C.
\]

Inputs have equal shape `(d,)` or `(n, d)` and finite values; `alpha` lies in `[0, 1]`. The caller supplies independent standard Gaussian noise. `stationary_reference_step` applies the same exact OU kernel to a current state.

`residual_noise_target` gives \(\epsilon-\sqrt{1-\alpha^2}u_t\). `reverse_mean_whitened` evaluates the conventional DDPM mean from that residual. Its `cumulative_alpha` argument is **alpha-bar**, whereas the forward kernel's `alpha` is its square root. The reverse helper specifies neither variance nor a complete sampler.

## Metrics

`retrieval_top1(similarities, targets)` returns one value per query. The score matrix is `(queries, gallery)`; targets are integer gallery indices. Top-1 ties use the first gallery index, matching NumPy `argmax`.

`image_two_way(similarities, targets)` compares each matched score against every non-target gallery score. Ties receive half credit. This pairwise metric differs from full-gallery Top-1. The supplied similarities must originate from the declared evaluation modality; the function cannot tell whether they represent EEG vectors or generated pictures.

`paired_query_bootstrap(treatment, control, repetitions=20000, seed=0)` resamples the paired difference by query. Repeats and generation seeds must first be grouped within a query. The gallery remains fixed; this is not the joint query/distractor concept bootstrap used by some historical studies. The returned interval is conditional on the subjects, fitted models and gallery. It does not turn reused development data into independent confirmation.

## Integrity and CLI

`verify_release(root)` checks both the frozen-source manifest and hashes of released aggregate evidence. It then checks summary consistency and displayed arithmetic. Path traversal, duplicate manifest entries, missing files and changed hashes are explicit failures. No historical confidence interval is reconstructed from aggregates.

```bash
python -m eeg_diffusion verify --root /path/to/EEG-Diffusion
python -m eeg_diffusion reference-demo --out outputs/synthetic-reference.json
python -m eeg_diffusion verify-stack
python -m eeg_diffusion check-o13-inputs --rad-root /path/to/rad
```

The demo reports mean `[0.4]`, covariance `[[0.5]]` and `unique_trials: 2` from synthetic data. `verify-stack` derives integer hit counts from rounded November participant rows and checks them against the published totals. Its paired t intervals use ten participant-level differences and keep the gallery and fitted models fixed.

The O13 preflight checks paths and NumPy array shapes and emits structured results. Success returns exit code 0. Invalid artifacts or missing inputs return 2. The `eeg-diffusion` console command remains available after installation.
