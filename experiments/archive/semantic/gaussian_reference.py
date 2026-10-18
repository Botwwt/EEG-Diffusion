"""Auditable Gaussian observation/reference kernels; not a learned EEG method.

The Gaussian posterior, OU kernel, and residual-score identities are classical.
This module isolates the candidate repeat-dependent diffusion geometry for tests.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Reference:
    mean: np.ndarray
    covariance: np.ndarray
    unique_trials: int

    def whiten(self, values: np.ndarray) -> np.ndarray:
        chol = np.linalg.cholesky(self.covariance)
        return np.linalg.solve(chol, (values-self.mean).T).T

    def unwhiten(self, values: np.ndarray) -> np.ndarray:
        return self.mean + values @ np.linalg.cholesky(self.covariance).T


def posterior_reference(observations, trial_ids, session_ids, operator, prior_mean,
                        prior_covariance, trial_noise, session_noise) -> Reference:
    """Integrate shared Gaussian session effects; never count copied trials twice.

    z_r = A y + b_session(r) + eta_r. Priors/noise must be fitted on training data.
    Distinct trials have independent eta; shared b induces correlated repeats.
    """
    values = np.asarray(observations, dtype=np.float64)
    operator, prior_mean, prior_covariance, trial_noise, session_noise = [
        np.asarray(x, dtype=np.float64) for x in
        [operator, prior_mean, prior_covariance, trial_noise, session_noise]]
    assert len(values) == len(trial_ids) == len(session_ids)
    assert values.ndim == 2 and np.isfinite(values).all()
    kept, known = [], {}
    for i, tid in enumerate(trial_ids):
        if tid in known:
            old = known[tid]
            if session_ids[old] != session_ids[i] or not np.array_equal(values[old], values[i]):
                raise ValueError('Same physical trial ID has inconsistent observations')
        else:
            known[tid] = i; kept.append(i)
    if not kept:
        return Reference(prior_mean.copy(), prior_covariance.copy(), 0)
    values = values[kept]
    sessions = [session_ids[i] for i in kept]
    count = len(values)
    same_session = np.equal.outer(sessions, sessions).astype(np.float64)
    covariance = np.kron(np.eye(count), trial_noise) + np.kron(same_session, session_noise)
    np.linalg.cholesky(trial_noise)
    design = np.tile(operator, (count, 1))
    # Kalman/Schur form avoids explicitly inverting a nearly singular precision.
    observation_covariance = design @ prior_covariance @ design.T + covariance
    gain = np.linalg.solve(observation_covariance, design @ prior_covariance).T
    mean = prior_mean + gain @ (values.reshape(-1)-design @ prior_mean)
    posterior_covariance = prior_covariance-gain @ design @ prior_covariance
    posterior_covariance = (posterior_covariance+posterior_covariance.T)*.5
    np.linalg.cholesky(posterior_covariance)
    return Reference(mean, posterior_covariance, count)


def forward_reference(clean, reference: Reference, alpha: float, noise):
    """x_t=m+alpha*(x_0-m)+sqrt(1-alpha²)*L*xi, xi~N(0,I)."""
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0,1]')
    return reference.mean + alpha*(clean-reference.mean) + (
        np.sqrt(1-alpha**2) * np.asarray(noise) @ np.linalg.cholesky(reference.covariance).T)


def residual_noise_target(whitened_noisy, noise, alpha: float):
    """epsilon residual: xi-sigma*u_t. Predict sigma*u_t+network_residual."""
    return np.asarray(noise)-np.sqrt(1-alpha**2)*np.asarray(whitened_noisy)


def stationary_reference_step(state, reference: Reference, retention: float, noise):
    """Exact reversible OU transition for the Gaussian reference, not Euler."""
    return forward_reference(state, reference, retention, noise)


def reverse_mean_whitened(noisy, residual_prediction, cumulative_alpha, step_beta):
    """DDPM mean in whitened coordinates with a residual epsilon parameterization.

    Uses the usual DDPM coefficient convention: cumulative_alpha=alpha_bar_t.
    Variance/initialization/discretization require a separately specified sampler.
    """
    sigma = np.sqrt(1-cumulative_alpha)
    if not (0 < sigma <= 1 and 0 < step_beta < 1):
        raise ValueError('Invalid DDPM coefficients')
    epsilon = sigma*np.asarray(noisy)+np.asarray(residual_prediction)
    return (np.asarray(noisy)-step_beta/sigma*epsilon)/np.sqrt(1-step_beta)
