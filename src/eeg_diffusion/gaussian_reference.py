"""Classical Gaussian observation kernels with shared session effects.

This typed numerical reference is a release-time refactor. The executed
historical implementation remains unchanged in experiments/archive/semantic/.
No learned EEG method or image-quality guarantee is implied by these identities.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _array(value: ArrayLike, name: str) -> FloatArray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _covariance(value: ArrayLike, size: int, name: str, *, positive: bool) -> FloatArray:
    result = _array(value, name)
    if result.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    scale = max(1.0, float(np.max(np.abs(result))))
    if not np.allclose(result, result.T, atol=1e-12 * scale, rtol=0):
        raise ValueError(f"{name} must be symmetric")
    result = (result + result.T) / 2
    if positive:
        try:
            np.linalg.cholesky(result)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{name} must be positive definite") from error
    elif np.linalg.eigvalsh(result).min() < -1e-12 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return result


def _points(value: ArrayLike, dimension: int, name: str) -> FloatArray:
    result = _array(value, name)
    if result.ndim not in (1, 2) or result.shape[-1] != dimension:
        raise ValueError(f"{name} must have shape ({dimension},) or (n, {dimension})")
    return result


def _retention(value: float) -> float:
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Retention must be finite and in [0, 1]")
    return float(value)


def _solve_spd(matrix: FloatArray, rhs: FloatArray) -> FloatArray:
    factor = np.linalg.cholesky(matrix)
    return np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))


@dataclass(frozen=True)
class Reference:
    """Immutable Gaussian moments and count of unique physical observations.

    Parameters
    ----------
    mean : array-like, shape (d,)
        Conditional mean in the original feature coordinates.
    covariance : array-like, shape (d, d)
        Symmetric positive-definite conditional covariance.
    unique_trials : int
        Count after deduplicating physical trial identifiers.
    """

    mean: FloatArray
    covariance: FloatArray
    unique_trials: int

    def __post_init__(self) -> None:
        mean = _array(self.mean, "mean").copy()
        if mean.ndim != 1 or not len(mean):
            raise ValueError("mean must be a nonempty vector")
        covariance = _covariance(self.covariance, len(mean), "covariance", positive=True).copy()
        if not isinstance(self.unique_trials, int) or self.unique_trials < 0:
            raise ValueError("unique_trials must be a nonnegative integer")
        mean.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

    def whiten(self, values: ArrayLike) -> FloatArray:
        """Transform a vector or row-batch into unit-covariance coordinates."""
        points = _points(values, len(self.mean), "values")
        factor = np.linalg.cholesky(self.covariance)
        return np.linalg.solve(factor, (points - self.mean).T).T

    def unwhiten(self, values: ArrayLike) -> FloatArray:
        """Transform unit-covariance coordinates back to the original space."""
        points = _points(values, len(self.mean), "values")
        return self.mean + points @ np.linalg.cholesky(self.covariance).T


def posterior_reference(
    observations: ArrayLike,
    trial_ids: Sequence[Hashable],
    session_ids: Sequence[Hashable],
    operator: ArrayLike,
    prior_mean: ArrayLike,
    prior_covariance: ArrayLike,
    trial_noise: ArrayLike,
    session_noise: ArrayLike,
) -> Reference:
    """Condition on correlated repeats without counting copied trials twice.

    The model is ``z_r = A y + b_session(r) + eta_r``. Session effects are
    independent across session IDs and shared within a session; trial noise is
    independent across unique trials. Fit all parameters on training data.

    Shapes: observations (R, m), operator (m, d), prior_mean (d,), prior_covariance
    (d, d), and both noise covariances (m, m). Trial noise and prior covariance
    must be positive definite; session noise may be zero. A repeated physical
    trial ID must have identical values and session membership. An empty (0, m)
    observation matrix returns the prior.

    Cholesky precision solves avoid subtraction of nearly equal covariances.
    Memory for the observation covariance is O((R m)^2).
    """
    values = _array(observations, "observations")
    design_one = _array(operator, "operator")
    mean = _array(prior_mean, "prior_mean")
    if mean.ndim != 1 or not len(mean):
        raise ValueError("prior_mean must be a nonempty vector")
    if design_one.ndim != 2 or design_one.shape[1] != len(mean) or not len(design_one):
        raise ValueError("operator must have shape (m, d), with m > 0")
    dimension, channels = len(mean), len(design_one)
    if values.ndim != 2 or values.shape[1] != channels:
        raise ValueError("observations must have shape (R, m)")
    if len(values) != len(trial_ids) or len(values) != len(session_ids):
        raise ValueError("Trial and session IDs must match the observation count")
    prior = _covariance(prior_covariance, dimension, "prior_covariance", positive=True)
    independent = _covariance(trial_noise, channels, "trial_noise", positive=True)
    shared = _covariance(session_noise, channels, "session_noise", positive=False)

    kept: list[int] = []
    known: dict[Hashable, int] = {}
    for index, trial_id in enumerate(trial_ids):
        if not isinstance(trial_id, Hashable) or not isinstance(session_ids[index], Hashable):
            raise ValueError("Physical trial and session IDs must be hashable")
        if trial_id in known:
            previous = known[trial_id]
            if session_ids[previous] != session_ids[index] or not np.array_equal(
                values[previous], values[index]
            ):
                raise ValueError("Same physical trial ID has inconsistent observations")
        else:
            known[trial_id] = index
            kept.append(index)
    if not kept:
        return Reference(mean, prior, 0)

    values = values[kept]
    sessions = [session_ids[index] for index in kept]
    same_session = np.array([[a == b for b in sessions] for a in sessions], dtype=float)
    noise = np.kron(np.eye(len(kept)), independent) + np.kron(same_session, shared)
    design = np.tile(design_one, (len(kept), 1))
    precision = _solve_spd(prior, np.eye(dimension)) + design.T @ _solve_spd(noise, design)
    covariance = _solve_spd(precision, np.eye(dimension))
    covariance = (covariance + covariance.T) / 2
    residual = values.reshape(-1) - design @ mean
    posterior_mean = mean + covariance @ design.T @ _solve_spd(noise, residual)
    return Reference(posterior_mean, covariance, len(kept))


def forward_reference(
    clean: ArrayLike, reference: Reference, alpha: float, noise: ArrayLike
) -> FloatArray:
    """Apply ``m + alpha (x - m) + sqrt(1 - alpha²) L noise``.

    The caller supplies independent unit Gaussian noise. Array checks cannot
    infer whether supplied values were sampled from the correct distribution.
    """
    alpha = _retention(alpha)
    clean_array = _points(clean, len(reference.mean), "clean")
    noise_array = _points(noise, len(reference.mean), "noise")
    if clean_array.shape != noise_array.shape:
        raise ValueError("clean and noise must have identical shapes")
    return (
        reference.mean
        + alpha * (clean_array - reference.mean)
        + np.sqrt(1 - alpha**2) * noise_array @ np.linalg.cholesky(reference.covariance).T
    )


def residual_noise_target(whitened_noisy: ArrayLike, noise: ArrayLike, alpha: float) -> FloatArray:
    """Return the residual epsilon target ``noise - sqrt(1 - alpha²) u_t``."""
    alpha = _retention(alpha)
    noisy_array = _array(whitened_noisy, "whitened_noisy")
    noise_array = _array(noise, "noise")
    if noisy_array.shape != noise_array.shape:
        raise ValueError("whitened_noisy and noise must have identical shapes")
    return noise_array - np.sqrt(1 - alpha**2) * noisy_array


def stationary_reference_step(
    state: ArrayLike, reference: Reference, retention: float, noise: ArrayLike
) -> FloatArray:
    """Apply an exact Gaussian OU transition; retention is not an Euler step."""
    return forward_reference(state, reference, retention, noise)


def reverse_mean_whitened(
    noisy: ArrayLike,
    residual_prediction: ArrayLike,
    cumulative_alpha: float,
    step_beta: float,
) -> FloatArray:
    """Compute the conventional DDPM mean from a residual epsilon prediction.

    ``cumulative_alpha`` is alpha-bar; ``forward_reference`` instead uses its
    square root as alpha. This function specifies no sampling variance.
    """
    if not np.isfinite(cumulative_alpha) or not 0 <= cumulative_alpha < 1:
        raise ValueError("cumulative_alpha must be finite and in [0, 1)")
    if not np.isfinite(step_beta) or not 0 < step_beta < 1:
        raise ValueError("step_beta must be finite and in (0, 1)")
    noisy_array = _array(noisy, "noisy")
    residual_array = _array(residual_prediction, "residual_prediction")
    if noisy_array.shape != residual_array.shape:
        raise ValueError("noisy and residual_prediction must have identical shapes")
    sigma = np.sqrt(1 - cumulative_alpha)
    epsilon = sigma * noisy_array + residual_array
    return (noisy_array - step_beta / sigma * epsilon) / np.sqrt(1 - step_beta)
