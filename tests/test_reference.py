import numpy as np
import pytest

from eeg_diffusion.gaussian_reference import (
    Reference,
    forward_reference,
    posterior_reference,
    residual_noise_target,
    reverse_mean_whitened,
)


def reference(values, ids, sessions, session_noise=0.5):
    return posterior_reference(
        np.asarray(values, dtype=float).reshape(-1, 1),
        ids,
        sessions,
        np.eye(1),
        np.zeros(1),
        np.eye(1),
        np.eye(1),
        session_noise * np.eye(1),
    )


def test_duplicate_physical_trial_does_not_create_evidence():
    single = reference([0.8], ["a"], [0])
    copied = reference([0.8, 0.8], ["a", "a"], [0, 0])
    np.testing.assert_array_equal(single.mean, copied.mean)
    np.testing.assert_array_equal(single.covariance, copied.covariance)
    assert copied.unique_trials == 1
    with pytest.raises(ValueError):
        reference([0.8, 0.7], ["a", "a"], [0, 0])


def test_correlated_repeats_are_less_precise_than_independent_repeats():
    same = reference([0.8, 0.8], ["a", "b"], [0, 0])
    independent = reference([0.8, 0.8], ["a", "b"], [0, 1])
    assert same.covariance.item() > independent.covariance.item()
    # Direct scalar conjugate calculation: session variance .5, two trials.
    np.testing.assert_allclose(same.covariance, [[0.5]], atol=1e-12)
    np.testing.assert_allclose(same.mean, [0.4], atol=1e-12)


def test_no_observations_returns_prior():
    r = reference([], [], [])
    np.testing.assert_array_equal(r.mean, [0.0])
    np.testing.assert_array_equal(r.covariance, [[1.0]])


def test_independent_scalar_repeats_match_conjugate_solution():
    # Prior variance 2, observation variance 3: precision = 1/2 + 2/3.
    r = posterior_reference(
        [[1.0], [2.0]], ["a", "b"], [0, 1], [[1.0]], [0.0], [[2.0]], [[3.0]], [[0.0]]
    )
    np.testing.assert_allclose(r.covariance, [[6 / 7]])
    np.testing.assert_allclose(r.mean, [6 / 7])


def test_reference_owns_its_buffers_and_preserves_round_trip():
    mean = np.array([1.0, -1.0])
    covariance = np.array([[2.0, 0.4], [0.4, 1.0]])
    r = Reference(mean, covariance, 0)
    mean[:] = 0
    covariance[:] = 0
    values = np.array([[0.2, 0.7], [1.0, 2.0]])
    np.testing.assert_allclose(r.unwhiten(r.whiten(values)), values, atol=1e-12)
    np.testing.assert_array_equal(r.mean, [1.0, -1.0])
    with pytest.raises(ValueError):
        r.covariance[0, 0] = 1


def test_forward_endpoints_and_residual_identity():
    r = Reference(np.array([0.3]), np.array([[2.0]]), 0)
    clean, noise = np.array([[0.7], [1.2]]), np.array([[-0.5], [0.8]])
    np.testing.assert_allclose(forward_reference(clean, r, 1, noise), clean)
    np.testing.assert_allclose(forward_reference(clean, r, 0, noise), r.unwhiten(noise))
    alpha = 0.4
    noisy = r.whiten(forward_reference(clean, r, alpha, noise))
    residual = residual_noise_target(noisy, noise, alpha)
    np.testing.assert_allclose(np.sqrt(1 - alpha**2) * noisy + residual, noise)
    # With a stationary Gaussian score, the conventional DDPM mean contracts.
    prediction = reverse_mean_whitened(noisy, np.zeros_like(noisy), alpha**2, 0.02)
    np.testing.assert_allclose(prediction, np.sqrt(0.98) * noisy)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_retention_is_rejected_before_numerical_operations(alpha):
    r = Reference(np.zeros(1), np.eye(1), 0)
    with pytest.raises(ValueError):
        forward_reference([[0]], r, alpha, [[0]])


@pytest.mark.parametrize("covariance", [[[0.0]], [[-1.0]], [[np.nan]], [[1.0, 0.0]]])
def test_invalid_covariance_is_rejected(covariance):
    with pytest.raises(ValueError):
        Reference(np.zeros(1), np.asarray(covariance), 0)


def test_high_precision_reference_remains_positive():
    r = posterior_reference([[1.0]], ["a"], [0], [[1.0]], [0.0], [[1.0]], [[1e-14]], [[0.0]])
    assert r.covariance.item() > 0
    np.testing.assert_allclose(r.covariance, [[1e-14]], rtol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_refactor_agrees_with_frozen_historical_gaussian_kernel(seed):
    import importlib.util
    import sys
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1] / "experiments/archive/semantic/gaussian_reference.py"
    )
    spec = importlib.util.spec_from_file_location("frozen_gaussian_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, 2))
    p = rng.normal(size=(2, 2))
    prior = p @ p.T + np.eye(2)
    arguments = (
        rng.normal(size=(4, 3)),
        ["a", "b", "c", "d"],
        [0, 0, 1, 1],
        a,
        rng.normal(size=2),
        prior,
        np.eye(3),
        0.3 * np.eye(3),
    )
    old = module.posterior_reference(*arguments)
    new = posterior_reference(*arguments)
    np.testing.assert_allclose(new.mean, old.mean, atol=1e-10, rtol=1e-11)
    np.testing.assert_allclose(new.covariance, old.covariance, atol=1e-10, rtol=1e-11)
