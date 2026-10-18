import numpy as np
import pytest
from eeg_diffusion.gaussian_reference import posterior_reference


def reference(values, ids, sessions, session_noise=.5):
    return posterior_reference(np.asarray(values, dtype=float).reshape(-1,1),ids,sessions,
                               np.eye(1),np.zeros(1),np.eye(1),np.eye(1),session_noise*np.eye(1))


def test_duplicate_physical_trial_does_not_create_evidence():
    single = reference([.8], ['a'], [0])
    copied = reference([.8,.8], ['a','a'], [0,0])
    np.testing.assert_array_equal(single.mean, copied.mean)
    np.testing.assert_array_equal(single.covariance, copied.covariance)
    assert copied.unique_trials == 1
    with pytest.raises(ValueError):
        reference([.8,.7], ['a','a'], [0,0])


def test_correlated_repeats_are_less_precise_than_independent_repeats():
    same = reference([.8,.8], ['a','b'], [0,0])
    independent = reference([.8,.8], ['a','b'], [0,1])
    assert same.covariance.item() > independent.covariance.item()
    # Direct scalar conjugate calculation: session variance .5, two trials.
    np.testing.assert_allclose(same.covariance, [[.5]], atol=1e-12)
    np.testing.assert_allclose(same.mean, [.4], atol=1e-12)


def test_no_observations_returns_prior():
    r = reference([], [], [])
    np.testing.assert_array_equal(r.mean, [0.])
    np.testing.assert_array_equal(r.covariance, [[1.]])

