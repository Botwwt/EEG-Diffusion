"""Exact algebra + Monte Carlo checks; all measurements here are synthetic."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from gaussian_reference import (posterior_reference, forward_reference,
                                residual_noise_target, stationary_reference_step,
                                reverse_mean_whitened)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--samples', type=int, default=100000)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); rng = np.random.default_rng(2026101501)
    prior_cov = np.array([[2., .4], [.4, 1.]])
    prior_mean = np.array([.3, -.2])
    operator = np.array([[1., .2], [.3, 1.]])
    trial_cov = np.array([[1.2, .1], [.1, .8]])
    session_cov = .3*np.eye(2)
    observations = np.array([[.4, .2], [.2, -.3], [.5, .4], [.3, -.1]])
    sessions, physical_ids = [0, 1, 0, 1], ['a', 'b', 'c', 'd']

    def reference(n, observations=observations, sessions=sessions, physical_ids=physical_ids):
        return posterior_reference(observations[:n], physical_ids[:n], sessions[:n],
            operator, prior_mean, prior_cov, trial_cov, session_cov)

    refs = [reference(n) for n in [0, 1, 2, 4]]
    for left, right in zip(refs, refs[1:]):
        assert np.linalg.eigvalsh(left.covariance-right.covariance).min() >= -1e-12
    assert np.linalg.eigvalsh(refs[1].covariance).min() > 0
    missing = refs[0]
    assert np.array_equal(missing.mean, prior_mean) and np.array_equal(missing.covariance, prior_cov)
    duplicated = reference(2, observations=observations[[0, 0]], sessions=[0, 0], physical_ids=['a', 'a'])
    assert np.array_equal(duplicated.mean, refs[1].mean)
    assert np.array_equal(duplicated.covariance, refs[1].covariance)
    try:
        reference(2, observations=observations[:2], sessions=[0, 0], physical_ids=['a', 'a'])
        raise AssertionError('Inconsistent duplicate was accepted')
    except ValueError:
        pass
    correct_correlated = reference(4, sessions=[0, 0, 0, 0])
    incorrect_independent = reference(4, sessions=[0, 1, 2, 3])
    assert np.linalg.eigvalsh(correct_correlated.covariance-incorrect_independent.covariance).min() > 0
    ref = refs[-1]
    n = args.samples
    clean_u, noise = rng.normal(size=(n, 2)), rng.normal(size=(n, 2))
    clean = ref.unwhiten(clean_u)
    assert np.allclose(ref.whiten(clean), clean_u, atol=1e-12)
    assert np.allclose(forward_reference(clean, ref, 1., noise), clean)
    assert np.allclose(forward_reference(clean, ref, 0., noise), ref.unwhiten(noise))
    alpha = .3; sigma = np.sqrt(1-alpha**2)
    noisy = forward_reference(clean, ref, alpha, noise)
    noisy_u = ref.whiten(noisy)
    assert np.allclose(noisy_u, alpha*clean_u+sigma*noise, atol=1e-12)
    residual = residual_noise_target(noisy_u, noise, alpha)
    original_energy = float(np.mean(np.sum(noise**2, axis=1)))
    residual_energy = float(np.mean(np.sum(residual**2, axis=1)))
    assert abs(residual_energy/(alpha**2*2)-1) < .015
    assert residual_energy < original_energy
    # For any predicted residual, translating back gives identical MSE samples.
    prediction = rng.normal(size=(n, 2))
    assert np.allclose(np.sum((prediction-residual)**2, axis=1),
        np.sum((sigma*noisy_u+prediction-noise)**2, axis=1))
    beta = .02
    mean = reverse_mean_whitened(noisy_u, np.zeros_like(noisy_u), alpha**2, beta)
    assert np.allclose(mean, np.sqrt(1-beta)*noisy_u)
    # Gaussian-reference probability-flow ODE has zero drift with exact score.
    score = -np.linalg.solve(ref.covariance, (noisy-ref.mean).T).T
    flow_drift = -.5*beta*(noisy-ref.mean)-.5*beta*(score @ ref.covariance)
    assert np.max(np.abs(flow_drift)) < 1e-12
    transitioned = stationary_reference_step(clean, ref, .6, noise)
    assert np.max(np.abs(np.cov(transitioned.T)-ref.covariance)) < .015
    assert np.max(np.abs(transitioned.mean(0)-ref.mean)) < .015
    # Exact expected sample-to-target MSE under the joint Gaussian model.
    risks = {str(r): float(2*np.trace(ref.covariance)) for r, ref in zip([0, 1, 2, 4], refs)}
    assert risks['4'] < risks['2'] < risks['1'] < risks['0']
    # A miscalibrated observation sign gives a concrete counterexample.
    # y~N(0,1), z=y+noise, Var(noise)=1; falsely predict mean=-z/2,var=.5.
    wrong_sign_MSE, unconditional_MSE = 3., 2.
    assert wrong_sign_MSE > unconditional_MSE
    # Matching moments is not distribution calibration: symmetric mixtures.
    mode_mean, within_std = 2., .2
    fourth_mixture = mode_mean**4+6*mode_mean**2*within_std**2+3*within_std**4
    fourth_gaussian = 3*(mode_mean**2+within_std**2)**2
    assert abs(fourth_mixture-fourth_gaussian) > 1
    # Finite-sample theorem: same linear head and ridge lambda, Gaussian data.
    # The analytic sigma*U skip removes regularization bias; variance is equal.
    dim, fit_samples, ridge, repetitions = 8, 32, 8., 5000
    design = rng.normal(size=(fit_samples, dim))
    gram = design.T @ design
    inverse = np.linalg.inv(gram+ridge*np.eye(dim))
    operator_ridge = inverse @ design.T
    conditional_errors = alpha*rng.normal(size=(repetitions, fit_samples, dim))
    residual_coefficients = np.einsum('ij,kjl->kil', operator_ridge, conditional_errors)
    regularization_bias = -sigma*ridge*inverse
    direct_excess = np.square(residual_coefficients+regularization_bias).sum(axis=(1, 2)).mean()
    residual_excess = np.square(residual_coefficients).sum(axis=(1, 2)).mean()
    predicted_residual_excess = alpha**2*dim*np.trace(inverse @ gram @ inverse)
    predicted_gain = sigma**2*ridge**2*np.trace(inverse @ inverse)
    assert abs(residual_excess/predicted_residual_excess-1) < .02
    assert abs((direct_excess-residual_excess)/predicted_gain-1) < .02
    assert direct_excess > residual_excess
    modules = {}
    for name in ['gaussian_reference.py', 'validate_reference.py']:
        data = (Path(__file__).parent/name).read_bytes()
        modules[name] = hashlib.sha256(data).hexdigest()
    result = {'status': 'completed', 'scope': 'synthetic algebra and Gaussian Monte Carlo only',
        'EEG_training_or_image_generation_run': False, 'novelty_proven': False, 'official_test_read': False,
        'seed': 2026101501, 'samples': n,
        'checks': ['posterior_PSD_order', 'missing_EEG_identity', 'single_trial_nonzero_uncertainty',
            'duplicate_trial_no_precision_gain', 'inconsistent_duplicate_rejected',
            'shared_session_covariance_accounted', 'whitening_roundtrip', 'forward_endpoints',
            'whitened_forward_kernel', 'residual_target_energy', 'loss_reparameterization_identity',
            'DDPM_stationary_reference_mean', 'Gaussian_probability_flow_stationarity',
            'exact_OU_reference_moments', 'Gaussian_pairing_MSE_gain', 'miscalibration_counterexample',
            'moment_matching_not_full_calibration', 'finite_sample_same_capacity_ridge_risk'],
        'Gaussian_oracle_expected_sample_MSE_by_R': risks,
        'residual_target_energy': residual_energy, 'standard_noise_target_energy': original_energy,
        'measured_target_energy_ratio': residual_energy/original_energy, 'theoretical_ratio': alpha**2,
        'miscalibration_counterexample': {'wrong_sign_expected_MSE': wrong_sign_MSE, 'prior_expected_MSE': unconditional_MSE},
        'finite_sample_ridge': {'dimension': dim, 'fit_samples': fit_samples, 'lambda': ridge,
            'Monte_Carlo_repetitions': repetitions, 'direct_excess_risk': float(direct_excess),
            'residual_excess_risk': float(residual_excess), 'theoretical_residual_excess_risk': float(predicted_residual_excess),
            'theoretical_risk_gain': float(predicted_gain), 'scope': 'Gaussian conditional, linear head, ridge penalty; not AdamW/UNet theorem'},
        'source_sha256': modules, 'elapsed_seconds': time.monotonic()-started}
    with (out/'results.json').open('x') as dest: json.dump(result, dest, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
