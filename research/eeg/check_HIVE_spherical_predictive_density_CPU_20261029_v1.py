"""Independent normalization/integration/Monte Carlo checks, not EEG efficacy."""
import json
import math
import numpy as np
from scipy import integrate, special
from brainhive_spherical_predictive_density_CPU_20261029_v1 import (
    fit_vmf_concentration, host_direction, log_area, power_spherical_entropy,
    power_to_vmf_kl, restore_host, sample_power_spherical,
    vmf_cosine_quantiles, vmf_log_normalizer, vmf_mean_cosine,
    fit_marginal_vmf_tilt, vmf_tilt_score,
)

rng = np.random.default_rng(1729)
checks = []
for concentration in (0, .01, 1, 20, 100):
    log_c = vmf_log_normalizer(3, concentration)
    mass = 2 * math.pi * integrate.quad(lambda t: math.exp(log_c + concentration * t), -1, 1, epsabs=1e-11)[0]
    assert abs(mass - 1) < 1e-10
checks.append('vMF_integrates_to_one_independent_S2_quadrature')
for dimension in (3, 17, 1023):
    for concentration in (0, .0001, 10, 200, 1000):
        mean = vmf_mean_cosine(dimension, concentration)
        if concentration:
            fitted = fit_vmf_concentration(dimension, mean)
            assert abs(fitted - concentration) < 1e-5
        assert np.isfinite(vmf_log_normalizer(dimension, concentration)) and 0 <= mean < 1
checks.append('finite_exact_high_dimension_normalizer_and_MLE_inversion_including_Bessel_underflow_regime')
q = vmf_cosine_quantiles(3, 0)
assert np.max(abs(q - np.array([-.95, 0, .95]))) < 1e-8
for dimension in (17, 1023):
    q = vmf_cosine_quantiles(dimension, 0)
    reference = 2 * special.betaincinv((dimension - 1) / 2, (dimension - 1) / 2, [.025, .5, .975]) - 1
    assert np.max(abs(q - reference)) < 1e-8
checks.append('predictive_cosine_quantiles_agree_with_uniform_sphere_exact_Beta_distribution')
for dimension in (3, 17, 1023):
    assert abs(power_spherical_entropy(dimension, 0) - log_area(dimension)) < 1e-10
    assert abs(power_to_vmf_kl(dimension, 0, 0, .3)) < 1e-10
checks.append('uniform_Power_and_vMF_have_same_entropy_and_zero_KL')
mu = np.array([.3, .4, math.sqrt(.75)])
sample = sample_power_spherical(mu, 4, 100000, rng)
assert np.max(abs(np.linalg.norm(sample, axis=-1) - 1)) < 1e-12
expected = 4 / 6 * mu
assert np.max(abs(sample.mean(axis=0) - expected)) < .008
direction_p = np.array([1., 0, 0])
beta = 1.
alpha = 5.
log_z = (alpha + beta) * math.log(2) + beta * math.log(math.pi) + special.gammaln(alpha) - special.gammaln(alpha + beta)
log_q = 4 * np.log1p(sample @ mu) - log_z
log_p = vmf_log_normalizer(3, 3) + 3 * (sample @ direction_p)
mc = log_q - log_p
analytic = power_to_vmf_kl(3, 4, 3, .3)
assert abs(mc.mean() - analytic) < 5 * mc.std() / math.sqrt(len(mc))
checks.append('independent_sampler_unit_support_analytic_mean_and_cross_direction_KL_Monte_Carlo')
for first in (1., -1.):
    mu = np.zeros(1023)
    mu[0] = first
    sample = sample_power_spherical(mu, 200, 2000, rng)
    assert np.isfinite(sample).all() and np.max(abs(np.linalg.norm(sample, axis=-1) - 1)) < 1e-12
checks.append('high_dimension_sampler_exact_and_near_pole_Householder_branches')
raw = rng.normal(size=(12, 1024))
raw -= raw.mean(axis=-1, keepdims=True)
weight = rng.uniform(.5, 1.5, size=1024)
bias = rng.normal(size=1024)
condition = bias + weight * raw
unit, radius, offset = host_direction(condition, weight, bias)
assert np.max(abs(restore_host(unit, radius, weight, bias) - condition)) < 1e-12
assert np.max(abs(offset)) < 1e-12
checks.append('real_host_affine_and_zero_mean_sphere_roundtrip')
# Exact finite analogue: ELBO gap is KL to the actual normalized posterior.
prior = np.array([.2, .5, .3])
likelihood = np.array([.1, .8, .3])
q = np.array([.25, .6, .15])
evidence = prior @ likelihood
posterior = prior * likelihood / evidence
bound = np.sum(q * np.log(likelihood)) - np.sum(q * np.log(q / prior))
gap = np.sum(q * np.log(q / posterior))
assert abs(math.log(evidence) - bound - gap) < 1e-12 and gap > 0
checks.append('normalized_semantic_latent_model_ELBO_gap_exact_posterior_KL_finite_example')
population_cos = np.array([.2, .4, .6, -.1])
eeg_cos = np.array([.1, .3, .2, .05])
common = np.array([.1, -.1, .2, .4])
zero_score, length, axis = vmf_tilt_score(17, 4, 0, common, eeg_cos, population_cos)
assert np.max(abs(zero_score - (vmf_log_normalizer(17, 4) + 4 * population_cos))) < 1e-12
checks.append('zero_EEG_tilt_exactly_recovers_normalized_population_prior')
scale = fit_marginal_vmf_tilt(17, 4, common, eeg_cos)
value = vmf_tilt_score(17, 4, scale, common, eeg_cos, population_cos)[0].mean()
grid = np.linspace(0, max(10, scale * 2), 2001)
assert value >= max(vmf_tilt_score(17, 4, candidate, common, eeg_cos, population_cos)[0].mean() for candidate in grid) - 1e-8
assert fit_marginal_vmf_tilt(17, 4, np.zeros(4), -np.ones(4) * .1) == 0
checks.append('convex_one_scalar_conditional_MLE_beats_independent_dense_grid_and_has_zero_boundary')
# Exact InfoNCE generative model: positives conditional, negatives marginal.
joint = np.array([[.10, .30, .10], [.25, .05, .20]])
eeg_marginal = joint.sum(axis=1)
visual_marginal = joint.sum(axis=0)
conditional = joint / eeg_marginal[:, None]
ratio = conditional / visual_marginal
recovered = visual_marginal * ratio
recovered /= recovered.sum(axis=1, keepdims=True)
wrong_uniform_base = ratio / ratio.sum(axis=1, keepdims=True)
assert np.max(abs(recovered - conditional)) < 1e-12
assert np.max(abs(wrong_uniform_base - conditional)) > .01
checks.append('finite_exact_density_ratio_requires_visual_base_to_recover_conditional')
print(json.dumps(dict(status='PASS_CLASSICAL_SPHERICAL_DENSITY_INDEPENDENT_CPU_CHECKS', checks=checks, actual_EEG_evidence=False, image_likelihood_is_not_Turbo_pixel_likelihood=True, new_GPU_calls=0, novelty_established=False)))
