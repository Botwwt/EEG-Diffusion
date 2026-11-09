"""Exhaustive SYNTHETIC probability checks; no neural or image evidence."""
from pathlib import Path
from datetime import datetime, timezone
from itertools import product
import hashlib
import json
import sys
import numpy as np
import EEG_multiscale_renewal_DRAFT_CPU_20261101_v1 as R
from EEG_multiscale_native_class_DRAFT_CPU_20261101_v1 import inference

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'audit/HIVE_independent_route_primary_review_20261101_v1/actual_synthetic_renewal_and_objective_bias_checks.json'


def ref(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def banks(q, samples):
    scales, classes = q.shape
    for values in product(range(classes), repeat=scales * samples):
        indices = np.asarray(values, np.int64).reshape(scales, samples)
        probability = float(np.prod(q[np.arange(scales)[:, None], indices]))
        yield probability, indices


def expected_objective(h, rho, ell, q, samples):
    return sum(weight * R.sampled_log_evidence(h, rho, ell, q, ids)
               for weight, ids in banks(q, samples))


def difference(function, h):
    gradient = np.zeros_like(h)
    for index in np.ndindex(h.shape):
        positive, negative = h.copy(), h.copy()
        positive[index] += 1e-5
        negative[index] -= 1e-5
        gradient[index] = (function(positive) - function(negative)) / 2e-5
    return gradient


def main():
    rng = np.random.default_rng(1729)
    exact_error = 0.
    for scales in (1, 3, 10):
        h = rng.normal(size=(scales, 7))
        ell = -rng.uniform(.1, 4., size=h.shape)
        for rho in (np.zeros(scales - 1), np.ones(scales - 1), rng.uniform(.1, .9, size=scales - 1)):
            actual = R.exact_log_evidence(h, rho, ell)
            expected = inference(h, rho, ell)['log_evidence']
            exact_error = max(exact_error, abs(actual - expected))
    assert exact_error < 2e-13

    h = np.array([[.2, -.4], [-.1, .6], [.5, -.2]])
    ell = np.log(np.array([[.8, .2], [.35, .65], [.7, .3]]))
    rho = np.array([.35, .7])
    q = np.array([[.7, .3], [.2, .8], [.6, .4]])
    exact = R.exact_log_evidence(h, rho, ell)
    unbiased_error = 0.
    evidence_results = []
    for samples in (1, 2):
        total = mean_likelihood = mean_log = 0.
        maximum_log = -np.inf
        count = 0
        for weight, ids in banks(q, samples):
            logvalue = R.sampled_log_evidence(h, rho, ell, q, ids)
            total += weight
            mean_likelihood += weight * np.exp(logvalue)
            mean_log += weight * logvalue
            maximum_log = max(maximum_log, logvalue)
            count += 1
        assert abs(total - 1.) < 1e-13 and mean_log <= exact + 1e-13
        unbiased_error = max(unbiased_error, abs(mean_likelihood - np.exp(exact)))
        assert maximum_log > exact
        evidence_results.append(dict(samples_per_start=samples, exhaustive_banks=count,
            total_probability=total, exact_likelihood=float(np.exp(exact)), expected_estimate=mean_likelihood,
            expected_log_estimate=mean_log, expected_log_gap=float(exact - mean_log),
            largest_individual_log_estimate_excess=float(maximum_log - exact)))
    assert unbiased_error < 1e-13

    uniform = np.full((2, 2), .5)
    same_emissions = np.log(np.array([[.8, .2], [.8, .2]]))
    reset = np.array([0.])
    shared = sum(.5 * np.exp(R.sampled_log_evidence(np.log(uniform), reset, same_emissions, uniform,
                        np.full((2, 1), c, np.int64))) for c in range(2))
    correct = np.exp(R.exact_log_evidence(np.log(uniform), reset, same_emissions))
    assert abs(shared - .34) < 1e-14 and abs(correct - .25) < 1e-14

    # Coarse class evidence is cancelled by the later scale: full evidence is flat.
    flat_full = np.log(np.array([[.8, .2], [.2, .8]]))
    proposal = np.array([[.8, .2], [.5, .5]])
    logits = np.zeros((2, 2))
    stay = np.array([1.])
    full = R.exact_log_evidence(logits, stay, flat_full)
    bound = expected_objective(logits, stay, flat_full, proposal, 1)
    KL = float(np.sum(proposal[0] * np.log(proposal[0] / .5)))
    assert abs(full - np.log(.16)) < 1e-14 and abs(full - bound - KL) < 1e-13
    true_gradient = inference(logits, stay, flat_full)['negative_log_evidence_gradient_logits']
    bound_gradient = difference(lambda value: expected_objective(value, stay, flat_full, proposal, 1), logits)
    expected_gradient = np.array([[.3, -.3], [0., 0.]])
    gradient_error = float(np.max(np.abs(bound_gradient - expected_gradient)))
    assert np.max(np.abs(true_gradient)) < 1e-14 and gradient_error < 1e-9
    pointwise = R.sampled_log_evidence(h, rho, ell, q, np.zeros((3, 1), np.int64))
    assert np.isfinite(pointwise)

    output = dict(status='PASS_EXHAUSTIVE_SYNTHETIC_RENEWAL_AND_OBJECTIVE_BIAS_AUDIT_NOT_EEG_EVIDENCE',
        UTC=datetime.now(timezone.utc).isoformat(), sources=[ref(Path(__file__).resolve()), ref(Path(R.__file__).resolve())],
        exact_renewal_vs_existing_HMM_maxerror=exact_error,
        independent_start_estimator_unbiased_likelihood_maxerror=unbiased_error,
        exhaustive_objective_results=evidence_results,
        shared_start_bank_counterexample=dict(shared_expected_estimate=shared, exact_likelihood=float(correct), bias=float(shared-correct)),
        flat_full_likelihood_counterexample=dict(exact_log_evidence=full, expected_bound=bound,
            gap_equals_KL=KL, true_negative_log_evidence_gradient=true_gradient.tolist(),
            sampled_expected_log_bound_gradient=bound_gradient.tolist(), analytic_gradient_maxerror=gradient_error,
            interpretation='A variational surrogate can demand prediction of its image proposal even when the exact complete likelihood has no class gradient.'),
        actual_native_emissions_used=False, actual_EEG_used=False, GPU_jobs=0,
        new_model_training_admitted=False, actual_native_cost_reduction_measured=False,
        next_action='Do not replace complete exact EEG likelihood training by a finite-bank log estimator merely because its likelihood estimate is unbiased. Native exact T8 evidence/cost and gradient-gap/variance diagnostics are still required.')
    assert not OUT.exists()
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=output['status'], exact_error=exact_error, unbiased_error=unbiased_error,
        phantom_bound_gradient_maxerror=gradient_error, shared_bank_bias=float(shared-correct), output=str(OUT)), ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
