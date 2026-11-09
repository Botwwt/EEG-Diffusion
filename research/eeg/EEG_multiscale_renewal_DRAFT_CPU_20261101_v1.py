"""DRAFT probability arithmetic; no EEG fit, native kernel, or GPU admission.

The copy/refresh chain can be summed over renewal segments exactly. The
optional importance estimator below requires independent banks for DIFFERENT
segment STARTS, fixed full-support proposals independent of model parameters,
and deterministic complete emissions for every queried class. It estimates
likelihood without bias, not log-likelihood or its original gradient. This is
classical Monte Carlo objective machinery, not a novelty claim.
"""
import numpy as np


def lse(value, axis=None):
    value = np.asarray(value, np.float64)
    maximum = np.max(value, axis=axis, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        shifted = np.where(np.isfinite(maximum), value - maximum, -np.inf)
        result = maximum + np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    result = np.where(np.isneginf(maximum), -np.inf, result)
    return np.squeeze(result, axis=axis)


def normalize_logits(logits):
    logits = np.asarray(logits, np.float64)
    assert logits.ndim == 2 and logits.shape[1] >= 2 and np.isfinite(logits).all()
    return logits - lse(logits, axis=1)[:, None]


def exact_segments(logits, emissions):
    logpi = normalize_logits(logits)
    ell = np.asarray(emissions, np.float64)
    assert ell.shape == logpi.shape and np.isfinite(ell).all() and np.all(ell <= 0)
    scales = len(logpi)
    result = np.full((scales, scales), -np.inf)
    for start in range(scales):
        cumulative = np.zeros(ell.shape[1])
        for end in range(start, scales):
            cumulative += ell[end]
            result[start, end] = lse(logpi[start] + cumulative)
    return result


def renewal_log_evidence(segment_log_evidence, stay):
    segment = np.asarray(segment_log_evidence, np.float64)
    rho = np.asarray(stay, np.float64)
    scales = len(segment)
    assert scales >= 1 and segment.shape == (scales, scales) and rho.shape == (scales - 1,)
    assert np.isfinite(segment[np.triu_indices(scales)]).all()
    assert np.isfinite(rho).all() and np.all((rho >= 0) & (rho <= 1))
    with np.errstate(divide='ignore'):
        logstay = np.log(rho)
        logreset = np.log1p(-rho)
    forward = np.full(scales + 1, -np.inf)
    forward[0] = 0.
    for end in range(scales):
        terms = []
        for start in range(end + 1):
            boundary = 0. if start == 0 else logreset[start - 1]
            within = logstay[start:end].sum()
            terms.append(forward[start] + boundary + within + segment[start, end])
        forward[end + 1] = lse(terms)
    return float(forward[-1])


def exact_log_evidence(logits, stay, emissions):
    return renewal_log_evidence(exact_segments(logits, emissions), stay)


def sampled_segments(logits, complete_emissions, proposal, indices):
    """Probability-only prototype with a synthetic/full supplied emission table.

    This implementation receives all emissions to make exhaustive correctness
    checks possible. It DOES NOT demonstrate sparse native evaluation or any
    reduction in actual VAR time. A future sparse kernel must provide the same
    complete scale emissions only for the sampled class union.
    """
    logpi = normalize_logits(logits)
    ell = np.asarray(complete_emissions, np.float64)
    q = np.asarray(proposal, np.float64)
    ids = np.asarray(indices)
    scales, classes = logpi.shape
    assert ell.shape == q.shape == logpi.shape and np.isfinite(ell).all() and np.all(ell <= 0)
    assert np.isfinite(q).all() and np.all(q > 0) and np.allclose(q.sum(1), 1., atol=1e-14, rtol=0)
    assert ids.ndim == 2 and ids.shape[0] == scales and ids.shape[1] >= 1
    assert np.issubdtype(ids.dtype, np.integer) and np.all((ids >= 0) & (ids < classes))
    samples = ids.shape[1]
    result = np.full((scales, scales), -np.inf)
    for start in range(scales):
        selected = ids[start]
        weight = logpi[start, selected] - np.log(q[start, selected])
        cumulative = np.zeros(samples)
        for end in range(start, scales):
            cumulative += ell[end, selected]
            result[start, end] = lse(weight + cumulative) - np.log(samples)
    return result


def sampled_log_evidence(logits, stay, emissions, proposal, indices):
    return renewal_log_evidence(sampled_segments(logits, emissions, proposal, indices), stay)


def independent_start_samples(proposal, samples, rng):
    q = np.asarray(proposal, np.float64)
    assert q.ndim == 2 and samples >= 1 and np.all(q > 0)
    assert np.allclose(q.sum(1), 1., atol=1e-14, rtol=0)
    return np.stack([rng.choice(q.shape[1], size=samples, replace=True, p=row) for row in q])
