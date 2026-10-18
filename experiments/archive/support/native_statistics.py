"""Concept-level paired uncertainty, without treating tokens or rounds as IID."""
import numpy as np


def concept_bootstrap_counts(n, repetitions=10000, seed=2026101693):
    assert isinstance(n, int) and n >= 2 and repetitions >= 100
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1 / n), size=repetitions)
    # Such draws have no between-concept pairs. Negligible for n=200; explicit
    # rejection keeps small synthetic tests mathematically well-defined too.
    invalid = (counts > 0).sum(1) < 2
    while invalid.any():
        counts[invalid] = rng.multinomial(n, np.full(n, 1 / n), size=int(invalid.sum()))
        invalid = (counts > 0).sum(1) < 2
    return counts.astype(np.float64)


def weighted_identification(matrix, counts):
    """Bootstrap the two-axis U statistic; repeated copies of a concept cannot distract itself."""
    matrix = np.asarray(matrix, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    n = len(matrix)
    assert matrix.shape == (n, n) and counts.ndim == 2 and counts.shape[1] == n
    assert np.isfinite(matrix).all() and np.isfinite(counts).all() and np.all(counts >= 0)
    assert np.all(np.diag(matrix) == 0), 'Same-concept comparisons must be excluded'
    denominator = counts.sum(1) ** 2 - (counts ** 2).sum(1)
    assert np.all(denominator > 0), 'Degenerate all-one-concept resample is undefined'
    return np.einsum('bi,bi->b', counts @ matrix, counts) / denominator


def interval(values, confidence=.95):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and np.isfinite(values).all() and 0 < confidence < 1
    lo, hi = np.quantile(values, [(1 - confidence) / 2, (1 + confidence) / 2])
    return [float(lo), float(hi)]


def paired_identification_interval(paired_matrix, control_matrix, counts):
    delta = np.asarray(paired_matrix, dtype=np.float64) - np.asarray(control_matrix, dtype=np.float64)
    n = len(delta)
    assert delta.shape == (n, n) and np.all(np.diag(delta) == 0)
    return {'paired_minus_control': float(delta.sum() / (n * (n - 1))),
            'interval95': interval(weighted_identification(delta, counts)),
            'resampling_unit': 'concept, query and distractor axes jointly',
            'independent_subjects': 1, 'independent_training_seeds': 1}


def finite_mean(values):
    """Propagate an undefined query instead of silently selecting surviving outputs."""
    assert len(values) > 0
    if any(value is None for value in values):
        return None
    x = np.asarray(values, dtype=np.float64)
    assert np.isfinite(x).all()
    return float(x.mean())


def paired_query_interval(paired, control, counts):
    assert len(paired) == len(control) == counts.shape[1]
    missing = sum(a is None or b is None for a, b in zip(paired, control))
    if missing:
        return {'paired_minus_control': None, 'interval95': None, 'undefined_queries': missing}
    delta = np.asarray(paired, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    assert np.isfinite(delta).all()
    return {'paired_minus_control': float(delta.mean()),
            'interval95': interval((counts @ delta) / counts.sum(1)),
            'resampling_unit': 'concept', 'undefined_queries': 0}


def average_rounds(rounds):
    """Average within each query before any query-level interval calculation."""
    assert len(rounds) > 0 and len({len(row) for row in rounds}) == 1
    return [finite_mean(list(values)) for values in zip(*rounds)]
