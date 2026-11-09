"""Classical covariance law for averaging exchangeable repeated measurements.

This is not a new theorem. Finite-pool identities do not prove R80 extrapolation.
"""
import itertools
import numpy as np

def predicted_variance(reference_variance, repeat_contrast_variance, count, reference_count=4):
    v = np.asarray(reference_variance, dtype=np.float64)
    n = np.asarray(repeat_contrast_variance, dtype=np.float64)
    assert isinstance(count, int) and count > 0 and reference_count == 4
    assert v.shape == n.shape and np.isfinite(v).all() and np.isfinite(n).all()
    assert np.all(v > 0) and np.all(n >= 0)
    if count == reference_count: return v.copy()
    result = v + (1 / count - 1 / reference_count) * n
    if not np.all(result > 0):
        raise ValueError('Nonpositive extrapolated variance; preserve failure, do not clamp.')
    return result

def finite_pool_moments(x):
    """x has stimulus, repeat4, channel, position axes; equal weights throughout."""
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 4 and x.shape[1] == 4
    mean4 = x.mean(1)
    mean = mean4.mean((0, 2))
    variance4 = np.square(mean4 - mean[None, :, None]).mean((0, 2))
    contrast = np.square(x - mean4[:, None]).sum(1).mean((0, 2)) / 3
    observed = {}
    for m in (1, 2, 3, 4):
        subsets = [x[:, s].mean(1) for s in itertools.combinations(range(4), m)]
        observed[m] = np.mean([np.square(z - mean[None, :, None]).mean((0, 2)) for z in subsets], axis=0)
    return dict(mean=mean, variance4=variance4, contrast_variance=contrast, subset_variances=observed)

def self_checks():
    rng = np.random.default_rng(1729)
    # These samples need not be independent or identically distributed for the finite-pool identity.
    x = rng.normal(size=(17, 4, 5, 11))
    x += np.arange(4)[None, :, None, None] * .3
    x[:, 1] = .7 * x[:, 0] + .3 * x[:, 1]
    m = finite_pool_moments(x)
    for count in (1, 2, 3, 4):
        predicted = m['variance4'] + (1 / count - .25) * m['contrast_variance']
        assert np.max(abs(predicted - m['subset_variances'][count])) < 2e-14
    assert np.array_equal(predicted_variance(m['variance4'], m['contrast_variance'], 4), m['variance4'])
    # A shared correlated component cancels from the difference between repeat counts.
    signal, shared, residual = 2., .4, .9
    v4 = np.array([signal + shared + residual / 4])
    assert abs(predicted_variance(v4, np.array([residual]), 80)[0] - (signal + shared + residual / 80)) < 1e-15
    try: predicted_variance(np.array([.1]), np.array([1.]), 80)
    except ValueError: pass
    else: raise AssertionError('Nonpositive physical forecast was silently accepted')
    return dict(finite_pool_all_subset_identities=True, exact_R4_identity=True,
                exchangeable_shared_covariance_cancels=True, negative_variance_rejected=True,
                finite_pool_identity_does_not_establish_R80_exchangeability=True)

if __name__ == '__main__':
    import json
    print(json.dumps(self_checks()))
