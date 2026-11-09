"""Classical normalized directional distributions, not a novel EEG method.

All densities use surface measure on S^(ambient_dimension-1). The host's
1024-vector has a zero-mean constraint, hence ambient_dimension=1023.
No Torch/CUDA or host weights are imported here.
"""
import math
import numpy as np
from scipy import integrate, optimize, special


def log_area(dimension):
    assert dimension >= 2
    return math.log(2) + dimension / 2 * math.log(math.pi) - special.gammaln(dimension / 2)


def _log_bessel_i(order, concentration):
    assert concentration > 0 and order >= 0
    scaled = special.ive(order, concentration)
    if scaled > 0 and np.isfinite(scaled):
        return float(math.log(scaled) + concentration)
    # Positive-term series, in log space, also covers large order/small kappa.
    terms = [0.0]
    term = 0.0
    for j in range(1, 100001):
        term += 2 * math.log(concentration / 2) - math.log(j) - math.log(order + j)
        terms.append(term)
        if j > concentration and term < max(terms) - 40:
            break
    else:
        raise ArithmeticError('Bessel series did not converge')
    return order * math.log(concentration / 2) - special.gammaln(order + 1) + float(special.logsumexp(terms))


def vmf_log_normalizer(dimension, concentration):
    assert dimension >= 2 and concentration >= 0 and np.isfinite(concentration)
    if concentration == 0:
        return -log_area(dimension)
    order = dimension / 2 - 1
    return order * math.log(concentration) - dimension / 2 * math.log(2 * math.pi) - _log_bessel_i(order, concentration)


def vmf_mean_cosine(dimension, concentration):
    if concentration == 0:
        return 0.0
    order = dimension / 2 - 1
    return float(math.exp(_log_bessel_i(order + 1, concentration) - _log_bessel_i(order, concentration)))


def fit_vmf_concentration(dimension, mean_cosine):
    assert np.isfinite(mean_cosine) and -1 <= mean_cosine < 1
    if mean_cosine <= 0:
        return 0.0
    high = max(1.0, dimension * mean_cosine / (1 - mean_cosine ** 2))
    while vmf_mean_cosine(dimension, high) < mean_cosine:
        high *= 2
    return float(optimize.brentq(lambda k: vmf_mean_cosine(dimension, k) - mean_cosine, 0, high, xtol=1e-10))


def vmf_cosine_quantiles(dimension, concentration, probabilities=(.025, .5, .975)):
    assert dimension >= 3
    constant = vmf_log_normalizer(dimension, concentration) + log_area(dimension - 1)

    def pdf(t):
        if abs(t) >= 1:
            return 0.0
        return math.exp(constant + concentration * t + (dimension - 3) / 2 * math.log1p(-t * t))

    # Split at the mode so quadrature cannot miss a narrow high-dimensional peak.
    if dimension == 3:
        mode = 1.0 if concentration else 0.0
    else:
        mode = 2 * concentration / ((dimension - 3) + math.sqrt((dimension - 3) ** 2 + 4 * concentration ** 2))

    def cdf(t):
        if t <= mode:
            return integrate.quad(pdf, -1, t, epsabs=1e-10, points=[min(t, mode)])[0]
        return 1 - integrate.quad(pdf, t, 1, epsabs=1e-10)[0]

    return np.array([optimize.brentq(lambda t: cdf(t) - probability, -1 + 1e-12, 1 - 1e-12, xtol=1e-10) for probability in probabilities])


def power_spherical_entropy(dimension, concentration):
    assert dimension >= 2 and concentration >= 0
    beta = (dimension - 1) / 2
    alpha = beta + concentration
    log_z = (alpha + beta) * math.log(2) + beta * math.log(math.pi) + special.gammaln(alpha) - special.gammaln(alpha + beta)
    expected_log = math.log(2) + special.digamma(alpha) - special.digamma(alpha + beta)
    return float(log_z - concentration * expected_log)


def power_to_vmf_kl(dimension, q_concentration, p_concentration, mean_direction_cosine):
    assert -1 <= mean_direction_cosine <= 1
    q_resultant = q_concentration / (q_concentration + dimension - 1)
    return float(-power_spherical_entropy(dimension, q_concentration) - vmf_log_normalizer(dimension, p_concentration) - p_concentration * q_resultant * mean_direction_cosine)


def sample_power_spherical(mean_direction, concentration, samples, generator):
    mean_direction = np.asarray(mean_direction, dtype=np.float64)
    dimension = mean_direction.size
    assert abs(np.linalg.norm(mean_direction) - 1) < 1e-10 and concentration >= 0
    beta = (dimension - 1) / 2
    axial = 2 * generator.beta(beta + concentration, beta, size=samples) - 1
    tangent = generator.normal(size=(samples, dimension - 1))
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    value = np.concatenate((axial[:, None], np.sqrt(np.maximum(0, 1 - axial ** 2))[:, None] * tangent), axis=-1)
    # Stable orthogonal map from e1 to mu. The sign choice avoids a near-zero
    # Householder denominator when mu approaches e1.
    sign = 1.0 if mean_direction[0] >= 0 else -1.0
    householder = mean_direction.copy()
    householder[0] += sign
    householder /= np.linalg.norm(householder)
    value[:, 0] *= -sign
    return value - 2 * (value @ householder)[:, None] * householder


def host_direction(condition, weight, bias):
    raw = (np.asarray(condition, dtype=np.float64) - np.asarray(bias, dtype=np.float64)) / np.asarray(weight, dtype=np.float64)
    centered = raw - raw.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    assert np.all(np.isfinite(centered)) and np.all(norm > 1e-12)
    return centered / norm, norm[:, 0], raw.mean(axis=-1)


def restore_host(direction, radius, weight, bias):
    direction = np.asarray(direction, dtype=np.float64)
    assert np.max(np.abs(direction.mean(axis=-1))) < 1e-10
    assert np.max(np.abs(np.linalg.norm(direction, axis=-1) - 1)) < 1e-10
    return np.asarray(bias) + np.asarray(weight) * np.asarray(radius)[..., None] * direction


def vmf_log_normalizer_array(dimension, concentration):
    concentration = np.asarray(concentration, dtype=np.float64)
    assert np.all(concentration >= 0) and np.isfinite(concentration).all()
    order = dimension / 2 - 1
    bessel = special.ive(order, concentration)
    valid = (concentration > 0) & (bessel > 0) & np.isfinite(bessel)
    answer = np.empty_like(concentration)
    answer[valid] = order * np.log(concentration[valid]) - dimension / 2 * math.log(2 * math.pi) - np.log(bessel[valid]) - concentration[valid]
    for index in np.argwhere(~valid):
        key = tuple(index)
        answer[key] = vmf_log_normalizer(dimension, float(concentration[key]))
    return answer


def vmf_mean_cosine_array(dimension, concentration):
    concentration = np.asarray(concentration, dtype=np.float64)
    order = dimension / 2 - 1
    denominator = special.ive(order, concentration)
    numerator = special.ive(order + 1, concentration)
    valid = (concentration > 0) & (denominator > 0) & (numerator > 0)
    answer = np.empty_like(concentration)
    answer[valid] = numerator[valid] / denominator[valid]
    for index in np.argwhere(~valid):
        key = tuple(index)
        answer[key] = vmf_mean_cosine(dimension, float(concentration[key]))
    return answer


def fit_marginal_vmf_tilt(dimension, population_concentration, EEG_population_cosine, EEG_target_cosine):
    """One nonnegative scalar MLE in p0(v) exp(a EEG_direction dot v).

    This is a normalized exponential-family tilt, not a new theorem. Population
    parameters are fixed before fitting a; a=0 exactly recovers p0.
    """
    common = np.asarray(EEG_population_cosine, dtype=np.float64)
    target = np.asarray(EEG_target_cosine, dtype=np.float64)
    assert common.shape == target.shape and common.ndim == 1

    def derivative(scale):
        length = np.sqrt(np.maximum(0, population_concentration ** 2 + scale ** 2 + 2 * population_concentration * scale * common))
        ratio = np.divide(scale + population_concentration * common, length, out=np.zeros_like(length), where=length > 0)
        return float(np.mean(target - vmf_mean_cosine_array(dimension, length) * ratio))

    if derivative(0) <= 0:
        return 0.0
    high = max(1., float(population_concentration))
    while derivative(high) > 0:
        high *= 2
        assert high < 1e8
    return float(optimize.brentq(derivative, 0, high, xtol=1e-8))


def vmf_tilt_score(dimension, population_concentration, scale, EEG_population_cosine, EEG_target_cosine, population_target_cosine):
    common = np.asarray(EEG_population_cosine)
    length = np.sqrt(np.maximum(0, population_concentration ** 2 + scale ** 2 + 2 * population_concentration * scale * common))
    natural_dot_target = population_concentration * np.asarray(population_target_cosine) + scale * np.asarray(EEG_target_cosine)
    score = vmf_log_normalizer_array(dimension, length) + natural_dot_target
    posterior_axis_cosine = np.divide(natural_dot_target, length, out=np.zeros_like(length), where=length > 0)
    return score, length, posterior_axis_cosine
