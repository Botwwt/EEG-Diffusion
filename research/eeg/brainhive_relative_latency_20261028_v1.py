"""Fixed D28 relative-latency measurement operator; NumPy and CPU only.

This is classical registration, not an original registration algorithm.  The
four-repeat zero-sum gauge preserves the estimated repeat centre; it does not
guarantee the absolute latency of every overlapping physiological component.

For edge (a,b), d maximises the symmetric mean of corr(x_a(t),x_b(t+d))
and corr(x_b(t),x_a(t-d)), t=12,...,237, after separately removing each
electrode's time mean. A tied minimum-absolute maximum at both +/-d abstains
to lag zero. Incidence rows are (-1 at a,+1 at b), hence tau=B.T@d/4 and
alignment is x_r(t+tau_r). The estimates are equivariant to repeat order.
All algorithmic constants below are fixed; no RNG or GPU framework is used.
"""
import numpy as np

MAX_LAG = 12
CORE_START, CORE_STOP = 12, 238
CORE_LENGTH = CORE_STOP - CORE_START
TIME_POINTS, ELECTRODES = 250, 17
EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
LAGS = np.arange(-MAX_LAG, MAX_LAG + 1, dtype=np.int16)
INCIDENCE = np.zeros((len(EDGES), 4), dtype=np.float32)
for _edge, (_a, _b) in enumerate(EDGES):
    INCIDENCE[_edge, _a], INCIDENCE[_edge, _b] = -1., 1.
_FFT_LENGTH = 512  # At least 2*250-1: the requested correlations are noncircular.


def _validate_trials(trials, repeats):
    if not isinstance(trials, np.ndarray) or trials.dtype != np.float32:
        raise TypeError('trials must be a float32 NumPy array')
    if trials.ndim != 4 or trials.shape[0] < 1 or trials.shape[1:] != (repeats, ELECTRODES, TIME_POINTS):
        raise ValueError('expected nonempty [B,%d,17,250] trials' % repeats)
    if not np.isfinite(trials).all():
        raise ValueError('nonfinite trials')


def _channel_indices(channels):
    if channels is None:
        return np.arange(ELECTRODES)
    indices = np.asarray(channels)
    if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in 'iu':
        raise ValueError('channels must be a nonempty integer sequence')
    if np.any(indices < 0) or np.any(indices >= ELECTRODES) or np.unique(indices).size != indices.size:
        raise ValueError('channels must be unique indices in [0,17)')
    return indices.astype(np.intp, copy=False)


def _scores(trials, edges, channels):
    """Exactly the symmetric fixed-window score, using reusable FFTs.

    Float64 internal reductions avoid cancellation; public scores are float32.
    Per-electrode offsets are removed before FFT only for numerical stability.
    The final window-specific centring is still performed for every lag.
    """
    selected = np.take(trials, _channel_indices(channels), axis=2).astype(np.float64)
    selected -= selected[..., CORE_START:CORE_STOP].mean(axis=-1, keepdims=True)
    middle = np.zeros_like(selected)
    middle[..., CORE_START:CORE_STOP] = selected[..., CORE_START:CORE_STOP]
    fa = np.fft.rfft(middle, n=_FFT_LENGTH, axis=-1)
    fb = np.fft.rfft(selected, n=_FFT_LENGTH, axis=-1)
    prefix = np.concatenate((np.zeros(selected.shape[:-1] + (1,)), np.cumsum(selected, axis=-1)), axis=-1)
    prefix2 = np.concatenate((np.zeros(selected.shape[:-1] + (1,)), np.cumsum(selected * selected, axis=-1)), axis=-1)
    starts = CORE_START + LAGS.astype(np.intp)
    stops = CORE_STOP + LAGS.astype(np.intp)
    sum_core = middle.sum(axis=-1)
    norm_core2 = (middle * middle).sum(axis=(2, 3)) - (sum_core * sum_core).sum(axis=2) / CORE_LENGTH
    sum_window = np.take(prefix, stops, axis=-1) - np.take(prefix, starts, axis=-1)
    square_window = np.take(prefix2, stops, axis=-1) - np.take(prefix2, starts, axis=-1)
    norm_window2 = (square_window - sum_window * sum_window / CORE_LENGTH).sum(axis=2)

    def directed(a, b):
        cross = np.fft.irfft((fa[:, a].conj() * fb[:, b]).sum(axis=1), n=_FFT_LENGTH, axis=-1)
        cross = cross[:, np.mod(LAGS.astype(np.intp), _FFT_LENGTH)]
        numerator = cross - (sum_core[:, a, :, None] * sum_window[:, b]).sum(axis=1) / CORE_LENGTH
        denominator = np.sqrt(np.maximum(norm_core2[:, a, None], 0.) * np.maximum(norm_window2[:, b], 0.))
        values = np.zeros_like(numerator)
        np.divide(numerator, denominator, out=values, where=denominator > 0.)
        return np.clip(values, -1., 1.)

    all_scores = np.empty((trials.shape[0], len(edges), len(LAGS)), dtype=np.float64)
    for k, (a, b) in enumerate(edges):
        all_scores[:, k] = .5 * (directed(a, b) + directed(b, a)[:, ::-1])
    return all_scores


def _best_lags(scores):
    # A sign preference would violate repeat exchangeability for +/-d ties.
    maxima = scores == scores.max(axis=-1, keepdims=True)
    absolute = np.abs(LAGS)
    minimum_absolute = np.where(maxima, absolute, MAX_LAG + 1).min(axis=-1, keepdims=True)
    nearest_maxima = maxima & (absolute == minimum_absolute)
    ambiguous = nearest_maxima.sum(axis=-1) > 1
    chosen = np.argmax(nearest_maxima, axis=-1)
    chosen = np.where(ambiguous, MAX_LAG, chosen)  # index MAX_LAG is lag zero.
    return LAGS[chosen], np.take_along_axis(scores, chosen[..., None], axis=-1)[..., 0]


def estimate_relative_shifts(trials, channels=None):
    """Return tau [B,4], six edge lags/scores/residuals and all 25 scores."""
    _validate_trials(trials, 4)
    scores = _scores(trials, EDGES, channels)
    edge_lags, edge_scores = _best_lags(scores)
    tau = (edge_lags.astype(np.float32) @ INCIDENCE) / np.float32(4.)
    residuals = tau @ INCIDENCE.T - edge_lags.astype(np.float32)
    return dict(tau=tau, edge_lags=edge_lags, edge_scores=edge_scores.astype(np.float32),
                edge_residuals=residuals, scores=scores.astype(np.float32))


def estimate_pair_lag(xa, xb, channels=None):
    """Estimate d for two [B,17,250] records with the same fixed convention."""
    if not isinstance(xa, np.ndarray) or not isinstance(xb, np.ndarray) or xa.shape != xb.shape:
        raise ValueError('pair record shapes must agree')
    pair = np.stack((xa, xb), axis=1)
    _validate_trials(pair, 2)
    scores = _scores(pair, ((0, 1),), channels)[:, 0]
    lag, score = _best_lags(scores)
    return dict(lag=lag, score=score.astype(np.float32), scores=scores.astype(np.float32))


def shift_records(trials, tau):
    """Linearly interpolate x(t+tau); values outside [0,249] are zero.

    trials is [B,R,17,250], R=2 or 4, and tau is float32 [B,R].  All-zero
    tau returns the original array; zero-shift records in mixed batches are
    copied exactly.  The caller's input arrays are never modified.
    """
    if not isinstance(trials, np.ndarray) or trials.ndim != 4 or trials.shape[1] not in (2, 4):
        raise ValueError('expected two or four repeats')
    _validate_trials(trials, trials.shape[1])
    if not isinstance(tau, np.ndarray) or tau.dtype != np.float32 or tau.shape != trials.shape[:2]:
        raise ValueError('tau must be float32 [B,R]')
    if not np.isfinite(tau).all():
        raise ValueError('nonfinite tau')
    if np.all(tau == 0.):
        return trials
    position = np.arange(TIME_POINTS, dtype=np.float64)[None, None, :] + tau.astype(np.float64)[..., None]
    # Clipping before integer conversion also handles arbitrarily large finite shifts.
    low_float = np.floor(position)
    low = np.clip(low_float, -1., TIME_POINTS).astype(np.intp)
    high = low + 1
    fraction = (position - low_float).astype(np.float32)
    shape = trials.shape
    low_values = np.take_along_axis(trials, np.broadcast_to(np.clip(low, 0, TIME_POINTS - 1)[:, :, None, :], shape), axis=-1)
    high_values = np.take_along_axis(trials, np.broadcast_to(np.clip(high, 0, TIME_POINTS - 1)[:, :, None, :], shape), axis=-1)
    low_valid = ((low_float >= 0.) & (low_float < TIME_POINTS))[:, :, None, :]
    high_valid = ((low_float + 1. >= 0.) & (low_float + 1. < TIME_POINTS))[:, :, None, :]
    result = (np.float32(1.) - fraction[:, :, None, :]) * np.where(low_valid, low_values, np.float32(0.))
    result += fraction[:, :, None, :] * np.where(high_valid, high_values, np.float32(0.))
    result = np.where((tau == 0.)[:, :, None, None], trials, result)
    return result.astype(np.float32, copy=False)


def cyclic_wrong_tau(tau):
    """Matched control: cyclically reassign the four estimated shifts."""
    if not isinstance(tau, np.ndarray) or tau.dtype != np.float32 or tau.ndim != 2 or tau.shape[1] != 4 or not np.isfinite(tau).all():
        raise ValueError('expected finite float32 [B,4] tau')
    return np.roll(tau, 1, axis=1)


def pair_align(pair_trials, channels=None):
    """Register a physical pair internally: tau=(-d/2,+d/2)."""
    _validate_trials(pair_trials, 2)
    estimate = estimate_pair_lag(pair_trials[:, 0], pair_trials[:, 1], channels)
    half = estimate['lag'].astype(np.float32) / np.float32(2.)
    tau = np.stack((-half, half), axis=1)
    aligned = shift_records(pair_trials, tau)
    return dict(estimate, tau=tau, aligned=aligned,
                mean=aligned.mean(axis=1, dtype=np.float32))


def four_align(trials, channels=None):
    """Convenience wrapper; official four-repeat estimation uses all 17 channels."""
    estimate = estimate_relative_shifts(trials, channels)
    aligned = shift_records(trials, estimate['tau'])
    wrong_tau = cyclic_wrong_tau(estimate['tau'])
    wrong_aligned = shift_records(trials, wrong_tau)
    return dict(estimate, aligned=aligned, mean=aligned.mean(axis=1, dtype=np.float32),
                baseline=trials.mean(axis=1, dtype=np.float32), wrong_tau=wrong_tau,
                wrong_mean=wrong_aligned.mean(axis=1, dtype=np.float32))
