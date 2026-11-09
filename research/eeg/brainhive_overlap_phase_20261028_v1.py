"""One bounded phase parameter, fitted on EEG samples only; no new decoder."""
import numpy as np


def fit_phase(x, y, previous, following, bound=.5, reversed_gradient=False):
    """Exact constrained LS for a piecewise linear two-tap delay model.

    p(delta)=y+delta*(following-y) for delta>=0, and
    p(delta)=y+delta*(y-previous) for delta<=0. One scalar per pair.
    """
    assert x.shape == y.shape == previous.shape == following.shape and x.ndim == 3
    plus, minus = following - y, y - previous
    if reversed_gradient:
        plus, minus = plus[..., ::-1], minus[..., ::-1]
    error = x - y
    axis = (1, 2)
    p_energy = np.square(plus).sum(axis)
    m_energy = np.square(minus).sum(axis)
    dp = np.clip(np.divide((plus * error).sum(axis), p_energy,
                           out=np.zeros(len(x)), where=p_energy > 1e-30), 0, bound)
    dm = np.clip(np.divide((minus * error).sum(axis), m_energy,
                           out=np.zeros(len(x)), where=m_energy > 1e-30), -bound, 0)
    lp = np.square(error - dp[:, None, None] * plus).sum(axis)
    lm = np.square(error - dm[:, None, None] * minus).sum(axis)
    choose_plus = (lp < lm) | ((lp == lm) & (np.abs(dp) <= np.abs(dm)))
    return np.where(choose_plus, dp, dm)


def predict_phase(y, previous, following, phase, reversed_gradient=False):
    plus, minus = following - y, y - previous
    if reversed_gradient:
        plus, minus = plus[..., ::-1], minus[..., ::-1]
    gradient = np.where(phase[:, None, None] >= 0, plus, minus)
    return y + phase[:, None, None] * gradient


def difference_segment(raw, start, count=64):
    """Both neighboring differences are available from raw[start-1:start+count+2]."""
    assert raw.ndim == 3 and 1 <= start and start + count + 1 < raw.shape[-1]
    segment = np.asarray(raw[..., start - 1:start + count + 2], np.float64)
    differences = segment[..., 1:] - segment[..., :-1]
    return differences[..., 1:-1], differences[..., :-2], differences[..., 2:]


def normalized_errors(x, predictions, base_y):
    denominator = (np.square(x).sum((1, 2)) + np.square(base_y).sum((1, 2))) / 2
    assert np.all(denominator > 1e-30)
    return np.stack([np.square(x - prediction).sum((1, 2)) / denominator
                     for prediction in predictions], axis=-1)
