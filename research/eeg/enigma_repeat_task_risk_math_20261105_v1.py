"""Finite recorded-subset task diagnostics; extrapolation is a hypothesis only."""
import itertools
from fractions import Fraction
import numpy as np

SUBSETS = tuple(s for n in (1, 2, 3, 4) for s in itertools.combinations(range(4), n))
COUNTS = np.asarray([len(s) for s in SUBSETS], dtype=np.int64)
METRICS = ('raw_visual_MSE', 'target_cosine', 'within_concept10_top1',
           'within_concept10_2AFC', 'condition_norm')


def lagrange_weights(counts, destination):
    xs = [Fraction(1, n) for n in counts]
    x = Fraction(1, destination)
    result = []
    for i, xi in enumerate(xs):
        w = Fraction(1)
        for j, xj in enumerate(xs):
            if i != j:
                w *= (x - xj) / (xi - xj)
        result.append(w)
    return tuple(result)


def task_metrics(prediction, targets10, correct_index):
    p = np.asarray(prediction, np.float64)
    y = np.asarray(targets10, np.float64)
    ix = np.asarray(correct_index, np.int64)
    assert p.ndim == 2 and p.shape[1] == 1024
    assert y.shape == (len(p), 10, 1024) and ix.shape == (len(p),)
    assert np.isfinite(p).all() and np.isfinite(y).all()
    pn = np.linalg.norm(p, axis=1)
    yn = np.linalg.norm(y, axis=2)
    assert np.all(pn > 0) and np.all(yn > 0)
    target = y[np.arange(len(p)), ix]
    scores = np.einsum('nd,nkd->nk', p / pn[:, None], y / yn[:, :, None])
    positive = scores[np.arange(len(p)), ix]
    afc = ((scores < positive[:, None]).sum(1) +
           .5 * ((scores == positive[:, None]).sum(1) - 1)) / 9
    values = np.column_stack((np.square(p - target).mean(1), positive,
                              scores.argmax(1) == ix, afc, pn))
    assert values.shape == (len(p), 5) and np.isfinite(values).all()
    return values


def summaries(per_view, loss_of_mean, dispersion):
    a = np.asarray(per_view, np.float64)
    b = np.asarray(loss_of_mean, np.float64)
    d = np.asarray(dispersion, np.float64)
    assert a.shape == (13230, 15, 5) and b.shape == (13230, 4, 5)
    assert d.shape == (13230, 4) and np.isfinite(a).all() and np.isfinite(b).all()
    count_means = np.stack([a[:, COUNTS == n].mean(1) for n in (1, 2, 3, 4)], axis=1)
    gap = count_means[:, :, 0] - b[:, :, 0]
    identity_error = float(np.max(np.abs(gap - d)))
    assert identity_error < 2e-12 and d.min() > -2e-12
    draws = np.random.default_rng(1729).integers(0, 1323, (1000, 1323))

    def summary(x):
        x = np.asarray(x, np.float64)
        assert x.shape == (13230,) and np.isfinite(x).all()
        groups = x.reshape(1323, 10).mean(1)
        return dict(mean=float(x.mean()), descriptive_concept_bootstrap95=
                    np.quantile(groups[draws].mean(1), [.025, .975]).tolist(),
                    positive_concepts=int((groups > 0).sum()), concepts=1323,
                    negative_fraction=float((x < 0).mean()))

    result = dict(by_count={}, count4_minus_count1={}, held_R3={}, prospective_R80={},
                  mean_prediction_vs_mean_loss={}, jensen_identity_max_error=identity_error)
    for j, n in enumerate((1, 2, 3, 4)):
        result['by_count'][str(n)] = {key: summary(count_means[:, j, k])
                                      for k, key in enumerate(METRICS)}
        result['mean_prediction_vs_mean_loss'][str(n)] = dict(
            loss_of_mean_prediction=summary(b[:, j, 0]),
            mean_per_view_loss=summary(count_means[:, j, 0]),
            prediction_dispersion_per_dimension=summary(d[:, j]))
    for k, key in enumerate(METRICS):
        result['count4_minus_count1'][key] = summary(count_means[:, 3, k] - count_means[:, 0, k])
    # Prospective models are fixed before reading the held R3 values. Ranks are not extrapolated.
    for key, risks in [('raw_visual_MSE', count_means[:, :, 0]),
                       ('one_minus_target_cosine', 1 - count_means[:, :, 1])]:
        linear3 = risks[:, 1] / 3 + 2 * risks[:, 3] / 3
        quadratic3 = -risks[:, 0] / 27 + 4 * risks[:, 1] / 9 + 16 * risks[:, 3] / 27
        actual3 = risks[:, 2]
        result['held_R3'][key] = {}
        for name, value in [('linear_R2_R4', linear3), ('quadratic_R1_R2_R4', quadratic3),
                            ('constant_R4_control', risks[:, 3])]:
            error = value - actual3
            result['held_R3'][key][name] = dict(predicted=summary(value), actual=summary(actual3),
                signed_error=summary(error), absolute_error=summary(abs(error)),
                root_mean_square_error=float(np.sqrt(np.square(error).mean())),
                absolute_error_minus_constant_control=summary(abs(error) - abs(risks[:, 3] - actual3)))
        result['prospective_R80'][key] = {}
        for name, counts in [('linear_R2_R4', (2, 4)), ('quadratic_R1_R2_R4', (1, 2, 4))]:
            weights = lagrange_weights(counts, 80)
            value = sum(float(w) * risks[:, n - 1] for n, w in zip(counts, weights))
            result['prospective_R80'][key][name] = dict(summary=summary(value),
                counts=list(counts), exact_rational_weights=[str(w) for w in weights],
                no_observed_R80_risk_in_this_study=True, not_a_validated_R80_prediction=True)
    return result, count_means
