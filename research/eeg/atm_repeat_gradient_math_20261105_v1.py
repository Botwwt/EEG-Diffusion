"""Actual output LayerNorm derivatives, with fixed R3 interpolation controls."""
import numpy as np

OBJECTIVES = ('raw_MSE', 'one_minus_cosine', 'within10_CE_temperature1')
METHODS = ('linear_R2_R4', 'quadratic_R1_R2_R4', 'constant_R4')
GRADIENT_METRICS = ('absolute_L2_error', 'relative_L2_error', 'cosine_to_actual_R3')


def loss_affine_gradients(p, targets10, index, gamma, beta):
    p = np.asarray(p, np.float64)
    y10 = np.asarray(targets10, np.float64)
    n = len(p)
    y = y10[np.arange(n), index]
    radius = np.linalg.norm(p, axis=1, keepdims=True)
    unit = p / radius
    visual = y10 / np.linalg.norm(y10, axis=2, keepdims=True)
    true = visual[np.arange(n), index]
    cosine = np.sum(unit * true, axis=1)
    scores = np.einsum('nd,nkd->nk', unit, visual)
    mx = scores.max(1, keepdims=True)
    prob = np.exp(scores-mx)
    total = prob.sum(1, keepdims=True)
    prob /= total
    gradient_unit = np.einsum('nk,nkd->nd', prob, visual) - true
    gs = [2*(p-y)/1024,
          -(true-cosine[:, None]*unit)/radius,
          (gradient_unit-(gradient_unit*unit).sum(1, keepdims=True)*unit)/radius]
    loss = np.stack([np.square(p-y).mean(1), 1-cosine,
                     mx[:, 0]+np.log(total[:, 0])-scores[np.arange(n), index]], axis=1)
    normalized_input = (p-beta)/gamma
    gradient = np.stack([np.concatenate((g*normalized_input, g), axis=1) for g in gs], axis=1)
    return loss, gradient


def compare_gradient_counts(count_gradients):
    g = np.asarray(count_gradients, np.float64)  # query,count,objective,2048
    true = g[:, 2]
    variants = [g[:, 1]/3+2*g[:, 3]/3, (-g[:, 0]+12*g[:, 1]+16*g[:, 3])/27, g[:, 3]]
    result = []
    for pred in variants:
        error = np.linalg.norm(pred-true, axis=-1)
        norm = np.linalg.norm(true, axis=-1)
        pnorm = np.linalg.norm(pred, axis=-1)
        assert np.all(norm > 0) and np.all(pnorm > 0)
        result.append(np.stack((error, error/norm, (pred*true).sum(-1)/(pnorm*norm)), axis=-1))
    return np.stack(result, axis=2)  # query,objective,method,metric


def gradient_summary(values):
    assert values.shape == (13230, 3, 3, 3) and np.isfinite(values).all()
    groups = values.reshape(1323, 10, 3, 3, 3).mean(1)
    draws = np.random.default_rng(1729).integers(0, 1323, (1000, 1323))
    result = {}
    for j, objective in enumerate(OBJECTIVES):
        result[objective] = {}
        for k, method in enumerate(METHODS):
            result[objective][method] = {}
            for m, metric in enumerate(GRADIENT_METRICS):
                x = groups[:, j, k, m]
                delta = x-groups[:, j, 2, m]
                result[objective][method][metric] = dict(mean=float(x.mean()),
                    descriptive_concept_bootstrap95=np.quantile(x[draws].mean(1), [.025,.975]).tolist(),
                    minus_constant=float(delta.mean()),
                    minus_constant_descriptive95=np.quantile(delta[draws].mean(1), [.025,.975]).tolist())
    return result
