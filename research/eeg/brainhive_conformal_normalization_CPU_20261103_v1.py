"""Classical conformal sphere normalization. No novelty or efficacy is implied.

Arrays contain row vectors. T_a is Rezende2020 Eq8 with omega=-a.
The barycenter potential is Cantarella-Schumacher2020 Eq25. We use finite
Euclidean L-BFGS, not their Riemannian Newton algorithm or convergence theorem.
"""
import numpy as np
from scipy.optimize import minimize


def unit(x):
    x = np.asarray(x, np.float64)
    lengths = np.linalg.norm(x, axis=-1, keepdims=True)
    assert np.isfinite(x).all() and np.all(lengths > 1e-12)
    return x / lengths


def zero_mean_basis(d):
    one = np.ones(d, np.float64) / np.sqrt(d)
    v = -one
    v[-1] += 1
    v /= np.linalg.norm(v)
    return (np.eye(d) - 2 * v[:, None] * v[None, :])[:, :-1]


def mobius(x, a):
    x, a = np.asarray(x, np.float64), np.asarray(a, np.float64)
    assert x.shape[-1] == a.size and a.ndim == 1
    aa = float(a @ a)
    assert np.isfinite(x).all() and np.isfinite(a).all() and aa < 1
    if aa == 0:
        return x.copy()
    dot = x @ a
    denominator = 1 + aa + 2 * dot
    assert np.all(denominator > 1e-14)
    return ((1-aa) * x + 2 * (1+dot)[..., None] * a) / denominator[..., None]


def conformal_scale(x, a):
    aa = float(a @ a)
    return (1-aa) / (1+aa+2*(x@a))


def barycenter_potential(a, x):
    aa = float(a @ a)
    assert aa < 1
    denominator = 1 + aa - 2 * (x @ a)
    assert np.all(denominator > 0)
    loss = float(np.log(denominator).mean() - np.log1p(-aa))
    gradient = 2 * (a * np.mean(1/denominator) - (x/denominator[:, None]).mean(0)) + 2*a/(1-aa)
    return loss, gradient


def fit_barycenter(x, max_iterations=80):
    x = unit(x)
    assert x.ndim == 2 and max_iterations == 80
    events = []

    def objective(v):
        scale = np.sqrt(1 + v @ v)
        a = v / scale
        loss, grad = barycenter_potential(a, x)
        gv = (grad - a * (a @ grad)) / scale
        events.append(dict(loss=loss, gradient_norm=float(np.linalg.norm(gv)), norm=float(np.linalg.norm(a))))
        return loss, gv

    solved = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=80, ftol=1e-14, gtol=1e-9, maxls=30))
    a = solved.x / np.sqrt(1 + solved.x @ solved.x)
    residual = float(np.linalg.norm(mobius(x, -a).mean(0)))
    proof = dict(success=bool(solved.success), message=str(solved.message), iterations=int(solved.nit),
                 evaluations=len(events), norm=float(np.linalg.norm(a)),
                 centered_mean_norm=residual, converged_center=residual < 1e-6,
                 inverse_scale_bound=float(np.exp(2*np.arctanh(np.linalg.norm(a)))),
                 optimization_events=events)
    return a, proof


def orthogonal_alignment(x, y):
    cross = y.T @ x
    left, singular, right = np.linalg.svd(cross, full_matrices=False)
    rotation = left @ right
    return rotation, dict(singular_min=float(singular.min()), singular_max=float(singular.max()),
                          singular_sum=float(singular.sum()),
                          optimality_gap=float(singular.sum() - np.sum((x @ rotation.T) * y)),
                          orthogonal_error=float(np.max(np.abs(rotation.T@rotation - np.eye(rotation.shape[0])))))


def transform(x, model, inverse=False):
    if inverse:
        return mobius(mobius(x, -model['target_center']) @ model['rotation'], model['EEG_center'])
    return mobius(mobius(x, -model['EEG_center']) @ model['rotation'].T, model['target_center'])


def fit_pair(x, y, permutation):
    x, y = unit(x), unit(y)
    ax, px = fit_barycenter(x)
    ay, py = fit_barycenter(y)
    proof = dict(EEG_barycenter=px, target_barycenter=py)
    if not px['converged_center'] or not py['converged_center']:
        return {}, proof
    xx, yy = mobius(x, -ax), mobius(y, -ay)
    paired, pp = orthogonal_alignment(xx, yy)
    wrong, pw = orthogonal_alignment(xx, yy[permutation])
    rotation, pr = orthogonal_alignment(x, y)
    identity = np.eye(x.shape[1])
    zero = np.zeros(x.shape[1])
    models = dict(conformal_paired=dict(EEG_center=ax,target_center=ay,rotation=paired),
                  conformal_wrong=dict(EEG_center=ax,target_center=ay,rotation=wrong),
                  rotation_only=dict(EEG_center=zero,target_center=zero,rotation=rotation),
                  concentration_only=dict(EEG_center=ax,target_center=ay,rotation=identity))
    proof.update(paired_rotation=pp, wrong_rotation=pw, ordinary_rotation=pr,
                 global_chord_distortion_bound=float(np.exp(2*np.arctanh(np.linalg.norm(ax)) + 2*np.arctanh(np.linalg.norm(ay)))))
    return models, proof
