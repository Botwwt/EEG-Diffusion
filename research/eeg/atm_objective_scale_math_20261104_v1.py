"""Classical MSE/temperature factorial; attribution controls, not a new method."""
import numpy as np

CELLS = {'nce_continued': (0., 'exp_clamped'),
         'nce_literal': (0., 'literal'),
         'mse9_literal': (9., 'literal'),
         'mse9_exp': (9., 'exp_clamped')}


def numpy_loss_gradient(prediction, target, ell, arm):
    x, y = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    assert x.shape == y.shape and x.ndim == 2 and np.isfinite(x).all() and np.isfinite(y).all()
    weight, kind = CELLS[arm]; n, d = x.shape
    xn, yn = np.linalg.norm(x, axis=1, keepdims=True), np.linalg.norm(y, axis=1, keepdims=True)
    assert xn.min() > 1e-9 and yn.min() > 1e-9
    u, v = x/xn, y/yn
    if kind == 'literal':
        scale, scale_derivative = float(ell), 1.
    else:
        exponential = np.exp(float(ell)); scale = min(exponential, 100.)
        scale_derivative = exponential if exponential <= 100 else 0.
    cosine = u @ v.T; logits = scale*cosine; i = np.arange(n)
    def ce(z):
        shifted = z-z.max(1, keepdims=True)
        p = np.exp(shifted); denominator = p.sum(1, keepdims=True); p /= denominator
        value = np.mean(np.log(denominator[:, 0])-shifted[i, i])
        p[i, i] -= 1
        return value, p/n
    row, gr = ce(logits); col, gc = ce(logits.T)
    derivative = .5*(gr+gc.T)
    du = scale*(derivative @ v)
    dx = (du-u*(du*u).sum(1, keepdims=True))/xn + weight*2*(x-y)/(n*d)
    mse = float(np.square(x-y).mean()); clip = float(.5*(row+col))
    return dict(loss=weight*mse+clip, raw_MSE=mse, clip_loss=clip,
                prediction_gradient=dx, ell_gradient=float((derivative*cosine).sum()*scale_derivative),
                actual_multiplier=float(scale), cosine_logits=logits)


def torch_loss(prediction, target, ell, arm):
    import torch
    import torch.nn.functional as F
    weight, kind = CELLS[arm]
    scale = ell if kind == 'literal' else ell.exp().clamp_max(100)
    logits = scale*F.normalize(prediction, dim=1) @ F.normalize(target, dim=1).T
    i = torch.arange(len(logits), device=logits.device)
    clip = .5*(F.cross_entropy(logits, i)+F.cross_entropy(logits.T, i))
    mse = F.mse_loss(prediction, target)
    return weight*mse+clip, dict(raw_MSE=mse, clip_loss=clip,
        mean_output_norm=prediction.norm(dim=1).mean(), logit_parameter=ell,
        actual_multiplier=scale)
