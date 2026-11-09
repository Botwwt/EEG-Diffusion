"""Fixed-Q attention derivatives; no model fitting or GPU operations."""
import numpy as np


def attention(q, k, v):
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(q.shape[-1])
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    return p @ v, p


def derivatives(q, k, v, target, wk, wv):
    """Differentiate mean response MSE through the K and V token paths.

    Shapes: Q=(H,S,d), K/V=(H,4,d), weights=(H*d,D).
    Q and target are constants. Returned derivatives are token-coordinate
    surrogates at the supplied K/V, not gradients of a sampled-image loss.
    """
    output, p = attention(q, k, v)
    residual = output - target
    go = 2 * residual / residual.size
    gv = p.swapaxes(-1, -2) @ go
    gp = go @ v.swapaxes(-1, -2)
    gs = p * (gp - (p * gp).sum(axis=-1, keepdims=True))
    gk = gs.swapaxes(-1, -2) @ q / np.sqrt(q.shape[-1])
    n = k.shape[-2]
    zk = gk.transpose(1, 0, 2).reshape(n, -1) @ wk
    zv = gv.transpose(1, 0, 2).reshape(n, -1) @ wv
    return dict(loss=float(np.mean(residual**2)), key=zk, value=zv,
                key_tensor=gk, value_tensor=gv)


def statistics(gk, gv):
    a, b = float(np.sum(gk*gk)), float(np.sum(gv*gv))
    cross = float(np.sum(gk*gv))
    total = float(np.sum((gk+gv)**2))
    denom = float(np.sqrt(a*b))
    return dict(key_squared_norm=a, value_squared_norm=b,
                key_value_dot=cross, shared_squared_norm=total,
                key_value_cosine=cross/denom if denom > 0 else None,
                key_token_mean_norm=float(np.linalg.norm(gk.mean(axis=0))),
                value_token_mean_norm=float(np.linalg.norm(gv.mean(axis=0))),
                shared_along_negative_value_derivative=-(b+cross),
                separated_value_along_negative_value_derivative=-b,
                negative_shared_gradient_derivative=-total,
                norm_identity_error=abs(total-(a+b+2*cross)))


def mathematical_checks():
    rng = np.random.default_rng(1729)
    z = rng.normal(size=(4, 9))
    wk, wv = [rng.normal(size=(6, 9)) / 3 for _ in range(2)]
    q = rng.normal(size=(2, 5, 3))
    target = rng.normal(size=q.shape)
    project = lambda a, w: (a @ w.T).reshape(4, 2, 3).transpose(1, 0, 2)
    k, v = project(z, wk), project(z, wv)
    d = derivatives(q, k, v, target, wk, wv)
    direction = rng.normal(size=z.shape)
    direction /= np.linalg.norm(direction)
    h = 1e-5
    def loss(x, y):
        return float(np.mean((attention(q, x, y)[0]-target)**2))
    errors = {}
    for mode in ('key', 'value', 'shared'):
        dk = project(direction, wk) if mode != 'value' else np.zeros_like(k)
        dv = project(direction, wv) if mode != 'key' else np.zeros_like(v)
        fd = (loss(k+h*dk, v+h*dv)-loss(k-h*dk, v-h*dv))/(2*h)
        grad = d[mode] if mode != 'shared' else d['key']+d['value']
        errors[mode] = abs(fd-float(np.sum(grad*direction)))
    shift = np.broadcast_to(rng.normal(size=(2, 1, 3)), k.shape)
    errors['common_key_translation_output'] = float(np.max(np.abs(
        attention(q, k+shift, v)[0]-attention(q, k, v)[0])))
    errors['key_gradient_token_sum'] = float(np.max(np.abs(d['key'].sum(0))))
    errors['norm_decomposition'] = statistics(d['key'], d['value'])['norm_identity_error']
    assert max(errors.values()) < 2e-9, errors
    return dict(status='PASS_LOCAL_SYNTHETIC_ALGEBRA_ONLY', checks=6,
                errors=errors, tolerance=2e-9, no_native_GPU_parity_claim=True)


if __name__ == '__main__':
    import json
    print(json.dumps(mathematical_checks()))
