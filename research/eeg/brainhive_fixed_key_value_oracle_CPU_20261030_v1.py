"""Privileged fixed-Q/K value capacity bounds, not an EEG prediction method.

Solve separate least-squares problems for each saved query/layer/time/head,
using its true image teacher response. Such oracle solutions are unavailable
at normal inference and may not be jointly realizable by a shared projector.
"""
import numpy as np


def attention(q, k, v):
    logits = q @ k.swapaxes(-1, -2) / np.sqrt(k.shape[-1])
    w = np.exp(logits - logits.max(axis=-1, keepdims=True))
    w /= w.sum(axis=-1, keepdims=True)
    return w @ v, w


def centered_basis():
    b = np.zeros((4, 3), dtype=np.float64)
    for j in range(3):
        scale = np.sqrt((j + 1) * (j + 2))
        b[:j + 1, j] = 1 / scale
        b[j + 1, j] = -(j + 1) / scale
    return b


def solve_value_oracles(weights, values, target, *, rcond=1e-10):
    assert weights.shape[-1] == values.shape[-2] == 4
    output = weights @ values
    difference = target - output
    b = centered_basis()
    design = weights @ b
    u, sv, vh = np.linalg.svd(design, full_matrices=False)
    weight_scale = np.linalg.svd(weights, compute_uv=False)[..., :1]
    # Scale the cutoff to the full attention operator. A numerically uniform
    # row must not turn rounding-sized centered singular values into a route.
    retained = sv > rcond * weight_scale
    inverse_sv = np.divide(1., sv, out=np.zeros_like(sv), where=retained)
    inverse = (vh.swapaxes(-1, -2) * inverse_sv[..., None, :]) @ u.swapaxes(-1, -2)
    coefficients = inverse @ difference
    centered_update = b @ coefficients
    free_update = np.linalg.pinv(weights, rcond=rcond) @ difference
    mean_update = np.broadcast_to(difference.mean(-2, keepdims=True), values.shape).copy()
    result = {}
    base_error = float(np.square(difference).mean())
    base_norm = max(float(np.linalg.norm(values)), 1e-30)
    for label, update in [('mean_only', mean_update), ('centered_only', centered_update), ('free_value', free_update)]:
        residual = difference - weights @ update
        error = float(np.square(residual).mean())
        assert error <= base_error + 1e-9 * max(1., base_error)
        result[label + '_remaining_MSE'] = error
        result[label + '_relative_value_update_norm'] = float(np.linalg.norm(update) / base_norm)
    assert result['free_value_remaining_MSE'] <= min(result['mean_only_remaining_MSE'], result['centered_only_remaining_MSE']) + 1e-9 * max(1., base_error)
    ranks = retained.sum(-1)
    result.update(base_MSE=base_error,
                  centered_update_token_mean_max=float(np.abs(centered_update.mean(-2)).max()),
                  centered_update_token_mean_relative=float(np.linalg.norm(centered_update.mean(-2)) / max(np.linalg.norm(centered_update), 1e-30)),
                  centered_design_min_rank=int(ranks.min()), centered_design_max_rank=int(ranks.max()),
                  centered_design_smallest_to_largest_singular_min=float((sv[..., -1] / np.maximum(sv[..., 0], 1e-300)).min()))
    assert result['centered_update_token_mean_relative'] < 1e-12
    assert all(np.isfinite(x) for x in result.values())
    return result


def describe_trace(q, k, v, correct):
    assert q.shape[0] == 1 and q.shape[-2] == 16
    assert k.shape == v.shape and k.shape[0] == 21 and k.shape[-2] == 4
    q, k, v = (np.asarray(x, dtype=np.float64) for x in (q, k, v))
    output, weights = attention(q, k, v)
    return solve_value_oracles(weights[0], v[0], output[11 + correct])


def mathematical_checks():
    rng = np.random.default_rng(1729)
    b = centered_basis()
    assert np.allclose(b.T @ b, np.eye(3), atol=1e-14)
    assert np.allclose(b.sum(0), 0, atol=1e-14)
    logits = rng.normal(size=(2, 16, 4))
    a = np.exp(logits); a /= a.sum(-1, keepdims=True)
    v = rng.normal(size=(2, 4, 7))
    delta = b @ rng.normal(size=(2, 3, 7))
    exact = solve_value_oracles(a, v, a @ (v + delta))
    assert exact['centered_only_remaining_MSE'] < 1e-24
    constant = rng.normal(size=(2, 1, 7))
    needs_mean = solve_value_oracles(a, v, a @ v + constant)
    assert needs_mean['mean_only_remaining_MSE'] < 1e-24
    assert needs_mean['centered_only_remaining_MSE'] > .001
    uniform = np.full((2, 16, 4), .25)
    zero_center_route = solve_value_oracles(uniform, v, uniform @ v + constant)
    assert zero_center_route['centered_design_max_rank'] == 0
    assert np.isclose(zero_center_route['centered_only_remaining_MSE'], zero_center_route['base_MSE'])
    assert zero_center_route['free_value_remaining_MSE'] < 1e-24
    return dict(status='PASS_PRIVILEGED_VALUE_ORACLE_CAPACITY_CHECKS', checks=5,
                centered_reachable_example_MSE=exact['centered_only_remaining_MSE'],
                constant_error_blocked_by_centering_MSE=needs_mean['centered_only_remaining_MSE'],
                uniform_attention_centered_path_cannot_change_output=True,
                no_actual_EEG_model_or_GPU_used=True)
