"""Read-only fixed-Q attention diagnostics; NumPy only, no model modification.

The value mean and routing remainder are NOT orthogonal. Their squared errors
require a cross term. Query-mean/query-centered errors, by contrast, are
orthogonal. These identities are classical, not a novelty or efficacy claim.
"""
import numpy as np


def attention(q, k, v):
    assert q.ndim == k.ndim == v.ndim == 4 and k.shape == v.shape
    assert q.shape[-1] == k.shape[-1] and q.shape[-3] == k.shape[-3]
    assert all(np.isfinite(x).all() for x in (q, k, v))
    logits = q @ k.swapaxes(-1, -2) / np.sqrt(k.shape[-1])
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v, weights


def parts(q, k, v):
    mean_k, mean_v = k.mean(axis=-2, keepdims=True), v.mean(axis=-2, keepdims=True)
    route, weights = attention(q, k - mean_k, v - mean_v)
    return mean_v, route, weights


def mse(x):
    return float(np.square(x).mean())


def describe_trace(q, k, v, correct):
    assert q.shape[0] == 1 and q.shape[-2] == 16
    assert k.shape == v.shape and k.shape[0] == 21 and k.shape[-2] == 4
    assert 0 <= correct < 10
    q, k, v = (np.asarray(x, dtype=np.float64) for x in (q, k, v))
    output, weights = attention(q, k, v)
    mean_v, route, centered_weights = parts(q, k, v)
    identity_error = float(np.abs(output - (mean_v + route)).max())
    weight_error = float(np.abs(weights - centered_weights).max())
    teacher = 11 + correct
    err_mean = mean_v[0] - mean_v[teacher]
    err_route = route[0] - route[teacher]
    err = output[0] - output[teacher]
    cross = float(2 * (err_mean * err_route).mean())
    query_mean = err.mean(axis=-2, keepdims=True)
    query_centered = err - query_mean
    # Two-order mean attribution; these are teacher K/V substitutions under
    # a frozen observed Q, not realizable normal-inference interventions.
    oe = output[0:1]
    ot = output[teacher:teacher + 1]
    kte, _ = attention(q, k[teacher:teacher + 1], v[0:1])
    vet, _ = attention(q, k[0:1], v[teacher:teacher + 1])
    delta_k = .5 * ((kte - oe) + (ot - vet))
    delta_v = .5 * ((vet - oe) + (ot - kte))
    total_delta = ot - oe
    squared_total = float(np.square(total_delta).sum())
    if squared_total == 0:
        key_projection_share = value_projection_share = None
    else:
        key_projection_share = float((delta_k * total_delta).sum() / squared_total)
        value_projection_share = float((delta_v * total_delta).sum() / squared_total)
    # Projection shares can be negative or >1. They are not nonnegative causal
    # percentages. Include the key/value interaction instead of hiding it.
    distances = np.square(output[11:21] - output[0]).mean(axis=(1, 2, 3))
    mean_distances = np.square(mean_v[11:21] - mean_v[0]).mean(axis=(1, 2, 3))
    route_distances = np.square(route[11:21] - route[0]).mean(axis=(1, 2, 3))
    def rank(values):
        target = values[correct]
        others = np.delete(values, correct)
        return dict(top1=float(np.argmin(values) == correct),
                    AFC=float(np.mean((others > target) + .5 * (others == target))))
    centered_v = v - mean_v
    sv = np.linalg.svd(centered_v[[0, teacher]], compute_uv=False)
    scale = np.maximum(sv[..., 0], 1e-30)
    record = dict(
        max_decomposition_error=identity_error, max_common_key_shift_weight_error=weight_error,
        teacher_error_total_MSE=mse(err), teacher_error_value_mean_MSE=mse(err_mean),
        teacher_error_route_MSE=mse(err_route), teacher_error_cross_term=cross,
        error_energy_identity_error=abs(mse(err) - mse(err_mean) - mse(err_route) - cross),
        teacher_error_query_constant_MSE=mse(query_mean),
        teacher_error_query_centered_MSE=mse(query_centered),
        orthogonal_error_identity_error=abs(mse(err) - mse(query_mean) - mse(query_centered)),
        key_projection_share=key_projection_share, value_projection_share=value_projection_share,
        key_value_attribution_closure_max_error=float(np.abs(delta_k + delta_v - total_delta).max()),
        key_value_interaction_MSE=mse(ot - kte - vet + oe),
        EEG_normalized_attention_entropy=float((-weights[0] * np.log(weights[0].clip(1e-300))).sum(-1).mean() / np.log(4)),
        teacher_normalized_attention_entropy=float((-weights[teacher] * np.log(weights[teacher].clip(1e-300))).sum(-1).mean() / np.log(4)),
        EEG_total_response_energy=mse(output[0]), EEG_route_energy=mse(route[0]),
        teacher_total_response_energy=mse(output[teacher]), teacher_route_energy=mse(route[teacher]),
        centered_value_fourth_to_first_singular_max=float((sv[..., 3] / scale).max()),
        total_response_within_concept=rank(distances),
        value_mean_within_concept=rank(mean_distances),
        route_within_concept=rank(route_distances))
    assert identity_error < 1e-10 and weight_error < 1e-12
    assert record['error_energy_identity_error'] < 1e-9
    assert record['orthogonal_error_identity_error'] < 1e-9
    assert record['key_value_attribution_closure_max_error'] < 1e-10
    assert record['centered_value_fourth_to_first_singular_max'] < 1e-12
    return record


def mathematical_checks():
    rng = np.random.default_rng(1729)
    q = rng.normal(size=(1, 2, 16, 8))
    k = rng.normal(size=(21, 2, 4, 8))
    v = rng.normal(size=k.shape)
    record = describe_trace(q, k, v, 0)
    out, _ = attention(q, k, v)
    shift = rng.normal(size=(21, 2, 1, 8))
    shifted_k, _ = attention(q, k + shift, v)
    shifted_v, _ = attention(q, k, v + shift)
    permuted, _ = attention(q, k[:, :, [3, 1, 0, 2]], v[:, :, [3, 1, 0, 2]])
    assert np.allclose(shifted_k, out, atol=1e-12, rtol=0)
    assert np.allclose(shifted_v, out + shift, atol=1e-12, rtol=0)
    assert np.allclose(permuted, out, atol=1e-12, rtol=0)
    constant = np.broadcast_to(v.mean(-2, keepdims=True), v.shape).copy()
    collapsed, _ = attention(q, k, constant)
    assert np.allclose(collapsed, constant[:, :, :1], atol=1e-12, rtol=0)
    # A zero-sum token update preserves the value mean but can alter attention.
    v_change = v + (k - k.mean(-2, keepdims=True))
    changed, _ = attention(q, k, v_change)
    assert np.allclose(v_change.mean(-2), v.mean(-2), atol=1e-12, rtol=0)
    assert np.max(np.abs(changed - out)) > .01
    assert abs(record['teacher_error_cross_term']) > 1e-5
    assert record['key_value_interaction_MSE'] > 1e-5
    return dict(status='PASS_ATTENTION_IDENTITIES_AND_COUNTEREXAMPLES', checks=9,
                maximum_identity_error=record['max_decomposition_error'],
                mean_preservation_does_not_preserve_attention=True,
                mean_and_route_errors_not_orthogonal=True,
                key_value_attributions_not_causal_shares=True,
                GPU_or_training_or_real_data_used=False)
