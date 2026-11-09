"""One fixed finite-budget historical capacity diagnostic, NumPy only.

Solve min .5||D X-T||_F^2 with ||X||_F <= radius, across all heads with one
common multiplier. The true-image teacher is privileged. This is not a shared
EEG predictor, a bounded neural-parameter update, or a normal inference method.
"""
import numpy as np


def attention(q, k, v):
    logits = q @ k.swapaxes(-1, -2) / np.sqrt(k.shape[-1])
    a = np.exp(logits - logits.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    return a @ v, a


def centered_basis():
    b = np.zeros((4, 3), dtype=np.float64)
    for j in range(3):
        scale = np.sqrt((j + 1) * (j + 2))
        b[:j + 1, j] = 1 / scale
        b[j + 1, j] = -(j + 1) / scale
    return b


def trust_region_least_squares(design, target, radius, *, cutoff_scale=None):
    assert radius >= 0 and np.isfinite(radius)
    assert np.isfinite(design).all() and np.isfinite(target).all()
    u, singular, vh = np.linalg.svd(design, full_matrices=False)
    scale = singular[..., :1] if cutoff_scale is None else cutoff_scale
    keep = singular > 1e-10 * scale
    projected = u.swapaxes(-1, -2) @ target
    effective_singular = np.where(keep, singular, 0.)
    inv = np.divide(1., singular, out=np.zeros_like(singular), where=keep)
    unconstrained_spectral = inv[..., None] * projected
    unconstrained_norm = float(np.linalg.norm(unconstrained_spectral))
    if radius == 0:
        value = np.zeros(design.shape[:-2] + (design.shape[-1], target.shape[-1]))
        return value, dict(multiplier=None, active=True, update_norm=0.,
                           stationarity_scaled=None, complementarity_scaled=None,
                           primal_violation=0., degenerate_zero_radius=True)
    if unconstrained_norm <= radius:
        multiplier = 0.
        spectral = unconstrained_spectral
    else:
        # ||X_lambda|| <= ||D^T T||/lambda brackets the common multiplier.
        low = 0.
        high = float(np.linalg.norm(effective_singular[..., None] * projected)) / radius
        assert high > 0 and np.isfinite(high)
        def coefficients(lam):
            factor = effective_singular / (effective_singular**2 + lam)
            return factor[..., None] * projected
        assert np.linalg.norm(coefficients(high)) <= radius * (1 + 1e-12)
        for _ in range(72):
            mid = (low + high) / 2
            if np.linalg.norm(coefficients(mid)) > radius:
                low = mid
            else:
                high = mid
        multiplier = high
        spectral = coefficients(multiplier)
    value = vh.swapaxes(-1, -2) @ spectral
    residual = design @ value - target
    norm = float(np.linalg.norm(value))
    stationarity = design.swapaxes(-1, -2) @ residual + multiplier * value
    stationarity_scaled = float(np.linalg.norm(stationarity) /
                                (1 + np.linalg.norm(design) * np.linalg.norm(target) + multiplier * norm))
    primal = max(0., norm - radius) / (1 + radius)
    complementarity = float(multiplier * abs(norm**2 - radius**2) / (1 + multiplier * radius**2))
    assert primal < 1e-12 and stationarity_scaled < 1e-10 and complementarity < 1e-10
    return value, dict(multiplier=float(multiplier), active=bool(multiplier > 0),
                       update_norm=norm, stationarity_scaled=stationarity_scaled,
                       complementarity_scaled=complementarity, primal_violation=primal,
                       degenerate_zero_radius=False)


def describe_trace(q, k, v, correct):
    assert q.shape[0] == 1 and q.shape[-2] == 16
    assert k.shape == v.shape and k.shape[0] == 21 and k.shape[-2] == 4
    q, k, v = (np.asarray(x, dtype=np.float64) for x in (q, k, v))
    output, weights = attention(q, k, v)
    a, original = weights[0], v[0]
    target = output[11 + correct] - output[0]
    radius = float(np.linalg.norm(original))  # Exactly ONE predeclared budget.
    scale = np.linalg.svd(a, compute_uv=False)[..., :1]
    bases = [('mean_only', np.ones((4, 1)) / 2),
             ('centered_only', centered_basis()), ('free_value', np.eye(4))]
    result = dict(base_MSE=float(np.square(target).mean()), radius_relative_to_original_value=1.,
                  original_value_F_norm=radius, budget_is_response_value_not_EEG_parameter_norm=True)
    diagnostics = []
    for name, basis in bases:
        design = a @ basis
        coefficient, diagnostic = trust_region_least_squares(design, target, radius, cutoff_scale=scale)
        update = basis @ coefficient
        error = float(np.square(target - a @ update).mean())
        result[name + '_remaining_MSE'] = error
        result[name + '_relative_update_norm'] = float(np.linalg.norm(update) / radius) if radius else 0.
        result[name + '_multiplier'] = diagnostic['multiplier']
        result[name + '_budget_active'] = diagnostic['active']
        assert error <= result['base_MSE'] + 1e-9 * max(1., result['base_MSE'])
        if name == 'centered_only':
            result['centered_update_token_mean_relative'] = float(np.linalg.norm(update.mean(-2)) / max(np.linalg.norm(update), 1e-300))
            assert result['centered_update_token_mean_relative'] < 1e-12
        diagnostics.append(diagnostic)
    assert result['free_value_remaining_MSE'] <= min(result['mean_only_remaining_MSE'], result['centered_only_remaining_MSE']) + 1e-9 * max(1., result['base_MSE'])
    result.update(KKT_stationarity_scaled_max=max((x['stationarity_scaled'] or 0.) for x in diagnostics),
                  KKT_complementarity_scaled_max=max((x['complementarity_scaled'] or 0.) for x in diagnostics),
                  primal_violation_max=max(x['primal_violation'] for x in diagnostics))
    return result


def mathematical_checks():
    rng = np.random.default_rng(1729)
    d = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    t = rng.normal(size=(2, 4, 7))
    radius = .3 * np.linalg.norm(t)
    got, cert = trust_region_least_squares(d, t, radius)
    analytic = radius * t / np.linalg.norm(t)
    error = float(np.abs(got - analytic).max())
    assert error < 1e-12 and abs(cert['multiplier'] - (np.linalg.norm(t) / radius - 1)) < 1e-12
    free, inactive = trust_region_least_squares(d, t, 2 * np.linalg.norm(t))
    assert np.allclose(free, t, atol=1e-12) and not inactive['active']
    zero, degenerate = trust_region_least_squares(d, t, 0.)
    assert np.count_nonzero(zero) == 0 and degenerate['degenerate_zero_radius']
    design = rng.normal(size=(2, 16, 3))
    target = rng.normal(size=(2, 16, 7))
    value, certificate = trust_region_least_squares(design, target, .1)
    lam = certificate['multiplier']
    direct = np.linalg.solve(design.swapaxes(-1, -2) @ design + lam * np.eye(3), design.swapaxes(-1, -2) @ target)
    assert np.max(np.abs(direct - value)) < 1e-12
    b = centered_basis()
    uniform = np.full((2, 16, 4), .25)
    flat, flat_cert = trust_region_least_squares(uniform @ b, target, 1., cutoff_scale=np.linalg.svd(uniform, compute_uv=False)[..., :1])
    assert np.count_nonzero(flat) == 0 and not flat_cert['active']
    return dict(status='PASS_FINITE_RADIUS_VALUE_ORACLE_CPU_CHECKS', checks=5,
                analytic_identity_solution_max_error=error,
                direct_normal_equation_max_error=float(np.abs(direct - value).max()),
                one_global_multiplier_across_heads=True, zero_budget_preserved=True,
                active_and_inactive_constraints_verified=True, uniform_centered_route_is_zero=True,
                actual_EEG_models_or_GPU_used=False)
