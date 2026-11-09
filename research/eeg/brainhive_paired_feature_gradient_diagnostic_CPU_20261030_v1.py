"""Closed-form CLIP-feature derivatives; not EEG/P parameter gradients.

Array libraries are supplied after choosing the runtime. All summaries describe
different T pairs at their logged parameter states, not a fixed-model test set.
"""

def values_and_gradients(features, targets, np):
    x = np.asarray(features, dtype=np.float64)
    v = np.asarray(targets, dtype=np.float64)
    assert x.shape == v.shape and x.ndim == 3 and x.shape[1:] == (2, 768)
    assert np.isfinite(x).all() and np.isfinite(v).all()
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    assert (norm > 0).all() and (np.linalg.norm(v, axis=-1) > 0).all()
    g = x / norm; v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    dg = g[:, 0] - g[:, 1]; dv = v[:, 0] - v[:, 1]
    effect = np.sum(dg * dv, axis=-1)
    absolute = 1 - np.sum(g * v, axis=-1).mean(-1)
    pairing = .07 * np.logaddexp(0, -effect / .07)
    s = 1 / (1 + np.exp(effect / .07))
    a = -.5 * v
    p = np.stack([-s[:, None] * dv, s[:, None] * dv], axis=1)
    # Pull back through each feature's normalization, including its true norm.
    a = (a - g * np.sum(g * a, axis=-1, keepdims=True)) / norm
    p = (p - g * np.sum(g * p, axis=-1, keepdims=True)) / norm
    return dict(loss=absolute + pairing, absolute=absolute, pairing=pairing,
                effect=effect, g=g, v=v, dg=dg, dv=dv,
                absolute_feature_gradient=a, pairing_feature_gradient=p)

def finite_difference_checks(features, targets, np):
    x = np.asarray(features, dtype=np.float64)
    v = np.asarray(targets, dtype=np.float64)
    original = values_and_gradients(x, v, np)
    rng = np.random.default_rng(1729)
    direction = rng.standard_normal(x.shape)
    direction /= np.linalg.norm(direction, axis=(1, 2), keepdims=True)
    h = 1e-5
    plus = values_and_gradients(x + h * direction, v, np)
    minus = values_and_gradients(x - h * direction, v, np)
    errors = {}
    for key, grad in (
        ('absolute', original['absolute_feature_gradient']),
        ('pairing', original['pairing_feature_gradient']),
        ('loss', original['absolute_feature_gradient'] + original['pairing_feature_gradient'])):
        numeric = (plus[key] - minus[key]) / (2 * h)
        analytic = np.sum(grad * direction, axis=(1, 2))
        errors[key] = float(np.abs(numeric - analytic).max())
        assert errors[key] < 2e-7
    return dict(actual_pairs=len(x), h=h, maximum_errors=errors,
                parameter_gradient_or_training_update_check=False)

def summarize(features, targets, np):
    z = values_and_gradients(features, targets, np)
    a = z['absolute_feature_gradient']; p = z['pairing_feature_gradient']
    na = np.linalg.norm(a, axis=(1, 2)); nb = np.linalg.norm(p, axis=(1, 2))
    cos = np.divide(np.sum(a * p, axis=(1, 2)), na * nb,
                    out=np.zeros_like(na), where=na * nb > 1e-12)
    ng = np.linalg.norm(z['dg'], axis=1); nv = np.linalg.norm(z['dv'], axis=1)
    angle = np.divide(z['effect'], ng * nv, out=np.zeros_like(ng), where=ng * nv > 1e-12)
    def summary(x):
        assert np.isfinite(x).all()
        return dict(mean=float(np.mean(x)), minimum=float(np.min(x)),
                    quantiles_05_25_50_75_95=np.quantile(x, [.05, .25, .5, .75, .95]).tolist(),
                    maximum=float(np.max(x)))
    return dict(pairs=len(features),
        training_loss={k:summary(z[k]) for k in ('loss', 'absolute', 'pairing', 'effect')},
        training_positive_assignment_margin_fraction=float(np.mean(z['effect'] > 0)),
        generated_CLIP_difference_norm=summary(ng), real_target_CLIP_difference_norm=summary(nv),
        difference_norm_ratio=summary(ng / nv), difference_alignment_cosine=summary(angle),
        zero_CLIP_difference_count=int((ng <= 1e-12).sum()),
        zero_target_difference_count=int((nv <= 1e-12).sum()),
        absolute_feature_gradient_norm=summary(na), pairing_feature_gradient_norm=summary(nb),
        feature_gradient_cosine=summary(cos),
        feature_gradient_negative_cosine_fraction=float(np.mean(cos < 0)),
        pairing_to_absolute_feature_gradient_norm_ratio=summary(nb / na),
        gradient_space='CLIP output features before their normalization only',
        EEG_or_P_parameter_gradient_conflict_NOT_measured=True,
        different_pairs_and_parameter_states_NOT_a_learning_curve=True,
        no_generalization_image_gain_or_pixel_collapse_claim=True)
