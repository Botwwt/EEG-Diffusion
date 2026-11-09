"""Euclidean nearest ordinary endpoint on a protected spherical EEG fiber.

This is classical constrained geometry, not an originality or physiology claim.
H describes measured candidate evidence axes; its validity must be established
on separate EEG observations. Preserving H does not preserve all information.

Let Q have orthonormal rows spanning rows(H) AND the constant mean direction.
For p=Q.T Q u, n0=u-p, r=||n0||, nO=(I-Q.T Q)uO, the fiber has the exact form
    v=p+r*q, q in ker(Q), ||q||=1.
Consequently minimising ||v-uO|| is equivalent to maximising q.nO. The
unconstrained optimum is q=nO/||nO||; this is a spherical optimisation proof,
NOT an application of the convex projection theorem.

For alpha=atan(rho), R=||u||, a null-sphere arc beta obeys
    cos(angle(u,v)) = 1 - (r*r/R/R)*(1-cos(beta)).
Thus beta_max=acos(clip(1-R*R*(1-cos(alpha))/(r*r),-1,1)).
The nearest feasible endpoint follows the shortest q0->qO arc, truncated at
beta_max. This is globally nearest if nO is nonzero and the necessary arc
direction is defined. If qO=-q0 and beta_max<pi, closest boundary points are
nonunique (provided the null space has dimension >=2). The required degenerate
fallback returns original u, which is FEASIBLE BUT NOT GENERALLY NEAREST.
When nO=0 every null-sphere point has equal objective, so returning u is optimal.

The original ordinary update has Delta perpendicular to u, ||Delta||<=rho||u||,
and renormalises u+Delta. Its actual angular bound is atan(rho), not a chord
bound rho or a null-sphere bound atan(rho). This module fixes rho=.25.

No old component or weight is modified. Torch imports are lazy; the Torch
operator preserves gradients through u/uO and holds the supplied H fixed.
"""
import math
import numpy as np

RHO = .25
ANGULAR_LIMIT = math.atan(RHO)
BASIS_RTOL = 1e-10
DEGENERATE_RTOL = 1e-12
ENDPOINT_RTOL = 2e-5
FEASIBLE_RTOL = 2e-8


def _np_matrix(value, name):
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(name + ' must be nonempty [B,D]')
    if value.dtype not in (np.dtype('float32'), np.dtype('float64')) or not np.isfinite(value).all():
        raise ValueError(name + ' must be finite float32/float64')


def build_protected_basis_numpy(H, dimension=None):
    """Orthonormal float64 row basis for span(rows(H), ones/sqrt(D))."""
    if H is None:
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError('dimension required with H=None')
        rows = np.empty((0, dimension), np.float64)
    else:
        H = np.asarray(H)
        if H.ndim != 2 or H.shape[1] < 1 or H.dtype.kind != 'f' or not np.isfinite(H).all():
            raise ValueError('H must be finite floating [K,D], allowing K=0')
        if dimension is not None and H.shape[1] != dimension:
            raise ValueError('H dimension mismatch')
        dimension = H.shape[1]
        rows = H.astype(np.float64, copy=True)
        if len(rows):
            scales = np.max(np.abs(rows), axis=1)
            rows = rows[scales > 0.] / scales[scales > 0., None]
            rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    rows = np.concatenate((rows, np.ones((1, dimension), np.float64) / math.sqrt(dimension)))
    _, singular, right = np.linalg.svd(rows, full_matrices=False)
    return right[singular > BASIS_RTOL * singular[0]]


def _np_angles(a, b):
    an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    cosine = np.ones_like(an)
    np.divide(np.sum(a*b, axis=1), an*bn, out=cosine, where=an*bn > 0.)
    return np.arccos(np.clip(cosine, -1., 1.))


def project_evidence_fiber_numpy(u, u_ordinary, H, *, enabled=True, rho=RHO):
    """Return (same-dtype [B,D] endpoint, diagnostics), without consuming RNG."""
    _np_matrix(u, 'u')
    if rho != RHO:
        raise ValueError('Only the original fixed rho=.25 is allowed')
    if not enabled:
        return u, dict(enabled=False, kept_original=np.ones(len(u), bool))
    _np_matrix(u_ordinary, 'u_ordinary')
    if u_ordinary.shape != u.shape or u_ordinary.dtype != u.dtype:
        raise ValueError('u/u_ordinary shape and dtype must match')
    x, target = u.astype(np.float64), u_ordinary.astype(np.float64)
    radius, target_radius = np.linalg.norm(x, axis=1), np.linalg.norm(target, axis=1)
    scale = np.maximum(radius, 1.)
    if np.any(np.abs(target_radius-radius) > ENDPOINT_RTOL*scale):
        raise ValueError('ordinary endpoint must preserve the original norm')
    if np.any(_np_angles(x, target) > ANGULAR_LIMIT+ENDPOINT_RTOL):
        raise ValueError('ordinary endpoint exceeds atan(.25) angular bound')
    Q = build_protected_basis_numpy(H, u.shape[1])
    p = (x @ Q.T) @ Q
    n0, nO = x-p, target-(target @ Q.T) @ Q
    null_radius, target_null_radius = np.linalg.norm(n0, axis=1), np.linalg.norm(nO, axis=1)
    zero = (radius <= DEGENERATE_RTOL) | (null_radius <= DEGENERATE_RTOL*scale) | (target_null_radius <= DEGENERATE_RTOL*scale)
    q0 = n0/np.maximum(null_radius[:, None], np.finfo(np.float64).tiny)
    qO = nO/np.maximum(target_null_radius[:, None], np.finfo(np.float64).tiny)
    dot = np.clip(np.sum(q0*qO, axis=1), -1., 1.)
    theta = np.arccos(dot)
    cos_cap = 1.-radius**2*(1.-math.cos(ANGULAR_LIMIT))/np.maximum(null_radius**2, np.finfo(np.float64).tiny)
    cap = np.arccos(np.clip(cos_cap, -1., 1.))
    beta = np.minimum(theta, cap)
    transverse = qO-dot[:, None]*q0
    transverse_norm = np.linalg.norm(transverse, axis=1)
    clipped = theta > cap+DEGENERATE_RTOL
    ambiguous = clipped & (transverse_norm <= DEGENERATE_RTOL) & ~zero
    direction = transverse/np.maximum(transverse_norm[:, None], np.finfo(np.float64).tiny)
    q = np.cos(beta)[:, None]*q0+np.sin(beta)[:, None]*direction
    q = np.where(clipped[:, None], q, qO)
    value64 = p+null_radius[:, None]*q
    already = (np.linalg.norm((target-x)@Q.T, axis=1) <= FEASIBLE_RTOL*scale)
    already &= np.abs(target_radius-radius) <= FEASIBLE_RTOL*scale
    already &= _np_angles(x, target) <= ANGULAR_LIMIT+FEASIBLE_RTOL
    unchanged = np.all(u_ordinary == u, axis=1)
    kept = zero | ambiguous | unchanged
    value = np.where(already[:, None], u_ordinary, value64.astype(u.dtype))
    value = np.where(kept[:, None], u, value)
    if np.all(kept):
        value = u
    result64 = value.astype(np.float64)
    info = dict(enabled=True, protected_rank=len(Q), kept_original=kept,
                already_feasible=already, degenerate=zero | ambiguous,
                antipodal_clipped_fallback=ambiguous,
                nearest_solution_guaranteed=~ambiguous,
                target_null_arc_angle=theta, allowed_null_arc_angle=cap,
                actual_global_angle=_np_angles(x, result64), angular_limit=ANGULAR_LIMIT,
                protected_error=np.linalg.norm((result64-x)@Q.T, axis=1),
                norm_error=np.abs(np.linalg.norm(result64, axis=1)-radius))
    return value, info


def build_protected_basis_torch(H, dimension=None, *, device=None):
    """Torch equivalent; H is fixed, detached, and may have zero rows."""
    import torch
    if H is None:
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError('dimension required with H=None')
        rows = torch.empty((0, dimension), dtype=torch.float64, device=device)
    else:
        if not torch.is_tensor(H) or H.ndim != 2 or H.shape[1] < 1 or not H.is_floating_point() or not torch.isfinite(H).all():
            raise ValueError('H must be finite floating [K,D], allowing K=0')
        if dimension is not None and H.shape[1] != dimension:
            raise ValueError('H dimension mismatch')
        dimension = H.shape[1]
        rows = H.detach().to(device=device, dtype=torch.float64)
        if len(rows):
            scales = rows.abs().amax(dim=1)
            rows = rows[scales > 0.]/scales[scales > 0., None]
            rows = rows/torch.linalg.vector_norm(rows, dim=1, keepdim=True)
    mean = torch.ones((1, dimension), dtype=torch.float64, device=rows.device)/math.sqrt(dimension)
    rows = torch.cat((rows, mean), dim=0)
    _, singular, right = torch.linalg.svd(rows, full_matrices=False)
    return right[singular > BASIS_RTOL*singular[0]]


def _torch_angles(a, b):
    import torch
    den = torch.linalg.vector_norm(a, dim=1)*torch.linalg.vector_norm(b, dim=1)
    cosine = torch.where(den > 0., (a*b).sum(1)/den.clamp_min(torch.finfo(a.dtype).tiny), torch.ones_like(den))
    return torch.acos(cosine.clamp(-1., 1.))


def project_evidence_fiber_torch(u, u_ordinary, H, *, enabled=True, rho=RHO):
    """Differentiable Torch endpoint operation; no module or learned parameter."""
    import torch
    if not torch.is_tensor(u) or u.ndim != 2 or min(u.shape) < 1 or u.dtype not in (torch.float32, torch.float64) or not torch.isfinite(u).all():
        raise ValueError('u must be finite float32/float64 [B,D]')
    if rho != RHO:
        raise ValueError('Only the original fixed rho=.25 is allowed')
    if not enabled:
        return u, dict(enabled=False, kept_original=torch.ones(len(u), dtype=torch.bool, device=u.device))
    if not torch.is_tensor(u_ordinary) or u_ordinary.shape != u.shape or u_ordinary.dtype != u.dtype or u_ordinary.device != u.device or not torch.isfinite(u_ordinary).all():
        raise ValueError('u/u_ordinary finite shape, dtype and device must match')
    x, target = u.double(), u_ordinary.double()
    radius, target_radius = torch.linalg.vector_norm(x, dim=1), torch.linalg.vector_norm(target, dim=1)
    scale = radius.clamp_min(1.)
    if torch.any((target_radius-radius).abs() > ENDPOINT_RTOL*scale):
        raise ValueError('ordinary endpoint must preserve the original norm')
    if torch.any(_torch_angles(x, target) > ANGULAR_LIMIT+ENDPOINT_RTOL):
        raise ValueError('ordinary endpoint exceeds atan(.25) angular bound')
    Q = build_protected_basis_torch(H, u.shape[1], device=u.device)
    p = (x @ Q.T) @ Q
    n0, nO = x-p, target-(target @ Q.T) @ Q
    null_radius, target_null_radius = torch.linalg.vector_norm(n0, dim=1), torch.linalg.vector_norm(nO, dim=1)
    tiny = torch.finfo(torch.float64).tiny
    zero = (radius <= DEGENERATE_RTOL) | (null_radius <= DEGENERATE_RTOL*scale) | (target_null_radius <= DEGENERATE_RTOL*scale)
    q0 = n0/null_radius[:, None].clamp_min(tiny)
    qO = nO/target_null_radius[:, None].clamp_min(tiny)
    dot = (q0*qO).sum(1).clamp(-1., 1.)
    theta = torch.acos(dot)
    cos_cap = 1.-radius.square()*(1.-math.cos(ANGULAR_LIMIT))/null_radius.square().clamp_min(tiny)
    cap = torch.acos(cos_cap.clamp(-1., 1.))
    beta = torch.minimum(theta, cap)
    transverse = qO-dot[:, None]*q0
    transverse_norm = torch.linalg.vector_norm(transverse, dim=1)
    clipped = theta > cap+DEGENERATE_RTOL
    ambiguous = clipped & (transverse_norm <= DEGENERATE_RTOL) & ~zero
    direction = transverse/transverse_norm[:, None].clamp_min(tiny)
    q = torch.cos(beta)[:, None]*q0+torch.sin(beta)[:, None]*direction
    q = torch.where(clipped[:, None], q, qO)
    value64 = p+null_radius[:, None]*q
    already = torch.linalg.vector_norm((target-x)@Q.T, dim=1) <= FEASIBLE_RTOL*scale
    already = already & ((target_radius-radius).abs() <= FEASIBLE_RTOL*scale)
    already = already & (_torch_angles(x, target) <= ANGULAR_LIMIT+FEASIBLE_RTOL)
    unchanged = (u_ordinary == u).all(dim=1)
    kept = zero | ambiguous | unchanged
    value = torch.where(already[:, None], u_ordinary, value64.to(u.dtype))
    value = torch.where(kept[:, None], u, value)
    if torch.all(kept):
        value = u
    result64 = value.double()
    info = dict(enabled=True, protected_rank=len(Q), kept_original=kept,
                already_feasible=already, degenerate=zero | ambiguous,
                antipodal_clipped_fallback=ambiguous,
                nearest_solution_guaranteed=~ambiguous,
                target_null_arc_angle=theta, allowed_null_arc_angle=cap,
                actual_global_angle=_torch_angles(x, result64), angular_limit=ANGULAR_LIMIT,
                protected_error=torch.linalg.vector_norm((result64-x)@Q.T, dim=1),
                norm_error=(torch.linalg.vector_norm(result64, dim=1)-radius).abs())
    return value, info


def restore_condition_numpy(condition, ordinary_condition, weight, bias, H, *, enabled=True, rho=RHO):
    """Undo/reapply the real affine; preserve original bytes on all off rows."""
    _np_matrix(condition, 'condition')
    if not enabled:
        return condition, dict(enabled=False, kept_original=np.ones(len(condition), bool))
    _np_matrix(ordinary_condition, 'ordinary_condition')
    if ordinary_condition.shape != condition.shape or ordinary_condition.dtype != condition.dtype:
        raise ValueError('condition endpoints must match')
    weight = np.broadcast_to(np.asarray(weight, dtype=condition.dtype), condition.shape)
    bias = np.broadcast_to(np.asarray(bias, dtype=condition.dtype), condition.shape)
    if not np.isfinite(weight).all() or not np.isfinite(bias).all() or np.any(np.abs(weight) <= 1e-6):
        raise ValueError('invalid original output affine')
    u, uO = (condition-bias)/weight, (ordinary_condition-bias)/weight
    star, info = project_evidence_fiber_numpy(u, uO, H, rho=rho)
    value = condition+weight*(star-u)
    value = np.where(info['already_feasible'][:, None], ordinary_condition, value)
    value = np.where(info['kept_original'][:, None], condition, value)
    if np.all(info['kept_original']):
        value = condition
    return value, info


def restore_condition_torch(condition, ordinary_condition, weight, bias, H, *, enabled=True, rho=RHO):
    """Torch affine wrapper; the old ordinary endpoint is supplied externally."""
    import torch
    if not enabled:
        return condition, dict(enabled=False, kept_original=torch.ones(len(condition), dtype=torch.bool, device=condition.device))
    weight = torch.broadcast_to(weight, condition.shape)
    bias = torch.broadcast_to(bias, condition.shape)
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all() or torch.any(weight.abs() <= 1e-6):
        raise ValueError('invalid original output affine')
    u, uO = (condition-bias)/weight, (ordinary_condition-bias)/weight
    star, info = project_evidence_fiber_torch(u, uO, H, rho=rho)
    value = condition+weight*(star-u)
    value = torch.where(info['already_feasible'][:, None], ordinary_condition, value)
    value = torch.where(info['kept_original'][:, None], condition, value)
    if torch.all(info['kept_original']):
        value = condition
    return value, info
