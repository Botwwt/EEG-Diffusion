"""CPU geometry tests, including counterexamples to unjustified guarantees."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import brainhive_ordinary_evidence_fiber_20261028_v1 as V


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('audit/crossdisciplinary_20261028_v1/ordinary_fiber_local_math_checks.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    torch.set_num_threads(2)
    numpy_state, torch_state = np.random.get_state(), torch.get_rng_state()
    rng = np.random.RandomState(1729)
    checks, diagnostics = {}, {}
    def check(name, value):
        if not bool(value):
            raise AssertionError(name)
        checks[name] = True
    def native_ordinary_endpoint(u, d):
        d = d-d.mean(1, keepdims=True)
        d -= np.sum(d*u, axis=1, keepdims=True)/np.sum(u*u, axis=1, keepdims=True)*u
        R = np.linalg.norm(u, axis=1, keepdims=True)
        delta = .25*R*d/np.sqrt(R*R+np.sum(d*d, axis=1, keepdims=True))
        out = u+delta
        return (out*R/np.linalg.norm(out, axis=1, keepdims=True)).astype(u.dtype)

    u = rng.standard_normal((31, 64))
    u -= u.mean(1, keepdims=True)
    u = u.astype(np.float32)
    uO = native_ordinary_endpoint(u, rng.standard_normal(u.shape).astype(np.float32))
    H = rng.standard_normal((8, 64)).astype(np.float32)
    H = np.concatenate((H, H[:1]*7., np.zeros((1, 64), np.float32)))
    Q = V.build_protected_basis_numpy(H)
    check('basis_protects_declared_rowspace_and_constant_mean_direction',
          np.linalg.norm(H.astype(np.float64)-(H@Q.T)@Q)/np.linalg.norm(H) < 1e-12
          and np.linalg.norm(Q@Q.T-np.eye(len(Q))) < 1e-12
          and np.linalg.norm(np.ones(64)-(np.ones(64)@Q.T)@Q) < 1e-12)
    result, info = V.project_evidence_fiber_numpy(u, uO, H)
    check('rowspace_measurements_and_mean_preserved',
          np.max(info['protected_error']) < 5e-7 and np.max(np.abs(result.astype(np.float64).mean(1)-u.astype(np.float64).mean(1))) < 2e-8)
    check('original_norm_and_actual_global_angular_budget_preserved',
          np.max(info['norm_error']) < 5e-7 and np.max(info['actual_global_angle']) <= V.ANGULAR_LIMIT+1e-7)
    check('dtype_shape_finite_and_original_arrays_unmodified',
          result.shape == u.shape and result.dtype == np.float32 and np.isfinite(result).all())
    check('off_returns_exact_original_object', V.project_evidence_fiber_numpy(u, None, None, enabled=False)[0] is u)
    check('zero_ordinary_update_returns_exact_original_object', V.project_evidence_fiber_numpy(u, u, H)[0] is u)
    check('zero_vector_degenerate_returns_exact_original_object',
          (lambda z: V.project_evidence_fiber_numpy(z, z, H)[0] is z)(np.zeros_like(u)))
    check('fully_protected_space_returns_exact_original_object', V.project_evidence_fiber_numpy(u, uO, np.eye(64))[0] is u)
    fixed_again, fixed_info = V.project_evidence_fiber_numpy(u, result, H)
    check('already_feasible_endpoint_is_idempotent', np.array_equal(fixed_again, result) and fixed_info['already_feasible'].all())

    # A four-dimensional fixture has a two-dimensional null plane once mean
    # and one evidence axis are protected. A dense circle supplies an
    # independent global nearest-endpoint comparison.
    paxis = np.array([1., 1., -1., -1.])/2
    naxis = np.array([1., -1., 0., 0.])/math.sqrt(2)
    taxis = np.array([0., 0., 1., -1.])/math.sqrt(2)
    fixture = (math.sqrt(.96)*paxis+.2*naxis)[None]
    target = (math.sqrt(1.-.04**2-.02**2)*paxis-.04*naxis+.02*taxis)[None]
    check('constrained_fixture_ordinary_endpoint_inside_original_angle', V._np_angles(fixture, target)[0] < V.ANGULAR_LIMIT)
    finite, details = V.project_evidence_fiber_numpy(fixture, target, paxis[None])
    check('global_cap_requires_larger_null_arc_than_atan_rho',
          details['target_null_arc_angle'][0] > details['allowed_null_arc_angle'][0]
          and details['allowed_null_arc_angle'][0] > V.ANGULAR_LIMIT
          and abs(details['actual_global_angle'][0]-V.ANGULAR_LIMIT) < 1e-12)
    angles = np.linspace(-math.pi, math.pi, 200001)
    circle = math.sqrt(.96)*paxis+.2*(np.cos(angles)[:, None]*naxis+np.sin(angles)[:, None]*taxis)
    allowed = V._np_angles(np.broadcast_to(fixture, circle.shape), circle) <= V.ANGULAR_LIMIT+1e-12
    optimum_grid = np.min(np.sum((circle[allowed]-target)**2, axis=1))
    exact_objective = float(np.sum((finite-target)**2))
    diagnostics['closed_form_objective_minus_dense_feasible_circle_min'] = exact_objective-float(optimum_grid)
    check('generic_closed_form_is_globally_nearest_on_nonconvex_spherical_cap', exact_objective <= optimum_grid+1e-12 and optimum_grid-exact_objective < 1e-5)
    antipodal = (math.sqrt(1.-.001**2)*paxis-.001*naxis)[None]
    fallback, anti_info = V.project_evidence_fiber_numpy(fixture, antipodal, paxis[None])
    cap = anti_info['allowed_null_arc_angle'][0]
    boundary = math.sqrt(.96)*paxis+.2*(math.cos(cap)*naxis+math.sin(cap)*taxis)
    check('antipodal_degenerate_fallback_exact_but_not_claimed_nearest',
          fallback is fixture and anti_info['antipodal_clipped_fallback'][0]
          and not anti_info['nearest_solution_guaranteed'][0]
          and np.sum((boundary-antipodal[0])**2) < np.sum((fixture-antipodal)**2))
    zero_null_target = paxis[None]
    equal_distance, equal_info = V.project_evidence_fiber_numpy(fixture, zero_null_target, paxis[None])
    check('zero_target_null_component_degenerate_keeps_original', equal_distance is fixture and equal_info['nearest_solution_guaranteed'][0])
    check('sphere_nonconvexity_counterexample_blocks_convex_projection_theorem',
          np.linalg.norm(((naxis+taxis)/2)) < 1. and abs(np.linalg.norm(naxis)-1.) < 1e-15 and abs(np.linalg.norm(taxis)-1.) < 1e-15)

    gamma = rng.uniform(.7, 1.3, (64,)).astype(np.float32)
    beta = rng.uniform(-.2, .2, (64,)).astype(np.float32)
    c, cO = (u*gamma+beta).astype(np.float32), (uO*gamma+beta).astype(np.float32)
    restored, affine_info = V.restore_condition_numpy(c, cO, gamma, beta, H)
    restored_u = ((restored-beta)/gamma).astype(np.float64)
    base_u = ((c-beta)/gamma).astype(np.float64)
    check('original_affine_coordinates_restored_with_preserved_measurements',
          np.max(np.linalg.norm((restored_u-base_u)@Q.T, axis=1)) < 1e-6)
    check('affine_disable_and_zero_update_exact',
          V.restore_condition_numpy(c, None, None, None, None, enabled=False)[0] is c
          and V.restore_condition_numpy(c, c, gamma, beta, H)[0] is c)

    tu, to, th = torch.from_numpy(u.copy()), torch.from_numpy(uO.copy()), torch.from_numpy(H.copy())
    tout, ti = V.project_evidence_fiber_torch(tu, to, th)
    parity = float(np.max(np.abs(tout.numpy()-result)))
    diagnostics['NumPy_Torch_float32_max_abs'] = parity
    check('Torch_and_NumPy_operators_agree_and_remain_CPU', parity < 1e-6 and tout.device.type == 'cpu')
    check('Torch_off_zero_and_full_protection_exact',
          V.project_evidence_fiber_torch(tu, None, None, enabled=False)[0] is tu
          and V.project_evidence_fiber_torch(tu, tu, th)[0] is tu
          and V.project_evidence_fiber_torch(tu, to, torch.eye(64))[0] is tu)
    check('Torch_feasible_idempotence', torch.equal(V.project_evidence_fiber_torch(tu, tout, th)[0], tout))
    tcond, tci = V.restore_condition_torch(torch.from_numpy(c), torch.from_numpy(cO), torch.from_numpy(gamma), torch.from_numpy(beta), th)
    check('Torch_affine_wrapper_matches_NumPy', np.max(np.abs(tcond.numpy()-restored)) < 1e-6)
    differentiable_target = to.double().requires_grad_(True)
    differentiable, _ = V.project_evidence_fiber_torch(tu.double(), differentiable_target, th.double())
    (differentiable.square()*torch.linspace(.5, 1.5, 64)).sum().backward()
    check('generic_Torch_operator_has_finite_endpoint_gradient', torch.isfinite(differentiable_target.grad).all())

    rejected = 0
    for call in (lambda: V.project_evidence_fiber_numpy(u, uO*1.1, H),
                 lambda: V.project_evidence_fiber_numpy(u, -u, H),
                 lambda: V.project_evidence_fiber_numpy(u, uO, np.full((1, 64), np.nan)),
                 lambda: V.project_evidence_fiber_numpy(u, uO, H, rho=.5),
                 lambda: V.restore_condition_numpy(c, cO, np.zeros(64), beta, H)):
        try:
            call()
        except (ValueError, TypeError):
            rejected += 1
    check('rejects_invalid_endpoint_norm_angle_affine_basis_and_radius', rejected == 5)
    new_np_state = np.random.get_state()
    same_np_state = numpy_state[0] == new_np_state[0] and np.array_equal(numpy_state[1], new_np_state[1]) and numpy_state[2:] == new_np_state[2:]
    check('no_NumPy_or_Torch_global_RNG_consumed_and_no_CUDA_initialised',
          same_np_state and torch.equal(torch_state, torch.get_rng_state()) and not torch.cuda.is_initialized())
    sources = {}
    for name in ('brainhive_ordinary_evidence_fiber_20261028_v1.py', Path(__file__).name):
        sources[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    value = dict(status='ORDINARY_EVIDENCE_FIBER_PURE_MATH_CPU_CHECKS_PASSED', checks=checks,
                 diagnostics=diagnostics, source_sha256=sources, seconds=time.perf_counter()-started,
                 rho=.25, actual_global_angular_limit_radians=V.ANGULAR_LIMIT,
                 H_learned=False, actual_EEG_read=False, new_images=0, new_training=0,
                 old_ordinary_source_or_checkpoint_modified=False, GPU_used=False,
                 proof_limits=['Spherical optimisation is nonconvex; convex projection theorem is inapplicable.',
                               'The forced clipped-antipodal fallback is feasible but need not be nearest.',
                               'H is a declared measured subspace, not established physiology.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(value, allow_nan=False))


if __name__ == '__main__':
    main()
