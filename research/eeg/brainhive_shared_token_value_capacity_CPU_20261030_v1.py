"""Shared-token, fixed-Q/K, privileged teacher capacity diagnostic.

One token change per historical stimulus/arm, shared across layers, heads and
times. A declared mapped-value norm budget is enforced. Matrix-free projected
iterations give feasible witnesses and convex lower bounds, not automatically
converged optima. This is not a learned EEG decoder or a whole-UNet derivative.
"""
import numpy as np


def bases():
    mean = np.ones((4, 1), dtype=np.float64) / 2
    center = np.zeros((4, 3), dtype=np.float64)
    for j in range(3):
        scale = np.sqrt((j + 1) * (j + 2))
        center[:j + 1, j] = 1 / scale
        center[j + 1, j] = -(j + 1) / scale
    return dict(mean_only=mean, centered_only=center,
                free_value=np.concatenate([mean, center], axis=1))


class SharedProblem:
    def __init__(self, blocks, value_maps, value_norms, basis):
        self.blocks, self.maps, self.norms, self.basis = blocks, value_maps, value_norms, basis
        self.rows = basis.shape[1]
        self.features = next(iter(value_maps.values())).shape[1]
        self.G, self.B = {}, {}
        self.target_squared = 0.
        self.rhs = np.zeros((self.rows, self.features), dtype=np.float64)
        for block in blocks:
            layer, a, target = block['layer'], block['attention'], block['target']
            h, positions, d = target.shape
            D = a @ basis
            weight = 1. / (len(blocks) * target.size)
            self.G.setdefault(layer, np.zeros((h, self.rows, self.rows)))
            self.B.setdefault(layer, np.zeros((h, self.rows, d)))
            self.G[layer] += weight * (D.swapaxes(-1, -2) @ D)
            self.B[layer] += weight * (D.swapaxes(-1, -2) @ target)
            self.target_squared += weight * float(np.square(target).sum())
        for layer, values in self.B.items():
            packed = values.transpose(1, 0, 2).reshape(self.rows, -1)
            self.rhs += packed @ self.maps[layer]

    def normal(self, x):
        result = np.zeros_like(x)
        for layer, gram in self.G.items():
            W = self.maps[layer]
            h = gram.shape[0]
            d = W.shape[0] // h
            projected = (x @ W.T).reshape(self.rows, h, d).transpose(1, 0, 2)
            weighted = np.matmul(gram, projected)
            result += weighted.transpose(1, 0, 2).reshape(self.rows, -1) @ W
        return result

    def objective(self, x, hx=None):
        hx = self.normal(x) if hx is None else hx
        return .5 * (float((x * hx).sum()) - 2 * float((x * self.rhs).sum()) + self.target_squared)

    def layer_ratios(self, x):
        result = {}
        for layer, W in self.maps.items():
            delta = float(np.linalg.norm(x @ W.T))
            original = self.norms[layer]
            result[layer] = delta / original if original else (0. if delta == 0 else float('inf'))
        return result

    def strict_witness(self, x, hx):
        ratios = self.layer_ratios(x)
        scale = min(1., 1. / max(1., max(ratios.values())))
        witness = scale * x
        objective = self.objective(witness, scale * hx)
        return witness, scale * hx, objective, scale


def projected_solve(problem, radius, lipschitz, *, iterations=64, initial=None):
    assert radius >= 0 and lipschitz > 0 and iterations == 64
    x = np.zeros_like(problem.rhs) if initial is None else initial.copy()
    assert np.linalg.norm(x) <= radius * (1 + 1e-10)
    hx = problem.normal(x)
    previous, h_previous = x.copy(), hx.copy()
    momentum = 1.
    best_x, best_h, best_f = x.copy(), hx.copy(), problem.objective(x, hx)
    strict_x, strict_h, strict_f, strict_scale = problem.strict_witness(x, hx)
    lower_bound = 0.
    history = []
    for iteration in range(iterations + 1):
        f = problem.objective(x, hx)
        gradient = hx - problem.rhs
        gap = float((gradient * x).sum() + radius * np.linalg.norm(gradient))
        assert gap >= -1e-10 * max(1., problem.target_squared)
        lower_bound = max(lower_bound, f - max(0., gap))
        if f < best_f:
            best_x, best_h, best_f = x.copy(), hx.copy(), f
        witness, hw, fw, scale = problem.strict_witness(x, hx)
        if fw < strict_f:
            strict_x, strict_h, strict_f, strict_scale = witness, hw, fw, scale
        history.append(dict(iteration=iteration, objective=f, convex_gap=gap,
                            aggregate_relative_norm=float(np.linalg.norm(x) / radius) if radius else 0.,
                            strict_witness_objective=fw))
        if iteration == iterations:
            break
        next_momentum = .5 * (1 + np.sqrt(1 + 4 * momentum**2))
        beta = (momentum - 1) / next_momentum
        y = x + beta * (x - previous)
        hy = hx + beta * (hx - h_previous)
        candidate = y - (hy - problem.rhs) / lipschitz
        norm = float(np.linalg.norm(candidate))
        if norm > radius:
            candidate *= radius / norm
        previous, h_previous = x, hx
        x, hx = candidate, problem.normal(candidate)
        momentum = next_momentum
    assert best_f >= -1e-10 and strict_f >= -1e-10
    assert np.linalg.norm(strict_x) <= radius * (1 + 1e-10)
    assert max(problem.layer_ratios(strict_x).values()) <= 1 + 1e-10
    certificate = dict(base_MSE=problem.target_squared, aggregate_ball_best_MSE=2 * best_f,
                       strict_per_layer_feasible_MSE=2 * strict_f, convex_MSE_lower_bound=2 * lower_bound,
                       aggregate_ball_MSE_gap_upper=2 * max(0., best_f - lower_bound),
                       strict_MSE_gap_to_relaxed_lower_bound=2 * max(0., strict_f - lower_bound),
                       strict_relative_norm=float(np.linalg.norm(strict_x) / radius) if radius else 0.,
                       strict_max_layer_value_relative_norm=max(problem.layer_ratios(strict_x).values()),
                       strict_scale_from_aggregate_iterate=float(strict_scale),
                       fixed_projected_iterations=iterations,
                       strict_witness_converged_within_one_percent_base=(2 * max(0., strict_f - lower_bound) <= .01 * problem.target_squared),
                       history=history)
    return certificate, strict_x, best_x


def mathematical_checks():
    rng = np.random.default_rng(1729)
    W = rng.normal(size=(8, 7))
    a = rng.uniform(size=(2, 5, 4)); a /= a.sum(-1, keepdims=True)
    target = rng.normal(size=(2, 5, 4))
    blocks = [dict(layer=0, attention=a, target=target), dict(layer=0, attention=a, target=target * .5)]
    problem = SharedProblem(blocks, {0: W}, {0: 10.}, bases()['centered_only'])
    x = rng.normal(size=problem.rhs.shape)
    explicit_rhs, explicit_normal = np.zeros_like(x), np.zeros_like(x)
    adjoint_error = 0.
    for block in blocks:
        D = block['attention'] @ problem.basis
        delta = (x @ W.T).reshape(3, 2, 4).transpose(1, 0, 2)
        response = D @ delta
        y = rng.normal(size=response.shape)
        adjoint = (D.swapaxes(-1, -2) @ y).transpose(1, 0, 2).reshape(3, 8) @ W
        adjoint_error = max(adjoint_error, abs(float((response * y).sum()) - float((x * adjoint).sum())))
        scale = 1 / (len(blocks) * block['target'].size)
        explicit_rhs += scale * (D.swapaxes(-1, -2) @ block['target']).transpose(1, 0, 2).reshape(3, 8) @ W
        explicit_normal += scale * (D.swapaxes(-1, -2) @ response).transpose(1, 0, 2).reshape(3, 8) @ W
    assert adjoint_error < 1e-12
    assert np.max(np.abs(explicit_rhs - problem.rhs)) < 1e-12
    assert np.max(np.abs(explicit_normal - problem.normal(x))) < 1e-12
    # A deliberately conservative bound suffices for this small fixture.
    L = float(np.square(W).sum())
    cert, solution, _ = projected_solve(problem, 1., L)
    direct = 0.
    for block in blocks:
        D = block['attention'] @ problem.basis
        response = D @ (solution @ W.T).reshape(3, 2, 4).transpose(1, 0, 2)
        direct += np.square(response - block['target']).mean() / len(blocks)
    assert abs(direct - cert['strict_per_layer_feasible_MSE']) < 1e-12
    # Build the full small normal operator to independently check the lower bound.
    n = problem.rhs.size
    dense = np.column_stack([problem.normal(np.eye(n)[i].reshape(problem.rhs.shape)).ravel() for i in range(n)])
    assert np.linalg.eigvalsh(dense).min() > -1e-12
    assert cert['convex_MSE_lower_bound'] <= cert['strict_per_layer_feasible_MSE'] + 1e-12
    return dict(status='PASS_SHARED_TOKEN_OPERATOR_ADJOINT_AND_FEASIBLE_SOLVER_CHECKS', checks=5,
                adjoint_max_error=adjoint_error, normal_operator_max_error=float(np.abs(explicit_normal - problem.normal(x)).max()),
                direct_response_MSE_error=abs(direct - cert['strict_per_layer_feasible_MSE']),
                finite_primal_witness_not_automatically_optimal=True, actual_EEG_or_GPU_used=False)
