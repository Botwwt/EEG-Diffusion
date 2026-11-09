"""Euclidean control of the whole ordinary correction geometry.

This is a mechanism control, not a new proposed method. It shares the original
ordinary MLP initialization/state keys, actual host affine coordinates and
residual scale. Only the centering, tangent projection, bounded denominator and
norm restoration are removed together. It requires its own complete training;
using original ordinary weights with this formula is not a matched experiment.
"""
import torch
from brainhive_bounded_component_v1 import BoundedConditionAdapter


class EuclideanConditionControl(BoundedConditionAdapter):
    def __init__(self):
        super().__init__(dimension=1024, hidden=256, predict_risk=False, radius=.25)

    def forward(self, condition, weight, bias, *, enabled=True, gate_mode='native',
                calibration=(1., 0.), risk_permutation=None):
        assert condition.ndim == 2 and condition.shape[-1] == 1024
        assert weight.shape == bias.shape == condition.shape
        assert torch.all(weight.abs() > 1e-6)
        if not enabled:
            return condition, {}
        assert gate_mode in ('native', 'fixed'), 'The control has no risk head or risk permutation'
        assert risk_permutation is None
        # radius is the original scalar interface used by Dfit strength search.
        # Here it is an unconstrained residual scale, not a guaranteed radius.
        assert 0. <= self.radius <= .25
        u = (condition - bias) / weight
        raw = self.direction(self.trunk(u))
        delta = self.radius * raw
        correction = weight * delta
        result = condition + correction
        assert torch.isfinite(result).all()
        return result, dict(risk=None, gate=condition.new_ones(len(condition)),
                            correction=correction,
                            standardized_delta_norm=delta.norm(dim=-1),
                            standardized_input_norm=u.norm(dim=-1))
