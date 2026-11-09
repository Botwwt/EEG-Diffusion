"""Frozen-head norm-only counterfactual; no new parameters or training.

Preserve the positive ray of the trained Euclidean-F condition while restoring
only the original affine-coordinate norm. This does not restore zero mean or
the ordinary tangent/angle constraint and is not claimed as a new method.
"""
import torch
from brainhive_ordinary_geometry_control_20261030_v1 import EuclideanConditionControl


def positive_ray_norm_projection(condition, raw, weight, bias):
    assert condition.shape == raw.shape == weight.shape == bias.shape
    assert condition.ndim == 2 and condition.shape[-1] == 1024
    assert torch.isfinite(raw).all() and torch.all(weight.abs() > 1e-6)
    c, z, w, b = (x.double() for x in (condition, raw, weight, bias))
    native = (c - b) / w
    target_r2 = native.square().sum(dim=-1, keepdim=True)
    z_over_w, b_over_w = z / w, b / w
    a = z_over_w.square().sum(dim=-1, keepdim=True)
    cross = (z_over_w * b_over_w).sum(dim=-1, keepdim=True)
    d = b_over_w.square().sum(dim=-1, keepdim=True) - target_r2
    assert torch.all(a > 0) and torch.all(d < 0)
    discriminant = (cross.square() - a * d).sqrt()
    # Stable positive quadratic root; the second root is strictly negative.
    factor = torch.where(cross >= 0, (cross + discriminant) / a,
                         -d / (discriminant - cross))
    assert torch.isfinite(factor).all() and torch.all(factor > 0)
    unchanged = (raw == condition).all(dim=-1, keepdim=True)
    factor = torch.where(unchanged, torch.ones_like(factor), factor)
    result = torch.where(unchanged, condition, (z * factor).to(condition.dtype))
    assert torch.isfinite(result).all()
    return result, factor


class NormOnlyConditionControl(EuclideanConditionControl):
    def forward(self, condition, weight, bias, *, enabled=True, gate_mode='native',
                calibration=(1., 0.), risk_permutation=None):
        if not enabled:
            return condition, {}
        raw, info = super().forward(condition, weight, bias, enabled=True,
                                    gate_mode=gate_mode, calibration=calibration,
                                    risk_permutation=risk_permutation)
        result, factor = positive_ray_norm_projection(condition, raw, weight, bias)
        info.update(raw_Euclidean_condition=raw, positive_norm_factor=factor,
                    correction=result-condition)
        return result, info
