"""The ordinary bounded-direction principle in a prior with no output LN affine.

Preserves each prior sample's mean and centered norm; no HIVE weight transfer.
This is an untrained port of the first method, not a new research contribution.
"""
import torch
from torch import nn


class NativePriorOrdinary(nn.Module):
    def __init__(self, dimension=1024, hidden=256, radius=.25):
        super().__init__()
        assert dimension == 1024 and hidden == 256 and radius == .25
        self.radius = radius
        self.trunk = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU())
        self.direction = nn.Linear(hidden, dimension)
        nn.init.zeros_(self.direction.weight)
        nn.init.zeros_(self.direction.bias)

    def forward(self, condition, *, enabled=True, strength=1.):
        assert condition.ndim == 2 and condition.shape[1] == 1024
        assert 0 <= strength <= 1 and torch.isfinite(condition).all()
        if not enabled:
            return condition, {}
        mean = condition.mean(-1, keepdim=True)
        centered = condition - mean
        norm = centered.norm(dim=-1, keepdim=True)
        valid = norm > 1e-6
        safe_norm = torch.where(valid, norm, torch.ones_like(norm))
        standardized = centered * (condition.shape[-1] ** .5) / safe_norm
        direction = self.direction(self.trunk(standardized))
        direction = direction - direction.mean(-1, keepdim=True)
        direction = direction - (direction * centered).sum(-1, keepdim=True) / safe_norm.square() * centered
        delta = self.radius * strength * norm * direction / (safe_norm.square() + direction.square().sum(-1, keepdim=True)).sqrt()
        scale = norm / (centered + delta).norm(dim=-1, keepdim=True).clamp_min(1e-12)
        correction = torch.where(valid, (scale - 1) * centered + scale * delta, torch.zeros_like(centered))
        result = condition + correction
        assert torch.isfinite(result).all()
        return result, {'correction': correction, 'centered_input_norm': norm.squeeze(-1),
                        'delta_norm': delta.norm(dim=-1), 'valid_centered_input': valid.squeeze(-1)}
