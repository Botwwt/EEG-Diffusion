"""A constrained visual coordinate chart, not a new probabilistic EEG likelihood."""
import math
import torch
from torch import nn


class RotationCoupling(nn.Module):
    """Triangular, unit-determinant, norm-preserving coupling with exact inverse.

    The gate reads unchanged coordinates only. This classical construction does
    not imply preservation of the empirical visual distribution or distances.
    """
    def __init__(self, dimension=1024, hidden=64, shift=0):
        super().__init__()
        assert dimension % 4 == 0
        self.dimension = dimension
        self.shift = shift
        self.first = nn.Linear(dimension // 2, hidden)
        self.last = nn.Linear(hidden, dimension // 4)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, value, inverse=False):
        z = torch.roll(value, self.shift, dims=-1)
        a, b = z.split(self.dimension // 2, dim=-1)
        normalized_a = a / torch.sqrt(1 + a.square().sum(dim=-1, keepdim=True))
        angle = (math.pi / 8) * torch.tanh(self.last(torch.tanh(self.first(normalized_a))))
        if inverse:
            angle = -angle
        pair = b.reshape(*b.shape[:-1], -1, 2)
        x, y = pair[..., 0], pair[..., 1]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotated = torch.stack([cosine * x - sine * y, sine * x + cosine * y], dim=-1).flatten(-2)
        return torch.roll(torch.cat([a, rotated], dim=-1), -self.shift, dims=-1)


class ConjugateChart(nn.Module):
    def __init__(self, dimension=1024, hidden=64, blocks=4):
        super().__init__()
        self.layers = nn.ModuleList([RotationCoupling(dimension, hidden, shift=i * (dimension // 8)) for i in range(blocks)])

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return value

    def inverse(self, value):
        for layer in reversed(self.layers):
            value = layer(value, inverse=True)
        return value


class OrdinaryCapacityControl(nn.Module):
    """A generic residual output head with nearly identical added parameter count."""
    def __init__(self, dimension=1024, hidden=96):
        super().__init__()
        self.first = nn.Linear(dimension, hidden)
        self.last = nn.Linear(hidden, dimension)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, value):
        return value + self.last(torch.tanh(self.first(value)))
