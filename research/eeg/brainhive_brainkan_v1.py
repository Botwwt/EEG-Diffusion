"""Physical-input, parameter-matched low-rank spline BrainKAN replacement.

This is a hybrid additive KAN, not the original dense multi-layer KAN. Its
named electrode/time coordinates remain available before the final LayerNorm;
these contributions are model attributions, not physiological causal effects.
"""
import hashlib
import json

import torch
from torch import nn
import torch.nn.functional as F


class BrainKAN(nn.Module):
    input_dim = 4250
    hidden_size = 1024
    rank = 60
    basis_count = 8
    expected_parameters = 6456512

    def __init__(self, dropout=.1):
        super().__init__()
        self.proj_in = nn.Linear(self.input_dim, self.hidden_size)
        self.spline_coefficients = nn.Parameter(torch.empty(self.input_dim, self.basis_count, self.rank))
        self.spline_out = nn.Linear(self.rank, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        # Eight cubic bases over five intervals: four repeated endpoint knots.
        self.register_buffer('knots', torch.tensor([-1.] * 4 + [-.6, -.2, .2, .6] + [1.] * 4))
        self.register_buffer('input_median', torch.zeros(self.input_dim))
        self.register_buffer('input_scale', torch.ones(self.input_dim))
        self.register_buffer('statistics_fitted', torch.tensor(False))
        # Neither factor is zero: both receive gradients on the first update.
        nn.init.normal_(self.spline_coefficients, std=.01 / self.input_dim ** .5)
        nn.init.xavier_uniform_(self.spline_out.weight)
        assert sum(p.numel() for p in self.parameters()) == self.expected_parameters

    @staticmethod
    def cubic_basis(u, knots):
        """Cox-de Boor basis, including the closed right endpoint, FP32/64."""
        knots = knots.to(device=u.device, dtype=u.dtype)
        assert knots.numel() == 12
        assert torch.isfinite(u).all()
        assert bool(((u >= knots[0]) & (u <= knots[-1])).all())
        z = u.unsqueeze(-1)
        values = ((z >= knots[:-1]) & (z < knots[1:])).to(u.dtype)
        for degree in range(1, 4):
            count = knots.numel() - degree - 1
            left_den = knots[degree:degree + count] - knots[:count]
            right_den = knots[degree + 1:degree + count + 1] - knots[1:count + 1]
            left = (z - knots[:count]) / left_den.clamp_min(torch.finfo(u.dtype).tiny)
            right = (knots[degree + 1:degree + count + 1] - z) / right_den.clamp_min(torch.finfo(u.dtype).tiny)
            left = torch.where(left_den > 0, left, torch.zeros_like(left))
            right = torch.where(right_den > 0, right, torch.zeros_like(right))
            values = left * values[..., :count] + right * values[..., 1:count + 1]
        endpoint = torch.zeros_like(values)
        endpoint[..., -1] = 1
        return torch.where(z == knots[-1], endpoint, values)

    def basis(self, x):
        # Basis recursion stays FP32 even inside native BF16 autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            values = torch.tanh((x.float() - self.input_median.float()) / self.input_scale.float())
            return self.cubic_basis(values, self.knots.float())

    @torch.no_grad()
    def fit_training_statistics(self, fit_x, provenance, coordinate_chunk=128):
        """Caller supplies ONLY the exact permitted T rows; provenance is required."""
        assert fit_x.device.type == 'cpu' and not bool(self.statistics_fitted)
        x = fit_x.reshape(len(fit_x), -1).float()
        assert x.shape[1] == self.input_dim and len(x) > 1
        assert provenance['split'] == 'T' and provenance['rows'] == len(x)
        assert len(provenance['array_rows']) == len(set(provenance['array_rows'])) == len(x)
        assert len(provenance['source_sha256']) == len(provenance['split_manifest_sha256']) == 64
        centers, scales, constant = [], [], 0
        for lo in range(0, self.input_dim, coordinate_chunk):
            chunk = x[:, lo:lo + coordinate_chunk]
            assert torch.isfinite(chunk).all()
            center = chunk.median(dim=0).values
            scale = 1.4826 * (chunk - center).abs().median(dim=0).values
            weak = scale < 1e-6
            constant += int(weak.sum())
            centers.append(center)
            scales.append(torch.where(weak, torch.ones_like(scale), scale))
        self.input_median.copy_(torch.cat(centers))
        self.input_scale.copy_(torch.cat(scales))
        self.statistics_fitted.fill_(True)
        record = dict(provenance, method='Coordinate median and 1.4826*MAD on exact fit T rows; scale<1e-6 ->1',
                      spline_input='tanh((physical_input-median_T)/scale_T)',
                      original_linear_branch_input='unchanged physical mean4 EEG; no new z-score',
                      constant_or_small_scale_coordinates=constant,
                      median_sha256=hashlib.sha256(self.input_median.cpu().numpy().tobytes()).hexdigest(),
                      scale_sha256=hashlib.sha256(self.input_scale.cpu().numpy().tobytes()).hexdigest())
        record['provenance_sha256'] = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return record

    def pre_norm(self, eeg):
        assert bool(self.statistics_fitted), 'Fit statistics on permitted T rows before any forward pass'
        x = eeg.reshape(len(eeg), -1)
        assert x.shape[1] == self.input_dim
        basis = self.basis(x)
        rank_values = F.linear(basis.flatten(1), self.spline_coefficients.reshape(-1, self.rank).T)
        return self.proj_in(x) + self.spline_out(self.dropout(rank_values))

    def forward(self, eeg):
        return self.norm(self.pre_norm(eeg))

    @torch.no_grad()
    def attribute_groups(self, eeg, groups, baseline=None):
        """Exact eval-mode pre-LN difference decomposition for a partition of inputs.

        Return [batch, groups, 1024]; never allocate [batch, 4250, 1024].
        LN/A/P/UNet contributions are NOT claimed to be additive.
        """
        assert not self.training
        x = eeg.reshape(len(eeg), -1).float()
        baseline = self.input_median if baseline is None else baseline.reshape(-1).to(x)
        assert baseline.shape == (self.input_dim,)
        flat = [int(j) for group in groups for j in group]
        assert sorted(flat) == list(range(self.input_dim)), 'Groups must partition all named input coordinates'
        basis_difference = self.basis(x) - self.basis(baseline.unsqueeze(0))
        result = []
        for group in groups:
            ids = torch.tensor(group, device=x.device, dtype=torch.long)
            linear = F.linear(x[:, ids] - baseline[ids], self.proj_in.weight[:, ids])
            rank = F.linear(basis_difference[:, ids].flatten(1), self.spline_coefficients[ids].reshape(-1, self.rank).T)
            result.append(linear + self.spline_out(rank))
        return torch.stack(result, dim=1)


def fixed_electrode_time_groups():
    """17 electrodes x ten non-overlapping 25-sample (100 ms nominal) windows."""
    return [[channel * 250 + t for t in range(start, start + 25)]
            for channel in range(17) for start in range(0, 250, 25)]
