import pytest
import torch
from torch.nn import functional as F

from paper_o13_composition import binned_layout, match_training_layout, mix_directions


def test_pooled_oof_decoder_matches_project_then_pool():
    rng = torch.Generator().manual_seed(43)
    coefficients = torch.randn(3, 6, generator=rng)
    basis = torch.randn(4 * 64 * 64, 6, generator=rng)
    mean = torch.randn(4 * 64 * 64, generator=rng)
    decoded = (coefficients @ basis.T + mean).view(3, 4, 64, 64)
    torch.testing.assert_close(binned_layout(coefficients, basis, mean), F.avg_pool2d(decoded, 4).flatten(1))


def test_composition_does_not_use_other_queries_to_set_center_or_scale():
    signal = torch.tensor([[2., 6.], [8., 18.]])
    calibration = torch.tensor([[1., 2.], [3., 10.]])
    visual = torch.tensor([[20., 30.], [24., 42.]])
    expected = match_training_layout(signal[:1], calibration, visual)
    torch.testing.assert_close(match_training_layout(signal, calibration, visual)[:1], expected)
    torch.testing.assert_close(expected, visual.mean(0, keepdim=True))


def test_equal_blend_of_two_unit_directions_ignores_component_scale():
    axis = torch.tensor([[1., 0., 0.]])
    first = torch.tensor([[8., 3., 0.]])
    second = torch.tensor([[4., 0., 12.]])
    expected = torch.tensor([[0., 1., 1.]]) / 2 ** .5
    torch.testing.assert_close(mix_directions(first, second, axis), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA checkpoint conversion requires a GPU")
def test_composition_uses_cpu_checkpoint_axis_with_cuda_readouts():
    axis = torch.tensor([[1., 0., 0.]])
    first = torch.tensor([[1., 1., 0.]], device="cuda")
    second = torch.tensor([[2., 0., 1.]], device="cuda")
    expected = torch.tensor([[0., 1., 1.]], device="cuda") / 2 ** .5
    torch.testing.assert_close(mix_directions(first, second, axis), expected)
