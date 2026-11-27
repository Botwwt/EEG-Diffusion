import torch
from torch.testing import assert_close

from paper_shared_response import group_mean, noise_regularized_views, shared_directions


def test_group_mean_uses_stimulus_identity_without_row_order_assumption():
    values = torch.tensor([[2., 1.], [4., 3.], [8., 5.], [10., 7.]])
    ids = torch.tensor([1, 0, 1, 0])
    result = group_mean(values, ids)
    assert_close(result, torch.tensor([[7., 5.], [5., 3.]]))


def test_shared_directions_find_between_subject_variation():
    torch.manual_seed(18)
    common = torch.linspace(-1, 1, 128)[:, None]
    views = [torch.cat((common, torch.randn(len(common), 2) * .16), 1) for _ in range(3)]
    whitened = [noise_regularized_views(view, .5) for view in views]
    space = shared_directions(whitened, 1)
    assert space.shape == (128, 1)
    correlation = torch.corrcoef(torch.cat((common, space), 1).T)[0, 1].abs()
    assert float(correlation) > .95
