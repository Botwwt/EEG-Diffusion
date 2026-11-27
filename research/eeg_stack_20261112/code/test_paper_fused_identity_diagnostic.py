import torch

from paper_fused_identity_diagnostic import identity_axes, matched_updates


def test_identity_axes_prefer_stable_object_information_to_photo_variance():
    offsets = torch.tensor([-1., 0., 1.]).repeat_interleave(4)
    nuisance = torch.tensor([-10., -4., 4., 10.]).repeat(3)
    images = torch.stack((offsets, nuisance), 1)
    labels = torch.arange(3).repeat_interleave(4)
    identity, pca = identity_axes(images, labels, 1)
    torch.testing.assert_close(identity.abs(), torch.tensor([[1.], [0.]]))
    torch.testing.assert_close(pca.abs(), torch.tensor([[0.], [1.]]))


def test_diagnostic_swaps_match_correction_norm_without_amplifying():
    hints = [torch.tensor([[4., 0.], [0., 0.]]),
             torch.tensor([[0., 3.], [3., 0.]]),
             torch.tensor([[0., -2.], [0., 1.]])]
    results = matched_updates(hints)
    for before, after in zip(hints, results, strict=True):
        torch.testing.assert_close(after.norm(dim=1), torch.tensor([2., 0.]))
        assert torch.all(after.abs() <= before.abs())
