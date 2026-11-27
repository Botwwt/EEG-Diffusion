import torch

from paper_shared_signal import boost_shared, peer_directions, rotate_photographs, whitening


def test_other_photo_control_preserves_every_training_object_and_changes_its_photo():
    object_ids = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    donor = rotate_photographs(object_ids, 1)
    assert torch.equal(donor.sort().values, torch.arange(len(object_ids)))
    assert torch.equal(object_ids[donor], object_ids)
    assert torch.all(donor != torch.arange(len(object_ids)))


def test_whitened_regression_is_the_same_unfiltered_ridge_control():
    g = torch.Generator().manual_seed(17)
    x = torch.randn(60, 7, generator=g)
    y = torch.randn(60, 12, generator=g)
    scaled = (x - x.mean(0)) / x.std(0)
    white, test = whitening(x, x)
    expected = torch.linalg.solve(scaled.T @ scaled / len(x) + .15 * torch.eye(x.shape[1]),
                                  scaled.T @ y / len(x))
    torch.testing.assert_close(test @ (white.T @ y / len(x)), scaled @ expected, atol=1e-6, rtol=1e-5)


def test_shared_mode_follows_the_paired_training_response():
    x = torch.tensor([[1., 1.], [-1., 1.], [1., -1.], [-1., -1.]])
    peer = torch.stack((x[:, 0], torch.zeros_like(x[:, 0])), dim=-1)
    axis = peer_directions(x, [peer], 1)
    vector = boost_shared(torch.ones(2, 1), axis)
    torch.testing.assert_close(vector.flatten(), torch.tensor([1., .5]))
