import torch

from paper_condition_calibration import adjust, fit_moments, orthogonal_update, relative_shape


def test_training_moments_restore_spread_and_predict_each_query_independently():
    rng = torch.Generator().manual_seed(11)
    photos = torch.randn(30, 4, 5, generator=rng)
    predicted = .25 * photos + torch.arange(5)
    calibration = fit_moments(predicted, photos)
    torch.testing.assert_close(adjust(predicted, calibration), photos)
    q = torch.randn(3, 4, 5, generator=rng)
    torch.testing.assert_close(adjust(q, calibration)[1:2], adjust(q[1:2], calibration))


def test_update_keeps_d2_mean_and_direction_and_matches_wrong_eeg_energy():
    rng = torch.Generator().manual_seed(3)
    old = torch.randn(6, 4, 9, generator=rng)
    new = torch.randn(6, 4, 9, generator=rng)
    old_direction = relative_shape(old - old.mean(0))
    reference = orthogonal_update(new, old_direction)
    sigma = reference.flatten(1).norm(dim=1)[:, None, None]
    wrong = orthogonal_update(new.roll(3, dims=0), old_direction, requested_norm=sigma)
    torch.testing.assert_close(wrong.flatten(1).norm(dim=1), sigma.flatten())
    torch.testing.assert_close((old + reference).mean(1), old.mean(1))
    torch.testing.assert_close((reference * old_direction).sum((1, 2)), torch.zeros(6), atol=2e-6, rtol=0)
    torch.testing.assert_close((wrong * old_direction).sum((1, 2)), torch.zeros(6), atol=2e-6, rtol=0)
