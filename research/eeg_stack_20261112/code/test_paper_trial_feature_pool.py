import torch
from torch import nn

from paper_trial_feature_pool import pool_features


def test_identical_trials_agree_with_or_without_early_pooling():
    trials = torch.randn(2, 1, 3, 19).repeat(1, 4, 1, 1)
    encode = nn.Sequential(nn.Conv1d(3, 5, 3, padding=1), nn.GELU()).eval()

    before = pool_features(encode, trials, "before")
    after = pool_features(encode, trials, "after")
    torch.testing.assert_close(before, after)


def test_channel_precision_is_permutation_invariant_and_commutes_with_linear_filters():
    trials = torch.randn(2, 4, 3, 19)
    linear = nn.Conv1d(3, 5, 3, padding=1).eval()
    permutation = [2, 0, 3, 1]

    before = pool_features(linear, trials, "before")
    after = pool_features(linear, trials, "after")
    torch.testing.assert_close(before, after)
    torch.testing.assert_close(after, pool_features(linear, trials[:, permutation], "after"))


def test_after_pooling_keeps_nonlinear_trial_information():
    trials = torch.tensor([-1., -1., 1., 1.]).reshape(1, 4, 1, 1).repeat(1, 1, 1, 12)
    nonlinear = nn.ReLU()

    assert torch.equal(pool_features(nonlinear, trials, "before"), torch.zeros(1, 1, 12))
    assert torch.equal(pool_features(nonlinear, trials, "after"), torch.full((1, 1, 12), .5))
