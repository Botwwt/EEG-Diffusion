import torch

from paper_fused_posterior import anchored_reconstruction, data_term


def test_repeat_likelihood_update_satisfies_normal_equations():
    rng = torch.Generator().manual_seed(95)
    predictor = torch.randn(7, 12, generator=rng)
    observed = torch.randn(3, 12, generator=rng)
    prior = torch.randn(3, 7, generator=rng)
    strength = 2.2
    gram, rhs = data_term(predictor, observed)
    matrix = torch.eye(7) + strength * gram
    solved = torch.cholesky_solve((prior + strength * rhs).T, torch.linalg.cholesky(matrix)).T
    torch.testing.assert_close(solved @ matrix, prior + strength * rhs, rtol=5e-6, atol=5e-6)
    assert ((solved @ predictor - observed).square().sum()
            < (prior @ predictor - observed).square().sum())


def test_identity_prior_without_eeg_data_term_keeps_original_coordinates():
    feature = torch.tensor([[1., 4.]])
    model = torch.eye(2)
    data_a = data_term(model, torch.ones_like(feature))
    data_b = data_term(model, -torch.ones_like(feature))
    assert torch.equal(anchored_reconstruction(feature, *data_a, 0., "linear"), feature)
    first = anchored_reconstruction(feature, *data_a, .7, "gaussian")
    other = anchored_reconstruction(feature, *data_b, .7, "gaussian")
    assert torch.all(first > other)
