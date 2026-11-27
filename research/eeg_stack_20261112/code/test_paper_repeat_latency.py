import torch

from paper_repeat_latency import aggregate, shift_marginals


def test_signed_alignment_recovers_known_independent_trial_delays():
    generator = torch.Generator().manual_seed(19)
    waveform = torch.randn(3, 250, generator=generator)
    delay = (-4, -2, 2, 4)
    repeats = torch.stack([waveform.roll(value, dims=-1) for value in delay])[None]

    result = aggregate(repeats, torch.zeros_like(waveform), torch.eye(3), 64.)
    target = waveform[:, 25:90]
    assert (result["aligned"][0, :, 25:90] - target).square().mean() < .005
    assert (result["symmetric"][0, :, 25:90] - target).square().mean() > .1
    assert torch.equal(result["mean"], repeats.mean(1))


def test_shift_marginals_preserve_trial_order_and_do_not_mix_queries():
    generator = torch.Generator().manual_seed(73)
    features = torch.randn(2, 4, 3, 250, generator=generator)
    permutation = [2, 0, 3, 1]
    weights = shift_marginals(features, 8.)
    permuted = shift_marginals(features[:, permutation], 8.)
    torch.testing.assert_close(weights[:, permutation], permuted)
    torch.testing.assert_close(weights[:1], shift_marginals(features[:1], 8.))
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4))


def test_zero_strength_uses_the_same_precision_pool_as_the_control():
    generator = torch.Generator().manual_seed(18)
    values = torch.randn(2, 2, 3, 250, generator=generator)
    center, axes = torch.randn(3, 250, generator=generator), torch.eye(3)
    result = aggregate(values, center, axes, 0.)
    power = values.square().mean(-1, keepdim=True).clamp_min(1e-6)
    weights = power.rsqrt()
    expected = (weights * values).sum(1) / weights.sum(1)
    torch.testing.assert_close(result["aligned"], expected)
    torch.testing.assert_close(result["aligned"], result["symmetric"])
