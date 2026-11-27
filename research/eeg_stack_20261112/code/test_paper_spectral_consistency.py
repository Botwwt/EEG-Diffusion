import torch

from paper_spectral_consistency import paired_residual_power


def test_a_single_outlying_trial_cannot_create_consistent_pair_power():
    coeff = torch.tensor([1., 1., 1., -20.], dtype=torch.complex64).reshape(1, 4, 1, 1, 1)
    assert paired_residual_power(coeff).item() == 0.


def test_two_pair_residual_agreement_is_invariant_to_trial_order():
    rng = torch.Generator().manual_seed(2026)
    coeff = torch.randn(3, 4, 5, 6, 7, generator=rng, dtype=torch.complex64)
    result = paired_residual_power(coeff)
    assert torch.all(result >= 0)
    torch.testing.assert_close(result, paired_residual_power(coeff[:, [3, 0, 2, 1]]), atol=1e-6, rtol=1e-6)
