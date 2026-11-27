import torch

from paper_trial_spectra import FEATURE_DIM, extract_spectra, repeat_moments, wavelet_bank


def test_repeat_moments_split_shared_and_disagreeing_phases():
    same = torch.ones(1, 4, 1, 1, 1, dtype=torch.complex64)
    opposed = same.clone()
    opposed[:, 2:] *= -1
    locked, cross, residual = repeat_moments(same)
    assert locked.item() == cross.item() == 1.
    assert residual.item() == 0.
    locked, cross, residual = repeat_moments(opposed)
    torch.testing.assert_close(locked, torch.zeros_like(locked))
    torch.testing.assert_close(cross, torch.full_like(cross, -1./3))
    torch.testing.assert_close(residual, torch.full_like(residual, 4./3))


def test_residual_power_is_invariant_to_an_added_common_waveform():
    rng = torch.Generator().manual_seed(44)
    x = torch.randn(2, 4, 17, 250, generator=rng)
    bank = wavelet_bank()
    one = extract_spectra(x, bank)
    two = extract_spectra(x + torch.randn(2, 1, 17, 250, generator=rng), bank)
    assert all(a.shape == (2, FEATURE_DIM) for a in one.values())
    torch.testing.assert_close(one["residual"], two["residual"], atol=1e-6, rtol=1e-5)
    assert not torch.equal(one["mean"], two["mean"])
