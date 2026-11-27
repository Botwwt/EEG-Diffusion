import torch

from paper_fused_prior import PhotoDenoiser, fit_coordinates, from_coordinates, to_coordinates


def test_visual_coordinates_preserve_full_condition():
    values = torch.randn(54, 8, generator=torch.Generator().manual_seed(8))
    state = fit_coordinates(values)
    scores = to_coordinates(values, state)
    torch.testing.assert_close(scores.mean(0), torch.zeros(8), atol=1e-6, rtol=0)
    torch.testing.assert_close(scores.var(0), torch.ones(8), atol=2e-5, rtol=0)
    torch.testing.assert_close(from_coordinates(scores, state), values, atol=3e-6, rtol=3e-6)


def test_zero_initialized_network_returns_gaussian_denoising():
    clean = torch.arange(18).reshape(3, 6).float()
    sigma = torch.tensor([.2, .7, 1.2])
    model = PhotoDenoiser(6, width=12)
    expected = clean / (1 + sigma[:, None].square())
    torch.testing.assert_close(model(clean, sigma), expected)
