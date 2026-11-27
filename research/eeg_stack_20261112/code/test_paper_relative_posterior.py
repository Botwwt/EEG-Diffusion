import unittest

import torch

from paper_relative_posterior import conditional_mixture_estimates, replace_relative_coordinates


class ConditionalMixtureTests(unittest.TestCase):
    def test_single_gaussian_has_analytic_mean_and_mode(self):
        mean, mode = conditional_mixture_estimates(
            torch.tensor([1.0]), torch.tensor([[1.0]]), torch.tensor([[4.0]]),
            torch.tensor([[3.0]]), torch.tensor([[6.0], [-4.0]]),
        )
        expected = torch.tensor([[6.25], [-3.75]]) / 3.25
        torch.testing.assert_close(mean, expected)
        torch.testing.assert_close(mode, expected)

    def test_bimodal_map_survives_even_when_mean_falls_between_peaks(self):
        mean, mode = conditional_mixture_estimates(
            torch.tensor([0.5, 0.5]), torch.tensor([[-4.0], [4.0]]), torch.tensor([[0.25]]),
            torch.zeros(1, 1), torch.zeros(1, 1),
        )
        torch.testing.assert_close(mean, torch.zeros(1, 1))
        torch.testing.assert_close(mode.abs(), torch.full((1, 1), 4.0))

    def test_mixture_mean_and_mode_agree_with_direct_posterior_density(self):
        masses = torch.tensor([0.55, 0.45], dtype=torch.float64)
        centers = torch.tensor([[-1.5], [1.3]], dtype=torch.float64)
        covariance = torch.tensor([[0.6]], dtype=torch.float64)
        precision = torch.tensor([[1.2]], dtype=torch.float64)
        information = torch.tensor([[0.4]], dtype=torch.float64)
        mean, mode = conditional_mixture_estimates(masses, centers, covariance, precision, information)

        x = torch.linspace(-8, 8, 16001, dtype=torch.float64)
        prior = (masses * torch.exp(-0.5 * (x[:, None] - centers.T).square() / covariance.item())).sum(1)
        density = prior * torch.exp(-0.5 * precision.item() * x.square() + information.item() * x)
        self.assertAlmostEqual(mean.item(), ((x * density).sum() / density.sum()).item(), places=9)
        self.assertAlmostEqual(mode.item(), x[density.argmax()].item(), delta=0.002)

    def test_replacing_coordinates_preserves_common_token_and_requested_residual(self):
        tokens = torch.tensor([[[2.], [4.], [6.], [8.]], [[1.], [0.], [3.], [4.]]])
        center = torch.tensor([1., 0., 0., -1.])
        analysis = torch.tensor([[.5], [-.5], [0.], [0.]])
        synthesis = torch.tensor([[1.], [-1.], [1.], [-1.]])
        source_common = tokens.mean(dim=1, keepdim=True)
        displacement = (tokens - source_common).flatten(1) - center
        original_coordinates = displacement @ analysis
        original = replace_relative_coordinates(tokens, center, analysis, synthesis, original_coordinates, 1.)
        torch.testing.assert_close(original, tokens)

        target = torch.tensor([[2.], [-1.]])
        changed = replace_relative_coordinates(tokens, center, analysis, synthesis, target, .5)
        offset = (changed - source_common).flatten(1) - center
        torch.testing.assert_close(changed.mean(dim=1, keepdim=True), source_common)
        torch.testing.assert_close(offset @ analysis, target)
        original_residual = displacement - original_coordinates @ synthesis.mT
        torch.testing.assert_close(offset - target @ synthesis.mT, .5 * original_residual)


if __name__ == "__main__":
    unittest.main()
