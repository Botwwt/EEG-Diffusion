import unittest

import torch

from paper_spectral_readout import power_components


class SpectralReadoutTests(unittest.TestCase):
    def test_coherent_and_induced_power_obey_the_repeat_variance_identity(self):
        wave = torch.randn(8, 4, 17, 4, 20)
        coherent, induced = power_components(wave)
        torch.testing.assert_close(coherent + .75 * induced, wave.square().mean(1))

    def test_antiphase_trials_have_induced_power_without_coherent_power(self):
        wave = torch.tensor([1., -1., 2., -2.]).reshape(1, 4, 1, 1, 1)
        coherent, induced = power_components(wave)
        self.assertEqual(coherent.item(), 0.)
        self.assertAlmostEqual(induced.item(), 10. / 3., places=6)


if __name__ == "__main__":
    unittest.main()
