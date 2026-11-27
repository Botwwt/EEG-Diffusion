import unittest

import torch
from torch.nn import functional as F

from paper_visual_likelihood import band_limited_step


class VisualLikelihoodTest(unittest.TestCase):
    def test_preserves_all_coarse_cells_and_keeps_gradient_ascent(self):
        value = torch.randn(2, 3, 16, 16, generator=torch.Generator().manual_seed(21))
        value[1] = F.interpolate(F.avg_pool2d(value[1:], 4), scale_factor=4, mode="nearest")[0]
        step = band_limited_step(value, block=4, strength=.15)
        self.assertTrue(torch.isfinite(step).all())
        torch.testing.assert_close(F.avg_pool2d(step, 4), torch.zeros(2, 3, 4, 4), atol=1e-7, rtol=0)
        torch.testing.assert_close(step[0].square().mean().sqrt(), torch.tensor(.15))
        self.assertGreater(float((step[0] * value[0]).sum()), 0.)
        torch.testing.assert_close(step[1], torch.zeros_like(step[1]), atol=1e-7, rtol=0)
