import unittest

import torch
from torch.nn.functional import normalize

from paper_tangent_calibration import aligned_tangent, fit_orthogonal, tangent


class TangentCalibrationTest(unittest.TestCase):
    def test_fit_recovers_paired_alignment_and_fixes_cone_axis(self):
        torch.manual_seed(9)
        axis = torch.tensor([1., 0., 0., 0., 0.])
        x = tangent(torch.randn(1200, 5), axis)
        orientation = torch.eye(5)
        orientation[1:3, 1:3] = torch.tensor([[0., -1.], [1., 0.]])
        y = x @ orientation
        edit = fit_orthogonal(x, y, axis, ridge=.001)
        torch.testing.assert_close(edit @ axis, axis)
        torch.testing.assert_close(edit.T @ edit, torch.eye(5), atol=2e-6, rtol=0.)
        self.assertGreater(float((x @ edit * y).sum(-1).mean()), .99)

    def test_forward_and_reverse_apply_same_angular_edit(self):
        torch.manual_seed(10)
        axis = normalize(torch.randn(9), dim=0)
        x = torch.randn(100, 9)
        photo = torch.randn(100, 9)
        center = tangent(x, axis).mean(0)
        reference = aligned_tangent(x, axis, center)
        warp = fit_orthogonal(reference, tangent(photo, axis), axis)
        correct = aligned_tangent(x, axis, center, warp)
        reverse = aligned_tangent(x, axis, center, warp.T)
        torch.testing.assert_close((reference*correct).sum(-1), (reference*reverse).sum(-1), atol=2e-6, rtol=0.)
        torch.testing.assert_close(correct @ axis, torch.zeros(len(x)), atol=2e-6, rtol=0.)
        torch.testing.assert_close(correct[:1], aligned_tangent(x[:1], axis, center, warp))
