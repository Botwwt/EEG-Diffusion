import unittest

import numpy as np
import torch

from paper_background_invariant import blend_contexts, choose_backgrounds, object_directions


class BackgroundInvariantTest(unittest.TestCase):
    def test_shared_background_enters_both_objects_at_the_same_positions(self):
        objects = torch.tensor([[[[[2., 2.]]], [[[3., 3.]]]]])
        alpha = torch.tensor([[[[[1., 0.]]], [[[1., 0.]]]]])
        backgrounds = torch.tensor([[[[[4., 4.]]], [[[5., 5.]]]]])
        result = blend_contexts(objects, alpha, backgrounds)

        self.assertEqual(tuple(result.shape), (1, 2, 2, 1, 1, 2))
        torch.testing.assert_close(result[0, :, :, 0, 0, 0], torch.tensor([[2., 2.], [3., 3.]]))
        torch.testing.assert_close(result[0, :, :, 0, 0, 1], torch.tensor([[4., 5.], [4., 5.]]))

    def test_removes_the_two_measured_background_directions(self):
        views = torch.tensor([[[[2., 0., 1.], [0., 0., 1.]], [[0., 2., 0.], [0., 0., 0.]]]])
        original, foreground, residual = object_directions(views, torch.tensor([[[1., 0., 0.], [0., 0., 1.]]]), ridge=0.)

        torch.testing.assert_close(original[0], torch.tensor([1., 0., -1.]) / 2 ** .5)
        torch.testing.assert_close(foreground[0], torch.tensor([1., -1., 1.]) / 3 ** .5)
        torch.testing.assert_close(residual[0], torch.tensor([0., 0., 1.]))

    def test_backgrounds_are_other_known_training_objects(self):
        groups = np.repeat(np.arange(6), 2)
        pairs = np.array([[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11]])
        b = choose_backgrounds(groups, pairs)
        self.assertEqual(b.shape, pairs.shape)
        self.assertTrue(np.all(groups[b[:, 0]] != groups[b[:, 1]]))
        self.assertTrue(np.all(groups[b][:, :, None] != groups[pairs][:, None, :]))

    def test_collinear_background_has_finite_ridge_projection(self):
        images = torch.tensor([[[[1., 0.], [0., 0.]], [[1., 1.], [0., 1.]]]])
        original = torch.tensor([[[0., 0.], [0., 1.]]])
        _, _, residual = object_directions(images, original)
        torch.testing.assert_close(residual, torch.tensor([[0., -1.]]))
