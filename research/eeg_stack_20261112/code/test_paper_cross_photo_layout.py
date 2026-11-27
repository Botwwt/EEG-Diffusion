"""Regression checks for the fixed-reference assignment comparison."""
import unittest

import torch

from paper_cross_photo_layout import assigned_references, paired_margin


class CrossPhotoLayoutTests(unittest.TestCase):
    def test_matching_and_crossing_use_the_same_distinct_photos(self):
        costs = torch.tensor([[[1., 2., 5.], [2., 8., 9.], [5., 9., 10.]]])
        left, right = assigned_references(costs)
        self.assertEqual(left.tolist(), [[0, 1]])
        self.assertEqual(right.tolist(), [[1, 0]])
        self.assertEqual(right.flip(-1).sort().values.tolist(), right.sort().values.tolist())
        self.assertLess(costs[0, left[0], right[0]].sum(), costs[0, left[0], right[0].flip(0)].sum())

    def test_common_query_offset_cancels_in_the_pair_margin(self):
        q = torch.tensor([[2., 1.]])
        p = torch.tensor([[1., 0.]])
        a = torch.tensor([[[1., 1.], [3., 2.]]])
        b = torch.tensor([[[0., 1.], [1., 0.]]])
        self.assertTrue(torch.equal(paired_margin(q, p, a, b), paired_margin(q + 7, p + 7, a, b)))
        self.assertTrue(torch.equal(paired_margin(q, p, a, b), -paired_margin(p, q, a, b)))

    def test_distinct_eeg_contexts_change_the_mean_margin(self):
        q = torch.tensor([[[2., 0.], [0., 2.]]])
        reference = torch.tensor([[[1., 0.], [0., 1.]]])
        aligned = paired_margin(q, torch.zeros_like(q), reference, torch.zeros_like(reference))
        crossed = paired_margin(q, torch.zeros_like(q), reference.flip(1), torch.zeros_like(reference))
        self.assertEqual(aligned.tolist(), [[2., 2.]])
        self.assertEqual(crossed.tolist(), [[0., 0.]])


if __name__ == "__main__":
    unittest.main()
