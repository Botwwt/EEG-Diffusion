import unittest

import torch

from paper_photo_top1 import keep_token_mean


class O13LateConditionTest(unittest.TestCase):
    def test_mean_is_from_recipient_and_slot_differences_are_from_blend(self):
        base = torch.tensor([[[1., 3.], [2., 4.], [3., 5.], [2., 4.]]])
        blend = torch.tensor([[[6., 5.], [8., 3.], [4., 7.], [6., 5.]]])
        projected = keep_token_mean(base, blend)

        torch.testing.assert_close(projected.mean(1), base.mean(1))
        torch.testing.assert_close(projected[:, 1:] - projected[:, :1], blend[:, 1:] - blend[:, :1])


if __name__ == '__main__':
    unittest.main()
