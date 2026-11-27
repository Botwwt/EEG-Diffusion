"""The R80 readout must normalize each EEG query independently."""
import io
import unittest

import numpy as np
import torch

from paper_fusion_siglip_probe import official_conditions


class OfficialConditionTests(unittest.TestCase):
    def test_other_stimuli_cannot_change_a_query(self):
        frame = np.ones((10, 200, 3), dtype=np.float32)
        frame[:, :, 1] *= 2
        frame[:, :, 2] *= 3
        changed = frame.copy()
        changed[0, 10, 1] *= 50
        hit = (np.arange(2000) < 937).reshape(10, 200)

        def read(embedding):
            buffer = io.BytesIO()
            np.savez(buffer, embedding=embedding, target=frame[0], feature_top1=hit,
                     subjects=np.arange(1, 11), row=np.arange(200))
            buffer.seek(0)
            return official_conditions(buffer, "cpu")[0]

        original, revised = read(frame), read(changed)
        torch.testing.assert_close(original.norm(dim=-1), torch.ones((10, 200)))
        self.assertTrue(torch.equal(original[0, 0], revised[0, 0]))
        self.assertFalse(torch.equal(original[0, 10], revised[0, 10]))


if __name__ == "__main__":
    unittest.main()
