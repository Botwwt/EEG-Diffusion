import unittest

import numpy as np
import torch

from paper_temporal_readout import TemporalReadout, paired_batches


class TemporalReadoutTests(unittest.TestCase):
    def test_paired_views_change_only_the_peer_photograph(self):
        groups = np.arange(40).reshape(4, 10)
        offsets = np.arange(40) % 10
        seen = []
        other_seen = []
        for people, photos in paired_batches(groups, offsets, 3, np.random.default_rng(2), batch_size=3):
            np.testing.assert_array_equal(photos[0], photos[1])
            np.testing.assert_array_equal(people[1], people[2])
            self.assertTrue((people[0] != people[1]).all())
            self.assertTrue((photos[0] != photos[2]).all())
            np.testing.assert_array_equal(photos[0] // 10, photos[2] // 10)
            self.assertEqual(len(set(photos[0] // 10)), photos.shape[1])
            seen.extend(photos[0])
            other_seen.extend(photos[2])
        self.assertEqual(sorted(seen), list(range(40)))
        self.assertEqual(sorted(other_seen), list(range(40)))

    def test_encoder_outputs_two_normalized_visual_embeddings(self):
        counts = []
        for segments in (1, 4):
            model = TemporalReadout(segments=segments)
            photo, semantic = model(torch.randn(4, 63, 100))
            self.assertEqual(photo.shape, semantic.shape)
            self.assertEqual(photo.shape, (4, 1024))
            for features in (photo, semantic):
                torch.testing.assert_close(features.norm(dim=-1), torch.ones(4))
            self.assertFalse(torch.equal(photo, semantic))
            counts.append(sum(parameter.numel() for parameter in model.parameters()))
        self.assertLess(abs(counts[0] / counts[1] - 1), .002)


if __name__ == "__main__":
    unittest.main()
