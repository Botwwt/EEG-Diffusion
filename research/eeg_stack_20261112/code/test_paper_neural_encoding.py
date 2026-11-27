import unittest

import numpy as np

from paper_neural_encoding import fit_templates, neural_cosine, other_photo_scores


class NeuralEncodingTest(unittest.TestCase):
    def test_concept_fit_ignores_balanced_photo_variation(self):
        images = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float32)
        response = images @ np.diag(np.array([2, 3], dtype=np.float32))
        noise = np.array([[.1, 0], [-.1, 0], [0, .1], [0, -.1]], dtype=np.float32)
        eeg = response[:, None, :] + noise
        center, heads, noise_cov, within_cov = fit_templates(images, eeg, np.repeat(np.arange(2), 2))

        np.testing.assert_array_equal(center, [0, 0])
        self.assertGreater(heads['photo'][1, 1], 2)
        np.testing.assert_array_equal(heads['concept'][1], [0, 0])
        self.assertGreater(within_cov[1, 1], noise_cov[1, 1])

    def test_whitened_cosine_uses_same_transform_on_both_sides(self):
        eeg = np.array([[1, 2], [0, 0]], dtype=np.float32)
        templates = np.array([[1, 0], [0, 2], [-1, -2]], dtype=np.float32)
        covariance = np.diag(np.array([1, 16], dtype=np.float32))
        scores = neural_cosine(eeg, templates, covariance)

        self.assertGreater(scores[0, 0], scores[0, 1])
        self.assertAlmostEqual(float(scores[0, 2]), -1, places=6)
        np.testing.assert_array_equal(scores[1], 0)

    def test_prototypes_exclude_the_query_photo_and_match_support_size(self):
        ranking = np.tile([4, 9, 3, 8], (4, 1))
        labels = np.array([4, 9, 4, 9])
        scores, keys = other_photo_scores(ranking, labels)

        np.testing.assert_array_equal(keys, [0, 1, 0, 1])
        np.testing.assert_array_equal(scores, [[3, 8], [3, 8], [4, 9], [4, 9]])


if __name__ == '__main__':
    unittest.main()
