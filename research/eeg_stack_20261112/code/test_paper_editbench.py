import unittest

import numpy as np

from paper_editbench import edit_outcomes


class EditBenchTest(unittest.TestCase):
    def test_each_endpoint_uses_its_own_alternatives_and_ties_are_misses(self):
        scores = np.array([[[.9, .5, .95, .8, 0.], [.8, .7, 0., 0., .7]]])
        source = np.array([0, 0, 1])
        original, edited_set, wins = edit_outcomes(scores, 2, source)

        np.testing.assert_array_equal(original, [[True, False]])
        np.testing.assert_array_equal(edited_set, [[False, False]])
        np.testing.assert_array_equal(wins, [[False, True, False]])
