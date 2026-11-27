import unittest

import numpy as np

from paper_shared_infer import study_budget


class SharedInferenceTest(unittest.TestCase):
    def test_only_registered_repetition_budgets_are_accepted(self):
        for repeats in (4, 80):
            trials = np.ones((repeats, 17, 250), dtype=np.float32)
            self.assertEqual(study_budget(trials, 3), f'official-r{repeats}')
        with self.assertRaisesRegex(ValueError, 'four or eighty'):
            study_budget(np.ones((5, 17, 250), dtype=np.float32), 3)

    def test_participant_and_waveform_coordinates_are_checked(self):
        for subject in (0, 11):
            with self.assertRaisesRegex(ValueError, 'participant'):
                study_budget(np.ones((4, 17, 250), dtype=np.float32), subject)
        with self.assertRaisesRegex(ValueError, 'posterior'):
            study_budget(np.ones((4, 63, 250), dtype=np.float32), 3)


if __name__ == '__main__':
    unittest.main()
