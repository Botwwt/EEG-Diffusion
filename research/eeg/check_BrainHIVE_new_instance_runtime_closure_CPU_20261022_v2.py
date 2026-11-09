"""CPU coverage/rejection checks for the read-only replacement-instance verifier."""
import copy
import sys
import unittest

import verify_BrainHIVE_new_instance_runtime_closure_CPU_20261022_v2 as verify


class ExactCoverage(unittest.TestCase):
    def setUp(self):
        self.expected = {f'/fixture/{i}': f'{i:064x}' for i in range(30)}
        self.rows = [dict(path=p, sha256=h) for p, h in self.expected.items()]

    def test_exact_full_identity_accepted(self):
        verify.exact_records(self.expected, list(reversed(self.rows)), 30)

    def test_empty_or_truncated_proof_rejected(self):
        for rows in [[], self.rows[:-1]]:
            with self.subTest(count=len(rows)), self.assertRaises(AssertionError):
                verify.exact_records(self.expected, rows, 30)

    def test_duplicate_paths_rejected(self):
        with self.assertRaises(AssertionError):
            verify.exact_records(self.expected, self.rows[:-1] + [self.rows[0]], 30)

    def test_changed_SHA_rejected(self):
        rows = copy.deepcopy(self.rows); rows[0]['sha256'] = 'f' * 64
        with self.assertRaises(AssertionError):
            verify.exact_records(self.expected, rows, 30)

    def test_wrong_path_or_extra_records_rejected(self):
        rows = copy.deepcopy(self.rows); rows[0]['path'] = '/fixture/unknown'
        for values in [rows, self.rows + [dict(path='/extra', sha256='a' * 64)]]:
            with self.subTest(count=len(values)), self.assertRaises(AssertionError):
                verify.exact_records(self.expected, values, 30)

    def test_copy_original_paths_use_exact_same_rule(self):
        copied = [dict(original_path=p, sha256=h) for p, h in self.expected.items()]
        verify.exact_records(self.expected, copied, 30, 'original_path')
        copied[-1]['original_path'] = '/wrong'
        with self.assertRaises(AssertionError):
            verify.exact_records(self.expected, copied, 30, 'original_path')

    def test_stage_status_required(self):
        with self.assertRaises(AssertionError):
            verify.stage_files('conditions', dict(status='RUNNING', files=self.rows))

    def test_original_input_stage_requires_all_JPEG_members(self):
        proof = dict(status=verify.STAGE_STATUS['inputs'], images_member_SHA_verified=16540,
            metadata=[dict(path='/metadata', sha256='a')], original_manifest_path='/container', container_sha256='b')
        self.assertEqual(len(verify.stage_files('inputs', proof)), 2)
        proof['images_member_SHA_verified'] = 1
        with self.assertRaises(AssertionError):
            verify.stage_files('inputs', proof)

    def test_latent_stage_rejects_training_or_generation(self):
        proof = dict(status=verify.STAGE_STATUS['latent'], T_images=13230, historical_receipt_unchanged=True,
            training_updates=0, UNet_forwards=0, images_generated=0, path='/latent', sha256='a')
        self.assertEqual(verify.stage_files('latent', proof), [dict(path='/latent', sha256='a')])
        for key in ['training_updates', 'UNet_forwards', 'images_generated']:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                verify.stage_files('latent', proof | {key: 1})

    def test_registered_exact_stage_denominators(self):
        self.assertEqual(verify.COUNTS, dict(public=19, inputs=5, conditions=30, latent=1, evaluation=25))
        self.assertEqual(sum(verify.COUNTS[s] for s in ['public', 'inputs', 'conditions', 'latent']), 55)
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
