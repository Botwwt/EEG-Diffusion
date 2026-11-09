"""Synthetic CPU payload rejection checks; no actual checkpoint or EEG cache."""
import os
import unittest
from pathlib import Path
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import audit_BrainHIVE_published_parent_payloads_CPU_20261022_v1 as A


class PayloadChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(1729)
        cls.model = A.source_class(torch)(4250, 1024, num_layers=1, dropout=.1)
        cls.state = dict(seed=1729, epochs=25, updates=425, registration_sha256=A.REG_SHA,
                         model={k: v.detach().clone() for k, v in cls.model.state_dict().items()}, logit_scale=torch.tensor(2.6592))
        cls.cache = dict(role='published', model_sha256='1' * 64, normalization='raw LayerNorm output',
                         array_rows=torch.arange(200), condition=torch.zeros(200, 1024))

    def check(self, state=None, cache=None):
        return A.validate_loaded(torch, self.model, state or self.state, cache or self.cache, '1' * 64)

    def test_original_metadata_pins_are_complete_and_match_original_files(self):
        for value in [A.REG_SHA, A.PROOF_SHA, A.FETCH_SHA, *A.SOURCE_PINS.values()]:
            self.assertEqual(len(value), 64)
            self.assertEqual(set(value) - set('0123456789abcdef'), set())
        original = A.ROOT / 'audit/BrainHIVE_native_proof_bundle_20261020_v4/native_work/runs/published_source_dropout_v2/registration.json'
        self.assertEqual(A.sha(original), A.REG_SHA)
        self.assertEqual(A.sha(A.ROOT / 'audit/BrainHIVE_all40_parent_training_CPU_verification_v1.json'), A.PROOF_SHA)
        self.assertEqual(A.sha(A.UPSTREAM / '_fetch_receipt.json'), A.FETCH_SHA)

    def test_positive_native_class_full_state_and_existing_cache_schema(self):
        result = self.check()
        self.assertEqual(result['parameters'], 6452224)
        self.assertTrue(result['existing_R80_cache_model_SHA_link_valid'])
        self.assertFalse(result['original_R80_EEG_or_export_recomputed'])
        self.assertEqual(result['forwards'], 0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_other_seed_scope_registration_or_incomplete_fit_rejected(self):
        for key, value in [('seed', 42), ('epochs', 24), ('updates', 325), ('registration_sha256', '2' * 64)]:
            state = dict(self.state, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(state=state)

    def test_missing_or_unexpected_model_keys_fail_strict_load(self):
        for mode in ('missing', 'extra'):
            weights = dict(self.state['model'])
            if mode == 'missing': weights.pop(next(iter(weights)))
            else: weights['unregistered'] = torch.tensor(1.)
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                self.check(state=dict(self.state, model=weights))

    def test_nonfinite_weights_and_logit_rejected(self):
        weights = dict(self.state['model']); key = A.AFFINE[0]
        weights[key] = weights[key].clone(); weights[key][0] = float('nan')
        with self.assertRaises(ValueError): self.check(state=dict(self.state, model=weights))
        with self.assertRaises(ValueError): self.check(state=dict(self.state, logit_scale=torch.tensor(float('nan'))))

    def test_affine_requires_raw_float32(self):
        weights = dict(self.state['model']); weights[A.AFFINE[0]] = weights[A.AFFINE[0]].double()
        with self.assertRaises(ValueError): self.check(state=dict(self.state, model=weights))

    def test_cache_role_normalization_and_model_SHA_are_bound(self):
        for key, value in [('role', 'full'), ('model_sha256', '2' * 64), ('normalization', 'unit norm')]:
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(cache=dict(self.cache, **{key: value}))

    def test_existing_cache_requires_complete_array_order_and_int64(self):
        for ids in (torch.arange(199), torch.arange(200).flip(0), torch.arange(200).float()):
            with self.assertRaises(ValueError): self.check(cache=dict(self.cache, array_rows=ids))

    def test_cache_requires_native_float32_full_finite_tensor(self):
        nonfinite = self.cache['condition'].clone(); nonfinite[0, 0] = float('nan')
        for tensor in (torch.zeros(199, 1024), torch.zeros(200, 1024, dtype=torch.float16), nonfinite):
            with self.assertRaises(ValueError): self.check(cache=dict(self.cache, condition=tensor))

    def test_file_identity_detects_wrong_hash_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic'; path.write_bytes(b'CPU fixture')
            before = path.read_bytes(); result = A.file_identity(path, A.sha(path))
            self.assertEqual(result['bytes'], len(before))
            with self.assertRaises(ValueError): A.file_identity(path, '0' * 64)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
