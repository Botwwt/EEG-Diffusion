"""Byte-identity and cap-preservation checks for the meta-loader observer."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
import run_BrainHIVE_frozen_function_low_memory_20261025_v3 as R


class LoaderAuditChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.audit = R.ConvertedWeightAudit(torch, self.directory)

    def tearDown(self):
        self.temporary.cleanup()

    def invoke(self, model, weights, loader):
        from unittest.mock import patch
        with patch.object(R, 'memory', return_value={'VmRSS_bytes': 100, 'VmHWM_bytes': 200}):
            return self.audit.inspect(model, weights, loader, (), {'dtype': torch.float16})

    def loader(self, model, weights, **kwargs):
        converted = {k: value.to(kwargs['dtype']) for k, value in weights.items()}
        model.load_state_dict(converted, strict=True, assign=True)
        return []

    def model(self):
        with torch.device('meta'):
            return torch.nn.Linear(3, 2)

    def weights(self):
        return {'weight': torch.tensor([[1.001, 2.002, 3.003], [4.004, 5.005, 6.006]]),
                'bias': torch.tensor([.125, -.25])}

    def test_exact_native_target_dtype_is_audited(self):
        model, weights = self.model(), self.weights()
        self.invoke(model, weights, self.loader)
        self.assertEqual(len(self.audit.records), 1)
        row = self.audit.records[0]
        self.assertTrue(row['all_tensors_exact'])
        self.assertEqual(row['expected_converted_state_sha256'], row['actual_loaded_state_sha256'])
        self.assertEqual(model.weight.dtype, torch.float16)
        self.assertTrue(torch.equal(weights['weight'], self.weights()['weight']))

    def test_non_meta_initialization_rejected(self):
        with self.assertRaises(AssertionError):
            self.invoke(torch.nn.Linear(3, 2), self.weights(), self.loader)

    def test_one_changed_weight_rejected(self):
        def changed(model, weights, **kwargs):
            self.loader(model, weights, **kwargs)
            with torch.no_grad(): model.weight[0, 0] += 1
            return []
        with self.assertRaises(AssertionError):
            self.invoke(self.model(), self.weights(), changed)
        self.assertEqual(self.audit.records, [])

    def test_unexpected_or_missing_keys_rejected(self):
        with self.assertRaises(AssertionError):
            self.invoke(self.model(), self.weights(), lambda *a, **k: ['unexpected'])
        def wrong(model, weights, **kwargs):
            self.loader(model, weights, **kwargs)
            model.register_buffer('unbound', torch.zeros(1))
            return []
        with self.assertRaises(AssertionError): self.invoke(self.model(), self.weights(), wrong)

    def test_incomplete_component_proof_cannot_close(self):
        with self.assertRaises(AssertionError): self.audit.complete()

    def test_four_component_coverage_is_exact(self):
        self.audit.records = [{'component': name} for name in
            ('UNet2DConditionModel', 'AutoencoderKL', 'CLIPTextModel', 'CLIPTextModelWithProjection')]
        self.assertEqual(self.audit.complete()['status'], 'ALL_FOUR_NATIVE_COMPONENTS_EXACT_CONVERTED_TENSOR_LOAD_VERIFIED')
        self.audit.records.append({'component': 'UNet2DConditionModel'})
        with self.assertRaises(AssertionError): self.audit.complete()

    def test_input_ref_hash_is_binding(self):
        path = self.directory / 'value.json'
        path.write_text('{"value": 1}')
        binding = R.ref(path)
        self.assertEqual(R.bound(binding), {'value': 1})
        path.write_text('{"value": 2}')
        with self.assertRaises(AssertionError): R.bound(binding)


if __name__ == '__main__':
    unittest.main(verbosity=2)
