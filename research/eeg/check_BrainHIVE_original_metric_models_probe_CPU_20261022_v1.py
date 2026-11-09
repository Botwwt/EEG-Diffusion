"""Small strict-load probe contracts only; imports no Torch and starts no child."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import probe_BrainHIVE_original_metric_models_CPU_20261022_v1 as P


def valid_model(stage):
    value = dict(status='ORIGINAL_METRIC_MODEL_CPU_STRICT_LOAD_COMPLETE', stage=stage,
        parameters=P.PARAMETERS[stage], strict_original_load=True,
        all_floating_parameters_and_buffers_finite=True, all_parameter_and_buffer_devices=['cpu'],
        GPU_initialized=False, images_read=0, forwards=0, scores_read=0)
    if stage == 'SwAV':
        value.update(missing_unused_head=list(P.SWAV_MISSING), unused_pretraining_heads=list(P.SWAV_UNUSED))
    if stage == 'CLIP_L14':
        value.update(source_commit=P.CLIP_COMMIT, visual_resolution=224, visual_layers=24,
            patch_kernel=[14, 14], embedding_dimensions=768)
    return value


class ProbeContracts(unittest.TestCase):
    def setUp(self):
        self.receipt = json.loads(P.ASSET_RECEIPT.read_text())

    def test_actual_closed_five_asset_schema_and_source_identity(self):
        self.assertEqual(hashlib.sha256(P.ASSET_RECEIPT.read_bytes()).hexdigest(), P.ASSET_RECEIPT_SHA)
        self.assertEqual(P.A.C.sha(P.A.__file__), P.RESTORER_SHA)
        rows = P.payload_rows(self.receipt)
        self.assertEqual(set(rows), set(P.STAGES))
        self.assertEqual(len(rows), 5)
        self.assertNotIn('torch', sys.modules)

    def test_bad_readiness_status_and_duplicate_missing_assets_rejected(self):
        bad = copy.deepcopy(self.receipt); bad['status'] = 'RUNNING'
        with self.assertRaises(AssertionError):
            P.payload_rows(bad)
        bad = copy.deepcopy(self.receipt); bad['assets'][-1] = copy.deepcopy(bad['assets'][0])
        with self.assertRaises(AssertionError):
            P.payload_rows(bad)

    def test_changed_sha_or_durable_path_rejected(self):
        for key, value in [('sha256', 'f' * 64), ('path', '/tmp/unrelated.pth')]:
            bad = copy.deepcopy(self.receipt); bad['assets'][0]['persistent_asset'][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                P.payload_rows(bad)

    def test_all_five_exact_cpu_load_records_accepted(self):
        for stage in P.STAGES:
            P.validate_model_record(valid_model(stage), stage)

    def test_gpu_init_forward_real_image_score_or_wrong_parameter_count_rejected(self):
        for key, value in [('GPU_initialized', True), ('images_read', 1), ('forwards', 1),
                ('scores_read', 1), ('parameters', 1), ('all_parameter_and_buffer_devices', ['cuda'])]:
            bad = valid_model('AlexNet'); bad[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                P.validate_model_record(bad, 'AlexNet')

    def test_nonfinite_or_partial_load_rejected(self):
        for key in ['strict_original_load', 'all_floating_parameters_and_buffers_finite']:
            bad = valid_model('Inception'); bad[key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError):
                P.validate_model_record(bad, 'Inception')

    def test_swav_exact_missing_and_original_unused_heads_only(self):
        P.swav_coverage(P.SWAV_MISSING, P.SWAV_UNUSED)
        for missing, unused in [(P.SWAV_MISSING + ('layer1.0.conv1.weight',), P.SWAV_UNUSED),
                (P.SWAV_MISSING, P.SWAV_UNUSED + ('projection_head.evil.weight',)),
                (P.SWAV_MISSING, P.SWAV_UNUSED[:-1]),
                (('fc.weight', 'fc.weight'), P.SWAV_UNUSED)]:
            with self.subTest(missing=missing, unused=unused), self.assertRaises(AssertionError):
                P.swav_coverage(missing, unused)

    def test_clip_original_architecture_required(self):
        for key, value in [('visual_resolution', 336), ('visual_layers', 12), ('patch_kernel', [16, 16]),
                ('embedding_dimensions', 512), ('source_commit', 'other')]:
            bad = valid_model('CLIP_L14'); bad[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                P.validate_model_record(bad, 'CLIP_L14')

    def test_sampled_actual_headroom_and_preflight_margin(self):
        with patch.object(P.A.C, 'resource_headroom', return_value=dict(effective_headroom_bytes=P.MIN_MEMORY)):
            P.memory()
            with self.assertRaises(AssertionError):
                P.memory(1)
        with patch.object(P.A.C, 'resource_headroom', return_value=dict(effective_headroom_bytes=P.MIN_MEMORY - 1)):
            with self.assertRaises(AssertionError):
                P.memory()

    def test_metadata_cap_keeps_partial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / 'fixture.log'; path.write_bytes(b'x' * (P.MAX_BYTES + 1))
            with self.assertRaises(AssertionError):
                P.output_bytes(root)
            self.assertEqual(path.stat().st_size, P.MAX_BYTES + 1)

    def test_owned_worker_stops_and_is_waited(self):
        process = Mock(); process.poll.return_value = None; process.wait.return_value = -15
        self.assertEqual(P.finish_child(process), -15)
        process.terminate.assert_called_once(); process.wait.assert_called_once_with(timeout=10)
        process.kill.assert_not_called()

    def test_owned_worker_kill_only_after_bounded_terminate_wait(self):
        process = Mock(); process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired(['fixture'], 10), -9]
        self.assertEqual(P.finish_child(process), -9)
        process.terminate.assert_called_once(); process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)

    def test_broken_failure_memory_snapshot_cannot_skip_owned_worker_cleanup(self):
        process = Mock(); process.poll.return_value = None; process.wait.return_value = -15
        status = {}
        with patch.object(P.A.C, 'resource_headroom', side_effect=OSError('fixture proc unavailable')):
            P.cleanup_after_failure(status, process)
        process.terminate.assert_called_once(); process.wait.assert_called_once_with(timeout=10)
        self.assertTrue(status['failure_owned_child_waited'])
        self.assertEqual(status['failure_resource_snapshot_error']['type'], 'OSError')


if __name__ == '__main__':
    unittest.main(verbosity=2)
