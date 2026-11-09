"""Synthetic CPU checks for the independent finite analysis queue.

No real registrations, training logs, D/V/test scores, server connection,
Torch import, model weights, GPU queries, or real process signals are used.
"""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import queue_BrainHIVE_decisive_T_analysis_20261025_v1 as Q


def fixture(base):
    roots = [base / 'payload', base / 'metadata']
    pins = {entry: 'a' * 64 for _, entry, _ in Q.STAGES}
    pins.update({Path(Q.__file__).name: 'b' * 64, 'run_gpu_job_v2.py': 'c' * 64,
        'brainhive_functional_training_contract_v2.py': 'd' * 64})
    stages = []
    for key, entry, gpu in Q.STAGES:
        config_path = str(base / (key + '_config.json'))
        output = roots[0] / key
        arguments = ['--config', config_path, '--output', str(output)] if key == 'layersCPU' else ['--registration', config_path]
        stages.append(dict(stage=key, entrypoint=entry, GPU=gpu, args=arguments,
            registration=dict(path=config_path, sha256='9' * 64),
            maximum_seconds=7200 if gpu else 600, job_name='synthetic_' + key,
            completion=dict(path=str(output / (key + '.json')), status='COMPLETE_' + key)))
    registration = dict(schema=Q.SCHEMA, source_sha256=pins,
        wrapper=dict(path=str(base / 'wrapper.py'), sha256='c' * 64),
        resource_helper=dict(path=str(base / 'helper.py'), sha256='d' * 64),
        authorization=dict(path=str(base / 'authorization.json'), sha256='e' * 64),
        upstream=dict(state_path=str(base / 'upstream.json'), lifecycle_lock=str(base / 'upstream.lock'),
            controller=dict(path=str(base / 'upstream.py'), sha256='f' * 64),
            recipes=[dict(arm=arm, path=str(base / (arm + '.json')), sha256='1' * 64,
                recipe_core_sha256=(str(index + 2) * 64)) for index, arm in enumerate(Q.ARMS)]),
        output_roots=[str(path) for path in roots], limits=dict(Q.LIMITS), stages=stages,
        auto_fit=False, new_candidate_training=False)
    records = {}
    completed = []
    for recipe in registration['upstream']['recipes']:
        arm = recipe['arm']; core = recipe['recipe_core_sha256']
        item = dict(arm=arm)
        for key, status in (
            ('final', 'FUNCTIONAL_ALIGNMENT_V2_ALL10_REAL_OOF_FIXED_FINAL_COMPONENT_COMPLETE'),
            ('generated', 'FUNCTIONAL_ALIGNMENT_V2_ALL16600_IMAGES_AND_TARGET_JOIN_COMPLETE'),
            ('evaluation', 'FUNCTIONAL_ALIGNMENT_V2_FULL16600_EVALUATION_CLOSED')):
            path = arm + '_' + key
            item[key] = dict(path=path, sha256='2' * 64)
            records[path] = dict(status=status, arm=arm, recipe_core_sha256=core, queries=16600,
                actual_full_V16600_evaluation_completed=True,
                full_identity_target_seed_noise_chain_verified=True)
        completed.append(item)
    records['comparison'] = dict(status=Q.COMPARISON, actual_full_V16600_comparison_completed=True)
    state = dict(status=Q.SUCCESS, pid=1, child_pid=None, current_stage=None, completed=completed,
        comparison=dict(path='comparison', sha256='3' * 64), attempts=[])
    return registration, state, records


class QueueContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.registration, self.state, self.records = fixture(self.base)

    def tearDown(self):
        self.temp.cleanup()

    def validate(self, value=None):
        return Q.validate_registration(value or self.registration, check_files=False, enforce_paths=False)

    def terminal(self, state=None, is_live=lambda pid: False):
        return Q.terminal_metadata(state or self.state, self.registration,
            reader=lambda item: self.records[item['path']], is_live=is_live)

    def test_valid_fixed_registration(self):
        self.assertIs(self.validate(), self.registration)

    def test_entrypoint_not_arbitrary(self):
        self.registration['stages'][0]['entrypoint'] = 'train_BrainHIVE_parent_v2.py'
        with self.assertRaises(AssertionError): self.validate()

    def test_fixed_stage_order(self):
        self.registration['stages'][0], self.registration['stages'][1] = self.registration['stages'][1], self.registration['stages'][0]
        with self.assertRaises(AssertionError): self.validate()

    def test_no_extra_stage(self):
        self.registration['stages'].append(copy.deepcopy(self.registration['stages'][0]))
        with self.assertRaises(AssertionError): self.validate()

    def test_no_gpu_reclassification(self):
        self.registration['stages'][0]['GPU'] = True
        with self.assertRaises(AssertionError): self.validate()

    def test_finite_gpu_timeout(self):
        self.registration['stages'][2]['maximum_seconds'] = float('inf')
        with self.assertRaises(AssertionError): self.validate()
        self.registration['stages'][2]['maximum_seconds'] = 7201
        with self.assertRaises(AssertionError): self.validate()

    def test_exact_CPU_stage_time_limits(self):
        self.registration['stages'][1]['maximum_seconds'] = 14400
        self.validate()
        self.registration['stages'][0]['maximum_seconds'] = 1801
        with self.assertRaises(AssertionError): self.validate()

    def test_output_escape_rejected(self):
        self.registration['stages'][0]['completion']['path'] = str(self.base / 'outside.json')
        with self.assertRaises(AssertionError): self.validate()

    def test_child_registration_argument_is_SHA_bound(self):
        self.registration['stages'][0]['args'][1] = str(self.base / 'unbound.json')
        with self.assertRaises(AssertionError): self.validate()

    def test_layers_output_argument_is_independent(self):
        self.registration['stages'][1]['args'][-1] = str(self.base / 'old_arm')
        with self.assertRaises(AssertionError): self.validate()

    def test_no_budget_lowering(self):
        self.registration['limits']['minimum_CPU_headroom_bytes'] = 1
        with self.assertRaises(AssertionError): self.validate()

    def test_no_automatic_fit(self):
        self.registration['auto_fit'] = True
        with self.assertRaises(AssertionError): self.validate()

    def test_required_source_pins(self):
        del self.registration['source_sha256']['run_gpu_job_v2.py']
        with self.assertRaises(AssertionError): self.validate()

    def test_running_S_not_completion(self):
        self.state['status'] = 'RUNNING_FIXED_FOUR_ARM_FULL_LIFECYCLE'
        self.assertIsNone(self.terminal())

    def test_success_written_before_controller_exit_still_waits(self):
        self.assertIsNone(self.terminal(is_live=lambda pid: pid == 1))

    def test_upstream_explicit_failure_stops(self):
        for status in ('RECOVERY_FAILED_PRESERVED_NO_FURTHER_AUTOMATIC_RETRY',
                'RECOVERY_STOP_REQUESTED_WRAPPER_STILL_LIVE'):
            self.state['status'] = status
            with self.assertRaises(RuntimeError): self.terminal()

    def test_success_complete_metadata_without_scores(self):
        proof = self.terminal()
        self.assertFalse(proof['score_values_accessed'])
        self.assertEqual(len(proof['closure_metadata_refs']), 12)
        self.assertEqual(proof['comparison'], self.state['comparison'])

    def test_S_fit_only_is_insufficient(self):
        del self.state['completed'][-1]['evaluation']
        with self.assertRaises(KeyError): self.terminal()

    def test_missing_arm_rejected(self):
        self.state['completed'].pop()
        with self.assertRaises(AssertionError): self.terminal()

    def test_wrong_identity_rejected(self):
        self.records['PN_evaluation']['recipe_core_sha256'] = '9' * 64
        with self.assertRaises(AssertionError): self.terminal()

    def test_incomplete_16600_rejected(self):
        self.records['S_generated']['queries'] = 16599
        with self.assertRaises(AssertionError): self.terminal()

    def test_comparison_not_complete_rejected(self):
        self.records['comparison']['status'] = 'FULL_V_COMPARISON_FAILED_NO_DECISION'
        with self.assertRaises(AssertionError): self.terminal()

    def test_live_gpu_child_rejected(self):
        self.records['old_job'] = dict(status='COMPLETED', pid=42, wrapper_pid=43)
        self.state['attempts'] = [dict(exit_code=0, job=dict(path='old_job', sha256='a' * 64))]
        with self.assertRaises(AssertionError): self.terminal(is_live=lambda pid: pid == 42)

    def test_unresolved_child_rejected(self):
        self.state['unresolved_live_child'] = True
        with self.assertRaises(AssertionError): self.terminal()

    def test_actual_sha_bound_reference(self):
        path = self.base / 'actual.json'; path.write_text('{"status":"OK"}', encoding='utf-8')
        item = Q.ref(path)
        self.assertEqual(Q.read_ref(item), dict(status='OK'))
        path.write_text('{"status":"CHANGED"}', encoding='utf-8')
        with self.assertRaises(AssertionError): Q.read_ref(item)

    def test_no_second_queue_or_retry(self):
        path = self.base / 'registration.json'; path.write_text('{}', encoding='utf-8')
        state_path = self.base / 'queue_state.json'
        with patch.object(Q, 'STATE', state_path):
            first = Q.Runner(self.registration, path)
            self.assertTrue(first.state['no_candidate_fit'])
            self.assertFalse(first.state['automatic_retry'])
            with self.assertRaises(AssertionError): Q.Runner(self.registration, path)

    def test_foreign_gpu_receipt_never_signalled(self):
        job = self.base / 'job.json'
        job.write_text(json.dumps(dict(wrapper_pid=999, name='x', command=['synthetic'], pid=42)), encoding='utf-8')
        class FakeProcess:
            pid = 111
            def poll(self): return None
        with patch.object(Q.os, 'killpg', create=True) as killed:
            with self.assertRaises(AssertionError): Q.stop_owned(FakeProcess(), True, job, ['synthetic'], 'x')
            killed.assert_not_called()

    def test_bounded_metadata_reader(self):
        path = self.base / 'large.json'; path.write_bytes(b' ' * 128)
        with patch.object(Q, 'JSON_CAP', 127):
            with self.assertRaises(AssertionError): Q.read(path)

    def test_distinct_disks_receive_own_output_allocation(self):
        payload, metadata = [Path(path) for path in self.registration['output_roots']]
        payload.mkdir(); metadata.mkdir()
        original_stat = Path.stat
        def fake_stat(path, *args, **kwargs):
            if path == payload: return SimpleNamespace(st_dev=1)
            if path == metadata: return SimpleNamespace(st_dev=2)
            return original_stat(path, *args, **kwargs)
        def disk(path):
            return SimpleNamespace(free=(2 << 30) if path == metadata else (10 << 30))
        with patch.object(Q, 'BASE', metadata), patch.object(Q, 'directory_bytes', return_value=0), \
                patch.object(Path, 'stat', autospec=True, side_effect=fake_stat), \
                patch.object(Q.shutil, 'disk_usage', side_effect=disk):
            # A2GiB root volume is sufficient for64MiB metadata +1GiB buffer;
            # the4.75GiB payload allocation belongs on the separate data disk.
            self.assertEqual(Q.budget(self.registration)['total_output_bytes'], 0)

    def test_metadata_buffer_not_lowered(self):
        payload, metadata = [Path(path) for path in self.registration['output_roots']]
        payload.mkdir(); metadata.mkdir()
        original_stat = Path.stat
        def fake_stat(path, *args, **kwargs):
            if path == payload: return SimpleNamespace(st_dev=1)
            if path == metadata: return SimpleNamespace(st_dev=2)
            return original_stat(path, *args, **kwargs)
        def disk(path):
            return SimpleNamespace(free=(1 << 30) if path == metadata else (10 << 30))
        with patch.object(Q, 'BASE', metadata), patch.object(Q, 'directory_bytes', return_value=0), \
                patch.object(Path, 'stat', autospec=True, side_effect=fake_stat), \
                patch.object(Q.shutil, 'disk_usage', side_effect=disk):
            with self.assertRaises(AssertionError): Q.budget(self.registration)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(QueueContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(status='SYNTHETIC_FINITE_ANALYSIS_QUEUE_CPU_CHECKS_PASSED' if result.wasSuccessful() else 'FAILED',
        checks=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        real_server_access=False, real_score_access=False, GPU_used=False, fitted_parameters=0)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
