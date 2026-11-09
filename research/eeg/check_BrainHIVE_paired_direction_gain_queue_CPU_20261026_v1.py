"""Boundary regressions for the independent frozen-probe waiting controller."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import queue_BrainHIVE_paired_direction_gain_20261026_v1 as A


def study_fixture():
    return dict(registration_path='/fixture/study.json', method={}, core_sources={}, input_bindings={}, authorization={},
        training_job_names={arm: [arm + '_train0', arm + '_train1'] for arm in A.C.ARMS},
        generation_job_names={arm: dict(T32=[arm + '_T32'], V=[arm + '_V0', arm + '_V1']) for arm in A.C.ARMS},
        functional_job_name='functional', live_O_takeover=dict(path='/fixture/takeover', sha256='c' * 64),
        disk_buffer_bytes=8 << 30)


def document_fixture():
    required = [Path(A.__file__).name, 'check_BrainHIVE_paired_direction_gain_queue_CPU_20261026_v1.py',
        A.ENTRY, 'run_gpu_job_v2.py', 'brainhive_paired_supervision_20261026_v3.py',
        'queue_BrainHIVE_decisive_T_analysis_20261025_v2.py', 'brainhive_paired_study_contract_20261025_v1.py']
    return dict(schema='BrainHIVE_paired_direction_gain_T80_20261026_v1',
        status='REGISTERED_SUPPLEMENTARY_T_ONLY_FROZEN_DIRECTION_GAIN_PROBE',
        seed=1729, batch_size=2, maximum_wait_seconds=18 * 86400, poll_seconds=60,
        metadata_directory=str(A.METADATA), metadata_maximum_bytes=64 << 20,
        output_directory=str(A.C.DATA / 'scientific_basis_probe_20261026_v1'), maximum_output_bytes=512 << 20,
        maximum_seconds=7200, maximum_extra_CPU_bytes=10 << 30, maximum_GPU_bytes=20 << 30,
        minimum_CPU_headroom_bytes=10 << 30, source_sha256={name: 'c' * 64 for name in required},
        formal_queue_completion=dict(state_path=str(A.ROOT / 'audit/paired_effect_study_20261025_v1/reaudit_v3/queue_state.json'), expected_status=A.SUCCESS),
        formal_launch_receipt=dict(path=str(A.ROOT / 'audit/paired_effect_study_20261025_v1/reaudit_v3/actual_launch_receipt.json'), sha256='d' * 64),
        original_study_registration=dict(path='/fixture/study.json', sha256='a' * 64),
        resource_authorization=dict(path='/fixture/auth.json', sha256='b' * 64), gpu_job_name='one_probe',
        query_selection=dict(path='/fixture/T80.json', sha256='e' * 64))


def closure_fixture(functional=True):
    study = study_fixture(); document = document_fixture()
    launch = dict(pid=136961, registration=document['original_study_registration'], command=['/fixture/python', 'queue'])
    specs = A.completion_specs(study)
    keys = [(arm, stage) for arm in A.C.ARMS for stage in ('train', 'postfit', 'generate', 'audit', 'features', 'score')]
    keys += ([(None, 'functional')] if functional else []) + [(None, 'compare')]
    completed = [dict(arm=arm, stage=stage, completion=dict(path=specs[(arm, stage)]['completion'], sha256='f' * 64)) for arm, stage in keys]
    state = dict(status=A.SUCCESS, pid=136961, registration=launch['registration'], study_core_sha256=A.C.core(study),
        current_stage=None, child_pid=None, completed=completed)
    if not functional: state['functional_probe'] = 'FAILED_PRESERVED_UNKNOWN_NO_AUTOMATIC_RETRY'
    statuses = {v['completion']: v['expected_status'] for v in specs.values()}
    return state, document, study, launch, statuses


class QueueChecks(unittest.TestCase):
    def test_fixed_limits_and_probe_output_path(self):
        document = document_fixture()
        A.validate_queue(document, check_files=False)
        for field, wrong in [('maximum_wait_seconds', 18 * 86400 + 1), ('poll_seconds', 61),
                             ('maximum_seconds', 7201), ('metadata_maximum_bytes', (64 << 20) + 1),
                             ('maximum_output_bytes', (512 << 20) + 1)]:
            changed = copy.deepcopy(document); changed[field] = wrong
            with self.assertRaises(AssertionError): A.validate_queue(changed, check_files=False)
        changed = copy.deepcopy(document); changed['output_directory'] = str(A.C.DATA / 'wrong')
        with self.assertRaises(AssertionError): A.validate_queue(changed, check_files=False)

    def test_queue_probe_common_schema_and_output_interoperate(self):
        import diagnose_BrainHIVE_paired_direction_gain_T80_20261026_v1 as probe
        document = document_fixture()
        document['source_sha256'].update({name: 'c' * 64 for name in probe.SOURCE_NAMES})
        study = study_fixture()
        study['functional_queries'] = document['query_selection']
        study['resource_authorization'] = document['resource_authorization']
        def reader(reference):
            return study if reference == document['original_study_registration'] else {}
        with mock.patch.object(probe.C, 'sha', return_value='c' * 64), \
             mock.patch.object(probe.C, 'bound', side_effect=reader), \
             mock.patch.object(probe.C, 'validate', return_value='core'):
            self.assertEqual(probe.validate_addon(document), study)
        A.validate_queue(document, check_files=False)

    def test_missing_required_source_and_changed_source_refuse(self):
        document = document_fixture(); del document['source_sha256'][A.ENTRY]
        with self.assertRaises(AssertionError): A.validate_queue(document, check_files=False)
        with mock.patch.object(A, 'sha', return_value='wrong'):
            with self.assertRaises(AssertionError): A.validate_queue(document_fixture(), check_files=True)

    def test_status_prefix_does_not_parse_or_read_score_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'summary.json'
            path.write_bytes(b'{\n  "status": "DONE",\n' + b'NOT_JSON_AND_NOT_A_SCORE_TO_PARSE')
            self.assertEqual(A.status_only(path), 'DONE')
            path.write_bytes(b'{\n  "scores": [1,2,3],\n  "status": "DONE"\n}')
            with self.assertRaises(AssertionError): A.status_only(path)

    def test_success_waits_for_original_controller_exit(self):
        state, document, study, launch, statuses = closure_fixture()
        self.assertIsNone(A.terminal(state, document, study, launch, is_live=lambda pid: True))
        result = A.terminal(state, document, study, launch, is_live=lambda pid: False,
            status_reader=lambda path: statuses[path], hash_reader=lambda path: 'f' * 64)
        self.assertEqual(len(result['completed_status_refs']), 26)
        self.assertFalse(result['V_score_values_read'])

    def test_prior_functional_unknown_is_accepted_only_as_registered(self):
        state, document, study, launch, statuses = closure_fixture(functional=False)
        result = A.terminal(state, document, study, launch, is_live=lambda pid: False,
            status_reader=lambda path: statuses[path], hash_reader=lambda path: 'f' * 64)
        self.assertTrue(result['old_functional_probe_unknown'])
        del state['functional_probe']
        with self.assertRaises(AssertionError):
            A.terminal(state, document, study, launch, is_live=lambda pid: False)

    def test_failed_dead_pending_or_missing_stage_never_launches(self):
        for bad in ('FAILED', 'STOP_REQUESTED', 'UNEXPECTED_STATE', 'DEAD_PENDING', 'MISSING_STAGE'):
            state, document, study, launch, statuses = closure_fixture()
            if bad == 'MISSING_STAGE': state['completed'].pop(3)
            else: state['status'] = 'PAIRED_EFFECT_FOUR_ARM_LIFECYCLES_RUNNING' if bad == 'DEAD_PENDING' else bad
            with self.assertRaises(AssertionError):
                A.terminal(state, document, study, launch, is_live=lambda pid: False,
                    status_reader=lambda path: statuses[path], hash_reader=lambda path: 'f' * 64)

    def test_unexpected_controller_pid_and_identity_reuse_refuse(self):
        state, _, _, launch, _ = closure_fixture()
        state['status'] = 'PAIRED_EFFECT_FOUR_ARM_LIFECYCLES_RUNNING'
        expected = dict(pid=136961, pgid=136961, sid=136961, start_ticks=123, argv=launch['command'])
        self.assertEqual(A.process_identity(state, launch, {}, inspect=lambda pid: expected,
            is_live=lambda pid: True), expected)
        changed = expected | dict(start_ticks=456)
        with self.assertRaises(AssertionError):
            A.process_identity(state, launch, {}, inspect=lambda pid: changed,
                is_live=lambda pid: True, previous=expected)
        state['pid'] += 1
        with self.assertRaises(AssertionError): A.process_identity(state, launch, {}, is_live=lambda pid: False)

    def test_unexpected_current_child_refuses(self):
        state, _, _, launch, _ = closure_fixture()
        state.update(child_pid=999, current_stage='unknown', attempts=[dict(name='unknown')])
        with self.assertRaises(AssertionError): A.process_identity(state, launch, {}, is_live=lambda pid: False)

    def test_raw_child_exit_codes_map_to_wrapper_zero_or_one(self):
        document = document_fixture(); child = ['/fixture/python', 'probe']
        for raw in (0, 1, 7, -15, -9):
            record = dict(wrapper_pid=100, pid=101, name='one_probe', command=child,
                authorization_sha256='b' * 64, max_seconds=7200, exit_code=raw,
                status='COMPLETED' if raw == 0 else 'FAILED_OR_JOB_TIMEOUT_RESUMABLE')
            result = A.validate_closed_job(record, 100, 0 if raw == 0 else 1, child, document, is_live=lambda pid: False)
            self.assertEqual(result['child_exit_code'], raw)
        with self.assertRaises(AssertionError):
            A.validate_closed_job(record, 100, -9, child, document, is_live=lambda pid: False)

    def test_supplemental_reserve_does_not_consume_original_reserve(self):
        document = document_fixture(); study = study_fixture()
        original = dict(reserved_remaining_bytes=96 << 30, free_bytes=(104 << 30) + (576 << 20), memory={})
        with mock.patch.object(A.C, 'resources', return_value=original), \
             mock.patch.object(A, 'safe_bytes_used', return_value=0), \
             mock.patch.object(A.shutil, 'disk_usage', return_value=SimpleNamespace(free=128 << 20)):
            result = A.resources(document, study)
            self.assertEqual(result['supplemental_reserved_bytes'], 576 << 20)
            original['free_bytes'] -= 1
            with self.assertRaises(AssertionError): A.resources(document, study)

    def test_delayed_receipt_binds_during_three_second_registration_grace(self):
        clock = [0.]; process = SimpleNamespace(pid=100, poll=lambda: None)
        record = dict(wrapper_pid=100, pid=101, name='probe', command=['probe'], status='RUNNING')
        adapter = mock.Mock(return_value='bound_identity')
        result = A.bind_owned_after_registration(process, Path('/fixture/job'), 'probe', ['probe'],
            exists=lambda path: clock[0] >= .2, reader=lambda path: record, adapter=adapter,
            is_live=lambda pid: True, monotonic=lambda: clock[0], sleep=lambda delay: clock.__setitem__(0, clock[0] + delay))
        self.assertEqual(result, 'bound_identity')
        self.assertGreaterEqual(clock[0], .2)
        self.assertLess(clock[0], 3)
        adapter.assert_called_once()

    def test_missing_receipt_stops_grace_without_signals(self):
        clock = [0.]; process = SimpleNamespace(pid=100, poll=lambda: None)
        result = A.bind_owned_after_registration(process, Path('/fixture/job'), 'probe', ['probe'],
            exists=lambda path: False, monotonic=lambda: clock[0], sleep=lambda delay: clock.__setitem__(0, clock[0] + delay))
        self.assertIsNone(result)
        self.assertAlmostEqual(clock[0], 3)
        stop = mock.Mock(); evidence = {}
        self.assertFalse(A.cleanup_owned(process, None, Path('/fixture/job'), 'probe', ['probe'], evidence,
            binder=lambda *args: None, exists=lambda path: False, stop_owned=stop))
        self.assertFalse(evidence['signals_sent']); stop.assert_not_called()

    def test_stable_identity_is_required_before_owned_cleanup(self):
        process = SimpleNamespace(pid=100, poll=lambda: None)
        owned = mock.Mock(); stop = mock.Mock(return_value=True); evidence = {}
        self.assertTrue(A.cleanup_owned(process, None, 'job', 'probe', ['probe'], evidence,
            binder=lambda *args: owned, stop_owned=stop))
        owned.validate_owned_identity_for_cleanup.assert_called_once()
        stop.assert_called_once_with(owned, True, 'job', ['probe'], 'probe', evidence)
        owned.validate_owned_identity_for_cleanup.side_effect = AssertionError('PID reused')
        with self.assertRaises(AssertionError):
            A.cleanup_owned(process, owned, 'job', 'probe', ['probe'], {}, stop_owned=stop)
        self.assertEqual(stop.call_count, 1)

    def test_already_closed_dead_job_cleanup_needs_no_signals(self):
        process = SimpleNamespace(pid=100, poll=lambda: 1)
        record = dict(wrapper_pid=100, pid=101, name='probe', command=['probe'], status='FAILED_OR_JOB_TIMEOUT_RESUMABLE')
        stop = mock.Mock(); evidence = {}
        self.assertTrue(A.cleanup_owned(process, None, 'job', 'probe', ['probe'], evidence,
            binder=lambda *args: None, exists=lambda path: True, reader=lambda path: record,
            is_live=lambda pid: False, stop_owned=stop))
        self.assertFalse(evidence['signals_sent']); stop.assert_not_called()


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(QueueChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise SystemExit(1)
    print(json.dumps(dict(status='DIRECTION_GAIN_ADDON_QUEUE_CPU_REGRESSIONS_PASSED', checks=result.testsRun,
        GPU_used=False, original_pipeline_modified=False, real_study_processes_touched=False)))
