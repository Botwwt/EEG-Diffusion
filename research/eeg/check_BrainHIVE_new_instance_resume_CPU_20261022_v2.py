"""Tiny CPU producer/observer fixtures for the replacement-instance guardian.

No SSH, GPU process, model, dataset, runtime aliases or experiment state are
used. Fake Popen handles deliberately model live and terminal observations;
they must never be interpreted as actual runtime or formal-training evidence.
"""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import queue_BrainHIVE_new_instance_resume_20261022_v2 as Q


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class Fixture:
    def __init__(self, root):
        self.root = root
        self.base = root / 'audit/new_instance_restore_20261022_v2'
        self.base.mkdir(parents=True)
        self.job = root / 'audit/latent_job.json'
        self.state = dict(attempts=[], source_sha256={}, formal_training_started=False, latent_wrapper_pid=777777777)

    def drivers(self, names=('public', 'inputs', 'conditions', 'latent', 'evaluation')):
        for name in names:
            write(self.base / (name + '_driver.json'), dict(status='CPU_SYNTHETIC_ONLY'))


class GuardianTests(unittest.TestCase):
    def context(self, stack, fixture):
        for name, value in [('ROOT', fixture.root), ('BASE', fixture.base), ('OUT', fixture.base / 'state.json')]:
            stack.enter_context(patch.object(Q, name, value))

    def test_driver_closure_before_gpu_wrapper_terminal_is_a_wait_not_failure(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            f = Fixture(Path(directory)); f.drivers(); self.context(stack, f)
            write(f.job, dict(status='RUNNING', pid=111111111, exit_code=None))
            sleeps = []
            def sleep(seconds):
                sleeps.append(seconds)
                write(f.job, dict(status='COMPLETED', pid=111111111, exit_code=0))
            stack.enter_context(patch.object(Q.time, 'sleep', sleep))
            stack.enter_context(patch.object(Q, 'live', lambda pid: pid == 111111111 and not sleeps))
            Q.wait_restoration(f.state, f.job, 222222222, 333333333)
            self.assertGreaterEqual(len(sleeps), 1)

    def test_transient_nonatomic_job_receipt_does_not_trigger_restart_or_failure(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            f = Fixture(Path(directory)); f.drivers(); self.context(stack, f)
            write(f.job, dict(status='COMPLETED', pid=111111111, exit_code=0))
            actual_read = Q.read; calls = []
            def read(path):
                if Path(path) == f.job and not calls:
                    calls.append('partial-write')
                    raise json.JSONDecodeError('CPU synthetic partial write', '{', 1)
                return actual_read(path)
            stack.enter_context(patch.object(Q, 'read', read))
            slept = []
            stack.enter_context(patch.object(Q.time, 'sleep', lambda seconds: slept.append(seconds)))
            stack.enter_context(patch.object(Q, 'live', lambda pid: pid == f.state['latent_wrapper_pid'] and not slept))
            Q.wait_restoration(f.state, f.job, 222222222, 333333333)
            self.assertEqual(calls, ['partial-write'])

    def test_completed_latent_receipt_still_waits_for_exact_wrapper_handle_exit(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            f = Fixture(Path(directory)); f.drivers(); self.context(stack, f)
            write(f.job, dict(status='COMPLETED', pid=111111111, exit_code=0))
            slept = []
            stack.enter_context(patch.object(Q.time, 'sleep', lambda seconds: slept.append(seconds)))
            stack.enter_context(patch.object(Q, 'live', lambda pid: pid == f.state['latent_wrapper_pid'] and not slept))
            Q.wait_restoration(f.state, f.job, 222222222, 333333333)
            self.assertEqual(len(slept), 1)

    def test_failed_latent_job_refuses_all_downstream_stages(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            f = Fixture(Path(directory)); f.drivers(); self.context(stack, f)
            write(f.job, dict(status='FAILED_OR_JOB_TIMEOUT_RESUMABLE', pid=111111111, exit_code=1))
            stack.enter_context(patch.object(Q, 'live', lambda pid: False))
            with self.assertRaises(AssertionError):
                Q.wait_restoration(f.state, f.job, 222222222, 333333333)
            self.assertFalse(f.state['formal_training_started'])
            self.assertEqual(f.state['attempts'], [])

    def test_missing_public_driver_and_dead_producer_is_not_a_wait(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            f = Fixture(Path(directory)); f.drivers(('inputs', 'conditions')); self.context(stack, f)
            write(f.job, dict(status='RUNNING', pid=111111111, exit_code=None))
            stack.enter_context(patch.object(Q, 'live', lambda pid: False))
            with self.assertRaises(AssertionError):
                Q.wait_restoration(f.state, f.job, 222222222, 333333333)

    def test_all_observed_downstream_stages_use_existing_exact_clis_and_order(self):
        tree = ast.parse(Path(Q.__file__).read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == 'run']
        calls.sort(key=lambda node: node.lineno)
        labels = [node.args[1].value for node in calls]
        self.assertEqual(labels, ['exact_runtime_closure', 'evaluation_GPU_probe', 'prepare_complete_recipes',
            'new_core_four_GPU_profiles', 'formal_admission', 'formal_full_four_arm_lifecycle'])
        entries = [node.args[2].value for node in calls]
        self.assertEqual(entries, ['verify_BrainHIVE_new_instance_runtime_closure_CPU_20261022_v2.py',
            'probe_BrainHIVE_evaluation_environment_GPU_20261022_v2.py',
            'prepare_BrainHIVE_functional_lifecycle_CPU_20261022_v1.py',
            'queue_BrainHIVE_functional_lifecycle_20261022_v1.py',
            'prepare_BrainHIVE_functional_lifecycle_CPU_20261022_v1.py',
            'queue_BrainHIVE_functional_lifecycle_20261022_v1.py'])
        self.assertEqual(ast.literal_eval(calls[0].args[3]), ['--evaluation', '--attempt', '1'])
        self.assertEqual(ast.literal_eval(calls[2].args[3]), ['prepare'])
        self.assertEqual(ast.literal_eval(calls[3].args[3]), ['--mode', 'profiles'])
        self.assertEqual(ast.literal_eval(calls[-1].args[3]), ['--mode', 'formal'])
        for entry in set(entries):
            self.assertTrue((Path(Q.__file__).parent / entry).is_file())
        # Formal mode cannot be reached directly by an earlier branch/child.
        self.assertLess(calls[4].lineno, calls[5].lineno)
        self.assertEqual(ast.literal_eval(calls[1].keywords[0].value), 'BrainHIVE_evaluation_environment_probe_20261022_v2_attempt1')

    def test_gpu_probe_command_uses_exclusive_wrapper_and_exact_registered_receipt(self):
        from contextlib import ExitStack
        class TerminalHandle:
            pid = 444444444
            returncode = 0
            def poll(self): return 0
            def wait(self, timeout): return 0
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            f = Fixture(Path(directory)); self.context(stack, f)
            entry = 'SYNTHETIC_SOURCE_NOT_EXECUTED.py'; (f.root / entry).write_text('CPU fixture only')
            f.state['source_sha256'][entry] = Q.sha(f.root / entry)
            name = 'BrainHIVE_evaluation_environment_probe_20261022_v2_attempt1'
            job_path = f.root / 'audit' / (name + '_job.json')
            commands = []
            def popen(argv, **kwargs):
                commands.append(argv)
                write(job_path, dict(status='COMPLETED', exit_code=0, pid=555555555, wrapper_pid=444444444))
                return TerminalHandle()
            stack.enter_context(patch.object(Q.subprocess, 'Popen', popen))
            stack.enter_context(patch.object(Q, 'live', lambda pid: False))
            Q.run(f.state, 'evaluation_GPU_probe', entry, ['--attempt', '1'], 900, GPU_name=name)
            argv = commands[0]
            self.assertEqual(argv[2], str(f.root / 'run_gpu_job_v2.py'))
            self.assertEqual(argv[3:7], ['--name', name, '--max-seconds', '900'])
            self.assertIn('--', argv)
            self.assertEqual(argv[-2:], ['--attempt', '1'])
            self.assertFalse(f.state['formal_training_started'])

    def test_nested_controller_observation_failure_preserves_live_handle_and_does_not_kill_it(self):
        from contextlib import ExitStack
        class LiveHandle:
            pid = 444444444
            returncode = None
            def poll(self): return None
            def wait(self, timeout):
                raise Q.subprocess.TimeoutExpired('CPU synthetic live nested controller', timeout)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            f = Fixture(Path(directory)); self.context(stack, f)
            entry = 'queue_BrainHIVE_functional_lifecycle_20261022_v1.py'
            (f.root / entry).write_text('CPU fixture only; not executable')
            f.state['source_sha256'][entry] = Q.sha(f.root / entry)
            stack.enter_context(patch.object(Q.subprocess, 'Popen', lambda *args, **kwargs: LiveHandle()))
            clock = iter([0., 2., 2., 2., 2.])
            stack.enter_context(patch.object(Q.time, 'monotonic', lambda: next(clock, 2.)))
            stack.enter_context(patch.object(Q.os, 'killpg', create=True,
                side_effect=AssertionError('Must not blindly kill nested controller groups')))
            with self.assertRaises((AssertionError, Q.subprocess.TimeoutExpired)):
                Q.run(f.state, 'formal_full_four_arm_lifecycle', entry, ['--mode', 'formal'], 1)
            self.assertTrue(f.state.get('unresolved_live_child'))
            self.assertNotEqual(f.state.get('status'), 'CONTINUATION_STOPPED_WITH_PRESERVED_EXPLICIT_FAILURE_NO_AUTOMATIC_RETRY')

    def test_missing_probe_job_metadata_still_waits_actual_live_gpu_wrapper(self):
        from contextlib import ExitStack
        class LiveHandle:
            pid = 444444444
            returncode = None
            waits = 0
            def poll(self): return None
            def wait(self, timeout):
                self.waits += 1
                raise Q.subprocess.TimeoutExpired('CPU synthetic GPU wrapper', timeout)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            f = Fixture(Path(directory)); self.context(stack, f)
            entry = 'SYNTHETIC_PROBE_SOURCE_NOT_EXECUTED.py'
            (f.root / entry).write_text('CPU fixture only')
            f.state['source_sha256'][entry] = Q.sha(f.root / entry)
            process = LiveHandle()
            stack.enter_context(patch.object(Q.subprocess, 'Popen', lambda *args, **kwargs: process))
            clock = iter([0., 100., 100.])
            stack.enter_context(patch.object(Q.time, 'monotonic', lambda: next(clock, 100.)))
            stack.enter_context(patch.object(Q.os, 'killpg', create=True,
                side_effect=AssertionError('No child PID has been verified')))
            with self.assertRaises(Q.subprocess.TimeoutExpired):
                Q.run(f.state, 'evaluation_GPU_probe', entry, ['--attempt', '1'], 1, GPU_name='SYNTHETIC_JOB_NOT_PUBLISHED')
            self.assertEqual(process.waits, 1)
            self.assertEqual(f.state['unresolved_live_child'], process.pid)


if __name__ == '__main__':
    unittest.main(verbosity=2)
