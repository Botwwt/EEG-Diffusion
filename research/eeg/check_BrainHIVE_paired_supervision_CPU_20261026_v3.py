"""CPU regression checks; --real-linux adds isolated /proc adoption fixtures.

The Linux fixture writes its own temporary CPU-only run_gpu_job_v2.py. It never
executes the study's GPU wrapper, acquires GPU locks, or touches the real ledger.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch

import brainhive_paired_supervision_20261026_v3 as S


class Backend:
    ticks_per_second = 100

    def __init__(self, identities):
        self.identities = identities
        self.clock = 180.0

    def inspect(self, pid):
        return copy.deepcopy(self.identities.get(pid))

    def boottime(self):
        return self.clock

    def monotonic(self):
        return self.clock

    def sleep(self, seconds):
        self.clock += seconds


class Fixture:
    name = 'isolated_CPU_fixture'
    wrapper_pid = 300
    child_pid = 301

    def __init__(self, root):
        self.root = Path(root).resolve()
        (self.root / 'audit').mkdir()
        (self.root / 'configs').mkdir()
        self.authorization = self.root / 'configs/resource_authorization_20261020_v3.json'
        self.authorization.write_text('{}\n')
        self.command = [sys.executable, '-B', '-u', str(self.root / 'cpu_child.py')]
        self.record = dict(name=self.name, pid=self.child_pid, wrapper_pid=self.wrapper_pid,
            command=self.command, start_utc='2026-10-26T01:00:00Z', max_seconds=120.0,
            charged_seconds_before_job=3.0, total_hours_cap=None, occupied_time_deadline=None,
            authorization=str(self.authorization),
            authorization_sha256=hashlib.sha256(self.authorization.read_bytes()).hexdigest(), status='RUNNING')
        self.path = self.root / 'audit' / (self.name + '_job.json')
        self.write()
        self.backend = Backend({
            self.wrapper_pid: dict(pid=self.wrapper_pid, ppid=42, pgid=self.wrapper_pid,
                sid=self.wrapper_pid, start_ticks=10000, state='S',
                argv=[sys.executable, '-B', str(self.root / 'run_gpu_job_v2.py'), '--name', self.name,
                      '--max-seconds', '120', '--', *self.command]),
            self.child_pid: dict(pid=self.child_pid, ppid=self.wrapper_pid, pgid=self.child_pid,
                sid=self.child_pid, start_ticks=10001, state='S', argv=self.command)})

    def write(self):
        self.path.write_text(json.dumps(self.record))

    def adopt(self, **kwargs):
        return S.AdoptedProcess(self.path, self.name, self.command, self.wrapper_pid, self.child_pid,
                                _root=self.root, _backend=self.backend, **kwargs)

    def close(self, code=0):
        self.backend.identities.clear()
        self.record.update(status='COMPLETED' if code == 0 else 'FAILED_OR_JOB_TIMEOUT_RESUMABLE',
                           exit_code=code, charged_seconds=80.0, end_utc='2026-10-26T01:01:20Z')
        self.write()


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(self.temp.name)

    def test_regular_file_count_uses_one_stat_and_excludes_directories(self):
        root = self.fixture.root / 'count'
        root.mkdir()
        (root / 'a').write_bytes(b'123')
        (root / 'sub').mkdir()
        (root / 'sub' / 'b').write_bytes(b'4567')
        original = Path.stat
        calls = {}
        def observed(path, *args, **kwargs):
            calls[path] = calls.get(path, 0) + 1
            return original(path, *args, **kwargs)
        with patch.object(Path, 'stat', observed):
            self.assertEqual(S.safe_bytes_used(root), 7)
        self.assertTrue(all(calls[root / p] == 1 for p in ('a', 'sub', 'sub/b')))

    def test_real_pending_rename_between_enumeration_and_stat(self):
        root = self.fixture.root / 'race'
        root.mkdir()
        pending = root / 'resume.pt.pending'
        target = root / 'resume.pt'
        pending.write_bytes(b'kept-checkpoint')
        original = Path.stat
        renamed = []
        def observed(path, *args, **kwargs):
            if path == pending and not renamed:
                os.replace(pending, target)
                renamed.append(True)
            return original(path, *args, **kwargs)
        with patch.object(Path, 'stat', observed):
            counted = S.safe_bytes_used(root)
        self.assertTrue(renamed)
        self.assertIn(counted, (0, len(b'kept-checkpoint')))
        self.assertEqual(target.read_bytes(), b'kept-checkpoint')
        self.assertEqual(S.safe_bytes_used(root), len(b'kept-checkpoint'))

    def test_count_permission_errors_are_not_suppressed(self):
        with patch.object(Path, 'stat', side_effect=PermissionError('fixture denied')):
            with self.assertRaises(PermissionError):
                S.safe_bytes_used(self.fixture.root)

    def test_child_wrapper_zero_and_nonzero_mapping(self):
        for child, wrapper in ((0, 0), (-15, 1), (7, 1), (-9, 1)):
            self.assertEqual(S.validate_wrapper_exit(child, wrapper)['child_exit_code'], child)
        for child, wrapper in ((0, 1), (-15, -15), (7, 7), (7, 0), (True, 1)):
            with self.assertRaises(AssertionError):
                S.validate_wrapper_exit(child, wrapper)

    def test_active_adapter_retains_original_age_and_identity(self):
        adopted = self.fixture.adopt()
        self.assertIsNone(adopted.poll())
        self.assertEqual(adopted.elapsed_seconds, 80)
        self.fixture.backend.clock += 7
        self.assertEqual(adopted.elapsed_seconds, 87)
        self.assertTrue(adopted.validate_owned_identity_for_cleanup()['identity_verified'])
        self.assertTrue(adopted.returncode_inferred)

    def test_valid_closure_preserves_raw_negative_and_positive_child_exits(self):
        for code in (-15, 7, 0):
            with tempfile.TemporaryDirectory() as root:
                fixture = Fixture(root)
                adopted = fixture.adopt()
                fixture.close(code)
                expected = 0 if code == 0 else 1
                self.assertEqual(adopted.poll(), expected)
                self.assertEqual(adopted.child_returncode, code)
                self.assertTrue(adopted.closure_valid)
                self.assertEqual(adopted.poll(), expected)

    def test_closed_receipt_does_not_finish_a_still_live_wrapper(self):
        adopted = self.fixture.adopt()
        live = copy.deepcopy(self.fixture.backend.identities)
        self.fixture.close(0)
        self.fixture.backend.identities = live
        self.assertIsNone(adopted.poll())
        self.assertFalse(adopted.closure_valid)

    def test_stale_running_receipt_is_nonzero_failure_within_one_second(self):
        adopted = self.fixture.adopt()
        self.fixture.backend.identities.clear()
        began = self.fixture.backend.clock
        self.assertEqual(adopted.poll(), 1)
        self.assertLessEqual(self.fixture.backend.clock - began, 1.000001)
        self.assertFalse(adopted.closure_valid)
        self.assertIsNone(adopted.child_returncode)
        self.assertIn('closure is invalid', adopted.closure_pending_reason)

    def test_dead_wrapper_live_child_fails_and_keeps_identity_for_cleanup(self):
        adopted = self.fixture.adopt()
        del self.fixture.backend.identities[self.fixture.wrapper_pid]
        self.fixture.backend.identities[self.fixture.child_pid]['ppid'] = 1
        self.assertEqual(adopted.poll(), 1)
        self.assertFalse(adopted.closure_valid)
        self.assertTrue(adopted.validate_owned_identity_for_cleanup()['child_live'])
        self.fixture.backend.identities[self.fixture.child_pid]['start_ticks'] += 1
        with self.assertRaises(AssertionError):
            adopted.validate_owned_identity_for_cleanup()
        with self.assertRaises(AssertionError):
            adopted.poll()

    def test_pid_reuse_and_group_command_changes_are_rejected(self):
        for key, value in (('start_ticks', 19999), ('pgid', 999), ('sid', 999), ('argv', ['another'])):
            original = copy.deepcopy(self.fixture.backend.identities)
            adopted = self.fixture.adopt()
            self.fixture.backend.identities[self.fixture.child_pid][key] = value
            with self.assertRaises(AssertionError):
                adopted.poll()
            self.fixture.backend.identities = original

    def test_bad_provenance_cannot_become_success(self):
        adopted = self.fixture.adopt()
        self.fixture.close(0)
        self.fixture.record['command'] = ['another']
        self.fixture.write()
        self.assertEqual(adopted.poll(), 1)
        self.assertFalse(adopted.closure_valid)
        with self.assertRaises(AssertionError):
            adopted.validate_owned_identity_for_cleanup()

    def test_truncated_receipt_cannot_become_success(self):
        adopted = self.fixture.adopt()
        self.fixture.backend.identities.clear()
        self.fixture.path.write_text('{')
        self.assertEqual(adopted.poll(), 1)
        self.assertFalse(adopted.closure_valid)

    def test_zombie_processes_count_as_dead_only_with_matching_start_ticks(self):
        adopted = self.fixture.adopt()
        bound = copy.deepcopy(self.fixture.backend.identities)
        self.fixture.close(0)
        for identity in bound.values():
            identity.update(state='Z', argv=[])
        self.fixture.backend.identities = bound
        self.assertEqual(adopted.poll(), 0)
        self.assertTrue(adopted.closure_valid)

    def test_constructor_rejects_closed_wrong_path_or_unbound_process(self):
        original = copy.deepcopy(self.fixture.backend.identities)
        self.fixture.backend.identities[self.fixture.child_pid]['ppid'] = 77
        with self.assertRaises(AssertionError):
            self.fixture.adopt()
        self.fixture.backend.identities = original
        self.fixture.close(0)
        with self.assertRaises(AssertionError):
            self.fixture.adopt()


def real_linux_checks():
    assert sys.platform.startswith('linux'), '--real-linux requires Linux'
    results = []
    for exit_code in (0, 7, -15):
        with tempfile.TemporaryDirectory(prefix='paired_supervision_CPU_') as directory:
            root = Path(directory).resolve()
            (root / 'audit').mkdir()
            (root / 'configs').mkdir()
            authorization = root / 'configs/resource_authorization_20261020_v3.json'
            authorization.write_text('{}\n')
            child = root / 'cpu_child.py'
            child.write_text(textwrap.dedent('''\
                from pathlib import Path
                import os, signal, sys, time
                gate = Path(sys.argv[1])
                while not gate.exists(): time.sleep(.01)
                code = int(sys.argv[2])
                if code < 0: os.kill(os.getpid(), -code)
                raise SystemExit(code)
                '''))
            wrapper = root / 'run_gpu_job_v2.py'
            wrapper.write_text(textwrap.dedent('''\
                from pathlib import Path
                import argparse, hashlib, json, os, subprocess, sys, time
                parser=argparse.ArgumentParser()
                parser.add_argument('--name',required=True)
                parser.add_argument('--max-seconds',type=float,required=True)
                parser.add_argument('command',nargs=argparse.REMAINDER)
                args=parser.parse_args()
                command=args.command[1:]
                root=Path(__file__).resolve().parent
                auth=root/'configs/resource_authorization_20261020_v3.json'
                process=subprocess.Popen(command,start_new_session=True)
                record=dict(name=args.name,pid=process.pid,wrapper_pid=os.getpid(),command=command,
                    start_utc='2026-10-26T01:00:00Z',max_seconds=args.max_seconds,
                    charged_seconds_before_job=0,total_hours_cap=None,occupied_time_deadline=None,
                    authorization=str(auth),authorization_sha256=hashlib.sha256(auth.read_bytes()).hexdigest(),status='RUNNING')
                receipt=root/'audit'/(args.name+'_job.json')
                receipt.write_text(json.dumps(record))
                started=time.monotonic()
                code=process.wait(timeout=args.max_seconds)
                record.update(status='COMPLETED' if code==0 else 'FAILED_OR_JOB_TIMEOUT_RESUMABLE',
                    exit_code=code,charged_seconds=time.monotonic()-started,end_utc='2026-10-26T01:00:02Z')
                receipt.write_text(json.dumps(record))
                raise SystemExit(0 if code==0 else 1)
                '''))
            gate = root / 'finish'
            command = [sys.executable, '-B', '-u', str(child), str(gate), str(exit_code)]
            name = 'isolated_CPU_' + str(abs(exit_code))
            receipt = root / 'audit' / (name + '_job.json')
            process = subprocess.Popen([sys.executable, '-B', str(wrapper), '--name', name,
                '--max-seconds', '10', '--', *command], start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 5
                while not receipt.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                assert receipt.exists(), 'CPU fixture wrapper did not register'
                record = json.loads(receipt.read_text())
                adopted = S.AdoptedProcess(receipt, name, command, process.pid, record['pid'], _root=root)
                assert adopted.poll() is None and adopted.elapsed_seconds < 10
                assert adopted.validate_owned_identity_for_cleanup()['child_live']
                gate.touch()
                observed_wrapper_exit = process.wait(timeout=5)
                inferred = adopted.poll()
                assert adopted.closure_valid and adopted.child_returncode == exit_code
                assert inferred == observed_wrapper_exit == (0 if exit_code == 0 else 1)
                results.append(dict(raw_child_exit=exit_code, observed_wrapper_exit=observed_wrapper_exit,
                    inferred_wrapper_exit=inferred, closure_valid=adopted.closure_valid,
                    elapsed_seconds=adopted.elapsed_seconds))
            finally:
                gate.touch()
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
                if process.stdout is not None:
                    process.stdout.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-linux', action='store_true')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SupervisionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful(), 'Supervision CPU regressions failed'
    linux = real_linux_checks() if args.real_linux else []
    print(json.dumps(dict(status='PAIRED_SUPERVISION_CPU_CHECKS_PASSED',
        CPU_checks=result.testsRun, real_linux_checks=linux, GPU_used=False,
        SSH_used=False, real_study_receipts_or_ledger_changed=False)))


if __name__ == '__main__':
    main()
