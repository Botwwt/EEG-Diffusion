"""Read-only supervision helpers for the unchanged paired-effect study.

An adopted process is not our subprocess: its return code is inferred from a
bound, closed wrapper receipt, never claimed to have been observed by waitpid.
The optional private backends exist only for CPU fixtures; production uses /proc.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import stat
import sys
import time


ROOT = Path(__file__).resolve().parent
FAILED_CLOSURE_RETURN_CODE = 1
_DEAD_STATES = frozenset(('Z', 'X', 'x'))


def safe_bytes_used(path):
    """Count regular files with one stat each; tolerate only vanished entries."""
    total = 0
    for entry in Path(path).rglob('*'):
        try:
            observed = entry.stat()
        except FileNotFoundError:
            # Atomic checkpoint/image replacement may remove a pending name
            # after enumeration. Other I/O and permission errors must surface.
            continue
        if stat.S_ISREG(observed.st_mode):
            total += observed.st_size
    return total


def validate_wrapper_exit(child_exit, wrapper_exit):
    """Validate run_gpu_job_v2's 0/1 mapping, retaining the raw child code."""
    assert type(child_exit) is int and type(wrapper_exit) is int
    expected = 0 if child_exit == 0 else 1
    assert wrapper_exit == expected, 'GPU wrapper exit disagrees with its raw child exit'
    return dict(child_exit_code=child_exit, wrapper_exit_code=wrapper_exit,
                expected_wrapper_exit_code=expected)


def _finite(value, *, positive=False):
    assert type(value) in (int, float) and math.isfinite(value)
    assert value > 0 if positive else value >= 0
    return float(value)


def _proc_stat(pid):
    text = Path('/proc').joinpath(str(pid), 'stat').read_text()
    # A process comm may itself contain whitespace or parentheses.
    tail = text[text.rfind(')') + 2:].split()
    assert text[:text.find('(')].strip() == str(pid) and len(tail) >= 20
    return dict(pid=pid, state=tail[0], ppid=int(tail[1]), pgid=int(tail[2]),
                sid=int(tail[3]), start_ticks=int(tail[19]))


def inspect_linux_process(pid):
    """Return a stable process identity, or None if the process disappeared."""
    try:
        before = _proc_stat(pid)
        raw = Path('/proc').joinpath(str(pid), 'cmdline').read_bytes()
        after = _proc_stat(pid)
    except FileNotFoundError:
        return None
    assert before['start_ticks'] == after['start_ticks'], 'PID reused during identity inspection'
    after['argv'] = [os.fsdecode(part) for part in raw.split(b'\0') if part]
    return after


class _LinuxBackend:
    def __init__(self):
        assert sys.platform.startswith('linux'), 'Live process adoption requires Linux /proc'
        assert hasattr(time, 'CLOCK_BOOTTIME')
        self.ticks_per_second = os.sysconf('SC_CLK_TCK')

    inspect = staticmethod(inspect_linux_process)
    monotonic = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)

    @staticmethod
    def boottime():
        return time.clock_gettime(time.CLOCK_BOOTTIME)


class AdoptedProcess:
    """Popen-like read-only adapter for one already running GPU wrapper.

    ``pid``, ``poll()`` and ``returncode`` support the existing ownership cleanup.
    Call ``validate_owned_identity_for_cleanup()`` immediately before cleanup;
    it rejects PID reuse and changed process groups/commands. ``elapsed_seconds``
    retains the original wrapper age across controller replacement.

    A dead wrapper with a live bound child or an unclosed receipt produces a
    nonzero failure sentinel. ``closure_valid`` distinguishes that sentinel from
    a valid receipt-derived 0/1 exit, and ``child_returncode`` preserves raw exits.
    This adapter never signals processes or writes receipts, ledgers, or locks.
    """
    def __init__(self, record_path, expected_name, expected_command,
                 expected_wrapper_pid, expected_child_pid, *, _root=None,
                 _backend=None, _receipt_reader=None):
        self._root = ROOT if _root is None else Path(_root).resolve()
        self._backend = _LinuxBackend() if _backend is None else _backend
        self._read_receipt = Path(record_path).read_bytes if _receipt_reader is None else _receipt_reader
        self.record_path = Path(record_path).resolve()
        assert isinstance(expected_name, str) and expected_name and '/' not in expected_name and '\\' not in expected_name
        assert self.record_path == self._root / 'audit' / (expected_name + '_job.json')
        assert type(expected_command) is list and expected_command and all(type(x) is str and x for x in expected_command)
        assert type(expected_wrapper_pid) is int and expected_wrapper_pid > 1
        assert type(expected_child_pid) is int and expected_child_pid > 1 and expected_child_pid != expected_wrapper_pid
        self.pid = expected_wrapper_pid
        self.child_pid = expected_child_pid
        self.expected_name = expected_name
        self.expected_command = list(expected_command)
        self.returncode = None
        self.child_returncode = None
        self.returncode_inferred = True
        self.inferred = True
        self.closure_valid = False
        self.closure_record = None
        self.closure_pending_reason = None
        self.wrapper_observed_dead = False
        self.child_observed_dead = False

        raw, record = self._stable_record()
        assert record.get('status') == 'RUNNING', 'Adopt only the registered active wrapper'
        self._validate_initial_record(record)
        self._initial_record = record
        self.max_seconds = _finite(record['max_seconds'], positive=True)
        wrapper = self._backend.inspect(self.pid)
        child = self._backend.inspect(self.child_pid)
        assert self._live(wrapper) and self._live(child), 'Wrapper and bound child must be live at adoption'
        assert wrapper['pid'] == self.pid and child['pid'] == self.child_pid
        assert wrapper['pgid'] == wrapper['sid'] == self.pid
        assert child['pgid'] == child['sid'] == self.child_pid
        assert child['ppid'] == self.pid, 'GPU child must originate from the registered wrapper'
        assert child['argv'] == self.expected_command
        self._validate_wrapper_command(wrapper['argv'])
        self._wrapper_identity = wrapper
        self._child_identity = child
        assert type(wrapper['start_ticks']) is int and wrapper['start_ticks'] > 0
        assert type(child['start_ticks']) is int and child['start_ticks'] >= wrapper['start_ticks']
        assert type(self._backend.ticks_per_second) in (int, float) and self._backend.ticks_per_second > 0
        # A final read rejects a closure or replacement during identity binding.
        final_raw, final_record = self._stable_record()
        assert final_raw == raw and final_record == record, 'Receipt changed while binding the running processes'
        stable_keys = ('pid', 'start_ticks', 'pgid', 'sid', 'argv')
        self.identity_evidence = dict(
            record_path=str(self.record_path), initial_receipt_sha256=hashlib.sha256(raw).hexdigest(),
            wrapper={key: wrapper[key] for key in stable_keys},
            child={key: child[key] for key in stable_keys}, clock='CLOCK_BOOTTIME',
            clock_ticks_per_second=self._backend.ticks_per_second,
            return_code_source='INFERRED_FROM_BOUND_CLOSED_RECEIPT_NOT_WAITPID')
        # Fail at binding, rather than silently resetting an inconsistent age.
        assert self.elapsed_seconds >= 0

    @staticmethod
    def _live(identity):
        return identity is not None and identity['state'] not in _DEAD_STATES

    def _stable_record(self):
        first = self._read_receipt()
        second = self._read_receipt()
        assert first == second, 'Wrapper receipt is being updated'
        result = json.loads(second)
        assert type(result) is dict
        return second, result

    def _validate_initial_record(self, record):
        assert record.get('name') == self.expected_name
        assert record.get('command') == self.expected_command
        assert record.get('wrapper_pid') == self.pid and type(record.get('wrapper_pid')) is int
        assert record.get('pid') == self.child_pid and type(record.get('pid')) is int
        assert type(record.get('start_utc')) is str and record['start_utc']
        _finite(record['max_seconds'], positive=True)
        _finite(record['charged_seconds_before_job'])
        assert record.get('total_hours_cap') is None and record.get('occupied_time_deadline') is None
        authorization = self._root / 'configs/resource_authorization_20261020_v3.json'
        assert Path(record.get('authorization', '')).resolve() == authorization
        assert record.get('authorization_sha256') == hashlib.sha256(authorization.read_bytes()).hexdigest()

    def _validate_wrapper_command(self, argv):
        assert len(argv) == 8 + len(self.expected_command)
        assert argv[:5] == [self.expected_command[0], '-B', str(self._root / 'run_gpu_job_v2.py'),
                            '--name', self.expected_name]
        assert argv[5] == '--max-seconds' and argv[7] == '--'
        assert math.isfinite(float(argv[6])) and float(argv[6]) == self.max_seconds
        assert argv[8:] == self.expected_command

    @property
    def elapsed_seconds(self):
        age = float(self._backend.boottime()) - self._wrapper_identity['start_ticks'] / self._backend.ticks_per_second
        assert math.isfinite(age) and age >= 0, 'Original wrapper age is inconsistent with CLOCK_BOOTTIME'
        return age

    def _current_identity(self, bound):
        current = self._backend.inspect(bound['pid'])
        if current is None:
            return None
        for key in ('pid', 'start_ticks', 'pgid', 'sid'):
            assert current[key] == bound[key], 'Bound process identity changed: ' + key
        # Linux empties cmdline for zombies. A still-running process must match.
        if self._live(current):
            assert current['argv'] == bound['argv'], 'Bound process command changed'
        return current

    def validate_owned_identity_for_cleanup(self):
        """Recheck both bound identities immediately before external signals."""
        wrapper = self._current_identity(self._wrapper_identity)
        child = self._current_identity(self._child_identity)
        _, record = self._stable_record()
        self._validate_unchanged(record)
        return dict(wrapper_pid=self.pid, child_pid=self.child_pid,
                    wrapper_live=self._live(wrapper), child_live=self._live(child),
                    wrapper_start_ticks=self._wrapper_identity['start_ticks'],
                    child_start_ticks=self._child_identity['start_ticks'], identity_verified=True)

    def _validate_unchanged(self, record):
        # Only wrapper closure fields may change; all original provenance stays.
        for key, value in self._initial_record.items():
            if key != 'status':
                assert record.get(key) == value, 'Wrapper receipt provenance changed: ' + key

    def _fail_closure(self, reason):
        self.closure_valid = False
        self.closure_pending_reason = reason
        self.returncode = FAILED_CLOSURE_RETURN_CODE
        return self.returncode

    def poll(self):
        wrapper = self._current_identity(self._wrapper_identity)
        child = self._current_identity(self._child_identity)
        # Existing cleanup polls repeatedly before later TERM/KILL decisions.
        # Keep rejecting PID reuse even after a failure sentinel was cached.
        if self.returncode is not None:
            return self.returncode
        if self._live(wrapper):
            return None
        self.wrapper_observed_dead = True
        self.child_observed_dead = not self._live(child)
        if not self.child_observed_dead:
            return self._fail_closure('Original wrapper died while its identity-verified child remains live; cleanup required')
        deadline = self._backend.monotonic() + 1.0
        while True:
            try:
                raw, record = self._stable_record()
                self._validate_unchanged(record)
                child_exit = record.get('exit_code')
                assert type(child_exit) is int, 'Closed wrapper receipt needs a raw child exit'
                expected_status = 'COMPLETED' if child_exit == 0 else 'FAILED_OR_JOB_TIMEOUT_RESUMABLE'
                assert record.get('status') == expected_status, 'Wrapper receipt has not reached a consistent closed status'
                _finite(record['charged_seconds'])
                assert type(record.get('end_utc')) is str and record['end_utc']
                self.child_returncode = child_exit
                self.returncode = 0 if child_exit == 0 else 1
                validate_wrapper_exit(child_exit, self.returncode)
                self.closure_valid = True
                self.closure_pending_reason = None
                self.closure_record = record
                self.closure_receipt_sha256 = hashlib.sha256(raw).hexdigest()
                return self.returncode
            except (AssertionError, FileNotFoundError, json.JSONDecodeError, KeyError) as error:
                reason = type(error).__name__ + ': ' + str(error)
                remaining = deadline - self._backend.monotonic()
                if remaining <= 0:
                    return self._fail_closure('Original wrapper and child are dead but receipt closure is invalid: ' + reason)
                self._backend.sleep(min(.05, remaining))
