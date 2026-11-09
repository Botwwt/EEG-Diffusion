"""Finite CPU waiting queue for one frozen T80 direction/gain diagnostic.

The original four-arm queue is read-only and keeps its complete lifecycle.
Only after its registered controller exits successfully does this queue acquire
the same lifecycle lock and launch one new, separately bounded GPU diagnostic.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import brainhive_paired_study_contract_20261025_v1 as C
import queue_BrainHIVE_decisive_T_analysis_20261025_v2 as Q
from brainhive_paired_supervision_20261026_v3 import safe_bytes_used, validate_wrapper_exit, inspect_linux_process, AdoptedProcess

ROOT = Path(__file__).resolve().parent
METADATA = ROOT / 'audit/scientific_basis_reaudit_20261026_v1'
STATE_NAME = 'addon_queue_state.json'
ENTRY = 'diagnose_BrainHIVE_paired_direction_gain_T80_20261026_v1.py'
SUCCESS = 'ALL_FOUR_PAIRED_EFFECT_FULL_T_FULL_V_LIFECYCLES_AND_FIXED_COMPARISON_COMPLETE'
DONE = 'COMPLETE_FIXED_PAIRED_DIRECTION_GAIN_T80'
WAIT_SECONDS = 18 * 86400
POLL_SECONDS = 60
METADATA_CAP = 64 << 20
LOG_CAP = 32 << 20
OUTPUT_CAP = 512 << 20
JSON_CAP = 16 << 20
SOURCE_CAP = 4 << 20
ADDITIONAL_RESERVATION = METADATA_CAP + OUTPUT_CAP
LOCK = ROOT / 'audit/functional_full_lifecycle_20261022_v1/lifecycle_controller.lock'
C.bytes_used = safe_bytes_used  # This read-only supervisor alone tolerates atomic renames.


def sha(path, maximum=JSON_CAP):
    path = Path(path)
    assert path.stat().st_size <= maximum
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    path = Path(path)
    assert path.stat().st_size <= JSON_CAP
    return json.loads(path.read_bytes())


def bound(reference):
    assert sha(reference['path']) == reference['sha256'], reference['path']
    return read(reference['path'])


def status_only(path):
    """Read the first status line only, without parsing image score values.

    The fixed C.atomic completion files put status first. Refuse another layout
    rather than falling back to loading a protected comparison summary.
    """
    path = Path(path)
    assert path.stat().st_size <= JSON_CAP
    with path.open('rb') as stream:
        assert stream.readline(32).strip() == b'{', 'Expected fixed pretty JSON layout'
        line = stream.readline(4096)
    match = re.fullmatch(rb'  "status": ("(?:[^"\\]|\\.)*")(?:,)?\r?\n?', line)
    assert match, 'Expected a bounded first top-level status field'
    value = json.loads(match.group(1))
    assert isinstance(value, str)
    return value


def completion_specs(study):
    # Importing this CPU controller does not execute its main or import Torch.
    import queue_BrainHIVE_paired_effect_20261026_v3 as original
    result = {}
    for arm in C.ARMS:
        for stage in ('train', 'postfit', 'generate', 'audit', 'features', 'score'):
            result[(arm, stage)] = original.command_spec(study, arm, stage, int(arm == 'O' and stage == 'train'))
    for stage in ('functional', 'compare'):
        result[(None, stage)] = original.command_spec(study, None, stage)
    return result


def allowed_attempts(study):
    specs = completion_specs(study)
    allowed = {value['name']: value for value in specs.values()}
    import queue_BrainHIVE_paired_effect_20261026_v3 as original
    for arm in C.ARMS:
        for stage in ('train', 'generate'):
            for segment in (0, 1):
                spec = original.command_spec(study, arm, stage, segment)
                allowed[spec['name']] = spec
    takeover = C.bound(study['live_O_takeover'])
    name = takeover['job']['name']
    allowed[name] = allowed[name] | dict(command=takeover['job']['command'])
    return allowed


def process_identity(state, launch, allowed, *, inspect=inspect_linux_process, is_live=Q.live, previous=None):
    assert type(state.get('pid')) is int and state['pid'] == launch['pid'], 'Unexpected upstream controller PID'
    assert state['registration'] == launch['registration']
    observed = None
    if is_live(launch['pid']):
        observed = inspect(launch['pid'])
        assert observed is not None and observed['argv'] == launch['command'], 'Upstream controller command changed'
        assert observed['pid'] == observed['pgid'] == observed['sid'] == launch['pid']
        if previous is not None:
            for key in ('pid', 'start_ticks', 'pgid', 'sid', 'argv'):
                assert observed[key] == previous[key], 'Upstream PID identity changed'
    child = state.get('child_pid')
    if child is not None:
        assert type(child) is int and child > 1
        assert state.get('attempts')
        entry = state['attempts'][-1]
        assert entry['name'] == state['current_stage'] and entry['name'] in allowed
        spec = allowed[entry['name']]
        assert entry['command'] == spec['command'] and entry['GPU'] is spec['GPU']
        assert entry.get('wrapper_pid') == child, 'Unexpected upstream child PID'
        expected = ([launch['command'][0], '-B', str(ROOT / 'run_gpu_job_v2.py'),
                     '--name', entry['name'], '--max-seconds', str(spec['maximum_seconds']), '--', *spec['command']]
                    if spec['GPU'] else spec['command'])
        assert entry['wrapper_command'] == expected
        if is_live(child):
            actual = inspect(child)
            assert actual is not None and actual['argv'] == expected, 'Unexpected process at upstream child PID'
            assert actual['pid'] == actual['pgid'] == actual['sid'] == child
    return observed


def terminal(state, document, study, launch, *, status_reader=status_only, hash_reader=sha, is_live=Q.live):
    assert state['pid'] == launch['pid'] and state['registration'] == launch['registration']
    assert state['study_core_sha256'] == C.core(study)
    status = state.get('status', '')
    if status != SUCCESS:
        assert not any(word in status for word in ('FAILED', 'FAILURE', 'STOP_REQUESTED', 'STOPPED')), 'Original lifecycle failed; preserve without diagnostic launch'
        assert status == 'PAIRED_EFFECT_FOUR_ARM_LIFECYCLES_RUNNING', 'Unexpected upstream state'
        assert is_live(state['pid']), 'Original controller died before a successful terminal state'
        return None
    assert state.get('current_stage') is None and state.get('child_pid') is None
    assert not state.get('unresolved_live_child')
    # Success is saved shortly before the original process releases its lock.
    if is_live(state['pid']):
        return None
    keys = [(entry.get('arm'), entry['stage']) for entry in state['completed']]
    mandatory = [(arm, stage) for arm in C.ARMS for stage in ('train', 'postfit', 'generate', 'audit', 'features', 'score')]
    assert keys[:24] == mandatory
    if len(keys) == 26:
        assert keys[24:] == [(None, 'functional'), (None, 'compare')]
    else:
        assert len(keys) == 25 and keys[24] == (None, 'compare')
        assert state.get('functional_probe') == 'FAILED_PRESERVED_UNKNOWN_NO_AUTOMATIC_RETRY'
    specs = completion_specs(study)
    proofs = []
    for entry, key in zip(state['completed'], keys):
        spec = specs[key]; reference = entry['completion']
        assert Path(reference['path']) == Path(spec['completion'])
        assert hash_reader(reference['path']) == reference['sha256']
        observed_status = status_reader(reference['path'])
        assert observed_status == spec['expected_status'], (key, observed_status)
        proofs.append(dict(arm=key[0], stage=key[1], completion=reference, status=observed_status))
    return dict(status=SUCCESS, original_controller_pid=launch['pid'], controller_exited=True,
                completed_status_refs=proofs, V_score_values_read=False,
                old_functional_probe_unknown=state.get('functional_probe') == 'FAILED_PRESERVED_UNKNOWN_NO_AUTOMATIC_RETRY')


def validate_queue(document, *, check_files=True, enforce_paths=True):
    assert document['schema'] == 'BrainHIVE_paired_direction_gain_T80_20261026_v1'
    assert document['status'] == 'REGISTERED_SUPPLEMENTARY_T_ONLY_FROZEN_DIRECTION_GAIN_PROBE'
    assert document['maximum_wait_seconds'] == WAIT_SECONDS
    assert document['poll_seconds'] == POLL_SECONDS
    assert document['metadata_maximum_bytes'] == METADATA_CAP
    assert document['maximum_seconds'] == 7200 and document['maximum_output_bytes'] == OUTPUT_CAP
    assert document['maximum_extra_CPU_bytes'] == 10 << 30
    assert document['maximum_GPU_bytes'] == 20 << 30
    assert document['minimum_CPU_headroom_bytes'] == 10 << 30
    assert document['formal_queue_completion']['expected_status'] == SUCCESS
    assert document.get('automatic_retry', False) is False
    metadata = Path(document['metadata_directory'])
    output = Path(document['output_directory'])
    assert metadata != output and not output.is_relative_to(metadata) and not metadata.is_relative_to(output)
    if enforce_paths:
        assert metadata == METADATA
        assert Path(document['formal_queue_completion']['state_path']) == ROOT / 'audit/paired_effect_study_20261025_v1/reaudit_v3/queue_state.json'
        assert Path(document['formal_launch_receipt']['path']) == ROOT / 'audit/paired_effect_study_20261025_v1/reaudit_v3/actual_launch_receipt.json'
        assert output == C.DATA / 'scientific_basis_probe_20261026_v1'
    required = {Path(__file__).name, 'check_BrainHIVE_paired_direction_gain_queue_CPU_20261026_v1.py',
                ENTRY, 'run_gpu_job_v2.py', 'brainhive_paired_supervision_20261026_v3.py',
                'queue_BrainHIVE_decisive_T_analysis_20261025_v2.py', 'brainhive_paired_study_contract_20261025_v1.py'}
    assert required <= set(document['source_sha256'])
    for name, digest in document['source_sha256'].items():
        assert Path(name).name == name and len(digest) == 64
        if check_files:
            assert sha(ROOT / name, SOURCE_CAP) == digest, name
    if not check_files:
        return document
    probe = importlib.import_module(Path(ENTRY).stem)
    study = probe.validate_addon(document)
    assert C.validate(study) == C.core(study)
    launch = bound(document['formal_launch_receipt'])
    assert launch['registration'] == document['original_study_registration']
    assert launch['scientific_core_sha256'] == C.core(study)
    assert launch['pid'] == 136961 and launch['initial_stages'] == 26
    assert launch['command'] == [sys.executable, '-B', '-u', str(ROOT / 'queue_BrainHIVE_paired_effect_20261026_v3.py'),
                                 '--registration', document['original_study_registration']['path']]
    admission = bound(launch['admission'])
    assert admission['status'] == 'FULL_PAIRED_STUDY_ADMITTED_AFTER_ACTUAL_ENGINEERING'
    assert admission['study_core_sha256'] == C.core(study)
    assert admission['config_sha256'] == document['original_study_registration']['sha256']
    assert Path(launch['admission']['path']) == C.DATA / 'admission.json'
    return study, launch


def resources(document, study):
    original = C.resources(study)
    metadata = safe_bytes_used(document['metadata_directory'])
    output = safe_bytes_used(document['output_directory'])
    assert metadata <= METADATA_CAP and output <= OUTPUT_CAP, 'Finite supplemental output cap exceeded'
    base_required = original['reserved_remaining_bytes'] + study['disk_buffer_bytes']
    assert original['free_bytes'] >= base_required + ADDITIONAL_RESERVATION, 'Preserve original full-lifecycle reserve plus576MiB supplement'
    metadata_path = Path(document['metadata_directory'])
    while not metadata_path.exists(): metadata_path = metadata_path.parent
    metadata_free = shutil.disk_usage(metadata_path).free
    assert metadata_free >= (METADATA_CAP - metadata) + METADATA_CAP, 'Metadata volume requires remaining cap plus64MiB headroom'
    return dict(original_remaining_and_buffer_bytes=base_required, supplemental_reserved_bytes=ADDITIONAL_RESERVATION,
                metadata_bytes=metadata, probe_bytes=output, metadata_free_bytes=metadata_free,
                persistent_free_bytes=original['free_bytes'], memory=original['memory'])


def preflight(path):
    assert os.name == 'posix' and ROOT == C.REMOTE
    document = read(path); study, launch = validate_queue(document)
    probe = importlib.import_module(Path(ENTRY).stem)
    report = probe.preflight(document, allow_upstream_pending=True)
    assert 'diffusers' not in sys.modules, 'Preflight must not load diffusion models'
    if 'torch' in sys.modules:
        assert not sys.modules['torch'].cuda.is_initialized(), 'CPU preflight must not initialize CUDA'
    state = read(document['formal_queue_completion']['state_path'])
    process_identity(state, launch, allowed_attempts(study))
    terminal(state, document, study, launch)
    return document, study, launch, dict(status='DIRECTION_GAIN_ADDON_QUEUE_CPU_PREFLIGHT_VALID_UPSTREAM_MAY_BE_PENDING',
        registration=dict(path=str(path), sha256=sha(path)), upstream_status=state['status'],
        maximum_wait_seconds=WAIT_SECONDS, one_GPU_job_only=True, automatic_retry=False,
        CUDA_initialized=False, no_GPU_query=True, V_score_values_read=False, probe_preflight=report)


def validate_closed_job(record, wrapper_pid, wrapper_code, child_command, document, *, is_live=Q.live):
    assert record['wrapper_pid'] == wrapper_pid and record['name'] == document['gpu_job_name']
    assert record['command'] == child_command
    assert not is_live(record['pid']) and not is_live(record['wrapper_pid'])
    assert record['authorization_sha256'] == document['resource_authorization']['sha256']
    assert record['max_seconds'] == document['maximum_seconds']
    assert record['status'] == ('COMPLETED' if record['exit_code'] == 0 else 'FAILED_OR_JOB_TIMEOUT_RESUMABLE')
    return validate_wrapper_exit(record['exit_code'], wrapper_code)


def bind_owned_after_registration(process, job, name, command, *, grace_seconds=3,
                                 exists=lambda path: Path(path).exists(), reader=read,
                                 adapter=AdoptedProcess, is_live=Q.live,
                                 monotonic=time.monotonic, sleep=time.sleep):
    """Allow only the wrapper's short atomic receipt-registration interval."""
    assert 0 < grace_seconds <= 3
    deadline = monotonic() + grace_seconds
    while True:
        if exists(job):
            record = reader(job)
            assert record['wrapper_pid'] == process.pid and record['command'] == command and record['name'] == name
            if record['status'] == 'RUNNING' and process.poll() is None and is_live(record['pid']):
                return adapter(job, name, command, process.pid, record['pid'])
            if record['status'] != 'RUNNING' and process.poll() is not None:
                return None
        if monotonic() >= deadline:
            return None
        sleep(min(.05, max(0, deadline - monotonic())))


def cleanup_owned(process, owned, job, name, command, evidence, *, binder=bind_owned_after_registration,
                  exists=lambda path: Path(path).exists(), reader=read, is_live=Q.live,
                  stop_owned=Q.stop_owned):
    if owned is None:
        owned = binder(process, job, name, command)
    if owned is not None:
        owned.validate_owned_identity_for_cleanup()
        evidence['stable_identity_bound_before_signals'] = True
        return stop_owned(owned, True, job, command, name, evidence)
    record = reader(job) if exists(job) else None
    if record is not None:
        assert record['wrapper_pid'] == process.pid and record['command'] == command and record['name'] == name
    closed = bool(record is not None and record.get('status') in ('COMPLETED', 'FAILED_OR_JOB_TIMEOUT_RESUMABLE'))
    stopped = bool(closed and process.poll() is not None and not is_live(record['pid']) and not is_live(record['wrapper_pid']))
    evidence.update(complete=stopped, signals_sent=False,
        reason='Closed receipt and both registered PIDs dead' if stopped else
               'Registration grace ended without stable live identities or a closed dead job; preserve unresolved')
    return stopped


class Runner:
    def __init__(self, document, study, launch, path):
        self.document, self.study, self.launch = document, study, launch
        self.path = Path(path); self.metadata = Path(document['metadata_directory'])
        self.state_path = self.metadata / STATE_NAME
        assert not self.state_path.exists(), 'Preserve previous queue; no automatic retry'
        self.state = dict(status='WAITING_FOR_ORIGINAL_FULL_LIFECYCLE_SUCCESS_AND_EXIT', pid=os.getpid(),
            child_pid=None, current_stage=None, attempts=[], completed=[],
            registration=dict(path=str(self.path), sha256=sha(self.path)), upstream_controller_pid=launch['pid'],
            V_score_values_read=False, no_training=True, automatic_retry=False, maximum_GPU_jobs=1)
        self.upstream_identity = None
        self.save()

    def save(self):
        self.state['UTC'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        C.atomic(self.state_path, self.state)
        assert safe_bytes_used(self.metadata) <= METADATA_CAP

    def validate(self):
        assert sha(self.path) == self.state['registration']['sha256']
        study, launch = validate_queue(self.document)
        assert study == self.study and launch == self.launch

    def wait(self):
        import fcntl
        started = time.monotonic()
        while True:
            elapsed = time.monotonic() - started
            assert elapsed < WAIT_SECONDS, 'Finite18day upstream waiting limit reached'
            self.validate(); self.state['resources'] = resources(self.document, self.study)
            upstream = read(self.document['formal_queue_completion']['state_path'])
            current = process_identity(upstream, self.launch, allowed_attempts(self.study), previous=self.upstream_identity)
            if current is not None:
                self.upstream_identity = current
                self.state['upstream_controller_identity'] = current
            proof = terminal(upstream, self.document, self.study, self.launch)
            if proof is not None:
                lock = LOCK.open('a')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close()
                else:
                    # Recheck successful closure after acquiring the original lock.
                    assert terminal(read(self.document['formal_queue_completion']['state_path']), self.document,
                                    self.study, self.launch) == proof
                    self.state['upstream_closure'] = proof
                    self.state['wait_elapsed_seconds'] = elapsed; self.save()
                    return lock
            self.state.update(wait_elapsed_seconds=elapsed, upstream_status=upstream['status'],
                              upstream_completed_stages=len(upstream.get('completed', [])))
            self.save()
            time.sleep(min(POLL_SECONDS, max(.001, WAIT_SECONDS - elapsed)))

    def stage(self):
        self.validate(); resources(self.document, self.study)
        probe = importlib.import_module(Path(ENTRY).stem)
        probe.preflight(self.document, allow_upstream_pending=False)
        Q.gpu_free()
        output = Path(self.document['output_directory'])
        assert not output.exists() or not any(output.iterdir()), 'Fresh diagnostic output only; no retry'
        name = self.document['gpu_job_name']; job = ROOT / 'audit' / (name + '_job.json')
        log = self.metadata / (name + '.log')
        assert not job.exists() and not log.exists()
        child = [sys.executable, '-B', '-u', str(ROOT / ENTRY), '--registration', str(self.path)]
        argv = [sys.executable, '-B', str(ROOT / 'run_gpu_job_v2.py'), '--name', name,
                '--max-seconds', str(self.document['maximum_seconds']), '--', *child]
        attempt = dict(command=argv, child_command=child, GPU=True, job_name=name, log=str(log), maximum_seconds=7200)
        self.state['attempts'].append(attempt)
        self.state.update(status='RUNNING_ONE_FIXED_FROZEN_T80_DIRECTION_GAIN_PROBE', current_stage='direction_gain_T80'); self.save()
        process = owned = None; started = time.monotonic(); last_resource = last_source = -100.
        env = os.environ.copy()
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            env[key] = '2'
        try:
            with log.open('xb') as stream:
                process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                self.state['child_pid'] = attempt['wrapper_pid'] = process.pid; self.save()
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    assert elapsed < 7200 + 120 and log.stat().st_size <= LOG_CAP
                    if owned is None and job.exists():
                        registered = read(job)
                        assert registered['wrapper_pid'] == process.pid and registered['command'] == child and registered['name'] == name
                        if registered['status'] == 'RUNNING':
                            owned = AdoptedProcess(job, name, child, process.pid, registered['pid'])
                            attempt['owned_process_identity'] = owned.identity_evidence
                    elif owned is not None:
                        owned.poll()  # Reject PID reuse even though Popen owns the wrapper.
                    if elapsed - last_resource >= 10:
                        resources(self.document, self.study); last_resource = elapsed
                    if elapsed - last_source >= 60:
                        self.validate(); last_source = elapsed
                    time.sleep(2)
                record = read(job)
                if owned is not None:
                    owned.validate_owned_identity_for_cleanup()
                attempt.update(validate_closed_job(record, process.pid, process.returncode, child, self.document))
                attempt['job'] = dict(path=str(job), sha256=sha(job))
                assert process.returncode == 0, 'Probe failure preserved; no automatic retry'
                assert status_only(output / 'completion.json') == DONE
                self.state['child_pid'] = None
        except BaseException as error:
            attempt['error'] = repr(error)
            if process is not None:
                attempt['owned_cleanup'] = {}
                try:
                    stopped = cleanup_owned(process, owned, job, name, child, attempt['owned_cleanup'])
                except BaseException as cleanup_error:
                    stopped = False; attempt['cleanup_error'] = repr(cleanup_error)
                self.state['child_pid'] = None if stopped else process.pid
                self.state['unresolved_live_child'] = not stopped
            self.save(); raise
        finally:
            attempt['seconds'] = time.monotonic() - started
            attempt['wrapper_exit_code'] = process.poll() if process is not None else None
            if log.exists():
                attempt['log_bytes'] = log.stat().st_size
                attempt['log_sha256'] = sha(log, LOG_CAP)
            self.save()
        resources(self.document, self.study)
        completion = output / 'completion.json'
        self.state['completed'].append(dict(stage='direction_gain_T80', completion=dict(path=str(completion), sha256=sha(completion)), status=DONE))
        self.state.update(status='ONE_FIXED_FROZEN_T80_DIRECTION_GAIN_PROBE_COMPLETE_NO_FIT', current_stage=None,
                          child_pid=None, method_effectiveness_established=False, A_conference_contribution_established=False)
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    document, study, launch, report = preflight(args.registration)
    if args.preflight_only:
        print(json.dumps(report)); return
    import fcntl
    metadata = Path(document['metadata_directory']); metadata.mkdir(parents=True, exist_ok=True)
    with (metadata / 'addon_controller.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Runner(document, study, launch, args.registration)
        lock = None
        def interrupt(signum, frame):
            raise InterruptedError('Supplemental controller received signal ' + str(signum))
        signal.signal(signal.SIGTERM, interrupt)
        try:
            lock = runner.wait(); runner.stage()
        except BaseException as error:
            runner.state.update(status='SUPPLEMENTAL_DIAGNOSTIC_FAILED_PRESERVED_NO_AUTOMATIC_RETRY', error=repr(error))
            runner.save(); raise
        finally:
            if lock is not None: lock.close()


if __name__ == '__main__':
    main()
