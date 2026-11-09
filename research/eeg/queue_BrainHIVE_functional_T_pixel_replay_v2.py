"""One CPU-only wait, then at most one registered three-image T probe.

This observer never imports Torch, fits a model, evaluates V/test, or changes an
original job. Transient observation failures only wait. Source/input drift,
existing probe artifacts, or a launch failure stop with preserved evidence.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from brainhive_functional_training_contract_v2 import ROOT, REPO, WORK, MAIN, DIAG, MAIN_TERMINAL, DIAG_TERMINAL, sha, read, resource_headroom
from diagnose_BrainHIVE_functional_T_pixel_replay_v2 import (
    OUT, CONTROLLERS, validate_config, fixed_input_binding, independent_preflight, proc_snapshot,
)

PERIOD = 1200
MAX_WAIT = 7 * 86400
CONFIG_OUT = str(REPO / 'configs/BrainHIVE_functional_T_pixel_replay_v2_reviewed_once.json')
STATE = str(WORK / 'audit/functional_T_pixel_replay_v2_wait_state.json')
LOG = str(WORK / 'audit/functional_T_pixel_replay_v2_wait.jsonl')
LOCK = str(WORK / 'audit/functional_T_pixel_replay_v2_wait.lock')
PROBE = 'diagnose_BrainHIVE_functional_T_pixel_replay_v2.py'


def validate_registration(registration, *, check_files=True):
    fixed = dict(schema='BrainHIVE_functional_T_pixel_replay_v2_wait_registration',
        status='REGISTERED_CPU_WAIT_ONLY_NO_GPU_LAUNCHED', automatic_retry=False,
        polling_seconds=PERIOD, maximum_wait_seconds=MAX_WAIT,
        maximum_probe_seconds=1200, maximum_launch_attempts=1,
        reviewed_config_path=CONFIG_OUT, queue_state_path=STATE, queue_log_path=LOG,
        queue_lock_path=LOCK, formal_training_enabled=False, full_V_generation_enabled=False)
    for key, expected in fixed.items():
        assert registration[key] == expected, (key, registration.get(key), expected)
    template = registration['probe_template']
    assert template['status'] == 'PENDING_TERMINAL_FACTS_NOT_EXECUTABLE'
    assert template.get('terminal_state_bindings', []) == []
    assert template['maximum_seconds'] == 1200 and template['output_directory'] == str(OUT)
    prospective = copy.deepcopy(template)
    prospective['status'] = 'REVIEWED_ENGINEERING_PROBE_READY'
    validate_config(prospective, check_files=check_files)
    needed = {Path(__file__).name, PROBE, 'run_gpu_job_v2.py', 'brainhive_functional_training_contract_v2.py'}
    assert needed <= set(registration['source_sha256'])
    for filename, digest in template['source_sha256'].items():
        assert registration['source_sha256'][filename] == digest
    if check_files:
        for filename, digest in registration['source_sha256'].items():
            assert Path(filename).name == filename and sha(ROOT / filename) == digest
        # Hashes + metadata only. The mapping is fixed before waiting/images.
        binding = fixed_input_binding(prospective)
        return dict(subject=binding['subject'], query_id=binding['query_id'],
            array_row=binding['query']['array_row'], cache_row=binding['cache_row'],
            parent_role=binding['source']['role'], image_seed=binding['image_seed'],
            static_input_identity_bound=True, actual_GPU_used=False)
    return dict(static_schema_valid=True, actual_GPU_used=False)


def readiness(main_state, main_process, diag_state, diag_process, *, other_controllers, GPU_idle):
    """Pure decision; no promotion from RUNNING or unverified process identity."""
    for name, state, snapshot, terminal in (
        ('main', main_state, main_process, MAIN_TERMINAL), ('T32', diag_state, diag_process, DIAG_TERMINAL)):
        if not state or state.get('status') not in terminal:
            return dict(ready=False, reason=name + '_NOT_TERMINAL')
        if not state.get('pid') or snapshot is None:
            return dict(ready=False, reason=name + '_PROCESS_IDENTITY_UNVERIFIED')
        if snapshot.get('live'):
            return dict(ready=False, reason=name + '_CONTROLLER_STILL_ACTIVE_OR_PID_REUSED')
    if other_controllers:
        return dict(ready=False, reason='ORIGINAL_CONTROLLER_INSTANCE_STILL_ACTIVE')
    if not GPU_idle:
        return dict(ready=False, reason='GPU_OR_GPU_LOCK_BUSY')
    return dict(ready=True, reason='BOTH_TERMINAL_INACTIVE_AND_GPU_IDLE')


def no_existing_probe_artifacts(registration):
    template = registration['probe_template']
    for path in (registration['reviewed_config_path'], template['output_directory'],
                 str(ROOT / 'audit' / (template['job_name'] + '_job.json'))):
        assert not Path(path).exists(), 'Preserve existing one-shot probe artifact; no automatic retry: ' + path


def gpu_idle_and_unlocked(template):
    observed = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=10)
    active = [int(value.strip()) for value in observed.stdout.splitlines() if value.strip()]
    capacity = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=10)
    free = [int(value.strip()) * (1 << 20) for value in capacity.stdout.splitlines() if value.strip()]
    assert len(free) == 1, 'Only the existing single GPU is authorized'
    if active or free[0] < template['minimum_free_GPU_bytes']:
        return False
    import fcntl
    with (ROOT / 'audit/GPU_JOB.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream, fcntl.LOCK_UN)
    return True


def observe(template):
    states, snapshots = {}, {}
    for label, path in (('main', MAIN), ('T32', DIAG)):
        try:
            state = read(str(path))
        except (FileNotFoundError, json.JSONDecodeError):
            state = None
        states[label] = state
        snapshots[label] = proc_snapshot(state['pid']) if state and state.get('pid') else None
    others = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():
            continue
        try:
            command = (folder / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        if any(script.encode() in command for script in CONTROLLERS):
            snapshot = proc_snapshot(int(folder.name))
            if snapshot['live']:
                others.append(dict(pid=int(folder.name), snapshot=snapshot))
    eligible = readiness(states['main'], snapshots['main'], states['T32'], snapshots['T32'],
                         other_controllers=others, GPU_idle=True)
    if eligible['ready']:
        eligible = readiness(states['main'], snapshots['main'], states['T32'], snapshots['T32'],
            other_controllers=others, GPU_idle=gpu_idle_and_unlocked(template))
    if eligible['ready']:
        headroom = resource_headroom()['effective_headroom_bytes']
        free = shutil.disk_usage(str(WORK)).free
        if (headroom < template['minimum_CPU_headroom_bytes'] or
                free < template['maximum_probe_bytes'] + template['minimum_retained_disk_bytes']):
            eligible = dict(ready=False, reason='SMALL_PROBE_RESOURCE_BUDGET_NOT_AVAILABLE')
    return dict(decision=eligible, main=states['main'], main_proc=snapshots['main'],
                T32=states['T32'], T32_proc=snapshots['T32'], other_controller_processes=others)


def save_state(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    assert not temporary.exists(), 'Preserve an interrupted observer write'
    data = dict(value, UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    with temporary.open('x') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return data


def create_reviewed_config(registration):
    """Bind real terminal facts once; no models or external approval is involved."""
    no_existing_probe_artifacts(registration)
    prospective = copy.deepcopy(registration['probe_template'])
    prospective['status'] = 'REVIEWED_ENGINEERING_PROBE_READY'
    prospective['terminal_state_bindings'] = [dict(path=str(path), sha256=sha(str(path))) for path in (MAIN, DIAG)]
    evidence = independent_preflight(prospective)
    # No state may change between preflight and the once-only config write.
    assert all(sha(item['path']) == item['sha256'] for item in prospective['terminal_state_bindings'])
    path = Path(registration['reviewed_config_path'])
    with path.open('x') as stream:
        stream.write(json.dumps(prospective, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return prospective, evidence


def inspect_probe_result(registration, code):
    template = registration['probe_template']
    job_path = ROOT / 'audit' / (template['job_name'] + '_job.json')
    summary_path = Path(template['output_directory']) / 'summary.json'
    if code != 0 or not job_path.exists() or not summary_path.exists():
        return dict(status='ONE_SHOT_PIXEL_PROBE_FAILED_OR_INCOMPLETE_PRESERVED_NO_RETRY',
            wrapper_exit_code=code, job_receipt_path=str(job_path), summary_path=str(summary_path),
            scientific_hypothesis_rejected=False)
    job, summary = read(job_path), read(summary_path)
    assert job['status'] == 'COMPLETED' and job['exit_code'] == 0
    assert summary['status'] == 'FUNCTIONAL_T_PIXEL_REPLAY_V2_EXACT_THREE_PATHS_PASSED'
    assert summary['actual_GPU_used'] and summary['actual_images_generated'] == 3 and summary['total_actual_UNet_NFE'] == 12
    assert summary['component_parameters_trained'] == 0 and not summary['optimizer_created']
    assert summary['native_IP_layers'] == 70 and summary['total_IP_calls'] == 840
    assert not summary['D_V_test_tensors_or_scores_read']
    for row in summary['cases']:
        assert sha(row['generated_path']) == row['generated_sha256']
    return dict(status='ONE_SHOT_THREE_IMAGE_T_PIXEL_REPLAY_COMPLETE',
        job_receipt_path=str(job_path), job_receipt_sha256=sha(job_path),
        summary_path=str(summary_path), summary_sha256=sha(summary_path),
        actual_images_generated=3, fitted_parameters=0, image_quality_gain_established=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--check-registration-only', action='store_true')
    args = parser.parse_args()
    registration_sha = sha(args.registration)
    registration = read(args.registration)
    checked = validate_registration(registration)
    if args.check_registration_only:
        print(json.dumps(dict(status='CPU_STATIC_WAIT_REGISTRATION_CHECK_ONLY', facts=checked, GPU_launched=False)), flush=True)
        return
    import fcntl
    with Path(registration['queue_lock_path']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not Path(STATE).exists() and not Path(LOG).exists(), 'Preserve a previous observer; no implicit restart'
        no_existing_probe_artifacts(registration)
        started = time.monotonic()
        base = dict(pid=os.getpid(), registration_path=str(args.registration), registration_sha256=registration_sha,
            interval_seconds=PERIOD, maximum_wait_seconds=MAX_WAIT, launch_attempts=0,
            queue_GPU_forward_calls=0, formal_training_enabled=False, full_V_generation_enabled=False,
            fixed_input_identity=checked, original_processes_modified=False, automatic_retry=False)
        def record(status, **fields):
            value = save_state(STATE, dict(base, status=status, **fields))
            # State includes authoritative full PID evidence; the log stays small.
            item = {key: value[key] for key in ('UTC', 'status', 'pid', 'launch_attempts')}
            if 'observation' in fields:
                item['wait_reason'] = fields['observation']['decision']['reason']
            with Path(LOG).open('a', buffering=1) as log:
                log.write(json.dumps(item) + '\n')
            print(json.dumps(item), flush=True)
        record('WAITING_CPU_ONLY_FOR_ORIGINAL_MAIN_AND_T32')
        while time.monotonic() - started < MAX_WAIT:
            try:
                assert sha(args.registration) == registration_sha, 'Static wait registration changed'
                validate_registration(registration)
                no_existing_probe_artifacts(registration)
            except BaseException as error:
                record('STOPPED_STATIC_DRIFT_OR_EXISTING_ARTIFACT_PRESERVED', error=repr(error))
                return 1
            try:
                observation = observe(registration['probe_template'])
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError, ValueError, json.JSONDecodeError) as error:
                record('WAITING_TRANSIENT_OBSERVATION_ERROR_NO_RESTART', observation_error=repr(error))
                time.sleep(min(PERIOD, max(0, MAX_WAIT - (time.monotonic() - started))))
                continue
            except AssertionError as error:
                record('STOPPED_OBSERVATION_CONTRACT_FAILURE_NO_RESTART', error=repr(error))
                return 1
            if not observation['decision']['ready']:
                record('WAITING_CPU_ONLY_FOR_ORIGINAL_MAIN_AND_T32', observation=observation)
                time.sleep(min(PERIOD, max(0, MAX_WAIT - (time.monotonic() - started))))
                continue
            try:
                # Recheck all static evidence just before the irreversible
                # once-only config/launch boundary. Nothing old is restarted.
                assert sha(args.registration) == registration_sha
                validate_registration(registration)
                config, preflight = create_reviewed_config(registration)
                base['launch_attempts'] = 1
                argv = [sys.executable, str(ROOT / 'run_gpu_job_v2.py'), '--name', config['job_name'],
                    '--max-seconds', '1200', '--', sys.executable, '-u', str(ROOT / PROBE),
                    '--config', registration['reviewed_config_path']]
                record('LAUNCHING_ONCE_THREE_T_IMAGES_NO_FIT', observation=observation,
                    reviewed_config_sha256=sha(registration['reviewed_config_path']),
                    reviewed_terminal_evidence=preflight['terminals'], command=argv)
                process = subprocess.Popen(argv, cwd=ROOT)
                record('ONE_SHOT_PIXEL_PROBE_LAUNCHED', wrapper_pid=process.pid,
                    reviewed_config_sha256=sha(registration['reviewed_config_path']))
                try:
                    code = process.wait(timeout=1260)
                except subprocess.TimeoutExpired:
                    record('PROBE_WRAPPER_TIMEOUT_PRESERVED_NO_RETRY', wrapper_pid=process.pid,
                        old_jobs_not_killed=True, probe_not_restarted=True)
                    return 1
                result = inspect_probe_result(registration, code)
                status = result.pop('status')
                record(status, **result)
                return 0 if status == 'ONE_SHOT_THREE_IMAGE_T_PIXEL_REPLAY_COMPLETE' else 1
            except BaseException as error:
                record('ONE_SHOT_BOUNDARY_OR_PROBE_FAILED_PRESERVED_NO_RETRY', error=repr(error))
                return 1
        record('STOPPED_FINITE_SEVEN_DAY_WAIT_NO_AUTOMATIC_RESTART')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
