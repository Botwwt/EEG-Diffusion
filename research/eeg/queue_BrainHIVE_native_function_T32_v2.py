"""Same frozen diagnostic, after the repaired main lifecycle; preserve v1."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
PYTHON = '/root/miniconda3/bin/python'
STATE = WORK / 'audit/native_function_T32_queue_v2.json'
OWNS_STATE = False
LOCK = None


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def state(status, **fields):
    pending = STATE.with_name(STATE.name + '.pending')
    pending.write_text(json.dumps(dict(status=status,
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pid=os.getpid(),
        component_training_enabled=False, KAN_enabled=False, **fields), indent=2) + '\n')
    pending.replace(STATE)


def gpu_idle():
    with (ROOT / 'audit/GPU_JOB.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def main():
    global OWNS_STATE, LOCK
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    LOCK = (WORK / 'audit/native_function_T32_queue_v2.lock').open('a')
    fcntl.flock(LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not STATE.exists(), 'Preserve diagnostic attempts'
    registration = read(args.registration)
    assert registration['diagnostic'] == 'native_function_T32_v2'
    assert registration['component_training_enabled'] is False and registration['KAN_enabled'] is False
    for name, expected in registration['queue_source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    registration_sha = sha(args.registration)
    assert 0 < registration['diagnostic_job_seconds'] <= 1200
    assert 0 < registration['maximum_queue_wait_seconds'] <= 72 * 3600
    previous_path = WORK / 'audit/main_storage_continuation_20261021_v1.json'
    terminal = {
        'ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED',
        'FINITE_MAIN_STORAGE_CONTINUATION_PARTIAL_WITH_EXPLICIT_FAILURES',
        'STORAGE_FORECAST_STOPPED_PRESERVED_NO_CASCADING_JOBS',
        'FAILED_MAIN_STORAGE_CONTINUATION_PRESERVED_NEEDS_DIAGNOSIS',
    }
    OWNS_STATE = True
    started = time.monotonic()
    while True:
        assert sha(args.registration) == registration_sha
        assert time.monotonic() - started < registration['maximum_queue_wait_seconds']
        previous = read(previous_path) if previous_path.exists() else {'status': 'CONTINUATION_NOT_STARTED'}
        if previous['status'] in terminal and gpu_idle():
            break
        state('WAITING_FOR_REPAIRED_MAIN_LIFECYCLE_AND_EXCLUSIVE_GPU',
              previous_main_status=previous['status'], next_check_seconds=600,
              formal_component_updates=0, generated_images=0,
              diagnostic_registration_sha256=registration_sha)
        time.sleep(600)
    assert shutil.disk_usage(WORK).free >= 128 << 20
    name = 'BrainHIVE_native_function_T32_v2'
    assert not (ROOT / 'audit' / (name + '_job.json')).exists()
    assert not Path(registration['output_directory']).exists()
    command = [PYTHON, str(ROOT / 'run_gpu_job_v2.py'), '--name', name,
        '--max-seconds', str(registration['diagnostic_job_seconds']), '--', PYTHON, '-u',
        str(ROOT / 'diagnose_BrainHIVE_native_function_T32_v2.py'), '--registration', str(args.registration)]
    state('RUNNING_ONE_FROZEN_T_ONLY_DIAGNOSTIC', argv=command)
    log = WORK / 'logs/native_function_T32_v2.log'
    with log.open('x') as stream:
        code = subprocess.call(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    completion_path = Path(registration['output_directory']) / 'completion.json'
    if code == 0 and completion_path.exists():
        completion = read(completion_path)
        state('FROZEN_FUNCTIONAL_DIAGNOSTIC_SAVED_NO_FORMAL_COMPONENT_TRAINING',
            completion_path=str(completion_path), completion_sha256=sha(completion_path),
            diagnostic_status=completion['status'], generated_images=0,
            next_action='Review measurements; no automatic component fit.')
    else:
        state('DIAGNOSTIC_FAILED_WITH_PRESERVED_LOG_NO_AUTOMATIC_RETRY', returncode=code, log=str(log))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OWNS_STATE:
            state('STOPPED_WITH_EXPLICIT_DIAGNOSTIC_FAILURE', error_type=type(error).__name__, error=str(error)[:1800])
        raise
