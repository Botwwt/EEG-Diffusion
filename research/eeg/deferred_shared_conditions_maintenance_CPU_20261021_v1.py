"""One finite, CPU-only cache maintenance attempt in the image-audit window.

The GPU lock is probed and immediately released, never held across maintenance.
The exact relocation helper independently rechecks its memory/inode guards.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from relocate_retired_ATM_features_CPU_20261021_v1 import ROOT, GIB, memory, sha, write


WORK = Path('/root/brainhive_reproduction_20261020_v1')
MAIN = WORK / 'audit/main_storage_continuation_20261021_v1.json'
STATE = WORK / 'audit/deferred_shared_conditions_maintenance_20261021_v1.json'
LOG = WORK / 'audit/deferred_shared_conditions_maintenance_20261021_v1.log'
OWN_LOCK = WORK / 'audit/deferred_shared_conditions_maintenance_20261021_v1.lock'
GPU_LOCK = ROOT / 'audit/GPU_JOB.lock'
RECEIPT = ROOT / 'audit/BrainHIVE_retired_shared_RAM_relocation_20261021_v1.json'
HELPER = ROOT / 'relocate_retired_shared_conditions_CPU_20261021_v1.py'
SOURCE = Path('/root/atm_diffusion_20261015/research/eeg_diffusion_next_20261018_v1/shared/conditions_EEG_ONLY.pt')
RAM = Path('/dev/shm/brainhive_retired_shared_20261021_v1')
BYTES = 416810752
SUCCESS = 'BYTE_IDENTICAL_LINK_AND_MAIN_FORECAST_VERIFIED'
AUDIT = 'audit_BrainHIVE_image_arm_CPU_v1.py'
TERMINAL = {
    'ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED',
    'FINITE_MAIN_STORAGE_CONTINUATION_PARTIAL_WITH_EXPLICIT_FAILURES',
    'STORAGE_FORECAST_STOPPED_PRESERVED_NO_CASCADING_JOBS',
    'FAILED_MAIN_STORAGE_CONTINUATION_PRESERVED_NEEDS_DIAGNOSIS',
}


def read(path):
    return json.loads(path.read_text())


def idle_gpu_lock():
    assert GPU_LOCK.is_file(), 'Probe only the existing global lock'
    with GPU_LOCK.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(lock, fcntl.LOCK_UN)
    return True


def receipt_state():
    if not RECEIPT.exists():
        return None
    report = read(RECEIPT)
    if report.get('status') == SUCCESS:
        assert report['original'] == str(SOURCE) and report['bytes'] == BYTES
        assert SOURCE.is_symlink() and SOURCE.exists() and sha(SOURCE) == report['sha256']
    return report['status']


def main():
    assert not STATE.exists() and not LOG.exists(), 'Preserve the maintenance attempt'
    with OWN_LOCK.open('a') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not STATE.exists() and not LOG.exists()
        helper_sha = sha(HELPER)
        dependency = ROOT / 'relocate_retired_ATM_features_CPU_20261021_v1.py'
        dependency_sha = sha(dependency)
        record = dict(pid=os.getpid(), source_sha256=sha(__file__), helper_sha256=helper_sha,
            dependency_sha256=dependency_sha, main_state=str(MAIN), helper=str(HELPER),
            maximum_wait_seconds=14400, poll_seconds=10, maximum_helper_seconds=60,
            GPU_jobs=0, helper_attempts=0, GPU_lock_held_during_helper=False,
            scientific_model_or_protocol_changes=False, RAM_is_persistent_backup=False)
        def save(status, **values):
            record.update(status=status, UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)
            write(STATE, record)
        save('WAITING_FOR_IMAGE_IDENTITY_CPU_WINDOW_AND_MEMORY_HEADROOM')
        started = time.monotonic()
        try:
            while time.monotonic() - started < 14400:
                status = receipt_state()
                if status is not None:
                    save('DONE_EXISTING_BYTE_IDENTICAL_RELOCATION' if status == SUCCESS else
                         'STOPPED_EXISTING_PARTIAL_RELOCATION_RECEIPT_PRESERVED', relocation_status=status)
                    return
                queue = read(MAIN)
                if queue['status'] in TERMINAL:
                    save('NOT_RUN_MAIN_QUEUE_TERMINAL', main_status=queue['status'])
                    return
                proc = Path('/proc') / str(queue['pid']) / 'stat'
                if not proc.exists() or proc.read_text().split()[2] in ['Z', 'X']:
                    save('NOT_RUN_MAIN_CONTROLLER_NO_LONGER_LIVE', main_pid=queue['pid'])
                    return
                mem = memory()
                enough = (mem['maximum'] - mem['current'] >= 4 * GIB + 2 * BYTES + (64 << 20)
                          and mem['shm_free'] >= GIB + BYTES)
                eligible = queue.get('cpu_entry') == AUDIT and enough
                save('WAITING_FOR_IMAGE_IDENTITY_CPU_WINDOW_AND_MEMORY_HEADROOM',
                     main_pid=queue['pid'], main_status=queue['status'], cpu_entry=queue.get('cpu_entry'),
                     memory=mem, memory_eligible=enough)
                if eligible and idle_gpu_lock():
                    # Recheck immediately before launching; the lock remains
                    # released so the main queue's nonblocking wrappers work.
                    fresh = read(MAIN)
                    fresh_mem = memory()
                    if (fresh.get('cpu_entry') != AUDIT or fresh['status'] in TERMINAL
                            or fresh_mem['maximum'] - fresh_mem['current'] < 4 * GIB + 2 * BYTES + (64 << 20)
                            or fresh_mem['shm_free'] < GIB + BYTES or not idle_gpu_lock()):
                        time.sleep(10)
                        continue
                    if 14400 - (time.monotonic() - started) < 60:
                        save('NOT_RUN_FINITE_WAIT_BOUND_NO_FULL_HELPER_WINDOW')
                        return
                    assert sha(HELPER) == helper_sha and sha(dependency) == dependency_sha
                    assert receipt_state() is None
                    before = SOURCE.lstat()
                    save('RUNNING_ONE_CPU_RELOCATION_ATTEMPT', helper_attempts=1, admitted_memory=fresh_mem)
                    try:
                        with LOG.open('x') as log:
                            result = subprocess.run(['/root/miniconda3/bin/python', '-u', str(HELPER)],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=False,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
                        status = receipt_state()
                        if result.returncode == 0 and status == SUCCESS:
                            save('DONE_BYTE_IDENTICAL_RELOCATION_AND_MAIN_FORECAST_VERIFIED',
                                 helper_exit_code=0, relocation_status=status)
                        else:
                            after = SOURCE.lstat()
                            unchanged = (not SOURCE.is_symlink()
                                and (before.st_ino, before.st_size, before.st_mtime_ns) ==
                                    (after.st_ino, after.st_size, after.st_mtime_ns))
                            before_write = status is None and unchanged and not RAM.exists()
                            save('REFUSED_BEFORE_RELOCATION_NO_RETRY' if before_write else
                                 'FAILED_OR_PARTIAL_RELOCATION_PRESERVED_NO_RETRY',
                                 helper_exit_code=result.returncode, relocation_status=status,
                                 refusal_before_relocation_receipt=before_write)
                    except subprocess.TimeoutExpired:
                        save('HELPER_TIMEOUT_PARTIAL_ARTIFACTS_PRESERVED_NO_RETRY', relocation_status=receipt_state())
                    return
                time.sleep(10)
            save('NOT_RUN_FINITE_FOUR_HOUR_WAIT_EXPIRED')
        except Exception as error:
            save('FAILED_CPU_MAINTENANCE_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)[:2000])
            raise


if __name__ == '__main__':
    main()
