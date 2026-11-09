"""Run exactly the four pre-registered two-update engineering profiles, once.

Each child uses the existing exclusive GPU wrapper. Failure retains all files
and stops this finite queue. There is no formal-training or generation branch.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'audit/functional_profiles_preparation_20261022_v1'
STATE = BASE / 'queue_state.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(value):
    value.update(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    temp = STATE.with_suffix('.pending')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(STATE)


def main():
    with (BASE / 'profile_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not STATE.exists(), 'No repeated profile queue or automatic retries'
        preparation = json.loads((BASE / 'completion.json').read_text())
        assert preparation['status'] == 'ALL_FOUR_ENGINEERING_ONLY_RECIPES_ACTUALLY_BOUND_AND_CHECKED'
        assert [r['arm'] for r in preparation['recipes']] == ['O', 'P', 'PN', 'S']
        assert preparation['maximum_profile_jobs'] == 4 and not preparation['formal_training_enabled']
        state = dict(status='RUNNING_FIXED_FOUR_ENGINEERING_PROFILES', pid=__import__('os').getpid(),
            source_sha256=sha(__file__), preparation_sha256=sha(BASE / 'completion.json'),
            completed=[], attempts=[], formal_training_enabled=False,
            new_images=0, maximum_seconds_per_job=1800, maximum_total_seconds=7200)
        save(state)
        started = time.monotonic()
        for item in preparation['recipes']:
            assert sha(item['path']) == item['sha256']
            recipe = json.loads(Path(item['path']).read_text())
            assert not Path(recipe['profile_output']).exists()
            job = ROOT / 'audit' / (recipe['profile_job_name'] + '_job.json')
            assert not job.exists(), 'Preserve previous job attempts'
            argv = [sys.executable, '-B', str(ROOT / 'run_gpu_job_v2.py'),
                    '--name', recipe['profile_job_name'], '--max-seconds', '1800', '--',
                    sys.executable, '-B', '-u', str(ROOT / 'train_BrainHIVE_functional_alignment_v2.py'),
                    '--recipe', item['path'], '--profile-only']
            log = BASE / (item['arm'] + '_wrapper.log')
            with log.open('x') as stream:
                child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                         cwd=ROOT, start_new_session=True)
                state.update(current_arm=item['arm'], wrapper_pid=child.pid)
                state['attempts'].append(dict(arm=item['arm'], argv=argv, log=str(log)))
                save(state)
                while child.poll() is None:
                    if time.monotonic() - started > 7200 or log.stat().st_size > 16 << 20:
                        child.terminate()
                        state.update(status='STOPPED_FINITE_ENGINEERING_QUEUE_BOUND', exit_code=None)
                        save(state)
                        return 2
                    time.sleep(2)
            status = json.loads(job.read_text()) if job.exists() else {}
            if child.returncode != 0 or status.get('status') != 'COMPLETED':
                state.update(status='ENGINEERING_PROFILE_FAILED_PRESERVED_NO_AUTOMATIC_RETRY',
                             exit_code=child.returncode, job_status=status.get('status'), log_sha256=sha(log))
                save(state)
                return 2
            output = Path(recipe['profile_output'])
            result = json.loads(output.read_text())
            assert result['status'] == 'ACTUAL_PARENT_Q_TWO_FORWARD_ENGINEERING_PROFILE_COMPLETE'
            assert result['arm'] == item['arm'] and result['updates'] == 2 and result['engineering_checks_passed']
            state['completed'].append(dict(arm=item['arm'], path=str(output), sha256=sha(output),
                job_path=str(job), job_sha256=sha(job), log_sha256=sha(log)))
            save(state)
        state.update(status='ALL_FOUR_ACTUAL_ENGINEERING_PROFILES_COMPLETE_NO_FORMAL_FIT',
                     current_arm=None, wrapper_pid=None, seconds=time.monotonic() - started)
        save(state)
        print(json.dumps(state))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
