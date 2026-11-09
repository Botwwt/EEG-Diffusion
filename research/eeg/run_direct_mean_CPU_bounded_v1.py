"""One bounded CPU-only execution of the pre-registered channel diagnosis."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temporary, path)


def main():
    plan_path = ROOT/'configs/direct_mean_channel_exploratory_v2.json'
    plan = json.loads(plan_path.read_text())
    registration = json.loads((ROOT/'audit/direct_mean_channel_v2_registration_v1.json').read_text())
    assert sha(plan_path) == registration['plan_sha256']
    assert sha(__file__) == registration['controller_sha256']
    assert plan['CPU_only'] and plan['maximum_attempts'] == 1 and plan['max_seconds'] == 1200
    receipt = ROOT/'audit/direct_mean_channel_v2_CPU_job_v1.json'
    assert not receipt.exists() and not Path(plan['output']).exists()
    command = ['/root/miniconda3/bin/python', str(ROOT/'diagnose_direct_mean_channel_v2.py'),
               '--plan', str(plan_path)]
    log = ROOT/'logs/IDEA1_direct_mean_channel_v2_CPU.log'
    assert not log.exists()
    started = time.time()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with log.open('xb') as stream:
        worker = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                  stdout=stream, stderr=subprocess.STDOUT,
                                  env=environment, start_new_session=True)
        state = dict(status='RUNNING_CPU_ONLY', controller_pid=os.getpid(), worker_pid=worker.pid,
                     command=command, start_epoch=started, max_seconds=1200, output=plan['output'],
                     log=str(log), plan_sha256=sha(plan_path), GPU_used=False,
                     maximum_attempts=1, may_launch_GPU_training=False, scientific_goal_complete=False)
        write(receipt, state)
        try:
            code = worker.wait(timeout=1200)
            state['status'] = 'COMPLETED_CPU_DIAGNOSIS' if code == 0 else 'FAILED_CPU_DIAGNOSIS'
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid, signal.SIGTERM)
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
            code = worker.returncode
            state['status'] = 'CPU_TIME_CAP_REACHED_PARTIALS_PRESERVED'
    state.update(exit_code=code, end_epoch=time.time(), seconds=time.time()-started)
    summary = Path(plan['output'])/'summary.json'
    if summary.exists():
        state['summary_sha256'] = sha(summary)
    write(receipt, state)
    with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:
        file.write(json.dumps(dict(event_id='direct_mean_channel_v2_CPU_scientific_attempt_v1',
                   receipt=str(receipt), receipt_sha256=sha(receipt), seconds=state['seconds'],
                   status=state['status'], GPU_used=False, scientific_goal_complete=False))+'\n')


if __name__ == '__main__':
    main()
