"""Finite, single-parent prerequisite queue; no scientific result is fabricated."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
ASSETS = WORK / 'assets'


def state(**kwargs):
    value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs)
    path = WORK / 'audit/alignment_queue_state.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)


def main():
    started = time.monotonic()
    plan = json.loads((ASSETS / 'transfer_plan.json').read_text())
    required = [ASSETS / a['path'] for a in plan['assets'] if not a['path'].startswith('prior_H14/attn_adapter/')]
    assert len(required) == 14
    state(status='WAITING_FOR_PUBLIC_TRAINING_PAYLOADS', required=14, completed=0, GPU_job_started=False)
    while True:
        completed = sum(p.exists() and Path(str(p) + '.ready.json').exists() for p in required)
        if completed == 14:
            break
        assert time.monotonic() - started < 7200, 'Finite public-transfer wait expired; retain all partial payloads'
        state(status='WAITING_FOR_PUBLIC_TRAINING_PAYLOADS', required=14, completed=completed, GPU_job_started=False)
        time.sleep(30)
    state(status='CPU_DATA_AND_NATIVE_PRIOR_CHECK', GPU_job_started=False)
    command = [sys.executable, str(ROOT / 'prepare_BrainHIVE_actual_inputs_CPU_v1.py')]
    code = subprocess.call(command, cwd=ROOT)
    assert code == 0, 'Data/weight/coordinate precondition failed; do not start training'
    state(status='RUNNING_FULL10_PUBLISHED_PROTOCOL_ALIGNMENT', GPU_job_started=True,
          epochs=25, full_training_stimuli_per_subject=16540, training_seed=1729)
    command = [sys.executable, str(ROOT/'run_gpu_job_v2.py'), '--name', 'BrainHIVE_published_parent_alignment_v1',
               '--max-seconds', '10800', '--', sys.executable, str(ROOT/'train_BrainHIVE_parent_v1.py'), '--role', 'published']
    code = subprocess.call(command, cwd=ROOT)
    assert code == 0, 'Native alignment failed; preserve checkpoints and inspect, no alternate model fallback'
    state(status='FULL10_ALIGNMENT_COMPLETE_ACTUAL_IMAGE_GENERATION_NEXT', GPU_job_started=False,
          actual_images_generated=0, component_gain_established=False, seconds=time.monotonic()-started)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        state(status='PRECONDITION_OR_EXECUTION_FAILED', error=type(exc).__name__ + ':' + str(exc),
              component_gain_established=False)
        raise
