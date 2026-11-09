"""Launch the single diagnostic queue, never an additional component fit."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')


def main():
    from diagnose_BrainHIVE_native_function_T32_v1 import check_registered_input_contract
    path = ROOT / 'configs/BrainHIVE_native_function_T32_v1.json'
    registration = json.loads(path.read_text())
    check_registered_input_contract(registration)
    state_path = WORK / 'audit/native_function_T32_queue_v1.json'
    launch_path = WORK / 'audit/native_function_T32_queue_launch_v1.json'
    log_path = WORK / 'logs/native_function_T32_queue_v1.log'
    assert not state_path.exists() and not launch_path.exists() and not log_path.exists()
    old = json.loads((WORK / 'audit/two_ideas_queue_v1.json').read_text())
    assert old['status'] == 'WITHDRAWN_BEFORE_NEW_TRAINING_PENDING_MECHANISM_REVIEW'
    command = ['/root/miniconda3/bin/python', '-u',
               str(ROOT / 'queue_BrainHIVE_native_function_T32_v1.py'), '--registration', str(path)]
    with log_path.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(50):
        assert process.poll() is None, log_path.read_text()[-2000:]
        if state_path.exists():
            state = json.loads(state_path.read_text())
            assert state['pid'] == process.pid
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Diagnostic queue did not produce verifiable live state')
    os.kill(process.pid, 0)
    receipt = dict(status='ONE_FROZEN_T_DIAGNOSTIC_QUEUE_LIVE',
                   utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   pid=process.pid, command=command, state=state,
                   KAN_training_enabled=False, response_rank_training_enabled=False,
                   existing_main_interrupted=False)
    launch_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
