"""Single-GPU lock and ledger under the explicit updated user authorization.

The old wrapper and all previous records remain unchanged. Each new job still
has a finite timeout; removing the total time cap never removes experiment gates.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == '--' else args.command
    assert command and '/' not in args.name and math.isfinite(args.max_seconds) and args.max_seconds > 0
    authorization_path = ROOT / 'configs/resource_authorization_20261020_v3.json'
    authorization_bytes = authorization_path.read_bytes()
    auth = json.loads(authorization_bytes)
    assert auth['user_instruction_verbatim'] == '\u4e0d\u7528\u603b\u4e0a\u9650 \u65e0\u9650\u65f6\u95f4'
    assert auth['total_GPU_job_hours_cap'] is None and auth['occupied_time_deadline_epoch'] is None
    assert auth['existing_single_GPU_only'] and not auth['purchase_or_resource_expansion_authorized']
    lock = (ROOT / 'audit/GPU_JOB.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger = ROOT / 'audit/compute_ledger.jsonl'
    used = sum(float(json.loads(line).get('charged_seconds', 0)) for line in ledger.read_text().splitlines())
    receipt = ROOT / 'audit' / (args.name + '_job.json')
    assert not receipt.exists(), 'Use a new receipt for a retry; preserve failed attempts'
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    record = dict(name=args.name, pid=process.pid, wrapper_pid=os.getpid(), command=command,
                  start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  max_seconds=args.max_seconds, charged_seconds_before_job=used,
                  total_hours_cap=None, occupied_time_deadline=None,
                  authorization=str(authorization_path),
                  authorization_sha256=hashlib.sha256(authorization_bytes).hexdigest(), status='RUNNING')
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    try:
        code = process.wait(timeout=args.max_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
        record['termination_reason'] = 'Finite registered job timeout; preserve resumable checkpoints'
    elapsed = time.monotonic() - started
    record.update(status='COMPLETED' if code == 0 else 'FAILED_OR_JOB_TIMEOUT_RESUMABLE',
                  exit_code=code, charged_seconds=elapsed,
                  end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    with ledger.open('a') as file:
        file.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)
    return 0 if code == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
