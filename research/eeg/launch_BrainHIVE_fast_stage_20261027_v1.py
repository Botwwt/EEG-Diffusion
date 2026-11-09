"""Launch exactly one predeclared, finite fast-development stage.

There is no automatic candidate search, retry, D resume, or next-stage launch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--stage', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    assert plan['D_stays_stopped'] is True
    for name, digest in plan['sources'].items():
        assert sha(ROOT / name) == digest, name
    stage = plan['stages'][args.stage]
    assert 0 < stage['max_seconds'] <= 1800
    assert stage['command'][0] == sys.executable
    assert stage['command'][3].startswith(str(ROOT) + '/')
    assert not (ROOT / 'audit' / (stage['name'] + '_job.json')).exists()
    log = Path(stage['log'])
    assert not log.exists(), 'Preserve previous attempts'
    processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not processes, processes
    for pid in (163400, 175778, 175779):
        assert not Path('/proc/' + str(pid)).exists(), 'Withdrawn D process unexpectedly alive'
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-B', '-u', str(ROOT / 'run_gpu_job_v2.py'),
               '--name', stage['name'], '--max-seconds', str(stage['max_seconds']), '--',
               *stage['command']]
    with log.open('xb') as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(status='FINITE_STAGE_WRAPPER_LAUNCHED', pid=process.pid,
                  plan=str(args.plan.resolve()), plan_sha256=sha(args.plan),
                  stage=args.stage, command=command, log=str(log),
                  D_stays_stopped=True, automatic_next_stage=False)
    receipt = ROOT / 'audit' / (stage['name'] + '_launch.json')
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
