"""One bounded S0 profiling job after the exact model acquisition completes."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parent
MODEL=Path('/dev/shm/eeg_sota_autoresearch_20261018_v1/sdxl_base_46216598')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model-pid',type=int,required=True)
    args=parser.parse_args();started=time.monotonic()
    def record(state,**data):
        (ROOT/'audit/standard_generator_profile_job_state.json').write_text(json.dumps(
            dict(stage=state,pid=os.getpid(),model_pid=args.model_pid,
                 elapsed_seconds=time.monotonic()-started,**data),indent=2)+'\n')
    while not (MODEL/'completion.json').exists():
        source=Path(f'/proc/{args.model_pid}/cmdline')
        assert source.exists() and b'acquire_sdxl_base.py' in source.read_bytes(), 'Model acquisition terminal without a verified complete snapshot'
        assert time.monotonic()-started<7200, 'Profiling prerequisite wait timeout'
        record('WAITING_ON_LIVE_MODEL_ACQUISITION')
        time.sleep(15)
    record('PROFILING_STARTING')
    command=['/root/miniconda3/bin/python',str(ROOT/'profile_standard_generator.py')]
    with (ROOT/'logs/standard_generator_profile.log').open('wb') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        record('PROFILING',child_pid=child.pid,command=command,timeout_seconds=600)
        try:
            code=child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise RuntimeError('S0 profiler exceeded its 600-second allocation')
    assert code==0,f'Profiler failed with code {code}'
    assert json.loads((ROOT/'audit/standard_generator_profile.json').read_text())['status']=='S0_PROFILE_COMPLETED'
    record('COMPLETED')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        (ROOT/'audit/standard_generator_profile_failure.json').write_text(json.dumps(
            dict(status='FAILED',type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()),indent=2)+'\n')
        raise
