"""Finite read-only 10-minute monitoring; never changes training or reads scores."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parent
WORK=Path('/root/brainhive_reproduction_20261020_v1')


def read(path):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def live(pid):
    try:os.kill(int(pid),0);return True
    except (ProcessLookupError,TypeError,ValueError):return False


def main():
    launch=read(ROOT/'audit/BrainHIVE_bounded_full_lifecycle_launch_v17.json')
    assert launch and launch['pid']
    log=WORK/'audit/bounded_10min_monitor_v1.jsonl'
    assert not log.exists()
    started=time.monotonic()
    with log.open('x',buffering=1) as file:
        while time.monotonic()-started<72*3600:
            state=read(WORK/'audit/bounded_full_lifecycle_queue_v1.json')
            job=read(ROOT/'audit'/(state['job']+'_job.json')) if state and state.get('job') else None
            record=dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                controller_live=live(launch['pid']),controller_state=state,active_job=job,
                disk_free={str(p):shutil.disk_usage(p).free for p in [WORK,Path('/root/autodl-tmp')]},
                memory_current=Path('/sys/fs/cgroup/memory.current').read_text().strip(),
                memory_events=Path('/sys/fs/cgroup/memory.events').read_text().strip(),
                GPU_status=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
                    '--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip(),
                model_or_hyperparameter_mutations=False,V_or_test_scores_read=False)
            record['component_training']={arm:read(WORK/'components/bounded_v1'/arm/'progress.json')
                for arm in ['risk','ordinary']}
            file.write(json.dumps(record)+'\n')
            tmp=WORK/'audit/bounded_10min_monitor_latest_v1.json.pending'
            tmp.write_text(json.dumps(record,indent=2)+'\n')
            tmp.replace(WORK/'audit/bounded_10min_monitor_latest_v1.json')
            terminal=state and state.get('status') in [
                'ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED',
                'FINITE_MAIN_COMPARISON_PARTIAL_WITH_EXPLICIT_FAILURES',
                'FAILED_PRESERVED_NEEDS_EXPLICIT_DIAGNOSIS']
            if terminal or not record['controller_live']:return
            for _ in range(20):time.sleep(30)


if __name__=='__main__':main()
