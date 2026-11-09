"""One180-second T-only GPU cost probe after both full reference evaluations."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parent
STATE=ROOT/'audit/projector_cost_probe_queue_state.json'


def write(value):
    STATE.write_text(json.dumps(value,indent=2)+'\n')


def main():
    config=json.loads((ROOT/'configs/projector_cost_probe_queue.json').read_text())
    launch=json.loads((ROOT/'audit/S1_queue_launch_batchsafe_v3.json').read_text())
    deadline=json.loads((ROOT/'configs/compute_time_bounds.json').read_text())['conservative_occupied_time_deadline_epoch']
    write(dict(status='WAITING_BOTH_COMPLETE_REFERENCE_EVALUATIONS',pid=os.getpid(),training_started=False))
    while True:
        prerequisite=json.loads((ROOT/'audit/S1_queue_state.json').read_text())
        if prerequisite['status']=='COMPLETED_TWO_FULL_REFERENCES':break
        assert prerequisite['status']!='FAILED_STOPPED_DEPENDENT_JOBS',prerequisite
        assert time.time()+225<deadline,'Insufficient campaign time even for the bounded cost probe'
        process=Path('/proc')/str(launch['pid'])/'cmdline'
        assert process.exists() and str(ROOT/'queue_reference_stage1.py').encode() in process.read_bytes(),'Registered reference queue stopped'
        time.sleep(30)
    for name,expected in config['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,(name,'Frozen probe source changed')
    command=['/root/miniconda3/bin/python',str(ROOT/'run_gpu_job.py'),'--name','S3_native_projector_cost_probe','--max-seconds','180','--',
        '/root/miniconda3/bin/python',str(ROOT/'profile_native_projector_cost.py')]
    log=ROOT/'logs/S3_native_projector_cost_probe.log';assert not log.exists()
    with log.open('xb') as file:
        child=subprocess.Popen(command,cwd=ROOT,stdout=file,stderr=subprocess.STDOUT)
        write(dict(status='RUNNING_ONE_BOUNDED_T_COST_PROBE',pid=child.pid,parent_pid=os.getpid(),command=command,log=str(log)))
        code=child.wait()
    assert code==0,f'Cost probe failed; inspect {log}; do not launch dependent training'
    write(dict(status='COST_PROBE_COMPLETED_FORMAL_ARMS_STILL_REQUIRE_BUDGET_GATE',formal_training_started=False,
        full_OOF_started=False,adapter_arms_started=False,final_test_unlocked=False))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        old=json.loads(STATE.read_text()) if STATE.exists() else {}
        write(old|dict(status='FAILED_OR_BUDGET_GATE_STOPPED_COST_PROBE',error=str(exc)));raise
