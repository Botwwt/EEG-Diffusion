"""One finite, preregistered OOF preparation/run after complete references.

No retries, controls, interface training or benchmark unlock. The existing
preparer/driver retain every scientific, source and complete-scope budget gate.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parent
PY='/root/miniconda3/bin/python'
STATE=ROOT/'audit/native_oof_stage_queue_state.json'


def load(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def write(value):
    temporary=STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,STATE)


def check_sources(plan):
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,('Locked queue source changed',name)
    policy=load(ROOT/plan['policy'])
    for name,digest in policy['source_sha256'].items():assert sha(ROOT/name)==digest,('Locked OOF source changed',name)


def prerequisites():
    post=ROOT/'audit/S1_postprocessing_state.json'
    probe=ROOT/'audit/projector_cost_probe_queue_state.json'
    ready=True
    for state,launch,script,terminal in [
        (post,'audit/S1_postprocessing_launch.json','queue_reference_postprocessing.py',
         ['COMPLETED_BOTH_FULL_REFERENCES_POSTPROCESSED']),
        (probe,'audit/projector_cost_probe_queue_launch.json','queue_projector_cost_probe.py',
         ['COST_PROBE_COMPLETED_FORMAL_ARMS_STILL_REQUIRE_BUDGET_GATE','FAILED_OR_BUDGET_GATE_STOPPED_COST_PROBE'])]:
        data=load(state) if state.exists() else {}
        if data.get('status') in terminal:continue
        assert not data.get('status','').startswith('FAILED'),data
        pid=load(ROOT/launch)['pid'];command=Path('/proc')/str(pid)/'cmdline'
        assert command.exists() and str(ROOT/script).encode() in command.read_bytes(),('Prerequisite controller stopped',script,pid)
        ready=False
    return ready


def main():
    plan_path=ROOT/'configs/native_oof_stage_queue_v1.json';plan=load(plan_path);check_sources(plan)
    assert plan['preparation_attempts']==1 and not plan['S4_or_final_authorized']
    deadline=load(ROOT/'configs/compute_time_bounds.json')['conservative_occupied_time_deadline_epoch']
    base=dict(controller_pid=os.getpid(),command=[PY,str(Path(__file__).resolve())],
        launch_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),plan_sha256=sha(plan_path),
        GPU_used_by_wait=False,interface_training_started=False,final_test_unlocked=False)
    write(base|dict(status='WAITING_COMPLETE_REFERENCES_CPU_AUDIT_AND_BOUNDED_COST_PROBE'))
    while True:
        assert not (ROOT/'STOP_OOF_STAGE_QUEUE').exists(),'Explicit queue stop file'
        assert time.time()+600<deadline,'Not enough remaining time even for preparation and preservation'
        if prerequisites():break
        time.sleep(30)
    check_sources(plan)
    # Exactly one preparation attempt. An unfunded decision is retained, not
    # converted into an undertrained teacher or a scientific negative result.
    command=[PY,str(ROOT/'prepare_native_oof_stage.py'),'--policy',str(ROOT/plan['policy'])]
    log=ROOT/'logs/native_oof_stage_queue_prepare.log'
    with log.open('xb') as file:
        result=subprocess.run(command,cwd=ROOT,stdout=file,stderr=subprocess.STDOUT,timeout=180)
    assert result.returncode==0,('Preparation did not complete',str(log),result.returncode)
    result=load(log);decision=ROOT/'audit/native_oof_stage_queue_preparation_result.json'
    with decision.open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    if result['status']=='NOT_RUN_COMPLETE_S3_PLUS_FULL_CONTROLS_FINAL_NOT_FUNDED':
        write(base|dict(status='COMPLETE_BUDGET_DECISION_NO_OOF_STARTED',preparation_result=str(decision),
            gate_path=result['gate_path'],hypothesis_rejected=False));return
    assert result['status']=='RESOLVED_AND_FUNDED_BUT_NOT_STARTED'
    resolved=Path(result['resolved_path']);assert sha(resolved)==result['resolved_sha256']
    command=[PY,str(ROOT/'run_native_oof_stage.py'),'--resolved',str(resolved)]
    log=ROOT/'logs/native_oof_stage_queue_driver.log';check_sources(plan)
    assert not (ROOT/'STOP_OOF_STAGE_QUEUE').exists()
    with log.open('xb') as file:
        child=subprocess.Popen(command,cwd=ROOT,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
        write(base|dict(status='RUNNING_FUNDED_NATIVE_OOF_DRIVER',pid=child.pid,command=command,log=str(log),
            preparation_result=str(decision),resolved=str(resolved)))
        code=child.wait()
    assert code==0,('OOF driver failed; preserve outputs and inspect before dependent work',str(log),code)
    write(base|dict(status='COMPLETE_FUNDED_NATIVE_OOF_DRIVER_REVIEW_REQUIRED',preparation_result=str(decision),
        resolved=str(resolved),driver_log=str(log),driver_exit_code=code,
        CPU_channel_validation_is_not_image_gain=True,S4_may_start=False,final_test_may_unlock=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        previous=load(STATE) if STATE.exists() else {}
        write(previous|dict(status='FAILED_OR_STOPPED_NO_DEPENDENT_STAGE_STARTED',error=repr(error)))
        raise
