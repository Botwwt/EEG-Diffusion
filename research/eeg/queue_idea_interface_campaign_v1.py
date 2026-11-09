"""Finite idea-first execution; arm release needs actual complete-scope funding.

No score-based continuation, scope reduction, retry, extra seed or final unlock.
Missing controls remain required even when the total40h cannot fund them.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
from prepare_idea_interface_arm_v1 import ROOT,load,sha,remaining,phase_done
from run_idea_interface_arm_v1 import write
from run_native_oof_stage import stop_group

PY='/root/miniconda3/bin/python'
TERMINAL_PREPARATION={
    'PREPARATION_COMPLETE_UNET_NOT_RUN_CPU_PREREQUISITE_FAILED',
    'PREPARATION_COMPLETE_UNET_NOT_RUN_NUMERICAL_STAGE_INCOMPLETE',
    'IMAGE_PREPARATION_TERMINAL_REVIEW_ACTUAL_RECEIPTS'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args()
    plan_path=Path(args.plan);plan=load(plan_path)
    assert plan['version']=='idea_interface_finite_campaign_v1'
    policy=load(plan['policy_path']);assert sha(plan['policy_path'])==plan['policy_sha256']
    assert plan['training_priority']==policy['prospective_arm_priority']
    assert plan['funding_rounds']==2 and not plan['scientific_goal_complete']
    state_path=ROOT/'audit/idea_interface_campaign_state_v1.json'
    assert not state_path.exists(),'Preserve previous controller state; no automatic restart'
    state=dict(status='WAITING_FOR_ACTUAL_PREPARATION',controller_pid=os.getpid(),plan_sha256=sha(plan_path),
        events=[],scientific_goal_complete=False,final_test_unlocked=False)
    write(state_path,state);process=None

    def guard():
        assert not (ROOT/'STOP_IDEA_INTERFACE_CAMPAIGN').exists(),'Explicit campaign stop'
        assert not (ROOT/'configs/FINAL_EVAL_UNLOCK.json').exists()
        assert remaining()>900,'Fixed total40h reserve reached'
        for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name

    def run(command,name,timeout):
        nonlocal process
        guard();log=ROOT/'logs'/f'{name}.log';assert not log.exists()
        tick=time.monotonic()
        with log.open('xb') as file:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='RUNNING',phase=name,child_pid=process.pid,log=str(log));write(state_path,state)
            while process.poll() is None:
                if (ROOT/'STOP_IDEA_INTERFACE_CAMPAIGN').exists() or time.monotonic()-tick>timeout:
                    # Arm controller owns GPU process groups and must perform
                    # graceful registered stop/reconciliation before any kill.
                    (ROOT/'STOP_IDEA_INTERFACE_ARM').touch(exist_ok=True)
                    try:process.wait(timeout=70)
                    except subprocess.TimeoutExpired:stop_group(process)
                    raise RuntimeError('Preserved stopped controller; inspect owned GPU receipts before resuming')
                time.sleep(5)
            code=process.wait()
        event=dict(phase=name,exit_code=code,seconds=time.monotonic()-tick,log=str(log))
        state['events'].append(event);write(state_path,state)
        return code,log

    try:
        while True:
            guard();aux=load(ROOT/'audit/idea_image_preparation_queue_state_v1.json')
            if aux['status'] in TERMINAL_PREPARATION:break
            pid=aux.get('controller_pid');assert pid and Path('/proc',str(pid)).exists(),('Preparation did not finish normally',aux)
            state.update(status='WAITING_FOR_ACTUAL_PREPARATION',preparation_phase=aux.get('phase'),
                preparation_state_sha256=sha(ROOT/'audit/idea_image_preparation_queue_state_v1.json'))
            write(state_path,state);time.sleep(20)
        for round_index in range(plan['funding_rounds']):
            for arm in plan['training_priority']:
                guard();resolved=ROOT/'configs'/f'idea_interface_{arm}_resolved_v1.json'
                if not resolved.exists():
                    code,log=run([PY,str(ROOT/'prepare_idea_interface_arm_v1.py'),'--arm',arm,'--policy',plan['policy_path']],
                        f'IDEA1_campaign_round{round_index}_{arm}_funding',120)
                    assert code in [0,2],('Actual preparer failed; no automatic retries',str(log))
                    decision=json.loads(log.read_text().strip().splitlines()[-1])
                    if code==2:assert decision['status']=='PREREQUISITES_PENDING_NO_ARM_RELEASE_CREATED'
                    state['events'].append(dict(arm=arm,round=round_index,funding_decision=decision));write(state_path,state)
                if not resolved.exists():continue
                resolved_data=load(resolved)
                if phase_done(resolved_data,'T_readiness'):continue
                arm_state=ROOT/'audit'/f'idea_interface_{arm}_state_v1.json'
                assert not arm_state.exists(),'Existing incomplete training requires failure audit, not automatic retry'
                code,log=run([PY,str(ROOT/'run_idea_interface_arm_v1.py'),'--resolved',str(resolved),'--through','T_readiness'],
                    f'IDEA1_campaign_round{round_index}_{arm}_training',sum(resolved_data['phase_caps_seconds'].values())+300)
                assert code==0,('Actual adapter training failed',str(log))
                assert load(arm_state)['status']=='PLANNED_PHASE_BOUNDARY_NEXT_PHASES_STILL_REQUIRED'
            # The candidate receives the earliest full images after matched
            # training; both reference populations have already been reserved.
            for arm in plan['image_priority']:
                guard();resolved=ROOT/'configs'/f'idea_interface_{arm}_resolved_v1.json'
                if not resolved.exists():continue
                resolved_data=load(resolved)
                if phase_done(resolved_data,'report'):continue
                command=[PY,str(ROOT/'run_idea_interface_arm_v1.py'),'--resolved',str(resolved),'--through','report']
                arm_state=ROOT/'audit'/f'idea_interface_{arm}_state_v1.json'
                if arm_state.exists():
                    assert load(arm_state)['status']=='PLANNED_PHASE_BOUNDARY_NEXT_PHASES_STILL_REQUIRED'
                    command.append('--resume-planned-phases')
                code,log=run(command,f'IDEA1_campaign_round{round_index}_{arm}_full_images',sum(resolved_data['phase_caps_seconds'].values())+300)
                assert code==0,('Actual full-image lifecycle failed',str(log))
                assert load(arm_state)['status']=='COMPLETED_FULL_ARM_IMAGES_METRICS_AND_IDENTITY'
        completed=[];missing=[]
        for arm in policy['required_interface_arms']+['Frozen']:
            path=ROOT/'configs'/f'idea_interface_{arm}_resolved_v1.json'
            (completed if path.exists() and phase_done(load(path),'report') else missing).append(arm)
        # Refresh matched deltas now that the shared references may be complete.
        if completed:
            code,log=run([PY,str(ROOT/'write_idea_interface_report_v1.py'),'--arm',completed[0]],'IDEA1_campaign_final_available_comparison',300)
            assert code==0,str(log)
        state.update(status='TERMINAL_REQUIRED_CONTROLS_MISSING' if missing else 'TERMINAL_IMAGE_ARMS_COMPLETE_EEG_CONTROLS_AND_CONFIRMATION_REQUIRED',
            completed_full_image_arms=completed,missing_required_image_arms=missing,
            scientific_goal_complete=False,independent_mechanism_confirmed=False,automatic_retry=False)
        write(state_path,state)
    except Exception as error:
        state.update(status='STOPPED_REQUIRES_ACTUAL_FAILURE_OR_RESOURCE_REVIEW',error=repr(error),
            scientific_goal_complete=False,hypothesis_rejected=False,automatic_retry=False)
        write(state_path,state);raise


if __name__=='__main__':main()
