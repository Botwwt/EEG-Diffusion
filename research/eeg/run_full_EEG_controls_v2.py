"""Two full matched EEG interventions, only after an actual stage release.

New attempts/state are separate from the preserved waiting-only v1 controller.
No OOF retry, interface training, final unlock or new selection is performed.
"""
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import time

from run_native_oof_stage import ROOT,PY,load,sha,fresh_remaining,stop_group,recover_gpu_receipt
from prepare_EEG_controls_stage_v2 import check_sources,reserve_seconds

DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
STATE=ROOT/'audit/S1_EEG_control_queue_v2_state.json'
STOP=ROOT/'STOP_FULL_EEG_CONTROLS_V2'


def write(value):
    value['last_update_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    temporary=STATE.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,STATE)
    # The existing protected status reader uses this current-state pointer.
    # Its original deferred bytes are archived once before the first v2 write.
    pointer=ROOT/'audit/S1_EEG_control_queue_state.json';temporary=pointer.with_suffix('.v2.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,pointer)


def run_phase(state,policy,name,seconds,script,*arguments,gpu=True):
    check_sources(policy);assert not STOP.exists(),'Explicit controls stop file'
    remaining=fresh_remaining()
    available=remaining-policy['final_reserve_seconds']-policy['global_margin_seconds']
    assert available>90,'Preserve final evaluation and cleanup margin before another control phase'
    phase_cap=min(seconds+90,available)
    log=ROOT/'logs'/f'{name}.log';assert not log.exists(),('No automatic retry or overwrite',str(log))
    command=[PY,str(ROOT/script),*map(str,arguments)]
    if gpu:command=[PY,str(ROOT/'run_gpu_job.py'),'--name',name,'--max-seconds',str(seconds),'--',*command]
    environment=os.environ.copy()
    if not gpu:environment.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    tick=time.monotonic();process=None;interrupted=False
    try:
        with log.open('xb') as file:
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='RUNNING_FULL_CONTROL_PHASE',stage=name,pid=process.pid,command=command,log=str(log),GPU_used=gpu,
                remaining_seconds_at_start=remaining,phase_cap_with_final_reserve_seconds=phase_cap)
            write(state)
            while process.poll() is None:
                if STOP.exists() or time.monotonic()-tick>=phase_cap:
                    interrupted=True;stop_group(process);break
                time.sleep(5)
            code=process.wait()
    finally:
        if process is not None:
            if process.poll() is None:interrupted=True;stop_group(process)
            elapsed=time.monotonic()-tick;code=process.returncode
            if gpu and interrupted:recover_gpu_receipt(name,code,elapsed)
            if not gpu:
                with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:
                    file.write(json.dumps(dict(stage=name,command=command,seconds=elapsed,exit_code=code,
                        GPU_used=False,controller_interrupted=interrupted,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))+'\n')
    assert code==0 and not interrupted,('Phase incomplete; preserve all outputs',name,code,interrupted)
    state.setdefault('completed_phases',[]).append(name);state.update(status='PHASE_COMPLETED',stage=name);write(state)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',required=True);args=parser.parse_args()
    release_path=Path(args.release).resolve();assert release_path==ROOT/'audit/EEG_controls_stage_release_v2.json'
    release=load(release_path);policy_path=Path(release['policy_path']);policy=load(policy_path)
    check_sources(policy);assert sha(policy_path)==release['policy_sha256']
    assert release['source_sha256']==sha(ROOT/'prepare_EEG_controls_stage_v2.py')
    assert release['status']=='CONTROL_STAGE_RELEASED_AFTER_ACTUAL_S3_S4_REVIEW'
    assert release['controls_may_proceed_to_fresh_complete_scope_budget_gate'] and not release['final_test_may_unlock']
    selection=ROOT/'audit/reference_control_selection.json';assert sha(selection)==release['reference_selection_sha256']
    method=load(selection)['selected_reference'];assert method==release['selected_reference'] and method in ['REF_REG','REF_PRIOR_V']
    for name in ['measured_generation_proof','measured_evaluation_proof']:
        proof=release[name];assert sha(proof['path'])==proof['sha256']
    assert not STATE.exists(),'Inspect saved v2 state before a separately reviewed restart'
    assert not (ROOT/'configs/FINAL_EVAL_UNLOCK.json').exists()
    old=ROOT/'audit/S1_EEG_control_queue_state.json'
    archive=None
    if old.exists():
        assert load(old)['status'].startswith('DEFERRED_'),'The original v1 controls must remain unstarted'
        archive=ROOT/'audit/source_snapshots/S1_EEG_control_queue_state.before_v2.json'
        archive.parent.mkdir(exist_ok=True)
        with archive.open('xb') as file:file.write(old.read_bytes())
    state=dict(status='PREPARED_FULL_CONTROLS_NOT_STARTED',controller_pid=os.getpid(),release_path=str(release_path),
        release_sha256=sha(release_path),policy_sha256=sha(policy_path),selected_reference=method,completed_phases=[],
        complete_queries_per_arm=16600,controls=['permuted','missing'],automatic_retry=False,
        new_training_started=False,final_test_unlocked=False,final_test_may_unlock=False,
        original_deferred_state_preserved=str(archive) if archive else None)
    write(state)
    image_base=DATA/'images/S1_full_native_batchsafe_v2/V';evaluation=DATA/'evaluation/V'
    gallery=DATA/'evaluation/gallery/V';correct=f'{method}_correct_R4'
    try:
        for i,condition in enumerate(['permuted','missing']):
            check_sources(policy);assert not STOP.exists()
            remaining=fresh_remaining();required=reserve_seconds(release['measured_complete_correct_arm_seconds'],2-i,policy)
            gate=dict(status='COMPLETE_REMAINING_CONTROL_SCOPE_BUDGET_CHECKED',selected_reference=method,
                controls_remaining=2-i,remaining_seconds=remaining,required_seconds=required,passed=remaining>=required,
                OOF_reservation_seconds=0,OOF_actual_status=release['S3_actual_status'],
                final_reserve_seconds=policy['final_reserve_seconds'],global_margin_seconds=policy['global_margin_seconds'],
                measured_correct_arm_seconds=release['measured_complete_correct_arm_seconds'],
                policy_sha256=sha(policy_path),release_sha256=sha(release_path),
                UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            path=ROOT/'audit'/f'EEG_controls_v2_budget_{2-i}_remaining.json'
            with path.open('x') as file:json.dump(gate,file,indent=2);file.write('\n')
            if not gate['passed']:
                state.update(status='NOT_RUN_REMAINING_FULL_CONTROLS_BUDGET',gate_path=str(path),hypothesis_rejected=False)
                write(state);return
            arm=f'{method}_{condition}_R4';images=image_base/arm;features=evaluation/arm/'features'
            assert not images.exists() and not features.exists(),'No automatic resume of a previous image attempt'
            run_phase(state,policy,f'S1_V_generate_{method}_{condition}_v2',18000,'generate_reference_images.py',
                '--method',method,'--condition',condition)
            run_phase(state,policy,f'S5_V_evaluate_{method}_{condition}_v2',10800,'extract_image_features.py',
                '--manifest',images/'manifest.jsonl','--output',features)
            run_phase(state,policy,f'S5_V_score_{method}_{condition}_v2',600,'score_full_manifest.py',
                '--manifest',images/'manifest.jsonl','--features',features,'--gallery-manifest',gallery/'manifest.jsonl',
                '--gallery-features',gallery/'features','--output',features.parent/'metrics',gpu=False)
            run_phase(state,policy,f'S5_V_low_level_{method}_{condition}_v2',3600,'score_low_level.py',
                '--manifest',images/'manifest.jsonl','--gallery-manifest',gallery/'manifest.jsonl',
                '--output',features.parent/'low_level',gpu=False)
            run_phase(state,policy,f'S5_V_paired_{method}_{condition}_v2',600,'paired_analysis.py',
                '--reference-manifest',image_base/correct/'manifest.jsonl','--candidate-manifest',images/'manifest.jsonl',
                '--reference-metrics',evaluation/correct/'metrics/per_query.csv','--candidate-metrics',features.parent/'metrics/per_query.csv',
                '--output',DATA/'evaluation/paired'/f'V_{method}_{condition}_minus_correct_v2',gpu=False)
        arguments=[]
        for arm in ['REF_REG_correct_R4','REF_PRIOR_V_correct_R4',f'{method}_permuted_R4',f'{method}_missing_R4']:
            arguments+=['--arm',arm,image_base/arm/'manifest.jsonl',evaluation/arm/'metrics/per_query.csv']
        out=DATA/'galleries/V_references_and_EEG_controls_batchsafe_v2'
        run_phase(state,policy,'S5_V_full_references_controls_gallery_v2',3600,'build_offline_gallery.py',
            *arguments,'--output',out,gpu=False)
        state.update(status='COMPLETED_FULL_V_EEG_CONTROLS',EEG_control_arms=2,full_query_count_per_arm=16600,
            gallery=str(out/'index.html'),paired_difference_direction='intervention minus correct',
            hypothesis_rejected=False,final_test_may_unlock=False)
        write(state)
    except BaseException as error:
        state.update(status='FAILED_OR_STOPPED_FULL_CONTROL_STAGE_OUTPUTS_PRESERVED',error=repr(error),hypothesis_rejected=False)
        write(state);raise


if __name__=='__main__':
    def terminate(signum,frame):raise SystemExit('Explicit supervisor signal '+str(signum))
    signal.signal(signal.SIGTERM,terminate)
    main()
