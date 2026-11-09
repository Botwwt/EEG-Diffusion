"""Finite real training -> T checks -> all V images -> all metrics/report files.

Accepts an actual funded resolved arm only. Every phase rechecks complete
outstanding commitments and the total cap. No final unlock, seed/model search,
automatic retries, or declaration of campaign success when controls are absent.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
from prepare_idea_interface_arm_v2 import load,sha,remaining,phase_done,storage_reservations
from run_native_oof_stage import stop_group,recover_gpu_receipt

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
PY='/root/miniconda3/bin/python'


def write(path,value):
    value['UTC']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resolved',required=True)
    parser.add_argument('--through',choices=['train','T_readiness','report'],default='report')
    parser.add_argument('--resume-planned-phases',action='store_true');args=parser.parse_args()
    resolved_path=Path(args.resolved);r=load(resolved_path);arm=r['arm']
    assert r['status'] in ['RESOLVED_FUNDED_NOT_STARTED','SHARED_REFERENCE_RESERVED_NOT_STARTED']
    assert r['executor_sha256']==sha(__file__) and r['formal_scope_queries']==16600
    assert r['preparation_implementation_sha256']==sha(ROOT/'prepare_idea_interface_arm_v2.py')
    assert sha(r['policy_path'])==r['policy_sha256'] and sha(r['cache_config_path'])==r['cache_config_sha256']
    assert r['required_all_five_arms_preserved'] and not r['scientific_goal_achieved']
    assert sha(r['recipe_path'])==r['recipe_sha256'] and sha(r['runtime_path'])==r['runtime_sha256']
    assert sha(r['full_training_config_path'])==r['full_training_config_sha256']
    assert sha(r['budget_receipt_path'])==r['budget_receipt_sha256']
    gate=load(r['budget_receipt_path']);assert gate['arm_lifecycle_funded']
    runtime=load(r['runtime_path']);policy=load(r['policy_path'])
    state_path=ROOT/'audit'/f'idea_interface_{arm}_state_v1.json'
    if state_path.exists():
        assert args.resume_planned_phases,'Inspect actual partial outputs before an explicit registered continuation'
        state=load(state_path);assert state['resolved_sha256']==sha(resolved_path)
        assert state['status']=='PLANNED_PHASE_BOUNDARY_NEXT_PHASES_STILL_REQUIRED','Failed phases need explicit failure audit/recovery'
        state['controller_pid']=os.getpid()
    else:
        assert not args.resume_planned_phases
        state=dict(status='PREPARED_NOT_STARTED',arm=arm,controller_pid=os.getpid(),resolved_path=str(resolved_path),
            resolved_sha256=sha(resolved_path),completed_phases=[],phase_receipts=[],full_five_arm_goal_complete=False,
            final_test_unlocked=False,formal_mechanism_increment_established=False)
    write(state_path,state)
    training=Path(r['training_root']);images=Path(r['image_root']);evaluation=Path(r['evaluation_root'])
    gallery=DATA/'evaluation/gallery/V';readiness=Path(r['T_readiness_root'])
    generation_common=['--recipe',r['recipe_path'],'--runtime-config',r['runtime_path'],
        '--full-training-config',r['full_training_config_path'],'--arm',arm]
    if arm!='Frozen':generation_common+=['--adapter-completion',str(training/'completion.json')]
    audit_common=['--manifest',str(images/'manifest.jsonl'),'--runtime-config',r['runtime_path'],
        '--full-training-config',r['full_training_config_path'],'--gallery-manifest',str(gallery/'manifest.jsonl')]
    phases=[]
    if arm!='Frozen':
        phases.append(('train','train_idea_interface_v1.py',['--recipe',r['recipe_path'],'--cache-config',r['cache_config_path'],
            '--validation-output',r['validation_output'],'--budget-receipt',r['budget_receipt_path'],'--output',str(training),'--arm',arm],True))
    phases += [
        ('T_readiness','generate_idea_interface_images_v1.py',generation_common+['--partition','T_readiness','--output',str(readiness)],True),
        ('generate','generate_idea_interface_images_v1.py',generation_common+['--partition','V','--output',str(images)],True),
        ('identity_before','audit_idea_image_pairing_v1.py',audit_common+['--output',str(evaluation/'identity_before')],False),
        ('features','extract_image_features.py',['--manifest',str(images/'manifest.jsonl'),'--output',str(evaluation/'features')],True),
        ('score','score_full_manifest.py',['--manifest',str(images/'manifest.jsonl'),'--features',str(evaluation/'features'),
            '--gallery-manifest',str(gallery/'manifest.jsonl'),'--gallery-features',str(gallery/'features'),'--output',str(evaluation/'metrics')],False),
        ('lowlevel','score_low_level.py',['--manifest',str(images/'manifest.jsonl'),'--gallery-manifest',str(gallery/'manifest.jsonl'),
            '--output',str(evaluation/'low_level')],False),
        ('identity_after','audit_idea_image_pairing_v1.py',audit_common+['--scores',str(evaluation/'metrics/per_query.csv'),
            '--output',str(evaluation/'identity_after')],False),
        ('report','write_idea_interface_report_v1.py',['--arm',arm],False)]
    process=None;name=None;gpu=False;tick=None
    try:
        for phase,script,arguments,gpu in phases:
            assert not (ROOT/'STOP_IDEA_INTERFACE_ARM').exists(),'Explicit arm stop flag'
            assert not (ROOT/'configs/FINAL_EVAL_UNLOCK.json').exists()
            for source,digest in policy['source_sha256'].items():assert sha(ROOT/source)==digest,source
            for source,digest in runtime['source_sha256'].items():assert sha(ROOT/source)==digest,source
            assert sha(r['executor_path'])==r['executor_sha256']
            # Read completion receipts and identities, never image-score values,
            # to decide whether the remaining complete commitments still fit.
            reservations=[]
            configurations={}
            for path in sorted((ROOT/'configs').glob('idea_interface_*_resolved_v1.json')):
                other=load(path)
                configurations[other['arm']]=other
                for stage,cap in other['phase_caps_seconds'].items():
                    if not phase_done(other,stage):reservations.append(dict(arm=other['arm'],phase=stage,seconds=cap))
            required=sum(x['seconds'] for x in reservations)+900;available=remaining()
            disk=storage_reservations(configurations)
            if available<required or disk['available_free_bytes']<disk['required_free_bytes']:
                state.update(status='RESOURCE_STOPPED_BEFORE_NEXT_PHASE',next_phase=phase,
                    required_remaining_seconds=required,actual_remaining_seconds=available,
                    hypothesis_rejected=False,uncompleted_required_arms_retained=True,storage_check=disk)
                write(state_path,state);return
            if phase_done(r,phase):
                if phase not in state['completed_phases']:state['completed_phases'].append(phase)
                write(state_path,state)
                if phase==args.through and phase!='report':
                    state.update(status='PLANNED_PHASE_BOUNDARY_NEXT_PHASES_STILL_REQUIRED',last_completed_phase=phase);write(state_path,state);return
                continue
            cap=r['phase_caps_seconds'][phase];name=f'IDEA1_{arm}_{phase}'
            command=[PY,str(ROOT/script),*arguments]
            if gpu:command=[PY,str(ROOT/'run_gpu_job.py'),'--name',name,'--max-seconds',str(cap),'--',*command]
            log=ROOT/'logs'/f'{name}.log';assert not log.exists()
            env=os.environ.copy()
            if not gpu:env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            tick=time.monotonic()
            with log.open('xb') as file:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='RUNNING',phase=phase,phase_supervisor_pid=process.pid,command=command,log=str(log),
                    GPU_used=gpu,actual_remaining_seconds_at_start=available,complete_outstanding_reservations=reservations)
                write(state_path,state)
                while process.poll() is None:
                    if (ROOT/'STOP_IDEA_INTERFACE_ARM').exists() or time.monotonic()-tick>cap+50:
                        stop_group(process)
                        if gpu:recover_gpu_receipt(name,process.returncode,time.monotonic()-tick)
                        break
                    time.sleep(5)
                code=process.wait()
            event=dict(phase=phase,exit_code=code,seconds=time.monotonic()-tick,GPU_used=gpu,log=str(log))
            state['phase_receipts'].append(event)
            if not gpu:
                with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:file.write(json.dumps(event)+'\n')
            if code!=0:
                if gpu:recover_gpu_receipt(name,code,event['seconds'])
                raise RuntimeError(f'Actual {phase} failed; preserve outputs, no automatic retry: {log}')
            assert phase_done(r,phase),('Process exit alone did not establish complete expected scope',phase)
            state['completed_phases'].append(phase);state.update(status='PHASE_COMPLETED',phase=phase);write(state_path,state)
            if phase==args.through and phase!='report':
                state.update(status='PLANNED_PHASE_BOUNDARY_NEXT_PHASES_STILL_REQUIRED',last_completed_phase=phase);write(state_path,state);return
        state.update(status='COMPLETED_FULL_ARM_IMAGES_METRICS_AND_IDENTITY',actual_formal_queries=16600,
            full_five_arm_goal_complete=False,scientific_comparison_requires_all_required_controls=True,
            next_action='Persist pairwise differences against same-encoder Frozen/RealPaired when complete; continue fixed remaining arm priority')
        write(state_path,state);print(json.dumps(state),flush=True)
    except Exception as error:
        if process is not None and process.poll() is None:
            stop_group(process)
            if gpu:recover_gpu_receipt(name,process.returncode,time.monotonic()-tick)
        state.update(status='FAILED_OR_RESOURCE_STOPPED_WITH_ALL_OUTPUTS_PRESERVED',error=repr(error),
            hypothesis_rejected=False,automatic_retry=False)
        write(state_path,state);raise


if __name__=='__main__':main()
