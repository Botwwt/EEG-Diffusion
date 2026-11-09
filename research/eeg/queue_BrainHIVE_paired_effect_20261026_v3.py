"""Finite pipeline v3: adopt the live O job; preserve its original deadline."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import brainhive_paired_study_contract_20261025_v1 as C
import queue_BrainHIVE_decisive_T_analysis_20261025_v2 as Q
from brainhive_paired_supervision_20261026_v3 import safe_bytes_used,validate_wrapper_exit,AdoptedProcess

# Pipeline-only accounting fix. GPU children keep their unchanged modules and
# single-writer counting; this supervisor must tolerate their atomic renames.
C.bytes_used=safe_bytes_used


def close_GPU_receipt(receipt,spec,wrapper_pid,wrapper_code):
    assert receipt['wrapper_pid']==wrapper_pid and receipt['name']==spec['name']
    assert receipt['command']==spec['command']
    assert not Q.live(receipt['pid']) and not Q.live(receipt['wrapper_pid'])
    assert receipt['status']==('COMPLETED' if receipt['exit_code']==0 else 'FAILED_OR_JOB_TIMEOUT_RESUMABLE')
    validate_wrapper_exit(receipt['exit_code'],wrapper_code)
    return dict(child_exit_code=receipt['exit_code'],wrapper_exit_code=wrapper_code)


def adoption(config):
    plan=C.bound(config['live_O_takeover'])
    assert plan['status']=='REGISTERED_LIVE_O_CONTROL_ONLY_TAKEOVER'
    assert plan['study_core_sha256']==C.core(config)
    old=C.bound(plan['original_registration']);C.validate(old)
    assert C.core(old)==C.core(config) and old['core_sources']==config['core_sources']
    expected=[sys.executable,'-B','-u',str(C.ROOT/'train_BrainHIVE_paired_effect_20261025_v1.py'),
        '--registration',old['registration_path'],'--arm','O','--resume']
    assert plan['job']['command']==expected and plan['job']['name']==config['training_job_names']['O'][1]
    assert plan['job']['max_seconds']==86400
    process=AdoptedProcess(Path(plan['job_path']),plan['job']['name'],expected,
        plan['job']['wrapper_pid'],plan['job']['pid'])
    assert process.identity_evidence==plan['process_identity'], 'Process identity changed before handover'
    assert process.poll() is None and process.elapsed_seconds<plan['job']['max_seconds']
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    assert {int(v) for v in active.splitlines() if v.strip()}=={plan['job']['pid']}
    return plan,process


def command_spec(config,arm,stage,segment=0):
    registration=config['registration_path'];root=C.DATA/'arms'/str(arm)
    train='train_BrainHIVE_paired_effect_20261025_v1.py'
    gen='generate_BrainHIVE_paired_effect_20261025_v1.py'
    cpu='evaluate_BrainHIVE_paired_effect_CPU_20261025_v2.py'
    if stage=='train':
        name=config['training_job_names'][arm][segment];seconds=86400;gpu=True
        args=[train,'--registration',registration,'--arm',arm]+(['--resume'] if segment else [])
        done=root/'train/completion.json';expected='PAIRED_EFFECT_FIXED_FINAL_FULL_T_COMPLETE'
    elif stage in ('postfit','generate'):
        split='T32' if stage=='postfit' else 'V'
        name=config['generation_job_names'][arm][split][segment];seconds=1200 if split=='T32' else 86400;gpu=True
        args=[gen,'--registration',registration,'--arm',arm,'--split',split,'--segment',str(segment)]
        done=root/('postfit_T32' if split=='T32' else 'images')/'completion.json';expected='PAIRED_GENERATION_COMPLETE'
    elif stage=='features':
        name='BrainHIVE_paired_effect_v1_'+arm+'_V_features_20261025';seconds=14400;gpu=True
        args=['run_BrainHIVE_paired_feature_guard_20261025_v2.py','--registration',registration,'--arm',arm]
        done=root/'features/completion.json';expected='COMPLETED'
    elif stage=='functional':
        name=config['functional_job_name'];seconds=7200;gpu=True
        args=['diagnose_BrainHIVE_paired_postfit_T80_20261025_v2.py','--registration',registration]
        done=C.DATA/'functional_T80/summary.json';expected='COMPLETE_FIXED_POSTFIT_PAIRED_T80'
    else:
        assert stage in ('audit','score','compare')
        name='BrainHIVE_paired_effect_v1_'+str(arm)+'_'+stage+'_20261025';seconds=14400 if stage=='score' else 1800;gpu=False
        args=[cpu,'--registration',registration,'--stage',stage]+(['--arm',arm] if arm else [])
        done=(C.DATA/'comparison/summary.json' if stage=='compare' else root/('artifact' if stage=='audit' else 'evaluation')/'completion.json')
        expected={'audit':'FULL16600_PAIRED_ARTIFACT_AND_EEG_IDENTITY_AUDIT_COMPLETE',
            'score':'FULL16600_REAL_EVALUATOR_INVOCATIONS_CLOSED','compare':'FOUR_FULL16600_PAIRED_EFFECT_COMPARISONS_COMPLETE'}[stage]
    return dict(name=name,maximum_seconds=seconds,GPU=gpu,
        command=[sys.executable,'-B','-u',str(C.ROOT/args[0]),*args[1:]],completion=str(done),expected_status=expected)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registration',required=True,type=Path)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    config=C.read(args.registration);C.validate(config,all_inputs=True);C.require_admission(config)
    C.resources(config)  # Existing training is already running; this starts no fit.
    takeover,adopted=adoption(config)
    retired=C.bound(config['retired_controller'])
    assert retired['status']=='OLD_CONTROLLER_RETIRED_LIVE_TRAINING_PRESERVED'
    assert retired['old_controller_pid']==takeover['old_controller_pid'] and not Q.live(retired['old_controller_pid'])
    specs=[command_spec(config,a,s,int(a=='O' and s=='train')) for a in C.ARMS for s in ('train','postfit','generate','audit','features','score')]
    specs.extend([command_spec(config,None,'functional'),command_spec(config,None,'compare')])
    assert len({s['name'] for s in specs})==26
    if args.preflight_only:
        print(json.dumps(dict(status='FULL_26_STAGE_PAIRED_LIFECYCLE_CPU_PREFLIGHT_PASSED',stages=len(specs),
            study_core_sha256=C.core(config),candidate_limit=13)));return
    import fcntl
    audit=C.ROOT/'audit/paired_effect_study_20261025_v1/reaudit_v3';audit.mkdir(parents=True,exist_ok=True)
    state_path=audit/'queue_state.json';assert not state_path.exists(),'Preserve previous queue; explicit continuation required'
    state=dict(status='PAIRED_EFFECT_FOUR_ARM_LIFECYCLES_RUNNING',pid=os.getpid(),registration=C.ref(args.registration),
        study_core_sha256=C.core(config),candidate_ordinal=13,current_stage=None,child_pid=None,
        completed=[],attempts=[],automatic_new_method=False,metric_driven_branching=False,
        live_O_takeover=config['live_O_takeover'],retired_controller=config['retired_controller'],
        training_scientific_core_unchanged=True,no_new_O_training_job=True)
    def save():state['UTC']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());C.atomic(state_path,state)
    def stop(signum,frame):raise InterruptedError('Controller received signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    with (C.ROOT/'audit/functional_full_lifecycle_20261022_v1/lifecycle_controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save()
        def execute(spec,existing=None):
            C.validate(config);C.require_admission(config)
            if spec['GPU'] and existing is None:Q.gpu_free()
            name=spec['name'];job=C.ROOT/'audit'/(name+'_job.json')
            log_path=Path(takeover['log_path']) if existing else audit/(name+'.log')
            assert not Path(spec['completion']).exists()
            if existing is None:
                assert not log_path.exists()
                if spec['GPU']:assert not job.exists()
            argv=([sys.executable,'-B',str(C.ROOT/'run_gpu_job_v2.py'),'--name',name,'--max-seconds',str(spec['maximum_seconds']),'--',*spec['command']]
                  if spec['GPU'] else spec['command'])
            entry=spec|dict(wrapper_command=argv,log=str(log_path),adopted_existing_job=existing is not None)
            state['attempts'].append(entry)
            state['current_stage']=name;save();process=None;started=time.monotonic()
            env=os.environ.copy()
            for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[key]='2'
            if not spec['GPU']:env['CUDA_VISIBLE_DEVICES']=''
            try:
                if existing is not None:
                    process=existing;entry['identity_evidence']=process.identity_evidence
                    entry['original_deadline_retained']=True
                else:
                    with log_path.open('xb') as stream:
                        process=subprocess.Popen(argv,cwd=C.ROOT,env=env,stdin=subprocess.DEVNULL,
                            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state['child_pid']=entry['wrapper_pid']=process.pid;save()
                while process.poll() is None:
                    elapsed=process.elapsed_seconds if existing is not None else time.monotonic()-started
                    assert elapsed<spec['maximum_seconds']+120
                    assert log_path.stat().st_size<=32<<20
                    C.resources(config);time.sleep(10)
                code=process.returncode;entry['exit_code']=code;entry['seconds']=time.monotonic()-started
                if existing is not None:
                    assert process.closure_valid,'Live job ended without a valid closed receipt'
                    entry['wrapper_exit_code_inferred_from_closed_receipt']=True
                entry['log_sha256']=C.sha(log_path)
                if spec['GPU']:
                    receipt=C.read(job);entry.update(close_GPU_receipt(receipt,spec,process.pid,code));entry['GPU_job']=C.ref(job)
                state['child_pid']=None;save();return code
            except BaseException:
                if process is not None:
                    if existing is not None:
                        try:process.validate_owned_identity_for_cleanup()
                        except BaseException as identity_error:
                            entry['cleanup']=dict(identity_validation_error=repr(identity_error),signals_sent=False,
                                wrapper_live=Q.live(process.pid),child_live=Q.live(process.child_pid))
                            terminal=not entry['cleanup']['wrapper_live'] and not entry['cleanup']['child_live']
                            state['child_pid']=None if terminal else process.pid;state['unresolved_live_child']=not terminal
                            save();raise
                    entry['cleanup']={};terminal=Q.stop_owned(process,spec['GPU'],job,spec['command'],name,entry['cleanup'])
                    state['child_pid']=None if terminal else process.pid;state['unresolved_live_child']=not terminal
                save();raise
        try:
            for arm in C.ARMS:
                for stage in ('train','postfit','generate','audit','features','score'):
                    initial_segment=int(arm=='O' and stage=='train')
                    spec=command_spec(config,arm,stage,initial_segment)
                    if arm=='O' and stage=='train':
                        spec['command']=takeover['job']['command']
                        code=execute(spec,existing=adopted)
                    else:code=execute(spec)
                    if code and stage in ('train','generate'):
                        assert initial_segment==0,'O review continuation already uses the second declared slot; no third fit'
                        root=C.DATA/'arms'/arm/('train' if stage=='train' else 'images')
                        progress=C.read(root/'progress.json')
                        allowed='PAIRED_TRAIN_FINITE_STOP_EXACT_RESUME' if stage=='train' else 'PAIRED_GENERATION_FINITE_STOP_EXACT_PREFIX'
                        assert progress['status']==allowed,'Do not retry engineering/model failures'
                        if stage=='train':assert (root/'resume.pt').exists()
                        spec=command_spec(config,arm,stage,1);code=execute(spec)
                    assert code==0,(arm,stage,'Failure preserved; no cascade')
                    done=C.read(spec['completion']);assert done['status']==spec['expected_status']
                    if stage=='train':assert done['updates']==24807 and done['primary_exposures']==done['paired_exposures']==396900
                    if stage in ('postfit','generate','features','audit','score'):assert done['queries']==(32 if stage=='postfit' else 16600)
                    state['completed'].append(dict(arm=arm,stage=stage,completion=C.ref(spec['completion'])));save()
            spec=command_spec(config,None,'functional');functional_code=execute(spec)
            from evaluate_BrainHIVE_paired_effect_CPU_20261025_v2 import functional_readout
            functional_closure=functional_readout(config)
            if functional_code==0 and functional_closure['closed']:
                state['completed'].append(dict(stage='functional',completion=C.ref(spec['completion'])))
            else:
                state['functional_probe']='FAILED_PRESERVED_UNKNOWN_NO_AUTOMATIC_RETRY'
                failure=C.DATA/'functional_T80/failure.json'
                if not failure.exists():C.atomic(failure,dict(status='FAILED_PRESERVED_UNKNOWN_NO_AUTOMATIC_RETRY',exit_code=functional_code,closure=functional_closure))
            save()
            spec=command_spec(config,None,'compare');assert execute(spec)==0
            assert C.read(spec['completion'])['status']==spec['expected_status']
            state['completed'].append(dict(stage='compare',completion=C.ref(spec['completion'])))
            state.update(status='ALL_FOUR_PAIRED_EFFECT_FULL_T_FULL_V_LIFECYCLES_AND_FIXED_COMPARISON_COMPLETE',current_stage=None);save()
        except BaseException as error:
            state.update(status='PAIRED_EFFECT_QUEUE_FAILED_PRESERVED_NO_ADAPTIVE_RETRY',error=repr(error));save();raise


if __name__=='__main__':main()
