"""Finite source/CPU-mean dependent native control lifecycle, OLD only."""
import argparse
import datetime
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import brainhive_training_mean_R80_contract_CPU_20261101_v1 as M


def atomic_progress(value):
    path=M.OUT/('progress.pending.'+str(os.getpid()))
    with path.open('x') as stream:__import__('json').dump(value,stream,indent=2);stream.write('\n')
    os.replace(path,M.OUT/'progress.json')


def run(protocol):
    s=M.validate(M.read(protocol));assert Path(protocol)==M.CODE/'protocol.json'
    for pin in s['source_pins']+s['inherited_source_pins']:M.checked(pin)
    review=M.read(M.CODE/'actual_conditional_execution_review.json')
    assert review['status']=='PASS_SOURCE_BOUND_CONDITIONAL_MEAN_CONTROL_LIFECYCLE_REVIEW'
    assert review['protocol']==M.ref(protocol) and review['source_pins']==s['source_pins']
    assert M.OUT.exists() and not (M.OUT/'queue_started.json').exists()
    M.write(M.OUT/'queue_started.json',dict(PID=os.getpid(),protocol=M.ref(protocol),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    spec=importlib.util.spec_from_file_location('mean_R80_owned_process_exact',M.checked(M.OWNED))
    owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
    sys.path.insert(0,str(Path(M.PARENT['path']).parent))
    from brainhive_gpu_exit_contract_CPU_20261101_v1 import verify as verify_exit
    parent=M.read(M.checked(M.PARENT));R=Path(parent['repository_root']);logs=M.OUT/'queue';logs.mkdir()
    owner=pinned=None;events=[];phase='WAIT';started=None
    def call(label,args,gpu=False,seconds=1800,allow_timeout=False):
        nonlocal owner,pinned,phase
        phase=label;assert time.monotonic()-started<s['execution_seconds']
        M.resources(s,starting=gpu)
        name='HIVE_train_mean_R80_20261101_v1_'+label
        command=[sys.executable,'-B']+list(map(str,args))
        if gpu:
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==M.GPU
            command=[sys.executable,'-B',str(R/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--']+command
            seconds=s['per_GPU_seconds']
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        tick=time.monotonic()
        with (logs/(label+'.stdout.log')).open('xb') as stdout,(logs/(label+'.stderr.log')).open('xb') as stderr:
            owner=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,cwd=R,env=env,start_new_session=True)
            pinned=owned.capture_wrapper_child(owner,R/'audit'/(name+'_job.json')) if gpu else None
            identity=M.process(owner.pid,-1)
            atomic_progress(dict(PID=os.getpid(),phase=phase,active_PID=owner.pid,active_start_ticks=identity['start_ticks'],
                GPU=gpu,job_name=name,completed_tasks=len(events),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            while owner.poll() is None:
                assert time.monotonic()-tick<seconds+65 and time.monotonic()-started<s['execution_seconds']
                M.resources(s);time.sleep(2)
            rc=owner.wait()
        event=dict(label=label,command=command,exit_code=rc,GPU=gpu,seconds=time.monotonic()-tick)
        if gpu:
            receipt=M.read(R/'audit'/(name+'_job.json'));event.update(verify_exit(rc,receipt,owner.pid))
            event['GPU_receipt']=M.ref(R/'audit'/(name+'_job.json'))
        owner=pinned=None;events.append(event);M.write(logs/(label+'.json'),event)
        if rc and allow_timeout:
            assert event['timed_out'] and receipt.get('termination_reason')=='Finite registered job timeout; preserve resumable checkpoints'
            return False
        assert rc==0;return True
    try:
        last=None;last_saved=0.
        while True:
            assert time.time()<=s['waiting_deadline_epoch'],'Finite dependency waiting deadline exhausted'
            gate=M.dependency(s);phase=gate['status']
            if phase.startswith('STOP_'):raise RuntimeError(phase)
            if phase=='READY_ACTUAL_CLOSED_SOURCE_AND_TRAIN_MEANS':
                available=M.resources(s)
                gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                if available['effective_CPU_headroom_bytes']>=s['minimum_start_CPU_headroom_bytes'] and not gpu:
                    M.write(M.OUT/'actual_dependency_closure.json',dict(**gate,resources=available,GPU_empty=True))
                    break
                phase='WAIT_RESOURCES_AFTER_BOTH_SOURCE_QUEUES_EXITED'
            if phase!=last or time.monotonic()-last_saved>300:
                atomic_progress(dict(PID=os.getpid(),phase=phase,dependency=gate,new_GPU_jobs=0,
                    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
                last=phase;last_saved=time.monotonic()
            time.sleep(15)
        started=time.monotonic()
        call('actual_sources_CPU',[M.CODE/'preflight_HIVE_training_mean_R80_CPU_20261101_v1.py','--protocol',protocol])
        for role in M.ROLES:
            call(role+'_profile',[M.CODE/'run_HIVE_training_mean_R80_GPU_20261101_v1.py','--protocol',protocol,'--role',role,'--mode','profile'],True)
            for lo in range(0,2000,200):
                call(role+f'_images_{lo:04d}',[M.CODE/'run_HIVE_training_mean_R80_GPU_20261101_v1.py',
                    '--protocol',protocol,'--role',role,'--mode','generate','--lo',lo],True)
            call(role+'_artifact_CPU',[M.CODE/'audit_HIVE_training_mean_R80_CPU_20261101_v1.py','--protocol',protocol,'--role',role])
        # Both complete image artifacts are required before interpreting any new semantic effect.
        for role in M.ROLES:
            root=M.OUT/role
            evaluation=dict(repository_root=str(R),output_root=str(root),source_pins=s['source_pins']+s['inherited_source_pins'],
                            gallery_manifest=parent['evaluation']['gallery_manifest'])
            folder=M.OUT/'evaluation_executions'/role
            M.write(folder/'protocol.json',evaluation)
            M.write(folder/'execution.json',dict(protocol=M.ref(folder/'protocol.json'),source_pins=evaluation['source_pins']))
            execution=folder/'execution.json';manifest=root/'loss_control/manifest.jsonl';features=root/'evaluation/features'
            def guarded(script,args,label):
                return [Path(M.PARENT['path']).parent/'run_HIVE_R80_evaluation_guarded_20261029_v1.py','--execution',execution,
                    '--audit-file',root/'evaluation_resources'/(label+'.json'),'--script',script,'--']+list(args)
            for part in range(s['feature_attempts']):
                label=role+'_features_'+str(part)
                if call(label,guarded(R/'extract_image_features.py',['--manifest',manifest,'--output',features],label),True,allow_timeout=True):break
                for path in features.rglob('*.tmp'):
                    assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(features.resolve())
                    path.rename(path.with_name(path.name+f'.part{part}_incomplete_retained'))
            else:raise RuntimeError('Fixed three feature prefix continuations exhausted')
            assert M.read(features/'completion.json')['queries']==2000
            gallery=parent['evaluation']['gallery_manifest']['path'];gfeatures=parent['evaluation']['gallery_features']
            for suffix,script,args in [('common',R/'score_full_manifest.py',['--manifest',manifest,'--features',features,'--gallery-manifest',gallery,
                    '--gallery-features',gfeatures,'--output',root/'evaluation/scores']),
                    ('low',R/'score_low_level.py',['--manifest',manifest,'--gallery-manifest',gallery,'--output',root/'evaluation/low_level'])]:
                label=role+'_'+suffix;call(label,guarded(script,args,label))
            for part in range(s['author_attempts']):
                label=role+'_author_'+str(part)
                if call(label,guarded(Path(M.PARENT['path']).parent/'score_HIVE_loss_control_R80_author_GPU_20261030_v1.py',
                    ['--execution',execution],label),True,allow_timeout=True):break
            else:raise RuntimeError('Fixed three author prefix continuations exhausted')
            author=M.read(root/'evaluation/original_paper_metrics/completion.json')
            assert author['queries']==2000 and M.read(root/'evaluation/scores/summary.json')['queries']==2000
            low=M.read(root/'evaluation/low_level/summary.json')
            assert low['status']=='COMPLETED' and low['expected_images']==low['verified_and_decoded']==2000
            M.write(root/'evaluation/evaluation_closure.json',dict(status='COMPLETE_TRAIN_MEAN_ACTUAL_ALL16_SCORE_CALLS_AND_FILES',
                protocol=M.ref(protocol),role=role,manifest=M.ref(manifest),feature_config=M.ref(features/'config.json'),
                feature_completion=M.ref(features/'completion.json'),score_CSVs=dict(common=M.ref(root/'evaluation/scores/per_query.csv'),
                    paper=M.ref(root/'evaluation/original_paper_metrics/per_query.csv')),
                author_completion=M.ref(root/'evaluation/original_paper_metrics/completion.json'),
                actual_score_call_receipts=[M.ref(logs/(role+'_'+suffix+'.json')) for suffix in ('common','low')]
                    +[M.ref(logs/(role+'_author_'+str(part)+'.json'))],queries=2000,all16_metrics_included=True))
        call('fixed4_all16_CPU',[M.CODE/'compare_HIVE_training_mean_R80_CPU_20261101_v1.py','--protocol',protocol])
        M.write(M.OUT/'completion.json',dict(status='COMPLETE_BOTH_TRAIN_MEAN_R80_4000_IMAGES_ALL16_FIXED4',protocol=M.ref(protocol),
            events=events,seconds=time.monotonic()-started,new_research_images=4000,new_training_updates=0,
            profile_engineering_images=100,segment_engineering_replays=40,cached_reference_images=6000,
            source_training_cost_inherited=True,mean_constructed_from_training_only=True,
            official200_previously_exposed=True,independent_H=False,NEW_contacted=False,second_method=False,SOTA=False,CCF_A=False))
    except BaseException as error:
        if owner is not None:owned.stop_owned(owner,pinned)
        M.write(M.OUT/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,no_automatic_retry=True));raise


if __name__=='__main__':
    def stop(*unused):raise InterruptedError('Stop only this owned train-mean control lifecycle')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
