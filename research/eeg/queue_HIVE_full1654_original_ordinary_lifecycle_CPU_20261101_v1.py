"""Future fully reviewed original-ordinary fit,2000 images,all16 metrics and fixed5 comparisons."""
import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import brainhive_full1654_ordinary_fit_contract_CPU_20261101_v1 as F
from brainhive_gpu_exit_contract_CPU_20261101_v1 import verify as verify_exit
E=F.E;K=F.K


def run(protocol):
    s=F.setup_fit(protocol);K.resources(s,starting=True)
    from brainhive_owned_process_CPU_20261029_v1 import capture_wrapper_child,stop_owned
    root=Path(s['output_root']);assert not root.exists();root.mkdir();logs=root/'queue';logs.mkdir()
    code=Path(protocol).parent;R=Path(s['repository_root']);events=[];owner=pinned=None;phase='INIT';started=time.monotonic()
    def call(label,args,gpu=False,seconds=1800,allow_timeout=False,training_start=False):
        nonlocal owner,pinned,phase
        phase=label;assert time.monotonic()-started<172800
        K.resources(s,starting=training_start)
        name='HIVE_full1654_original_ordinary_20261101_v1_'+label
        command=[sys.executable,'-B']+list(map(str,args))
        if gpu:
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==K.GPU
            command=[sys.executable,'-B',str(R/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--']+command;seconds=900
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        tick=time.monotonic()
        with (logs/(label+'.stdout.log')).open('xb') as stdout,(logs/(label+'.stderr.log')).open('xb') as stderr:
            owner=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,cwd=R,env=env,start_new_session=True)
            pinned=capture_wrapper_child(owner,R/'audit'/(name+'_job.json')) if gpu else None
            fields=Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()
            F.atomic_metadata(root/'progress.json',dict(PID=os.getpid(),phase=phase,active_PID=owner.pid,
                active_start_ticks=int(fields[19]),GPU=gpu,job_name=name,completed_tasks=len(events),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            rc=owner.wait(timeout=seconds+65)
        event=dict(label=label,command=command,exit_code=rc,GPU=gpu,seconds=time.monotonic()-tick)
        if gpu:
            receipt=K.read(R/'audit'/(name+'_job.json'))
            event.update(verify_exit(rc,receipt,owner.pid))
            event['GPU_receipt']=K.ref(R/'audit'/(name+'_job.json'))
        owner=pinned=None;events.append(event);K.write(logs/(label+'.json'),event)
        if rc and allow_timeout:
            assert receipt.get('termination_reason')=='Finite registered job timeout; preserve resumable checkpoints'
            return False
        assert rc==0;return True
    try:
        for segment in range(128):
            if (root/'ordinary_full1654/fit_completion.json').exists():break
            call(f'fit_{segment:03d}',[code/'train_HIVE_full1654_original_ordinary_GPU_20261101_v1.py','--protocol',protocol,'--segment',segment],True,training_start=True)
            call(f'latest_CPU_{segment:03d}',[code/'audit_HIVE_full1654_original_ordinary_fit_CPU_20261101_v1.py','--protocol',protocol,'--latest-only'],seconds=90)
        assert (root/'ordinary_full1654/fit_completion.json').exists(),'Finite128 planned segments exhausted; no scientific retry'
        call('full_fit_CPU',[code/'audit_HIVE_full1654_original_ordinary_fit_CPU_20261101_v1.py','--protocol',protocol])
        call('native_profile',[code/'run_HIVE_full1654_original_ordinary_R80_GPU_20261101_v1.py','--protocol',protocol,'--mode','profile'],True)
        for lo in range(0,2000,200):
            call(f'images_{lo:04d}',[code/'run_HIVE_full1654_original_ordinary_R80_GPU_20261101_v1.py','--protocol',protocol,'--mode','generate','--lo',lo],True)
        call('artifact_CPU',[code/'audit_HIVE_full1654_original_ordinary_R80_CPU_20261101_v1.py','--protocol',protocol])
        arm=root/'ordinary_full1654'
        evaluation=dict(repository_root=s['repository_root'],output_root=str(arm),source_pins=s['source_pins'],gallery_manifest=s['evaluation']['gallery_manifest'])
        K.write(code/'evaluation_protocol.json',evaluation)
        K.write(code/'evaluation_execution.json',dict(protocol=K.ref(code/'evaluation_protocol.json'),source_pins=s['source_pins']))
        execution=code/'evaluation_execution.json';manifest=arm/'loss_control/manifest.jsonl';features=arm/'evaluation/features'
        def guarded(script,args,label):
            return [code/'run_HIVE_R80_evaluation_guarded_20261029_v1.py','--execution',execution,
                '--audit-file',arm/'evaluation_resources'/(label+'.json'),'--script',script,'--']+list(args)
        for part in range(3):
            label='features_'+str(part)
            if call(label,guarded(R/'extract_image_features.py',['--manifest',manifest,'--output',features],label),True,allow_timeout=True):break
            for path in features.rglob('*.tmp'):
                assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(features.resolve())
                path.rename(path.with_name(path.name+f'.part{part}_incomplete_retained'))
        else:raise RuntimeError('Fixed three feature timeout continuations exhausted')
        assert K.read(features/'completion.json')['queries']==2000
        gallery=s['evaluation']['gallery_manifest']['path'];gfeatures=s['evaluation']['gallery_features']
        for label,script,args in [('common',R/'score_full_manifest.py',['--manifest',manifest,'--features',features,'--gallery-manifest',gallery,'--gallery-features',gfeatures,'--output',arm/'evaluation/scores']),
                                  ('low',R/'score_low_level.py',['--manifest',manifest,'--gallery-manifest',gallery,'--output',arm/'evaluation/low_level'])]:
            call(label,guarded(script,args,label))
        for part in range(3):
            label='author_'+str(part)
            if call(label,guarded(code/'score_HIVE_loss_control_R80_author_GPU_20261030_v1.py',['--execution',execution],label),True,allow_timeout=True):break
        else:raise RuntimeError('Fixed three author timeout continuations exhausted')
        author=K.read(arm/'evaluation/original_paper_metrics/completion.json')
        assert author['queries']==2000 and K.read(arm/'evaluation/scores/summary.json')['queries']==2000
        low=K.read(arm/'evaluation/low_level/summary.json')
        assert low['status']=='COMPLETED' and low['expected_images']==low['verified_and_decoded']==2000
        K.write(arm/'evaluation/evaluation_closure.json',dict(status='COMPLETE_ACTUAL_NEW_ORDINARY_ALL16_SCORE_CALLS_AND_FILES',
            manifest=K.ref(manifest),feature_config=K.ref(features/'config.json'),feature_completion=K.ref(features/'completion.json'),
            score_CSVs=dict(common=K.ref(arm/'evaluation/scores/per_query.csv'),paper=K.ref(arm/'evaluation/original_paper_metrics/per_query.csv')),
            author_completion=K.ref(arm/'evaluation/original_paper_metrics/completion.json'),
            actual_score_call_receipts=[K.ref(logs/(label+'.json')) for label in ('common','low')]+[K.ref(logs/('author_'+str(part)+'.json'))],
            queries=2000,all16_metrics_included=True))
        call('all4_fixed5',[code/'compare_HIVE_full1654_all4_methods_CPU_20261101_v1.py','--protocol',protocol])
        K.write(root/'completion.json',dict(status='COMPLETE_FULL1654_ORIGINAL_ORDINARY_FIT_2000_R80_ALL16_AND_FIXED5',
            protocol=K.ref(protocol),events=events,head_updates=31014,EEG_exposures=496200,new_research_images=2000,
            cached_reference_images_reused=6000,seconds=time.monotonic()-started,training_seed=1729,
            official200_previously_exposed=True,independent_H=False,NEW_contacted=False,SOTA=False,CCF_A=False))
    except BaseException as error:
        if owner is not None:stop_owned(owner,pinned)
        K.write(root/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,no_automatic_retry=True));raise


if __name__=='__main__':
    def stop(*unused):raise InterruptedError('Stop only this owned full1654 ordinary lifecycle')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);run(parser.parse_args().protocol)
