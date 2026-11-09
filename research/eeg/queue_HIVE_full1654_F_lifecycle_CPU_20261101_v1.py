"""Two complete fits, all4000 images, fixed all16-metric comparison; finite."""
import argparse,datetime,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import brainhive_full1654_F_study_contract_20261101_v1 as G
from brainhive_owned_process_CPU_20261029_v1 import capture_wrapper_child,stop_owned

def run(path):
    s=G.setup(path);G.resources(s,training_start=True);root=Path(s['output_root']);assert not root.exists();root.mkdir();logs=root/'queue';logs.mkdir();code=Path(path).parent;R=Path(s['repository_root'])
    start=time.monotonic();events=[];owner=None;pinned=None;phase='INIT'
    def call(label,args,gpu=False,seconds=1800,allow_timeout=False):
        nonlocal owner,pinned,phase
        phase=label;assert time.monotonic()-start<21600;G.resources(s,training_start=label.startswith('fit_'))
        command=[sys.executable,'-B']+list(map(str,args));env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0' if gpu else '')
        name='HIVE_full1654_F_20261101_v2_'+label
        if gpu:
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==G.GPU
            command=[sys.executable,'-B',str(R/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--']+command;seconds=900
        with (logs/(label+'.stdout.log')).open('xb') as so,(logs/(label+'.stderr.log')).open('xb') as se:
            tick=time.monotonic();owner=subprocess.Popen(command,stdout=so,stderr=se,stdin=subprocess.DEVNULL,cwd=R,env=env,start_new_session=True)
            pinned=capture_wrapper_child(owner,R/'audit'/(name+'_job.json')) if gpu else None
            fields=Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()
            G.write(root/'progress.json',dict(PID=os.getpid(),phase=phase,active_PID=owner.pid,active_start_ticks=int(fields[19]),GPU=gpu,job_name=name,completed_tasks=len(events),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()),replace=True)
            rc=owner.wait(timeout=seconds+65)
        event=dict(label=label,command=command,GPU=gpu,exit_code=rc,seconds=time.monotonic()-tick)
        if gpu:
            receipt=G.read(R/'audit'/(name+'_job.json'));assert receipt['wrapper_pid']==owner.pid and receipt['exit_code']==rc
            event['GPU_receipt']=G.ref(R/'audit'/(name+'_job.json'))
        owner=None;pinned=None;events.append(event);G.write(logs/(label+'.json'),event)
        if rc and allow_timeout:
            assert gpu and receipt.get('termination_reason')=='Finite registered job timeout; preserve resumable checkpoints'
            return False
        assert rc==0
        return True
    def guarded(arm,script,args,label):
        return [code/'run_HIVE_R80_evaluation_guarded_20261029_v1.py','--execution',code/(arm+'_evaluation_execution.json'),'--audit-file',root/arm/'evaluation_resources'/(label+'.json'),'--script',script,'--']+args
    try:
        for arm in G.ARMS:
            call('fit_'+arm,[code/'train_HIVE_full1654_F_heads_GPU_20261101_v1.py','--study',path,'--arm',arm],True)
            call('fit_audit_'+arm,[code/'audit_HIVE_full1654_F_heads_CPU_20261101_v1.py','--study',path,'--arm',arm])
        for arm in G.ARMS:call('native_profile_'+arm,[code/'run_HIVE_full1654_F_R80_GPU_20261101_v1.py','--study',path,'--arm',arm,'--mode','profile'],True)
        # All declared images precede any new method scoring.
        for arm in G.ARMS:
            for lo in range(0,2000,200):call(f'images_{arm}_{lo:04d}',[code/'run_HIVE_full1654_F_R80_GPU_20261101_v1.py','--study',path,'--arm',arm,'--mode','generate','--lo',lo],True)
        for arm in G.ARMS:call('artifact_'+arm,[code/'audit_HIVE_full1654_R80_CPU_20261101_v1.py','--study',path,'--arm',arm])
        for arm in G.ARMS:
            out=root/arm;spec=dict(repository_root=s['repository_root'],output_root=str(out),source_pins=s['source_pins'],gallery_manifest=s['evaluation']['gallery_manifest'])
            G.write(code/(arm+'_evaluation_protocol.json'),spec)
            e=dict(protocol=G.ref(code/(arm+'_evaluation_protocol.json')),source_pins=s['source_pins']);G.write(code/(arm+'_evaluation_execution.json'),e)
            manifest=out/'loss_control/manifest.jsonl';features=out/'evaluation/features'
            for part in range(3):
                label=f'features_{arm}_{part}'
                if call(label,guarded(arm,R/'extract_image_features.py',['--manifest',manifest,'--output',features],label),True,allow_timeout=True):break
                for p in features.rglob('*.tmp'):
                    assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(features.resolve());p.rename(p.with_name(p.name+f'.part{part}_incomplete_retained'))
            else:raise RuntimeError('Fixed three feature timeout segments exhausted; preserve full denominator')
            assert G.read(features/'completion.json')['queries']==2000
            gallery=s['evaluation']['gallery_manifest']['path'];gallery_features=s['evaluation']['gallery_features']
            for label,script,args in [('common_'+arm,R/'score_full_manifest.py',['--manifest',manifest,'--features',features,'--gallery-manifest',gallery,'--gallery-features',gallery_features,'--output',out/'evaluation/scores']),('low_'+arm,R/'score_low_level.py',['--manifest',manifest,'--gallery-manifest',gallery,'--output',out/'evaluation/low_level'])]:
                call(label,guarded(arm,script,args,label))
            for part in range(3):
                label=f'author_{arm}_{part}'
                if call(label,guarded(arm,code/'score_HIVE_loss_control_R80_author_GPU_20261030_v1.py',['--execution',code/(arm+'_evaluation_execution.json')],label),True,allow_timeout=True):break
            else:raise RuntimeError('Fixed three author metric timeout segments exhausted')
        call('compare_all16',[code/'compare_HIVE_full1654_F_R80_CPU_20261101_v1.py','--study',path])
        G.write(root/'completion.json',dict(status='COMPLETE_FULL1654_TWO_F_FITS_4000_R80_IMAGES_ALL16_FIXED_COMPARISONS',study=G.ref(path),events=events,training_seed=1729,new_head_updates=62028,new_EEG_exposures=992400,new_research_images=4000,reference_native_images_reused=2000,seconds=time.monotonic()-start,official200_previously_exposed=True,independent_H=False,NEW_contacted=False,SOTA=False,CCF_A=False))
    except BaseException as error:
        if owner is not None:stop_owned(owner,pinned)
        G.write(root/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,no_automatic_retry=True));raise

if __name__=='__main__':
    def stop(*unused):raise InterruptedError('Stop only this owned full1654 lifecycle')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);run(p.parse_args().study)
