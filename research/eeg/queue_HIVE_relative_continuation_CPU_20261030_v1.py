"""Single finite OLD-only train/profile/audit/all-four 80->400 queue."""
import argparse,datetime,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_relative_continuation_contract_CPU_20261030_v1 import ARMS,atomic,read,ref,spec
from brainhive_owned_process_CPU_20261029_v1 import capture_wrapper_child,stop_owned

def run(path):
    s=spec(path);root=Path(s['output_root']);root.mkdir(exist_ok=False);logs=root/'queue';logs.mkdir()
    R=Path(s['repository_root']);code=Path(path).parent;started=time.monotonic();phase='PREFLIGHT';events=[];owner=pinned=None
    def budget():
        assert time.monotonic()-started<21600 and shutil.disk_usage(root).free>2<<30
        assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<2<<30
    def call(label,args,GPU=False,seconds=1800):
        nonlocal owner,pinned
        budget();env=dict(os.environ);env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');command=[sys.executable,'-B']+args;job='HIVE_relative_'+label+'_20261030_v1'
        if GPU:
            assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            command=[sys.executable,'-B',str(R/'run_gpu_job_v2.py'),'--name',job,'--max-seconds','900','--']+command;seconds=900
        else:env['CUDA_VISIBLE_DEVICES']=''
        with (logs/(label+'.stdout.log')).open('xb') as so,(logs/(label+'.stderr.log')).open('xb') as se:
            owner=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True)
            pinned=capture_wrapper_child(owner,R/'audit'/(job+'_job.json')) if GPU else None
            f=Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()
            atomic(logs/'progress.json',dict(phase=phase,label=label,own_PID=owner.pid,start_ticks=int(f[19]),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),events=events,NEW_contacted=False))
            rc=owner.wait(timeout=seconds+60);owner=pinned=None
        event=dict(label=label,GPU=GPU,returncode=rc,job=job if GPU else None,seconds_since_start=time.monotonic()-started);events.append(event);atomic(logs/(label+'.receipt.json'),event)
        assert rc==0,(label,rc);budget()
    try:
        assert shutil.disk_usage(root).free>=8<<30
        assert read(code/'actual_CPU_admission_review.json')['status']=='PASS_REAL_INPUT_SCHEMA_SOURCE_RESOURCE_AND_FIXED_POLICY_CPU_REVIEW'
        phase='FOUR_ARM_ACTUAL_GRADIENT_PROFILE_TEMPORARY_WEIGHTS_DISCARDED'
        call('fit_profile',[str(code/'train_HIVE_relative_continuation_GPU_20261030_v1.py'),'--study',str(path),'--mode','profile'],True)
        for arm in ARMS:
            phase='FIXED_THREE_ADDITIONAL_T_PASSES_'+arm
            call('fit_'+arm,[str(code/'train_HIVE_relative_continuation_GPU_20261030_v1.py'),'--study',str(path),'--mode','fit','--arm',arm],True)
        phase='FULL_STATE_COVERAGE_AND_NUMPY_EQUATIONS_AUDIT'
        call('fit_CPU_audit',[str(code/'audit_HIVE_relative_continuation_CPU_20261030_v1.py'),'--study',str(path),'--mode','fit'])
        assert read(root/'CPU_audit/fit_completion.json')['status']=='PASS_ALL4_FULL_OOF_STATE_AND_INDEPENDENT_HEAD_EQUATIONS'
        execution=code/'image_execution.json';assert read(execution)['training_protocol']==ref(path)
        def image(label,mode,stage,lo=0,hi=80):
            call(label,[str(code/'run_HIVE_relative_continuation_images_20261030_v1.py'),'--execution',str(execution),'--mode',mode,'--stage',str(stage),'--lo',str(lo),'--hi',str(hi),'--arms']+list(ARMS),mode!='score')
        phase='ALL_FOUR_FIXED80_IMAGES_AND_PAIRED_READOUT'
        image('images80','generate',80);image('features80','features',80);image('score80','score',80)
        decision=read(root/'images/score80.json')['decision'];stage=80
        if decision['action']=='EXPAND400':
            phase='ALL_FOUR_EXPAND400_NO_INTERMEDIATE_EFFECT_READ'
            for lo in range(80,400,80):image('images_'+str(lo)+'_'+str(lo+80),'generate',400,lo,lo+80)
            image('features400','features',400);image('score400','score',400);stage=400;decision=read(root/'images/score400.json')['decision']
        phase='INDEPENDENT_ALL_IMAGE_IDENTITY_NOISE_RANKS_AND_RAW_PAIRS'
        call('images_CPU_audit',[str(code/'audit_HIVE_relative_continuation_CPU_20261030_v1.py'),'--study',str(path),'--mode','images'])
        assert read(root/'CPU_audit/images_completion.json')['images']==4*stage
        atomic(root/'completion.json',dict(status='COMPLETE_ALL4_RELATIVE_CONTINUATIONS_AND_FIXED_IMAGE_SCREEN',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),study=ref(path),events=events,
            primary_arm='conditional_same',final_stage=stage,decision=decision,new_research_images=4*stage,additional_models=4,additional_updates=4*24807,additional_query_exposures=4*396900,
            seconds=time.monotonic()-started,training_seed=1729,NEW_contacted=False,not_SOTA=True,no_automatic_new_fit_or_official_test=True))
    except BaseException as e:
        if owner is not None:stop_owned(owner,pinned)
        atomic(root/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-started,automatic_science_retry=False));raise

if __name__=='__main__':
    def stop(*_):raise InterruptedError('Owned relative continuation queue stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);p=argparse.ArgumentParser();p.add_argument('--study',required=True);run(p.parse_args().study)
