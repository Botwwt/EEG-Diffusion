"""Explicitly invoked, finite admitted queue. A DRAFT never dispatches a GPU."""
import argparse,datetime,json,os,signal,subprocess,time,traceback
from pathlib import Path
from AR_anchored_training_contract_CPU_20261029_v1 import load_admission,validate_sources,validate_profiles,validate_image_binding,verify_protected_files,ref

def write_once(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def replace_json(path,value):
    pending=path.with_name(path.name+f'.pending.{os.getpid()}.{time.time_ns()}');write_once(pending,value);os.replace(pending,path)

def main(admission_path):
    admission,admission_sha=load_admission(admission_path)
    validate_sources(admission,controller_path=Path(__file__))
    validate_profiles(admission);validate_image_binding(admission);verify_protected_files(admission)
    root=Path(admission['output_root']);root.mkdir(exist_ok=True);assert root.resolve()==root and not root.is_symlink()
    assert not (root/'queue_launch.json').exists() and not (root/'queue_failure.json').exists()
    write_once(root/'registration.json',dict(admission_sha256=admission_sha,admission=ref(admission_path)))
    write_once(root/'queue_launch.json',dict(status='EXPLICITLY_INVOKED_FORMAL_AR_QUEUE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        pid=os.getpid(),admission=ref(admission_path),arms=admission['arms']))
    stop=[];events=[];start=time.monotonic();resources=admission['resources']
    def stop_signal(signum,frame):
        stop.append(signum);p=root/'STOP_REQUESTED.json'
        if not p.exists():write_once(p,dict(status='USER_OR_SIGNAL_STOP_REQUESTED_WHOLE_LOGICAL_STEP_BOUNDARY',signal=signum,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    R=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
    wrapper=R/('run_gpu_job_v2.py' if admission['host_role']=='old' else 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py')
    try:
        for arm in admission['arms']:
            while not (root/arm/'completion.json').exists():
                assert time.monotonic()-start<resources['maximum_queue_seconds']
                if stop or (root/'STOP_REQUESTED.json').exists():
                    write_once(root/'queue_stopped.json',dict(status='AR_QUEUE_STOPPED_WITHOUT_NEW_JOB',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),events=events));return
                while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                    assert time.monotonic()-start<resources['maximum_queue_seconds']
                    if stop:break
                    replace_json(root/'queue_progress.json',dict(status='ADMITTED_AR_QUEUE_WAITING_GPU_IDLE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=arm,events=events));time.sleep(10)
                if stop:continue
                attempts=root/arm/'attempts';existing=sorted(p for p in attempts.iterdir() if p.is_dir()) if attempts.exists() else []
                assert [p.name for p in existing]==[f'attempt_{i:05d}' for i in range(1,len(existing)+1)]
                for prior in existing:assert (prior/'completion.json').is_file() and not (prior/'failure.json').exists(), 'No automatic retry after a failed attempt'
                index=len(existing)+1;limit=1 if index==1 else resources['maximum_updates_per_job']
                label=f"AR_anchor_{admission['host_role']}_{arm}_a{index:05d}_20261029_v1"
                command=['/root/miniconda3/bin/python','-B',str(wrapper),'--name',label,'--max-seconds','900','--','/root/miniconda3/bin/python','-B',admission['producer'],
                    '--admission',str(admission_path),'--arm',arm,'--attempt-index',str(index),'--maximum-updates',str(limit),'--job-name',label]
                replace_json(root/'queue_progress.json',dict(status='FORMAL_AR_QUEUE_EXECUTING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=arm,attempt_index=index,command=command,events=events))
                with (root/(label+'.stdout.log')).open('xb') as out,(root/(label+'.stderr.log')).open('xb') as err:
                    began=time.monotonic();result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=950)
                event=dict(arm=arm,attempt_index=index,exit_code=result.returncode,seconds=time.monotonic()-began,job_receipt=ref(R/'audit'/(label+'_job.json')));events.append(event)
                assert result.returncode==0,event
                completion=json.loads((root/arm/'attempts'/f'attempt_{index:05d}'/'completion.json').read_bytes())
                assert completion['status']=='COMPLETE_FINITE_FORMAL_AR_LOGICAL_STEP_CHUNK' and completion['admission_sha256']==admission_sha
                manifest=json.loads((root/arm/'resume/manifest.json').read_bytes());assert manifest['status']=='READY' and manifest['current']['updates']==completion['end_update']
                if completion['stop_requested']:stop.append('producer_whole_step_stop')
            final=json.loads((root/arm/'completion.json').read_bytes());assert final['updates']==8270 and final['query_exposures']==132300
        verify_protected_files(admission)
        write_once(root/'queue_completion.json',dict(status='ALL_HOST_ASSIGNED_FIXED_T_AR_ARMS_COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),host_role=admission['host_role'],
            arms=admission['arms'],events=events,seconds=time.monotonic()-start,new_research_images=0,all10_formal_training_claimed=False))
    except BaseException as error:
        write_once(root/'queue_failure.json',dict(status='FORMAL_AR_QUEUE_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            error=repr(error),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-start));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--admission',type=Path,required=True);a=p.parse_args()
    try:main(a.admission)
    except ValueError as error:
        if str(error).startswith('NOT_ADMITTED_'):print(str(error));raise SystemExit(64)
        raise
