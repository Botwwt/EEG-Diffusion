"""One finite OLD GPU engineering job, no scientific fit follow-on."""
import argparse,datetime,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):Path(p).write_bytes((json.dumps(v,indent=2)+'\n').encode())
def run(path):
    s=read(path);R=Path(s['repository_root']);sys.path.insert(0,str(R))
    from brainhive_owned_process_CPU_20261029_v1 import capture_wrapper_child,stop_owned
    code=Path(path).parent;out=Path(s['output_root']);logs=code/'queue';logs.mkdir(exist_ok=False);started=time.monotonic();owner=pinned=None
    try:
        assert read(code/'CPU_review.json')['status']=='PASS_ACTUAL_COMMON_DIRECTION_MATH_AND_FIXED_INPUT_CPU_REVIEW'
        assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');job='HIVE_common_descent_profile_20261030_v2'
        cmd=[sys.executable,'-B',str(R/'run_gpu_job_v2.py'),'--name',job,'--max-seconds','300','--',sys.executable,'-B',str(code/'profile_HIVE_common_descent_GPU_20261030_v2.py'),'--protocol',str(path)]
        with (logs/'stdout.log').open('xb') as so,(logs/'stderr.log').open('xb') as se:
            owner=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True);pinned=capture_wrapper_child(owner,R/'audit'/(job+'_job.json'))
            fields=Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split();write(logs/'progress.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),own_PID=owner.pid,start_ticks=int(fields[19]),job=job,phase='ACTUAL_GPU_ENGINEERING_ONLY'))
            rc=owner.wait(timeout=360);owner=pinned=None
        assert rc==0;done=read(out/'completion.json');assert done['status']=='PASS_ACTUAL_FOUR_ARM_COMMON_DESCENT_FP32_GPU_ENGINEERING'
        write(code/'controller_completion.json',dict(status='COMPLETE_ONE_GPU_ENGINEERING_PROFILE_NO_FIT_OR_IMAGES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-started,job=job,
            temporary_updates=done['temporary_updates'],scientific_updates=0,followon_queue=None))
    except BaseException as e:
        if owner is not None:stop_owned(owner,pinned)
        write(code/'controller_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-started,automatic_retry=False));raise
if __name__=='__main__':
    def stop(*_):raise InterruptedError('Owned profile stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
