"""One finite GPU engineering task after the exact original replication queue."""
import argparse,datetime,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_teacher_profile_contract_CPU_20261030_v1 import dependency_state
from brainhive_owned_process_CPU_20261029_v1 import capture_wrapper_child,stop_owned


def read_optional(path):
    path=Path(path)
    return json.loads(path.read_bytes()) if path.is_file() else None
def live(pid,ticks):
    path=Path('/proc',str(pid),'stat')
    try:fields=path.read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    return fields[0] not in ('Z','X') and int(fields[19])==ticks
def atomic(path,data):
    path=Path(path);temporary=path.with_name(path.name+'.pending');temporary.write_text(json.dumps(data,indent=2)+'\n');os.replace(temporary,path)


def main(protocol_path):
    s=read_optional(protocol_path);assert s['host']=='old'
    root=Path(s['controller_output_root']);root.mkdir();started=time.monotonic();owner=None;pinned=None;phase='WAIT_EXACT_FIXED_TRAINING_AND_ALL_R80'
    try:
        while True:
            assert time.monotonic()-started<s['maximum_CPU_wait_seconds']
            state=dependency_state(s['dependencies'],live,read_optional)
            atomic(root/'progress.json',dict(phase=phase,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),dependencies=state,new_GPU_jobs_started=0,NEW_contacted=False))
            if state['ready']:break
            time.sleep(30)
        # Nothing here cancels, changes, resumes or retries either original study.
        phase='RESOURCE_PREFLIGHT'
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
        assert shutil.disk_usage(root).free>5<<30
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        assert available>14<<30
        R=Path(s['repository_root']);job=s['job_name'];code=Path(protocol_path).parent
        command=[sys.executable,'-B',str(R/'run_gpu_job_v2.py'),'--name',job,'--max-seconds','900','--',sys.executable,'-B',str(code/'profile_HIVE_normalized_semantic_teacher_T8_GPU_20261030_v1.py'),'--protocol',str(protocol_path)]
        phase='ONE_FIXED_T8_ENGINEERING_JOB'
        with (root/'job.stdout.log').open('xb') as so,(root/'job.stderr.log').open('xb') as se:
            owner=subprocess.Popen(command,cwd=R,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True)
            pinned=capture_wrapper_child(owner,R/'audit'/(job+'_job.json'))
            fields=Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()
            atomic(root/'progress.json',dict(phase=phase,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),wrapper_PID=owner.pid,start_ticks=int(fields[19]),new_GPU_jobs_started=1,command=command))
            rc=owner.wait(timeout=960);owner=None;pinned=None
        assert rc==0
        result=read_optional(Path(s['output_root'])/'completion.json')
        assert result['status']=='COMPLETE_FIXED_T8_NORMALIZED_SEMANTIC_TEACHER_E_M_ENGINEERING'
        atomic(root/'completion.json',dict(status='COMPLETE_ONE_DEPENDENT_T8_ENGINEERING_ONLY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),profile=result,new_full_T_fit_or_research_images=False))
    except BaseException as error:
        if owner is not None:stop_owned(owner,pinned)
        atomic(root/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True));raise


if __name__=='__main__':
    def stop(*unused):raise InterruptedError('Explicit dependent engineering stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);main(parser.parse_args().protocol)
