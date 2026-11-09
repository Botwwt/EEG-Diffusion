"""Finite follow-on after exact fits, R80 evaluation and actual-G T8 profile.

Never resumes an upstream job or admits a full-T fit. Binds the actual upstream
q artifact only after its exact controller has exited successfully.
"""
import argparse,datetime,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_joint_student_followon_contract_CPU_20261030_v1 import dependencies_ready,validate_teacher_arrays
from brainhive_owned_process_CPU_20261029_v1 import capture_wrapper_child,stop_owned


def read(path):
    path=Path(path)
    return json.loads(path.read_bytes()) if path.is_file() else None


def ref(path):
    path=Path(path)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest)


def checked(item):
    actual=ref(item['path']);assert actual['sha256']==item['sha256']
    if 'bytes' in item:assert actual['bytes']==item['bytes']
    return actual


def live(pid,ticks):
    try:fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    return fields[0] not in ('Z','X') and int(fields[19])==ticks


def atomic(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)


def main(protocol_path):
    s=read(protocol_path);assert s['host']=='old' and not s['NEW_GPU_use_allowed']
    root=Path(s['controller_output_root']);root.mkdir();started=time.monotonic();owner=None;pinned=None;phase='WAIT_EXACT_FITS_R80_AND_ACTUAL_G_TEACHERS'
    try:
        while True:
            assert time.monotonic()-started<s['maximum_CPU_wait_seconds']
            state=dependencies_ready(s['dependencies'],live,read)
            atomic(root/'progress.json',dict(phase=phase,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),dependencies=state,new_GPU_jobs_started=0,NEW_contacted=False))
            if state['ready']:break
            time.sleep(30)
        phase='BIND_ACTUAL_TERMINAL_TEACHER_ARTIFACT'
        for item in s['source_pins']+s['payloads']:checked(item)
        upstream=read(s['upstream_teacher_completion_path'])
        assert upstream['status']=='COMPLETE_FIXED_T8_NORMALIZED_SEMANTIC_TEACHER_E_M_ENGINEERING'
        assert upstream['protocol']==s['upstream_teacher_protocol']
        assert upstream['queries']==8 and upstream['actual_T8_auxiliary_teachers_temporarily_optimized']
        assert upstream['teacher_arms']==s['teacher_arms'] and upstream['temporary_E_updates']==96
        import numpy as np
        artifact=ref(s['upstream_teacher_parameters_path'])
        data=dict(np.load(artifact['path'],allow_pickle=False));validation=validate_teacher_arrays(data,s['teacher_arms'])
        completion=ref(s['upstream_teacher_completion_path'])
        validation.update(status='PASS_ALL24_ACTUAL_UPSTREAM_POWER_SPHERICAL_ARTIFACT_MOMENTS',teacher_parameters=artifact,upstream_completion=completion)
        atomic(root/'actual_upstream_teacher_CPU_validation.json',validation)
        binding=dict(upstream_protocol=s['upstream_teacher_protocol'],teacher_parameters=artifact,input_pins=[artifact,completion]+[ref(item['completion']) for item in s['dependencies']],validation=validation)
        binding_path=root/'actual_teacher_binding.json';atomic(binding_path,binding);binding_ref=ref(binding_path)
        phase='RESOURCE_PREFLIGHT'
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
        assert shutil.disk_usage(root).free>5<<30
        available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
        assert available>14<<30
        R=Path(s['repository_root']);code=Path(protocol_path).parent;job=s['job_name']
        command=[sys.executable,'-B',str(R/'run_gpu_job_v2.py'),'--name',job,'--max-seconds','900','--',sys.executable,'-B',str(code/'profile_HIVE_joint_student_readout_noise_T8_GPU_20261030_v1.py'),'--protocol',str(protocol_path),'--binding',str(binding_path),'--binding-sha',binding_ref['sha256']]
        phase='ONE_JOINT_STUDENT_READOUT_AND_NOISE_ENGINEERING_JOB'
        with (root/'job.stdout.log').open('xb') as so,(root/'job.stderr.log').open('xb') as se:
            owner=subprocess.Popen(command,cwd=R,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True)
            pinned=capture_wrapper_child(owner,R/'audit'/(job+'_job.json'))
            fields=Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()
            atomic(root/'progress.json',dict(phase=phase,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),wrapper_PID=owner.pid,start_ticks=int(fields[19]),new_GPU_jobs_started=1,command=command,binding=binding_ref))
            rc=owner.wait(timeout=960);owner=None;pinned=None
        assert rc==0
        actual_root=Path(s['output_root']);profile=read(actual_root/'completion.json')
        assert profile['status']=='COMPLETE_JOINT_STUDENT_READOUT_NOISE_T8_ENGINEERING'
        assert profile['protocol']==ref(protocol_path) and profile['binding']==binding_ref
        phase='INDEPENDENT_CPU_STATE_READOUT_AND_TRAINING_CRITIC_AUDIT'
        environment=dict(os.environ);environment.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        command=[sys.executable,'-B',str(code/'audit_HIVE_joint_student_readout_noise_CPU_20261030_v1.py'),'--protocol',str(protocol_path),'--binding',str(binding_path)]
        with (root/'audit.stdout.log').open('xb') as so,(root/'audit.stderr.log').open('xb') as se:
            owner=subprocess.Popen(command,cwd=R,env=environment,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True)
            rc=owner.wait(timeout=180);owner=None
        assert rc==0
        independent=read(actual_root/'independent_CPU_audit.json')
        assert independent['status']=='PASS_JOINT_STUDENT_T8_INDEPENDENT_CPU_STATE_FORWARD_READOUT_AND_NOISE_ARITHMETIC'
        atomic(root/'completion.json',dict(status='COMPLETE_ONE_JOINT_STUDENT_READOUT_NOISE_T8_ENGINEERING_ONLY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=ref(protocol_path),binding=binding_ref,profile=profile,independent_CPU_audit=independent,new_full_T_fit_or_image_efficacy=False,no_automatic_retry=True,NEW_contacted=False))
    except BaseException as error:
        if owner is not None:stop_owned(owner,pinned)
        atomic(root/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True,NEW_contacted=False));raise


if __name__=='__main__':
    def stop(*unused):raise InterruptedError('Explicit own follow-on engineering stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);main(parser.parse_args().protocol)
