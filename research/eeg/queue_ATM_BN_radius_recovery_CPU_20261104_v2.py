"""Explicit projection-only recovery from completed BN state, old deadline/cap."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,json,subprocess,sys,time,traceback
from pathlib import Path
recovery=json.loads(Path(sys.argv[1]).read_bytes());sys.path.insert(0,str(Path(recovery['original_policy']['path']).parent))
import atm_BN_radius_contract_20261104_v1 as B
c=B.setup(recovery['original_policy']['path']);q=c['q'];K=c['K'];out=c['out'];repo=c['repo'];owner=child=None;started=time.monotonic();code=Path(sys.argv[1]).parent
import brainhive_owned_process_CPU_20261029_v1 as P
until=time.monotonic()+10
while not (code/'actual_launch.json').exists():assert time.monotonic()<until;time.sleep(.05)
launch=K.read(code/'actual_launch.json');assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks'];deadline=launch['absolute_deadline_epoch']
try:
    assert deadline==recovery['original_deadline_epoch'] and time.time()+1900<deadline;B.resources(c,starting=True)
    for pin in recovery['completed_prefix'].values():K.checked(pin)
    receipt=repo/'audit'/(recovery['job_name']+'_job.json');assert not receipt.exists() and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    command=[sys.executable,'-B',str(repo/'run_gpu_job_v2.py'),'--name',recovery['job_name'],'--max-seconds','900','--',sys.executable,'-B',recovery['sources']['GPU']['path'],sys.argv[1]]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';K.write(out/'recovery_progress_v2.json',dict(phase='PROJECTOR_AND_FIXED2_INPUT_PROBES_NO_BN_REFIT'))
    with (out/'GPU_recovery_v2.log').open('xb') as log:
        owner=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);child=P.capture_wrapper_child(owner,receipt);rc=owner.wait(timeout=930)
    assert rc==0;job=K.read(receipt);assert job['status']=='COMPLETED' and job['exit_code']==0 and not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live'];owner=child=None
    assert time.time()+930<deadline;B.resources(c,starting=True);K.write(out/'recovery_progress_v2.json',dict(phase='CPU_FULL_AUDIT_REUSING_ORIGINAL_BN_FINAL_STATE'))
    with (out/'CPU_recovery_v2.log').open('xb') as log:owner=subprocess.Popen([sys.executable,'-B',q['sources']['CPU']['path'],recovery['original_policy']['path']],cwd=repo,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);rc=owner.wait(timeout=900)
    assert rc==0;owner=None;audit=K.read(out/'CPU_audit.json');assert audit['status']=='PASS_FULL_T_BN_STATE_AND_ALL400_NATIVE_CPU_CONDITIONS_RADIUS_PROJECTOR_AUDIT'
    for pin in recovery['completed_prefix'].values():K.checked(pin)
    K.write(out/'completion.json',dict(status='COMPLETE_BN_RADIUS_NATIVE_FUNCTION_PREPARATION_NO_IMAGES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=recovery['original_policy'],recovery_policy=K.ref(sys.argv[1]),original_failure_retained=True,original_failed_job=recovery['original_failed_job'],completed_BN_prefix_SHA_unchanged=True,
        GPU_completion=K.ref(out/'GPU_completion.json'),CPU_audit=K.ref(out/'CPU_audit.json'),job=K.ref(receipt),seconds=time.monotonic()-started,absolute_deadline_epoch=deadline,BN_statistic_updates=830,optimizer_updates=0,new_images=0,temporary_fixed_input_probe_statistic_updates_discarded=2,not_new_EEG_method_or_image_gain=True))
    K.write(out/'recovery_progress_v2.json',dict(phase='COMPLETE_RECOVERED_BN_RADIUS_PREPARATION_NO_IMAGES'))
except BaseException as exc:
    cleanup=P.stop_owned(owner,child);K.write(out/'recovery_failure_v2.json',dict(error=repr(exc),traceback=traceback.format_exc(),cleanup=cleanup,absolute_deadline_epoch=deadline,partial_evidence_retained=True));raise
