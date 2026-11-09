"""One bounded GPU diagnostic followed by a full CPU audit; no images/retries."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,subprocess,sys,time,traceback
from pathlib import Path
import atm_BN_radius_contract_20261104_v1 as B
c=B.setup(sys.argv[1]);q=c['q'];K=c['K'];out=c['out'];repo=c['repo'];started=time.monotonic();owner=child=None
import brainhive_owned_process_CPU_20261029_v1 as P
until=time.monotonic()+10
while not (Path(q['code_root'])/'actual_launch.json').exists():assert time.monotonic()<until;time.sleep(.05)
launch=K.read(Path(q['code_root'])/'actual_launch.json');assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks'];deadline=launch['absolute_deadline_epoch']
try:
    B.resources(c,starting=True);assert time.time()+1900<deadline
    receipt=repo/'audit'/(q['job_name']+'_job.json');assert not receipt.exists() and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    command=[sys.executable,'-B',str(repo/'run_gpu_job_v2.py'),'--name',q['job_name'],'--max-seconds','900','--',sys.executable,'-B',q['sources']['GPU']['path'],sys.argv[1]]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';K.write(out/'progress.json',dict(phase='RUNNING_T_ONLY_BN_AND_NATIVE_PROJECTOR_DIAGNOSTIC'))
    with (out/'GPU.log').open('xb') as log:
        owner=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);child=P.capture_wrapper_child(owner,receipt);rc=owner.wait(timeout=930)
    assert rc==0;job=K.read(receipt);assert job['status']=='COMPLETED' and job['exit_code']==0 and not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live'];owner=child=None
    assert time.time()+930<deadline;B.resources(c,starting=True);K.write(out/'progress.json',dict(phase='CPU_FULL_BN_AND400_CONDITIONS_AUDIT'))
    with (out/'CPU.log').open('xb') as log:owner=subprocess.Popen([sys.executable,'-B',q['sources']['CPU']['path'],sys.argv[1]],cwd=repo,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);rc=owner.wait(timeout=900)
    assert rc==0;owner=None;audit=K.read(out/'CPU_audit.json');assert audit['status']=='PASS_FULL_T_BN_STATE_AND_ALL400_NATIVE_CPU_CONDITIONS_RADIUS_PROJECTOR_AUDIT'
    K.write(out/'completion.json',dict(status='COMPLETE_BN_RADIUS_NATIVE_FUNCTION_PREPARATION_NO_IMAGES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=K.ref(sys.argv[1]),GPU_completion=K.ref(out/'GPU_completion.json'),CPU_audit=K.ref(out/'CPU_audit.json'),job=K.ref(receipt),seconds=time.monotonic()-started,absolute_deadline_epoch=deadline,BN_statistic_updates=830,optimizer_updates=0,new_images=0,not_new_EEG_method_or_image_gain=True))
    K.write(out/'progress.json',dict(phase='COMPLETE_BN_RADIUS_DIAGNOSTIC_NO_IMAGES'))
except BaseException as exc:
    cleanup=P.stop_owned(owner,child);K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),cleanup=cleanup,absolute_deadline_epoch=deadline,partial_evidence_retained=True));raise
