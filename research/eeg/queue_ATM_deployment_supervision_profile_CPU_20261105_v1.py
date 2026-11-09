import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,subprocess,sys,time,traceback
from pathlib import Path
A=Path(__file__).resolve().parent;q=json.loads((A/'policy.json').read_bytes());O=A/'output';start=time.monotonic();owner=child=None
sys.path[:0]=[str(Path(q['parent_resource_policy']).parent)]
import atm_parent_R80_contract_20261105_v174 as R
c=R.setup(q['parent_resource_policy']);K=c['K'];pq=q['physical']
spec=importlib.util.spec_from_file_location('owned',K.checked(pq['owned_process']));P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
while not (A/'actual_launch.json').exists():
 assert time.monotonic()-start<15;time.sleep(.05)
launch=K.read(A/'actual_launch.json');assert launch['PID']==os.getpid()
try:
 quote=R.resources(c,starting=True);assert quote['free_bytes']>=quote['required_including_this_project']+q['extra_persistent_wrapper_reserve_bytes']
 assert K.ref(__file__)==q['sources']['controller']
 receipt=Path(pq['GPU_wrapper']['path']).parent/'audit'/(q['job_prefix']+'_job.json');assert not receipt.exists()
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
 cmd=[sys.executable,'-B',pq['GPU_wrapper']['path'],'--name',q['job_prefix'],'--max-seconds','900','--',sys.executable,'-B',q['sources']['GPU']['path'],'--policy',str(A/'policy.json')]
 with (A/'GPU.log').open('xb') as log:
  owner=subprocess.Popen(cmd,cwd=receipt.parent.parent,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  child=P.capture_wrapper_child(owner,receipt);rc=owner.wait(timeout=min(935,launch['absolute_deadline_epoch']-time.time()))
 j=K.read(receipt);assert rc==0 and j['status']=='COMPLETED' and j['exit_code']==0
 assert not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live'];owner=child=None
 d=K.read(O/'completion.json');assert d['status']=='PASS_REAL_TWO_BATCH_THREE_ROUTE_PROFILE' and d['temporary_parameter_updates']==10
 K.write(A/'controller_completion.json',dict(status='COMPLETE_GPU_DEPLOYMENT_SUPERVISION_PROFILE_CPU_AUDIT_PENDING',receipt=K.ref(receipt),worker_completion=K.ref(O/'completion.json'),seconds=time.monotonic()-start,new_fits=0,new_images=0))
except BaseException as e:
 cleanup=P.stop_owned(owner,child);K.write(A/'controller_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),cleanup=cleanup,seconds=time.monotonic()-start));raise
