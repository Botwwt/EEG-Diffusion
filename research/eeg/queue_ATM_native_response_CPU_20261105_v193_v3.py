"""Finite CPU conditions -> one native GPU probe -> independent CPU review."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,subprocess,sys,time,traceback
from pathlib import Path
q=json.loads(Path(sys.argv[1]).read_bytes());root=Path(q['parent_root']);out=Path(q['output_root']);started=time.monotonic()
sys.path.insert(0,str(root/'code'));import atm_fixed_projection_contract_20261105_v190 as C
c=C.setup(root/'code/policy.json');K=c['K'];assert K.ref(__file__)==q['sources']['controller']
sp=importlib.util.spec_from_file_location('owned',K.checked(q['owned_process']));P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P)
owner=child=None;events=[]
try:
 for key,seconds,gpu in [('review',180,False)]:
  assert time.time()+seconds+10<q['absolute_deadline_epoch'];C.resources(c,True)
  cmd=[sys.executable,'-B',q['sources'][key]['path'],sys.argv[1]];receipt=c['repo']/'audit'/(q['job_name']+'_job.json')
  if gpu:
   assert not receipt.exists();cmd=[sys.executable,'-B',K.checked(q['GPU_wrapper']).as_posix(),'--name',q['job_name'],'--max-seconds','900','--',*cmd]
  K.write(out/'controller_progress.json',dict(stage=key,completed=events))
  with (out/(key+'.log')).open('xb') as f:
   owner=subprocess.Popen(cmd,cwd=c['repo'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if gpu else '',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   identity=K.process(owner.pid);child=P.capture_wrapper_child(owner,receipt) if gpu else None
   rc=owner.wait(timeout=seconds)
  assert rc==0,(key,rc);assert not K.process(identity['PID'],identity['start_ticks'])['live']
  if gpu:
   j=K.read(receipt);assert j['status']=='COMPLETED' and not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
  events.append(dict(stage=key,exit_code=rc,GPU=gpu));owner=child=None
 assert K.read(out/'review_completion.json')['status']=='PASS_ALL22400_RESPONSE_UNITS_EIGHT_CPU64_CASES_T80_READOUT'
 K.write(out/'completion.json',dict(status='COMPLETE_NATIVE_T80_RESPONSE_DIAGNOSIS',events=events,policy=K.ref(sys.argv[1]),review=K.ref(out/'review_completion.json'),GPU_receipt=K.ref(receipt),seconds=time.monotonic()-started,no_new_fit=True,NEW_contacted=False))
except BaseException as exc:
 cleanup=P.stop_owned(owner,child);K.write(out/'controller_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),events=events,cleanup=cleanup));raise
