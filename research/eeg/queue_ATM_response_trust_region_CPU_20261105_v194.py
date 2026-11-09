import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,subprocess,sys,time,traceback
from pathlib import Path
q=json.loads(Path(sys.argv[1]).read_bytes());root=Path(q['parent_root']);out=Path(q['output_root']);start=time.monotonic()
sys.path.insert(0,str(root/'code'));import atm_fixed_projection_contract_20261105_v190 as C
c=C.setup(root/'code/policy.json');K=c['K'];assert K.ref(__file__)==q['sources']['controller'];C.resources(c,True)
spec=importlib.util.spec_from_file_location('owned',K.checked(q['owned_process']));P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P);child=None
try:
 assert time.time()+220<q['absolute_deadline_epoch']
 with (out/'worker.log').open('xb') as f:child=subprocess.Popen([sys.executable,'-B',q['sources']['worker']['path'],sys.argv[1]],cwd=c['repo'],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 who=K.process(child.pid);K.write(out/'worker_identity.json',who);rc=child.wait(timeout=180);assert rc==0,rc;assert not K.process(who['PID'],who['start_ticks'])['live'];child=None
 assert K.read(out/'completion.json')['status']=='COMPLETE_CN_RADIUS_TRUST_REGION_DIAGNOSTIC'
 K.write(out/'controller_completion.json',dict(status='COMPLETE_CPU_ONLY_TRUST_REGION',worker=who,exit_code=rc,completion=K.ref(out/'completion.json'),seconds=time.monotonic()-start,NEW_contacted=False))
except BaseException as exc:
 cleanup=P.stop_owned(child,None);K.write(out/'controller_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),cleanup=cleanup));raise
