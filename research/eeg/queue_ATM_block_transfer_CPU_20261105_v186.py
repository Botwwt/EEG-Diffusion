"""Finite read-only wait, then ONE bounded CPU factorial readout; never train/restart."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,json,subprocess,sys,time,traceback
from pathlib import Path
p=Path(sys.argv[1]);q=json.loads(p.read_bytes());root=Path(q['project_root']);out=Path(q['output_root'])
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):
 p=Path(p);tmp=p.with_name(p.name+'.pending');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def proc(a):
 p=Path('/proc',str(a['PID']),'stat')
 if not p.exists():return False
 s=p.read_text().rsplit(')',1)[1].split();return s[0] not in ('Z','X') and int(s[19])==a['start_ticks']
try:
 original=read(root/'code_recovery_v3/actual_launch.json');assert original['PID']==752098 and original['start_ticks']==152300045
 while time.time()+130<q['absolute_deadline_epoch']:
  image_live=proc(original);rd=root/'terminal_review';rla=rd/'actual_launch.json'
  state=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),image_owner_live=image_live,effects_read=False)
  if not image_live:
   assert (root/'image_output/completion.json').exists() and not (root/'image_output/failure.json').exists(),'IMAGE_FAILURE_PRESERVED_NO_RESTART'
   assert rla.exists(),'ORIGINAL_CONTROLLER_DID_NOT_LAUNCH_REVIEW'
   rl=read(rla);state['review_owner_live']=proc(rl)
   if not state['review_owner_live']:
    assert (rd/'completion.json').exists(),'REVIEW_FAILURE_PRESERVED_NO_RESTART'
    break
  write(out/'wait_status.json',state);time.sleep(min(60,max(1,q['absolute_deadline_epoch']-time.time()-130)))
 else:raise TimeoutError('FINITE_WAIT_EXPIRED_SCIENCE_UNCHANGED')
 cmd=['timeout','--signal=TERM','--kill-after=10s','120',sys.executable,'-B',q['source']['path'],str(p)]
 with (out/'analysis.log').open('xb') as f:
  child=subprocess.Popen(cmd,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  st=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()
  write(out/'analysis_launch.json',dict(PID=child.pid,start_ticks=int(st[19]),maximum_seconds=120,command=cmd))
  rc=child.wait(timeout=135)
 assert rc==0,rc
 write(out/'queue_completion.json',dict(status='COMPLETE_TERMINAL_ONLY_BLOCK_TRANSFER_ANALYSIS',analysis=read(out/'completion.json')['status'],UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
except BaseException as exc:
 write(out/'queue_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),original_science_never_restarted=True));raise
