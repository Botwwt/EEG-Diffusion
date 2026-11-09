"""Finite stdlib-only wait for BOTH exact science/review owners, then one CPU analysis."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
path=Path(sys.argv[1]);q=json.loads(path.read_bytes());out=Path(q['output_root']);parent=Path(q['parent_root']);child=None
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+'.pending');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def proc(pid,ticks=None):
 p=Path('/proc',str(pid),'stat')
 if not p.exists():return False
 s=p.read_text().rsplit(')',1)[1].split();return s[0] not in ['Z','X'] and (ticks is None or int(s[19])==ticks)
try:
 assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==q['waiter']['sha256']
 while time.time()+900<q['absolute_deadline_epoch']:
  la=read(parent/'code/actual_launch.json');assert la['PID']==759064 and la['start_ticks']==153164658
  live=proc(la['PID'],la['start_ticks'])
  if not live:
   assert not (parent/'image_output/failure.json').exists() and (parent/'image_output/completion.json').exists(),'PARENT_FAILED_NO_RESTART'
   ra=parent/'terminal_review/actual_launch.json'
   if ra.exists():
    r=read(ra)
    if not proc(r['PID'],r['start_ticks']):
     done=read(parent/'terminal_review/completion.json');assert done['status']=='PASS_ALL2400_IDENTITIES_PIXELS_NOISES_11200_RANK_VALUES_7200_LOWLEVEL_138_CIS';break
  write(out/'wait_progress.json',dict(status='WAIT_SAME_IMAGE_AND_REVIEW_OWNERS',image_owner_live=live,effects_read=0));time.sleep(20)
 else:raise TimeoutError('FINITE_WAIT_EXPIRED_NO_RESTART')
 worker=Path(q['worker']['path']);assert hashlib.sha256(worker.read_bytes()).hexdigest()==q['worker']['sha256']
 with (out/'worker.log').open('xb') as f:
  child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=10s','880',sys.executable,'-B',str(worker),'--plan',str(path)],cwd=q['repo'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  write(out/'worker_launch.json',dict(PID=child.pid,after_parent_and_review_terminal=True));rc=child.wait(timeout=895)
 assert rc==0,rc;write(out/'wait_completion.json',dict(status='COMPLETE_ONE_TERMINAL_CPU_ANALYSIS',exit_code=rc,new_GPU_jobs=0))
except BaseException as exc:
 if child is not None and child.poll() is None:
  import signal
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
 write(out/'wait_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),parent_never_restarted=True));raise
