"""Two predeclared complete fits, finite jobs, no retry/resume or seed selection."""
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
def ref(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def atomic(p,x):
 p=Path(p);pending=p.with_name(p.name+'.pending');pending.write_text(json.dumps(x,indent=2)+'\n');os.replace(pending,p)
def run(path):
 s=json.loads(Path(path).read_bytes());assert s['host']=='old' and s['seeds']==[2027,3407] and s['no_automatic_resume'];root=Path(s['output_root']);root.mkdir();started=time.monotonic();active=None;events=[];phase='INIT'
 try:
  for p in s['source_pins']+s['payloads']:assert ref(p['path'])['sha256']==p['sha256']
  for number in s['seeds']:
   assert time.monotonic()-started<102000;assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();phase=f'seed{number}_full24807';job=f'HIVE_ordinary_seed{number}_full_T1323_20261029_v3';command=[sys.executable,'-B',str(Path(s['repository_root'])/'run_gpu_job_v2.py'),'--name',job,'--max-seconds',str(s['job_seconds']),'--',sys.executable,'-B',s['trainer']['path'],'--protocol',str(path),'--mode','train','--seed',str(number),'--part','0']
   with (root/(phase+'.stdout.log')).open('xb') as so,(root/(phase+'.stderr.log')).open('xb') as se:
    active=subprocess.Popen(command,cwd=s['repository_root'],stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True);st=(Path('/proc')/str(active.pid)/'stat').read_text().rsplit(')',1)[1].split();atomic(root/'queue_progress.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,wrapper_PID=active.pid,start_ticks=int(st[19]),command=command,events=events));rc=active.wait(timeout=s['job_seconds']+60);active=None
   ev=dict(seed=number,exit_code=rc,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());events.append(ev);atomic(root/(phase+'_terminal.json'),ev);assert rc==0,'Preserve failure; no automatic restart/resume'
   p=root/f'seed{number}/part00/completion.json';done=json.loads(p.read_bytes());assert done['status']=='COMPLETE_FULL10_TRUE_OOF_T_THREE_PASSES_ORDINARY_REPLICATE' and done['updates']==24807 and done['seen']==396900;assert ref(done['component']['path'])==done['component']
  atomic(root/'training_completion.json',dict(status='BOTH_PREDECLARED_UNINTERRUPTED_ORDINARY_REPLICATES_FULL_T_COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=ref(path),events=events,seeds=s['seeds'],existing_seed1729_reused=True,new_images=0,image_evaluation_not_yet_run=True,seconds=time.monotonic()-started,SOTA_established=False))
 except BaseException as error:
  if active is not None and active.poll() is None:
   try:os.killpg(active.pid,signal.SIGTERM);active.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait(timeout=10)
   except ProcessLookupError:pass
  atomic(root/'queue_failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-started));raise
if __name__=='__main__':
 def stop(*_):raise InterruptedError('Explicit seed queue stop')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
