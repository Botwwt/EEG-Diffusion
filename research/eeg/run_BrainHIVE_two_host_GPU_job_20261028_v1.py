"""Finite, exclusive GPU jobs on the additional host explicitly supplied by user."""
import argparse,datetime,fcntl,hashlib,json,math,os,platform,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def exclusive_json(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def update_json(path,value):
 tmp=path.with_name(path.name+'.pending');assert not tmp.exists()
 with tmp.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--max-seconds',type=int,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
 assert '/' not in a.name and '\\' not in a.name and 0<a.max_seconds<=3600
 command=a.command[1:] if a.command and a.command[0]=='--' else a.command;assert command
 ap=ROOT/'configs/research_parallel_authorization_20261028_v1.json';raw=ap.read_bytes();auth=json.loads(raw)
 assert auth['new_ordinary_versions_permitted'] and auth['existing_ordinary_baseline_bytes_immutable'] and auth['per_host_exclusive_GPU_lock_and_finite_jobs_required'] and not auth['purchase_or_provider_account_changes_authorized']
 allowed=next(h for h in auth['authorized_hosts'] if h['host']=='CLOUD_LOGIN_HOST' and h['port']==15107)
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==allowed['GPU_UUID']
 audit=ROOT/'audit';audit.mkdir(parents=True,exist_ok=True)
 with (audit/'GPU_JOB.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not active,'Additional GPU already has a compute owner'
  receipt=audit/(a.name+'_job.json');assert not receipt.exists();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';start=time.monotonic();proc=subprocess.Popen(command,cwd=ROOT,env=env,start_new_session=True)
  state=dict(name=a.name,status='RUNNING',hostname=platform.node(),GPU_UUID=uuid,pid=proc.pid,wrapper_pid=os.getpid(),command=command,start_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),maximum_seconds=a.max_seconds,authorization=dict(path=str(ap),sha256=hashlib.sha256(raw).hexdigest()),host_specific_ledger=True)
  exclusive_json(receipt,state);cancel=[None]
  for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,lambda number,_frame:cancel.__setitem__(0,number))
  reason=None
  while proc.poll() is None:
   if cancel[0] is not None or time.monotonic()-start>=a.max_seconds:
    reason='WRAPPER_SIGNAL' if cancel[0] is not None else 'FINITE_JOB_TIMEOUT';os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=15)
    break
   time.sleep(.25)
  code=proc.wait();state.update(status='COMPLETED' if code==0 and reason is None else 'FAILED_OR_TIMEOUT_PRESERVED',exit_code=code,termination_reason=reason,seconds=time.monotonic()-start,end_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
  update_json(receipt,state)
  with (audit/'compute_ledger_two_hosts.jsonl').open('a') as f:f.write(json.dumps(state)+'\n');f.flush();os.fsync(f.fileno())
  print(json.dumps(state),flush=True);return 0 if state['status']=='COMPLETED' else 1
if __name__=='__main__':raise SystemExit(main())
