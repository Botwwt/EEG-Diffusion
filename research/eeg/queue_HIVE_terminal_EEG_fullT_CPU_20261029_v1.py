"""Finite whole-T bank and two fixed fit arms; no automatic test access."""
import argparse,datetime,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from run_HIVE_terminal_EEG_fullT_20261029_v1 import checked,read,ref,write,atomic_json
def run(path):
 s=read(path);root=Path(s['output_root']);root.mkdir();logs=root/'queue';logs.mkdir();start=time.monotonic();events=[];active=None;phase='INIT'
 try:
  for x in s['source_pins']+s['payloads']:checked(x)
  tasks=[('bank','',i) for i in range(s['bank_parts'])]+[('train',arm,i) for arm in ('real_prefix','Gaussian') for i in range(s['training_parts_per_arm'])]
  for mode,arm,part in tasks:
   phase=f'{mode}_{arm}_{part:02d}';name='HIVE_terminal_fullT_'+phase+'_20261029_v1';cmd=[sys.executable,'-B',str(Path(s['repository_root'])/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--',sys.executable,'-B',s['program']['path'],'--protocol',str(path),'--mode',mode,'--part',str(part),'--arm',arm]
   with (logs/(phase+'.stdout.log')).open('xb') as so,(logs/(phase+'.stderr.log')).open('xb') as se:
    active=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True,cwd=s['repository_root']);f=(Path('/proc')/str(active.pid)/'stat').read_text().rsplit(')',1)[1].split();atomic_json(root/'progress.json',dict(phase=phase,wrapper_PID=active.pid,start_ticks=int(f[19]),completed_tasks=len(events),total_tasks=len(tasks),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()));rc=active.wait(timeout=940);active=None
   event=dict(phase=phase,exit_code=rc);events.append(event);write(logs/(phase+'_terminal.json'),event);assert rc==0,event
  finals={arm:read(root/f'{arm}_part_{s["training_parts_per_arm"]-1:02d}/completion.json') for arm in ('real_prefix','Gaussian')};assert all(x['updates']==1654 and x['exposures']==13230 for x in finals.values());assert finals['real_prefix']['initial_model_sha256']==finals['Gaussian']['initial_model_sha256'];assert finals['real_prefix']['order_sha256']==finals['Gaussian']['order_sha256'];write(root/'completion.json',dict(status='COMPLETE_SUB01_FULL_T_ONE_PASS_MATCHED_TERMINAL_EEG_TRAINING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=ref(path),events=events,final_models={a:x['final_model'] for a,x in finals.items()},updates_per_arm=1654,exposures_per_arm=13230,training_seed=1729,model_choice='fixed last update only',Dcheck_or_official_results_read_by_this_queue=False,new_research_images=0,NEW_contacted=False,seconds=time.monotonic()-start,next='Independent full-state/coverage audit and normal EEG image comparison required; no automatic claim of efficacy or official access.'))
 except BaseException as error:
  if active is not None and active.poll() is None:
   try:os.killpg(active.pid,signal.SIGTERM);active.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait(timeout=10)
   except ProcessLookupError:pass
  write(root/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-start));raise
if __name__=='__main__':
 def stop(*_):raise InterruptedError('Explicit full T queue stop')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
