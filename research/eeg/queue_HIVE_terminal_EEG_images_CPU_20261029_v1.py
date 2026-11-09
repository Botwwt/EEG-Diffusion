"""Wait for the exact fixed-fit queue, audit, then paired80->400 images.

CPU wait never reserves CUDA. No candidate choice from controls; real_prefix
alone controls the fixed screening expansion. Failed stages remain failed.
"""
import argparse,datetime,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from run_HIVE_terminal_EEG_images_20261029_v1 import checked,read,ref,write
def atomic(p,x):
 p=Path(p);temp=p.with_name(p.name+'.pending');temp.write_text(json.dumps(x,indent=2)+'\n');os.replace(temp,p)
def live(pid,start):
 p=Path('/proc')/str(pid)/'stat'
 if not p.is_file():return False
 f=p.read_text().rsplit(')',1)[1].split();return f[0] not in ('Z','X') and int(f[19])==start
def run(path):
 s=read(path);assert s['status']=='REGISTERED_NATIVE_TERMINAL_EEG_PAIRED_IMAGE_SCREEN';root=Path(s['output_root']);root.mkdir();logs=root/'queue';logs.mkdir();start=time.monotonic();active=None;phase='WAIT_FIXED_FITS';events=[]
 try:
  for x in s['source_pins']:checked(x)
  checked(s['training_protocol']);up=Path(s['training_output_root']);a=s['training_controller'];assert a['controller_PID']==314506 and a['start_ticks']==92219247
  while True:
   assert time.monotonic()-start<13000,'Finite upstream waiting window exhausted; no retry or training restart'
   if (up/'failure.json').is_file():raise RuntimeError('Upstream fixed training failed; preserve its exact state and stop this dependent queue')
   alive=live(a['controller_PID'],a['start_ticks']);atomic(root/'progress.json',dict(phase=phase,upstream_controller_live=alive,upstream_progress=read(up/'progress.json') if (up/'progress.json').is_file() else None,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
   if not alive:
    assert (up/'completion.json').is_file(),'Exact training controller terminal without a successful final completion';break
   time.sleep(20)
  def task(label,cmd,gpu=False,stdin=None):
   nonlocal active,phase
   phase=label
   if gpu:cmd=[sys.executable,'-B',str(Path(s['repository_root'])/'run_gpu_job_v2.py'),'--name','HIVE_terminal_EEG_images_'+label+'_20261029_v1','--max-seconds','900','--']+cmd
   with (logs/(label+'.stdout.log')).open('xb') as so,(logs/(label+'.stderr.log')).open('xb') as se:
    active=subprocess.Popen(cmd,stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True,cwd=s['repository_root']);f=(Path('/proc')/str(active.pid)/'stat').read_text().rsplit(')',1)[1].split();atomic(root/'progress.json',dict(phase=phase,pid=active.pid,start_ticks=int(f[19]),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    if stdin is not None:active.communicate(stdin,timeout=180)
    else:active.wait(timeout=940 if gpu else 180)
    rc=active.returncode;active=None
   event=dict(phase=label,exit_code=rc);events.append(event);write(logs/(label+'_terminal.json'),event);assert rc==0,event
  training_audit=up/'independent_full_T_training_CPU_audit.json'
  if training_audit.is_file():assert read(training_audit)['status']=='ALL13230_PREFIXES_AND_BOTH1654_UPDATE_FINAL_STATES_CPU_AUDITED'
  else:task('training_CPU_audit',[sys.executable,'-B',s['training_CPU_auditor']['path']],stdin=b'{"host":"old"}\n')
  state=read(training_audit);assert state['prefix_queries']==13230 and set(state['arms'])=={'real_prefix','Gaussian'}
  bound=dict(s);bound['registered_protocol']=ref(path);bound['training_audit_expected']=ref(training_audit);bound_path=Path(path).parent/'bound_protocol.json';write(bound_path,bound)
  base=[sys.executable,'-B',s['program']['path'],'--protocol',str(bound_path)]
  task('all400_CPU_conditions',base+['--mode','conditions'])
  for mode in ('generate','features'):task(mode+'_000_080',base+['--mode',mode,'--lo','0','--hi','80'],gpu=True)
  task('score80',base+['--mode','score','--count','80']);task('audit80',base+['--mode','audit','--count','80']);first=read(root/'score80.json')
  if first['primary_decision']['action']=='EXPAND400':
   for lo,hi in ((80,240),(240,400)):
    for mode in ('generate','features'):task(f'{mode}_{lo:03d}_{hi:03d}',base+['--mode',mode,'--lo',str(lo),'--hi',str(hi)],gpu=True)
   task('score400',base+['--mode','score','--count','400']);task('audit400',base+['--mode','audit','--count','400']);count=400
  else:assert first['primary_decision']['reason']=='LARGE_HARM';count=80
  final=read(root/f'score{count}.json');write(root/'completion.json',dict(status='COMPLETE_FIXED_TERMINAL_EEG_IMAGE_SCREEN',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=ref(path),bound_protocol=ref(bound_path),queries=count,new_images=3*count,ordinary_exact_replays=count,score=ref(root/f'score{count}.json'),independent_CPU_audit=ref(root/f'independent_images_and_statistics{count}_CPU_audit.json'),primary_decision=final['primary_decision'],events=events,no_automatic_official_or_multisubject_access=True,NEW_contacted=False,seconds=time.monotonic()-start,training_wait_seconds_included=True))
 except BaseException as error:
  if active is not None and active.poll() is None:
   try:os.killpg(active.pid,signal.SIGTERM);active.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait(timeout=10)
   except ProcessLookupError:pass
  write(root/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-start));raise
if __name__=='__main__':
 def stop(*_):raise InterruptedError('Explicit dependent image queue stop')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
