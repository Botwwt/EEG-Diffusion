"""One frozen GPU probe followed by one CPU readout; no scientific retries."""
import argparse,subprocess,sys,time,traceback
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();started=time.monotonic()
 import json
 s=json.loads(Path(a.protocol).read_bytes());sys.path.insert(0,str(Path(s['base_protocol']['path']).parent))
 import brainhive_functional_mean_contract_20261103_v1 as G
 base=G.setup(G.checked(s['base_protocol']));root=Path(s['output_root']);code=Path(a.protocol).parent
 assert s['status']=='REGISTERED_FROZEN_FULL_UNET_PAIRED_EEG_AND_SYMMETRIC_PATH_DIAGNOSTIC'
 for pin in s['source_pins']:G.checked(pin)
 out=root/'queue';assert not out.exists();out.mkdir();events=[]
 try:
  G.write(out/'started.json',dict(UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(),protocol=G.ref(a.protocol),own_PID=G.os.getpid()))
  name='HIVE_complete_EEG_influence_20261103_v1'
  steps=[('GPU_probe',[sys.executable,'-B',str(Path(base['repository_root'])/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--',
                      sys.executable,'-B',str(code/'probe_HIVE_complete_EEG_influence_GPU_20261103_v1.py'),'--protocol',a.protocol,'--job-name',name],990),
         ('CPU_readout',[sys.executable,'-B',str(code/'readout_HIVE_complete_EEG_influence_CPU_20261103_v1.py'),'--protocol',a.protocol],120)]
  for label,command,limit in steps:
   then=time.monotonic()
   with (out/(label+'.stdout.log')).open('xb') as stdout,(out/(label+'.stderr.log')).open('xb') as stderr:
    worker=subprocess.Popen(command,cwd=code,stdout=stdout,stderr=stderr,start_new_session=True)
    G.write(out/(label+'_process.json'),dict(PID=worker.pid,command=command,UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat()))
    rc=worker.wait(timeout=limit)
   event=dict(stage=label,returncode=rc,seconds=time.monotonic()-then,stdout=G.ref(out/(label+'.stdout.log')),stderr=G.ref(out/(label+'.stderr.log')))
   if label=='GPU_probe':
    job_path=Path(base['repository_root'])/'audit'/(name+'_job.json');job=G.read(job_path)
    event['GPU_job']=G.ref(job_path);assert job['status']=='COMPLETED' and job['exit_code']==0 and job['wrapper_pid']==worker.pid
   assert rc==0,event;events.append(event);G.write(out/(label+'_complete.json'),event)
  probe=G.read(root/'probe/completion.json');readout=G.read(root/'readout/completion.json')
  assert probe['status']=='COMPLETE_FROZEN_FULL_UNET_ALL10_SUBJECT80_T_THREE_DOMAINS'
  assert readout['status']=='COMPLETE_ALL10_T80_FULL_UNET_FIXED_STATE_AND_TWO_PATH_READOUT'
  G.write(out/'completion.json',dict(status='COMPLETE_FROZEN_EEG_PROPAGATION_PROBE_AND_READOUT',protocol=G.ref(a.protocol),events=events,
     probe=G.ref(root/'probe/completion.json'),readout=G.ref(root/'readout/completion.json'),seconds=time.monotonic()-started,
     independent_CPU_raw_replay_pending=True,models_trained=0,new_images=0,NEW_contacted=False))
 except BaseException as error:
  G.write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-started,automatic_retry=False));raise
if __name__=='__main__':main()
