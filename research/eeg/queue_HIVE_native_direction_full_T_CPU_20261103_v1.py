"""One finite OLD queue: all ten subjects, three matched arms, three full T passes."""
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import brainhive_native_direction_full_contract_20261103_v1 as C
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();started=time.monotonic();s,e,old,base,G,H=C.setup(a.protocol);out=Path(s['output_root'])/'queue';out.mkdir();current=None
 def stopped(sig,frame):raise SystemExit('Queue received signal '+str(sig))
 signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
 try:
  C.resources(s,e,True);command=[sys.executable,'-B',str(Path(a.protocol).parent/'prepare_HIVE_native_direction_full_T_CPU_20261103_v1.py'),'--protocol',a.protocol]
  with (out/'preflight.stdout.log').open('xb') as stdout,(out/'preflight.stderr.log').open('xb') as stderr:
   cpu=subprocess.Popen(command,cwd=Path(a.protocol).parent,stdout=stdout,stderr=stderr,start_new_session=True);G.write(out/'preflight_process.json',dict(PID=cpu.pid,command=command));rc=cpu.wait(timeout=300)
  assert rc==0 and C.read(Path(s['output_root'])/'inputs/completion.json')['status']=='COMPLETE_T10_THREE_PASS_ALL_CONCEPT_PAIR_ORDERS_AND_NATIVE_INPUTS_CPU';results=[]
  for si,subject in enumerate(s['subjects']):
   for ai,arm in enumerate(C.ARMS):
    for segment in range(s['maximum_segments_per_model']):
     assert time.monotonic()-started<s['maximum_queue_seconds']-1000;C.resources(s,e,True)
     name=f'HIVE_native_direction_full_20261103_s{si+1:02d}_a{ai}_seg{segment:02d}'
     command=[sys.executable,'-B',str(Path(base['repository_root'])/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--',sys.executable,'-B',str(Path(a.protocol).parent/'train_HIVE_native_direction_full_T_GPU_20261103_v1.py'),'--protocol',a.protocol,'--subject',subject,'--arm',arm,'--segment',str(segment),'--job-name',name]
     label=f'{subject}_{arm}_segment{segment:03d}';log=out/label
     with log.with_suffix('.stdout.log').open('xb') as stdout,log.with_suffix('.stderr.log').open('xb') as stderr:
      worker=subprocess.Popen(command,cwd=Path(a.protocol).parent,stdout=stdout,stderr=stderr,start_new_session=True);current=dict(PID=worker.pid,command=command,name=name,subject=subject,arm=arm,segment=segment)
      G.write(out/(label+'_process.json'),current);C.atomic_json(out/'current.json',current);rc=worker.wait(timeout=990)
     jobpath=Path(base['repository_root'])/'audit'/(name+'_job.json');job=C.read(jobpath);assert rc==0 and job['status']=='COMPLETED' and job['exit_code']==0 and job['wrapper_pid']==worker.pid
     segmentpath=Path(s['output_root'])/'models'/subject/arm/f'segment_{segment:03d}'/'completion.json';done=C.read(segmentpath)
     assert done['subject']==subject and done['arm']==arm and done['updates_this_segment']>0
     G.write(out/(label+'_closed.json'),dict(segment=C.ref(segmentpath),job=C.ref(jobpath),wrapper_returncode=rc))
     if done['status']=='COMPLETE_FIXED_EPOCH3_NATIVE_EEG_MODEL':
      assert done['updates']==2481 and done['primary_exposures']==39690;results.append(C.ref(Path(s['output_root'])/'models'/subject/arm/'completion.json'));break
     assert done['status']=='RESUMABLE_FIXED_NATIVE_EEG_SEGMENT_SOFT_LIMIT'
    else:raise RuntimeError('Finite segment allocation exhausted; keep actual checkpoint, no reset or automatic extension')
  assert len(results)==30
  G.write(out/'completion.json',dict(status='COMPLETE_ALL30_NATIVE_EEG_FIXED_EPOCH3_T_FITS_READY_FOR_FROZEN_IMAGE_EVALUATION',protocol=C.ref(a.protocol),models=results,
   total_updates=74430,primary_exposures=1190700,seconds=time.monotonic()-started,ordinary_immutable=True,NEW_contacted=False,new_images=0,image_effect_unknown=True,publication_ready=False,SOTA=False))
 except BaseException as error:
  if current and 'worker' in locals() and worker.poll() is None:
   # The legacy wrapper has no TERM handler. Stop only its recorded child group,
   # then let the wrapper record the real signal exit and charged time.
   jp=Path(base['repository_root'])/'audit'/(current['name']+'_job.json')
   if jp.exists():
    job=C.read(jp);assert job['wrapper_pid']==worker.pid;child=job['pid']
    if Path('/proc',str(child)).exists():
     cmd=Path('/proc',str(child),'cmdline').read_bytes();assert b'train_HIVE_native_direction_full_T_GPU_20261103_v1.py' in cmd and current['name'].encode() in cmd
     os.killpg(child,signal.SIGTERM)
     try:worker.wait(timeout=35)
     except subprocess.TimeoutExpired:
      if Path('/proc',str(child)).exists():os.killpg(child,signal.SIGKILL)
      worker.wait(timeout=15)
   else:
    worker.terminate();worker.wait(timeout=10)
  if 'cpu' in locals() and cpu.poll() is None:
   os.killpg(cpu.pid,signal.SIGTERM)
   try:cpu.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(cpu.pid,signal.SIGKILL);cpu.wait(timeout=10)
  G.write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),current=current,seconds=time.monotonic()-started,automatic_retry=False));raise
if __name__=='__main__':main()
