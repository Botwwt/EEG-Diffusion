"""CPU wait for exact two-fit terminal; bind finals then all R80, no selection."""
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_owned_process_CPU_20261029_v1 import capture_wrapper_child,stop_owned
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(x):assert ref(x['path'])==x;return Path(x['path'])
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def atomic(p,x):
 p=Path(p);temp=p.with_name(p.name+'.pending');temp.write_text(json.dumps(x,indent=2)+'\n');os.replace(temp,p)
def live(pid,ticks):
 p=Path('/proc')/str(pid)/'stat'
 if not p.exists():return False
 f=p.read_text().rsplit(')',1)[1].split();return f[0] not in ('Z','X') and int(f[19])==ticks
def run(path):
 study=read(path);assert study['host']=='old' and study['training_seeds']==[1729,2027,3407];fit=read(checked(study['training_protocol']));root=Path(study['output_root']);root.mkdir();code=Path(path).parent;R=Path(study['repository_root']);fitroot=Path(fit['output_root']);active=None;pinned=None;events=[];phase='WAIT_EXACT_FIXED_FITS';started=time.monotonic()
 def call(name,command,GPU=False,seconds=1800):
  nonlocal active,pinned
  if GPU:
   assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();command=[sys.executable,'-B',str(R/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--']+command
  with (root/(name+'.stdout.log')).open('xb') as so,(root/(name+'.stderr.log')).open('xb') as se:
   active=subprocess.Popen(command,cwd=R,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True);pinned=capture_wrapper_child(active,R/'audit'/(name+'_job.json')) if GPU else None;st=(Path('/proc')/str(active.pid)/'stat').read_text().rsplit(')',1)[1].split();atomic(root/'progress.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,child_PID=active.pid,start_ticks=int(st[19]),command=command,events=events));rc=active.wait(timeout=seconds+60);active=None;pinned=None
  event=dict(name=name,exit_code=rc,GPU=GPU,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());events.append(event);write(root/(name+'_terminal.json'),event);assert rc==0,event
 try:
  for x in study['source_pins']+study['payloads']:checked(x)
  while True:
   assert time.monotonic()-started<102000;is_live=live(study['upstream_controller_PID'],study['upstream_start_ticks']);failure=fitroot/'queue_failure.json'
   if failure.exists():raise RuntimeError('Exact fixed-fit queue failed; dependency not admitted: '+str(failure))
   if not is_live:
    assert (fitroot/'training_completion.json').is_file(),'Missing exact training handle without success closure; no restart';done=read(fitroot/'training_completion.json');assert done['protocol']==study['training_protocol'] and done['seeds']==[2027,3407];break
   atomic(root/'progress.json',dict(phase=phase,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),upstream_controller_live=True,upstream_progress=read(fitroot/'queue_progress.json'),new_GPU_jobs_started=0));time.sleep(30)
  phase='INDEPENDENT_FINAL_FIT_AUDIT';audit=root/'both_final_fit_CPU_audit.json';call('audit_both_fits',[sys.executable,'-B',str(code/'audit_HIVE_ordinary_seed_final_CPU_20261029_v1.py'),'--study',str(path),'--output',str(audit)],seconds=300);finals=read(audit);assert finals['status']=='PASS_BOTH_FIXED_FINAL_SEEDS_ALL10_OOF_THREE_PASSES_MODEL_ADAM_COVERAGE_AND_DFIT_CALIBRATION'
  old=read(checked(study['original_R80_protocol']));olde=read(checked(study['original_R80_execution']));runs=[study['original_seed1729_run']]
  for item in finals['reports']:
   number=item['seed'];phase=f'BIND_SEED{number}';s=dict(old);s.update(status='REGISTERED_SEED_REPLICATE_R80_MATCHED_BENCHMARK_v1',output_root=str(root/f'seed{number}'),ordinary=item['component'],ordinary_strength=item['selected_strength'],training_seed=number,training_protocol=study['training_protocol'],final_training_CPU_audit=ref(audit),source_pins=study['source_pins'],reserved_disk_bytes=6<<30,training_seed_variability_claim=False,official200_already_exposed=True,independent_H=False)
   spec=code/f'seed{number}_R80_protocol.json';write(spec,s);phase=f'PROFILE_SEED{number}';job=f'HIVE_seed{number}_R80_profile_20261029_v1';call(job,[sys.executable,'-B',str(code/'profile_HIVE_seed_replicate_R80_GPU_20261029_v1.py'),'--spec',str(spec),'--job-name',job],GPU=True,seconds=900)
   profile=Path(s['output_root'])/'profile/completion.json';p=read(profile);assert p['ordinary_strength']==s['ordinary_strength'] and p['all10_parent_pixels_exact'] and p['all10_ordinary_native_RNG_paired']
   e=dict(olde);e.update(status='BOUND_SEED_REPLICATE_R80_FULL_LIFECYCLE_v1',protocol=ref(spec),profile_completion=ref(profile),source_pins=study['source_pins'],final_training_CPU_audit=ref(audit),training_seed=number,official200_already_exposed=True)
   execution=code/f'seed{number}_R80_execution.json';write(execution,e);phase=f'FULL_R80_SEED{number}';call(f'seed{number}_full_R80_lifecycle',[sys.executable,'-B',str(code/'queue_HIVE_seed_replicate_R80_CPU_20261029_v1.py'),'--execution',str(execution)],seconds=21600)
   runroot=Path(s['output_root']);runs.append(dict(seed=number,checkpoint=item['component'],lifecycle_completion=ref(runroot/'full_queue/completion.json'),common_csv=ref(runroot/'evaluation/scores/per_query.csv'),paper_csv=ref(runroot/'evaluation/original_paper_metrics/per_query.csv'),paired_summary=ref(runroot/'evaluation/paired_comparison/summary.json')))
  phase='JOINT_ALL3_SEEDS_FULL16';joint=dict(study);joint.update(runs=runs,joint_output=str(root/'joint_all3_seeds'));jointpath=code/'bound_all3_seed_comparison.json';write(jointpath,joint);call('joint_all3_seeds',[sys.executable,'-B',str(code/'compare_HIVE_ordinary_three_seeds_CPU_20261029_v1.py'),'--study',str(jointpath)],seconds=1800)
  write(root/'completion.json',dict(status='COMPLETE_TWO_NEWSEED_R80_AND_ALL3_SEED_FULL16_COMPARISON',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),study=ref(path),seeds=[1729,2027,3407],new_research_images=4000,events=events,joint_summary=ref(root/'joint_all3_seeds/summary.json'),SOTA_established=False))
 except BaseException as error:
  if active is not None:stop_owned(active,pinned)
  write(root/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events));raise
if __name__=='__main__':
 def stop(*_):raise InterruptedError('Explicit dependent R80 study stop')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);p=argparse.ArgumentParser();p.add_argument('--study',required=True);run(p.parse_args().study)
