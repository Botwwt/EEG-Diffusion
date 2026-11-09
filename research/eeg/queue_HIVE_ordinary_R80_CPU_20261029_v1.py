"""OLD-only full2000 generation, complete artifacts/features and fixed metrics."""
import argparse,datetime,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
PY='/root/miniconda3/bin/python'
def ref(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def run(path):
 e=json.loads(Path(path).read_bytes());s=json.loads(Path(e['protocol']['path']).read_bytes());assert ref(e['protocol']['path'])==e['protocol'] and s['host']=='old'
 root=Path(s['output_root']);logs=root/'full_queue';logs.mkdir();code=Path(path).parent;R=Path(s['repository_root']);started=time.monotonic();events=[]
 def guarded(script,args,name):return [PY,'-B',str(code/'run_HIVE_R80_evaluation_guarded_20261029_v1.py'),'--execution',str(path),'--audit-file',str(root/'evaluation_resources'/(name+'.json')),'--script',str(script),'--']+args
 def call(name,command,GPU=False,seconds=900):
  assert time.monotonic()-started<21600
  if GPU:
   assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
   command=[PY,'-B',str(R/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--']+command
  with (logs/(name+'.stdout.log')).open('xb') as so,(logs/(name+'.stderr.log')).open('xb') as se:
   child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True,cwd=R);rc=child.wait(timeout=seconds+60)
  ev=dict(name=name,returncode=rc,GPU=GPU,seconds_since_queue_start=time.monotonic()-started);events.append(ev);write(logs/(name+'.receipt.json'),ev);return rc
 try:
  for p in s['source_pins']+e['source_pins']:assert ref(p['path'])==p
  profile=json.loads(Path(e['profile_completion']['path']).read_bytes());assert ref(e['profile_completion']['path'])==e['profile_completion'] and profile['all10_parent_pixels_exact'] and profile['all10_ordinary_native_RNG_paired']
  for lo in range(0,2000,200):
   name=f'HIVE_ordinary_R80_subject{lo//200+1:02d}_20261029_v1';assert call(name,[PY,'-B',str(code/'generate_HIVE_ordinary_R80_GPU_20261029_v1.py'),'--execution',str(path),'--lo',str(lo),'--hi',str(lo+200),'--job-name',name],True)==0
   c=json.loads((root/'generation_attempts'/f'{lo:04d}_{lo+200:04d}/completion.json').read_bytes());assert c['queries']==200 and not c['effect_scores_read']
  assert call('artifact_full2000',[PY,'-B',str(code/'audit_HIVE_ordinary_R80_CPU_20261029_v1.py'),'--execution',str(path)],False,1800)==0
  manifest=root/'ordinary/manifest.jsonl';features=root/'evaluation/features';gallery=s['gallery_manifest']['path']
  for i in range(3):
   name=f'HIVE_ordinary_R80_features_segment{i}_20261029_v1';rc=call(name,guarded(R/'extract_image_features.py',['--manifest',str(manifest),'--output',str(features)],name),True)
   if rc==0:break
   job=json.loads((R/'audit'/(name+'_job.json')).read_bytes());assert job.get('termination_reason')=='Finite registered job timeout; preserve resumable checkpoints','Only registered finite timeouts may continue committed feature prefixes'
   for p in features.rglob('*.tmp'):
    assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(features.resolve());p.rename(p.with_name(p.name+f'.segment{i}_incomplete_retained'))
  else:raise RuntimeError('Fixed three feature segments exhausted; preserve prefixes')
  assert json.loads((features/'completion.json').read_bytes())['queries']==2000
  assert call('common_score_full2000',guarded(R/'score_full_manifest.py',['--manifest',str(manifest),'--features',str(features),'--gallery-manifest',gallery,'--gallery-features',e['gallery_features'],'--output',str(root/'evaluation/scores')],'common_score_full2000'),False,1800)==0
  assert call('low_level_full2000',guarded(R/'score_low_level.py',['--manifest',str(manifest),'--gallery-manifest',gallery,'--output',str(root/'evaluation/low_level')],'low_level_full2000'),False,1800)==0
  for i in range(3):
   name=f'HIVE_ordinary_R80_author_segment{i}_20261029_v1';rc=call(name,guarded(code/'score_HIVE_ordinary_R80_author_GPU_20261029_v1.py',['--execution',str(path)],name),True)
   if rc==0:break
   job=json.loads((R/'audit'/(name+'_job.json')).read_bytes());assert job.get('termination_reason')=='Finite registered job timeout; preserve resumable checkpoints','No automatic retries for non-timeout errors'
  else:raise RuntimeError('Fixed three author metric segments exhausted')
  assert call('paired_full2000',[PY,'-B',str(code/'compare_HIVE_ordinary_R80_CPU_20261029_v1.py'),'--execution',str(path)],False,1800)==0
  write(logs/'completion.json',dict(status='COMPLETE_FIXED_ORDINARY_R80_GENERATION_AND_FULL_PAIRED_EVALUATION',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution=ref(path),events=events,queries=2000,new_research_images=2000,engineering_images=30,seconds=time.monotonic()-started,SOTA_established=False))
 except BaseException as error:
  write(logs/'failure.json',dict(status='ORDINARY_R80_LIFECYCLE_FAILURE_PRESERVED_NO_SCIENCE_RETRY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(error),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-started));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execution',required=True);q=p.parse_args();run(q.execution)
