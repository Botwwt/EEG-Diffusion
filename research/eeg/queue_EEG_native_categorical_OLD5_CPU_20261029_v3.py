"""Finite OLD-only continuation; retain first80 and never read partial effects."""
import argparse,datetime,hashlib,json,subprocess,time,traceback
from pathlib import Path


def ref(p):
 p=Path(p)
 with p.open('rb') as f: h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)


def write(p,x):
 with p.open('x') as f: json.dump(x,f,indent=2);f.write('\n')


def run(path):
 s=json.loads(Path(path).read_bytes());assert s['status']=='REGISTERED_CATEGORICAL_OLD_ONLY_EXACT_LOGITS_STAGE400_v2' and s['execution_host']=='old'
 root=Path(s['queue_root']);root.mkdir();started=time.monotonic();events=[]
 try:
  for item in s['source_pins']:assert ref(item['path'])==item
  for lo in range(80,400,20):
   assert time.monotonic()-started<3600
   assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
   job=f'categorical_OLD5_{lo:03d}_20261029_v3'
   cmd=['/root/miniconda3/bin/python','-B',str(Path(s['repository_root'])/'run_gpu_job_v2.py'),'--name',job,'--max-seconds','900','--','/root/miniconda3/bin/python','-B',str(Path(path).parent/'generate_EEG_native_categorical_OLD5_GPU_20261029_v2.py'),'--spec',str(path),'--lo',str(lo),'--hi',str(lo+20),'--job-name',job]
   with (root/(job+'.stdout.log')).open('xb') as so,(root/(job+'.stderr.log')).open('xb') as se:
    child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True,cwd=Path(path).parent)
    try:rc=child.wait(timeout=950)
    except subprocess.TimeoutExpired:
     # Wrapper's own900s deadline owns cleanup; do not launch another child.
     raise RuntimeError('Original finite wrapper still missing terminal status; inspect same handle; no automatic retry')
   events.append(dict(job=job,returncode=rc,range=[lo,lo+20]));write(root/f'committed_{lo:04d}.json',events[-1]);assert rc==0,events[-1]
   p=Path(s['output_root'])/f'chunk_{lo:04d}_{lo+20:04d}/completion.json';done=json.loads(p.read_bytes())
   assert done['status']=='COMPLETE_CATEGORICAL_OLD_ONLY_FIVE_ARM_IMAGE_CHUNK_v2' and done['spec']==ref(path) and done['range']==[lo,lo+20]
  import numpy as np
  from PIL import Image
  panel=json.loads(Path(s['panel']['path']).read_bytes());paths=set();inodes=set();total=0
  feature_ranges={arm:[] for arm in s['arms']}
  for lo in range(80,400,20):
   p=Path(s['output_root'])/f'chunk_{lo:04d}_{lo+20:04d}/completion.json';done=json.loads(p.read_bytes())
   for arm in s['arms']:
    f=done['features'][arm];assert ref(f['path'])==f;x=np.load(f['path'],allow_pickle=False)
    assert x.shape==(20,768) and x.dtype==np.float32 and np.isfinite(x).all() and np.max(np.abs(np.linalg.norm(x,axis=1)-1))<2e-5
    r=done['records'][arm];assert ref(r['path'])==r;rows=json.loads(Path(r['path']).read_bytes());assert len(rows)==20
    for index,row in enumerate(rows,lo):
     q=panel['queries'][index];assert row['index']==index and row['arm']==arm and all(row[k]==q[k] for k in ('query_id','stimulus_id','array_row'))
     image=Path(row['image']['path']);assert ref(image)==row['image'];stat=image.stat();assert str(image) not in paths and (stat.st_dev,stat.st_ino) not in inodes
     paths.add(str(image));inodes.add((stat.st_dev,stat.st_ino))
     with Image.open(image) as im:assert im.size==(256,256);pixels=np.array(im.convert('RGB'),copy=True)
     assert hashlib.sha256(pixels.tobytes()).hexdigest()==row['pixel_SHA256']
     assert all(np.std(pixels[:,:,k])>1e-3 for k in range(3));total+=1
    feature_ranges[arm].extend(range(lo,lo+20))
  assert total==1600 and all(v==list(range(80,400)) for v in feature_ranges.values())
  audit=root/'independent_new320_artifact_CPU_audit.json'
  write(audit,dict(status='PASS_OLD_ONLY_ALL5_NEW320_PIXELS_IDENTITIES_FEATURES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),images=total,unique_inodes=len(inodes),queries_per_arm=320,first80_reused_from_original_receipts=True,effect_scores_read=False,CUDA_imported=False))
  write(root/'completion.json',dict(status='COMPLETE_CATEGORICAL_OLD_ONLY_FIVE_ARM_STAGE400_CONTINUATION',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(path),events=events,new_research_images=1600,original80_images_reused=400,artifact_audit=ref(audit),seconds=time.monotonic()-started,effect_scores_read=False))
 except BaseException as error:
  write(root/'failure.json',dict(status='OLD_ONLY_CATEGORICAL_QUEUE_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(error),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-started));raise


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);q=p.parse_args();run(q.spec)
