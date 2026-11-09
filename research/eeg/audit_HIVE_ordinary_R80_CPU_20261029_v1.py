"""Full2000 identities/pixels and independently written NumPy correction replay."""
import argparse,hashlib,json,os,sys,time
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def check(r):p=Path(r['path']);assert sha(p)==r['sha256'];return p
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')

def run(path):
 e=json.loads(Path(path).read_bytes());s=json.loads(check(e['protocol']).read_bytes());sys.path.insert(0,s['repository_root']);import brainhive_native_models_v1 as C
 import torch,numpy as np
 from scipy.special import erf
 from PIL import Image
 assert not torch.cuda.is_initialized();torch.set_num_threads(2);started=time.monotonic();root=Path(s['output_root']);out=root/'ordinary';assert not (out/'artifact_audit.json').exists()
 source=json.loads(check(s['input_completion']).read_bytes());bank=np.load(check(source['bank']),allow_pickle=False)
 c=bank['conditions'].reshape(2000,1024).astype(np.float64);gamma=np.repeat(bank['gammas'],200,axis=0).astype(np.float64);beta=np.repeat(bank['betas'],200,axis=0).astype(np.float64)
 weights=torch.load(check(s['ordinary']),weights_only=True,map_location='cpu')['model'];w={k:v.numpy().astype(np.float64) for k,v in weights.items()}
 u=(c-beta)/gamma;a=u@w['trunk.0.weight'].T+w['trunk.0.bias'];h=.5*a*(1+erf(a/np.sqrt(2)));d=h@w['direction.weight'].T+w['direction.bias'];d-=d.mean(1,keepdims=True);r2=np.square(u).sum(1,keepdims=True);d-=(d*u).sum(1,keepdims=True)/r2*u
 delta=.25*np.sqrt(r2)*d/np.sqrt(r2+np.square(d).sum(1,keepdims=True));scale=np.sqrt(r2)/np.linalg.norm(u+delta,axis=1,keepdims=True);predicted=c+gamma*((scale-1)*u+scale*delta)
 actual=np.concatenate([np.load(check(json.loads((root/'generation_attempts'/f'{i:04d}_{i+200:04d}/completion.json').read_bytes())['corrected_conditions']),allow_pickle=False) for i in range(0,2000,200)])
 assert actual.shape==(2000,1024) and actual.dtype==np.float32;maxerr=float(np.max(np.abs(actual-predicted)));relative=float(np.linalg.norm(actual-predicted)/np.linalg.norm(actual));assert maxerr<1e-4 and relative<1e-5
 baseline=[json.loads(x) for x in check(s['baseline_scoring_manifest']).read_text().splitlines()];assert len(baseline)==2000
 gallery=[json.loads(x) for x in check(s['gallery_manifest']).read_text().splitlines()];g={r['stimulus_id']:r for r in gallery};assert len(g)==200
 for r in gallery:assert sha(r['image_path'])==r['image_sha256']
 rows=[];inodes=set();paths=set();baseline_inodes=set()
 for i,old in enumerate(baseline):
  r=json.loads((out/'records'/f'{i:04d}.json').read_bytes());assert all(r[k]==old[k] for k in ('query_id','subject','stimulus_id','array_row','image_seed','repeat_protocol','EEG_source_stimulus_id'))
  assert r['ordinary_checkpoint']==s['ordinary'] and r['ordinary_strength']==1. and r['ordinary_radius']==.25 and r['UNet_NFE']==4 and r['native_IP_K_calls']==r['native_IP_V_calls']==280 and len(r['ancestral_noise_observations'])==4
  assert r['parent_checkpoint']==source['reports'][i//200]['model'] and r['source_execution']==ref(path)
  assert r['gamma_sha256']==hashlib.sha256(bank['gammas'][i//200].tobytes()).hexdigest() and r['beta_sha256']==hashlib.sha256(bank['betas'][i//200].tobytes()).hexdigest()
  assert not r['normal_inference_target_access'] and r['one_image_per_query']
  x=actual[i:i+1];assert hashlib.sha256(x.tobytes()).hexdigest()==r['condition_sha256'];assert hashlib.sha256(x.astype(np.float16).tobytes()).hexdigest()==r['actual_A_input_fp16_sha256']
  raw=bank['conditions'].reshape(2000,1024)[i:i+1];assert hashlib.sha256(raw.tobytes()).hexdigest()==r['original_condition_sha256']
  p=Path(r['generated_path']);assert sha(p)==r['generated_sha256'] and sha(old['generated_path'])==old['generated_sha256'];st=p.stat();oldst=Path(old['generated_path']).stat();assert (st.st_dev,st.st_ino)!=(oldst.st_dev,oldst.st_ino)
  assert str(p) not in paths and (st.st_dev,st.st_ino) not in inodes;paths.add(str(p));inodes.add((st.st_dev,st.st_ino));baseline_inodes.add((oldst.st_dev,oldst.st_ino))
  with Image.open(p) as im:assert im.size==(512,512);pixels=np.array(im.convert('RGB'),copy=True)
  assert hashlib.sha256(pixels.tobytes()).hexdigest()==r['generated_pixel_sha256']
  target=g[r['stimulus_id']];assert old['target_image_sha256']==target['image_sha256']
  r.update(concept_id=old['concept_id'],target_image_sha256=target['image_sha256'],target_image_path=target['image_path'],score_target_path=target['image_path'],target_mapping_attached_after_inference=True)
  rows.append(r)
 assert len(inodes)==len(baseline_inodes)==2000 and not inodes.intersection(baseline_inodes)
 manifest=out/'manifest.jsonl'
 with manifest.open('x') as f:
  for row in rows:f.write(json.dumps(row)+'\n')
 result=dict(status='PASS_ALL2000_ORDINARY_R80_IMAGES_SEEDS_TARGETS_AND_NUMPY_CORRECTIONS',queries=2000,subjects=10,gallery=200,manifest=ref(manifest),source_execution=ref(path),numpy64_vs_actual_GPU_max_absolute=maxerr,numpy64_vs_actual_GPU_relative=relative,actual_corrected_scalar_count=2048000,unique_ordinary_inodes=2000,baseline_inodes_disjoint=True,effect_scores_read=False,CUDA_initialized=False,seconds=time.monotonic()-started)
 write(out/'artifact_audit.json',result);print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execution',required=True);q=p.parse_args();run(q.execution)
