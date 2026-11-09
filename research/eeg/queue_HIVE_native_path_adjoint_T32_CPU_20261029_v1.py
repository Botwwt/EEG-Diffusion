"""Finite OLD-only probe after the exact running R80 lifecycle exits successfully."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np

PY='/root/miniconda3/bin/python'
def ref(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def live(identity):
 try:
  f=(Path('/proc')/str(identity['controller_PID'])/'stat').read_text().rsplit(')',1)[1].split()
  return f[0] not in ('Z','X') and int(f[19])==identity['start_ticks']
 except FileNotFoundError:return False
def run(path):
 s=json.loads(Path(path).read_bytes());assert s['host']=='old';root=Path(s['output_root']);logs=root/'queue';logs.mkdir();began=time.monotonic();events=[]
 try:
  for item in s['source_pins']:assert ref(item['path'])==item
  identity=s['wait_for_R80_controller'];progress=logs/'waiting.json'
  write(progress,dict(status='WAITING_FOR_EXACT_OLD_R80_FULL_LIFECYCLE',identity=identity,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
  while live(identity):
   assert time.monotonic()-began<10800,'Finite three-hour wait elapsed; preserve pending work without launching'
   time.sleep(5)
  R=Path(s['repository_root']);prior=Path(s['wait_for_R80_output'])/'full_queue'
  assert not (prior/'failure.json').exists(),'R80 lifecycle failed; do not hide the failure by automatically running another direction'
  assert json.loads((prior/'completion.json').read_bytes())['status']=='COMPLETE_FIXED_ORDINARY_R80_GENERATION_AND_FULL_PAIRED_EVALUATION'
  assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
  for lo,hi in ((0,16),(16,32)):
   assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
   name=f'HIVE_native_path_adjoint_T32_{lo:02d}_{hi:02d}_20261029_v1'
   cmd=[PY,'-B',str(R/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--',PY,'-B',str(Path(path).parent/'probe_HIVE_native_path_adjoint_T32_GPU_20261029_v1.py'),'--protocol',str(path),'--lo',str(lo),'--hi',str(hi)]
   with (logs/(name+'.stdout.log')).open('xb') as so,(logs/(name+'.stderr.log')).open('xb') as se:
    child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True,cwd=R);rc=child.wait(timeout=960)
   event=dict(name=name,returncode=rc,seconds=time.monotonic()-began);events.append(event);write(logs/(name+'.receipt.json'),event);assert rc==0,'Probe failure retained; no automatic retry or changed precision gate'
  for lo,hi in ((0,16),(16,32)):
   complete=json.loads((root/f'segment_{lo:02d}_{hi:02d}/completion.json').read_bytes());assert complete['queries']==16 and complete['all_native_endpoints_bit_exact'] and complete['protocol']==ref(path)
  rows=[];values=[]
  for index,q in enumerate(s['queries']):
   segment=root/('segment_00_16' if index<16 else 'segment_16_32');r=json.loads((segment/f'query_{index:02d}.json').read_bytes());assert r['query']==q and ref(r['archive']['path'])==r['archive'];z=np.load(r['archive']['path'],allow_pickle=False)
   u=z['u'].astype(np.float64);basis=np.stack([np.ones(1024),u]);gram=basis@basis.T
   def tangent(g):
    g=np.asarray(g,dtype=np.float64);return g-basis.T@np.linalg.solve(gram,basis@g)
   ge=tangent(z['grad_end']);gp=tangent(z['grad_epsilon'].astype(np.float64).mean(axis=0)+.1*z['grad_token'].astype(np.float64));den=np.linalg.norm(ge)*np.linalg.norm(gp);assert den>1e-20
   cosine=float(ge@gp/den);slope=float(-ge@gp/np.linalg.norm(gp));assert np.isfinite([cosine,slope]).all();values.append(cosine);rows.append(dict(index=index,query=q,tangent_cosine_point_vs_endpoint=cosine,predicted_endpoint_slope_along_unit_point_descent=slope,endpoint_grad_norm=float(np.linalg.norm(ge)),point_grad_norm=float(np.linalg.norm(gp)),record=ref(segment/f'query_{index:02d}.json')))
  groups={}
  for i,q in enumerate(s['queries']):groups.setdefault(q['concept_id'],[]).append(i)
  keys=sorted(groups);rng=np.random.default_rng(1729);arr=np.array(values);boot=[]
  for _ in range(1000):
   drawn=rng.integers(len(keys),size=len(keys));indices=[i for j in drawn for i in groups[keys[j]]];boot.append(float(arr[indices].mean()))
  result=dict(status='COMPLETE_ALL32_FROZEN_NATIVE_PATH_INPUT_GRADIENT_DIAGNOSTIC',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),queries=32,concepts=len(keys),protocol=ref(path),events=events,rows=rows,mean_tangent_cosine=float(arr.mean()),negative_cosine_queries=int((arr<0).sum()),descriptive_concept_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),all32_native_latents_bit_exact=True,old_uniform_t_draws=128,new_model_updates=0,new_research_images=0,automatic_followup_training=False,seconds_including_wait=time.monotonic()-began,limits=s['interpretation_limits'],SOTA_established=False)
  write(root/'completion.json',result);print(json.dumps({k:result[k] for k in ('status','queries','mean_tangent_cosine','negative_cosine_queries','descriptive_concept_bootstrap95')}))
 except BaseException as error:
  write(logs/'failure.json',dict(status='PATH_ADJOINT_QUEUE_FAILED_PRESERVE_ALL_PREFIXES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(error),traceback=traceback.format_exc(),events=events));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);q=p.parse_args();run(q.protocol)
