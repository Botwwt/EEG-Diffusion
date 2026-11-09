"""Independent CPU64 all600 inference and fixed projection/optimizer checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,hashlib,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_fixed_projection_contract_20261105_v190 as C
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=['fit_audit'],required=True);a=ap.parse_args();start=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
c=C.setup(a.policy);q,K,tc=c['q'],c['K'],c['train'];out=c['fit'];assert K.ref(__file__)==q['sources']['fit_CPU'];C.resources(c)
sys.path[:0]=tc['physical']['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
torch.set_num_threads(2)
def same(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
try:
 base=torch.load(K.checked(tc['q']['checkpoint']),map_location='cpu',weights_only=True,mmap=True);done=K.read(out/'fit_10_completion.json');assert done['updates']==830 and done['exposures']==132300
 final=torch.load(K.checked(done['checkpoint']),map_location='cpu',weights_only=True,mmap=True);net=ATMS();names=[n for n,p in net.named_parameters()];pids=base['optimizer']['param_groups'][0]['params'];assert pids==final['optimizer']['param_groups'][0]['params'] and len(names)==len(pids)
 changed=[];frozen=0
 for name,pid in zip(names,pids):
  if name.startswith('proj_eeg.'):
   assert torch.equal(base['model'][name],final['model'][name]) and same(base['optimizer']['state'].get(pid),final['optimizer']['state'].get(pid));frozen+=base['model'][name].numel()
  elif not torch.equal(base['model'][name],final['model'][name]):changed.append(name)
 assert frozen==2527232 and any(n.startswith('encoder.') for n in changed) and any(n.startswith('enc_eeg.') for n in changed)
 epochs=[json.loads(s) for s in (out/'epochs.jsonl').read_text().splitlines()];assert len(epochs)==10
 for e in range(10):
  v=epochs[e];order=np.random.default_rng(1739+e).permutation(1323);assert v['epoch']==e+1 and v['updates']==83*(e+1) and v['exposures']==13230*(e+1)
  assert v['coverage_min']==v['coverage_max']==1 and v['concept_order_sha256']==hashlib.sha256(order.tobytes()).hexdigest()
 net.load_state_dict(final['model'],strict=True);net.double().eval();mean=torch.load(K.checked(tc['physical']['EEG']),map_location='cpu',weights_only=True,mmap=True);test=torch.load(K.checked(tc['R80']['EEG_R80']),map_location='cpu',weights_only=True,mmap=True)
 saved=np.load(K.checked(K.read(out/'conditions_completion.json')['conditions']));ids=[r['array_row'] for r in tc['queries']];mx=0.;sq=0.;source_square=0.;count=0
 with torch.inference_mode():
  for lo in range(0,600,50):
   x=mean[ids[lo:lo+50]] if lo<400 else test[lo-400:lo-350]
   pred=net(x.double(),torch.ones(len(x),dtype=torch.long)).numpy();delta=pred-saved[lo:lo+50];mx=max(mx,float(abs(delta).max()));sq+=float((delta**2).sum());source_square+=float((pred**2).sum());count+=len(x)
   np.testing.assert_allclose(pred,saved[lo:lo+50],atol=3e-5,rtol=3e-5)
   assert time.monotonic()-start<850 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<3<<30
 assert count==600
 C.resources(c);K.write(out/'fit_audit_completion.json',dict(status='PASS_ALL600_CPU64_FIXED_HEAD_FULL10_EPOCHS',checkpoint=done['checkpoint'],conditions=K.ref(out/'conditions.npy'),projection_parameters=frozen,projection_weights_and_Adam_exact=True,changed_nonprojection_names=changed,
  full_T_epochs=10,updates=830,exposures=132300,all_orders_coverage_exact=True,CPU64_forward_queries=count,max_abs_error=mx,relative_L2=(sq/source_square)**.5,CUDA_initialized=torch.cuda.is_initialized(),seconds=time.monotonic()-start))
 assert not torch.cuda.is_initialized()
except BaseException as exc:
 K.write(out/'fit_audit_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
