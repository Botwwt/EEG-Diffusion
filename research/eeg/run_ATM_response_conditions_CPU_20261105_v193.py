"""Canonical T80 four frozen EEG models, independently repeated in CPU64."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
q=json.loads(Path(sys.argv[1]).read_bytes());root=Path(q['parent_root']);out=Path(q['output_root']);started=time.monotonic()
sys.path.insert(0,str(root/'code'));import atm_fixed_projection_contract_20261105_v190 as C
c=C.setup(root/'code/policy.json');K=c['K'];C.resources(c,True)
assert K.ref(__file__)==q['sources']['conditions'];rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
sys.path[:0]=c['train']['physical']['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
torch.set_num_threads(2)
try:
 from atm_native_response_geometry_20261105_v193 import numpy_attention,numpy_measures
 fixture=np.random.default_rng(1729).normal(size=(14,2,16,5))
 dd,ss,ee,aa=numpy_measures(fixture,3)
 assert np.max(abs(dd[:,:,0]-dd[:,:,1]-dd[:,:,2]))<1e-12
 assert np.max(abs(dd[1:,3]-dd[:1,3]-ee+2*aa))<1e-12
 zero=np.zeros((1,2,16,5));kk=np.ones((14,2,4,5));vv=np.arange(14*2*4*5).reshape(14,2,4,5).astype(float)
 assert np.max(abs(numpy_attention(zero,kk,vv)-vv.mean(-2,keepdims=True)))<1e-12
 ids=np.asarray(q['T80_array_rows'],np.int64);assert np.array_equal(ids,c['train']['T'].reshape(-1,10)[:8].reshape(-1))
 eeg=torch.load(K.checked(q['EEG']),map_location='cpu',weights_only=True,mmap=True)[ids].clone()
 target=torch.load(K.checked(q['visual_target']),map_location='cpu',weights_only=True,mmap=True)[ids].clone()
 source={k:torch.load(K.checked(v),map_location='cpu',weights_only=True,mmap=True)['model'] for k,v in q['models'].items()}
 states=dict(NN=source['NN'],CC=source['CC'],CN={n:source['NN'][n] if n.startswith('proj_eeg.') else v for n,v in source['CC'].items()},FH=source['FH'])
 values=[];errors={}
 for arm,state in states.items():
  net=ATMS().eval().requires_grad_(False);net.load_state_dict(state,strict=True)
  with torch.inference_mode():
   a=net(eeg,torch.ones(80,dtype=torch.long)).numpy();net.double();b=net(eeg.double(),torch.ones(80,dtype=torch.long)).numpy()
  np.testing.assert_allclose(a,b,atol=3e-5,rtol=3e-5);values.append(a);errors[arm]=float(abs(a-b).max());del net
  assert time.monotonic()-started<110 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<3<<30
 np.savez_compressed(out/'conditions.npz',conditions=np.stack(values,1),teacher=target.numpy(),T_rows=ids)
 assert not torch.cuda.is_initialized();C.resources(c)
 K.write(out/'conditions_completion.json',dict(status='PASS_ALL320_T80_CPU32_CPU64_FROZEN_CONDITIONS',conditions=K.ref(out/'conditions.npz'),errors=errors,models=q['models'],canonical_precision='native CPU32; CPU64 all320 independently repeated; no old image identity implied',seconds=time.monotonic()-started,new_updates=0))
except BaseException as exc:
 K.write(out/'conditions_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
