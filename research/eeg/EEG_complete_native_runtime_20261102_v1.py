"""Shared EEG-only model initialization and SHA-bound transaction helpers."""
import hashlib,os,sys
from pathlib import Path
import EEG_complete_native_study_contract_CPU_20261102_v1 as C
def cpu_tree(x):
 import torch
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu_tree(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
 return x
def tree_equal(a,b):
 import torch
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
 return a==b
def digest(state):
 import torch
 h=hashlib.sha256()
 for name,value in sorted(state.items()):
  if isinstance(value,torch.Tensor):
   x=value.detach().cpu().contiguous();h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def save_state(path,state):
 import torch
 path=Path(path);assert path.resolve().is_relative_to(C.OUT.resolve());pending=path.with_name(path.name+'.pending');assert not pending.exists()
 with pending.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
 os.replace(pending,path);return C.ref(path)
def model(q,arm,device):
 import torch
 C.roots();base,a,image,stats=C.source_inputs(q,binary=False)
 from EEG_complete_native_study_core_20261102_v1 import EEGDistribution,ImagePrior
 if arm=='image_prior_full':p=ImagePrior()
 else:
  from eeg_labram_pretrained_20261028_v1 import build_encoder
  encoder=build_encoder(a['author_source'],a['vendor'],seed=1729)
  warm=torch.load(C.checked(base['full_EEG']['model']),map_location='cpu',weights_only=True);assert warm['role']=='full' and warm['epoch']==50 and warm['updates']==5200
  encoder.load_state_dict(warm['model'],strict=True);p=EEGDistribution(encoder,arm)
 p=p.to(device).eval();return p
def channels(q):
 C.roots();base,a,image,stats=C.source_inputs(q,binary=False)
 from models.labram import get_input_chans
 return get_input_chans([x.upper() for x in stats['channel_names']])
def inputs(q,role='T'):
 import numpy as np
 done=C.read(C.OUT/'inputs/completion.json');assert done['status']=='PASS_COLD_COMPLETE_T13230_DCHECK400_REAL_EEG_INPUT_JOIN_TRANSFORM' and done['protocol']==C.ref(C.CODE/'protocol.json')
 x=np.load(C.checked(done['files'][role]),mmap_mode='r',allow_pickle=False);assert x.shape==((13230 if role=='T' else 400),63,1,200) and x.dtype==np.float32;return x
