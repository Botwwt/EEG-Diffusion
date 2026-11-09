"""Read complete bank rows only after canonical whole success and exact exit."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
BANK_PATH='/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/multiscale_EEG_20261028_v1/VAR_full_T_native_bank_code_20261102_v1/protocol.json'
BANK_SHA='26994a32e29844a8423a8222fed1022116d11fb2f3e3b5ff6b94028287196c2b'
def checked(pin):
 p=Path(pin['path']);assert p.is_file() and not p.is_symlink()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 assert p.stat().st_size==pin['bytes'] and h==pin['sha256'];return p
def read(p):return json.loads(Path(p).read_bytes())
class CompleteNativeBank:
 def __init__(self,protocol):
  assert protocol['path']==BANK_PATH and protocol['sha256']==BANK_SHA and protocol['bytes']==3379
  q=read(checked(protocol));pin=next(v for v in q['source_pins'] if Path(v['path']).name=='VAR_full_T_native_bank_contract_CPU_20261102_v1.py')
  s=importlib.util.spec_from_file_location('closed_complete_native_bank',checked(pin));c=importlib.util.module_from_spec(s);s.loader.exec_module(c);c.setup(protocol['path'])
  assert not c.process(525462,119859959)['exact_live'],'Whole bank controller is still live; no EEG access admitted.'
  self.root=Path(q['output_root']);assert not (self.root/'failure.json').exists()
  done=read(self.root/'completion.json');assert done['status']=='COMPLETE_FULL_T13230_NATIVE_K1000_ALL680_BANK_AND_WHOLE_CPU_AUDIT' and done['protocol']==protocol
  a=read(checked(done['CPU_audit']));assert a['status']=='PASS_ALL13230_ROWS_132300000_RETAINED_SCALE_CELLS_AND_COMPLETE_CLASS_COVERAGE' and a['protocol']==protocol and len(a['parts'])==207
  assert sum(v['rows'] for v in a['parts'])==13230 and sum(v['retained_scale_cells'] for v in a['parts'])==132300000
  self.parts=a['parts'];self.cache={};self.protocol=protocol
 def rows(self,positions):
  ids=np.asarray(positions,dtype=np.int64)
  if ids.ndim!=1 or len(ids)<1 or (ids<0).any() or (ids>=13230).any():raise ValueError('Complete T row positions required')
  out=np.empty((len(ids),10,1000),np.float64)
  for part in np.unique(ids//64):
   k=int(part);lo=k*64;hi=min(lo+64,13230)
   if k not in self.cache:
    done=read(checked(self.parts[k]['completion']));pin=next(v for v in done['files'] if Path(v['path']).name=='scale_log_likelihoods.npy')
    a=np.load(checked(pin),mmap_mode='r',allow_pickle=False);assert a.shape==(hi-lo,1000,10) and a.dtype==np.float64;self.cache[k]=a
   take=np.flatnonzero(ids//64==k);out[take]=self.cache[k][ids[take]-lo].transpose(0,2,1)
  assert np.isfinite(out).all() and (out<=0).all();return out
