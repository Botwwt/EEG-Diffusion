"""Cold whole-T/heldout EEG input join after complete bank. No CUDA."""
import argparse,datetime,hashlib,json,os,resource,sys,time,traceback
from pathlib import Path
import EEG_complete_native_study_contract_CPU_20261102_v1 as C
def run(protocol):
 tick=time.monotonic();base_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 q=C.setup(protocol);C.resources(q);base,a,image,stats=C.source_inputs(q,binary=True);C.roots();out=C.OUT/'inputs';out.mkdir()
 try:
  import numpy as np,torch
  from eeg_labram_interface_20261028_v1 import CacheNativeEEGTransform
  from scipy.signal import resample_poly
  assert not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
  ids=C.read(a['data']['EEG_identities']['path']);tokens=C.read(a['data']['token_identities']['path']);T=ids['T'];assert len(T)==len(tokens)==13230
  assert len({r['stimulus_id'] for r in T})==13230 and len({r['concept_id'] for r in T})==1323
  assert stats['training_array_rows']==[int(r['array_row']) for r in T]
  for i,(x,y) in enumerate(zip(T,tokens)):
   assert y['query_index']==i and all(x[k]==y[k] for k in ('stimulus_id','array_row','concept_id'))
  panel=C.read(image['panel']['path']);P=panel['queries'];gallery=panel['target_gallery'];physical=C.read(image['physical_queries']['path'])
  assert len(P)==len(gallery)==len(physical['Dcheck'])==400
  groups={}
  for i,(x,y,z) in enumerate(zip(P,gallery,physical['Dcheck'])):
   assert all(x[k]==y[k]==z[k] for k in ('stimulus_id','array_row','concept_id'));groups.setdefault(x['concept_id'],[]).append(i)
  assert len(groups)==40 and {len(v) for v in groups.values()}=={10}
  assert not ({r['concept_id'] for r in T}&set(groups))
  stage80=sorted(i for v in groups.values() for i in v[:2]);assert len(stage80)==80 and len({P[i]['concept_id'] for i in stage80})==40
  transform=CacheNativeEEGTransform(tuple(stats['channel_names']),np.array(stats['center']),np.array(stats['scale']));files={};errors={}
  for role,pin,count in [('T',base['T_mean63'],13230),('Dcheck',image['Dcheck_mean63'],400)]:
   raw=np.load(pin['path'],mmap_mode='r',allow_pickle=False);assert raw.shape==(count,63,250) and raw.dtype==np.float32
   p=out/(role+'_EEG_normalized.npy');dest=np.lib.format.open_memmap(p,mode='w+',dtype=np.float32,shape=(count,63,1,200));maximum=0.
   for lo in range(0,count,64):
    hi=min(lo+64,count);x=np.array(raw[lo:hi]);value=transform(x,channel_names=tuple(stats['channel_names']))
    # Independent explicit standardization and public polyphase operator,
    # every input row, not just a selected query.
    standard=(x.astype(np.float64)-np.asarray(stats['center'])[None,:,None])/np.asarray(stats['scale'])[None,:,None]
    independent=resample_poly(standard,4,5,axis=2,padtype='constant').astype(np.float32)[:,:,None,:]
    err=float(np.abs(value-independent).max());maximum=max(maximum,err);assert np.array_equal(value,independent);dest[lo:hi]=value
    assert time.monotonic()-tick<q['CPU_input_seconds'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-base_RSS<q['maximum_extra_CPU_bytes']
   dest.flush();del dest;files[role]=C.ref(p);errors[role]=maximum
  warm=torch.load(base['full_EEG']['model']['path'],map_location='cpu',weights_only=True)
  assert warm['role']=='full' and warm['subject']=='sub-01' and warm['epoch']==50 and warm['updates']==5200 and warm['query_exposures']==661500
  assert all(torch.isfinite(v).all() for v in warm['model'].values() if isinstance(v,torch.Tensor) and v.is_floating_point())
  order=dict(stage80=stage80,stage400=list(range(400)),T_rows=[dict(stimulus_id=r['stimulus_id'],array_row=r['array_row'],concept_id=r['concept_id']) for r in T],Dcheck_rows=P)
  C.write(out/'joined_identity_and_stages.json',order)
  C.write(out/'completion.json',dict(status='PASS_COLD_COMPLETE_T13230_DCHECK400_REAL_EEG_INPUT_JOIN_TRANSFORM',protocol=C.ref(protocol),bank=C.bank_state(),files=files,identities=C.ref(out/'joined_identity_and_stages.json'),source_T=base['T_mean63'],source_Dcheck=image['Dcheck_mean63'],normalization=base['full_EEG']['normalization'],initial_EEG=base['full_EEG']['model'],warm_source_updates=5200,warm_source_query_exposures=661500,all_T_and_Dcheck_transform_rows_checked=True,max_transform_error=errors,independent_original_physical_repeat_average_replayed=False,permission='Pinned published physical-R4 caches and original T-only construction evidence reused',stage80_query_count=80,stage80_concepts=40,query_images_read=0,new_updates=0,CUDA_initialized=False,seconds=time.monotonic()-tick))
 except BaseException as e:C.write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),no_retry=True));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
