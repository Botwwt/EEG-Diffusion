"""Independent full condition replay, all training logs and saved Adam states."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,hashlib,json,sys,time
from pathlib import Path
import atm_deployment_training_contract_20261105_v179 as C
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);args=ap.parse_args();started=time.monotonic()
c=C.setup(args.policy);q,K,pq,out=c['q'],c['K'],c['physical'],c['out'];C.resources(c,starting=True)
sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import numpy as np
import torch
from models.atms import ATMS
torch.set_num_threads(2)
T=np.asarray(q['T_array_rows'],np.int64);assert len(np.unique(T))==13230
assert hashlib.sha256(T.tobytes()).hexdigest()==q['T_rows_sha256']
base=torch.load(K.checked(q['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
raw=torch.load(K.checked(pq['EEG']),map_location='cpu',weights_only=True,mmap=True)
test=torch.load(K.checked(c['old']['train']['R80']['EEG_R80']),map_location='cpu',weights_only=True,mmap=True)
d=K.read(out/'conditions_completion.json');assert d['status']=='COMPLETE_EEG_ONLY_ALL2400_CONDITIONS'
values=np.load(K.checked(d['conditions']));rows=K.read(K.checked(d['queries']));ids=[r['array_row'] for r in rows['R4']]
assert values.shape==(600,4,1024) and not set(ids)&set(T)
logs=[];maxerrors={}
for arm in C.ARMS:
 directory=out/'fits'/arm;ev=[json.loads(s) for s in (directory/'epochs.jsonl').read_text().splitlines()];assert len(ev)==10
 for epoch,e in enumerate(ev):
  order=np.random.default_rng(1739+epoch).permutation(1323)
  assert e['arm']==arm and e['epoch']==epoch+1 and e['updates']==83*(epoch+1) and e['exposures']==13230*(epoch+1)
  assert e['concept_order_sha256']==hashlib.sha256(order.tobytes()).hexdigest() and e['coverage_min']==e['coverage_max']==1
  assert e['BN_updates']==83 and e['EEG_forward_calls']==e['derivative_calls']==166
  assert e['lambda_P']==1. and not e['D_or_R80_fit'] and all(np.isfinite(v) for v in e['online_mean'].values())
 for epoch in (5,10):
  pin=K.read(directory/f'completion{epoch:02d}.json')['checkpoint'];ck=torch.load(K.checked(pin),map_location='cpu',weights_only=True,mmap=True)
  assert ck['updates']==epoch*83 and ck['exposures']==epoch*13230 and ck['arm']==arm and ck['epoch']==epoch
  assert all(int(z['step'])==830+epoch*83 for z in ck['optimizer']['state'].values())
  for name,v in ck['model'].items():
   assert torch.isfinite(v).all()
   if name.endswith('num_batches_tracked'):assert int(v)==int(base['model'][name])+epoch*83
  for z in ck['optimizer']['state'].values():assert all(torch.isfinite(v).all() for v in z.values() if torch.is_tensor(v))
  assert ck['parent']==q['checkpoint'] and ck['CPU_rng'].dtype==ck['CUDA_rng'].dtype==torch.uint8
 logs+=ev
for j,arm in enumerate(C.ALL_ARMS):
 pin=q['reference_checkpoints'][arm] if arm in q['reference_checkpoints'] else K.read(out/'fits'/arm/'completion10.json')['checkpoint']
 ck=torch.load(K.checked(pin),map_location='cpu',weights_only=True,mmap=True);net=ATMS();net.load_state_dict(ck['model'],strict=True);net.eval();err=0.
 with torch.inference_mode():
  for lo in range(0,600,100):
   assert time.monotonic()-started<850
   x=raw[ids[lo:lo+100]] if lo<400 else test[lo-400:lo-300]
   pred=net(x,torch.ones(len(x),dtype=torch.long)).numpy();err=max(err,float(np.max(abs(pred-values[lo:lo+100,j]))))
 assert err<5e-5,(arm,err);maxerrors[arm]=err;del net
baseline_error=float(np.max(abs(values[:,:2]-c['old']['values'][:,:2])));assert baseline_error<5e-5
assert not torch.cuda.is_initialized();K.checked(q['checkpoint'])
K.write(out/'independent_CPU_audit.json',dict(status='PASS_ALL2400_CPU_CONDITIONS_ALL20_EPOCHS_FOUR_SAVED_STATES',
 seconds=time.monotonic()-started,condition_max_errors=maxerrors,baseline_original_cache_error=baseline_error,
 coverage_and_order_epochs=20,saved_model_optimizer_RNG_states_checked=4,full_gradient_replay=False,
 all_original_checkpoints_unchanged=True,new_images=0,OLD_only=True,NEW_contacted=False))
