"""Independent terminal checkpoint, full condition and coverage audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,time,hashlib,json
from pathlib import Path
import numpy as np
import atm_repeat_count_training_contract_20261105_v1 as C
import atm_repeat_count_training_core_20261105_v1 as F
start=time.monotonic();c=C.setup(sys.argv[1]);q,K,out,pq=c['q'],c['K'],c['out'],c['physical'];assert K.ref(__file__)==q['sources']['CPU'];C.resources(c,starting=True)
sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
torch.set_num_threads(2)
done=K.read(out/'conditions_completion.json');assert done['status']=='COMPLETE_EEG_ONLY_ALL2400_CONDITIONS'
values=np.load(K.checked(done['conditions']));assert values.shape==(600,4,1024)
raw=torch.load(pq['EEG']['path'],map_location='cpu',weights_only=True,mmap=True)
test=torch.load(K.checked(c['R80']['EEG_R80']),map_location='cpu',weights_only=True,mmap=True);assert test.shape==(200,63,250)
ids=[r['array_row'] for r in c['queries']];errors={};checkpoints={}
base=torch.load(K.checked(q['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
old=np.load(K.checked(q['positive_R4_conditions']));assert old.shape==(400,5,1024)
assert np.max(abs(values[:400,0]-old[:,2]))<5e-5
for ai,arm in enumerate(('frozen_positive_NCE',)+F.ARMS):
 state=base if ai==0 else torch.load(K.checked(K.read(out/'fits'/arm/'completion10.json')['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
 if ai:
  assert state['epoch']==10 and state['updates']==830 and state['exposures']==132300 and state['view_exposures']==1984500
  events=[json.loads(s) for s in (out/'fits'/arm/'epochs.jsonl').read_text().splitlines()];assert len(events)==10
  for epoch,e in enumerate(events):
   order=np.random.default_rng(1739+epoch).permutation(1323);assert hashlib.sha256(order.tobytes()).hexdigest()==e['concept_order_sha256']
   assert e['coverage_min']==e['coverage_max']==1 and e['updates']==83*(epoch+1) and e['exposures']==13230*(epoch+1)
   assert not e['R80_data_fitted'] and not e['Dfit_or_Dcheck_used']
   assert abs(float(F.weights(arm)@np.array(e['online_mean_losses_by_count']))-e['online_risk'])<1e-10
  for entry in state['optimizer']['state'].values():assert int(entry['step'])==1660
  # Unused native modules are truly unchanged, not silently weight-decayed.
  for name in base['model']:
   if name.startswith('subject_wise_linear'):assert torch.equal(state['model'][name],base['model'][name])
  checkpoints[arm]=K.read(out/'fits'/arm/'completion10.json')['checkpoint']
 net=ATMS().eval();net.load_state_dict(state['model'],strict=True);errs=[]
 with torch.inference_mode():
  for lo in range(0,600,100):
   assert time.monotonic()-start<850
   x=raw[ids[lo:lo+100]] if lo<400 else test[lo-400:lo-300]
   pred=net(x,torch.ones(len(x),dtype=torch.long)).numpy();errs.append(float(abs(pred-values[lo:lo+100,ai]).max()))
 assert max(errs)<5e-5;errors[arm]=max(errs);del net
K.write(out/'independent_CPU_audit.json',dict(status='PASS_ALL2400_CPU_CONDITIONS_AND_ALL30_EPOCHS',CPU_forwards=2400,
 maximum_errors=errors,all_30_orders_and_risk_means_checked=True,all_final_Adam_steps1660=True,
 frozen_positive_R4_reference_matched=True,checkpoints=checkpoints,seconds=time.monotonic()-start,
 no_full_training_gradient_replay=True,no_image_efficacy_claim=True))
C.resources(c)
