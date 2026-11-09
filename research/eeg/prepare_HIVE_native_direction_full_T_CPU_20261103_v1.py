"""Verify the exact full T join and all fixed pair orders before any formal update."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import brainhive_native_direction_full_contract_20261103_v1 as C
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();started=time.monotonic();s,e,old,base,G,H=C.setup(a.protocol);before=C.resources(s,e,True)
 import torch
 from brainhive_native_models_v1 import classes
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();scope=classes();bindings=C.read(C.checked(e['actual_bindings']));split=C.read(C.checked(base['input_pins']['split']))
 rows=np.array(split['image_rows']['T'],np.int64);allowed=set(rows);assert len(rows)==13230 and len(np.unique(rows))==13230
 manifest=[r for r in csv.DictReader(C.checked(e['image_manifest']).open()) if r['dataset_split']=='train'];concepts={}
 for r in manifest:
  if int(r['array_row']) in allowed:concepts.setdefault(r['concept_id'],[]).append(int(r['array_row']))
 assert len(concepts)==1323 and all(len(r)==10 for r in concepts.values())
 plans=np.stack([[C.pair_plan(concepts,subject,epoch) for epoch in range(3)] for subject in s['subjects']]);reports=[]
 for si,subject in enumerate(s['subjects']):
  mp=next(v for v in e['full_models'] if v['subject']==subject);rp=next(v for v in e['R4_payloads'] if v['subject']==subject)
  reg=C.read(C.checked(mp['registration']));assert set(reg['training_array_rows'])==set(rows) and not reg['excluded_oof_rows']
  raw=torch.load(C.checked(rp['payload']),weights_only=True,map_location='cpu',mmap=True);assert raw.shape==(16540,63,250)
  model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval();model.load_state_dict(torch.load(C.checked(mp['model']),weights_only=True,map_location='cpu')['model'],strict=True)
  from brainhive_native_direction_contract_20261103_v1 import freeze_affine
  freeze_affine(model)
  for start in range(0,len(rows),256):assert torch.isfinite(raw[rows[start:start+256]][:,bindings['selected_indices']]).all()
  for epoch in range(3):
   plan=plans[si,epoch];assert sorted(plan[:,:2].ravel())==sorted(rows) and set(plan.ravel())==set(rows)
  reports.append(dict(subject=subject,model=mp['model'],raw_R4=rp['payload'],registration=mp['registration'],epochs=3,pair_batches_per_epoch=827,updates=2481,primary_exposures=39690))
  del raw,model
 targets=torch.load(C.checked(base['input_pins']['T_F_targets']),weights_only=True,map_location='cpu');assert targets.shape==(16540,1024) and torch.isfinite(targets[rows]).all()
 latents=np.load(C.checked(old['T_latent']),mmap_mode='r',allow_pickle=False);assert latents.shape==(13230,4,64,64) and np.isfinite(latents).all()
 # Serialize/restore the two actual training RNG families and compare their next draws.
 nr=torch.Generator(device='cpu').manual_seed(C.stable_seed('sub-01',0,'noise'));ar=np.random.default_rng(C.stable_seed('sub-01',0,'axial'))
 import copy
 ns=nr.get_state().clone();ass=copy.deepcopy(ar.bit_generator.state);probe=C.rng_probe(nr,ar,torch);nr.set_state(ns);ar.bit_generator.state=ass;assert C.rng_probe(nr,ar,torch)==probe
 root=Path(s['output_root'])/'inputs';root.mkdir();np.savez_compressed(root/'fixed_full_T_pair_plans.npz',training_rows=rows,pairs=plans)
 done=dict(status='COMPLETE_T10_THREE_PASS_ALL_CONCEPT_PAIR_ORDERS_AND_NATIVE_INPUTS_CPU',protocol=C.ref(a.protocol),source=C.ref(__file__),plans=C.ref(root/'fixed_full_T_pair_plans.npz'),subjects=reports,
  tensor_shape=[10,3,6615,4],primary_T_coverage_exactly_once_per_epoch=True,full_parent_T_fit_not_OOF=True,ordinary_frozen=True,all30_models_same_training_permissions=True,
  RNG_serialized_next_draw_probe=probe,seconds=time.monotonic()-started,resources_before=before,resources_after=C.resources(s,e),new_updates=0,new_images=0,NEW_contacted=False)
 assert not torch.cuda.is_initialized() and time.monotonic()-started<240;G.write(root/'completion.json',done);print(json.dumps(done),flush=True)
if __name__=='__main__':main()
