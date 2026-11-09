"""All real T80 raw R4 inputs and ten actual full parent encoders, no CUDA."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from brainhive_native_direction_contract_20261103_v1 import setup,resources
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);cli=parser.parse_args();started=time.monotonic()
 spec,old,base,G,H=setup(cli.protocol);before=resources(spec,old,base,G,H,True)
 import torch
 from brainhive_native_models_v1 import classes
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();scope=classes()
 prior=G.read(G.checked(old['prior_prepared']));prior_arrays=dict(np.load(G.checked(prior['arrays']),allow_pickle=False))
 split=G.read(G.checked(base['input_pins']['split']));allowed=set(split['image_rows']['T']);rows=old['queries'];assert len(rows)==80
 manifest=[row for row in csv.DictReader(G.checked(spec['image_manifest']).open()) if row['dataset_split']=='train']
 concepts={}
 for row in manifest:
  if int(row['array_row']) in allowed:concepts.setdefault(row['concept_id'],[]).append(int(row['array_row']))
 for key in concepts:concepts[key].sort();assert len(concepts[key])==10
 bindings=G.read(G.checked(spec['actual_bindings']));eeg=np.empty((80,4,4250),np.float32);CPU_conditions=np.empty((80,4,1024),np.float32)
 gammas=np.empty((10,1024),np.float32);betas=np.empty_like(gammas);source_rows=np.empty((80,4),np.int64);reports=[];coverage=np.zeros(80,np.int64)
 for si,subject in enumerate(sorted(bindings['subjects'])):
  item=next(v for v in spec['R4_payloads'] if v['subject']==subject);modelpin=next(v for v in spec['full_models'] if v['subject']==subject)
  raw=torch.load(G.checked(item['payload']),weights_only=True,map_location='cpu',mmap=True);assert raw.shape==(16540,63,250)
  saved=torch.load(G.checked(modelpin['model']),weights_only=True,map_location='cpu');model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval();model.load_state_dict(saved['model'],strict=True)
  reg=G.read(G.checked(modelpin['registration']));assert set(reg['training_array_rows'])==allowed and not reg['excluded_oof_rows']
  for index,row in enumerate(rows):
   if row['subject']!=subject:continue
   cr=concepts[row['concept_id']];i=cr.index(row['array_row']);j=cr.index(row['donor_array_row']);assert j==(i+1)%10
   ids=[cr[i],cr[j],cr[(i+2)%10],cr[(j+2)%10]];assert len(set(ids))==4
   x=raw[ids][:,bindings['selected_indices'],:].reshape(4,4250).contiguous();assert torch.isfinite(x).all();eeg[index]=x.numpy();source_rows[index]=ids;coverage[index]+=1
   with torch.no_grad():CPU_conditions[index]=model(x).numpy()
  gammas[si]=saved['model']['block.layers.0.norm.weight'].numpy();betas[si]=saved['model']['block.layers.0.norm.bias'].numpy()
  reports.append(dict(subject=subject,model=modelpin['model'],registration=modelpin['registration'],raw_R4=item['payload'],training_rows=13230,
   native_CPU_eval_mode=True,no_OOF_claim=True));del model,saved,raw
 assert np.all(coverage==1) and np.isfinite(eeg).all() and np.isfinite(CPU_conditions).all()
 out=Path(spec['output_root'])/'inputs';assert not out.exists();out.mkdir()
 np.savez_compressed(out/'T80_raw_R4_and_full_parent_CPU.npz',EEG=eeg,conditions_CPU=CPU_conditions,gammas=gammas,betas=betas,source_rows=source_rows,F_targets=prior_arrays['target'])
 assert not torch.cuda.is_initialized() and time.monotonic()-started<180
 done=dict(status='COMPLETE_REAL_T80_RAW_R4_TEN_FULL_PARENT_CPU_INPUTS_NO_CUDA',protocol=G.ref(cli.protocol),source=G.ref(__file__),arrays=G.ref(out/'T80_raw_R4_and_full_parent_CPU.npz'),
  rows=rows,models=reports,ordinary=base['ordinary'],native_parent_T_fit_not_OOF=True,source_labels_only_for_training_controls=True,
  seconds=time.monotonic()-started,new_GPU_jobs=0,new_models=0,new_images=0,NEW_contacted=False,resources_before=before,resources_after=resources(spec,old,base,G,H))
 G.write(out/'completion.json',done);print(json.dumps(done))
if __name__=='__main__':main()
