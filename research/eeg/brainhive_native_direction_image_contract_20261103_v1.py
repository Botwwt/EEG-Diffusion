"""Postfit evaluation under the original full native EEG study's unchanged 10GiB cap."""
import json,sys
from pathlib import Path
ARMS=('reencoded_ordinary','positive_only','directional','sameconcept_wrong_EEG')
OLD_ARMS=('parent','ordinary','risk')
CONTRASTS={'directional_minus_positive_only':('directional','positive_only'),
 'directional_minus_wrong_EEG':('directional','sameconcept_wrong_EEG'),
 'directional_minus_old_ordinary':('directional','ordinary'),
 'positive_only_minus_old_ordinary':('positive_only','ordinary'),
 'directional_minus_reencoded_ordinary':('directional','reencoded_ordinary'),
 'reencoded_minus_old_ordinary':('reencoded_ordinary','ordinary')}
STATUS='REGISTERED_NATIVE_EEG_FIXED_FINAL_DCHECK400_FOUR_ARM_POSTFIT'
def setup(path):
 s=json.loads(Path(path).read_bytes());sys.path.insert(0,str(Path(s['training_protocol']['path']).parent))
 import brainhive_native_direction_full_contract_20261103_v1 as C
 C.checked(s['training_protocol']);train,e,old,base,G,H=C.setup(s['training_protocol']['path'])
 assert s['status']==STATUS and s['host']=='old' and s['seed']==1729 and s['GPU_UUID']==train['GPU_UUID']
 assert tuple(s['arms'])==ARMS and s['fixed_contrasts']=={k:list(v) for k,v in CONTRASTS.items()}
 assert s['subject']=='sub-01' and s['stages']==[80,400] and s['gallery_targets']==400 and s['queries_per_segment']==80
 assert s['image_phase_cap']==2<<30 and s['maximum_GPU_bytes']==20<<30 and s['maximum_extra_CPU_bytes']==10<<30
 assert s['profile_seconds']==420 and s['generation_segment_seconds']==900 and s['CPU_stage_seconds']==360 and s['maximum_execution_seconds']==6*3600
 assert s['ordinary_immutable'] and not s['NEW_contacted'] and s['all30_final_models_and_training_owner_exit_required']
 root=Path(train['output_root']);assert s['runtime_revision']==2 and Path(s['code_root'])==root/'postfit_code_20261103_v2' and Path(s['output_root'])==root/'postfit_20261103_v2'
 assert s['preserved_postfit_roots']==[str(root/'postfit_code_20261103_v1'),str(root/'postfit_20261103_v1')]
 for pin in s['source_pins']+s['reused_source_pins']:C.checked(pin)
 return s,train,e,old,base,G,H,C
def dependency(s,train,C):
 owner=s['training_owner'];proc=Path('/proc',str(owner['PID']))
 if proc.exists():
  try:stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
  except FileNotFoundError:stat=None
  if stat is not None and int(stat[19])==owner['start_ticks'] and stat[0] not in ('Z','X'):
   return dict(status='WAIT_SAME_ORIGINAL_TRAINING_OWNER',owner_state=stat[0],new_GPU_jobs=0)
 qroot=Path(train['output_root'])/'queue';done=qroot/'completion.json'
 if not done.exists():return dict(status='STOP_SOURCE_TRAINING_TERMINAL_WITHOUT_FULL_SUCCESS',failure=C.ref(qroot/'failure.json') if (qroot/'failure.json').exists() else None)
 document=C.read(done);assert document['status']=='COMPLETE_ALL30_NATIVE_EEG_FIXED_EPOCH3_T_FITS_READY_FOR_FROZEN_IMAGE_EVALUATION'
 assert document['protocol']==s['training_protocol'] and document['total_updates']==74430 and document['primary_exposures']==1190700 and len(document['models'])==30
 models={}
 for si,subject in enumerate(train['subjects']):
  for ai,arm in enumerate(train['arms']):
   pin=document['models'][3*si+ai];row=C.read(C.checked(pin));assert row['subject']==subject and row['arm']==arm and row['status']=='COMPLETE_FIXED_EPOCH3_NATIVE_EEG_MODEL'
   assert row['updates']==2481 and row['primary_exposures']==39690 and row['epoch']==3 and row['coverage_count']==39690 and row['ordinary_frozen'] and row['host_frozen'] and row['output_LN_affine_frozen']
   C.checked(row['final_model']);models[subject+'/'+arm]=row['final_model']
 return dict(status='READY_ALL30_FIXED_FINALS_AND_ACTUAL_OWNER_EXIT',source_completion=C.ref(done),models=models)
def assets(s,base,C,load_features=False):
 d09=C.read(C.checked(s['D09_completion']));inputs=C.read(C.checked(d09['inputs']));panel=C.read(C.checked(inputs['panel']))['Dcheck'];metadata=C.read(C.checked(inputs['target_metadata']))['Dcheck']
 assert len(panel)==len(metadata)==400 and len({v['query_id'] for v in panel})==400 and all(v['subject_id']=='sub-01' and v['partition']=='Dcheck' for v in panel)
 split=C.read(C.checked(base['input_pins']['split']));assert set(v['array_row'] for v in panel).issubset(set(split['image_rows']['D'])) and set(v['concept_id'] for v in panel).issubset(set(split['d_check_concepts'])) and not set(v['array_row'] for v in panel).intersection(split['image_rows']['T'])
 groups={}
 for i,row in enumerate(panel):
  for k,v in row.items():assert metadata[i][k]==v
  groups.setdefault(row['concept_id'],[]).append(i)
 assert len(groups)==40 and all(len(v)==10 for v in groups.values()) and all(len([i for i in v if i<80]) in (0,10) for v in groups.values())
 rows=[json.loads(line) for line in C.checked(d09['check_manifest']).read_text().splitlines()];assert len(rows)==1200
 for i,row in enumerate(panel):
  for j,arm in enumerate(OLD_ARMS):
   old=rows[3*i+j];assert old['arm']==arm and old['UNet_NFE']==4 and old['normal_inference_target_access'] is False
   for k,v in row.items():assert old[k]==v
   for key in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations'):assert old[key]==rows[3*i][key]
 result=dict(D09=d09,inputs=inputs,panel=panel,metadata=metadata,old_manifest=rows)
 if load_features:
  import numpy as np
  result['target_features']=np.load(C.checked(d09['check_target_features']),allow_pickle=False);result['old_features']=np.load(C.checked(d09['check_generated_features']),allow_pickle=False)
  assert result['target_features'].shape==(400,768) and result['old_features'].shape==(400,3,768)
 return result
def resources(s,train,e,C,starting=False,admit_remaining_phase=False):
 status=C.resources(train,e,starting);used=sum(C.allocated(p) for p in [s['code_root'],s['output_root']]+s['preserved_postfit_roots']);assert used<=s['image_phase_cap']
 if admit_remaining_phase:assert status['own_remaining_bytes']>=s['image_phase_cap']-used
 return dict(full_lifecycle=status,own_phase_used_bytes=used,phase_cap_within_same_full10GiB=s['image_phase_cap'],new_external_reservation_bytes=0)
