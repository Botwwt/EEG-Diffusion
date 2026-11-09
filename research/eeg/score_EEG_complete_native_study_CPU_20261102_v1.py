"""Fixed ten-arm, 40-concept EEG study readout after full artifact closure.

Only the frozen metric_arrays/compare arithmetic is reused. Historical HIVE
load_panel, query-prefix selection, generation contracts and gates are not used.
"""
import argparse,csv,os,time
from pathlib import Path
import numpy as np
import EEG_complete_native_study_contract_CPU_20261102_v1 as C
from audit_EEG_complete_native_predictions_images_CPU_20261102_v1 import all_image_arms,finals,guard_factory

METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc')
REFERENCES={'global_full':('global_coarse','visual_global_full','image_prior_full'),
 'chain_full':('global_full','independent_full','visual_global_full','image_prior_full')}

def pairs():
 result={f'{a}_minus_{b}':(a,b) for a,refs in REFERENCES.items() for b in refs}
 for arm in C.CANDIDATES:
  for suffix in ('same','different'):result[f'{arm}_minus_{suffix}_EEG']=(arm,arm+'_'+suffix)
 return result

def decision(contrasts,stage,prior80=None):
 assert stage in (80,400) and set(contrasts)==set(pairs());result={}
 for arm,refs in REFERENCES.items():
  if stage==80:
   harms=[]
   for ref in refs:
    row=contrasts[f'{arm}_minus_{ref}']['DINO2AFC']
    if row['difference']<=-.02 and row['descriptive_concept_bootstrap95'][1]<0:harms.append(ref)
   result[arm]=dict(action='STOP_CLEAR_HARM' if harms else 'EXPAND400',clear_harm_references=harms,efficacy_established=False)
  else:
   assert prior80 is not None and prior80['stage']==80
   checks={}
   for ref in refs:
    row=contrasts[f'{arm}_minus_{ref}'];afc=row['DINO2AFC']
    checks[ref]=bool(afc['difference']>=.01 and afc['descriptive_concept_bootstrap95'][0]>0 and row['target_cosine']['difference']>0 and row['within_concept10_top1']['difference']>=0)
   diff=contrasts[f'{arm}_minus_different_EEG']['DINO2AFC'];same=contrasts[f'{arm}_minus_same_EEG']['DINO2AFC']
   pairing=bool(diff['difference']>0 and diff['descriptive_concept_bootstrap95'][0]>0)
   details=bool(same['difference']>0 and same['descriptive_concept_bootstrap95'][0]>0 and contrasts[f'{arm}_minus_same_EEG']['within_concept10_top1']['difference']>0)
   stopped=prior80['candidates'][arm]['action']=='STOP_CLEAR_HARM';passed=all(checks.values()) and pairing and not stopped
   result[arm]=dict(action='PROMISING_FOR_MULTISUBJECT_AND_COMPLETE_METRICS' if passed else 'STOP_INSTANCE',fixed_comparator_checks=checks,paired_EEG_over_different_concept_control=pairing,paired_stimulus_detail_supported=details,prior80_stop_preserved=stopped,independent_confirmation=False,SOTA=False)
 return dict(stage=stage,candidates=result,expand_any=any(v['action']=='EXPAND400' for v in result.values()) if stage==80 else False,selected_winner=None,no_hyperparameter_or_epoch_selection=True,training_seed_uncertainty_assessed=False)

def run(protocol,stage):
 q=C.setup(protocol,formal=True);guard=guard_factory(q,q['CPU_score_seconds']);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 finals(protocol);C.roots()
 import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as locked
 from EEG_complete_native_independent_scores_CPU_20261102_v1 import independent_arrays,independent_contrast
 import EEG_complete_native_independent_scores_CPU_20261102_v1 as independent_source
 for module in (locked,independent_source):
  pin=next(p for p in q['metric_sources'] if Path(p['path']).name==Path(module.__file__).name);assert C.ref(module.__file__)==pin
 assert tuple(locked.METRICS)==METRICS
 audit_path=C.OUT/f'artifacts_stage{stage}_CPU_audit.json';audit=C.read(audit_path)
 assert audit['status']=='PASS_ALL_EXPECTED_IMAGES_IDENTITIES_SEEDS_LATENT_CHAINS_AND_FILES' and audit['protocol']==C.ref(protocol)
 assert audit['arms']==all_image_arms() and audit['queries_per_arm']==stage and audit['images']==stage*10 and audit['all_samples_retained']
 for pin in audit['record_files'].values():C.checked(pin)
 for pin in audit['image_files']:C.checked(pin)
 joined=C.read(C.checked(C.read(C.OUT/'inputs/completion.json')['identities']));image=C.read(C.checked(q['legacy_image_spec']));panel=C.read(C.checked(image['panel']))
 selected=joined['stage80'] if stage==80 else list(range(400));queries=[panel['queries'][i] for i in selected];gallery=panel['target_gallery']
 assert len(selected)==stage and len(gallery)==400 and len({r['concept_id'] for r in queries})==40
 assert all(sum(r['concept_id']==k for r in queries)==(2 if stage==80 else 10) for k in {r['concept_id'] for r in queries})
 # Every fixed arm's complete files/identity closure precedes target-score reads.
 features={}
 for arm in all_image_arms():
  guard();by_index={}
  for pin in audit['features'][arm]:
   data=np.load(C.checked(pin),allow_pickle=False);root=Path(pin['path']).parent.parent;chunk=C.read(root/'completion.json')
   assert data.shape==(20,768) and len(chunk['query_indices'])==20
   for i,row in zip(chunk['query_indices'],data):assert i not in by_index;by_index[i]=row
  assert set(by_index)==set(selected);features[arm]=np.stack([by_index[i] for i in selected]).astype(np.float64)
 cache=C.read(C.checked(q['target_cache']));assert cache['status']=='EXISTING_LOCKED_TARGET_FEATURES_REUSED_WITH_COMPLETE_AR_EEG_IDENTITY_JOIN'
 assert cache['panel']==image['panel'] and cache['encoder']['weight_sha256']==image['DINO_weight']['sha256']
 assert all(cache['encoder']['image_evaluation'][k]==image['image_evaluation'][k] for k in ('bytes','sha256'))
 assert cache['features']==q['target_features'];targets=np.load(C.checked(cache['features']),allow_pickle=False).astype(np.float64)
 assert targets.shape==(400,768) and np.isfinite(targets).all() and np.all(np.linalg.norm(targets,axis=-1)>0)
 selfpair=locked.metric_arrays(targets,targets,gallery,gallery);assert all(np.all(selfpair[m]==1) for m in ('DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc'))
 arrays={a:locked.metric_arrays(x,targets,queries,gallery) for a,x in features.items()};independent={a:independent_arrays(x,targets,queries,gallery) for a,x in features.items()}
 error=max(float(np.abs(arrays[a][m]-independent[a][m]).max()) for a in arrays for m in METRICS);assert error<=2e-6
 contrasts={name:locked.compare(arrays[a],arrays[b],queries) for name,(a,b) in pairs().items()}
 separate={name:independent_contrast(independent[a],independent[b],queries) for name,(a,b) in pairs().items()}
 cierror=max(abs(x-y) for name in contrasts for m in METRICS for x,y in zip(contrasts[name][m]['descriptive_concept_bootstrap95'],separate[name][m]['descriptive_concept_bootstrap95']));assert cierror<=2e-6
 prior80=C.read(C.OUT/'screens/stage80/summary.json') if stage==400 else None
 if prior80:assert prior80['protocol']==C.ref(protocol) and audit['preserved_stage80']==prior80['artifacts']
 outcome=decision(contrasts,stage,prior80);assert outcome==decision(separate,stage,prior80)
 out=C.OUT/'screens'/f'stage{stage}';out.mkdir(parents=True);assert not (out/'summary.json').exists()
 np.savez_compressed(out/'all_metrics.npz',**{a+'__'+m:v for a,row in arrays.items() for m,v in row.items()})
 with (out/'per_query.csv').open('x',encoding='utf8',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=['arm','query_index','query_id','stimulus_id','concept_id',*METRICS]);writer.writeheader()
  for arm,row in arrays.items():
   for j,(i,r) in enumerate(zip(selected,queries)):writer.writerow(dict(arm=arm,query_index=i,query_id=r['query_id'],stimulus_id=r['stimulus_id'],concept_id=r['concept_id'],**{m:float(v[j]) for m,v in row.items()}))
 guard();C.write(out/'summary.json',dict(status='COMPLETE_FIXED_ALL10_EEG_FULL_GENERATOR_IMAGE_SCREEN',protocol=C.ref(protocol),artifacts=C.ref(audit_path),target_cache=q['target_cache'],target_features=q['target_features'],means={a:{m:float(v.mean()) for m,v in row.items()} for a,row in arrays.items()},contrasts=contrasts,independent_metric_max_error=error,independent_interval_max_error=cierror,all_query_metric_cells=stage*10*5,all_metrics=C.ref(out/'all_metrics.npz'),per_query=C.ref(out/'per_query.csv'),all_fixed_models_and_images_complete_before_effect_reads=True,already_exposed_single_subject_development=True,complete_semantic_metrics_or_official_R80_completed=False,**outcome))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--stage',type=int,choices=[80,400],required=True);v=p.parse_args();run(v.protocol,v.stage)
