"""Frozen first80 plus OLD-only missing320: all-five whole-panel readout."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import csv,datetime,hashlib,json,time
from pathlib import Path
import numpy as np
import score_EEG_native_categorical80_CPU_20261029_v1 as S80
import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as math
from AR_anchored_images_score_CPU_20261029_v1 import independent_arrays,independent_contrast

L=Path(__file__).resolve().parent
D=L/'audit/EEG_native_categorical_20261029_v1'
R=D/'received_stage400_OLD_v2'
ARMS=['coupled','same_donor','different_donor','visual','uniform']


def run():
 started=time.monotonic();receipt=S80.read(R/'receiving_completion.json')
 assert receipt['status']=='ALL5_CATEGORICAL_OLD_ONLY_NEW320_METADATA_EXACT_RECEIVED'
 for item in receipt['files'].values():
  p=Path(item['local']);assert p.stat().st_size==item['remote']['bytes'] and S80.sha(p)==item['remote']['sha256']
 def local(name,expected=None):
  item=receipt['files'][name];p=Path(item['local'])
  if expected is not None:assert all(expected[k]==item['remote'][k] for k in ('bytes','sha256'))
  return p
 queue=S80.read(local('queue_completion.json'));audit=S80.read(local('artifact_CPU_audit.json',queue['artifact_audit']));spec=S80.read(local('spec.json',queue['spec']))
 assert queue['status']=='COMPLETE_CATEGORICAL_OLD_ONLY_FIVE_ARM_STAGE400_CONTINUATION' and queue['new_research_images']==1600 and queue['original80_images_reused']==400
 assert audit['status']=='PASS_OLD_ONLY_ALL5_NEW320_PIXELS_IDENTITIES_FEATURES' and audit['images']==1600 and audit['unique_inodes']==1600 and not audit['effect_scores_read']
 assert spec['execution_host']=='old' and spec['arms']==ARMS and not spec['NEW_GPU_used']
 prior=S80.read(D/'stage80_results_v1/summary.json');assert prior['status']=='COMPLETE_ALL5_CATEGORICAL_FIXED80_IMAGE_SCREEN' and prior['decision']=='EXPAND400'
 assert S80.sha(D/'stage80_results_v1/summary.json')==spec['stage80_decision']['sha256']
 first_receipt=S80.read(S80.RECEIVED/'receiving_completion.json')
 for item in first_receipt['files'].values():assert S80.sha(item['local'])==item['remote']['sha256']
 panel=S80.read(S80.RECEIVED/'old/panel400.json');assert S80.sha(S80.RECEIVED/'old/panel400.json')==S80.PANEL_SHA
 queries=panel['queries'];gallery=panel['target_gallery'];groups={}
 for i,q in enumerate(queries):groups.setdefault(q['concept_id'],[]).append(i)
 keys=sorted(groups);assert len(keys)==40 and {len(g) for g in groups.values()}=={10}
 same={i:groups[k][(j+1)%10] for k in keys for j,i in enumerate(groups[k])};different={i:groups[keys[(n+1)%40]][j] for n,k in enumerate(keys) for j,i in enumerate(groups[k])}
 first=np.load(D/'stage80_results_v1/all6_features.npz',allow_pickle=False)
 for arm in ARMS:
  host='old' if arm in ('coupled','same_donor','different_donor') else 'new'
  original=np.concatenate([np.load(S80.RECEIVED/host/f'{arm}_features_{lo:04d}.npy',allow_pickle=False) for lo in range(0,80,20)])
  assert np.array_equal(first[arm],original),'Original first80 feature data changed'
 features={a:[first[a]] for a in ARMS};identities={a:[] for a in ARMS};seen=set();new_counts={a:0 for a in ARMS}
 logits={host:np.load(S80.RECEIVED/host/'EEG_logits400.npy',allow_pickle=False) for host in ('old','new')}
 models={host:S80.read(S80.RECEIVED/host/'fit_completion.json')['final_model']['sha256'] for host in ('old','new')}
 for lo in range(80,400,20):
  done=S80.read(local(f'chunk_{lo:04d}_completion.json'))
  assert done['status']=='COMPLETE_CATEGORICAL_OLD_ONLY_FIVE_ARM_IMAGE_CHUNK_v2' and done['range']==[lo,lo+20] and done['spec']==queue['spec']
  assert done['execution_host']=='old' and done['arms']==ARMS and done['native_VAR_VQ_unchanged'] and done['normal_input_EEG_only'] and done['effect_scores_read'] is False
  assert done['DINO_weight']['sha256']==S80.DINO_SHA and done['image_evaluation']==spec['image_evaluation']
  if lo==80:
   assert done['original80_class_decisions_replayed']==400 and done['original80_pixel_replays']==5
   assert all(v<2e-6 for v in done['original80_feature_max_errors'].values())
  for arm in ARMS:
   x=np.load(local(f'{arm}_features_{lo:04d}.npy',done['features'][arm]),allow_pickle=False);rows=S80.read(local(f'{arm}_records_{lo:04d}.json',done['records'][arm]));assert x.shape==(20,768) and x.dtype==np.float32 and np.isfinite(x).all() and len(rows)==20
   assert np.max(np.abs(np.linalg.norm(x,axis=1)-1))<2e-5
   host='old' if arm in ('coupled','same_donor','different_donor') else 'new'
   for i,row in enumerate(rows,lo):
    q=queries[i];assert row['arm']==arm and row['index']==i and all(row[k]==q[k] for k in ('query_id','stimulus_id','array_row'))
    donor=same[i] if arm=='same_donor' else different[i] if arm=='different_donor' else i
    assert row['EEG_row']==(None if arm=='uniform' else donor)
    arr=np.zeros((1,1000),np.float32) if arm=='uniform' else logits[host][donor:donor+1]
    assert row['EEG_logit_row_sha256']==hashlib.sha256(arr.tobytes()).hexdigest()
    assert row['class_draw_seed']==S80.seed(q['stimulus_id'],'latent') and row['token_seed']==S80.seed(q['stimulus_id'],'token')
    assert row['training_source_host']==host and row['execution_host']=='old' and row['cached_fixed_final_EEG_logits']
    assert row['model_SHA256']==models[host] and row['native_class_table_SHA256']==done['native_class_table_SHA256'] and 0<=row['native_class_id']<1000
    assert row['generation']==dict(one_class_draw=True,one_image=True,best_of_N=1,guidance=0.,top_k=0,top_p=0.,resolution=[256,256],scales=[1,2,3,4,5,6,8,10,13,16])
    assert not row['normal_inference_reads_query_image_label_or_gallery'] and row['labels_used_only_to_construct_experimental_EEG_permutations']==(arm in ('same_donor','different_donor'))
    assert row['image']['path'] not in seen;seen.add(row['image']['path']);identities[arm].append(row['query_id'])
   features[arm].append(x);new_counts[arm]+=len(rows)
 assert all(v==320 for v in new_counts.values())
 for arm in ARMS:
  features[arm]=np.concatenate(features[arm]);assert features[arm].shape==(400,768)
  assert np.array_equal(features[arm][:80],first[arm]) and identities[arm]==[q['query_id'] for q in queries[80:]]
 # All five complete gates precede any access to the whole-panel effect.
 target_path=S80.RECEIVED/'old/targets400.npy';assert S80.sha(target_path)==prior['target_features']['sha256'];targets=np.load(target_path,allow_pickle=False)
 gr=L/'audit/EEG_native_marginal_images_20261029_v2/received_stage400_v1';greceipt=S80.read(gr/'receiving_completion.json');gpath=gr/'old/coupled_features.npy'
 matching=[v for v in greceipt['files'].values() if Path(v['local']).resolve()==gpath.resolve()];assert len(matching)==1 and S80.sha(gpath)==matching[0]['remote']['sha256']
 assert S80.sha(gr/'old/panel400.json')==S80.PANEL_SHA and S80.sha(gr/'old/targets400.npy')==S80.sha(target_path)
 grows=S80.read(gr/'old/coupled_records.json');assert [r['query_id'] for r in grows]==[q['query_id'] for q in queries]
 features['frozen_Gaussian']=np.load(gpath,allow_pickle=False);assert np.array_equal(features['frozen_Gaussian'][:80],first['frozen_Gaussian'])
 assert S80.sha(math.__file__)==S80.METRIC_SHA
 arrays={a:math.metric_arrays(x,targets,queries,gallery) for a,x in features.items()};separate={a:independent_arrays(x,targets,queries,gallery) for a,x in features.items()}
 errors={a:{m:float(np.max(np.abs(row[m]-separate[a][m]))) for m in math.METRICS} for a,row in arrays.items()};assert max(v for row in errors.values() for v in row.values())<=2e-6
 pairs={f'coupled_minus_{b}':('coupled',b) for b in ('visual','uniform','same_donor','different_donor','frozen_Gaussian')};pairs['visual_minus_uniform']=('visual','uniform')
 contrasts={name:math.compare(arrays[a],arrays[b],queries) for name,(a,b) in pairs.items()}
 independent={name:independent_contrast(separate[a],separate[b],queries) for name,(a,b) in pairs.items()}
 cierror=max(abs(x-y) for name,row in contrasts.items() for m,item in row.items() for x,y in zip(item['descriptive_concept_bootstrap95'],independent[name][m]['descriptive_concept_bootstrap95']));assert cierror<=2e-6
 def practical(name):
  x=contrasts[name]['DINO2AFC'];return x['difference']>=.01 and x['descriptive_concept_bootstrap95'][0]>0
 checks=dict(visual_DINO_practical=practical('coupled_minus_visual'),uniform_DINO_practical=practical('coupled_minus_uniform'),
  no_within10_loss_vs_visual=contrasts['coupled_minus_visual']['within_concept10_top1']['difference']>=0,no_within10_loss_vs_uniform=contrasts['coupled_minus_uniform']['within_concept10_top1']['difference']>=0,
  gain_over_frozen_Gaussian=practical('coupled_minus_frozen_Gaussian'),EEG_pairing_over_different_donor=contrasts['coupled_minus_different_donor']['DINO2AFC']['difference']>0 and contrasts['coupled_minus_different_donor']['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0)
 out=D/'stage400_results_v2';out.mkdir(exist_ok=False)
 np.savez_compressed(out/'all6_features.npz',**features);np.savez_compressed(out/'all_metrics.npz',**{a+'__'+m:x for a,row in arrays.items() for m,x in row.items()})
 with (out/'per_query.csv').open('x',newline='',encoding='utf8') as stream:
  writer=csv.DictWriter(stream,fieldnames=['arm','query_index','query_id','stimulus_id','concept_id',*math.METRICS]);writer.writeheader()
  for arm,row in arrays.items():
   for i,q in enumerate(queries):writer.writerow(dict(arm=arm,query_index=i,query_id=q['query_id'],stimulus_id=q['stimulus_id'],concept_id=q['concept_id'],**{m:float(x[i]) for m,x in row.items()}))
 S80.write(out/'independent_CPU_rescore.json',dict(status='PASS_ALL6_400_METRIC_CELLS_AND_ALL6_CONCEPT_BOOTSTRAPS',metric_cells=12000,array_max_errors=errors,max_interval_error=cierror,feature_encoder_independently_rerun=False,first80_features_bit_exact=True))
 result=dict(status='COMPLETE_ALL5_CATEGORICAL_FULL400_OLD_ONLY_SCREEN',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=400,gallery_targets=400,concepts=40,
  decision='PROMISING_FOR_MULTISUBJECT_FULL_EVALUATION' if all(checks.values()) else 'STOP_INSTANCE',preregistered_checks=checks,means={a:{m:float(x.mean()) for m,x in row.items()} for a,row in arrays.items()},contrasts=contrasts,
  receiving_completion=S80.ref(R/'receiving_completion.json'),prior80=S80.ref(D/'stage80_results_v1/summary.json'),source=S80.ref(__file__),queue=queue,
  all5_complete_before_whole_panel_score=True,first80_per_arm_reused=80,source_fits_unchanged=True,NEW_GPU_work_started=False,SOTA_established=False,independent_confirmation=False,
  limits='Exposed single-subject development panel; seed1729; intervals do not cover training randomness or screening selection. Gaussian comparison jointly changes likelihood/condition family and top600-to-top0. Ordinary remains a stronger different-system reference.',seconds=time.monotonic()-started)
 S80.write(out/'summary.json',result)
 print(json.dumps(dict(status=result['status'],decision=result['decision'],checks=checks,means=result['means'],DINO_contrasts={k:v['DINO2AFC'] for k,v in contrasts.items()})))


if __name__=='__main__':run()
