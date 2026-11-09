"""Fixed complete R4/R80 comparison in the already reserved training project."""
import json,sys
from pathlib import Path
import numpy as np
ARMS=('frozen_positive_NCE','R4_continued','positive_views','count_corrected')
CONTRASTS=(('count_corrected','R4_continued'),('count_corrected','positive_views'),('R4_continued','frozen_positive_NCE'),('count_corrected','frozen_positive_NCE'))
PAPER=('eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav')
R4_METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
R80_METRICS=('target_cosine','DINO2AFC','gallery_top1','gallery_top5','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')+PAPER
def setup(path):
 q=json.loads(Path(path).read_bytes());assert q['status']=='LOCKED_COMPLETE_COUNT_RISK_R4_R80_IMAGES'
 assert q['host']=='old' and q['seed']==1729 and q['new_images']==2000 and q['arms']==list(ARMS)
 assert q['contrasts']==[list(v) for v in CONTRASTS] and q['metrics_R4']==list(R4_METRICS) and q['metrics_R80']==list(R80_METRICS)
 sys.path.insert(0,str(Path(q['training_policy']['path']).parent))
 import atm_repeat_count_training_contract_20261105_v1 as T
 train=T.setup(q['training_policy']['path']);K=train['K'];K.checked(q['training_policy'])
 assert q['project_root']==train['q']['project_root'] and q['maximum_output_bytes']==train['q']['maximum_output_bytes']
 launch=K.read(Path(q['training_policy']['path']).parent/'actual_launch.json');assert not K.process(launch['PID'],launch['start_ticks'])['live']
 done=K.read(K.checked(q['training_completion']));assert done['status']=='COMPLETE_THREE_ARM_T10_AND_ALL2400_CONDITIONS_IMAGES_PENDING'
 for e in done['events']:
  j=K.read(K.checked(e['receipt']));assert j['status']=='COMPLETED' and not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
 audit=K.read(K.checked(done['CPU_audit']));assert audit['status']=='PASS_ALL2400_CPU_CONDITIONS_AND_ALL30_EPOCHS'
 for pin in q['sources'].values():K.checked(pin)
 for pin in q['runtime_sources'].values():K.checked(pin)
 # Original host / metrics / target permissions, not a reimplementation.
 sys.path.insert(0,str(Path(train['q']['image_reference_policy']['path']).parent))
 import atm_conditional_stimulus_contract_20261104_v1 as O
 old=O.setup(train['q']['image_reference_policy']['path'])
 sys.path.insert(0,str(Path(train['q']['R80_reference_policy']['path']).parent))
 import enigma_R80_count_image_contract_20261104_v1 as R
 r80=R.setup(train['q']['R80_reference_policy']['path'])
 prep=K.read(old['out']/'prepared.json');positive=O.records(old,400)['nce_continued'];assert len(positive)==400
 exported=K.read(train['out']/'conditions_completion.json');assert exported['status']=='COMPLETE_EEG_ONLY_ALL2400_CONDITIONS'
 values=np.load(K.checked(exported['conditions']),allow_pickle=False);assert values.shape==(600,4,1024)
 # R4 baseline reuses exact historical conditioning bytes, not a numerically-close
 # re-export at a different EEG forward batch size. All new arms keep new bytes.
 oldvalues=np.load(K.checked(prep['conditions']),allow_pickle=False);assert oldvalues.shape==(400,5,1024)
 difference=float(abs(values[:400,0]-oldvalues[:,2]).max());assert difference<5e-5
 values=values.copy();values[:400,0]=oldvalues[:,2]
 for i,row in enumerate(positive):
  assert row['query_index']==i and all(row[k]==train['queries'][i][k] for k in O.IDENTITY)
 prior_records=K.read(K.checked(K.read(r80['out']/'image_audit.json')['ordered_records']))
 r80noise=[r for r in prior_records if r['arm']=='EMA_original'];assert [r['query_index'] for r in r80noise]==list(range(200))
 return dict(q=q,K=K,T=T,train=train,old=old,r80=r80,repo=old['repo'],parent=old['parent'],cv=old['cv'],out=Path(q['output_root']),
  values=values,positive=positive,prepared=prep,queries={'R4':train['queries'],'R80':r80['rows']},r80noise=r80noise,
  baseline_reexport_max_difference=difference,physical=train['physical'])

def resources(c,starting=False):return c['T'].resources(c['train'],starting=starting)

def records(c,domain):
 n=400 if domain=='R4' else 200;result={}
 for arm in ARMS:
  if domain=='R4' and arm==ARMS[0]:
   result[arm]=[dict(r,arm=arm,reused_original_arm='nce_continued') for r in c['positive']]
  else:
   path=c['out']/'images'/domain/arm/'images.jsonl';result[arm]=[json.loads(s) for s in path.read_text().splitlines()]
  assert [r['query_index'] for r in result[arm]]==list(range(n))
 return result

def paired(arrays,queries,domain):
 concepts=sorted({r['concept_id'] for r in queries});groups=[[i for i,r in enumerate(queries) if r['concept_id']==g] for g in concepts]
 assert len(groups)==(40 if domain=='R4' else 200) and all(len(g)==(10 if domain=='R4' else 1) for g in groups)
 draw=np.random.default_rng(1729).integers(0,len(groups),(1000,len(groups)));result={};diffs={}
 for a,b in CONTRASTS:
  name=a+'-'+b;result[name]={}
  for key in (R4_METRICS if domain=='R4' else R80_METRICS):
   d=arrays[a][key]-arrays[b][key];diffs[name+'__'+key]=d;valid=np.isfinite(d)
   means=np.array([d[g].mean() for g in groups]);result[name][key]=dict(difference=float(d.mean()) if valid.all() else None,
    descriptive_concept_bootstrap95=np.quantile(means[draw].mean(1),[.025,.975]).tolist() if valid.all() else None,
    defined_queries=int(valid.sum()),total_queries=len(queries),concepts=len(groups))
 return result,diffs
