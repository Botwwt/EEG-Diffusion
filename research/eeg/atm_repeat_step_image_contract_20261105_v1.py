"""Fixed complete R4/R80 comparison in the already reserved training project."""
import json,sys
from pathlib import Path
import numpy as np
ARMS=('frozen_positive_NCE','R4_continued','positive_views','repeat_step','sign_control','norm_control')
CONTRASTS=tuple(('repeat_step',b) for b in ('R4_continued','sign_control','norm_control','frozen_positive_NCE','positive_views'))+(('sign_control','R4_continued'),('norm_control','R4_continued'))
PAPER=('eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav')
R4_METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
R80_METRICS=('target_cosine','DINO2AFC','gallery_top1','gallery_top5','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')+PAPER
def setup(path):
 q=json.loads(Path(path).read_bytes());assert q['status']=='LOCKED_COMPLETE_REPEAT_STEP_R4_R80_IMAGES'
 assert q['host']=='old' and q['seed']==1729 and q['new_images']==1800 and q['arms']==list(ARMS)
 assert q['contrasts']==[list(v) for v in CONTRASTS] and q['metrics_R4']==list(R4_METRICS) and q['metrics_R80']==list(R80_METRICS)
 sys.path.insert(0,str(Path(q['training_policy']['path']).parent));import atm_repeat_step_training_contract_20261105_v1 as T
 train=T.setup(q['training_policy']['path']);K=train['K'];K.checked(q['training_policy'])
 assert q['project_root']==train['q']['project_root'] and q['maximum_output_bytes']==train['q']['maximum_output_bytes']
 launch=K.read(Path(q['training_policy']['path']).parent/'actual_launch.json');assert not K.process(launch['PID'],launch['start_ticks'])['live']
 done=K.read(K.checked(q['training_completion']));assert done['status']=='COMPLETE_THREE_ARM_T10_AND_ALL3600_CONDITIONS_IMAGES_PENDING'
 for e in done['events']:
  j=K.read(K.checked(e['receipt']));assert j['status']=='COMPLETED' and not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
 assert K.read(K.checked(done['CPU_audit']))['status']=='PASS_ALL3600_CPU_CONDITIONS_AND_ALL30_EPOCHS'
 for pin in q['sources'].values():K.checked(pin)
 for pin in q['runtime_sources'].values():K.checked(pin)
 sys.path.insert(0,str(Path(q['reference_image_policy']['path']).parent));import atm_repeat_count_image_contract_20261105_v1 as H
 old=H.setup(K.checked(q['reference_image_policy']));refdone=K.read(K.checked(q['reference_image_completion']));assert refdone['status']=='COMPLETE_THREE_ARM_COUNT_RISK_TRAINING_IMAGE_EVALUATION'
 refprofile=K.read(old['out']/'profile_completion.json');historical=np.load(K.checked(refprofile['conditions']))
 exported=K.read(train['out']/'conditions_completion.json');assert exported['status']=='COMPLETE_EEG_ONLY_ALL3600_CONDITIONS'
 values=np.load(K.checked(exported['conditions']));assert values.shape==(600,6,1024) and historical.shape==(600,4,1024)
 difference=float(abs(values[:,:3]-historical[:,:3]).max());assert difference<5e-5
 values=values.copy();values[:,:3]=historical[:,:3]
 refs={domain:{arm:rows for arm,rows in H.records(old,domain).items() if arm in ARMS[:3]} for domain in ['R4','R80']}
 return dict(q=q,K=K,T=T,train=train,old=old['old'],r80=old['r80'],repo=old['repo'],parent=old['parent'],cv=old['cv'],out=Path(q['output_root']),
  values=values,positive=refs['R4'][ARMS[0]],prepared=old['prepared'],queries=old['queries'],r80noise=old['r80noise'],
  baseline_reexport_max_difference=difference,physical=train['physical'],reference=old,reference_records=refs)

def resources(c,starting=False):return c['T'].resources(c['train'],starting=starting)

def records(c,domain):
 n=400 if domain=='R4' else 200;result={}
 for arm in ARMS:
  if arm in ARMS[:3]:result[arm]=c['reference_records'][domain][arm]
  else:result[arm]=[json.loads(s) for s in (c['out']/'images'/domain/arm/'images.jsonl').read_text().splitlines()]
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
