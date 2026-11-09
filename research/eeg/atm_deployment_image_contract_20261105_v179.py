"""Two fixed EEG supervision paths, two pre-existing NCE references, full galleries."""
import json,sys
from pathlib import Path
import numpy as np
ARMS=('frozen_positive_NCE','R4_continued','stochastic_P','deployment_P')
CONTRASTS=(('deployment_P','stochastic_P'),('deployment_P','frozen_positive_NCE'),('deployment_P','R4_continued'),('stochastic_P','frozen_positive_NCE'),('stochastic_P','R4_continued'))
PAPER=('eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav')
R4_METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
R80_METRICS=('target_cosine','DINO2AFC','gallery_top1','gallery_top5','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')+PAPER
def setup(path):
 q=json.loads(Path(path).read_bytes());assert q['status']=='LOCKED_FULL_DEPLOYMENT_SUPERVISION_R4_R80_IMAGES'
 assert q['host']=='old' and q['seed']==1729 and q['arms']==list(ARMS) and q['new_images']==1200
 assert q['contrasts']==[list(z) for z in CONTRASTS] and q['metrics_R4']==list(R4_METRICS) and q['metrics_R80']==list(R80_METRICS)
 sys.path.insert(0,str(Path(q['training_policy']['path']).parent));import atm_deployment_training_contract_20261105_v179 as T
 train=T.setup(q['training_policy']['path']);K=train['K'];K.checked(q['training_policy'])
 assert q['project_root']==train['q']['project_root'] and q['maximum_output_bytes']==train['q']['maximum_output_bytes']
 la=K.read(K.checked(q['training_launch']));assert not K.process(la['PID'],la['start_ticks'])['live']
 done=K.read(K.checked(q['training_completion']));assert done['status']=='COMPLETE_TWO_ARM_T10_ALL2400_CONDITIONS_IMAGES_PENDING'
 for e in done['events']:
  if e['GPU']:
   j=K.read(K.checked(e['receipt']));assert j['status']=='COMPLETED' and j['exit_code']==0 and not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
 assert K.read(K.checked(done['CPU_audit']))['status']=='PASS_ALL2400_CPU_CONDITIONS_ALL20_EPOCHS_FOUR_SAVED_STATES'
 for p in list(q['sources'].values())+list(q['runtime_sources'].values()):K.checked(p)
 for arm,p in q['new_model_pins'].items():
  K.checked(p);assert K.read(train['out']/'fits'/arm/'completion10.json')['checkpoint']==p
 n=train['old'];export=K.read(K.checked(done['conditions']));values=np.load(K.checked(export['conditions']))
 assert values.shape==(600,4,1024) and export['arms']==list(ARMS)
 difference=float(abs(values[:,:2]-n['values'][:,:2]).max());assert difference<5e-5
 values=values.copy();values[:,:2]=n['values'][:,:2]
 refs={domain:{arm:n['reference_records'][domain][arm] for arm in ARMS[:2]} for domain in ['R4','R80']}
 result=dict(n);result.update(q=q,K=K,T=T,train=train,out=Path(q['output_root']),values=values,reference=n,reference_records=refs,
  baseline_reexport_max_difference=difference,positive=refs['R4'][ARMS[0]],physical=train['physical'])
 return result
def resources(c,starting=False):return c['T'].resources(c['train'],starting)
def records(c,domain):
 n=400 if domain=='R4' else 200;result={}
 for arm in ARMS:
  result[arm]=c['reference_records'][domain][arm] if arm in ARMS[:2] else [json.loads(s) for s in (c['out']/'images'/domain/arm/'images.jsonl').read_text().splitlines()]
  assert [r['query_index'] for r in result[arm]]==list(range(n))
 return result
def paired(arrays,queries,domain):
 concepts=sorted({r['concept_id'] for r in queries});groups=[[i for i,r in enumerate(queries) if r['concept_id']==g] for g in concepts]
 assert len(groups)==(40 if domain=='R4' else 200) and all(len(g)==(10 if domain=='R4' else 1) for g in groups)
 draw=np.random.default_rng(1729).integers(0,len(groups),(1000,len(groups)));result={};diffs={}
 for a,b in CONTRASTS:
  name=a+'-'+b;result[name]={}
  for key in R4_METRICS if domain=='R4' else R80_METRICS:
   d=arrays[a][key]-arrays[b][key];diffs[name+'__'+key]=d;valid=np.isfinite(d);means=np.array([d[g].mean() for g in groups])
   result[name][key]=dict(difference=float(d.mean()) if valid.all() else None,
    descriptive_concept_bootstrap95=np.quantile(means[draw].mean(1),[.025,.975]).tolist() if valid.all() else None,
    defined_queries=int(valid.sum()),total_queries=len(queries),concepts=len(groups))
 return result,diffs
