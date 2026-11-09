"""Frozen baselines, one EEG response-repair study, complete image comparison."""
import json,os,sys
from pathlib import Path
import numpy as np
ARMS=('frozen_positive_NCE','C_trunk_N_head','response_repair','matched_direction')
CONTRASTS=tuple((ARMS[j],ARMS[i]) for i in range(4) for j in range(i+1,4))
PAPER=('eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav')
R4_METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
R80_METRICS=('target_cosine','DINO2AFC','gallery_top1','gallery_top5','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')+PAPER

def setup(path):
 q=json.loads(Path(path).read_bytes());assert q['status']=='LOCKED_EEG_RESPONSE_REPAIR_FULL_T10_R4_R80'
 assert q['host']=='old' and q['seed']==1729 and q['arms']==list(ARMS)
 assert q['epochs']==10 and q['updates_per_arm']==830 and q['maximum_output_bytes']==768<<20
 sys.path.insert(0,str(Path(q['prior_policy']['path']).parent))
 import atm_fixed_projection_contract_20261105_v190 as B
 base=B.setup(q['prior_policy']['path']);K=base['K'];K.checked(q['prior_policy'])
 for pin in list(q['sources'].values())+list(q['runtime_sources'].values()):K.checked(pin)
 prof=K.read(K.checked(q['profile_completion']));assert prof['status']=='PASS_TRAINING_ONLY_RESPONSE_CONTRAST_TARGETS_AND_MATCHED_REPAIR_GRADIENTS'
 assert prof['all_steps_discarded'] and prof['all_baselines_unchanged']
 for pin in q['models'].values():K.checked(pin)
 refs={d:{a:B.records(base,d)[a] for a in ARMS[:2]} for d in ['R4','R80']}
 feats={}
 for d in ['R4','R80']:
  ec=K.read(base['out']/('encode_'+d+'_completion.json'))
  feats[d]=dict(arms=[dict(features=ec['features'],index=j) for j in [0,2]],target=ec['target'])
 out=Path(q['output_root']);fit=Path(q['project_root'])/'fit';values=None
 if (fit/'conditions_completion.json').exists():
  done=K.read(fit/'conditions_completion.json');assert done['status']=='COMPLETE_RESPONSE_REPAIR_EEG_ONLY1200_CONDITIONS'
  val=np.load(K.checked(done['conditions']));assert val.shape==(600,2,1024) and np.isfinite(val).all()
  values=np.concatenate([base['values'][:,[0,2]],val],axis=1)
 c=dict(base);c.update(q=q,K=K,reference=base,train=base['train'],out=out,fit=fit,values=values,
  reference_records=refs,reference_features=feats,reference_papers={a:K.ref(base['out']/('paper_'+a+'_completion.json')) for a in ARMS[:2]},positive=refs['R4'][ARMS[0]],baseline_reexport_max_difference=0.)
 return c

def resources(c,starting=False):
 q,K=c['q'],c['K'];ledger=q['resource_ledger'];s=ledger['one_new_settlement']
 assert s['cap']==5<<30 and s['unused_future_bytes']==905576448
 assert sum(K.allocated(p) for p in s['roots'])==s['actual_allocated_bytes']==s['cap']-s['unused_future_bytes']
 for p in s['policies']:K.checked(p)
 for z in s['completions']:assert K.read(K.checked(z['pin']))['status']==z['status']
 for pin in s['launches']:
  la=K.read(K.checked(pin))
  if 'PID' in la:assert not K.process(la['PID'],la.get('start_ticks'))['live']
 assert K.allocated(ledger['frozen_parent_root'])==ledger['frozen_parent_allocated_bytes']
 used=K.allocated(q['project_root']);assert used<=q['maximum_output_bytes']
 required=ledger['prior_quote']['required_including_this_project']-s['unused_future_bytes']+q['maximum_output_bytes']+q['wrapper_reserve_bytes']-used
 assert required>=58<<30
 fs=os.statvfs(q['project_root']);free=fs.f_bavail*fs.f_frsize;assert free>=required,(free,required)
 mi={s.split(':',1)[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if ':' in s and len(s.split())>=2 and s.split()[1].isdigit()}
 head=mi['MemAvailable'];mx=Path('/sys/fs/cgroup/memory.max');cu=Path('/sys/fs/cgroup/memory.current')
 if mx.exists() and cu.exists() and mx.read_text().strip()!='max':head=min(head,int(mx.read_text())-int(cu.read_text()))
 assert head>=(24<<30 if starting else 2<<30)
 return dict(required_including_this_project=required,free_bytes=free,own_used_bytes=used,own_remaining_bytes=q['maximum_output_bytes']-used,CPU_headroom_bytes=head,
  closed_VAR_unused_once=s['unused_future_bytes'],all_older_credits_correction_50GiB_8GiB_and_other_forecasts_retained=True,no_deletion=True)

def records(c,domain):
 n=400 if domain=='R4' else 200;result={}
 for a in ARMS:
  result[a]=c['reference_records'][domain][a] if a in ARMS[:2] else [json.loads(v) for v in (c['out']/'images'/domain/a/'images.jsonl').read_text().splitlines()]
  assert [r['query_index'] for r in result[a]]==list(range(n))
 return result

def paired(arrays,queries,domain):
 groups=[[i for i,r in enumerate(queries) if r['concept_id']==g] for g in sorted({r['concept_id'] for r in queries})]
 assert len(groups)==(40 if domain=='R4' else 200)
 draw=np.random.default_rng(1729).integers(0,len(groups),(1000,len(groups)));result={};diffs={}
 for a,b in CONTRASTS:
  name=a+'-'+b;result[name]={}
  for key in R4_METRICS if domain=='R4' else R80_METRICS:
   delta=arrays[a][key]-arrays[b][key];valid=np.isfinite(delta);gm=np.array([delta[g].mean() for g in groups])
   result[name][key]=dict(difference=float(delta.mean()) if valid.all() else None,descriptive_concept_bootstrap95=np.quantile(gm[draw].mean(1),[.025,.975]).tolist() if valid.all() else None,defined_queries=int(valid.sum()),total_queries=len(queries),concepts=len(groups))
   diffs[name+'__'+key]=delta
 return result,diffs
