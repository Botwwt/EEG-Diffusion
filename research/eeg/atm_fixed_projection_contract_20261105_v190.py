"""One fixed-head continuation; inherited baselines and explicit full budget."""
import json,os,sys
from pathlib import Path
import numpy as np
ARMS=('frozen_positive_NCE','R4_continued','C_trunk_N_head','fixed_N_head')
CONTRASTS=tuple((ARMS[j],ARMS[i]) for i in range(4) for j in range(i+1,4))
PAPER=('eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav')
R4_METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
R80_METRICS=('target_cosine','DINO2AFC','gallery_top1','gallery_top5','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')+PAPER
def setup(path):
 q=json.loads(Path(path).read_bytes());assert q['status']=='LOCKED_FIXED_N_HEAD_T10_COMPLETE_R4_R80'
 assert q['host']=='old' and q['seed']==1729 and q['arms']==list(ARMS)
 assert q['epochs']==10 and q['updates']==830 and q['maximum_output_bytes']==384<<20
 sys.path.insert(0,str(Path(q['prior_policy']['path']).parent))
 import atm_CN_image_contract_20261105_v188 as C
 base=C.setup(q['prior_policy']['path']);K=base['K'];K.checked(q['prior_policy'])
 for pin in list(q['sources'].values())+list(q['runtime_sources'].values()):K.checked(pin)
 probe=K.read(K.checked(q['reserve_probe']));assert probe['status']=='PASS_PROFILE_AND_SEVEN_CLOSED_STILL_CHARGED_ALLOWANCES_NO_DELETION'
 assert probe['proposed_settlements']==q['settlements']
 prof=K.read(K.checked(q['profile_completion']));assert prof['status']=='PASS_NATIVE_C_FAST_PATH_BITEXACT_AND_FIXED_HEAD_MASKED_ADAM'
 sys.path.insert(0,str(Path(q['training_policy']['path']).parent))
 import atm_repeat_count_training_contract_20261105_v1 as T
 train=T.setup(K.checked(q['training_policy']));out=Path(q['output_root']);fit=Path(q['project_root'])/'fit'
 refs={d:{a:C.records(base,d)[a] for a in ARMS[:3]} for d in ['R4','R80']}
 ref_feats={}
 for d in ['R4','R80']:
  ec=K.read(base['out']/('encode_'+d+'_completion.json'))
  ref_feats[d]=dict(arms=[dict(features=ec['features'],index=j) for j in [0,1,3]],target=ec['target'])
 papers={a:K.ref(base['out']/('paper_'+a+'_completion.json')) for a in ARMS[:3]}
 values=None
 if (fit/'conditions_completion.json').exists():
  done=K.read(fit/'conditions_completion.json');assert done['status']=='COMPLETE_FIXED_HEAD_EEG_ONLY600_CONDITIONS'
  new=np.load(K.checked(done['conditions']));assert new.shape==(600,1024) and np.isfinite(new).all()
  values=np.concatenate([base['values'][:,[0,1,3]],new[:,None]],axis=1)
 result=dict(base);result.update(q=q,K=K,C=C,reference=base,train=train,out=out,fit=fit,values=values,
  reference_records=refs,reference_features=ref_feats,reference_papers=papers,positive=refs['R4'][ARMS[0]],baseline_reexport_max_difference=0.)
 return result
def resources(c,starting=False):
 q,K=c['q'],c['K'];p=K.read(K.checked(q['reserve_probe']));credit=0
 for s in q['settlements']:
  assert K.allocated(s['project'])==s['actual_allocated_bytes']
  assert s['unused_future_bytes']==s['cap']-s['actual_allocated_bytes'];credit+=s['unused_future_bytes']
  for pin in s['policies']:K.checked(pin)
  for pin in s['completions']:assert K.read(K.checked(pin['pin']))['status']==pin['status']
  for pin in s['launches']:
   la=K.read(K.checked(pin))
   if 'PID' in la:assert not K.process(la['PID'],la.get('start_ticks'))['live']
 assert credit==p['credit'] and len(q['settlements'])==7
 used=K.allocated(q['project_root']);assert used<=q['maximum_output_bytes']
 required=p['quote']['required_including_this_project']-credit+q['maximum_output_bytes']+(1<<20)-used
 assert required>=58<<30
 fs=os.statvfs(q['project_root']);free=fs.f_bavail*fs.f_frsize;assert free>=required,(free,required)
 mi={s.split(':',1)[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if ':' in s and len(s.split())>=2 and s.split()[1].isdigit()}
 head=mi['MemAvailable'];mx=Path('/sys/fs/cgroup/memory.max');cu=Path('/sys/fs/cgroup/memory.current')
 if mx.exists() and cu.exists() and mx.read_text().strip()!='max':head=min(head,int(mx.read_text())-int(cu.read_text()))
 assert head>=(24<<30 if starting else 2<<30)
 return dict(required_including_this_project=required,free_bytes=free,own_used_bytes=used,own_remaining_bytes=q['maximum_output_bytes']-used,CPU_headroom_bytes=head,
 seven_closed_unused_once=credit,all_prior_credits_and_correction_retained=True,all_50GiB_floor_8GiB_buffer_and_other_forecasts_held=True,no_deletion=True)
def records(c,domain):
 n=400 if domain=='R4' else 200;result={}
 for a in ARMS:
  result[a]=c['reference_records'][domain][a] if a in ARMS[:3] else [json.loads(v) for v in (c['out']/'images'/domain/a/'images.jsonl').read_text().splitlines()]
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
