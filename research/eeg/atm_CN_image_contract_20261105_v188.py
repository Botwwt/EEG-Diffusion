"""Missing same-head continuation images; exact old references, one new arm."""
import json,os,sys
from pathlib import Path
import numpy as np
ARMS=('frozen_positive_NCE','R4_continued','P_trunk_N_head','C_trunk_N_head')
CONTRASTS=tuple((ARMS[j],ARMS[i]) for i in range(4) for j in range(i+1,4))
PAPER=('eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav')
R4_METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
R80_METRICS=('target_cosine','DINO2AFC','gallery_top1','gallery_top5','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')+PAPER

def setup(path):
 q=json.loads(Path(path).read_bytes());assert q['status']=='LOCKED_MATCHED_HEAD_CN_FULL600_IMAGES'
 assert q['host']=='old' and q['seed']==1729 and q['arms']==list(ARMS)
 assert q['maximum_output_bytes']==384<<20 and q['wrapper_reserve_bytes']==1<<20
 assert q['new_images']==600 and q['reused_images']==1800 and q['new_training']==0
 assert q['contrasts']==[list(x) for x in CONTRASTS]
 sys.path.insert(0,str(Path(q['prior_policy']['path']).parent))
 import atm_block_image_contract_20261105_v185 as B
 old=B.setup(q['prior_policy']['path']);K=old['K'];K.checked(q['prior_policy'])
 for pin in list(q['sources'].values())+list(q['runtime_sources'].values()):K.checked(pin)
 probe=K.read(K.checked(q['reserve_probe']))
 assert probe['proposed_settlements']==q['settlements'] and probe['new_cap_bytes']==q['maximum_output_bytes']
 for pin in q['closed_launches']:
  la=K.read(K.checked(pin));assert not K.process(la['PID'],la['start_ticks'])['live']
 audit=K.read(K.checked(q['diagnostic_audit']));assert audit['status']=='COMPLETE_ALL600_MATCHED_HEAD_CONDITIONS_2400_CPU64_FORWARD_CHECKS'
 assert audit['arms']==['NN','PN','CN','CC'] and audit['paired_intervals']==130
 for pin in audit['parents']:K.checked(pin)
 for v in probe['completion_pins']:assert K.read(K.checked(v['pin']))['status']==v['status']
 rows=K.read(K.checked(q['diagnostic_queries']));assert rows==old['queries']
 bank=np.load(K.checked(q['diagnostic_conditions']));assert bank.shape==(600,4,1024) and np.isfinite(bank).all()
 values=bank[:,[0,3,1,2]].copy()
 assert np.array_equal(values[:,0],old['values'][:,0]) and np.array_equal(values[:,2],old['values'][:,3])
 assert np.array_equal(values[:,1],old['reference']['values'][:,1])
 refs={};papers={};ref_feats={}
 for domain in ('R4','R80'):
  br=B.records(old,domain);pr=old['O'].records(old['reference'],domain)
  refs[domain]={ARMS[0]:br[ARMS[0]],ARMS[1]:pr[ARMS[1]],ARMS[2]:br[ARMS[2]]}
  ep=K.read(old['out']/('encode_'+domain+'_completion.json'))
  ec=K.read(old['reference']['out']/('encode_'+domain+'_completion.json'))
  assert np.array_equal(np.load(K.checked(ep['target'])),np.load(K.checked(ec['target'])))
  ref_feats[domain]=dict(arms=[dict(features=ep['features'],index=0),dict(features=ec['features'],index=1),dict(features=ep['features'],index=3)],target=ep['target'])
 for arm in ARMS[:3]:
  p=(old['reference']['out'] if arm==ARMS[1] else old['out'])/('paper_'+arm+'_completion.json')
  papers[arm]=K.ref(p);v=K.read(p);assert v['status']=='COMPLETE_ALL200_AUTHOR_EIGHT_METRICS';K.checked(v['values'])
 result=dict(old);result.update(q=q,K=K,B=B,reference=old,reference_records=refs,reference_papers=papers,reference_features=ref_feats,
  values=values,out=Path(q['output_root']),positive=refs['R4'][ARMS[0]],baseline_reexport_max_difference=0.)
 return result

def resources(c,starting=False):
 q,K=c['q'],c['K'];base=c['reference'];bq=base['q']
 oldprobe=K.read(K.checked(bq['reserve_probe']));r=oldprobe['inherited_quote']
 prior=base['reference']['train'];ledger=prior['ledger']
 K.checked(prior['q']['reserve_ledger']);K.checked(ledger['source_probe'])
 assert K.allocated(prior['q']['project_root'])==r['own_used_bytes']
 assert r['required_including_this_project']==ledger['prior_quote']['required_including_this_project']-ledger['settled_unused_future_bytes']+ledger['profile_wrapper_reserve_bytes']+prior['q']['maximum_output_bytes']-r['own_used_bytes']
 for old in ledger['settlements']:
  assert K.allocated(old['project'])==old['actual_allocated_bytes']
  for pin in old['all_launches']:
   la=K.read(K.checked(pin));assert not K.process(la['PID'],la.get('start_ticks'))['live']
  assert K.read(K.checked(old['completion']))['status']==old['completion_status']
 credit=0
 for s in bq['settlements']:
  assert sum(K.allocated(p) for p in s['roots'])==s['actual_allocated_bytes']
  assert s['unused_future_allowance']==s['prior_charged_remaining']==s['cap']-s['actual_allocated_bytes']
  for pin in s['policies']+s['source_policy_pins']+s['launch_pins']:K.checked(pin)
  for pin in s['completion_pins']:assert K.read(K.checked(pin['pin']))['status']==pin['status']
  credit+=s['unused_future_allowance']
 assert credit==1334829056 and bq['duplicate_credit_correction_bytes']==663662592
 probe=K.read(K.checked(q['reserve_probe']));settled=0
 assert len(q['settlements'])==2 and q['settlements'][0]['project']==bq['project_root']
 assert q['settlements'][1]['project'].endswith('/ATM_exact_parent_R80_20261105_v174')
 assert not any(x['project'] in y['roots'] for x in q['settlements'] for y in bq['settlements'])
 for s in q['settlements']:
  assert K.allocated(s['project'])==s['actual_allocated_bytes']
  assert s['unused_future_bytes']==s['cap']-s['actual_allocated_bytes'];settled+=s['unused_future_bytes']
 for pin in q['closed_launches']:
  la=K.read(K.checked(pin));assert not K.process(la['PID'],la['start_ticks'])['live']
 assert K.read(K.checked(probe['old_exact_parent_policy']))['maximum_output_bytes']==192<<20
 old_required=r['required_including_this_project']+663662592-credit+bq['maximum_output_bytes']-q['settlements'][0]['actual_allocated_bytes']
 assert old_required==probe['quote']['required_including_this_project']
 used=K.allocated(q['project_root']) if Path(q['project_root']).exists() else 0
 assert used<=q['maximum_output_bytes']
 required=old_required-settled+q['maximum_output_bytes']+q['wrapper_reserve_bytes']-used
 assert required>=58<<30
 fs=os.statvfs(Path(q['project_root']).parent);free=fs.f_bavail*fs.f_frsize;assert free>=required,(free,required)
 mi={s.split(':',1)[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if ':' in s and len(s.split())>=2 and s.split()[1].isdigit()}
 head=mi['MemAvailable'];mx=Path('/sys/fs/cgroup/memory.max');cu=Path('/sys/fs/cgroup/memory.current')
 if mx.exists() and cu.exists() and mx.read_text().strip()!='max':head=min(head,int(mx.read_text())-int(cu.read_text()))
 assert head>=(24<<30 if starting else 2<<30)
 return dict(required_including_this_project=required,free_bytes=free,own_used_bytes=used,own_remaining_bytes=q['maximum_output_bytes']-used,
 CPU_headroom_bytes=head,two_new_closed_unused_credits_once=settled,wrapper_reserve_held=q['wrapper_reserve_bytes'],
 old_duplicate_credit_correction_held=663662592,old_nine_settlement_not_repeated=1334829056,
 all_actual_files_retained=True,all_other_forecasts_50GiB_floor_8GiB_buffer_held=True)

def records(c,domain):
 n=400 if domain=='R4' else 200;result={}
 for a in ARMS:
  result[a]=c['reference_records'][domain][a] if a in ARMS[:3] else [json.loads(v) for v in (c['out']/'images'/domain/a/'images.jsonl').read_text().splitlines()]
  assert [r['query_index'] for r in result[a]]==list(range(n))
 return result

def paired(arrays,queries,domain):
 concepts=sorted({r['concept_id'] for r in queries});groups=[[i for i,r in enumerate(queries) if r['concept_id']==g] for g in concepts]
 assert len(groups)==(40 if domain=='R4' else 200)
 draw=np.random.default_rng(1729).integers(0,len(groups),(1000,len(groups)));result={};diffs={}
 for a,b in CONTRASTS:
  name=a+'-'+b;result[name]={}
  for key in R4_METRICS if domain=='R4' else R80_METRICS:
   delta=arrays[a][key]-arrays[b][key];valid=np.isfinite(delta);gm=np.array([delta[g].mean() for g in groups])
   result[name][key]=dict(difference=float(delta.mean()) if valid.all() else None,
    descriptive_concept_bootstrap95=np.quantile(gm[draw].mean(1),[.025,.975]).tolist() if valid.all() else None,
    defined_queries=int(valid.sum()),total_queries=len(queries),concepts=len(groups))
   diffs[name+'__'+key]=delta
 return result,diffs
