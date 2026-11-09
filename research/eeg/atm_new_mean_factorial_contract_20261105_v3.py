"""Within-concept permutation interventions on one frozen positive EEG encoder."""
import json,os,sys,hashlib
from pathlib import Path
import numpy as np
ARMS=('correct','wrong_mean','wrong_center','wrong_both')
METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
IDENTITY=('query_id','subject_id','concept_id','stimulus_id','array_row','partition','image_seed')
CONTRASTS={'mean_given_correct_center':[1,-1,0,0], 'mean_given_wrong_center':[0,0,1,-1],
 'center_given_correct_mean':[1,0,-1,0], 'center_given_wrong_mean':[0,1,0,-1],
 'paired_total':[1,0,0,-1], 'mean_average':[.5,-.5,.5,-.5],
 'center_average':[.5,.5,-.5,-.5], 'interaction':[1,-1,-1,1]}
def sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def donors(rows):
 assert len(rows)==400 and len({r['query_id'] for r in rows})==400
 result=np.full(400,-1,np.int64)
 groups=sorted({r['concept_id'] for r in rows});assert len(groups)==40
 for concept in groups:
  ids=sorted([i for i,r in enumerate(rows) if r['concept_id']==concept],key=lambda i:rows[i]['stimulus_id'])
  assert len(ids)==10 and len({rows[i]['stimulus_id'] for i in ids})==10
  result[ids]=np.roll(ids,-1)
 assert sorted(result.tolist())==list(range(400)) and np.all(result!=np.arange(400))
 return result
def tokens(native,d):
 assert native.shape==(400,4,2048) and native.dtype==np.float16 and np.isfinite(native).all()
 assert d.shape==(400,) and sorted(d.tolist())==list(range(400))
 p=native.astype(np.float64);m=p.mean(1,keepdims=True);shift=m[d]-m
 # Difference form gives exact identity for a zero intervention.
 return np.stack([native,(p+shift).astype(np.float16),(p[d]-shift).astype(np.float16),native[d]],axis=1)
def paired(arrays,rows):
 groups=[[i for i,r in enumerate(rows) if r['concept_id']==g] for g in sorted({r['concept_id'] for r in rows})]
 assert len(groups)==40 and all(len(g)==10 for g in groups)
 draws=np.random.default_rng(1729).integers(0,40,(1000,40));pairs={};diffs={}
 for name,weights in CONTRASTS.items():
  pairs[name]={}
  for key in METRICS:
   v=sum(w*arrays[a][key] for a,w in zip(ARMS,weights));assert v.shape==(400,) and np.isfinite(v).all()
   means=np.array([v[g].mean() for g in groups]);diffs[name+'__'+key]=v
   pairs[name][key]=dict(difference=float(v.mean()),descriptive_concept_bootstrap95=np.quantile(means[draws].mean(1),[.025,.975]).tolist())
 return pairs,diffs
def setup(path):
 q=json.loads(Path(path).read_bytes());assert q['status']=='LOCKED_NEW_NCE_MEAN_CENTER_FACTORIAL_DIAGNOSTIC'
 assert q['host']=='old' and q['seed']==1729 and q['arms']==list(ARMS) and q['contrasts']==CONTRASTS
 assert q['new_images']==1200 and q['new_updates']==0 and q['maximum_output_bytes']==640<<20
 sys.path.insert(0,str(Path(q['prior_policy']['path']).parent));import atm_repeat_step_image_contract_20261105_v1 as P
 old=P.setup(q['prior_policy']['path']);K=old['K'];K.checked(q['prior_policy'])
 done=K.read(K.checked(q['prior_completion']));assert done['status']=='COMPLETE_PAIRED_REPEAT_STEP_TRAINING_IMAGE_EVALUATION'
 for pin in q['sources'].values():K.checked(pin)
 for pin in q['inputs'].values():K.checked(pin)
 for pin in q['closed_launches']:
  launch=K.read(K.checked(pin));assert not K.process(launch['PID'],launch.get('start_ticks'))['live']
 for e in done['events']:
  if e['GPU']:
   job=K.read(K.checked(e['receipt']));assert job['status']=='COMPLETED' and job['exit_code']==0
   assert not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live']
 values=old['values'][:400,0].copy();native=np.load(K.checked(q['inputs']['tokens']),mmap_mode='r')[:400,0].copy()
 rows=old['queries']['R4'];d=donors(rows)
 for i,r in enumerate(old['positive']):assert sha(native[i])==r['tokens_FP16_sha256'] and sha(values[i])==r['condition_FP32_sha256']
 return dict(q=q,K=K,P=P,prior=old,repo=old['repo'],out=Path(q['output_root']),rows=rows,values=values,native=native,donors=d,bank=tokens(native,d))
def resources(c,starting=False):
 q,K=c['q'],c['K'];train=c['prior']['train'];ledger=train['ledger']
 # Reconstruct the frozen inherited quote, checking all original allocations,
 # receipts and ownership, before applying the two explicit closed credits.
 assert K.allocated(train['prior']['q']['project_root'])==ledger['prior_project_used']
 K.checked(ledger['charge_snapshot']);old_credit=0
 for s in ledger['settlements']:
  pq=K.read(K.checked(s['policy']));done=K.read(K.checked(s['completion']))
  assert done['status']==s['completion_status'] and pq['maximum_output_bytes']==s['maximum_output_bytes']
  assert K.allocated(s['project'])==s['actual_allocated_bytes']
  for pin in s['all_launches']:
   a=K.read(K.checked(pin));assert not K.process(a['PID'],a.get('start_ticks'))['live']
  for pin in s['jobs']:
   j=K.read(K.checked(pin));assert j['status']=='COMPLETED' and j['exit_code']==0
   assert not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
  assert s['unused_future_allowance']==s['maximum_output_bytes']-s['actual_allocated_bytes'];old_credit+=s['unused_future_allowance']
 assert old_credit==ledger['settled_unused_future_bytes']
 base=ledger['prior_quote']['required_including_this_project']-old_credit;assert base>=58<<30
 old_used=K.allocated(train['q']['project_root']);assert old_used<=train['q']['maximum_output_bytes']
 inherited=base+train['q']['maximum_output_bytes']-old_used;credit=0
 for s in q['settlements']:
  assert K.allocated(s['project'])==s['actual_allocated_bytes']
  pq=K.read(K.checked(s['policy']));done=K.read(K.checked(s['completion']))
  assert pq['maximum_output_bytes']==s['maximum_output_bytes'] and done['status']==s['completion_status']
  assert s['unused_future_allowance']==s['maximum_output_bytes']-s['actual_allocated_bytes'];credit+=s['unused_future_allowance']
 assert credit==663662592 and inherited==68043997184
 used=K.allocated(q['project_root']);assert used<=q['maximum_output_bytes']
 required=inherited-credit+q['maximum_output_bytes']-used
 fs=os.statvfs(q['project_root']);free=fs.f_bavail*fs.f_frsize
 assert required>=58<<30 and free>=required,dict(free=free,required=required)
 mi={s.split(':',1)[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if ':' in s and len(s.split())>=2 and s.split()[1].isdigit()}
 head=mi['MemAvailable'];mx=Path('/sys/fs/cgroup/memory.max');cur=Path('/sys/fs/cgroup/memory.current')
 if mx.exists() and cur.exists() and mx.read_text().strip()!='max':head=min(head,int(mx.read_text())-int(cur.read_text()))
 assert head>=(24<<30 if starting else 2<<30)
 return dict(free_bytes=free,required_including_this_project=required,own_used_bytes=used,own_remaining_bytes=q['maximum_output_bytes']-used,
  CPU_headroom_bytes=head,closed_unused_future_settled=credit,inherited_before_settlement=inherited,all_other_forecasts_held=True,files_deleted=0)
def records(c):
 result={'correct':[dict(r,arm='correct') for r in c['prior']['positive']]}
 for arm in ARMS[1:]:result[arm]=[json.loads(s) for s in (c['out']/'images'/arm/'images.jsonl').read_text().splitlines()]
 assert all([r['query_index'] for r in result[a]]==list(range(400)) for a in ARMS)
 return result


def save_array(path,value):
 path=Path(path);assert not path.exists()
 temporary=path.with_name(path.name+".pending")
 with temporary.open("xb") as f:np.save(f,value,allow_pickle=False)
 temporary.replace(path)
