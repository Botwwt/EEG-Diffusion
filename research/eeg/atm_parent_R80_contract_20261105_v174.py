"""Exact original ATMS versus frozen positive NCE, same 200 R80 queries."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
ARMS=('original_ATMS','frozen_positive_NCE')
PAPER=('eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav')
METRICS=('target_cosine','DINO2AFC','gallery_top1','gallery_top5','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')+PAPER
IDENTITY=('query_id','subject_id','stimulus_id','concept_id','array_row','image_seed')
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(path,a):
 p=Path(path);assert not p.exists();tmp=p.with_name(p.name+'.pending')
 with tmp.open('xb') as f:np.save(f,a,allow_pickle=False)
 tmp.replace(p)
def setup(path):
 q=json.loads(Path(path).read_bytes());assert q['status']=='LOCKED_EXACT_PARENT_R80_200_MATCHED_BASELINE'
 assert q['host']=='old' and q['seed']==1729 and q['arms']==list(ARMS) and q['metrics']==list(METRICS)
 assert q['new_images']==200 and q['reused_images']==200 and q['new_updates']==0 and q['maximum_output_bytes']==192<<20
 sys.path.insert(0,str(Path(q['prior_policy']['path']).parent))
 import atm_new_mean_factorial_contract_20261105_v3 as F
 f=F.setup(q['prior_policy']['path']);K=f['K'];K.checked(q['prior_policy']);done=K.read(K.checked(q['prior_completion']))
 assert done['status']=='COMPLETE_NEW_MEAN_FACTORIAL_FULL400'
 for lp in q['closed_launches']:
  a=K.read(K.checked(lp));assert not K.process(a['PID'],a.get('start_ticks'))['live']
 for e in done['events']:
  if e['GPU']:
   j=K.read(K.checked(e['receipt']));assert j['status']=='COMPLETED' and j['exit_code']==0
   assert not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
 for pin in q['sources'].values():K.checked(pin)
 for pin in q['runtime_sources'].values():K.checked(pin)
 K.checked(q['encoder']);assert q['encoder']['sha256']=='f00082d532ec0227d00d037982113c5d273813b91e34a37d6fd4b9d5e9d0dafe'
 n=f['prior'];ref=n['reference'];rows=ref['queries']['R80'];positive=n['reference_records']['R80']['frozen_positive_NCE']
 assert len(rows)==len(positive)==200 and len({r['concept_id'] for r in rows})==200
 return dict(q=q,K=K,F=F,f=f,n=n,ref=ref,rows=rows,positive=positive,repo=n['repo'],parent=n['parent'],cv=n['cv'],physical=n['physical'],out=Path(q['output_root']))
def resources(c,starting=False):
 # Replay the already sealed reserve chain, then settle ONLY the closed factorial remainder.
 f=c['f'];q=c['q'];K=c['K'];train=f['prior']['train'];ledger=train['ledger']
 assert K.allocated(train['prior']['q']['project_root'])==ledger['prior_project_used'];K.checked(ledger['charge_snapshot']);oldcredit=0
 for s in ledger['settlements']:
  pq=K.read(K.checked(s['policy']));d=K.read(K.checked(s['completion']))
  assert d['status']==s['completion_status'] and pq['maximum_output_bytes']==s['maximum_output_bytes']
  assert K.allocated(s['project'])==s['actual_allocated_bytes']
  for pin in s['all_launches']:
   a=K.read(K.checked(pin));assert not K.process(a['PID'],a.get('start_ticks'))['live']
  for pin in s['jobs']:
   j=K.read(K.checked(pin));assert j['status']=='COMPLETED' and j['exit_code']==0
   assert not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
  assert s['unused_future_allowance']==s['maximum_output_bytes']-s['actual_allocated_bytes'];oldcredit+=s['unused_future_allowance']
 assert oldcredit==ledger['settled_unused_future_bytes'];base=ledger['prior_quote']['required_including_this_project']-oldcredit
 oldused=K.allocated(train['q']['project_root']);assert oldused<=train['q']['maximum_output_bytes']
 inherited=base+train['q']['maximum_output_bytes']-oldused;credit=0
 for s in f['q']['settlements']:
  assert K.allocated(s['project'])==s['actual_allocated_bytes']
  pq=K.read(K.checked(s['policy']));d=K.read(K.checked(s['completion']))
  assert pq['maximum_output_bytes']==s['maximum_output_bytes'] and d['status']==s['completion_status']
  assert s['unused_future_allowance']==s['maximum_output_bytes']-s['actual_allocated_bytes'];credit+=s['unused_future_allowance']
 assert credit==663662592 and inherited==68043997184
 s=q['settlement'];assert s['policy']==q['prior_policy'] and s['completion']==q['prior_completion']
 assert K.allocated(f['q']['project_root'])==s['actual_allocated_bytes']==406142976
 assert f['q']['maximum_output_bytes']-s['actual_allocated_bytes']==s['unused_future_allowance']==264945664
 used=K.allocated(q['project_root']);assert used<=q['maximum_output_bytes']
 # The prior project is CLOSED: no future allocation remains; actual files stay on disk.
 required=inherited-credit+q['maximum_output_bytes']-used
 fs=os.statvfs(Path(q['project_root']).parent);free=fs.f_bavail*fs.f_frsize
 assert required>=58<<30 and free>=required,dict(free=free,required=required)
 mi={s.split(':',1)[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if ':' in s and len(s.split())>=2 and s.split()[1].isdigit()}
 head=mi['MemAvailable'];mx=Path('/sys/fs/cgroup/memory.max');cur=Path('/sys/fs/cgroup/memory.current')
 if mx.exists() and cur.exists() and mx.read_text().strip()!='max':head=min(head,int(mx.read_text())-int(cur.read_text()))
 assert head>=(24<<30 if starting else 2<<30)
 return dict(free_bytes=free,required_including_this_project=required,own_used_bytes=used,own_remaining_bytes=q['maximum_output_bytes']-used,CPU_headroom_bytes=head,
  newly_settled_closed_unused_bytes=264945664,all_actual_files_retained=True,all_other_forecasts_held=True,floor50GiB_buffer8GiB_held=True)
def records(c):
 p=c['out']/'images'/'images.jsonl';a=[json.loads(s) for s in p.read_text().splitlines()]
 assert [v['query_index'] for v in a]==list(range(200))
 return {'original_ATMS':a,'frozen_positive_NCE':c['positive']}
def paired(a,rows):
 order=np.argsort([r['concept_id'] for r in rows]);draw=np.random.default_rng(1729).integers(0,200,(1000,200));pairs={};diffs={}
 for key in METRICS:
  d=a[ARMS[1]][key]-a[ARMS[0]][key];valid=np.isfinite(d);diffs[key]=d
  pairs[key]=dict(difference=float(d.mean()) if valid.all() else None,descriptive_concept_bootstrap95=np.quantile(d[order][draw].mean(1),[.025,.975]).tolist() if valid.all() else None,
   defined_queries=int(valid.sum()),total_queries=200)
 return pairs,diffs
