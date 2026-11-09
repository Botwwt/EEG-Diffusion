"""Four frozen factor cells; complete R4/R80 images; corrected closed reserve ledger."""
import json,os,sys
from pathlib import Path
import numpy as np
ARMS=("frozen_positive_NCE","stochastic_P","N_trunk_P_head","P_trunk_N_head")
CONTRASTS=tuple((ARMS[j],ARMS[i]) for i in range(4) for j in range(i+1,4))
PAPER=('eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav')
R4_METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
R80_METRICS=('target_cosine','DINO2AFC','gallery_top1','gallery_top5','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')+PAPER
def setup(path):
 q=json.loads(Path(path).read_bytes());assert q["status"]=="LOCKED_FROZEN_BLOCK_FACTORIAL_FULL_R4_R80_IMAGES"
 assert q["host"]=="old" and q["seed"]==1729 and q["arms"]==list(ARMS)
 assert q["maximum_output_bytes"]==640<<20 and q["new_images"]==1200 and q["reused_images"]==1200
 assert q["contrasts"]==[list(v) for v in CONTRASTS]
 assert q["metrics_R4"]==list(R4_METRICS) and q["metrics_R80"]==list(R80_METRICS)
 sys.path.insert(0,str(Path(q["prior_policy"]["path"]).parent))
 import atm_deployment_image_contract_20261105_v179 as O
 old=O.setup(q["prior_policy"]["path"]);K=old["K"];K.checked(q["prior_policy"])
 for pin in q["sources"].values():K.checked(pin)
 for pin in q["runtime_sources"].values():K.checked(pin)
 for pin in q["closed_launches"]:
  a=K.read(K.checked(pin));assert not K.process(a["PID"],a["start_ticks"])["live"]
 assert K.read(K.checked(q["diagnostic_completion"]))["status"]=="COMPLETE_FROZEN_EEG_BLOCK_FACTORIAL"
 assert K.read(K.checked(q["diagnostic_audit"]))["CPU_forward_queries"]==2528
 assert K.read(K.checked(q["prior_completion"]))["status"]=="COMPLETE_DEPLOYMENT_SUPERVISION_TRAINING_IMAGE_EVALUATION"
 bank=np.load(K.checked(q["diagnostic_conditions"]));assert bank.shape==(600,4,1024)
 values=bank[:,[0,3,1,2]].copy()
 er=float(abs(values[:,:2]-old["values"][:,[0,2]]).max());assert er<5e-5
 values[:,:2]=old["values"][:,[0,2]]
 assert np.isfinite(values).all()
 refs={}
 for domain in ("R4","R80"):
  rr=O.records(old,domain);refs[domain]={a:rr[a] for a in ARMS[:2]}
 # Check this allowance was not already credited in any inherited active ledger.
 seen=set();credited=set()
 def scan(v):
  if not isinstance(v,(dict,list,tuple)) or id(v) in seen:return
  seen.add(id(v))
  if isinstance(v,dict):
   ss=list(v.get("settlements",[]))
   if isinstance(v.get("settlement"),dict):ss.append(v["settlement"])
   for s in ss:
    if isinstance(s,dict):
     if "project" in s:credited.add(s["project"])
     credited.update(s.get("roots",[]))
   for vv in v.values():scan(vv)
  else:
   for vv in v:scan(vv)
 scan(old)
 roots=[]
 for s in q['settlements']:
  for p in s['roots']:
   assert not any(p==r or p.startswith(r+'/') or r.startswith(p+'/') for r in roots)
   assert not any(p==r or r.startswith(p+'/') for r in credited)
   roots.append(p)
 probe=K.read(K.checked(q['reserve_probe']))
 assert probe['groups']==q['settlements'] and probe['duplicate_credit_correction_bytes']==q['duplicate_credit_correction_bytes']==663662592
 result=dict(old);result.update(q=q,K=K,O=O,reference=old,reference_records=refs,
  values=values,positive=refs["R4"][ARMS[0]],out=Path(q["output_root"]),baseline_reexport_max_difference=er)
 return result
def resources(c,starting=False):
 q,K=c["q"],c["K"]
 # The old quote predates these settlements. Reapplying its free-space assertion
 # would charge released forecasts again after new images consume real space.
 # Keep its SHA-bound conservative amount, verify its frozen project/credits,
 # then perform ONE current assertion on the reconciled complete ledger below.
 probe=K.read(K.checked(q["reserve_probe"]));r=probe['inherited_quote']
 prior=c['reference']['train'];ledger=prior['ledger']
 K.checked(prior['q']['reserve_ledger']);K.checked(ledger['source_probe'])
 assert K.allocated(prior['q']['project_root'])==r['own_used_bytes']
 assert r['required_including_this_project']==ledger['prior_quote']['required_including_this_project']-ledger['settled_unused_future_bytes']+ledger['profile_wrapper_reserve_bytes']+prior['q']['maximum_output_bytes']-r['own_used_bytes']
 for old in ledger['settlements']:
  assert K.allocated(old['project'])==old['actual_allocated_bytes']
  for pin in old['all_launches']:
   la=K.read(K.checked(pin));assert not K.process(la['PID'],la.get('start_ticks'))['live']
  assert K.read(K.checked(old['completion']))['status']==old['completion_status']
 credit=0
 for s in q['settlements']:
  actual=sum(K.allocated(p) for p in s['roots'])
  assert actual==s['actual_allocated_bytes']
  assert s['unused_future_allowance']==s['prior_charged_remaining']==s['cap']-actual
  for pin in s['policies']+s['source_policy_pins']+s['launch_pins']:K.checked(pin)
  for d in s['completion_pins']:assert K.read(K.checked(d['pin']))['status']==d['status']
  credit+=s['unused_future_allowance']
 assert credit==1334829056
 used=K.allocated(q["project_root"]) if Path(q["project_root"]).exists() else 0
 assert used<=q["maximum_output_bytes"]
 required=r["required_including_this_project"]+q['duplicate_credit_correction_bytes']-credit+q["maximum_output_bytes"]-used
 assert required>=58<<30
 fs=os.statvfs(Path(q["project_root"]).parent);free=fs.f_bavail*fs.f_frsize;assert free>=required,(free,required)
 mi={s.split(':',1)[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if ':' in s and len(s.split())>=2 and s.split()[1].isdigit()}
 head=mi['MemAvailable'];mx=Path('/sys/fs/cgroup/memory.max');cu=Path('/sys/fs/cgroup/memory.current')
 if mx.exists() and cu.exists() and mx.read_text().strip()!='max':head=min(head,int(mx.read_text())-int(cu.read_text()))
 assert head>=(24<<30 if starting else 2<<30)
 return dict(required_including_this_project=required,free_bytes=free,own_used_bytes=used,
 own_remaining_bytes=q["maximum_output_bytes"]-used,CPU_headroom_bytes=head,
 closed_nine_group_credit_once=credit,duplicate_credit_correction_bytes=q['duplicate_credit_correction_bytes'],inherited_quote=r,
 no_deleted_files=True,all_other_forecasts_50GiB_floor_8GiB_buffer_held=True)
def records(c,domain):
 n=400 if domain=="R4" else 200;result={}
 for a in ARMS:
  result[a]=c["reference_records"][domain][a] if a in ARMS[:2] else [json.loads(v) for v in (c["out"]/"images"/domain/a/"images.jsonl").read_text().splitlines()]
  assert [r["query_index"] for r in result[a]]==list(range(n))
 return result
def paired(arrays,queries,domain):
 concepts=sorted({r["concept_id"] for r in queries});groups=[[i for i,r in enumerate(queries) if r["concept_id"]==g] for g in concepts]
 assert len(groups)==(40 if domain=="R4" else 200)
 draw=np.random.default_rng(1729).integers(0,len(groups),(1000,len(groups)));result={};diffs={}
 for a,b in CONTRASTS:
  name=a+"-"+b;result[name]={}
  for key in R4_METRICS if domain=="R4" else R80_METRICS:
   delta=arrays[a][key]-arrays[b][key];valid=np.isfinite(delta);gm=np.array([delta[g].mean() for g in groups])
   result[name][key]=dict(difference=float(delta.mean()) if valid.all() else None,
    descriptive_concept_bootstrap95=np.quantile(gm[draw].mean(1),[.025,.975]).tolist() if valid.all() else None,
    defined_queries=int(valid.sum()),total_queries=len(queries),concepts=len(groups))
   diffs[name+"__"+key]=delta
 return result,diffs
