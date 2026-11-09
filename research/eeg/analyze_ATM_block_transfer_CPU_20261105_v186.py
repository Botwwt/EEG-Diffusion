"""Terminal-only paired factorial interactions; descriptive, no fitting or selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import datetime,hashlib,json,resource,sys,time
from pathlib import Path
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
plan=json.loads(Path(sys.argv[1]).read_bytes());root=Path(plan['project_root']);out=Path(plan['output_root'])
assert plan['host']=='old' and plan['seed']==1729
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(p):assert ref(p['path'])==p;return Path(p['path'])
def live(a):
 p=Path('/proc',str(a['PID']),'stat')
 if not p.exists():return False
 s=p.read_text().rsplit(')',1)[1].split();return s[0] not in ('Z','X') and int(s[19])==a['start_ticks']
assert not live(read(root/'code_recovery_v3/actual_launch.json'))
assert not live(read(root/'terminal_review/actual_launch.json'))
proof=read(root/'terminal_review/completion.json')
assert proof['status']=='PASS_ALL2400_IDENTITIES_PIXELS_NOISES_11200_RANK_VALUES_7200_LOWLEVEL_138_CIS'
checked(proof['image_completion'])
assert ref(__file__)==plan['source']
checked(plan['image_policy'])
sys.path.insert(0,str(Path(plan['image_policy']['path']).parent))
import atm_block_image_contract_20261105_v185 as C
c=C.setup(plan['image_policy']['path']);K=c['K'];before=C.resources(c)
import numpy as np
from scipy.stats import spearmanr
diag=np.load(checked(plan['condition_metrics']));queries=read(checked(plan['queries']))
images=np.load(root/'image_output/metrics.npz');summary=read(root/'image_output/readout.json');checked(summary['arrays'])
ordered=read(root/'image_output/ordered_images.json')
maparm={'NN':'frozen_positive_NCE','NP':'N_trunk_P_head','PN':'P_trunk_N_head','PP':'stochastic_P'}
condarm={'NN':'N_trunk_N_head','NP':'N_trunk_P_head','PN':'P_trunk_N_head','PP':'P_trunk_P_head'}
effects={};result={};identity_count=0;identity_fields=('query_id','subject_id','stimulus_id','concept_id','array_row','image_seed')
def factors(v):
 hN=v['PN']-v['NN'];hP=v['PP']-v['NP'];gN=v['NP']-v['NN'];gP=v['PP']-v['PN'];j=v['PP']-v['PN']-v['NP']+v['NN']
 np.testing.assert_allclose(hP-hN,j,atol=1e-12,rtol=0);np.testing.assert_allclose(gP-gN,j,atol=1e-12,rtol=0)
 return dict(trunk_at_Nhead=hN,trunk_at_Phead=hP,head_at_Ntrunk=gN,head_at_Ptrunk=gP,interaction=j)
def paired(x,groups,draw):
 if not np.isfinite(x).all():return dict(mean=None,CI95=None,defined=int(np.isfinite(x).sum()),total=len(x))
 gm=np.array([x[g].mean() for g in groups]);ci=np.quantile(gm[draw].mean(1),[.025,.975])
 return dict(mean=float(x.mean()),CI95=ci.tolist(),defined=len(x),total=len(x),positive=int((x>0).sum()),negative=int((x<0).sum()),zero=int((x==0).sum()))
def correlation(x,y):
 if not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(x)==0 or np.ptp(y)==0:return None
 return dict(Pearson=float(np.corrcoef(x,y)[0,1]),Spearman=float(spearmanr(x,y).statistic))
for domain in ('R4','R80'):
 n=400 if domain=='R4' else 200;q=queries[domain];assert len(q)==n
 groups=[[i for i,r in enumerate(q) if r['concept_id']==g] for g in sorted({r['concept_id'] for r in q})]
 assert len(groups)==(40 if domain=='R4' else 200)
 draw=np.random.default_rng(1729).integers(0,len(groups),(1000,len(groups)))
 idx={(r['arm'],r['query_index']):r for r in ordered if r['domain']==domain}
 for a in maparm.values():
  for i in range(n):
   assert all(idx[a,i][k]==q[i][k] for k in identity_fields);identity_count+=1
 metrics=C.R4_METRICS if domain=='R4' else C.R80_METRICS
 d=dict(images={},conditions={},cross_metric_associations={})
 image_effects={};condition_effects={}
 for metric in metrics:
  f=factors({a:images[domain+'__'+v+'__'+metric] for a,v in maparm.items()});image_effects[metric]=f
  d['images'][metric]={a:paired(x,groups,draw) for a,x in f.items()}
  for a,x in f.items():effects[domain+'__image__'+metric+'__'+a]=x
 for metric in plan['condition_metrics_to_compare']:
  f=factors({a:diag[domain+'__'+v+'__'+metric] for a,v in condarm.items()});condition_effects[metric]=f
  d['conditions'][metric]={a:paired(x,groups,draw) for a,x in f.items()}
  for a,x in f.items():effects[domain+'__condition__'+metric+'__'+a]=x
 for effect in ('trunk_at_Nhead','trunk_at_Phead','head_at_Ntrunk','head_at_Ptrunk','interaction'):
  d['cross_metric_associations'][effect]={}
  for cm in plan['condition_metrics_to_compare']:
   d['cross_metric_associations'][effect][cm]={im:correlation(condition_effects[cm][effect],image_effects[im][effect]) for im in plan['image_metrics_to_compare']}
 # Independently evaluate interactions using a scalar four-cell loop.
 for metric in metrics:
  vv={a:images[domain+'__'+v+'__'+metric] for a,v in maparm.items()}
  loop=np.array([float(vv['PP'][i])-float(vv['PN'][i])-float(vv['NP'][i])+float(vv['NN'][i]) for i in range(n)])
  np.testing.assert_allclose(loop,image_effects[metric]['interaction'],atol=1e-12,rtol=0,equal_nan=True)
 result[domain]=d
assert identity_count==2400 and time.monotonic()-started<100
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<2<<30
np.savez_compressed(out/'effects.npz',**effects)
answer=dict(status='COMPLETE_ALL600_QUERY_FOUR_CELL_IMAGE_AND_CONDITION_INTERACTIONS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 domains=result,identity_count=identity_count,plan=ref(sys.argv[1]),image_review=ref(root/'terminal_review/completion.json'),
 effects=ref(out/'effects.npz'),new_fits=0,new_GPU=0,new_images=0,
 inference='Descriptive exposed sub01 factorial; interactions depend on metric scale and donor coadaptation; correlations are not causal mediation or information loss. No fitted selector, chosen checkpoint, threshold, or new candidate.',
 statistics='1000 concept resamples per domain seed1729; unchanged raw metric signs; 23 image interactions secondary/exploratory; no multiplicity correction.',
 seconds=time.monotonic()-started,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0)
assert K.allocated(out)<plan['maximum_output_bytes'];after=C.resources(c)
answer['resources_before']=before;answer['resources_after']=after
K.write(out/'completion.json',answer)
print(json.dumps(dict(status=answer['status'],seconds=answer['seconds'],identities=identity_count)))
