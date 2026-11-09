import argparse,datetime,hashlib,json,os
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();E=a.directory
read=lambda p:json.loads(Path(p).read_bytes())
for name,item in read(E/'received_manifest.json')['manifest'].items():
    b=(E/name).read_bytes();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
s=read(E/'summary.json');v=np.load(E/'all8200_feature_diagnostics.npz',allow_pickle=False);scores=np.load(E/'same_concept_scores.npz',allow_pickle=False);sid,cid,kappa,slot=(v[k] for k in ('sid','cid','kappa','slots'));n=len(sid);assert n==8200
maximum=0.;cells=0;grids={}
for name in s['means']:
    x=scores[name];assert x.shape==(8200,10);truth=x[np.arange(n),slot];CE=logsumexp(kappa[:,None]*x,axis=1)-kappa*truth
    independent=dict(F_cosine=truth,same_CE_unscaled=CE,same_CE_scaled=.1*CE/kappa)
    for key in ('same_Top1','same_2AFC','mean_margin'):independent[key]=np.empty(n)
    for i in range(n):
        others=np.delete(x[i],slot[i]);t=truth[i];ties=np.count_nonzero(np.abs(others-t)<=1e-12)
        independent['same_Top1'][i]=0 if np.any(others>t+1e-12) else 1/(ties+1)
        independent['same_2AFC'][i]=(np.count_nonzero(others<t-1e-12)+.5*ties)/9
        independent['mean_margin'][i]=t-others.mean()
    for key,values in independent.items():
        err=float(np.abs(values-v[name+'::'+key]).max());assert err<1e-12;maximum=max(maximum,err);cells+=n
    grids[name]={key:np.asarray([[v[name+'::'+key][(sid==i)&(cid==j)].mean() for j in range(82)] for i in range(10)]) for key in s['means'][name]}
rng=np.random.default_rng(1729);weights=[]
for _ in range(1000):weights.append(np.multiply.outer(np.bincount(rng.integers(0,10,10),minlength=10),np.bincount(rng.integers(0,82,82),minlength=82)).ravel()/820)
weights=np.asarray(weights);statistics=0
def check(grid,r):
    global maximum,statistics
    ci=np.quantile(weights@grid.ravel(),[.025,.975]);err=max(abs(grid.mean()-r['mean']),float(np.abs(ci-r['descriptive_CI95']).max()),float(np.abs(grid.mean(axis=1)-r['subject_means']).max()));assert err<1e-12;maximum=max(maximum,err);statistics+=1
for name,fields in s['means'].items():
    for key,r in fields.items():check(grids[name][key],r)
for name,fields in s['comparisons'].items():
    left,right=name.split('_minus_',1)
    for key,r in fields.items():check(grids[left][key]-grids[right][key],r)
r=dict(status='PASS_LOCAL_ALL11_DFIT_FEATURE_RANKS_CE_AND_BOOTSTRAP',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),queries=8200,methods=11,
    score_cells=sum(scores[k].size for k in scores.files),independently_recomputed_metric_cells=cells,statistic_records=statistics,maximum_error=maximum,
    population_projections_and_head_outputs_not_independently_replayed=True,AP_only_saved_aggregate_curves=True,not_new_image_efficacy=True,not_fresh_test=True)
(E/'independent_local_audit.json').write_bytes((json.dumps(r,indent=2)+'\n').encode());print(json.dumps(r))
