"""Independent direct rank definitions and weighted bootstrap on all8200."""
import argparse,datetime,hashlib,json,os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
from scipy.special import erf
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);args=p.parse_args();root=args.directory
read=lambda p:json.loads(Path(p).read_bytes())
manifest=read(root/'received_manifest.json')['manifest']
for name,item in manifest.items():
    b=(root/name).read_bytes();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
done=read(root/'actual/completion.json');assert done['status']=='COMPLETE' and done['Dfit_queries']==8200
assert done['new_training_updates']==done['new_images']==0 and not done['CUDA_initialized']
s=read(root/'protocol/protocol.json');summary=read(root/'actual/summary.json')
bank=np.load(root/'actual/same_concept_score_bank.npz',allow_pickle=False)
metrics=np.load(root/'actual/all_query_metrics.npz',allow_pickle=False)
identity=np.load(root/'actual/identity_population.npz',allow_pickle=False)
sid,cid=identity['subject_ids'],identity['concept_ids'];slot=bank['true_slots'];n=len(sid)
assert n==8200 and len(set(sid.tolist()))==10 and len(set(cid.tolist()))==82
assert not set(identity['T_array_rows']) & set(identity['Dfit_array_rows'])
assert len(identity['T_array_rows'])==13230 and len(identity['Dfit_array_rows'])==820
assert abs(np.linalg.norm(identity['population'])-1)<1e-14
max_query_error=0.;checks=0
for method in s['methods']:
    for label,bank_key in [('same_concept_F',method+'_same_scores'),('same_concept_perp_F',method+'_perp_same_scores')]:
        if bank_key not in bank.files:
            assert method=='T_population_null' and 'perp' in label
            assert method+'__'+label+'_2AFC' not in metrics.files
            continue
        scores=bank[bank_key];assert scores.shape==(8200,10)
        values={k:np.empty(n) for k in ('Top1_tie_fraction','2AFC','mean_margin','best_wrong_margin')}
        for i in range(n):
            truth=float(scores[i,int(slot[i])]);others=np.delete(scores[i],int(slot[i]))
            wins=sum(1. if truth>x+1e-12 else .5 if abs(truth-x)<=1e-12 else 0. for x in others)
            ties=sum(abs(truth-x)<=1e-12 for x in others)
            values['Top1_tie_fraction'][i]=0. if any(x>truth+1e-12 for x in others) else 1/(ties+1)
            values['2AFC'][i]=wins/9
            values['mean_margin'][i]=truth-float(np.mean(others))
            values['best_wrong_margin'][i]=truth-max(others)
        for key,x in values.items():
            err=float(np.max(np.abs(x-metrics[method+'__'+label+'_'+key])));max_query_error=max(max_query_error,err);assert err<1e-13;checks+=n
        if label=='same_concept_F':
            err=float(np.max(np.abs(scores[np.arange(n),slot]-metrics[method+'__F_cosine'])));assert err<1e-13;max_query_error=max(max_query_error,err)
    total=metrics[method+'__F_common_projection']+metrics[method+'__F_residual_projection']
    assert np.max(np.abs(total-metrics[method+'__F_cosine']))<2e-14
    cc=summary['concentrations'][method];assert abs(cc['unit_prediction_population_energy']+cc['unit_prediction_variance']-1)<1e-12
assert abs(metrics['T_population_null__same_concept_F_Top1_tie_fraction'].mean()-.1)<1e-12
assert abs(metrics['T_population_null__same_concept_F_2AFC'].mean()-.5)<1e-12
rng=np.random.default_rng(1729);weights=[]
for _ in range(1000):
    sw=np.bincount(rng.integers(0,10,10),minlength=10)
    cw=np.bincount(rng.integers(0,82,82),minlength=82)
    weights.append(np.multiply.outer(sw,cw).reshape(-1)/820)
weights=np.asarray(weights);grids={}
for name in metrics.files:
    values=metrics[name];assert values.shape==(8200,) and np.isfinite(values).all()
    grid=np.empty((10,82))
    for subject in range(10):
        for concept in range(82):
            ids=(sid==subject)&(cid==concept);assert ids.sum()==10;grid[subject,concept]=values[ids].sum()/10
    grids[name]=grid
max_statistics_error=0.;statistics_checked=0
def check_stat(x,grid,record):
    global max_statistics_error,statistics_checked
    boot=weights@grid.reshape(-1)
    test=[(float(x.mean()),record['mean'])]
    test.extend(zip(np.quantile(boot,[.025,.975]).tolist(),record['descriptive_CI95']))
    test.extend(zip(grid.mean(axis=1).tolist(),record['subject_means']))
    err=max(abs(a-b) for a,b in test);assert err<2e-13
    max_statistics_error=max(max_statistics_error,err);statistics_checked+=1
for method,fields in summary['methods'].items():
    for key,record in fields.items():
        name=method+'__'+key;check_stat(metrics[name],grids[name],record)
for first,second in s['fixed_comparisons']:
    for key,record in summary['comparisons'][first+'_minus_'+second].items():
        left,right=first+'__'+key,second+'__'+key
        check_stat(metrics[left]-metrics[right],grids[left]-grids[right],record)
w=np.load(root/'actual/head_witnesses64.npz',allow_pickle=False);witness_error={}
for method in ('ordinary','bounded_F','Euclidean_F'):
    get=lambda key:w[method+'__'+key]
    c,a,b=(w[k].astype(np.float64) for k in ('condition','weight','bias'));u=(c-b)/a
    h=np.dot(u,get('trunk.0.weight').T)+get('trunk.0.bias');h=.5*h*(1+erf(h/np.sqrt(2)))
    d=np.dot(h,get('direction.weight').T)+get('direction.bias')
    if method=='Euclidean_F':x=c+a*.25*d
    else:
        d=d-d.mean(axis=1)[:,None];r=np.sum(u*u,axis=1)[:,None];d=d-u*np.sum(d*u,axis=1)[:,None]/r
        delta=.25*np.sqrt(r)*d/np.sqrt(r+np.sum(d*d,axis=1)[:,None]);scale=np.sqrt(r/np.sum((u+delta)**2,axis=1)[:,None]);x=c+a*((scale-1)*u+scale*delta)
    err=float(np.abs(x-w[method+'_independent64']).max());assert err<1e-10;witness_error[method]=err
result=dict(status='PASS_LOCAL_ALL8200_RANK_AND_STATISTICS_AUDIT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    SHA_files=len(manifest),queries=8200,methods=6,same_concept_raw_scores=sum(bank[k].size for k in bank.files if k.endswith('_scores')),
    independently_recomputed_rank_metric_cells=checks,rank_metric_max_error=max_query_error,
    all_method_and_predeclared_contrast_statistic_records=statistics_checked,statistics_max_error=max_statistics_error,
    independent_head_equations_witness_queries_per_model=64,witness_max_error=witness_error,
    original_fullT_FP64_guard_passed=False,
    audit_limits=['All8200sameconceptscoresandallquerystatisticsindependentlyreduced;full8200headoutputsnottransferred',
                  'Head64equationwitnessesreplayedlocally;T13230populationindependentchunkreplayonserveronly',
                  'All820globalretrievalscoresnottransferred;thesequerystatisticsarebootstrapcheckedonly',
                  'Dfitpreviouslyusedforcalibration;noimageefficacyorindependentconfirmation'])
(root/'independent_local_audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
