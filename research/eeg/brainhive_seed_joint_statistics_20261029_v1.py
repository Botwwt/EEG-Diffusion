"""Descriptive crossed training-seed/subject/concept bootstrap, all3 seeds."""
import numpy as np
def joint_statistics(difference,rounds=1000,seed=1729):
    d=np.asarray(difference,dtype=np.float64)
    assert d.shape==(3,10,200) and np.isfinite(d).all(),'Retain full6000 denominator; no undefined-query dropping'
    rng=np.random.default_rng(seed);samples=[];weights=[]
    for _ in range(rounds):
        k=rng.integers(3,size=3);s=rng.integers(10,size=10);c=rng.integers(200,size=200)
        samples.append(float(d[np.ix_(k,s,c)].mean()));weights.append((np.bincount(k,minlength=3)/3,np.bincount(s,minlength=10)/10,np.bincount(c,minlength=200)/200))
    wk,ws,wc=[np.stack([w[i] for w in weights]) for i in range(3)];second=np.einsum('rk,rs,rc,ksc->r',wk,ws,wc,d,optimize=True);err=float(np.max(abs(second-np.array(samples))));assert err<1e-12
    per_seed=d.mean((1,2));return dict(mean_difference=float(d.mean()),descriptive_seed_subject_concept_bootstrap95=np.quantile(samples,[.025,.975]).tolist(),per_seed_mean_differences=per_seed.tolist(),training_run_mean_range=[float(per_seed.min()),float(per_seed.max())],positive_training_runs=int((per_seed>0).sum()),training_runs=3,queries_per_run=2000,full_query_cells=6000,bootstrap_rounds=rounds,bootstrap_seed=seed,independent_weighted_check_max_error=err,population_training_seed_confidence_not_established=True)
