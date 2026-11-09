"""Classical cross-repeat distances; no encoder feature or novelty claim."""
import itertools
import numpy as np

REPEAT_EDGES=tuple(itertools.combinations(range(4),2))
PARTITIONS=((0,5),(1,4),(2,3))

def pair_products(trials,variance):
    x=np.asarray(trials,dtype=np.float64)
    assert x.ndim==3 and x.shape[1]==4 and x.shape[0]>=2 and np.isfinite(x).all()
    v=np.asarray(variance,dtype=np.float64);assert v.shape==(x.shape[2],) and np.isfinite(v).all() and (v>0).all()
    pairs=np.array(list(itertools.combinations(range(len(x)),2)),dtype=np.int64)
    difference=(x[pairs[:,0]]-x[pairs[:,1]])/np.sqrt(v)[None,None,:]
    result=np.stack([np.einsum('np,np->n',difference[:,a],difference[:,b])/x.shape[2] for a,b in REPEAT_EDGES])
    assert result.shape==(6,len(pairs)) and np.isfinite(result).all()
    return result

def shifted_pair_indices(stimuli=10):
    pairs=list(itertools.combinations(range(stimuli),2));index={pair:k for k,pair in enumerate(pairs)}
    return np.array([index[tuple(sorted(((a+1)%stimuli,(b+1)%stimuli)))] for a,b in pairs],dtype=np.int64)

def concept_statistics(products):
    p=np.asarray(products,dtype=np.float64);assert p.ndim==3 and p.shape[1:]==(6,45) and np.isfinite(p).all()
    p=p-p.mean(-1,keepdims=True);wrong=shifted_pair_indices()
    return np.stack([np.stack(((p[:,a]**2).sum(-1),(p[:,b]**2).sum(-1),
        (p[:,a]*p[:,b]).sum(-1),(p[:,a]*p[:,b,wrong]).sum(-1)),axis=-1) for a,b in PARTITIONS],axis=1)

def statistics_value(statistics):
    total=np.asarray(statistics,dtype=np.float64).sum(axis=0);assert total.shape==(3,4)
    denominator=np.sqrt(total[:,0]*total[:,1])
    if not (denominator>0).all():return None
    correct=total[:,2]/denominator;wrong=total[:,3]/denominator
    assert (np.abs(correct)<=1+1e-10).all() and (np.abs(wrong)<=1+1e-10).all()
    return np.array((correct.mean(),wrong.mean(),(correct-wrong).mean())),correct,wrong

def fixed_readout(products):
    stats=concept_statistics(products);point=statistics_value(stats)
    if point is None:return dict(status='UNKNOWN_ZERO_DISTANCE_VARIANCE',all_expected_concepts=len(stats)),stats,None
    rng=np.random.default_rng(1729);bootstrap=[]
    for _ in range(1000):
        value=statistics_value(stats[rng.integers(0,len(stats),size=len(stats))])
        if value is None:return dict(status='UNKNOWN_BOOTSTRAP_ZERO_VARIANCE',all_expected_concepts=len(stats)),stats,None
        bootstrap.append(value[0])
    boot=np.asarray(bootstrap);ci=np.quantile(boot,[.025,.975],axis=0).T;estimate=point[0]
    support=bool(estimate[0]>=.1 and ci[0,0]>0 and ci[2,0]>0)
    return dict(status='COMPLETE_FIXED_FULL_T_REPEAT_GEOMETRY_DIAGNOSTIC',concepts=len(stats),pairs_per_concept=45,
        main_correct_correlation=float(estimate[0]),same_concept_shift_control=float(estimate[1]),correct_minus_control=float(estimate[2]),
        descriptive_concept_bootstrap95=ci.tolist(),all3_correct_correlations=point[1].tolist(),all3_control_correlations=point[2].tolist(),
        candidate_measurement_gate=support,decision='SEPARATE_TARGET_DESIGN_MAY_BE_CONSIDERED_NOT_ADMITTED' if support else 'STOP_THIS_TARGET_GEOMETRY_PROPOSAL',
        numerical_threshold_r=.1,independent_repetition_noise_verified=False,true_neural_distance_identification='UNKNOWN',
        new_decoder_or_GPU_or_image_admitted=False),stats,boot
