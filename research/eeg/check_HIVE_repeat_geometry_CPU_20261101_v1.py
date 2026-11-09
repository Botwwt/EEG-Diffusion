"""Unbiasedness assumptions, shared-noise counterexample and statistic audits."""
import itertools,json
import numpy as np
import brainhive_repeat_geometry_CPU_20261101_v1 as M

def run():
    checks=[];values=[];zero_signal_values=[]
    for signs in itertools.product((-1.,1.),repeat=8):
        x=np.array(signs).reshape(2,4,1)+np.array([0.,2.])[:,None,None]
        values.append(M.pair_products(x,np.ones(1))[:,0])
        zero_signal_values.append(M.pair_products(np.array(signs).reshape(2,4,1),np.ones(1))[:,0])
    assert np.allclose(np.mean(values,axis=0),4.,atol=1e-14)
    assert np.min(zero_signal_values)<0 and np.allclose(np.mean(zero_signal_values,axis=0),0.)
    checks.append('exact256_independent_zero_mean_noise_crossdist_unbiased_and_signed')
    shared=[]
    for a,b in itertools.product((-1.,1.),repeat=2):
        x=np.broadcast_to(np.array([a,2+b])[:,None,None],(2,4,1)).copy();shared.append(M.pair_products(x,np.ones(1))[:,0])
    assert np.allclose(np.mean(shared,axis=0),6.)
    checks.append('shared_repeat_noise_bias_counterexample_prevents_unconditional_neural_distance_claim')
    rng=np.random.default_rng(1729);x=rng.normal(size=(10,4,7));variance=rng.uniform(.3,2.,size=7)
    actual=M.pair_products(x,variance);brute=[]
    for a,b in M.REPEAT_EDGES:
        row=[]
        for i,j in itertools.combinations(range(10),2):row.append(sum((x[i,a,k]-x[j,a,k])*(x[i,b,k]-x[j,b,k])/variance[k] for k in range(7))/7)
        brute.append(row)
    error=float(np.max(np.abs(actual-brute)));assert error<1e-14
    checks.append('explicit4250_analogue_coordinate_sum_matches_vectorized_distances')
    products=rng.normal(size=(7,6,45));stats=M.concept_statistics(products);v=M.statistics_value(stats)
    centered=products-products.mean(-1,keepdims=True);wrong=M.shifted_pair_indices();direct=[]
    for a,b in M.PARTITIONS:
        direct.append((np.corrcoef(centered[:,a].ravel(),centered[:,b].ravel())[0,1],
                       np.corrcoef(centered[:,a].ravel(),centered[:,b,wrong].ravel())[0,1]))
    assert np.max(np.abs(v[1]-np.array(direct)[:,0]))<1e-14 and np.max(np.abs(v[2]-np.array(direct)[:,1]))<1e-14
    for indices in (np.array([0,0,3,5,1,2,2]),np.array([6,5,4,3,2,1,0])):
        new=M.statistics_value(stats[indices]);t=centered[indices]
        expected=np.mean([np.corrcoef(t[:,a].ravel(),t[:,b].ravel())[0,1] for a,b in M.PARTITIONS])
        assert abs(new[0][0]-expected)<1e-14
    checks.append('independent_corrcoef_and_duplicate_concept_resampling_match_additive_statistics')
    assert M.fixed_readout(np.zeros((7,6,45)))[0]['status']=='UNKNOWN_ZERO_DISTANCE_VARIANCE'
    checks.append('zero_variance_remains_UNKNOWN_full_concept_count_retained')
    r,unused,b=M.fixed_readout(products);r2,unused,b2=M.fixed_readout(products)
    assert r==r2 and np.array_equal(b,b2) and b.shape==(1000,3)
    checks.append('all3_partitions_equal_weight_and_shared1000_fixed_seed_bootstraps_repeat_exactly')
    return dict(status='PASS_REPEAT_GEOMETRY_CLASSICAL_MATH_AND_STATISTIC_CHECKS',checks=checks,explicit_sum_maxerror=error,
        real_EEG_results_read=False,new_GPU_jobs=0,novelty_claimed=False)
if __name__=='__main__':print(json.dumps(run()))
