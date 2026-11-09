"""Algebra, correlated-noise counterexample, and independent joint-ridge check."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import brainhive_crossrepeat_quadratic_20261028_v1 as Q


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();start=time.monotonic();rng=np.random.default_rng(1729);checks=[]
    for R in (2,3,4,80):
        x,y=rng.normal(size=(7,R,5)),rng.normal(size=(7,R,5))
        naive,cross,noise,error=Q.features(x,y)
        direct=sum(x[:,r]*y[:,s] for r in range(R) for s in range(R) if r!=s)/(R*(R-1))
        assert np.allclose(cross,direct,atol=1e-13,rtol=1e-12)
        permutation=rng.permutation(R)
        assert np.allclose(Q.features(x[:,permutation],y[:,permutation])[1],cross,atol=1e-13)
    checks += ['All ordered cross pairs equal compressed formula for R2/R3/R4/R80','Common repeat permutation invariant','U=mean product-sample cross covariance/R']
    # An exhaustive symmetric noise distribution verifies the expectation
    # without relying on a Monte Carlo confidence interval.
    from itertools import product
    epsilon=np.array(list(product((-1.,1.),repeat=4)))[:,:,None]
    signal=.7
    naive,cross,noise,_=Q.features(signal+epsilon,signal+epsilon)
    assert abs(cross.mean()-signal**2)<1e-14 and abs(naive.mean()-signal**2-.25)<1e-14
    checks.append('Independent zero-mean finite noise: exact unbiased expectation, naive bias1/R')
    common=np.repeat(np.array([-1.,1.])[:,None,None],4,axis=1)
    correlated=Q.features(signal+common,signal+common)[1]
    assert abs(correlated.mean()-signal**2-1)<1e-14
    assert np.any(cross<0)
    # Unbiasedness does not guarantee smaller squared-error risk.
    zero=Q.features(epsilon,epsilon)
    assert np.mean(zero[1]**2)>np.mean(zero[0]**2)
    checks += ['Shared repeat noise remains: unbiased neural moment not guaranteed','Finite U can be negative; do not PSD-clip','Unbiased U can have larger MSE than naive product']
    fit=rng.normal(size=(15,5));held=rng.normal(size=(7,5))
    first=Q.fit_standardize(fit,held)
    second=Q.fit_standardize(fit,held*100+33)
    assert np.array_equal(first[0],second[0]) and np.array_equal(first[2]['mean'],second[2]['mean'])
    checks.append('FIT-only normalizer independent of held values')
    import torch
    torch.set_num_threads(2)
    X=torch.from_numpy(rng.normal(size=(19,8)));Z=torch.from_numpy(rng.normal(size=(19,3)));Y=torch.from_numpy(rng.normal(size=(19,4)));penalty=.19
    chol=torch.linalg.cholesky(X.T@X+penalty*torch.eye(8,dtype=torch.float64))
    base=torch.cholesky_solve(X.T@Y,chol)
    linear,quad=Q.joint_ridge_extension(torch,chol,base,X,Z,Y,penalty)
    joint=np.concatenate((X.numpy(),Z.numpy()),axis=1)
    oracle=np.linalg.solve(joint.T@joint+penalty*np.eye(11),joint.T@Y.numpy())
    error=np.max(np.abs(np.concatenate((linear.numpy(),quad.numpy()))-oracle));assert error<1e-12
    normal=np.max(np.abs((joint.T@joint+penalty*np.eye(11))@oracle-joint.T@Y.numpy()));assert normal<1e-12
    checks.append('Schur joint ridge equals independent NumPy full normal-equation solution')
    target=rng.normal(size=(10,4));candidates=np.tile(np.arange(10),(10,1));labels=np.arange(10)
    metric=Q.score(target,target,candidates,labels)
    assert np.all(metric[:,0]==0) and np.all(metric[:,2:4]==1)
    tied=Q.score(np.zeros_like(target),target,candidates,labels)
    assert np.allclose(tied[:,2],.1) and np.allclose(tied[:,3],.5)
    checks.append('True-target and zero-prediction fractional-tie score checks')
    assert not torch.cuda.is_initialized()
    value=dict(status='CROSS_REPEAT_QUADRATIC_MATH_AND_CPU_SOLVER_CHECKS_PASSED',checks=checks,checks_count=len(checks),
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),module_sha256=hashlib.sha256(Path(Q.__file__).read_bytes()).hexdigest(),
               independent_solver_max_absolute_error=float(error),normal_equation_max_absolute_error=float(normal),
               independent_noise_naive_MSE=float(np.mean(zero[0]**2)),independent_noise_U_MSE=float(np.mean(zero[1]**2)),
               correlated_noise_U_bias=1.,seconds=time.monotonic()-start,actual_EEG_read=False,GPU_initialized=False,new_models=0,new_images=0)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value))


if __name__=='__main__':main()
