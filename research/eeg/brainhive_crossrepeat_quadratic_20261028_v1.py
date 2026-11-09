"""Classical cross-repeat quadratic statistics and an exact ridge block solve.

No claim of a new U-statistic, crossnobis estimator, or certified denoising.
NumPy computations are float64; no GPU imports or context initialization.
"""
import numpy as np


def features(left,right):
    a,b=np.asarray(left,np.float64),np.asarray(right,np.float64)
    assert a.shape==b.shape and a.ndim==3 and a.shape[1]>=2
    repeats=a.shape[1];ma,mb=a.mean(1),b.mean(1)
    naive=ma*mb
    noise=((a-ma[:,None])*(b-mb[:,None])).sum(1)/(repeats*(repeats-1))
    cross=(a.sum(1)*b.sum(1)-(a*b).sum(1))/(repeats*(repeats-1))
    assert np.isfinite(cross).all()
    scale=np.maximum(1.,np.maximum(np.abs(cross),np.abs(naive)+np.abs(noise)))
    error=np.max(np.abs(cross-(naive-noise))/scale)
    assert error<1e-12
    return naive,cross,noise,float(error)


def fit_standardize(fit,held):
    fit=np.asarray(fit,np.float64);held=np.asarray(held,np.float64)
    mean=fit.mean(0);scale=np.sqrt(np.mean(np.square(fit-mean),axis=0))
    zero=scale<1e-10;scale[zero]=1.
    return (fit-mean)/scale,(held-mean)/scale,dict(mean=mean,scale=scale,constant_columns=zero)


def joint_ridge_extension(torch,base_cholesky,base_coefficient,X,Z,Y,penalty):
    """Exact Schur complement for the same joint ridge, not residual-only fit."""
    XZ=X.T@Z
    E=torch.cholesky_solve(XZ,base_cholesky)
    schur=Z.T@Z-XZ.T@E+penalty*torch.eye(Z.shape[1],dtype=Z.dtype)
    schur=(schur+schur.T)/2
    L=torch.linalg.cholesky(schur)
    q=torch.cholesky_solve(Z.T@Y-XZ.T@base_coefficient,L)
    linear=base_coefficient-E@q
    return linear,q


def score(prediction,target,candidate_rows,labels):
    p=np.asarray(prediction,np.float64);y=np.asarray(target,np.float64)
    pn=p/np.maximum(np.linalg.norm(p,axis=1,keepdims=True),1e-30)
    yn=y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-30)
    cosine=np.einsum('nd,nkd->nk',pn,yn[candidate_rows])
    correct=cosine[np.arange(len(p)),labels]
    top=(correct==cosine.max(1))/(cosine==cosine.max(1)[:,None]).sum(1)
    afc=((correct[:,None]>cosine).sum(1)+.5*(correct[:,None]==cosine).sum(1)-.5)/9
    margin=correct-(cosine.sum(1)-correct)/9
    return np.stack((np.square(p-y).mean(1),correct,top,afc,margin),axis=-1)
