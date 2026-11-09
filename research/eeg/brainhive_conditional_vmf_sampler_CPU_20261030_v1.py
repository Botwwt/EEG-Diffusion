"""Classical Ulrich-Wood vMF sampling, explicit private RNG, no image selection."""
import numpy as np
from brainhive_contrastive_marginal_mixture_CPU_20261029_v1 import condition_mixture


def sample_vmf(direction,concentration,samples,generator):
    mu=np.asarray(direction,dtype=np.float64)
    assert mu.ndim==1 and len(mu)>=3 and abs(np.linalg.norm(mu)-1)<1e-10
    assert concentration>=0 and np.isfinite(concentration) and samples>=1
    dimension=len(mu);n=dimension-1
    # Stable version of (sqrt(4*k*k+n*n)-2*k)/n.
    b=n/(np.hypot(2*concentration,n)+2*concentration)
    x0=(1-b)/(1+b)
    accepted=[];remaining=samples;proposals=0
    for unused in range(1000):
        if remaining==0:break
        z=generator.beta(n/2,n/2,size=remaining)
        axial=(1-(1+b)*z)/(1-(1-b)*z)
        log_accept=concentration*(axial-x0)+n*(np.log1p(-x0*axial)-np.log1p(-x0*x0))
        keep=np.log(generator.uniform(size=remaining))<=log_accept
        proposals+=remaining;accepted.append(axial[keep]);remaining-=int(keep.sum())
    if remaining:
        raise RuntimeError('Finite rejection budget exhausted; do not silently change distribution')
    axial=np.concatenate(accepted)
    tangent=generator.normal(size=(samples,dimension-1));tangent/=np.linalg.norm(tangent,axis=-1,keepdims=True)
    values=np.concatenate((axial[:,None],np.sqrt(np.maximum(0,1-axial**2))[:,None]*tangent),axis=1)
    # Stable Householder map from the signed first coordinate to mu.
    sign=1. if mu[0]>=0 else -1.
    reflector=mu.copy();reflector[0]+=sign;reflector/=np.linalg.norm(reflector)
    values[:,0]*=-sign
    values-=2*(values@reflector)[:,None]*reflector
    assert np.isfinite(values).all() and np.max(abs(np.linalg.norm(values,axis=-1)-1))<1e-12
    return values,dict(proposals=proposals,accepted=samples)


def sample_conditional_mixture(weights,means,concentrations,EEG_direction,scale,samples,generator):
    model=condition_mixture(weights,means,concentrations,EEG_direction,scale)
    assignments=generator.choice(len(weights),size=samples,p=model['weights'])
    values=np.empty((samples,means.shape[1]),dtype=np.float64);proposals=0
    for component in np.unique(assignments):
        indices=np.flatnonzero(assignments==component)
        draws,record=sample_vmf(model['means'][component],float(model['concentrations'][component]),len(indices),generator)
        values[indices]=draws;proposals+=record['proposals']
    return values,dict(component=assignments,proposals=proposals,samples=samples,log_evidence=model['log_evidence'])
