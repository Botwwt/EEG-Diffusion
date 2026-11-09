"""Classical second log-signature and exact conditional ridge algebra.

An identifiability probe, not a deployed encoder or a novelty assertion.
"""
import numpy as np

EDGES=(0,50,100,150,200,250)
ARMS=('c_only','c_raw','c_raw_scatter','c_raw_scatter_area',
      'c_raw_scatter_random_sign_area','c_raw_scatter_wrong_area')
METRICS=('F_MSE','correct_F_cosine','same_concept_Top1','same_concept_2AFC','correct_minus_other9_cosine')


def area_and_scatter(increments):
    """Five ordered native-projection increments; retain all off-diagonal pairs."""
    x=np.asarray(increments,np.float64)
    assert x.ndim==3 and x.shape[1]==5
    prefix=np.cumsum(x,axis=1)-x
    ordered=np.einsum('nti,ntj->nij',prefix,x)
    area=.5*(ordered-ordered.swapaxes(-1,-2))
    scatter=np.einsum('nti,ntj->nij',x,x)
    indices=np.triu_indices(x.shape[-1],1)
    return area[:,indices[0],indices[1]],scatter[:,indices[0],indices[1]]


def standardize(fit,held):
    fit=np.asarray(fit,np.float64);held=np.asarray(held,np.float64)
    mean=fit.mean(0);scale=np.sqrt(np.square(fit-mean).mean(0))
    constant=scale<1e-10;scale[constant]=1.
    return (fit-mean)/scale,(held-mean)/scale,dict(mean=mean,scale=scale,constant=constant)


def fit_basis(contributions,fit_rows,rank=32):
    """Use FIT EEG only; no labels, held rows, target or decoder scores."""
    x=np.asarray(contributions,np.float64)
    mean=x[fit_rows].mean(0)
    centered=x[fit_rows]-mean
    cov=np.einsum('nti,ntj->ij',centered,centered)/(len(fit_rows)*5)
    eigenvalues,eigenvectors=np.linalg.eigh(cov)
    values=eigenvalues[-rank:][::-1].copy()
    basis=eigenvectors[:,-rank:][:,::-1].copy()
    # Deterministic column gauge; the full wedge feature is gauge equivalent.
    selected=np.argmax(np.abs(basis),axis=0)
    basis*=np.where(basis[selected,np.arange(rank)]<0,-1.,1.)
    scales=np.sqrt(np.maximum(values,1e-12))
    projected=((x-mean)@basis)/scales
    note=dict(mean=mean,basis=basis,scales=scales,eigenvalues=values,
              retained_variance=float(values.sum()/np.maximum(eigenvalues.sum(),1e-30)))
    return projected,note


def solve(torch,X,Y,penalty):
    gram=X.T@X
    gram.diagonal().add_(penalty)
    L=torch.linalg.cholesky(gram)
    return L,torch.cholesky_solve(X.T@Y,L)


def extend(torch,L,base,X,Z,Y,penalty):
    cross=X.T@Z
    E=torch.cholesky_solve(cross,L)
    schur=Z.T@Z-cross.T@E
    schur.diagonal().add_(penalty)
    schur=(schur+schur.T)/2
    extra=torch.cholesky_solve(Z.T@Y-cross.T@base,torch.linalg.cholesky(schur))
    return base-E@extra,extra


def metrics(prediction,targets,candidates,labels):
    p=np.asarray(prediction,np.float64);y=np.asarray(targets,np.float64)
    yn=y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-30)
    result=np.empty((len(p),5),np.float64)
    for lo in range(0,len(p),128):
        v=p[lo:lo+128];v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-30)
        sim=np.einsum('nd,nkd->nk',v,yn[candidates[lo:lo+len(v)]])
        correct=sim[np.arange(len(v)),labels[lo:lo+len(v)]]
        top=(correct==sim.max(1))/(sim==sim.max(1)[:,None]).sum(1)
        afc=((correct[:,None]>sim).sum(1)+.5*(correct[:,None]==sim).sum(1)-.5)/9
        result[lo:lo+len(v)]=np.stack((np.square(p[lo:lo+len(v)]-y[lo:lo+len(v)]).mean(1),
              correct,top,afc,correct-(sim.sum(1)-correct)/9),axis=-1)
    return result


def summarize(concept_values,bootstrap_indices):
    x=np.asarray(concept_values,np.float64)
    draws=np.concatenate([x[bootstrap_indices[i:i+20]].mean(1) for i in range(0,len(bootstrap_indices),20)])
    lo,hi=np.quantile(draws,[.025,.975],axis=0)
    return dict(mean=x.mean(0).tolist(),descriptive95_lower=lo.tolist(),descriptive95_upper=hi.tolist()),draws


def decision(comparisons):
    main=comparisons['c_raw_scatter']
    passed=(main['relative_MSE_improvement']>=.005 and main['mean'][2]>=.01
       and all(v['descriptive95_lower'][0]>0 and v['mean'][2]>=0 for v in comparisons.values()))
    return dict(predictive_gate_passed=bool(passed),
        action='PREDICTIVE_SUPPORT_REQUIRES_SEPARATE_ARCHITECTURE_AND_IMAGES' if passed else 'STOP_THIS_FIXED_ORDERED_AREA_INSTANCE',
        automatic_training_or_images=False,novelty_established=False)
