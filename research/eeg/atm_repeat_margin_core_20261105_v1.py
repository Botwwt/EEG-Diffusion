"""Frozen replicate agreement and task derivative mass; no new estimator claim."""
import numpy as np
STATES=('frozen_positive_NCE','R4_continued','positive_views','count_corrected')
METHODS=('native_NCE','relative_pairwise_logistic','uniform')
REGIONS=('all','within_concept','cross_concept')
CATEGORIES=('both_correct','both_wrong','discordant','zero_boundary')
SPLITS=(((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2)))

def sigmoid(x):
    return np.exp(-np.logaddexp(0.,-x))

def objective(logits,kind):
    """Float64 independent loss and complete dL/dS, including diagonal terms."""
    s=np.asarray(logits,np.float64);n=len(s);assert s.shape==(n,n) and n>1
    d=np.diag(s);off=~np.eye(n,dtype=bool)
    if kind=='native_NCE':
        a=s-s.max(1,keepdims=True);p=np.exp(a);p/=p.sum(1,keepdims=True)
        b=s-s.max(0,keepdims=True);q=np.exp(b);q/=q.sum(0,keepdims=True)
        loss=.5*((np.log(np.exp(a).sum(1))+s.max(1)-d).mean()+(np.log(np.exp(b).sum(0))+s.max(0)-d).mean())
        w=np.stack((p,q))/(2*n);g=w.sum(0);g[np.diag_indices(n)]-=1/n
    elif kind=='relative_pairwise_logistic':
        a=s-d[:,None];b=s-d[None,:]
        loss=(np.logaddexp(0.,a)[off].sum()+np.logaddexp(0.,b)[off].sum())/(2*n*(n-1))
        w=np.stack((sigmoid(a),sigmoid(b)))/(2*n*(n-1));w[:,~off]=0
        g=w.sum(0);g[np.diag_indices(n)]-=w[0].sum(1)+w[1].sum(0)
    else:
        assert kind=='uniform';loss=0.;w=np.ones((2,n,n))/(2*n*(n-1));g=np.zeros_like(s)
    w[:,~off]=0
    return float(loss),g,w

def torch_losses(s):
    import torch
    n=len(s);ids=torch.arange(n,device=s.device);off=~torch.eye(n,device=s.device,dtype=torch.bool)
    return (.5*(torch.nn.functional.cross_entropy(s,ids)+torch.nn.functional.cross_entropy(s.T,ids)),
        .5*(torch.nn.functional.softplus(s-s.diag()[:,None])[off].mean()+torch.nn.functional.softplus(s-s.diag()[None,:])[off].mean()))

def agreement(single_logits,permuted=False):
    s=np.asarray(single_logits,np.float64);r,n,m=s.shape;assert r==4 and n==m and n%10==0
    if permuted:
        perm=(np.arange(n)//10)*10+(np.arange(n)%10+1)%10;s=s[:,:,perm]
    d=np.diagonal(s,axis1=1,axis2=2)
    margins=np.stack((d[:,:,None]-s,d[:,None,:]-s),axis=1)
    a=np.zeros((2,n,n,4),np.float64)
    for left,right in SPLITS:
        x=margins[list(left)].mean(0);y=margins[list(right)].mean(0)
        a[...,0]+=(x>0)&(y>0);a[...,1]+=(x<0)&(y<0);a[...,2]+=(x*y<0);a[...,3]+=(x==0)|(y==0)
    a/=3;assert np.max(abs(a.sum(-1)-1))<1e-15
    return a

def measure(logits):
    """Per-query numerators; preserve both orientations before ratio reductions."""
    s=np.asarray(logits,np.float64);assert s.shape[0]==5
    n=s.shape[1];off=~np.eye(n,dtype=bool);same=(np.arange(n)//10)[:,None]==(np.arange(n)//10)[None,:]
    masks=(off,off&same,off&~same);w=np.stack([objective(s[4],k)[2] for k in METHODS])
    ev=np.stack((agreement(s[:4]),agreement(s[:4],True)))
    nums=np.zeros((n,3,3,2,4));dens=np.zeros((n,3,3))
    for ri,mask in enumerate(masks):
        for mi in range(3):
            v=w[mi]*mask
            dens[:,ri,mi]=v[0].sum(1)+v[1].sum(0)
            nums[:,ri,mi]=np.einsum('ij,cijk->ick',v[0],ev[:,0])+np.einsum('ij,cijk->jck',v[1],ev[:,1])
    assert np.max(abs(nums.sum(-1)-dens[:,:,:,None]))<1e-14
    return nums,dens

def math_checks():
    rng=np.random.default_rng(1729);s=rng.normal(size=(20,20));errors={}
    for kind in METHODS[:2]:
        loss,g,w=objective(s,kind);assert loss>=0 and abs(g.sum())<1e-12 and np.all(w>=0)
        errors[kind]=[]
        for i,j in ((0,0),(1,1),(0,1),(1,0),(3,12),(12,3)):
            a=s.copy();b=s.copy();a[i,j]+=1e-5;b[i,j]-=1e-5
            err=abs((objective(a,kind)[0]-objective(b,kind)[0])/2e-5-g[i,j]);assert err<1e-8;errors[kind].append(err)
    x=rng.normal(size=(5,20,20));num,den=measure(x);assert num.shape==(20,3,3,2,4)
    assert np.allclose(num[:,0],num[:,1]+num[:,2]) and np.allclose(den[:,0],den[:,1]+den[:,2])
    assert np.allclose(den[:,0,2],1/20)
    identical=np.repeat(s[None],4,axis=0);a=agreement(identical);assert np.all(a[...,2]==0)
    zero=agreement(np.zeros_like(identical));assert np.all(zero[...,3]==1)
    # Recordings can share arbitrary common noise: agreement alone is not unbiased neural signal.
    common=rng.normal(size=(20,20));assert np.all(agreement(np.repeat(common[None],4,0))[...,2]==0)
    return dict(status='PASS_ANALYTIC_LOSS_DERIVATIVES_AND_REPLICATE_PARTITION',max_FD_error=max(max(v) for v in errors.values()),
        derivative_coordinates=12,agreement_is_NOT_independent_signal_estimator=True,zero_and_shared_noise_counterexamples_pass=True)
