"""Classical full-T moment decomposition and fixed mean interventions, no fit."""
import numpy as np
ARMS=('direct','nce_continued','conditional','restore_old_mean','restore_visual_mean','opposite_old_shift')
METRICS=('F_cosine','F_raw_MSE','F_within10_top1','F_within10_two_afc','norm')

def moments(values,block=128):
    x=np.asarray(values,dtype=np.float64);assert x.ndim==2 and np.isfinite(x).all()
    sums=np.stack([v.sum(0) for v in np.array_split(x,np.arange(block,len(x),block))])
    sum_squares=np.stack([(v*v).sum(0) for v in np.array_split(x,np.arange(block,len(x),block))])
    mean=sums.sum(0)/len(x);variance=sum_squares.sum(0)/len(x)-mean*mean
    assert variance.min()>-1e-10
    return dict(mean=mean,variance=variance,block_sums=sums,block_squares=sum_squares)

def interventions(old,nce,fine,means):
    old=np.asarray(old,dtype=np.float64);nce=np.asarray(nce,dtype=np.float64);fine=np.asarray(fine,dtype=np.float64)
    assert old.shape==nce.shape==fine.shape and old.shape[1]==1024
    b=means['direct']-means['nce_continued'];v=means['visual']-means['nce_continued']
    return np.stack([old,nce,fine,nce+b,nce+v,nce-b],1)

def score_within(values,teacher):
    x=np.asarray(values,dtype=np.float64);f=np.asarray(teacher,dtype=np.float64);n,k,d=x.shape
    assert f.shape==(n,d) and n%10==0 and k==6 and d==1024
    norms=np.linalg.norm(x,axis=-1);fn=np.linalg.norm(f,axis=-1);assert norms.min()>1e-8 and fn.min()>1e-8
    xunit=x/norms[...,None];funit=f/fn[:,None]
    matrix=np.einsum('gaid,gjd->gaij',xunit.reshape(n//10,10,k,d).transpose(0,2,1,3),funit.reshape(n//10,10,d),optimize=True)
    # Canonical shape: arm, concept, query-in-concept, target-in-concept.
    matrix=matrix.transpose(1,0,2,3);i=np.arange(10);diag=matrix[:,:,i,i].transpose(1,2,0).reshape(n,k)
    top1=(matrix.argmax(-1)==i[None,None,:]).transpose(1,2,0).reshape(n,k)
    diag_a=matrix[:,:,i,i]
    afc=((matrix<diag_a[:,:,:,None]).sum(-1)+.5*((matrix==diag_a[:,:,:,None]).sum(-1)-1))/9
    afc=afc.transpose(1,2,0).reshape(n,k)
    mse=((norms*norms+fn[:,None]**2-2*norms*fn[:,None]*diag)/d)
    return matrix,dict(F_cosine=diag,F_raw_MSE=mse,F_within10_top1=top1.astype(float),F_within10_two_afc=afc,norm=norms)

def decompose_delta(old,new):
    delta=np.asarray(new,dtype=np.float64)-np.asarray(old,dtype=np.float64);mu=delta.mean(0)
    total=float((delta*delta).sum(1).mean());common=float(mu@mu);centered=float(((delta-mu)**2).sum(1).mean())
    assert abs(total-common-centered)<1e-9*max(1,total)
    return dict(total_energy=total,common_mean_shift_energy=common,centered_energy=centered,common_fraction=common/max(total,1e-300),mean_shift=mu)
