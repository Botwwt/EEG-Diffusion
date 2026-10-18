"""SORT endpoint pilot: inductive ordinal sets and certified convex projection.

These numerical routines contain no evaluation gallery or target-image lookup.
The conformal quantile needs exchangeability; it does not certify reused dev data.
"""
import math
import numpy as np


def unit(x, eps=1e-12):
    x=np.asarray(x,dtype=np.float64)
    norm=np.linalg.norm(x,axis=-1,keepdims=True)
    if np.any(norm<=eps) or not np.isfinite(x).all():
        raise ValueError('A finite nonzero vector is required')
    return x/norm


def anchor_directions(a, bank):
    """32 fixed local + 32 far-anchor pairs selected using EEG and fit bank only."""
    a=unit(a);bank=unit(bank)
    if a.ndim!=2 or bank.ndim!=2 or a.shape[1]!=bank.shape[1] or len(bank)<64:
        raise ValueError('Expected [B,d] and a fit bank with at least64 rows')
    score=a@bank.T
    # Stable ID tie-breaking is part of the frozen selection rule.
    order=np.argsort(-score,axis=1,kind='stable')
    local=[(i,j) for i in range(16) for j in range(i+1,16)][:32]
    far=np.rint(np.linspace(.40,.99,32)*(len(bank)-1)).astype(int)
    p=np.column_stack([order[:,i] for i,j in local]+[order[:,k%8] for k in range(32)])
    n=np.column_stack([order[:,j] for i,j in local]+[order[:,far[k]] for k in range(32)])
    diff=bank[p]-bank[n];norm=np.linalg.norm(diff,axis=-1)
    valid=norm>1e-10
    d=diff/np.maximum(norm[...,None],1e-12)
    signed=np.einsum('bld,bd->bl',d,a)
    d*=np.where(signed>=0,1.,-1.)[...,None]
    mu=np.abs(signed);valid &= mu>1e-10
    d=np.where(valid[...,None],d,0.)
    return d,mu,valid,np.stack([p,n],axis=-1)


def repeat_scale(d, trials, fit_floor):
    """A quality feature, not an independent-trial variance or a posterior SE.

    No1/R factor: duplicating an entire trial group cannot improve precision.
    A single trial or zero dispersion still retains the fitted positive floor.
    """
    if not np.isfinite(fit_floor) or fit_floor<=0:
        raise ValueError('A strictly positive fit-only scale floor is required')
    trials=unit(trials)
    projected=np.einsum('bld,brd->blr',d,trials)
    dispersion=np.mean((projected-projected.mean(-1,keepdims=True))**2,axis=-1)
    return np.sqrt(dispersion+float(fit_floor)**2)


def conformal_scores(d,mu,sigma,valid,target,rho=.5):
    if not 0<rho<1 or np.any(sigma<=0):raise ValueError('rho and scale')
    y=unit(target)
    residual=(rho*mu-np.einsum('bld,bd->bl',d,y))/sigma
    return np.maximum(np.where(valid,residual,0.),0.).max(axis=1)


def conformal_quantile(scores,alpha=.1):
    scores=np.asarray(scores,dtype=np.float64)
    if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all() or np.any(scores<0):
        raise ValueError('Finite nonnegative calibration scores required')
    if not 0<alpha<1:raise ValueError('alpha')
    k=math.ceil((len(scores)+1)*(1-alpha))
    return float(np.partition(scores,k-1)[k-1]) if k<=len(scores) else float('inf')


def boundaries(mu,sigma,valid,q,rho=.5):
    if q<0 or np.any(sigma<=0) or not 0<rho<1:raise ValueError('invalid set parameters')
    raw=rho*mu-q*sigma
    active=valid & (raw>1e-8)
    # Inactive entries must stay finite for dual certificates (0 * inf is NaN).
    return np.where(active,raw,0.),active


def random_rank_directions(a, reference_d, seed=2026101715):
    """Secondary control: fixed random span, same per-query numerical row rank."""
    a=unit(a);rng=np.random.default_rng(seed);d=a.shape[1];L=reference_d.shape[1]
    basis=np.linalg.qr(rng.normal(size=(d,L)),mode='reduced')[0]
    coefficients=rng.normal(size=(L,L));result=[];ranks=[]
    for v,ref in zip(a,reference_d):
        rank=int(np.linalg.matrix_rank(ref,tol=1e-8));ranks.append(rank)
        if not rank:result.append(np.zeros_like(ref));continue
        draw=unit(coefficients[:,:rank]@basis[:,:rank].T)
        draw*=np.where(draw@v>=0,1.,-1.)[:,None];result.append(draw)
    result=np.asarray(result);mu=np.einsum('bld,bd->bl',result,a)
    return result,mu,mu>1e-10,np.asarray(ranks)


def project_intersection(v,d,b,active,feasible_a,max_cycles=1200,tol=1e-7):
    """Dykstra in the span of(v,D), with primal/dual optimality certificates.

    Halfspace corrections are scalars times rows of D. The ball correction is
    represented in the same span, avoiding B*L*d state per Dykstra iteration.
    A failed certificate falls back to the known feasible EEG direction and is
    explicitly marked. This is not a silent claim of a minimal projection.
    """
    supplied=np.asarray(v,dtype=np.float64);v=unit(supplied)
    v=np.where((np.abs(np.linalg.norm(supplied,axis=-1)-1)<1e-12)[:,None],supplied,v)
    a=unit(feasible_a);d=np.asarray(d,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64);active=np.asarray(active,dtype=bool)
    if d.shape[:2]!=b.shape or b.shape!=active.shape or d.shape[0]!=len(v):raise ValueError('shape')
    if np.any(b[active]<=0) or np.any(np.einsum('bld,bd->bl',d,a)[active]<b[active]-1e-8):
        raise ValueError('The supplied EEG direction must be feasible')
    out=v.copy();meta=dict(cycles=np.zeros(len(v),dtype=int),fallback=np.zeros(len(v),bool),
        projection_gap=np.zeros(len(v)),pre_normalization_violation=np.zeros(len(v)),
        pre_normalization_norm=np.ones(len(v)),ball_active=np.zeros(len(v),bool),
        movement=np.zeros(len(v)),active_count=active.sum(1))
    selected=np.flatnonzero(active.any(axis=1))
    if not len(selected):return out,meta,out.copy()
    D=d[selected];B=b[selected];mask=active[selected];V=v[selected]
    G=D@D.swapaxes(1,2);dv=np.einsum('bld,bd->bl',D,V);vv=np.sum(V*V,1)
    count,L=B.shape
    coef=np.zeros((count,L));ac=np.ones(count);pc=np.zeros_like(coef);pa=np.zeros(count)
    hp=np.zeros_like(coef);dots=dv.copy();converged=np.zeros(count,bool);cycles=np.zeros(count,int)
    gap=np.full(count,np.inf);violation=gap.copy()
    def mat(x):return np.einsum('bij,bj->bi',G,x,optimize=True)
    def norm2(av,cv):return np.maximum(0,av*av*vv+2*av*np.sum(cv*dv,1)+np.sum(cv*mat(cv),1))
    for cycle in range(1,max_cycles+1):
        for j in range(L):
            use=mask[:,j] & ~converged
            if not use.any():continue
            denom=np.maximum(G[:,j,j],1e-15)
            displacement=np.maximum(B[:,j]-dots[:,j]-hp[:,j]*denom,0.)/denom
            delta=np.where(use,hp[:,j]+displacement,0.)
            coef[:,j]+=delta;dots+=delta[:,None]*G[:,j,:]
            hp[:,j]=np.where(use,-displacement,hp[:,j])
        yc=coef+pc;ya=ac+pa;r=np.sqrt(norm2(ya,yc));scale=np.maximum(1.,r)
        nc=yc/scale[:,None];na=ya/scale
        # Hold converged query state exactly while the remaining queries finish.
        use=~converged
        pc=np.where(use[:,None],yc-nc,pc);pa=np.where(use,ya-na,pa)
        coef=np.where(use[:,None],nc,coef);ac=np.where(use,na,ac)
        dots=ac[:,None]*dv+mat(coef)
        if cycle%5==0 or cycle==max_cycles:
            lam=np.maximum(-hp,0.)*mask
            wn=np.sqrt(norm2(np.ones(count),lam))
            phi=np.where(wn<=1,.5*wn*wn,wn-.5)
            dual=.5*vv+np.sum(lam*B,1)-phi
            primal=.5*norm2(ac-1,coef)
            gap=np.maximum(primal-dual,0.)
            violation=np.maximum(np.where(mask,B-dots,0.),0.).max(1)
            done=(violation<=tol)&(gap<=tol)
            newly=done&~converged;cycles[newly]=cycle;converged|=done
            if converged.all():break
    cycles[~converged]=max_cycles
    projected=ac[:,None]*V+np.einsum('bl,bld->bd',coef,D,optimize=True)
    n=np.linalg.norm(projected,axis=1)
    geometric=np.einsum('bld,bd->bl',D,projected)
    violation=np.maximum(np.where(mask,B-geometric,0.),0.).max(1)
    certified=converged & (violation<=2*tol) & (n<=1+2*tol) & (n>1e-12)
    projected[~certified]=a[selected[~certified]]
    normed=unit(projected)
    final_violation=np.maximum(np.where(mask,B-np.einsum('bld,bd->bl',D,normed),0.),0.).max(1)
    if np.any(final_violation>1e-6):raise ArithmeticError('Final normalized endpoint violates its fixed constraints')
    out[selected]=normed;pre=v.copy();pre[selected]=projected
    meta['cycles'][selected]=cycles;meta['fallback'][selected]=~certified
    meta['projection_gap'][selected]=gap;meta['pre_normalization_violation'][selected]=violation
    meta['pre_normalization_norm'][selected]=n;meta['ball_active'][selected]=n>=1-1e-6
    meta['movement']=np.linalg.norm(out-v,axis=1)
    return out,meta,pre
