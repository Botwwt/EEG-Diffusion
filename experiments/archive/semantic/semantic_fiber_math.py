"""Gaussian semantic tilt controls. Mathematical properties, not a novelty claim."""
import numpy as np


def sym(a): return (a+a.T)/2


def sqrt_spd(a):
    w,v=np.linalg.eigh(sym(a));assert w.min()>0
    return (v*np.sqrt(w))@v.T


def kl_cov(q,c):
    d=len(c);sq,lq=np.linalg.slogdet(q);sc,lc=np.linalg.slogdet(c);assert sq==sc==1
    return float(.5*(np.trace(np.linalg.solve(c,q))-d+lc-lq))


def tilt_cov(c,p,beta):
    if beta==0:return c.copy()
    return sym(c-c@p.T@np.linalg.solve(np.eye(len(p))/beta+p@c@p.T,p@c))


def budget_tilt(c,p,budget):
    if budget==0:return c.copy(),0.
    lower=0.;upper=1.
    while kl_cov(tilt_cov(c,p,upper),c)<budget:upper*=2
    for _ in range(80):
        middle=(lower+upper)/2
        if kl_cov(tilt_cov(c,p,middle),c)<budget:lower=middle
        else:upper=middle
    beta=(lower+upper)/2;q=tilt_cov(c,p,beta)
    assert abs(kl_cov(q,c)-budget)<1e-9
    return q,beta


def uniform_kl_scale(d,budget):
    lo=1e-12;hi=1.
    for _ in range(80):
        r=(lo+hi)/2
        if .5*d*(r-1-np.log(r))>budget:lo=r
        else:hi=r
    return (lo+hi)/2


def fiber_check(c,q,p):
    _,_,v=np.linalg.svd(p,full_matrices=True);n=v[len(p):]
    def parts(a):
        uu=p@a@p.T;vu=n@a@p.T
        coefficient=np.linalg.solve(uu,vu.T).T
        conditional=sym(n@a@n.T-coefficient@vu.T)
        return coefficient,conditional
    cb,vb=parts(c);cq,vq=parts(q)
    return dict(conditional_regression_max_error=float(abs(cb-cq).max()),
        conditional_covariance_max_error=float(abs(vb-vq).max()))


def exact_gaussian_vp_flow(mean,c,noise,steps=50):
    """Exact affine probability flow for Gaussian VP marginals, including endpoints."""
    w,v=np.linalg.eigh(sym(c));assert w.min()>0
    x=noise.copy();previous_a=0.;previous_variance=np.ones(len(c))
    for k in range(1,steps+1):
        a=1. if k==steps else np.sin(k/steps*np.pi/2)
        variance=a*a*w+(1-a*a)
        x=a*mean+((x-previous_a*mean)@v*np.sqrt(variance/previous_variance))@v.T
        previous_a=a;previous_variance=variance
    return x


def validate_math():
    rng=np.random.default_rng(2026101668);a=rng.normal(size=(8,8));c=a@a.T+.5*np.eye(8)
    p=np.linalg.qr(rng.normal(size=(8,3)))[0].T
    q,beta=budget_tilt(c,p,1.)
    assert np.array_equal(tilt_cov(c,p,0),c)
    inverse_error=float(abs(np.linalg.inv(q)-(np.linalg.inv(c)+beta*p.T@p)).max());assert inverse_error<1e-10
    f=fiber_check(c,q,p);assert max(f.values())<1e-10
    # Ordinary orthogonal-noise contraction fails to preserve correlated fibers.
    naive=(np.eye(8)-.5*p.T@p)@c@(np.eye(8)-.5*p.T@p)
    assert max(fiber_check(c,naive,p).values())>1e-3
    noise=rng.normal(size=(11,8));mean=rng.normal(size=(11,8))
    endpoint=mean+noise@sqrt_spd(q)
    e1=float(abs(exact_gaussian_vp_flow(mean,q,noise,1)-endpoint).max())
    e50=float(abs(exact_gaussian_vp_flow(mean,q,noise,50)-endpoint).max())
    assert e1<1e-10 and e50<1e-10
    vals=[np.trace(p@tilt_cov(c,p,b)@p.T) for b in [0,.01,.1,1,10]]
    assert np.all(np.diff(vals)<0)
    r=uniform_kl_scale(8,1.);assert abs(kl_cov(r*c,c)-1)<1e-10
    return dict(status='completed',validation='synthetic-pass only',null_identity=True,
        precision_identity_max_error=inverse_error,fiber_preservation=f,
        incorrect_projected_noise_counterexample_passed=True,
        exact_flow_endpoint_error_1step=e1,exact_flow_endpoint_error_50step=e50,
        projected_risk_monotone=True,matched_KL_passed=True,
        image_quality_or_real_performance_proven=False)
