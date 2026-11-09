"""Classical spherical quadratic MAP used as a training-target candidate.

This solver is not an originality claim. H is a measured local generator
semantic metric; R is an empirical repeated-EEG dispersion proxy, not a
validated physiological posterior covariance. All coordinates precede the
actual host output affine. No test image is an inference input.
"""
import numpy as np

def centered_basis(d):
    v=np.zeros(d,dtype=np.float64);v[-1]=1.;v-=1./np.sqrt(d)
    return (np.eye(d)-2.*np.outer(v,v)/(v@v))[:,:-1]

def spherical_quadratic_target(m,f,H,precision):
    """Minimize .5(u-f)'H(u-f)+.5(u-m)'precision(u-m).

    Constraints: sum(u)=0 and ||u||=||m||. Solves the full equality-constrained
    quadratic via its scalar secular equation, including the hard case.
    H and precision must be symmetric, H PSD and precision positive definite.
    """
    m=np.asarray(m,dtype=np.float64);f=np.asarray(f,dtype=np.float64)
    d=m.size;Q=centered_basis(d);radius=float(np.linalg.norm(m))
    assert m.shape==f.shape==(d,) and radius>0
    assert abs(m.mean())<1e-8 and abs(f.mean())<1e-8
    assert H.shape==precision.shape==(d,d)
    assert np.allclose(H,H.T,atol=1e-10) and np.allclose(precision,precision.T,atol=1e-10)
    A=Q.T@(H+precision)@Q;b=Q.T@(H@f+precision@m)
    eigen,U=np.linalg.eigh(A);z=U.T@b
    assert eigen[0]>0 and np.isfinite(eigen).all()
    tolerance=1e-11*max(1.,abs(eigen[-1]));minimum=np.abs(eigen-eigen[0])<=tolerance
    hard=bool(np.linalg.norm(z[minimum])<1e-11*max(1.,np.linalg.norm(z)))
    y=np.zeros_like(z)
    if hard:
        y[~minimum]=z[~minimum]/(eigen[~minimum]-eigen[0])
        hard=bool(np.linalg.norm(y)<=radius)
    if hard:
        multiplier=-float(eigen[0]);y[np.flatnonzero(minimum)[0]]=np.sqrt(max(0.,radius**2-y@y))
    else:
        lo=-float(eigen[0]);hi=max(1.,np.linalg.norm(z)/radius)
        def norm_at(value):return float(np.linalg.norm(z/(eigen+value)))
        while norm_at(hi)>radius:hi=2.*hi+1.
        for _ in range(160):
            mid=(lo+hi)*.5
            if mid==lo or mid==hi:break
            if norm_at(mid)>radius:lo=mid
            else:hi=mid
        multiplier=hi;y=z/(eigen+multiplier)
    u=Q@(U@y)
    def objective(v):
        a=v-f;c=v-m
        return float(.5*a@H@a+.5*c@precision@c)
    residual=Q.T@((H+precision)@u-(H@f+precision@m)+multiplier*u)
    result=dict(multiplier=multiplier,hard_case=hard,
        centered_error=float(abs(u.mean())),radius_relative_error=float(abs(np.linalg.norm(u)-radius)/radius),
        KKT_relative_residual=float(np.linalg.norm(residual)/max(1.,np.linalg.norm(b))),
        minimum_shifted_eigenvalue=float(eigen[0]+multiplier),objective=objective(u),
        objective_at_EEG=objective(m),objective_at_visual=objective(f))
    assert result['centered_error']<1e-8 and result['radius_relative_error']<1e-7
    assert result['KKT_relative_residual']<1e-7 and result['minimum_shifted_eigenvalue']>-1e-8
    return u,result

def empirical_repeat_precision(views):
    """Delete-one dispersion with fixed half-isotropic shrinkage.

    Four R3 views yield jackknife 3/4 sum outer products. This is explicitly
    an empirical geometry; correlated sessions and nonlinear encoders can
    invalidate any interpretation as the true R4 estimator covariance.
    """
    views=np.asarray(views,dtype=np.float64);assert views.ndim==2 and len(views)==4
    d=views.shape[1];D=views-views.mean(0,keepdims=True);D-=D.mean(1,keepdims=True)
    S=.75*D.T@D;scale=float(np.trace(S)/(d-1))
    if scale<=1e-14:return np.eye(d),dict(zero_dispersion=True,scale=scale)
    R=.5*S+.5*scale*np.eye(d);precision=np.linalg.inv(R)
    Q=centered_basis(d);precision*=((d-1)/np.trace(Q.T@precision@Q))
    return precision,dict(zero_dispersion=False,scale=scale,shrinkage=.5,
        covariance_is_empirical_proxy=True,true_iid_or_calibrated_posterior_not_established=True)
