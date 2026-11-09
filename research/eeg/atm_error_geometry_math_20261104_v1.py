"""EEG prediction-error coordinates, not a physiological noise model.

For z=a*f+b*eps, m=c+mean_T(f-c), s^2=a^2*lambda+b^2:
u=(z-a*m)/s, A=a*sqrt(lambda)/s, B=b/s,
v=A*eps-B*(f-m)/sqrt(lambda), D=m+sqrt(lambda)*(A*u-B*v).
The rotation is classical v/EDM preconditioning in paired error coordinates.
"""
import numpy as np

MODES=('native_v','iso_error','paired_error','wrong_error')

def coefficients(alpha,lam):
    alpha=np.asarray(alpha,np.float64).reshape(-1,1);lam=np.asarray(lam,np.float64).reshape(1,-1)
    assert np.isfinite(alpha).all() and np.isfinite(lam).all() and np.all((alpha>0)&(alpha<1)) and np.all(lam>0)
    a=np.sqrt(alpha);b=np.sqrt(1-alpha);s=np.sqrt(alpha*lam+1-alpha)
    return a,b,s,a*np.sqrt(lam)/s,b/s

def target_and_clean(f,c,noise,alpha,mu,lam):
    a,b,s,A,B=coefficients(alpha,lam);m=np.asarray(c)+mu;z=a*f+b*noise;u=(z-a*m)/s
    v=A*noise-B*(f-m)/np.sqrt(lam)
    clean=m+np.sqrt(lam)*(A*u-B*v)
    return dict(z=z,u=u,v=v,clean=clean,A=A,B=B,s=s)

def statistics(c,f):
    c=np.asarray(c,np.float64);f=np.asarray(f,np.float64);assert c.shape==f.shape and c.ndim==2 and np.isfinite(c).all() and np.isfinite(f).all()
    error=f-c;mu=error.mean(0);r=error-mu;cov=r.T@r/len(r)
    values,basis=np.linalg.eigh(cov);average=float(values.mean());assert average>0 and values.min()>-average*1e-8
    floor=average*1e-6;lam=np.maximum(values,floor);permutation=np.random.default_rng(1729).permutation(len(lam))
    return dict(mean_error=mu,basis=basis,eigenvalues_unfloored=values,eigenvalues=lam,
        isotropic_eigenvalues=np.full_like(lam,average),wrong_eigenvalues=lam[permutation],permutation=permutation,
        eigenvalue_floor=np.array(floor),mean_error_rotated=mu@basis,
        covariance_reconstruction_maxabs=np.array(abs((basis*values)@basis.T-cov).max()),
        orthogonality_maxabs=np.array(abs(basis.T@basis-np.eye(len(values))).max()),
        target_maximum_norm=np.array(np.linalg.norm(f,axis=1).max()),
        target_mean_norm=np.array(np.linalg.norm(f,axis=1).mean()),
        residual_second_moment=np.array(np.mean(r*r)))
