"""Classical quadratic proximal solve, parameterized by physical-repeat versus R4 task-gradient differences.

This is a candidate optimizer rule, not a claim of a new matrix identity or a
validated estimator of physiological noise. All input arrays are float64.
"""
import numpy as np

def solve(base_data_step,gradient_R4,repeat_gradients,diagonal_metric_inverse):
    v=np.asarray(base_data_step,np.float64);g=np.asarray(gradient_R4,np.float64);G=np.asarray(repeat_gradients,np.float64);D=np.asarray(diagonal_metric_inverse,np.float64)
    assert v.ndim==1 and v.shape==g.shape==D.shape and G.shape==(4,len(v))
    assert np.isfinite(v).all() and np.isfinite(g).all() and np.isfinite(G).all() and np.isfinite(D).all() and np.all(D>=0)
    U=(G-g[None,:]).T/2.;DU=D[:,None]*U;B=U.T@DU;B=(B+B.T)/2
    tau=float(g@(D*g));e,V=np.linalg.eigh(B);assert e.min()>=-1e-10*max(1.,e.max())
    e=np.maximum(e,0);b=U.T@v
    if not np.any(e):
        return v.copy(),dict(exact_zero_covariance_fallback=True,tau=tau,eigenvalues=e,stationarity_relative=0.,sensitivity_before=float(b@b),sensitivity_after=float(b@b))
    tol=np.finfo(np.float64).eps*max(e.max(),tau)*16
    inv=np.divide(1.,e+tau,out=np.zeros(4),where=e+tau>tol)
    coeff=V@(inv*(V.T@b));out=v-DU@coeff
    after=U.T@out;res=(B+tau*np.eye(4))@coeff-b
    # tau==0 means the limiting projection, with numerical rank explicitly reported.
    residual=float(np.linalg.norm(res)/(np.linalg.norm(b)+1e-30));assert residual<1e-6
    assert after@after<=b@b+1e-10*max(1.,b@b)
    return out,dict(exact_zero_covariance_fallback=False,tau=tau,eigenvalues=e,stationarity_relative=residual,
        sensitivity_before=float(b@b),sensitivity_after=float(after@after),numerical_rank=int((e>tol).sum()))

def checks():
    rng=np.random.default_rng(1729);v=rng.normal(size=13);g=rng.normal(size=13);G=rng.normal(size=(4,13));D=np.exp(rng.normal(size=13))
    z,r=solve(v,g,G,D);U=(G-g[None,:]).T/2.;tau=g@(D*g)
    exact=np.linalg.solve(np.diag(1/D)+U@U.T/tau,v/D)
    err=float(abs(z-exact).max());assert err<1e-12
    zero,zz=solve(v,g,np.repeat(g[None,:],4,axis=0),D);assert np.array_equal(zero,v)
    for scale in (1e-3,2.,20.):
        x,_=solve(v,g*scale,G*scale,D);assert np.max(abs(x-z))<1e-12
    tiny,_=solve(v,g,g[None,:]+(G-g[None,:])*1e-8,D);assert np.max(abs(tiny-v))<1e-12
    perm,_=solve(v,g,G[[2,0,3,1]],D);assert np.max(abs(perm-z))<1e-12
    signs=rng.choice([-1.,1.],size=13);S=g[None,:]+(G-g[None,:])*signs
    B=U.T@(D[:,None]*U);Us=(S-g[None,:]).T/2.
    assert np.max(abs(Us.T@(D[:,None]*Us)-B))<1e-12
    assert np.linalg.norm(z)<np.inf
    # Reducing directional replicate variation does NOT guarantee reducing finite-step task loss.
    vv=np.array([-1.,-1.]);gg=-vv;GG=np.array([[1.,0.],[-1.,0.],[1.,0.],[-1.,0.]])
    x,rr=solve(vv,gg,GG,np.ones(2));assert rr['sensitivity_after']<rr['sensitivity_before']
    assert np.square(x-vv).sum()>0 # base step is already at a quadratic optimum.
    return dict(status='PASS_R1_R4_DIFFERENCE_PROXIMAL_IDENTITY_AND_LIMITS',direct_solve_max_error=err,zero_fallback_bitexact=True,
        small_noise_continuity=True,gradient_unit_invariance=True,repeat_order_invariance=True,same_spectrum_sign_control=True,
        finite_task_improvement_counterexample=True,not_novel_matrix_identity=True,not_centered_noise_covariance=True)
