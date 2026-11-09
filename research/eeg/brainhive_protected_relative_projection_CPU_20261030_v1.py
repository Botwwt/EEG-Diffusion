"""DRAFT frozen-field composition using a classical constrained projection.

Protect the original ordinary condition's affine norm, mean and one T-derived
linear population score. Preserve the original native angular budget. This
does NOT preserve the entire A/P response or prove image benefit/novelty.
"""
import numpy as np


def protected_relative_projection(native, ordinary, initial_F, relative, gamma, beta,
                                  population, *, enabled=True, radius=.25):
    assert radius == .25
    arrays=[np.asarray(x) for x in (native,ordinary,initial_F,relative,gamma,beta)]
    assert all(x.shape==arrays[0].shape for x in arrays)
    assert arrays[0].ndim==2 and arrays[0].shape[1]==1024
    assert all(np.isfinite(x).all() for x in arrays) and np.all(np.abs(arrays[4])>1e-6)
    dtype=arrays[1].dtype
    if not enabled:return arrays[1].copy(),dict(disabled_exact=True)
    m=np.asarray(population,dtype=np.float64);assert m.shape==(1024,) and abs(np.linalg.norm(m)-1)<1e-12
    cn,c0,cf,ce,g,b=[x.astype(np.float64) for x in arrays]
    un=(cn-b)/g;u0=(c0-b)/g
    # Pure learning increment relative to the very same starting F head.
    v=u0+(ce-cf)/g
    b1=np.ones_like(u0)/np.sqrt(1024)
    a=g*m[None,:];a-=b1*np.sum(a*b1,axis=1,keepdims=True)
    an=np.linalg.norm(a,axis=1,keepdims=True)
    b2=np.divide(a,an,out=np.zeros_like(a),where=an>1e-12)
    def null(x):return x-b1*np.sum(x*b1,axis=1,keepdims=True)-b2*np.sum(x*b2,axis=1,keepdims=True)
    p=u0-null(u0);r2=np.sum(u0*u0,axis=1);pn2=np.sum(p*p,axis=1)
    free2=r2-pn2;assert np.min(free2)>-1e-10
    R=np.sqrt(np.maximum(free2,0));rn=np.linalg.norm(un,axis=1);assert np.min(r2)>1e-12 and np.min(rn)>1e-12
    w=null(v);wn=np.linalg.norm(w,axis=1);h=null(un);hn=np.linalg.norm(h,axis=1)
    tiny_free=R<=1e-12*np.sqrt(r2)
    no_axis=hn<=1e-12*rn
    cap_cos=1/np.sqrt(1+radius*radius)
    rhs=rn*np.sqrt(r2)*cap_cos-np.sum(un*p,axis=1)
    tau=np.divide(rhs,R*hn,out=np.zeros_like(rhs),where=~tiny_free&~no_axis)
    regular=~tiny_free&~no_axis
    if np.any(tau[regular]>1+1e-10):raise ValueError('Protected plane and native cap infeasible; no radius relaxation')
    if np.any(rhs[no_axis]>1e-10):raise ValueError('Native cap infeasible on protected plane')
    tau=np.clip(tau,-1,1)
    axis=np.divide(h,hn[:,None],out=np.zeros_like(h),where=hn[:,None]>0)
    reference=np.divide(null(u0),R[:,None],out=np.zeros_like(u0),where=R[:,None]>0)
    z=np.divide(w,wn[:,None],out=reference.copy(),where=wn[:,None]>1e-12)
    active=regular&(wn>1e-12)&(np.sum(z*axis,axis=1)<tau)
    near_parallel=np.zeros(len(u0),dtype=bool)
    optimal=wn.copy()
    for i in np.flatnonzero(active):
        wa=float(w[i]@axis[i]);orth=w[i]-wa*axis[i];on=np.linalg.norm(orth)
        if on<=1e-12*max(1,wn[i]):
            # A true objective tie: choose a deterministic null-space basis,
            # never a direction chosen using labels, targets or effect scores.
            leverage=1-b1[i]**2-b2[i]**2-axis[i]**2;j=int(np.argmax(leverage))
            orth=np.zeros(1024);orth[j]=1;orth-=b1[i,j]*b1[i]+b2[i,j]*b2[i]+axis[i,j]*axis[i]
            orth/=np.linalg.norm(orth);near_parallel[i]=True
        else:orth/=on
        z[i]=tau[i]*axis[i]+np.sqrt(max(0,1-tau[i]**2))*orth
        optimal[i]=tau[i]*wa+np.sqrt(max(0,1-tau[i]**2))*on
    projected=p+R[:,None]*z
    projected[tiny_free]=u0[tiny_free]
    # An exactly zero field increment returns the original ordinary bytes.
    identity=np.all(arrays[2]==arrays[3],axis=1)
    result=(c0+g*(projected-u0)).astype(dtype);result[identity]=arrays[1][identity]
    actual=(result.astype(np.float64)-b)/g
    native_cos=np.sum(actual*un,axis=1)/(np.linalg.norm(actual,axis=1)*rn)
    old_cos=np.sum(u0*un,axis=1)/(np.linalg.norm(u0,axis=1)*rn)
    certificate_error=np.abs(np.sum(w*z,axis=1)-optimal)/(1+np.abs(optimal))
    assert np.max(certificate_error[~tiny_free])<1e-9
    info=dict(active_native_caps=active,degenerate_direction_ties=near_parallel,zero_increment_exact=identity,
        ideal_objective_certificate_scaled_error=certificate_error,
        affine_norm_relative_error=np.abs(np.linalg.norm(actual,axis=1)/np.sqrt(r2)-1),
        affine_mean_error=np.abs(np.mean(actual-u0,axis=1)),
        protected_linear_score_error=np.abs((result.astype(np.float64)-c0)@m),
        native_cosine_before=old_cos,native_cosine_after=native_cos,
        native_cosine_lower_bound=cap_cos,protects_complete_AP_response=False)
    return result,info
