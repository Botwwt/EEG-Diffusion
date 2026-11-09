"""Fixed-EEG equal-angle direction interventions; classical geometry only.

The actual old corrected condition defines each query's angle. Both controls
still depend on query EEG geometry and old-head amplitude. No EEG-free claim.
"""
import numpy as np

ARMS=('population_direction','varying_direction')
def project(x,u):
    x=np.asarray(x,np.float64);u=np.asarray(u,np.float64)
    return x-u*np.sum(x*u,axis=-1,keepdims=True)/np.sum(u*u,axis=-1,keepdims=True)

def interventions(native,corrected,gamma,beta,centered_direction,population_mean):
    native=np.asarray(native,np.float64);corrected=np.asarray(corrected,np.float64)
    gamma=np.broadcast_to(np.asarray(gamma,np.float64),native.shape);beta=np.broadcast_to(np.asarray(beta,np.float64),native.shape)
    assert native.ndim==2 and native.shape[1]==1024 and corrected.shape==native.shape and np.min(np.abs(gamma))>1e-6
    d=np.asarray(centered_direction,np.float64);mean=np.broadcast_to(np.asarray(population_mean,np.float64),native.shape)
    assert d.shape==mean.shape==native.shape
    u=(native-beta)/gamma;v=(corrected-beta)/gamma;radius=np.linalg.norm(u,axis=1,keepdims=True)
    cosine=np.sum(u*v,axis=1,keepdims=True)/(radius*np.linalg.norm(v,axis=1,keepdims=True));cosine=np.clip(cosine,-1.,1.)
    theta=np.arccos(cosine);assert np.max(theta)<=np.arctan(.25)+1e-5
    pop=project(mean,u);vary=project(d-mean,u);full=project(d,u)
    assert np.max(np.abs(full-pop-vary))<1e-9
    values=[];reports={}
    for arm,q in zip(ARMS,(pop,vary)):
        qnorm=np.linalg.norm(q,axis=1,keepdims=True);fallback=(qnorm[:,0]<=1e-12)|(theta[:,0]<=1e-12)
        unit=q/np.maximum(qnorm,1e-12);new_u=np.cos(theta)*u+np.sin(theta)*radius*unit
        result=beta+gamma*new_u;result[fallback]=corrected[fallback]
        actual=result.astype(np.float32);a=(actual.astype(np.float64)-beta)/gamma
        angles=np.arccos(np.clip(np.sum(a*u,axis=1,keepdims=True)/(np.linalg.norm(a,axis=1,keepdims=True)*radius),-1,1))
        norm_error=np.abs(np.linalg.norm(a,axis=1,keepdims=True)/radius-1)
        error=float(np.max(np.abs(angles-theta)));assert error<2e-6 and np.max(norm_error)<2e-6
        values.append(actual);reports[arm]=dict(zero_component_or_zero_angle_fallback_queries=np.flatnonzero(fallback).tolist(),maximum_angle_error_radians=error,maximum_standardized_norm_relative_error=float(norm_error.max()),minimum_projected_component_norm=float(qnorm.min()),mean_direction_cosine_with_full=float(np.mean(np.sum(q*full,axis=1)/(np.linalg.norm(q,axis=1)*np.linalg.norm(full,axis=1)).clip(1e-20))))
    return np.stack(values,axis=1),dict(arms=list(ARMS),queries=len(native),original_angles_radians=theta[:,0].tolist(),reports=reports,max_decomposition_error=float(np.abs(full-pop-vary).max()))

def mathematical_checks():
    rng=np.random.default_rng(1729);u=rng.normal(size=(13,1024));u-=u.mean(1,keepdims=True);g=rng.uniform(.7,1.3,(1,1024));b=rng.normal(scale=.03,size=(1,1024));native=b+g*u
    d=rng.normal(size=u.shape);d-=d.mean(1,keepdims=True);full=project(d,u);angle=np.linspace(.01,.24,len(u))[:,None]
    target=b+g*(np.cos(angle)*u+np.sin(angle)*np.linalg.norm(u,axis=1,keepdims=True)*full/np.linalg.norm(full,axis=1,keepdims=True))
    result,report=interventions(native,target,g,b,d,d.mean(0));assert result.shape==(13,2,1024)
    # A zero population direction has an explicit original-output fallback.
    z,proof=interventions(native,target,g,b,d,np.zeros(1024));assert np.array_equal(z[:,0],target.astype(np.float32)) and len(proof['reports']['population_direction']['zero_component_or_zero_angle_fallback_queries'])==13
    # A purely population direction has a zero varying component, same fallback.
    constant=np.broadcast_to(d[0],d.shape);z,proof=interventions(native,target,g,b,constant,d[0]);assert np.array_equal(z[:,1],target.astype(np.float32))
    return dict(status='PASS_EQUAL_ANGLE_NORM_DECOMPOSITION_AND_BOTH_ZERO_COMPONENT_FALLBACKS',queries=13,maximum_angle_error=max(x['maximum_angle_error_radians'] for x in report['reports'].values()),classical_formula_not_a_novelty_claim=True)
