"""Classical Noise2Noise/cycle validity checks; SYNTHETIC, not EEG or novelty."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def symmetric_quadrature(dimension):
    # Exact degree-two moments of a standard Gaussian; no Monte Carlo tolerance.
    z=np.concatenate((np.eye(dimension),-np.eye(dimension)))*np.sqrt(dimension)
    assert np.allclose(z.mean(0),0,atol=1e-15)
    assert np.allclose(z.T@z/len(z),np.eye(dimension),atol=2e-15)
    return z


def positive_matrix(rng,d):
    a=rng.normal(size=(d,d));return a@a.T/d+np.eye(d)*.3


def weighted_risk(prediction,target,precision):
    error=prediction-target
    return float(np.einsum('ni,ij,nj->',error,precision,error)/len(error))


def points(signal_cov,shared_cov,noise_cov,k,inclusive):
    d=len(signal_cov);z=symmetric_quadrature(4*d)
    def root(matrix):
        vals,vec=np.linalg.eigh(matrix);assert vals.min()>-1e-12
        return vec@np.diag(np.sqrt(np.maximum(vals,0)))
    signal=z[:,:d]@root(signal_cov).T;shared=z[:,d:2*d]@root(shared_cov).T
    first=z[:,2*d:3*d]@root(noise_cov).T;second=z[:,3*d:]@root(noise_cov).T
    target=signal+shared+first
    query=signal+shared+((first+np.sqrt(k-1)*second)/k if inclusive else second/np.sqrt(k))
    return query,target,signal,target-signal


def checks():
    rng=np.random.default_rng(1729);d=7;k=3;S=positive_matrix(rng,d);N=positive_matrix(rng,d)
    W=positive_matrix(rng,d);zero=np.zeros((d,d));Q=positive_matrix(rng,d)*.4;ident=np.eye(d)
    names=[];maximum=0.
    # The pointwise identity has no independence or zero-mean assumption.
    query,target,signal,noise=points(S,Q,N,k,False);B=rng.normal(size=(d,d))*.2;pred=query@B.T
    observed=weighted_risk(pred,target,W);clean=weighted_risk(pred,signal,W)
    noise_risk=weighted_risk(noise,np.zeros_like(noise),W)
    cross=float(np.einsum('ni,ij,nj->',pred-signal,W,noise)/len(noise))
    error=abs(observed-(clean+noise_risk-2*cross));assert error<1e-12;maximum=max(maximum,error)
    names.append('pointwise_weighted_cycle_risk_includes_explicit_target_noise_cross_term')
    # Same three-repeat input count, different relation to the target repeat.
    a,t,s,n=points(S,zero,N,k,True);cov=a.T@a/len(a);cross_cov=t.T@a/len(a)
    inclusive=np.linalg.solve(cov.T,cross_cov.T).T
    assert np.max(np.abs(inclusive-ident))<1e-12
    names.append('inclusive_target_cycle_linear_optimum_is_identity_even_with_true_signal')
    a,t,s,n=points(S,zero,N,k,False);cov=S+N/k
    disjoint=np.linalg.solve(cov.T,S.T).T
    empirical=np.linalg.solve((a.T@a/len(a)).T,(t.T@a/len(a)).T).T
    error=float(np.max(np.abs(disjoint-empirical)));assert error<1e-12;maximum=max(maximum,error)
    for candidate in (ident,zero,disjoint,B):
        p=a@candidate.T
        error=abs(weighted_risk(p,t,W)-weighted_risk(p,s,W)-np.trace(W@N))
        assert error<2e-12;maximum=max(maximum,error)
    names.append('disjoint_conditionally_independent_repeats_have_oracle_risk_plus_constant_and_Wiener_optimum')
    a,t,s,n=points(zero,zero,N,k,False)
    no_signal=np.linalg.solve((a.T@a/len(a)).T,(t.T@a/len(a)).T).T
    assert np.max(np.abs(no_signal))<1e-12
    a,t,s,n=points(zero,zero,N,k,True)
    noise_identity=np.linalg.solve((a.T@a/len(a)).T,(t.T@a/len(a)).T).T
    assert np.max(np.abs(noise_identity-ident))<1e-12
    names.append('zero_signal_counterexample_inclusive_cycle_rewards_noise_copy_disjoint_cycle_does_not')
    a,t,s,n=points(zero,Q,N,k,False);cov=Q+N/k
    shared_predictor=np.linalg.solve(cov.T,Q.T).T
    assert np.linalg.norm(shared_predictor)>0 and weighted_risk(a@shared_predictor.T,s,W)>0
    empirical=np.linalg.solve((a.T@a/len(a)).T,(t.T@a/len(a)).T).T
    error=float(np.max(np.abs(shared_predictor-empirical)));assert error<1e-12;maximum=max(maximum,error)
    names.append('disjoint_indices_with_shared_noise_still_reward_nuisance_and_do_not_identify_clean_signal')
    # Centering R iid residuals itself creates negative off-diagonal covariance.
    R=4;z=symmetric_quadrature(R*d);root=np.linalg.cholesky(N)
    repeats=np.stack([z[:,j*d:(j+1)*d]@root.T for j in range(R)],axis=1)
    residual=repeats-repeats.mean(1,keepdims=True)
    for r in range(R):
        for q in range(R):
            expected=N*((1 if r==q else 0)-1/R)
            actual=residual[:,r].T@residual[:,q]/len(z)
            error=float(np.max(np.abs(actual-expected)));assert error<1e-12;maximum=max(maximum,error)
    names.append('same_R4_centering_creates_minus_noise_covariance_over4_even_when_repeats_independent')
    # A forward observer does not constrain image changes in its blind subspace.
    observer=np.array([[1.,0.]])
    image_good=np.array([.4,.5]);image_bad=np.array([.4,500.])
    assert np.array_equal(observer@image_good,observer@image_bad) and np.linalg.norm(image_good-image_bad)>100
    names.append('low_neural_cycle_error_does_not_control_image_content_in_observer_blind_subspace')
    # Joint latent cycles have a trivial zero-code solution unless anchored by other objectives.
    codes=np.zeros((9,4));arbitrary_images=np.tile(np.array([1.,-5.]),(9,1));encoded_images=np.zeros_like(codes)
    assert np.square(codes-encoded_images).sum()==0 and np.unique(arbitrary_images,axis=0).shape[0]==1
    names.append('latent_cycle_alone_allows_joint_zero_code_collapse_not_a_claim_about_combined_published_losses')
    return dict(status='PASS_SYNTHETIC_DISJOINT_REPEAT_CYCLE_VALIDITY_CHECKS',checks=names,maximum_exact_moment_error=maximum,
        dimensions=d,total_physical_repeats=4,input_repeats=3,matched_input_repeat_count=True,
        classical_formulas_not_novel=True,actual_EEG_noise_independence_tested=False,
        actual_frozen_forward_observer_tested=False,new_EEG_models=0,new_training_updates=0,new_GPU_jobs=0,new_reconstruction_images=0,
        published_method_failure_or_our_past_failure_cause_inferred=False,new_method_or_SOTA_or_CCF_A_confirmed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    value=checks();source=Path(__file__);raw=source.read_bytes()
    value['source']=dict(path=str(source.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    output=Path(a.output);assert not output.exists();output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(value))
