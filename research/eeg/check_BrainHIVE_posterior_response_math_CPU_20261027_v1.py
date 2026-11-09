"""Independent algebra/counterexample checks; no EEG fit or GPU use."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa
import brainhive_posterior_response_math_20261027_v1 as M


def main():
    p=argparse.ArgumentParser();p.add_argument('--output', type=Path, required=True);a=p.parse_args()
    assert not a.output.exists(); started=time.monotonic();rng=np.random.default_rng(1729)
    torch.set_num_threads(2);torch.manual_seed(1729);checks={}
    n,k=7,4;x=rng.normal(size=(n,n));s=x@x.T+np.eye(n)
    x=rng.normal(size=(k,k));r=x@x.T+np.eye(k);h=rng.normal(size=(k,n))
    m=rng.normal(size=n);b=rng.normal(size=k);e=rng.normal(size=(11,k))
    mu,cov=M.gaussian_conditional(e,m,s,h,b,r)
    precision=np.linalg.solve(s,np.eye(n))+h.T@np.linalg.solve(r,h)
    rhs=np.linalg.solve(s,m)[:,None]+h.T@np.linalg.solve(r,(e-b).T)
    assert np.allclose(mu,np.linalg.solve(precision,rhs).T,atol=1e-11)
    assert np.allclose(cov,np.linalg.solve(precision,np.eye(n)),atol=1e-11)
    checks['Gaussian_gain_and_precision_forms_agree']=True
    spectrum=M.whitened_spectrum(s,h,r);z=np.linalg.solve(spectrum['noise_factor'],(e-b-h@m).T).T
    tmp=np.zeros((len(e),n));tmp[:,:k]=z@spectrum['left']
    spectral_mean=m+(tmp*spectrum['mean_gain'])@spectrum['right'].T@spectrum['prior_factor'].T
    spectral_cov=spectrum['prior_factor']@spectrum['right']@np.diag(spectrum['remaining_variance'])@spectrum['right'].T@spectrum['prior_factor'].T
    assert np.allclose(mu,spectral_mean,atol=1e-11) and np.allclose(cov,spectral_cov,atol=1e-11)
    assert np.all(spectrum['remaining_variance'][k:]==1)
    checks['same_spectrum_controls_mean_variance_and_null_space']=True
    null_mu,null_cov=M.gaussian_conditional(e,m,s,h*0,b,r)
    assert np.array_equal(null_mu,np.broadcast_to(m,null_mu.shape)) and np.array_equal(null_cov,s)
    checks['uninformative_EEG_returns_prior_not_fabricated_precision']=True
    noisy=rng.normal(size=(11,n));alpha,sigma=.7,.8
    dynamic=M.state_conditioned_innovation(e,noisy,m,s,h,b,r,alpha,sigma)
    joint_h=np.concatenate([alpha*np.eye(n),h]);joint_r=np.zeros((n+k,n+k))
    joint_r[:n,:n]=sigma**2*np.eye(n);joint_r[n:,n:]=r
    joint_mu,joint_cov=M.gaussian_conditional(np.concatenate([noisy,e],1),m,s,joint_h,np.r_[np.zeros(n),b],joint_r)
    assert np.allclose(joint_mu,dynamic['corrected_mean'],atol=1e-11)
    assert np.allclose(joint_cov,dynamic['corrected_covariance'],atol=1e-11)
    checks['state_then_EEG_innovation_equals_joint_Bayes_solution']=True
    reverse_mu,reverse_cov=M.gaussian_conditional(noisy,mu[0],cov,alpha*np.eye(n),np.zeros(n),sigma**2*np.eye(n))
    assert np.allclose(reverse_mu[0],joint_mu[0],atol=1e-11) and np.allclose(reverse_cov,joint_cov,atol=1e-11)
    checks['joint_solution_independent_of_observation_order']=True
    close=M.state_conditioned_innovation(e,noisy,m,s,h,b,r,.999,1e-5)
    assert np.linalg.norm(close['gain'])<1e-8
    checks['already_observed_clean_state_suppresses_redundant_EEG_gain']=True
    absent=M.state_conditioned_innovation(e,noisy,m,s,h,b,r,1e-5,1.)
    assert np.allclose(absent['corrected_mean'],mu,atol=3e-4)
    checks['uninformative_image_state_reduces_to_EEG_only_solution']=True
    assert np.linalg.norm(dynamic['innovation'])>0 and np.linalg.norm(dynamic['gain'])>0
    checks['time_state_dependence_is_in_the_derived_gain_not_a_scalar_gate']=True
    # Conditional epsilon: Cov(epsilon,[z,e])=[sigma*I,0]. This is an
    # independent joint-Gaussian derivation, not a restatement of the code.
    observation_cov=joint_h@s@joint_h.T+joint_r
    epsilon_cross=np.concatenate([sigma*np.eye(n),np.zeros((n,k))],1)
    eps_direct=(np.concatenate([noisy,e],1)-np.r_[alpha*m,h@m+b])@np.linalg.solve(observation_cov,epsilon_cross.T)
    assert np.allclose(eps_direct,dynamic['conditional_epsilon'],atol=1e-11)
    checks['conditional_epsilon_equals_independent_joint_Gaussian_derivation']=True
    sm=dynamic['prior_conditional_covariance'];predicted_eeg=dynamic['prior_denoised_mean']@h.T+b
    schur=M.conditional_innovation(e,dynamic['prior_denoised_mean'],predicted_eeg,sm,sm@h.T,h@sm@h.T+r)
    assert np.allclose(schur['mean'],joint_mu,atol=1e-11) and np.allclose(schur['covariance'],joint_cov,atol=1e-11)
    checks['general_conditional_Schur_solution_equals_linear_measurement_case']=True
    no_extra=M.conditional_innovation(e,dynamic['prior_denoised_mean'],predicted_eeg,sm,np.zeros((n,k)),r)
    assert np.array_equal(no_extra['mean'],dynamic['prior_denoised_mean']) and np.array_equal(no_extra['covariance'],sm)
    checks['zero_conditional_increment_is_exact_noop_even_for_nonzero_EEG_residual']=True
    rotation,_=np.linalg.qr(rng.normal(size=(k,k)))
    rotated=M.conditional_innovation(e@rotation,dynamic['prior_denoised_mean'],predicted_eeg@rotation,sm,sm@h.T@rotation,rotation.T@(h@sm@h.T+r)@rotation)
    assert np.allclose(rotated['mean'],schur['mean'],atol=1e-11) and np.allclose(rotated['covariance'],schur['covariance'],atol=1e-11)
    checks['EEG_coordinate_rotation_preserves_inferred_image_state']=True
    assert np.linalg.eigvalsh(sm-schur['covariance']).min()>-1e-12
    checks['valid_conditional_covariance_cannot_increase_after_observation']=True
    try:M.conditional_innovation(e,dynamic['prior_denoised_mean'],predicted_eeg,sm,100*sm@h.T,r)
    except np.linalg.LinAlgError:checks['inconsistent_joint_covariance_rejected_not_ridge_repaired']=True
    else:raise AssertionError('non-PSD joint model accepted')
    q=torch.randn(2,2,5,3,dtype=torch.float64);ks=torch.randn(2,3,2,4,3,dtype=torch.float64);vs=torch.randn_like(ks)
    w=torch.tensor([[.2,.3,.5],[.4,.1,.5]],dtype=torch.float64)
    actual=M.marginal_attention(q,ks,vs,w)
    logits=torch.einsum('bhnd,bmhkd->bmhnk',q,ks)/(3**.5)
    explicit=torch.einsum('bmhnk,bmhkd,bm->bhnd',logits.softmax(-1),vs,w)
    assert torch.allclose(actual,explicit,atol=3e-7,rtol=0)
    checks['separate_four_token_normalizers_equal_explicit_finite_expectation']=True
    assert torch.equal(M.marginal_attention(q,ks[:,:1],vs[:,:1],torch.ones(2,1,dtype=q.dtype)),sdpa(q,ks[:,0],vs[:,0]))
    checks['delta_law_uses_native_SDPA_exactly_CPU']=True
    permutation=torch.tensor([2,0,1])
    assert torch.allclose(actual,M.marginal_attention(q,ks[:,permutation],vs[:,permutation],w[:,permutation]),atol=3e-7,rtol=0)
    checks['hypothesis_order_does_not_change_response']=True
    duplicated_k=torch.cat([ks[:,:1],ks],dim=1);duplicated_v=torch.cat([vs[:,:1],vs],dim=1)
    duplicated_w=torch.cat([w[:,:1]/2,w],dim=1);duplicated_w[:,1]/=2
    assert torch.allclose(actual,M.marginal_attention(q,duplicated_k,duplicated_v,duplicated_w),atol=3e-7,rtol=0)
    checks['splitting_an_atom_preserves_distribution_and_response']=True
    # Adding a constant to every logit WITHIN one hypothesis cannot change its
    # native attention. A single global M*4 softmax violates this invariance.
    q2=torch.ones(1,1,1,1,dtype=torch.float64)
    k2=torch.tensor([[[[[0.],[1.],[2.],[3.]]],[[[0.],[1.],[2.],[3.]]]]],dtype=q2.dtype)
    v2=torch.tensor([[[[[0.],[0.],[0.],[0.]]],[[[1.],[1.],[1.],[1.]]]]],dtype=q2.dtype)
    w2=torch.tensor([[.5,.5]],dtype=q2.dtype);shift=k2.clone();shift[:,1]+=10
    correct=M.marginal_attention(q2,k2,v2,w2);correct_shift=M.marginal_attention(q2,shift,v2,w2)
    wrong=sdpa(q2,shift.flatten(1,3).reshape(1,1,8,1),v2.flatten(1,3).reshape(1,1,8,1))
    assert torch.equal(correct,correct_shift) and abs(correct.item()-.5)<1e-12 and wrong.item()>.999
    checks['concatenating_all_hypotheses_is_wrong_counterexample']=dict(correct=correct.item(),wrong=wrong.item())
    # Jensen gap in a two-hypothesis, fixed-Q example, without a learned module.
    t1=torch.tensor([[[[[-3.],[0.],[0.],[0.]]],[[[3.],[0.],[0.],[0.]]]]],dtype=q2.dtype)
    v3=torch.tensor([[[[[1.],[0.],[0.],[0.]]],[[[1.],[0.],[0.],[0.]]]]],dtype=q2.dtype)
    integrated=M.marginal_attention(q2,t1,v3,w2)
    at_mean=sdpa(q2,t1.mean(1),v3.mean(1))
    gap=abs(integrated.item()-at_mean.item());assert gap>.15
    checks['mean_condition_and_mean_attention_do_not_commute']=dict(integrated=integrated.item(),at_mean=at_mean.item(),gap=gap)
    try:M.marginal_attention(q2,k2,v2,torch.tensor([[-.5,1.5]],dtype=q2.dtype))
    except AssertionError:checks['negative_quadrature_weights_rejected']=True
    else:raise AssertionError('invalid probability weights accepted')
    samples=rng.multivariate_normal(mu[0],cov,size=30000)
    checks['Gaussian_synthetic_covariance_calibration']=float(np.mean(np.square(np.linalg.solve(np.linalg.cholesky(cov),(samples-mu[0]).T))))
    assert abs(checks['Gaussian_synthetic_covariance_calibration']-1)<.015
    value=dict(status='POSTERIOR_RESPONSE_MATH_CPU_CHECKED_NOT_AN_EEG_OR_IMAGE_RESULT',checks=checks,check_count=len(checks),
               GPU_jobs=0,EEG_models_fitted=0,new_images=0,actual_native_UNet_replay=False,
               full_diffusion_posterior_proved=False,novelty_established=False,seconds=time.monotonic()-started)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value))


if __name__=='__main__':main()
