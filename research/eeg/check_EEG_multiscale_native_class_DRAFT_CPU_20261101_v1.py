"""Exact enumeration/normalization/gradient and cache counterexamples, synthetic."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import EEG_multiscale_native_class_DRAFT_CPU_20261101_v1 as M


def enumerate_joint(logits,rho,ell):
    pi=np.exp(logits-M.logsumexp(logits,axis=1,keepdims=True));S,K=pi.shape
    z=0.;gamma=np.zeros_like(pi);stay=np.zeros(S-1);reset=np.zeros_like(pi)
    for states in itertools.product(range(K),repeat=S):
        for switches in itertools.product((0,1),repeat=S-1):  # 0=stay, 1=refresh, including refresh-to-same-class.
            value=pi[0,states[0]]*np.exp(sum(ell[k,c] for k,c in enumerate(states)))
            for k in range(1,S):
                value*=(1-rho[k-1])*pi[k,states[k]] if switches[k-1] else rho[k-1]*(states[k]==states[k-1])
            z+=value
            for k,c in enumerate(states):gamma[k,c]+=value
            for k,switch in enumerate(switches,1):
                if switch:reset[k,states[k]]+=value
                else:stay[k-1]+=value
    return dict(log_evidence=float(np.log(z)),class_posterior=gamma/z,expected_stay=stay/z,expected_reset_class=reset/z)


def checks():
    rng=np.random.default_rng(1729);S,K=4,3;logits=rng.normal(size=(S,K));stay_logits=rng.normal(size=S-1)
    rho=M.sigmoid(stay_logits);ell=-rng.uniform(.1,3,size=(S,K));actual=M.inference(logits,rho,ell)
    explicit=enumerate_joint(logits,rho,ell);maximum=0.;checks=[]
    for key,value in explicit.items():
        error=float(np.max(np.abs(np.asarray(actual[key])-value)));assert error<1e-12;maximum=max(maximum,error)
    checks.append('all81_class_paths_and8_stay_refresh_paths_match_log_evidence_and_all_posterior_moments')
    h=1e-6;grad_error=0.
    for index in np.ndindex(logits.shape):
        plus=logits.copy();minus=logits.copy();plus[index]+=h;minus[index]-=h
        numerical=-(M.inference(plus,rho,ell)['log_evidence']-M.inference(minus,rho,ell)['log_evidence'])/(2*h)
        grad_error=max(grad_error,abs(numerical-actual['negative_log_evidence_gradient_logits'][index]))
    for index in range(S-1):
        plus=stay_logits.copy();minus=stay_logits.copy();plus[index]+=h;minus[index]-=h
        numerical=-(M.inference(logits,M.sigmoid(plus),ell)['log_evidence']-M.inference(logits,M.sigmoid(minus),ell)['log_evidence'])/(2*h)
        grad_error=max(grad_error,abs(numerical-actual['negative_log_evidence_gradient_stay_logits'][index]))
    assert grad_error<2e-8
    checks.append('all12_EEG_prior_logits_and3_stay_logits_negative_log_evidence_gradients_match_central_differences')
    logpi=logits-M.logsumexp(logits,axis=1,keepdims=True)
    sticky=M.inference(logits,np.ones(S-1),ell)
    expected=float(M.logsumexp(logpi[0]+ell.sum(0)))
    assert abs(sticky['log_evidence']-expected)<1e-12
    assert np.max(np.abs(sticky['negative_log_evidence_gradient_logits'][1:]))==0
    reset=M.inference(logits,np.zeros(S-1),ell)
    assert abs(reset['log_evidence']-float(M.logsumexp(logpi+ell,axis=1).sum()))<1e-12
    checks.append('rho1_exact_global_class_mixture_and_rho0_independent_scale_mixtures_are_explicit_limits')
    # Each emission is a proper native conditional of a binary visible token given its full history.
    params=rng.normal(size=(3,3));total=0.
    for image in itertools.product((0,1),repeat=3):
        probabilities=np.stack([M.sigmoid(params[k]+.6*sum(image[:k])) for k in range(3)])
        logprob=np.log(np.where(np.asarray(image)[:,None],probabilities,1-probabilities))
        total+=np.exp(M.inference(logits[:3],rho[:2],logprob)['log_evidence'])
    assert abs(total-1)<1e-12
    checks.append('complete_visible_image_distribution_normalizes_with_history_dependent_native_emissions')
    # A scale-level common offset may dominate raw NLL but contains no class evidence.
    offsets=np.array([-1000.,-2000.,-3000.,-4000.]);shifted=M.inference(logits,rho,ell+offsets[:,None])
    assert abs(shifted['log_evidence']-actual['log_evidence']-offsets.sum())<1e-10
    for key in ('class_posterior','expected_stay','expected_reset_class','negative_log_evidence_gradient_logits'):
        assert np.max(np.abs(shifted[key]-actual[key]))<2e-11
    checks.append('large_class_independent_late_scale_NLL_changes_evidence_but_not_class_posterior_or_EEG_gradient')
    # Mixing parallel token probabilities independently changes a single global-class mixture.
    probabilities=np.array([[.9,.8],[.1,.2]]);prior=np.array([.5,.5])
    mixture_of_products=float(prior@probabilities.prod(1));product_of_mixtures=float((prior@probabilities).prod())
    assert abs(mixture_of_products-product_of_mixtures)>.1
    checks.append('mixture_of_within_scale_token_products_is_not_product_of_token_mixtures')
    # A cache made under class a is not a cache for native fixed class b.
    previous=np.array([[.2,1.],[-.4,.1]]);condition_matrix=np.array([[.7,.4],[-.2,.9]])
    old_class=np.array([1.,0.]);new_class=np.array([0.,1.]);query=np.array([.3,-.8])
    old_keys=previous+old_class@condition_matrix;new_keys=previous+new_class@condition_matrix
    # Token-dependent conditioning is needed to make the softmax differ, as in a nonlinear native block.
    old_keys=np.tanh(old_keys);new_keys=np.tanh(new_keys)
    old_logits=query@old_keys.T;new_logits=query@new_keys.T
    assert np.max(np.abs(np.exp(old_logits-M.logsumexp(old_logits))-np.exp(new_logits-M.logsumexp(new_logits))))>.01
    checks.append('synthetic_condition_dependent_prefix_KV_cache_cannot_be_reused_after_class_change')
    # Same coarse evidence can conceal fully discriminative fine-scale evidence.
    same_prefix=np.array([[-.5,-.5],[-.5,-.5]])
    fine_left=np.vstack((same_prefix,[-.01,-6.]));fine_right=np.vstack((same_prefix,[-6.,-.01]))
    left=M.inference(np.zeros((3,2)),np.ones(2),fine_left);right=M.inference(np.zeros((3,2)),np.ones(2),fine_right)
    coarse=M.inference(np.zeros((2,2)),np.ones(1),same_prefix)
    assert np.allclose(coarse['class_posterior'],.5) and left['class_posterior'][0,0]>.99 and right['class_posterior'][0,1]>.99
    checks.append('synthetic_equal_coarse_likelihood_does_not_determine_full_scale_class_posterior')
    return dict(status='PASS_SYNTHETIC_MULTISCALE_NATIVE_CLASS_DRAFT_PROBABILITY_CHECKS',checks=checks,
        exact_enumeration_maxerror=maximum,analytical_gradient_central_difference_maxerror=grad_error,
        full_image_normalization_error=abs(total-1),classes=K,scales=S,synthetic_only=True,
        actual_EEG_or_VAR_likelihoods_tested=False,actual_native_cache_replay_done=False,
        full680_emission_bank_built=False,actual_cost_profile_done=False,new_GPU_jobs=0,new_training_updates=0,new_reconstruction_images=0,
        HMM_or_sticky_transition_or_forward_backward_claimed_original=False,publication_novelty_or_effect_established=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();value=checks()
    sources=[Path(__file__),Path(M.__file__)];value['sources']=[dict(path=str(path.resolve()),bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources]
    output=Path(a.output);assert not output.exists();output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(value))
