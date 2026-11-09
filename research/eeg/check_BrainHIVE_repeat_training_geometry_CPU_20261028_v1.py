"""Mathematical counterexamples for proposed repeated-EEG training objectives.

Synthetic NumPy-only checks. These do not establish any model for the real EEG,
do not read experimental arrays, and cannot authorize fitting or image claims.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np


def run():
    results=[]
    def record(name, **values):
        results.append(dict(name=name, passed=True, **values))
    # Deattenuation estimates a different risk from prediction on noisy R4.
    S=np.diag([1., .2]); N=np.diag([4., 8.]); R=4
    W_observed=S@np.linalg.inv(S+N/R); W_clean=np.eye(2)
    def risk(W):
        A=W-np.eye(2)
        return float(np.trace(A@S@A.T)+np.trace(W@N@W.T)/R)
    assert risk(W_observed)<risk(W_clean)
    record('R4_DEPLOYMENT_NOISE_TERM_IS_REAL_PREDICTION_RISK',
        observed_optimum_diagonal=W_observed.diagonal().tolist(),
        observed_risk_at_observed_optimum=risk(W_observed),
        observed_risk_at_clean_deattenuated_optimum=risk(W_clean),
        statement='Subtracting noise covariance changes the target to latent-clean-input risk; it is not a free improvement to noisy-input deployment.')
    # All deterministic signal-preserving linear estimates have this limit.
    rng=np.random.default_rng(1729)
    weights=rng.normal(size=(10000,4));weights-=(weights.sum(1)-1)[:,None]/4
    excess=(weights**2).sum(1)-.25
    assert np.all(excess>=-1e-14)
    assert np.max(np.abs(excess-((weights-.25)**2).sum(1)))<1e-13
    record('ONLY_UNIFORM_MEAN_HAS_R4_VARIANCE_WITH_FOUR_IID_REPEATS',
        identity='sum(a)=1 => sum(a*a)=1/4+sum((a-1/4)**2)',
        independent_same_variance_linear_replicates_available=False,
        assumptions='Equal conditional covariance and mutually uncorrelated repeat noise; not proved for real EEG.')
    choices=list(itertools.product(range(4),repeat=4))
    coefficients=np.array([np.bincount(choice,minlength=4)/4 for choice in choices])
    variance=(coefficients**2).sum(1);covariance=coefficients@np.full(4,.25)
    assert np.allclose(covariance,.25) and np.count_nonzero(variance==.25)==24
    assert set(np.unique(variance))=={.25,.375,.5,.625,1.}
    record('BOOTSTRAP_R4_IS_NOT_A_NEW_INDEPENDENT_EQUAL_NOISE_MEASUREMENT',
        bootstrap_draws=len(choices),exact_uniform_mean_draws=24,
        bootstrap_mean_noise_variance_factors=sorted(np.unique(variance).tolist()),
        covariance_factor_with_original_mean=float(covariance[0]),
        mean_variance_factor=float(variance.mean()))
    # Gaussianity is what makes orthogonal mean and contrasts independent.
    eta=np.array(list(itertools.product([-1.,1.],repeat=4)))
    mean=eta.mean(1);contrast=eta@np.array([1.,1.,-1.,-1.])/4
    cross=float(np.mean(mean*contrast))
    mixed=float(np.mean(mean**2*contrast**2))
    product=float(np.mean(mean**2)*np.mean(contrast**2))
    assert cross==0 and abs(mixed-product)>1e-3
    record('UNCORRELATED_MEAN_AND_REPEAT_CONTRAST_NEED_NOT_BE_INDEPENDENT',
        zero_mean_independent_Rademacher_repeats=True,covariance=cross,
        fourth_mixed_moment=mixed,product_of_second_moments=product,
        EEG_Gaussianity_or_independence_established=False)
    # Noise2Noise risk expansion. Merely observing the same stimulus twice is
    # insufficient: E[eta_B | s,Y_A]=0 is a sufficient condition for a free
    # target-noise constant in the squared risk.
    A=np.array([-1.,1.]);noisy_pair_risk=0.;clean_risk=float(np.mean(A**2))
    cross_term=float(np.mean(A*A));target_noise=float(np.mean(A**2))
    assert noisy_pair_risk==clean_risk+target_noise-2*cross_term
    record('CORRELATED_TARGET_NOISE_BREAKS_THE_NOISE2NOISE_EQUIVALENCE',
        example='s=0, Y_A=Y_B=eta, g(y)=y',pair_risk=noisy_pair_risk,
        clean_risk=clean_risk,discarded_cross_term_twice=2*cross_term,
        sufficient_condition='E[eta_B | s,Y_A]=0; covariance zero alone does not establish this for nonlinear g.')
    # The exact cross-entropy Hessian includes curvature of the logits.
    def softmax(x):
        z=np.exp(x-np.max(x));return z/z.sum()
    def ce(logits, label=0):
        z=logits-np.max(logits)
        return float(np.log(np.exp(z).sum())-z[label])
    J=np.array([.7,-.4,.2]);z=np.array([.3,-.1,.8]);p=softmax(z)
    fisher=np.diag(p)-np.outer(p,p);ggn=float(J@fisher@J)
    h=1e-4
    finite=(ce(z+h*J)-2*ce(z)+ce(z-h*J))/h**2
    assert abs(finite-ggn)<1e-7
    record('AFFINE_LOGIT_CROSS_ENTROPY_HESSIAN_EQUALS_FISHER_PULLBACK',
        generalized_Gauss_Newton=ggn,finite_difference=finite)
    nonlinear=lambda x:ce(np.array([x*x,0.]))
    finite=(nonlinear(h)-2*nonlinear(0)+nonlinear(-h))/h**2
    assert abs(finite+1)<1e-7
    record('NONLINEAR_LOGIT_CURVATURE_CANNOT_BE_DROPPED_FROM_AN_EXACT_RISK_CLAIM',
        example='logits=(x*x,0), correct_class=0, x=0',
        Fisher_pullback=0.,actual_loss_second_derivative=finite,
        exact_formula='H_x L=J_logits.T*(diag(p)-p*p.T)*J_logits+sum_k (p_k-y_k)*H_x logit_k',
        implication='A positive Fisher penalty is a local surrogate, not the full expected noisy-EEG loss.')
    # Train InfoNCE can improve without any ordering or direction improvement.
    logits=np.array([[1.,.2,.1],[.1,1.,.3],[.2,.1,1.]])
    def nce(a):return float(np.mean([ce(a*row,i) for i,row in enumerate(logits)]))
    assert nce(4)<nce(1) and np.array_equal(np.argmax(logits,1),np.argmax(4*logits,1))
    record('TRAIN_CONTRASTIVE_LOSS_ALONE_DOES_NOT_IDENTIFY_THE_CHANGED_MECHANISM',
        fixed_logits_loss_scale1=nce(1),fixed_logits_loss_scale4=nce(4),
        all_rankings_unchanged=True,real_D30_temperature_change_not_measured=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    tick=time.monotonic();checks=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve()
    result=dict(status='REPEAT_TRAINING_MATH_IDENTITIES_AND_COUNTEREXAMPLES_VERIFIED',
        source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        checks=checks,number_of_checks=len(checks),seconds=time.monotonic()-tick,
        real_EEG_or_image_data_read=False,Torch_imported=False,CUDA_initialized=False,
        new_fit_or_new_images=0,real_EEG_noise_model_established=False,
        candidate_or_image_admission=False,originality_established=False)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],checks=len(checks),seconds=result['seconds'])))


if __name__=='__main__':main()
