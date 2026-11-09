"""Small probability identities and an actual preterminal refusal, not EEG evidence."""
import itertools
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import diagnose_VAR_full_target_sampling_gradient_CPU_20261101_v2 as D


def run():
    groups=[]
    theta=np.array([.2,-.1,.4]);ell=np.log([.8,.2,.5]);q=np.array([.5,.3,.2])
    state=D.setup(theta,ell,q);ids=np.array([[0,1,1,2],[2,0,2,0]])
    bank=D.fixed_bank(state,ids)
    direct=[]
    for row in ids:
        weights=state['pi'][row]*np.exp(ell[row])/q[row]
        p=np.bincount(row,weights=weights,minlength=3)/weights.sum()
        direct.append(p)
    assert np.allclose(bank['posterior'],direct,atol=1e-14,rtol=0)
    groups.append('INDEPENDENT_UNNORMALIZED_PROBABILITY_AND_DUPLICATE_COUNT_REFERENCE')
    h=1e-6;maximum=0.
    for j in range(3):
        direction=np.eye(3)[j]*h
        left=D.fixed_bank(D.setup(theta+direction,ell,q),ids)['centered_log_estimate']
        right=D.fixed_bank(D.setup(theta-direction,ell,q),ids)['centered_log_estimate']
        fd=(left-right)/(2*h)
        maximum=max(maximum,float(abs(fd-(bank['posterior'][:,j]-state['pi'][j])).max()))
    assert maximum < 1e-8
    groups.append('FIXED_BANK_LOG_OBJECTIVE_PARAMETER_LOGIT_GRADIENT_FINITE_DIFFERENCE')
    all_one=D.fixed_bank(state,np.arange(3)[:,None])
    assert np.allclose(q @ (all_one['posterior']-state['pi']),q-state['pi'],atol=1e-14,rtol=0)
    assert abs(D.exact_one_sample(state)['expected_likelihood_ratio']-1)<1e-13
    groups.append('EXACT_M1_UNBIASED_LIKELIHOOD_BUT_PROPOSAL_ONLY_EXPECTED_LOG_GRADIENT')
    pi=np.array([.5,.5]);proposal=np.array([.8,.2])
    flat=D.setup(np.zeros(2),np.log([.16,.16]),proposal)
    expected_kl=float(np.dot(proposal,np.log(proposal/pi)))
    value=D.exact_one_sample(flat)
    assert abs(value['exact_log_evidence_minus_expected_log_estimate']-expected_kl)<1e-14
    assert np.array_equal(flat['positive_log_likelihood_gradient'],np.zeros(2))
    assert np.allclose(proposal-flat['pi'],[.3,-.3],atol=1e-14,rtol=0)
    groups.append('FLAT_FULL_LIKELIHOOD_ZERO_TRUE_GRADIENT_AND_NONZERO_M1_SURROGATE')
    shifted=D.setup(theta,ell-1e6,q);other=D.fixed_bank(shifted,ids)
    assert np.allclose(bank['posterior'],other['posterior'],atol=1e-10,rtol=0)
    assert abs(shifted['log_evidence']-state['log_evidence']+1e6)<1e-9
    groups.append('COMMON_FULL_LIKELIHOOD_OFFSET_GRADIENT_AND_EVIDENCE_INVARIANCE')
    banks=np.array(list(itertools.product(range(3),repeat=2)))
    probabilities=np.prod(q[banks],axis=1)
    all_two=D.fixed_bank(state,banks)
    assert abs(np.dot(probabilities,np.exp(all_two['centered_log_estimate']-state['centered_log_evidence']))-1)<1e-13
    expected_gradient=probabilities @ all_two['posterior']-state['pi']
    fd=np.zeros(3)
    for j in range(3):
        direction=np.eye(3)[j]*h
        a=D.fixed_bank(D.setup(theta+direction,ell,q),banks)['centered_log_estimate']
        b=D.fixed_bank(D.setup(theta-direction,ell,q),banks)['centered_log_estimate']
        fd[j]=np.dot(probabilities,a-b)/(2*h)
    assert np.allclose(fd,expected_gradient,atol=1e-8,rtol=0)
    groups.append('ALL_NINE_M2_BANKS_EXACT_EXPECTATION_AND_LOG_OBJECTIVE_GRADIENT')
    for bad in (np.array([.5,.5,0.]),np.array([.5,.4,.2])):
        try:D.setup(theta,ell,bad)
        except AssertionError:pass
        else:raise AssertionError('Invalid proposal accepted')
    groups.append('NO_ZERO_SUPPORT_OR_UNNORMALIZED_PROPOSAL')
    with tempfile.TemporaryDirectory(prefix='native_gradient_diagnostic_checks_') as temp:
        temp=Path(temp);missing=temp/'missing';destination=temp/'must_not_exist'
        refusal=D.run(missing,destination)
        assert refusal['result_payloads_read']==0 and not destination.exists()
        incomplete=temp/'incomplete';incomplete.mkdir()
        refusal=D.run(incomplete,destination)
        assert refusal['result_payloads_read']==0 and not destination.exists()
    groups.append('ACTUAL_MISSING_ARCHIVE_AND_MISSING_INDEPENDENT_AUDIT_NO_OUTPUT_REFUSALS')
    sampled=D.sampled_statistics(state,4,128,np.random.SeedSequence([1729]),deadline=time.monotonic()+30)
    rng=np.random.default_rng(np.random.SeedSequence([1729]));draws=[]
    for begin in range(0,128,64):draws.append(D.fixed_bank(state,rng.choice(3,size=(64,4),p=q))['posterior'])
    p=np.concatenate(draws)
    assert abs(sampled['mean_gradient_L2_distance_to_exact']-np.linalg.norm(p.mean(0)-state['posterior']))<1e-14
    assert abs(sampled['mean_gradient_MC_noise_RMS']-np.sqrt(p.var(0,ddof=1).sum()/128))<1e-13
    groups.append('INDEPENDENT_REPLICATE_VARIANCE_AND_MEAN_GRADIENT_REFERENCE')
    return dict(status='PASS_LOCAL_SYNTHETIC_PROBABILITY_AND_PRETERMINAL_GUARD_CHECKS_ONLY',
        groups=groups,maximum_fixed_bank_gradient_FD_error=maximum,
        native_pretrained_likelihoods_read=0,EEG_models_or_images_read=0,
        real_native_sampling_bias_tested=False,new_GPU_jobs=0,new_optimizer_updates=0)


if __name__=='__main__':print(json.dumps(run(),indent=2,allow_nan=False))
