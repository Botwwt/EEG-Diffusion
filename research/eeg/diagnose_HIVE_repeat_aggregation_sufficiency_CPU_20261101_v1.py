"""Classical correlated-measurement posterior audit, not a new EEG efficacy test."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np

def posterior(prior_mean,prior_cov,H,noise,observations):
    prior_precision=np.linalg.inv(prior_cov)
    precision=prior_precision+H.T@np.linalg.solve(noise,H)
    covariance=np.linalg.inv(precision)
    mean=covariance@(prior_precision@prior_mean+H.T@np.linalg.solve(noise,observations))
    return mean,covariance

def run():
    started=time.monotonic();rng=np.random.default_rng(1729)
    def spd(n):
        a=rng.normal(size=(n,n));return a@a.T+.5*np.eye(n)
    H=rng.normal(size=(4,3));V=spd(3);Sigma=spd(4);C=spd(4);prior=rng.normal(size=3)
    rows=[]
    for repeats in (1,2,4,80):
        e=rng.normal(size=(repeats,4));all_H=np.tile(H,(repeats,1))
        all_noise=np.kron(np.eye(repeats),Sigma)+np.kron(np.ones((repeats,repeats)),C)
        full=posterior(prior,V,all_H,all_noise,e.reshape(-1))
        averaged=posterior(prior,V,H,C+Sigma/repeats,e.mean(0))
        error=max(float(np.max(np.abs(a-b))) for a,b in zip(full,averaged))
        assert error<2e-12
        zero_sum=rng.normal(size=e.shape);zero_sum-=zero_sum.mean(0)
        other=posterior(prior,V,all_H,all_noise,(e+zero_sum).reshape(-1))
        counterfactual_error=max(float(np.max(np.abs(a-b))) for a,b in zip(full,other))
        assert counterfactual_error<2e-12
        # Multiplication of nominal single-trial likelihoods neglects shared u.
        naive=posterior(prior,V,H,(C+Sigma)/repeats,e.mean(0))
        eig=float(np.linalg.eigvalsh(averaged[1]-naive[1]).min())
        assert eig>=-2e-12
        rows.append(dict(repetitions=repeats,full_block_vs_mean_posterior_maxerror=error,
                         same_mean_zero_sum_perturbation_posterior_maxerror=counterfactual_error,
                         correct_minus_naive_posterior_covariance_min_eigenvalue=eig))
    # Known unequal noise permits extra information in individual trials. This
    # is a mathematical counterexample, not measured trial quality in THINGS.
    prior_mean=np.zeros(1);prior_cov=np.eye(1);values=np.array([0.,10.])
    weighted=posterior(prior_mean,prior_cov,np.ones((2,1)),np.diag([.01,100.]),values)
    mean_only=posterior(prior_mean,prior_cov,np.ones((1,1)),np.array([[(.01+100.)/4]]),np.array([5.]))
    assert abs(float(weighted[0][0]))<.002 and abs(float(mean_only[0][0]))>.1
    local=Path(__file__).resolve().parent
    reports=['reports/BRAINHIVE_FAST_ITERATION_AND_TRIALSET_RESULTS_20261027_v1.md',
             'reports/BRAINHIVE_EEG_REPEAT_RESPONSE_INVESTIGATION_20261028_v1.md',
             'reports/BRAINHIVE_EEG_COVARIANCE_IDENTIFIABILITY_20261028_v1.md']
    refs=[]
    for name in reports:
        raw=(local/name).read_bytes();refs.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    return dict(status='CLASSICAL_MODEL_SUFFICIENCY_AUDIT_NO_NEW_METHOD_ADMITTED',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        working_model='e_r=H z+u+epsilon_r; z~N(m0,V),u~N(0,C),epsilon_r iid N(0,Sigma); all mutually independent',
        exact_solution=dict(V_R='(V^-1+H.T@(C+Sigma/R)^-1@H)^-1',
                            m_R='V_R@(V^-1@m0+H.T@(C+Sigma/R)^-1@mean(e))'),
        implication='With shared fixed H and homoscedastic exchangeable Gaussian noise, the mean is sufficient; merely replacing mean pooling by a Gaussian product cannot recover extra stimulus evidence.',
        shared_noise_limit='As R grows, C remains. Multiplying single-trial likelihoods as independent uses (C+Sigma)/R and overstates precision when C is nonzero.',
        exact_matrix_checks=rows,
        heteroscedastic_counterexample=dict(known_trial_noise_variances=[.01,100.],
            exact_weighted_posterior_mean=float(weighted[0][0]),mean_only_posterior_mean=float(mean_only[0][0]),
            not_EEG_observation_or_performance=True),
        actual_EEG_shared_noise_or_trial_quality_estimated=False,
        source_scope_limit='Four exported repeats lack verified session IDs. Neither iid noise nor identifiable sample-specific precision is established by these algebraic checks.',
        existing_EEG_evidence=refs,
        primary_neighbours=[
            dict(url='https://openreview.net/pdf?id=ufZN2-aehFa',title='Bayesian Context Aggregation for Neural Processes',
                 implication='Bayesian Gaussian aggregation is existing methodology, not a novel component.'),
            dict(url='https://proceedings.mlr.press/v216/tailor23a/tailor23a.pdf',title='Exploiting Inferential Structure in Neural Processes',
                 implication='Mixture and robust aggregation also have established prior work.'),
            dict(url='https://arxiv.org/abs/2608.16360',title='Contrastive Learning with Variational Regularization for Multi-Session EEG-to-Speech Decoding',
                 implication='Repeated-EEG contrastive consistency plus variational regularization has a direct cross-task neighbour; speech findings do not prove image gains.')],
        decision='Do not launch generic repeated-trial Gaussian/PoE or repeat-consistency as a newly justified architecture. Require real T-only evidence of useful heterogeneity or non-Gaussian residual structure beyond the mean plus matched image gains before admission. Current full1654 study unchanged.',
        novelty_claimed=False,new_GPU_jobs=0,new_model_updates=0,new_images=0,new_R80_results_opened=0,
        seconds=time.monotonic()-started)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();value=run();assert not args.receipt.exists();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    raw=Path(__file__).read_bytes();value['source']=dict(path=Path(__file__).name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    args.receipt.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=value['status'],seconds=value['seconds'],matrix_checks=len(value['exact_matrix_checks']),new_GPU_jobs=0)))
