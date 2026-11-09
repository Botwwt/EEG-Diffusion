"""Synthetic limits of covariance-classification and Gaussian preprocessing.

No real EEG, weights or images are opened. Exact classical formulas are checked,
including a counterexample where smoothing an already optimal decoder hurts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np


def phi(x):
    return np.exp(-np.square(x)/2)/math.sqrt(2*math.pi)


def Phi(x):
    return .5*(1+np.vectorize(math.erf)(np.asarray(x)/math.sqrt(2)))


def gelu_normal_mean(mu, variance):
    scale=math.sqrt(1+variance)
    return float(mu*Phi(mu/scale)+variance/scale*phi(mu/scale))


def mixture_risks(nodes):
    # Fixed normal-noise grid avoids slow Hermite convergence at the mixture
    # posterior's narrow transition. Omitted |N|>12 MSE is below2e-32.
    z=np.linspace(-12.,12.,nodes);w=phi(z)*(z[1]-z[0]);w[[0,-1]]*=.5
    risks=np.zeros(2)
    scales=np.array([.5,2.])
    for truth in (-1.,1.):
        for sigma in scales:
            y=truth+sigma*z
            positive=np.sum(phi((y[:,None]-1)/scales)/scales,axis=1)
            negative=np.sum(phi((y[:,None]+1)/scales)/scales,axis=1)
            without=(positive-negative)/(positive+negative)
            with_scale=np.tanh(y/sigma**2)
            risks += .25*np.array([np.sum(w*(truth-without)**2),np.sum(w*(truth-with_scale)**2)])
    return risks


def run():
    low=mixture_risks(10001);high=mixture_risks(20001)
    assert np.max(np.abs(low-high))<1e-9
    assert high[1]<high[0]-.01
    results=[dict(name='CLASS_INDEPENDENT_NOISE_SCALE_CAN_HAVE_CONDITIONAL_DECODING_VALUE',
        passed=True,model='S uniform{-1,+1}; sigma independent uniform{.5,2}; Y=S+sigma*N(0,1)',
        sigma_only_class_accuracy=.5,I_sigma_S_zero_by_factorized_joint_distribution=True,
        Bayes_MSE_without_sigma=float(high[0]),Bayes_MSE_with_sigma=float(high[1]),
        normal_grid10001_20001_max_difference=float(np.max(np.abs(low-high))),
        normal_noise_domain=[-12.,12.],omitted_normal_tail_MSE_upper_bound=2e-32,
        identity='E Var(S|Y)=E Var(S|Y,sigma)+E Var(E[S|Y,sigma]|Y)',
        statement='A near-chance covariance-only classifier does not rule out conditional value of measurement variability.')]
    nodes,weights=np.polynomial.hermite.hermgauss(128)
    maximum=0.
    for mu in (-4.,-1.,0.,2.):
        for variance in (0.,.1,1.,4.):
            values=mu+math.sqrt(2*variance)*nodes
            numeric=float(np.sum(weights*values*Phi(values))/math.sqrt(math.pi))
            analytic=gelu_normal_mean(mu,variance)
            maximum=max(maximum,abs(numeric-analytic))
    assert maximum<1e-10
    assert gelu_normal_mean(0.,0.)==0.
    assert abs(gelu_normal_mean(0.,1.)-1/math.sqrt(4*math.pi))<1e-15
    results.append(dict(name='GAUSSIAN_GELU_MEAN_IDENTITY_NOT_WHOLE_NETWORK_EXACTNESS',passed=True,
        formula='E[Z*Phi(Z)]=mu*Phi(mu/sqrt(1+v))+v/sqrt(1+v)*phi(mu/sqrt(1+v))',
        quadrature_max_abs=maximum,mean_at_mu0_variance1=gelu_normal_mean(0.,1.),
        ordinary_GELU_of_mu0=0.,assumptions='One Gaussian scalar preactivation; exact erf GELU, not approximate-tanh GELU.',
        does_not_prove_exact_Gaussian_residual_or_LayerNorm_propagation=True))
    S=np.array([[1.,.2],[.2,.5]]);N=np.array([[.6,.1],[.1,.3]]);mu=np.array([.1,-.2]);y=np.array([.7,.4])
    K=S@np.linalg.inv(S+N)
    posterior_mu=mu+K@(y-mu);posterior_cov=S-K@S
    assert np.allclose(posterior_cov,posterior_cov.T) and np.linalg.eigvalsh(posterior_cov).min()>0
    zero_mu=mu+S@np.linalg.inv(S)@(y-mu)
    assert np.allclose(zero_mu,y)
    results.append(dict(name='KNOWN_LINEAR_GAUSSIAN_MEASUREMENT_POSTERIOR',passed=True,
        posterior_mean=posterior_mu.tolist(),posterior_covariance=posterior_cov.tolist(),
        zero_noise_recovers_observation=True,
        real_role_specific_signal_or_noise_covariances_not_fitted=True,
        does_not_establish_real_EEG_Gaussianity_or_repeat_noise_independence=True))
    # S~N(0,1), Y=S+N, optimal direct decoder f(Y)=Y/2. Inserting
    # E[S|Y]=Y/2 before this already calibrated f produces Y/4.
    original=(1-.5)**2+.5**2
    twice=(1-.25)**2+.25**2
    assert original==.5 and twice==.625 and twice>original
    results.append(dict(name='DENOSING_AN_ALREADY_OPTIMAL_NOISY_INPUT_DECODER_CAN_DOUBLE_CORRECT',passed=True,
        model='S,N independent N(0,1); Y=S+N; trained direct f(y)=y/2',
        original_prediction_risk=original,post_denoising_f_of_E_S_given_Y_risk=twice,
        statement='A latent-clean posterior estimate and a pretrained noisy-input decoder need not compose into an optimal task decoder.'))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();started=time.monotonic();checks=run();a.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='MEASUREMENT_CONDITIONING_COUNTEREXAMPLES_AND_SCALAR_IDENTITIES_VERIFIED',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=checks,seconds=time.monotonic()-started,
        real_EEG_or_model_or_images_read=False,Torch_imported=False,CUDA_initialized=False,
        original_math_or_architecture_claim=False,new_candidate_admission=False)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],checks=len(checks),seconds=result['seconds'])))


if __name__=='__main__':main()
