"""DRAFT copy-or-refresh latent class chain; classical HMM, no EEG/VAR admission.

Emission ell[k,c] must be the sum of ALL true token log probabilities at scale k
under a frozen native fixed-c full-prefix kernel. Changing c invalidates a KV
cache made under another class. This file implements probability arithmetic
only, not a generator, EEG model, experiment protocol or novelty claim.
"""
import numpy as np


def logsumexp(value,axis=None,keepdims=False):
    maximum=np.max(value,axis=axis,keepdims=True)
    result=maximum+np.log(np.exp(value-maximum).sum(axis=axis,keepdims=True))
    return result if keepdims else np.squeeze(result,axis=axis)


def sigmoid(logits):return np.exp(-np.logaddexp(0.,-np.asarray(logits,np.float64)))


def inference(logits,stay_probability,emission_log_probability):
    logits=np.asarray(logits,np.float64);rho=np.asarray(stay_probability,np.float64)
    ell=np.asarray(emission_log_probability,np.float64)
    assert logits.ndim==2 and ell.shape==logits.shape and logits.shape[0]>=1 and logits.shape[1]>=2
    S,K=logits.shape
    assert rho.shape==(S-1,) and np.isfinite(logits).all() and np.isfinite(ell).all()
    assert np.isfinite(rho).all() and np.all((rho>=0)&(rho<=1)) and np.all(ell<=0)
    logpi=logits-logsumexp(logits,axis=1,keepdims=True);pi=np.exp(logpi)
    with np.errstate(divide='ignore'):logstay=np.log(rho);logreset=np.log1p(-rho)
    alpha=np.empty_like(ell);normalizers=np.empty(S)
    first=logpi[0]+ell[0];normalizers[0]=logsumexp(first);alpha[0]=first-normalizers[0]
    for k in range(1,S):
        predicted=np.logaddexp(logstay[k-1]+alpha[k-1],logreset[k-1]+logpi[k])
        value=predicted+ell[k];normalizers[k]=logsumexp(value);alpha[k]=value-normalizers[k]
    beta=np.zeros_like(ell)
    for k in range(S-1,0,-1):
        following=ell[k]+beta[k]
        reset=logreset[k-1]+logsumexp(logpi[k]+following)
        beta[k-1]=np.logaddexp(logstay[k-1]+following,reset)-normalizers[k]
    posterior=np.exp(alpha+beta-logsumexp(alpha+beta,axis=1,keepdims=True))
    reset_class=np.zeros_like(ell);stay=np.zeros(S-1)
    gradient_logits=np.zeros_like(ell);gradient_logits[0]=pi[0]-posterior[0]
    for k in range(1,S):
        following=ell[k]+beta[k]-normalizers[k]
        reset_class[k]=np.exp(logreset[k-1]+logpi[k]+following)
        stay[k-1]=np.exp(logstay[k-1]+alpha[k-1]+following).sum()
        assert abs(stay[k-1]+reset_class[k].sum()-1)<1e-10
        gradient_logits[k]=reset_class[k].sum()*pi[k]-reset_class[k]
    return dict(log_evidence=float(normalizers.sum()),class_posterior=posterior,
        expected_stay=stay,expected_reset_class=reset_class,
        negative_log_evidence_gradient_logits=gradient_logits,
        negative_log_evidence_gradient_stay_logits=rho-stay,
        scale_log_normalizers=normalizers,forward_log_alpha=alpha,backward_log_beta=beta,
        arithmetic='FP64 scaled log-space O(scales*classes); emission computation not included')
