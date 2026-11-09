"""Exact classical vMF-mixture tilt in raw contrastive score coordinates.

This is an auxiliary probability model, not an admitted HIVE generation input.
Its unit directions must not bypass native affine/radius or ordinary contracts.
"""
import numpy as np
from scipy.special import logsumexp
from brainhive_spherical_predictive_density_CPU_20261029_v1 import (
    vmf_log_normalizer_array, vmf_mean_cosine_array,
)


def condition_mixture(weights, means, concentrations, EEG_direction, scale):
    weights=np.asarray(weights,dtype=np.float64)
    means=np.asarray(means,dtype=np.float64)
    concentrations=np.asarray(concentrations,dtype=np.float64)
    EEG_direction=np.asarray(EEG_direction,dtype=np.float64)
    assert means.ndim==2 and weights.shape==concentrations.shape==(len(means),)
    assert np.all(weights>0) and abs(weights.sum()-1)<1e-10
    assert np.all(concentrations>=0) and scale>=0
    assert np.max(abs(np.linalg.norm(means,axis=1)-1))<1e-10
    assert abs(np.linalg.norm(EEG_direction)-1)<1e-10
    dimension=means.shape[1]
    natural=concentrations[:,None]*means+scale*EEG_direction
    lengths=np.linalg.norm(natural,axis=1)
    directions=np.divide(natural,lengths[:,None],out=np.tile(EEG_direction,(len(means),1)),where=lengths[:,None]>0)
    unnormalized=np.log(weights)+vmf_log_normalizer_array(dimension,concentrations)-vmf_log_normalizer_array(dimension,lengths)
    log_evidence=float(logsumexp(unnormalized))
    return dict(weights=np.exp(unnormalized-log_evidence),means=directions,concentrations=lengths,log_evidence=log_evidence)


def mixture_log_density(values, weights, means, concentrations):
    values=np.asarray(values,dtype=np.float64)
    means=np.asarray(means,dtype=np.float64)
    return logsumexp(np.log(weights)+vmf_log_normalizer_array(means.shape[1],concentrations)+(values@means.T)*concentrations,axis=-1)


def vmf_kl(q_means,q_concentrations,p_means,p_concentrations):
    dimension=q_means.shape[1]
    return vmf_log_normalizer_array(dimension,q_concentrations)-vmf_log_normalizer_array(dimension,p_concentrations)+vmf_mean_cosine_array(dimension,q_concentrations)*(q_concentrations-p_concentrations*np.sum(q_means*p_means,axis=1))


def joint_mixture_kl(q_weights,q_means,q_concentrations,p_weights,p_means,p_concentrations):
    """KL for joint component+direction, not KL between marginal mixtures."""
    return float(np.sum(q_weights*(np.log(q_weights/p_weights)+vmf_kl(q_means,q_concentrations,p_means,p_concentrations))))
