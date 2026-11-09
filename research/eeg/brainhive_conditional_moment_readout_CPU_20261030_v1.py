"""Exact conditional-mixture moments and a fixed latent chordal Bayes action.

The EEG natural-parameter direction is not generally the conditional mean.
This action minimizes latent squared chordal loss, not image semantic loss.
"""
import numpy as np
from scipy.special import logsumexp
from brainhive_spherical_predictive_density_CPU_20261029_v1 import vmf_log_normalizer_array,vmf_mean_cosine_array


def conditional_moments(weights,means,concentrations,EEG_directions,scale):
    weights=np.asarray(weights,dtype=np.float64)
    means=np.asarray(means,dtype=np.float64)
    kappas=np.asarray(concentrations,dtype=np.float64)
    directions=np.asarray(EEG_directions,dtype=np.float64)
    assert directions.ndim==2 and directions.shape[1]==means.shape[1]
    assert weights.shape==kappas.shape==(len(means),) and np.all(weights>0)
    assert abs(weights.sum()-1)<1e-10 and scale>=0
    assert np.max(abs(np.linalg.norm(directions,axis=-1)-1))<1e-10
    assert np.max(abs(np.linalg.norm(means,axis=-1)-1))<1e-10
    dimension=means.shape[1]
    squared=np.maximum(0,kappas**2+scale**2+2*scale*(directions@means.T)*kappas)
    lengths=np.sqrt(squared)
    logits=np.log(weights)+vmf_log_normalizer_array(dimension,kappas)-vmf_log_normalizer_array(dimension,lengths)
    logZ=logsumexp(logits,axis=-1)
    probabilities=np.exp(logits-logZ[:,None])
    resultant=vmf_mean_cosine_array(dimension,lengths)
    # A_d(k)/k -> 1/d at k=0, so no division by a vanishing natural vector.
    ratio=np.divide(resultant,lengths,out=np.full_like(lengths,1./dimension),where=lengths>0)
    coefficients=probabilities*ratio
    moment=(coefficients*kappas)@means+(scale*coefficients.sum(axis=1))[:,None]*directions
    assert np.isfinite(moment).all() and np.max(np.linalg.norm(moment,axis=-1))<=1+1e-10
    return moment,dict(log_evidence=logZ,conditional_component_weights=probabilities)


def latent_chordal_action(moment):
    moment=np.asarray(moment,dtype=np.float64)
    assert moment.ndim==2 and np.isfinite(moment).all()
    norms=np.linalg.norm(moment,axis=-1)
    # If the exact moment is zero every unit direction has the same latent
    # risk; choose a declared fixed first-axis tie, never a gallery answer.
    ties=norms<=1e-12
    direction=np.zeros_like(moment);direction[ties,0]=1.
    direction[~ties]=moment[~ties]/norms[~ties,None]
    return direction,dict(moment_norm=norms,canonical_zero_moment_tie=ties)


def model_chordal_risk(moment,unit_action):
    return 2.-2.*np.einsum('ij,ij->i',moment,unit_action)
