"""Classical minimizer of average KL(q||p_s) for EEG tilted priors.

This averages KL penalties; it is not a claim that different subjects' EEG are
conditionally independent, nor a claim about a true multi-brain posterior.
"""
import numpy as np
from brainhive_contrastive_marginal_mixture_CPU_20261029_v1 import condition_mixture


def EEG_prior_KL_barycenter(conditions,scales,weights,means,concentrations):
    conditions=np.asarray(conditions,dtype=np.float64)
    scales=np.asarray(scales,dtype=np.float64)
    norms=np.linalg.norm(conditions,axis=1)
    assert conditions.ndim==2 and scales.shape==(len(conditions),) and np.all(norms>1e-12) and np.all(scales>=0)
    unit=conditions/norms[:,None]
    evidence=np.mean(scales[:,None]*unit,axis=0)
    strength=float(np.linalg.norm(evidence))
    direction=evidence/strength if strength>1e-12 else unit[0]
    model=condition_mixture(weights,means,concentrations,direction,strength)
    individual=[condition_mixture(weights,means,concentrations,mu,float(scale))['log_evidence'] for mu,scale in zip(unit,scales)]
    gap=float(np.mean(individual)-model['log_evidence'])
    assert gap>=-1e-10
    return model,dict(effective_direction=direction,effective_strength=strength,average_individual_log_evidence=float(np.mean(individual)),barycenter_log_evidence=model['log_evidence'],q_independent_Jensen_gap=gap)
