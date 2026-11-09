"""Exact first derivative of a conditional mixture cross-entropy.

The teacher moment must be held fixed during this M-step. A point teacher gives
a finite NLL gradient, not a finite Dirac-to-continuous KL. This file does not
implement a differentiable HIVE decoder or claim a complete EEG method.
"""
import numpy as np
from brainhive_contrastive_marginal_mixture_CPU_20261029_v1 import condition_mixture
from brainhive_spherical_predictive_density_CPU_20261029_v1 import vmf_mean_cosine_array


def conditional_mean(weights,means,concentrations,EEG_direction,scale):
    model=condition_mixture(weights,means,concentrations,EEG_direction,scale)
    resultant=vmf_mean_cosine_array(means.shape[1],model['concentrations'])
    mean=np.sum((model['weights']*resultant)[:,None]*model['means'],axis=0)
    return mean,model


def EEG_cross_entropy_value_and_gradient(condition,teacher_mean,weights,means,concentrations,scale):
    condition=np.asarray(condition,dtype=np.float64)
    teacher_mean=np.asarray(teacher_mean,dtype=np.float64)
    radius=np.linalg.norm(condition)
    assert radius>1e-12 and teacher_mean.shape==condition.shape
    assert np.linalg.norm(teacher_mean)<=1+1e-10
    direction=condition/radius
    posterior_mean,model=conditional_mean(weights,means,concentrations,direction,scale)
    difference=posterior_mean-teacher_mean
    gradient=scale/radius*(difference-direction*np.dot(direction,difference))
    # E_q[-log p0(v)] and E_q[log q(v)] are constants for this fixed-q M-step.
    # Return only the theta-dependent cross-entropy, never call it total KL.
    value=model['log_evidence']-scale*np.dot(direction,teacher_mean)
    return float(value),gradient,dict(posterior_mean=posterior_mean,teacher_mean=teacher_mean,theta_independent_terms_omitted=True)
