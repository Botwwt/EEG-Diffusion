import json
import numpy as np
from scipy.special import logsumexp
from numpy.polynomial.legendre import leggauss
from brainhive_conditional_moment_readout_CPU_20261030_v1 import conditional_moments,latent_chordal_action,model_chordal_risk
from brainhive_contrastive_marginal_mixture_CPU_20261029_v1 import condition_mixture
from brainhive_spherical_predictive_density_CPU_20261029_v1 import vmf_mean_cosine_array,vmf_log_normalizer_array
rng=np.random.default_rng(1729);checks=[]
for dimension in (3,17,1023):
    means=rng.normal(size=(4,dimension));means/=np.linalg.norm(means,axis=-1,keepdims=True)
    directions=rng.normal(size=(12,dimension));directions/=np.linalg.norm(directions,axis=-1,keepdims=True)
    weights=np.array([.1,.2,.3,.4]);kappas=np.array([0.,1.,5.,10.]);scale=2.3
    moment,information=conditional_moments(weights,means,kappas,directions,scale)
    for j,mu in enumerate(directions):
        posterior=condition_mixture(weights,means,kappas,mu,scale)
        reference=(posterior['weights'][:,None]*vmf_mean_cosine_array(dimension,posterior['concentrations'])[:,None]*posterior['means']).sum(axis=0)
        assert np.max(abs(moment[j]-reference))<1e-12
    action,data=latent_chordal_action(moment)
    improvement=model_chordal_risk(moment,directions)-model_chordal_risk(moment,action)
    assert improvement.min()>-1e-12
checks+=['batched_moment_matches_independent_explicit_component_updates_in3_17_1023_dimensions','posterior_mean_action_has_no_greater_model_latent_chordal_risk_than_evidence_direction']
dimension=3;means=np.eye(3);weights=np.array([.2,.3,.5]);kappas=np.array([1.,3.,5.]);mu=np.array([[1.,-2.,3.]]);mu/=np.linalg.norm(mu,axis=-1,keepdims=True)
moment,info=conditional_moments(weights,means,kappas,mu,2.)
z,wz=leggauss(96);phi=np.arange(256)*2*np.pi/256
grid=np.stack(np.broadcast_arrays(np.sqrt(1-z[:,None]**2)*np.cos(phi),np.sqrt(1-z[:,None]**2)*np.sin(phi),z[:,None]),axis=-1)
logp0=logsumexp(np.log(weights)+vmf_log_normalizer_array(3,kappas)+(grid@means.T)*kappas,axis=-1)
integrand=np.exp(logp0+2*(grid@mu[0])-info['log_evidence'][0])
mass=np.sum(integrand*wz[:,None])*(2*np.pi/256)
quadrature=np.einsum('ab,abk,a->k',integrand,grid,wz)*(2*np.pi/256)
assert abs(mass-1)<1e-12 and np.max(abs(quadrature-moment[0]))<1e-12
checks.append('independent_S2_surface_quadrature_confirms_normalized_density_and_conditional_mean')
zero,_=conditional_moments(weights,means,kappas,mu,0.)
population=(weights[:,None]*vmf_mean_cosine_array(3,kappas)[:,None]*means).sum(axis=0)
assert np.max(abs(zero[0]-population))<1e-12
checks.append('zero_EEG_evidence_returns_visual_population_moment_not_raw_EEG_direction')
isotropic,_=conditional_moments(np.array([1.]),np.array([[1.,0.,0.]]),np.array([0.]),mu,4.)
action,_=latent_chordal_action(isotropic);assert np.max(abs(action-mu))<1e-12
checks.append('uniform_population_is_the_special_case_where_evidence_and_mean_directions_coincide')
opposite,_=conditional_moments(np.array([1.]),np.array([[1.,0.,0.]]),np.array([4.]),np.array([[-1.,0.,0.]]),4.)
assert np.max(abs(opposite))<1e-12
tie,tiedata=latent_chordal_action(opposite);assert np.array_equal(tie,np.array([[1.,0.,0.]])) and tiedata['canonical_zero_moment_tie'][0]
checks.append('cancelled_natural_parameter_has_finite_uniform_moment_and_explicit_canonical_tie')
print(json.dumps(dict(status='PASS_EXACT_CONDITIONAL_MOMENT_AND_LATENT_BAYES_READOUT_CPU',checks=checks,quadrature_mass_error=abs(mass-1),quadrature_moment_error=float(np.max(abs(quadrature-moment[0]))),new_GPU_calls=0)))
