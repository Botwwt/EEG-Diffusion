import json
import math
import numpy as np
from scipy import integrate
from brainhive_contrastive_marginal_mixture_CPU_20261029_v1 import condition_mixture,mixture_log_density,joint_mixture_kl
from brainhive_spherical_predictive_density_CPU_20261029_v1 import vmf_log_normalizer

rng=np.random.default_rng(1729)
checks=[]
weights=np.array([.3,.7])
means=np.array([[1.,0,0],[-1.,0,0]])
concentrations=np.array([3.,7.])
EEG=np.array([1.,0,0])
posterior=condition_mixture(weights,means,concentrations,EEG,2.)
def prior_at(t):
    return .3*math.exp(vmf_log_normalizer(3,3)+3*t)+.7*math.exp(vmf_log_normalizer(3,7)-7*t)
evidence=2*math.pi*integrate.quad(lambda t:prior_at(t)*math.exp(2*t),-1,1,epsabs=1e-12)[0]
assert abs(math.log(evidence)-posterior['log_evidence'])<1e-12
mass=2*math.pi*integrate.quad(lambda t:prior_at(t)*math.exp(2*t-posterior['log_evidence']),-1,1,epsabs=1e-12)[0]
assert abs(mass-1)<1e-12
checks.append('mixture_EEG_evidence_and_normalization_match_independent_S2_quadrature')
zero=condition_mixture(weights,means,concentrations,EEG,0)
assert np.max(abs(zero['weights']-weights))<1e-12 and abs(zero['log_evidence'])<1e-12
assert np.max(abs(zero['means']-means))<1e-12
checks.append('zero_EEG_tilt_exactly_returns_original_nonuniform_marginal')
means=rng.normal(size=(5,17));means/=np.linalg.norm(means,axis=1,keepdims=True)
weights=np.array([.1,.2,.2,.3,.2]);concentrations=np.array([0.,1.,2.,3.,8.])
EEG=rng.normal(size=17);EEG/=np.linalg.norm(EEG)
values=rng.normal(size=(100,17));values/=np.linalg.norm(values,axis=1,keepdims=True)
posterior=condition_mixture(weights,means,concentrations,EEG,5.)
explicit=mixture_log_density(values,posterior['weights'],posterior['means'],posterior['concentrations'])
tilted=mixture_log_density(values,weights,means,concentrations)+5*(values@EEG)-posterior['log_evidence']
assert np.max(abs(explicit-tilted))<1e-12
checks.append('closed_component_weight_update_equals_normalized_density_ratio_for_all_points')
rotation=np.linalg.qr(rng.normal(size=(17,17)))[0]
rotated=condition_mixture(weights,means@rotation,concentrations,EEG@rotation,5.)
assert np.max(abs(rotated['weights']-posterior['weights']))<1e-12
assert np.max(abs(rotated['means']-posterior['means']@rotation))<1e-12
checks.append('joint_orthogonal_change_of_contrastive_coordinates_is_equivariant')
kl=joint_mixture_kl(posterior['weights'],posterior['means'],posterior['concentrations'],posterior['weights'],posterior['means'],posterior['concentrations'])
assert abs(kl)<1e-12
q_weights=posterior['weights'][::-1].copy()
assert joint_mixture_kl(q_weights,posterior['means'],posterior['concentrations'],posterior['weights'],posterior['means'],posterior['concentrations'])>0
checks.append('joint_component_direction_KL_zero_identity_and_positive_changed_probabilities')
print(json.dumps(dict(status='PASS_CLASSICAL_NONUNIFORM_MARGINAL_MIXTURE_EEG_TILT_CPU_MATH',checks=checks,new_EEG_models=0,new_images=0,new_GPU_calls=0,real_T_mixture_not_fitted=True,HIVE_input_not_admitted=True,novelty_established=False)))
