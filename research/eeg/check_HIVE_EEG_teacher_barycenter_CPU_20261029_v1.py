import json
import numpy as np
from brainhive_contrastive_marginal_mixture_CPU_20261029_v1 import condition_mixture,mixture_log_density
from brainhive_EEG_teacher_barycenter_CPU_20261029_v1 import EEG_prior_KL_barycenter

rng=np.random.default_rng(1729);checks=[]
means=rng.normal(size=(4,17));means/=np.linalg.norm(means,axis=1,keepdims=True)
weights=np.array([.1,.2,.3,.4]);kappa=np.array([1.,2.,3.,8.]);conditions=rng.normal(size=(10,17));scales=rng.uniform(1,10,size=10)
q=rng.normal(size=(100,17));q/=np.linalg.norm(q,axis=1,keepdims=True)
model,info=EEG_prior_KL_barycenter(conditions,scales,weights,means,kappa)
individual=[]
for c,scale in zip(conditions,scales):
    m=condition_mixture(weights,means,kappa,c/np.linalg.norm(c),scale)
    individual.append(mixture_log_density(q,m['weights'],m['means'],m['concentrations']))
logbar=mixture_log_density(q,model['weights'],model['means'],model['concentrations'])
assert np.max(abs(logbar-np.mean(individual,axis=0)-info['q_independent_Jensen_gap']))<1e-12
checks.append('average_fixed_teacher_KL_equals_barycenter_KL_plus_q_independent_Jensen_gap')
single,unused=EEG_prior_KL_barycenter(conditions[:1],scales[:1],weights,means,kappa)
reference=condition_mixture(weights,means,kappa,conditions[0]/np.linalg.norm(conditions[0]),scales[0])
assert np.max(abs(single['weights']-reference['weights']))<1e-12
checks.append('one_subject_exactly_reduces_to_individual_conditional_prior')
zero,unused=EEG_prior_KL_barycenter(conditions,np.zeros(10),weights,means,kappa)
assert np.max(abs(zero['weights']-weights))<1e-12
assert np.max(abs(zero['concentrations']-kappa))<1e-12
checks.append('no_EEG_evidence_exactly_returns_visual_marginal')
permuted,unused=EEG_prior_KL_barycenter(conditions[::-1],scales[::-1],weights,means,kappa)
assert np.max(abs(model['weights']-permuted['weights']))<1e-12
checks.append('subject_order_invariant_no_product_of_independent_subjects_assumption')
print(json.dumps(dict(status='PASS_TRAINING_ONLY_EEG_KL_BARYCENTER_CPU',checks=checks,real_EEG_teacher_not_fitted=True,new_GPU_calls=0,novelty_established=False)))
