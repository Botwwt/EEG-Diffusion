import json
import numpy as np
from brainhive_EEG_mixture_moment_objective_CPU_20261029_v1 import conditional_mean,EEG_cross_entropy_value_and_gradient

rng=np.random.default_rng(1729);checks=[];dimension=1024
means=rng.normal(size=(8,dimension));means/=np.linalg.norm(means,axis=1,keepdims=True)
weights=rng.uniform(size=8);weights/=weights.sum();concentrations=rng.uniform(600,1600,size=8)
condition=rng.normal(size=dimension);teacher=rng.normal(size=dimension);teacher/=np.linalg.norm(teacher)
value,gradient,info=EEG_cross_entropy_value_and_gradient(condition,teacher,weights,means,concentrations,180.)
errors=[]
for coordinate in (0,17,129,511,767,1023):
    h=1e-4;plus=condition.copy();minus=condition.copy();plus[coordinate]+=h;minus[coordinate]-=h
    fplus=EEG_cross_entropy_value_and_gradient(plus,teacher,weights,means,concentrations,180.)[0]
    fminus=EEG_cross_entropy_value_and_gradient(minus,teacher,weights,means,concentrations,180.)[0]
    errors.append(abs((fplus-fminus)/(2*h)-gradient[coordinate]))
assert max(errors)<1e-7
checks.append('full1024_native_score_coordinate_gradient_matches_independent_finite_differences')
assert abs(np.dot(gradient,condition))<1e-10
rescaled=EEG_cross_entropy_value_and_gradient(condition*3,teacher,weights,means,concentrations,180.)
assert abs(rescaled[0]-value)<1e-10 and np.max(abs(rescaled[1]*3-gradient))<1e-10
checks.append('normalized_score_chain_rule_has_zero_radial_derivative_and_inverse_norm_scaling')
mean,model=conditional_mean(weights,means,concentrations,condition/np.linalg.norm(condition),180.)
matched=EEG_cross_entropy_value_and_gradient(condition,mean,weights,means,concentrations,180.)
assert np.max(abs(matched[1]))<1e-12
checks.append('teacher_posterior_moment_match_gives_exact_zero_M_step_gradient')
new=EEG_cross_entropy_value_and_gradient(condition-.001*gradient,teacher,weights,means,concentrations,180.)
assert new[0]<value
checks.append('one_small_exact_gradient_step_reduces_synthetic_fixed_teacher_cross_entropy')
print(json.dumps(dict(status='PASS_EXACT_EEG_CONDITIONAL_MIXTURE_MOMENT_GRADIENT_CPU',checks=checks,maximum_finite_difference_error=max(errors),actual_EEG_training=False,actual_HIVE_teacher_fitted=False,new_GPU_calls=0,novelty_established=False)))
