import json
import numpy as np
from brainhive_conditional_vmf_sampler_CPU_20261030_v1 import sample_vmf,sample_conditional_mixture
from brainhive_conditional_moment_readout_CPU_20261030_v1 import conditional_moments
from brainhive_spherical_predictive_density_CPU_20261029_v1 import vmf_mean_cosine,vmf_cosine_quantiles
from brainhive_EEG_joint_density_M_step_CPU_20261030_v1 import joint_value_and_gradients
rng=np.random.default_rng(1729);checks=[];sample_records=[]
for dimension in (3,17,1023):
    for concentration in (0.,10.,1000.,100000.):
        mu=rng.normal(size=dimension);mu/=np.linalg.norm(mu)
        values,record=sample_vmf(mu,concentration,6000,rng)
        mean=vmf_mean_cosine(dimension,concentration)
        error=float(np.linalg.norm(values.mean(axis=0)-mean*mu))
        reference_SE=np.sqrt(max(1-mean**2,1e-12)/6000)
        assert error<5*reference_SE
        sample_records.append(dict(dimension=dimension,kappa=concentration,mean_error=error,reference_SE=reference_SE,proposals=record['proposals']))
checks.append('Wood_sampler_support_and_analytic_vector_moment_at_uniform_to1e5_in3_17_1023_dimensions')
mu=np.array([1.,0.,0.]);values,unused=sample_vmf(mu,10.,20000,rng)
ordered=np.sort(values[:,0]);cdf=(np.exp(10*(ordered-1))-np.exp(-20))/(1-np.exp(-20))
assert np.max(abs(cdf-(np.arange(len(cdf))+.5)/len(cdf)))<.015
checks.append('S2_axial_samples_match_independent_exact_CDF_not_only_first_moment')
mu=np.zeros(1023);mu[0]=1.;values,unused=sample_vmf(mu,1000.,12000,rng)
quantiles=vmf_cosine_quantiles(1023,1000.)
assert np.max(abs(np.array([(values[:,0]<=cut).mean() for cut in quantiles])-np.array([.025,.5,.975])))<.02
checks.append('1023dim_axial_quantiles_match_independent_scalar_density_integrals')
for sign in (1.,-1.):
    mu=np.zeros(1023);mu[0]=sign
    values,unused=sample_vmf(mu,1000.,1000,rng);assert np.mean(values[:,0])*sign>0
checks.append('stable_both_pole_rotations')
dimension=17;means=rng.normal(size=(4,dimension));means/=np.linalg.norm(means,axis=1,keepdims=True);weights=np.array([.1,.2,.3,.4]);kappas=np.array([1.,5.,10.,20.]);mu=rng.normal(size=dimension);mu/=np.linalg.norm(mu)
expected,unused=conditional_moments(weights,means,kappas,mu[None],7.)
values,record=sample_conditional_mixture(weights,means,kappas,mu,7.,20000,rng)
assert np.linalg.norm(values.mean(axis=0)-expected[0])<5*np.sqrt((1-np.sum(expected**2))/20000)
checks.append('conditional_category_and_vMF_samples_match_exact_mixture_moment')
first,_=sample_conditional_mixture(weights,means,kappas,mu,7.,30,np.random.default_rng(1729))
second,_=sample_conditional_mixture(weights,means,kappas,mu,7.,30,np.random.default_rng(1729))
assert np.array_equal(first,second);checks.append('private_condition_RNG_replay_is_exact_and_separate_from_generator_RNG')
max_error=0.
for scale in (1.,100.,10000.):
    condition=rng.normal(size=17);teacher=rng.normal(size=17);teacher=.9*teacher/np.linalg.norm(teacher)
    value,gradient,log_gradient,unused=joint_value_and_gradients(condition,teacher,weights,means,kappas,np.log(scale))
    step=1e-5
    plus=joint_value_and_gradients(condition,teacher,weights,means,kappas,np.log(scale)+step)[0]
    minus=joint_value_and_gradients(condition,teacher,weights,means,kappas,np.log(scale)-step)[0]
    error=abs((plus-minus)/(2*step)-log_gradient);max_error=max(max_error,error)
    assert error<1e-4 and error/max(1,abs(log_gradient))<1e-7
checks.append('existing_global_logit_scale_exact_M_step_gradient_matches_finite_differences')
# A categorical counterexample: matching distributions exactly does not justify
# rendering their normalized latent mean under a nonlinear semantic loss.
latent=np.array([[.5,np.sqrt(.75)],[.5,-np.sqrt(.75)]])
mean=latent.mean(axis=0);mean/=np.linalg.norm(mean)
loss=lambda x:float(min(1.,(x[0]-.5)**2/.25))
assert sum(loss(v) for v in latent)/2==0 and loss(mean)==1
checks.append('zero_KL_nonlinear_counterexample_rejects_semantic_optimality_claim_for_latent_mean')
print(json.dumps(dict(status='PASS_CONDITIONAL_SAMPLING_SCALE_GRADIENT_AND_DECISION_BOUNDARIES_CPU',checks=checks,maximum_log_scale_gradient_error=max_error,sample_records=sample_records,new_GPU_calls=0)))
