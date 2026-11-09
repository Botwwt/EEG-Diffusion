"""Pinned Torch CPU E/M graph checks, no real semantic teacher or CUDA."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import gammaln
from brainhive_semantic_teacher_Torch_20261029_v1 import power_rsample,power_entropy,power_mean,power_parameters,mixture_prior_log_density,F_native_coordinates,native_F_condition,semantic_teacher_loss,exact_moment_M_step
from brainhive_spherical_predictive_density_CPU_20261029_v1 import power_spherical_entropy,vmf_log_normalizer_array,vmf_log_normalizer
from brainhive_contrastive_marginal_mixture_CPU_20261029_v1 import condition_mixture,mixture_log_density
from brainhive_EEG_mixture_moment_objective_CPU_20261029_v1 import EEG_cross_entropy_value_and_gradient

parser=argparse.ArgumentParser();parser.add_argument('--F-geometry',type=Path);args=parser.parse_args()
torch.set_num_threads(2);torch.manual_seed(1729);rng=np.random.default_rng(1729);checks=[]
for dimension in (17,1023):
    for sign in (-1.,1.):
        raw=torch.zeros(dimension,dtype=torch.float64,requires_grad=True)
        with torch.no_grad():raw[0]=sign
        concentration=torch.tensor(1000.,dtype=torch.float64,requires_grad=True)
        sample=power_rsample(raw,concentration,torch.Size([128]))
        assert torch.isfinite(sample).all() and float((sample.norm(dim=-1)-1).abs().max())<1e-12
        loss=sample[...,1].mean()+power_entropy(raw,concentration)*.001
        g=torch.autograd.grad(loss,(raw,concentration));assert all(torch.isfinite(value).all() for value in g)
checks.append('intrinsic17_and1023_sphere_sample_support_and_both_pole_gradients')
raw=torch.randn(17,dtype=torch.float64);concentration=torch.tensor(15.,dtype=torch.float64,requires_grad=True)
sample=power_rsample(raw,concentration,torch.Size([50000]));mean,kappa,unused,unused2=power_parameters(raw,concentration)
expected=power_mean(raw,concentration)
assert float((sample.mean(dim=0)-expected).abs().max())<.008
axial=(sample*mean).sum(dim=-1).mean();gradient=torch.autograd.grad(axial,concentration)[0]
exact=16/(kappa+16).square()*torch.sigmoid(concentration)
assert abs(float(gradient-exact))<2e-4
checks.append('implicit_Beta_rsample_mean_and_concentration_gradient_match_analytic_moment')
maximum_entropy_error=0.
for dimension in (17,1023):
    for value in (-1000.,10.,1000.,100000.):
        raw=torch.randn(dimension,dtype=torch.float64);parameter=torch.tensor(value,dtype=torch.float64)
        k=float(torch.nn.functional.softplus(parameter));entropy=float(power_entropy(raw,parameter));error=abs(entropy-power_spherical_entropy(dimension,k));maximum_entropy_error=max(maximum_entropy_error,error);assert error<1e-7
checks.append('Torch_entropy_agrees_with_independent_SciPy_at_uniform_and_high_concentration')
dimension=17;means=rng.normal(size=(4,dimension));means/=np.linalg.norm(means,axis=1,keepdims=True);weights=np.array([.1,.2,.3,.4]);kappas=np.array([1.,5.,10.,20.]);evidence=rng.normal(size=dimension);strength=np.linalg.norm(evidence)
posterior=condition_mixture(weights,means,kappas,evidence/strength,strength)
v=rng.normal(size=(8,dimension));v/=np.linalg.norm(v,axis=1,keepdims=True)
prior=mixture_prior_log_density(torch.tensor(v),torch.tensor(weights),torch.tensor(means),torch.tensor(kappas),torch.tensor(vmf_log_normalizer_array(dimension,kappas)),torch.tensor(evidence),torch.tensor(posterior['log_evidence'],dtype=torch.float64))
reference=mixture_log_density(v,posterior['weights'],posterior['means'],posterior['concentrations'])
assert np.max(abs(prior.numpy()-reference))<1e-12
checks.append('differentiable_Torch_prior_equals_exact_normalized_conditional_mixture')
try:
    mixture_prior_log_density(torch.tensor(v),torch.tensor(weights),torch.tensor(means),torch.tensor(kappas),torch.tensor(vmf_log_normalizer_array(dimension,kappas)),torch.tensor(evidence),torch.tensor(posterior['log_evidence'],dtype=torch.float32))
except ValueError:
    pass
else:
    raise AssertionError('Low precision probability constants must be rejected')
checks.append('reject_probability_constant_precision_loss_before_model_arithmetic')
weight=torch.linspace(.8,1.2,1024,dtype=torch.float64);bias=torch.linspace(-.01,.01,1024,dtype=torch.float64);pre=torch.randn(7,1024,dtype=torch.float64);pre-=pre.mean(dim=-1,keepdim=True);radii=pre.norm(dim=-1);condition=bias+weight*pre
coordinates=F_native_coordinates(condition,weight,bias);restored=native_F_condition(coordinates,weight,bias,radii)
assert float((restored-condition).abs().max())<1e-12
assert float((((restored-bias)/weight).mean(dim=-1)).abs().max())<1e-12
checks.append('intrinsic_F_affine_zero_mean_and_per_row_radius_roundtrip')
raw=torch.randn(17,dtype=torch.float64,requires_grad=True);concentration=torch.tensor(1000.,dtype=torch.float64,requires_grad=True);sample=power_rsample(raw,concentration)
frozen=torch.randn(17,8,dtype=torch.float64,requires_grad=False);feature=torch.tanh(sample@frozen);target=torch.randn(8,dtype=torch.float64)
prior=mixture_prior_log_density(sample,torch.tensor(weights),torch.tensor(means),torch.tensor(kappas),torch.tensor(vmf_log_normalizer_array(17,kappas)),torch.tensor(evidence),torch.tensor(posterior['log_evidence'],dtype=torch.float64))
loss,terms=semantic_teacher_loss(feature,target,8.,vmf_log_normalizer(8,8.),power_entropy(raw,concentration),prior)
gradients=torch.autograd.grad(loss,(raw,concentration));assert all(torch.isfinite(g).all() for g in gradients) and float(gradients[0].norm())>0 and frozen.grad is None
checks.append('normalized_auxiliary_semantic_E_step_graph_reaches_teacher_with_frozen_toy_decoder')
condition=torch.randn(3,17,dtype=torch.float64,requires_grad=True);teacher=torch.randn(3,17,dtype=torch.float64);teacher=torch.nn.functional.normalize(teacher,dim=-1).detach().requires_grad_(True)
loss=exact_moment_M_step(condition,teacher,weights,means,kappas,5.);loss.sum().backward();assert teacher.grad is None
for row in range(3):
    value,g,unused=EEG_cross_entropy_value_and_gradient(condition[row].detach().numpy(),teacher[row].detach().numpy(),weights,means,kappas,5.)
    assert abs(float(loss[row])-value)<1e-12 and np.max(abs(condition.grad[row].numpy()-g))<1e-12
checks.append('exact_M_step_injects_correct_first_derivative_and_detaches_teacher')
real_geometry=None
if args.F_geometry:
    data=np.load(args.F_geometry,allow_pickle=False);weight=torch.from_numpy(data['weight']);bias=torch.from_numpy(data['bias']);reference=float(np.median(data['native_pre_affine_radius']))
    latent=torch.randn(8,1023,dtype=torch.float64,requires_grad=True);condition=native_F_condition(latent,weight,bias,reference);back=F_native_coordinates(condition,weight,bias)
    assert float((back.norm(dim=-1)-reference).abs().max())<1e-10
    assert float((torch.nn.functional.normalize(back,dim=-1)-torch.nn.functional.normalize(latent,dim=-1)).abs().max())<1e-12
    real_geometry=dict(actual_F_weight_exactly_one=bool(np.array_equal(data['weight'],np.ones(1024))),actual_F_bias_nonzero=bool(np.any(data['bias']!=0)),T_only_reference_pre_affine_radius=reference,T_count=len(data['T_array_rows']))
    checks.append('actual_F_weights_nonzero_bias_and_T_radius_bound_in_intrinsic_roundtrip')
    raw_condition=torch.randn(2,1024,dtype=torch.float64,requires_grad=True)
    native_means=rng.normal(size=(4,1023));native_means/=np.linalg.norm(native_means,axis=1,keepdims=True)
    native_teacher=torch.nn.functional.normalize(torch.randn(2,1023,dtype=torch.float64),dim=-1)*.7
    native_kappas=np.array([600.,800.,1000.,1200.])
    coordinates=F_native_coordinates(raw_condition,weight,bias)
    native_loss=exact_moment_M_step(coordinates,native_teacher,weights,native_means,native_kappas,120.)
    native_loss.sum().backward();assert torch.isfinite(raw_condition.grad).all() and float(raw_condition.grad.norm())>0
    native_errors=[]
    for coordinate in (0,511,1023):
        plus=raw_condition.detach().clone();minus=raw_condition.detach().clone();plus[0,coordinate]+=1e-4;minus[0,coordinate]-=1e-4
        lp=exact_moment_M_step(F_native_coordinates(plus,weight,bias),native_teacher,weights,native_means,native_kappas,120.)[0]
        lm=exact_moment_M_step(F_native_coordinates(minus,weight,bias),native_teacher,weights,native_means,native_kappas,120.)[0]
        native_errors.append(abs(float((lp-lm)/2e-4-raw_condition.grad[0,coordinate])))
    assert max(native_errors)<1e-7
    real_geometry['native_1024_to_intrinsic1023_M_step_finite_difference_error']=max(native_errors)
    checks.append('EEG_output_gradient_through_actual_F_affine_and_intrinsic_basis_matches_finite_differences')
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_TORCH_NORMALIZED_TEACHER_E_M_AND_NATIVE_F_COORDINATES_CPU',checks=checks,Torch_version=torch.__version__,maximum_entropy_error=maximum_entropy_error,implicit_sample_gradient_error=abs(float(gradient-exact)),actual_F_geometry=real_geometry,CUDA_initialized=False,new_GPU_calls=0,actual_G_semantic_teacher_or_EEG_fit=False)))
