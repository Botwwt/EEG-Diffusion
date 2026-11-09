"""EM/moment/normalization coverage checks with independently sampled S2."""
import json
import math
import numpy as np
from scipy.special import logsumexp
from brainhive_spherical_predictive_density_CPU_20261029_v1 import fit_vmf_concentration
from brainhive_contrastive_marginal_mixture_CPU_20261029_v1 import condition_mixture,mixture_log_density
from brainhive_marginal_mixture_fit_CPU_20261029_v1 import fit_tied_vmf_mixture,conditional_parameters,fit_conditional_tilt,component_coverage

rng=np.random.default_rng(1729)
checks=[]
def draw_s2(kappa,count):
    u=rng.uniform(size=count)
    t=np.log(np.exp(-kappa)+u*(np.exp(kappa)-np.exp(-kappa)))/kappa
    phi=rng.uniform(0,2*np.pi,size=count)
    return np.stack((t,np.sqrt(1-t*t)*np.cos(phi),np.sqrt(1-t*t)*np.sin(phi)),axis=1)
data=np.concatenate((draw_s2(8,500),-draw_s2(8,500)))
model=fit_tied_vmf_mixture(data,components=2,iterations=20)
assert np.all(np.diff([row['mean_logscore'] for row in model['history']])>=-1e-7)
assert np.min(abs(model['means'][:,0]))>.95
checks.append('actual_EM_logscore_monotone_and_recovers_two_separated_S2_components')
single=fit_tied_vmf_mixture(data[:500],components=1,iterations=3)
mean=data[:500].mean(axis=0)
assert np.max(abs(single['means'][0]-mean/np.linalg.norm(mean)))<1e-10
assert abs(single['concentrations'][0]-fit_vmf_concentration(3,float(np.linalg.norm(mean))))<1e-8
checks.append('one_component_EM_equals_independent_closed_direction_and_vMF_MLE')
weights=np.array([.3,.7]);means=np.array([[1.,0,0],[-1.,0,0]]);kappa=np.array([3.,7.]);E=np.array([1.,0,0])
posterior=condition_mixture(weights,means,kappa,E,2.)
labels=rng.choice(2,size=20000,p=posterior['weights'])
sample=draw_s2(5,20000);sample[labels==1]*=-1
cosE=np.tile([1.,-1.],(len(sample),1))
cosF=sample[:,0,None]*np.array([1.,-1.])
coverage,quantiles,details=component_coverage(weights,kappa,cosE,cosF,sample[:,0],2.,3)
assert abs(coverage.mean()-.95)<.008
assert np.max(abs(details['responsibilities'].sum(axis=1)-1))<1e-12
lengths,pw,logz=conditional_parameters(weights,kappa,cosE,2.,3)
assert np.max(abs(pw-posterior['weights']))<1e-12
checks.append('responsibility_integrated_component_PIT_has_nominal_coverage_under_true_mixture')
dirs=rng.normal(size=(100,3));dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
cosE=dirs@means.T
target=np.full(100,.5)
alpha,report=fit_conditional_tilt(weights,kappa,cosE,target,3)
assert 0<alpha<4096 and abs(report['derivative_at_solution'])<1e-7
def loss(a):
    unused,unused2,z=conditional_parameters(weights,kappa,cosE,a,3)
    return float(np.mean(z-a*target))
assert loss(alpha)<=min(loss(a) for a in np.linspace(0,max(alpha*2,10),1001))+1e-8
assert abs((loss(alpha+1e-4)-loss(alpha-1e-4))/2e-4)<1e-7
checks.append('conditional_scalar_MLE_matches_independent_objective_grid_and_finite_difference')
print(json.dumps(dict(status='PASS_VMF_MIXTURE_EM_AND_LATENT_COMPONENT_COVERAGE_CPU',checks=checks,real_EEG_result=False,new_GPU_calls=0)))
