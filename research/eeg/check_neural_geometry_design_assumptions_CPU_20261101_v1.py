"""Counterexamples constrain the next research design, not current efficacy gates."""
import json
from pathlib import Path
import numpy as np
L=Path(__file__).resolve().parent;O=L/'audit/EEG_geometry_design_assumptions_20261101_v1';O.mkdir(exist_ok=False)
eps=1e-5;n=8;z=np.zeros((n,n+1));z[:,0]=np.sqrt(1-eps**2);z[:,1:]=eps*np.eye(n)
pred=np.argmax(z[:,1:],axis=1);assert np.array_equal(pred,np.arange(n))
centred=z-z.mean(0);variance=float(np.mean(np.sum(centred**2,axis=1)));expected=eps**2*(1-1/n)
assert abs(variance-expected)<1e-20
rng=np.random.default_rng(1729);labels=np.repeat(np.arange(8),1000);noise=rng.normal(size=(len(labels),9));noise/=np.linalg.norm(noise,axis=1,keepdims=True)
noise_variance=float(np.mean(np.sum((noise-noise.mean(0))**2,axis=1)));noise_accuracy=float(np.mean(np.argmax(noise[:,1:],axis=1)==labels))
assert noise_variance>.99 and .09<noise_accuracy<.16
# Exact enumeration, not a stochastic significance test. Independent zero-mean
# epsilon produces zero cross-distance; an unmodelled shared pattern survives.
independent=np.array([-1.,1.]);zero=[];shared=[]
for a in independent:
    for b in independent:zero.append(a*b);shared.append((1+a)*(1+b))
assert np.mean(zero)==0. and np.mean(shared)==1.
result=dict(status='COMPLETE_CPU_DESIGN_COUNTEREXAMPLES_NO_NEW_METHOD',
    low_variance_identifiable=dict(samples=8,epsilon=eps,centred_angular_variance=variance,analytic_expected_variance=expected,analytic_coordinate_readout_accuracy=1.,meaning='Small angular variance alone does not prove loss of stimulus information; no claim about learnability or noise robustness.'),
    high_variance_noninformative=dict(samples=8000,centred_angular_variance=noise_variance,fixed_coordinate_readout_accuracy=noise_accuracy,analytic_X_Y_independence_by_construction=True,meaning='Increasing variance can add label-independent noise and cannot substitute for held-out stimulus decoding.'),
    cross_repeat_model=dict(independent_noise_only_exact_cross_product=float(np.mean(zero)),shared_artifact_exact_cross_product=float(np.mean(shared)),
        equation='If x_ir=s_i+a_i+epsilon_ir, cross-repeat contrasts retain Delta a as well as Delta s. An iid epsilon model does not separate a shared artifact from neural signal.',
        classical_crossnobis_unbiasedness_not_disproved=True,independence_assumption_required=True,biological_specificity_not_established=True),
    relevant_actual_evidence=['audit/HIVE_paired_semantic_closed_transmission_20261101_v1/local_transmission_and_log_geometry.json','audit/relative_latency_20261028_v1/actual/candidate/measurement/summary.json'],
    actual_R4_independent_session_metadata_available=False,actual_R4_iid_noise_established=False,
    next_design_constraints=['Do not promote a variance-floor regularizer as an information-preserving innovation.','Classical crossnobis/RSA or repeat-covariance weighting alone is not new; D08 already used repeat-noise reliability eigenmodes.','Any new EEG-target mechanism needs held-concept stimulus-specific evidence and improvement beyond the existing c/ordinary control.','Current noO fit and its image/secondary gates remain unchanged.'],
    novelty_established=False,new_training_updates=0,new_images=0,new_GPU_jobs=0)
(O/'actual_counterexamples_and_design_constraints.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
