"""Check an exact objective identity on closed-run features; not model gradients."""
import json
from pathlib import Path
import numpy as np
L=Path(__file__).resolve().parent;D=L/'audit/HIVE_paired_semantic_closed_transmission_20261101_v1'
x=np.load(D/'observed_arrays.npz',allow_pickle=False);rng=np.random.default_rng(1729);tau=.07;maximum_identity=0.;maximum_gradient=0.;maximum_FD=0.;cases=0
def unit(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)
def components(h,v):
    g=unit(h);v=unit(v);mean=(g[0]+g[1])/2;target_mean=(v[0]+v[1])/2;m=np.dot(g[0]-g[1],v[0]-v[1]);pair=tau*np.logaddexp(0,-m/tau)
    original=1-np.mean(np.sum(g*v,axis=-1))+pair
    common=1-np.dot(mean,target_mean);differential=-m/4+pair
    return np.array([original,common,differential]),g,v,m
for arm in ('joint_correct','EEG_only_correct','P_only_correct','joint_swapped_targets','joint_absolute_only'):
    for block in ('first512','last512'):
        gs=x[f'features__{arm}__{block}'];vs=x[f'targets__{arm}__{block}']
        for h,v in zip(gs,vs):
            if arm=='joint_swapped_targets':v=v[::-1]
            values,g,v,m=components(h,v);maximum_identity=max(maximum_identity,abs(values[0]-sum(values[1:])))
            sigma=np.exp(-np.logaddexp(0,m/tau));dv=v[0]-v[1];vbar=v.mean(0)
            direct=np.stack([-v[0]/2-sigma*dv,-v[1]/2+sigma*dv]);common=np.stack([-vbar/2,-vbar/2]);diff=np.stack([-(.25+sigma)*dv,(.25+sigma)*dv])
            maximum_gradient=max(maximum_gradient,float(np.max(np.abs(direct-common-diff))))
            # Unit-normalization derivative is included; this checks h, not a
            # fictional unconstrained unit-feature perturbation or EEG weights.
            grads=[direct,common,diff];direction=rng.normal(size=h.shape);direction/=np.linalg.norm(direction)
            predicted=[]
            for grad in grads:
                grad=(grad-g*np.sum(grad*g,axis=-1,keepdims=True))/np.linalg.norm(h,axis=-1,keepdims=True);predicted.append(np.sum(grad*direction))
            eps=1e-5;observed=(components(h+eps*direction,v)[0]-components(h-eps*direction,v)[0])/(2*eps)
            maximum_FD=max(maximum_FD,float(np.max(np.abs(np.array(predicted)-observed))));cases+=1
assert cases==160 and maximum_identity<1e-12 and maximum_gradient<1e-12 and maximum_FD<1e-8
value=dict(status='COMPLETE_CLASSICAL_MEAN_DIFFERENCE_OBJECTIVE_IDENTITY_CPU_CHECK',actual_closed_feature_pairs=cases,tau=tau,
    exact_objective='L = 1 - dot(g_bar,v_bar) - m/4 + tau*softplus(-m/tau); m=dot(g_i-g_j,v_i-v_j)',
    exact_parameter_chain_rule='grad_theta L = -J_bar.T v_bar - (1/4 + sigmoid(-m/tau)) (J_i-J_j).T (v_i-v_j); J_i=d(normalized_generated_CLIP_i)/dtheta',
    conceptual_constant_output_family='If g_i=g_j for every EEG and J_i=J_j, differential gradient is zero while common-target gradient can remain. Equal outputs ALONE do not imply equal parameter Jacobians.',
    maximum_loss_identity_error=maximum_identity,maximum_feature_gradient_identity_error=maximum_gradient,maximum_normalization_aware_directional_FD_error=maximum_FD,
    actual_EEG_or_P_parameter_Jacobians_NOT_measured=True,gradient_conflict_or_dominance_NOT_established=True,absolute_only_scope='The identity check evaluates the paired formula on its recorded features as a mathematical counterfactual, not its actual absolute-only training loss.',
    interpretation='Observed input/token/semantic contraction plus this identity motivates testing preservation of EEG stimulus sensitivity; it is not proof of unique cause or a new published principle.',
    novelty_established=False,new_GPU_jobs=0,new_model_updates=0,new_images=0,original_STOP_unchanged=True,live_noO_study_unchanged=True)
(D/'mean_difference_derivation_and_checks.json').write_bytes((json.dumps(value,indent=2)+'\n').encode());print(json.dumps(value))
