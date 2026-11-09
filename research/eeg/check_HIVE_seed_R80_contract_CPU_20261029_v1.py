"""Actual new behavior: nonunit calibrated strengths, identity and seed guards."""
import copy,json
import numpy as np
from brainhive_seed_repro_R80_contract_20261029_v1 import GRID,validate_replicate_state,bounded_numpy64
rng=np.random.default_rng(1729);n=7;d=1024;h=256;u=rng.normal(size=(n,d));u-=u.mean(1,keepdims=True);u*=np.sqrt(d)/np.linalg.norm(u,axis=1,keepdims=True);g=rng.uniform(.7,1.3,(n,d));b=rng.normal(0,.05,(n,d));c=b+g*u
w={'trunk.0.weight':rng.normal(0,.03,(h,d)),'trunk.0.bias':rng.normal(0,.01,h),'direction.weight':rng.normal(0,.02,(d,h)),'direction.bias':rng.normal(0,.01,d)};checks=[]
for alpha in GRID:
    changed=bounded_numpy64(c,g,b,w,alpha);v=(changed-b)/g
    assert np.max(abs(np.linalg.norm(v,axis=1)-np.linalg.norm(u,axis=1)))<1e-10
    assert np.max(abs(v.mean(1)))<1e-12
    angle=np.arccos(np.clip(np.sum(u*v,axis=1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)),-1,1));assert np.max(angle)<=np.arctan(.25*alpha)+1e-7
    if alpha==0:assert np.array_equal(changed,c)
checks.append('all5actual_strengths_apply_inside_sphere_preserve_norm_mean_and_bound_angle')
half=bounded_numpy64(c,g,b,w,.5);full=bounded_numpy64(c,g,b,w,1.);assert np.max(abs(half-(c+full)/2))>1e-5;checks.append('half_strength_is_not_linear_interpolation_outside_host_sphere')
for alpha in GRID:
    errors=[2.,2.,2.,2.,2.];errors[GRID.index(alpha)]=1.
    state=dict(seed=2027,arm='ordinary',updates=24807,seen_queries=396900,passes=3,protocol_sha256='training_spec',calibration=dict(kind='Dfit_only_global_strength_calibration',strengths=GRID,Dfit_token_errors=errors,selected_strength=alpha,selected_without_V_or_test_images=True));spec=dict(training_seed=2027,ordinary_strength=alpha,training_protocol=dict(sha256='training_spec'));assert validate_replicate_state(state,spec)==alpha
checks.append('all5_Dfit_selected_strengths_admitted_without_forcing_one')
state['calibration']['Dfit_token_errors']=[1.,1.,1.,1.,1.];state['calibration']['selected_strength']=0.;spec['ordinary_strength']=0.;assert validate_replicate_state(state,spec)==0.;checks.append('Dfit_ties_choose_smaller_strength')
mutations=[('wrong_training_seed',lambda s:s.update(seed=3407)),('partial_not_final',lambda s:s.update(updates=128)),('wrong_exposure_count',lambda s:s.update(seen_queries=2048)),('wrong_training_protocol',lambda s:s.update(protocol_sha256='other')),('image_selected_strength',lambda s:s['calibration'].update(selected_strength=1.)),('nonfinite_calibration',lambda s:s['calibration'].update(Dfit_token_errors=[float('nan')]*5)),('wrong_arm',lambda s:s.update(arm='risk'))]
for name,mutate in mutations:
    x=copy.deepcopy(state);mutate(x)
    try:validate_replicate_state(x,spec)
    except AssertionError:checks.append('reject_'+name)
    else:raise AssertionError('Accepted invalid replicate: '+name)
print(json.dumps(dict(status='PASS_NEWSEED_R80_CALIBRATION_IDENTITY_AND_SPHERE_CONTRACT',checks=checks,real_newseed_image_effect=False,new_GPU_calls=0)))
