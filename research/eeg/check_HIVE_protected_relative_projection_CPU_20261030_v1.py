"""Meaningful analytic constraints, degeneracies and sampled optimum checks."""
import json
from pathlib import Path
import numpy as np
from brainhive_protected_relative_projection_CPU_20261030_v1 import protected_relative_projection as project
rng=np.random.default_rng(1729);n,d=24,1024
un=rng.normal(size=(n,d));un-=un.mean(axis=1,keepdims=True);un*=32/np.linalg.norm(un,axis=1,keepdims=True)
g=rng.uniform(.7,1.3,size=(n,d));b=rng.normal(scale=.03,size=(n,d));m=rng.normal(size=d);m/=np.linalg.norm(m)
direction=rng.normal(size=(n,d));direction-=direction.mean(axis=1,keepdims=True);direction-=un*np.sum(direction*un,axis=1,keepdims=True)/1024;direction/=np.linalg.norm(direction,axis=1,keepdims=True)
u0=un+32*.21*direction;u0*=32/np.linalg.norm(u0,axis=1,keepdims=True)
native=b+g*un;ordinary=b+g*u0;initial=b+g*(un+.1*rng.normal(size=(n,d)));relative=initial+g*rng.normal(scale=1.4,size=(n,d));checks=[]
out,info=project(native,ordinary,initial,initial,g,b,m);assert np.array_equal(out,ordinary);checks.append('ZERO_FIELD_INCREMENT_EXACT_ORDINARY_BYTES')
out,_=project(native,ordinary,initial,relative,g,b,m,enabled=False);assert np.array_equal(out,ordinary);checks.append('DISABLED_EXACT_ORDINARY_BYTES')
out,info=project(native,ordinary,initial,relative,g,b,m)
assert max(info['affine_norm_relative_error'])<1e-12 and max(info['affine_mean_error'])<1e-12 and max(info['protected_linear_score_error'])<1e-11
assert min(info['native_cosine_after'])>=info['native_cosine_lower_bound']-1e-12;checks.append('ACTUAL_AFFINE_NORM_MEAN_LINEAR_SCORE_AND_ORIGINAL_NATIVE_CAP')
arrays=[x.astype(np.float32) for x in (native,ordinary,initial,relative,g,b)]
out32,proof32=project(*arrays,m);assert max(proof32['affine_norm_relative_error'])<2e-7 and max(proof32['protected_linear_score_error'])<2e-6 and min(proof32['native_cosine_after'])>=proof32['native_cosine_lower_bound']-2e-7;checks.append('FP32_CAST_WITH_EXPLICIT_SMALL_ROUNDING_ERRORS')
anti=initial+g*(-un-u0);_,tie=project(native,ordinary,initial,anti,g,b,m);assert tie['degenerate_direction_ties'].all();checks.append('ANTIPODAL_OBJECTIVE_TIE_DETERMINISTIC_NULL_BASIS')
outtie,_=project(native,ordinary,initial,anti,g,b,m);outtie2,_=project(native,ordinary,initial,anti,g,b,m);assert np.array_equal(outtie,outtie2);checks.append('DEGENERATE_TIE_REPEAT_IDENTICAL')
try:project(native[:1],(-native[:1]),initial[:1],relative[:1],np.ones((1,d)),np.zeros((1,d)),un[0]/32)
except (ValueError,AssertionError):checks.append('INFEASIBLE_PROTECTED_PLANE_REJECTED_NO_RADIUS_RELAXATION')
else:raise AssertionError('Infeasible cap accepted')
gap=-np.inf
for i in range(4):
    e1=np.ones(d)/32;a=g[i]*m;a-=e1*(a@e1);e2=a/np.linalg.norm(a)
    null=lambda x:x-np.outer(x@e1,e1)-np.outer(x@e2,e2) if x.ndim==2 else x-e1*(x@e1)-e2*(x@e2)
    p=u0[i]-null(u0[i]);R=np.sqrt(u0[i]@u0[i]-p@p);axis=null(un[i]);hn=np.linalg.norm(axis);axis/=hn
    tau=(np.linalg.norm(un[i])*np.linalg.norm(u0[i])/np.sqrt(1+.25**2)-un[i]@p)/(R*hn)
    v=u0[i]+(relative[i]-initial[i])/g[i];candidate=(out[i]-b[i])/g[i]
    random=null(rng.normal(size=(2000,d)));random-=np.outer(random@axis,axis);random/=np.linalg.norm(random,axis=1,keepdims=True)
    cosines=rng.uniform(max(-1,tau),1,size=2000);z=cosines[:,None]*axis+np.sqrt(1-cosines**2)[:,None]*random
    feasible=p+R*z;difference=feasible@v-candidate@v;assert difference.max()<1e-9;gap=max(gap,float(difference.max()))
checks.append('8000_INDEPENDENT_RANDOM_FEASIBLE_POINTS_DO_NOT_BEAT_CLOSED_OPTIMUM')
root=Path(__file__).resolve().parent/'audit/HIVE_protected_relative_projection_CPU_checks_20261030_v2';root.mkdir(exist_ok=False)
r=dict(status='PASS_EIGHT_CLASSICAL_PROJECTION_CPU_CHECKS',checks=checks,checks_count=len(checks),sampled_feasible_points=8000,max_sampled_advantage=gap,
    FP32_max_norm_relative_error=float(max(proof32['affine_norm_relative_error'])),FP32_max_protected_score_error=float(max(proof32['protected_linear_score_error'])),
    actual_EEG_inputs_tested=False,actual_AP_response_preserved=False,real_method_or_image_gain_not_proven=True,new_training_updates=0,new_GPU_calls=0)
(root/'completion.json').write_bytes((json.dumps(r,indent=2)+'\n').encode());print(json.dumps(r))
