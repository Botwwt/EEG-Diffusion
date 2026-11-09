import json
import numpy as np
from brainhive_native_chart_inverse_CPU_20261030_v1 import chart_parameters,inverse_direction,layernorm_preimage
rng=np.random.default_rng(1729);checks=[]
for dimension in (7,1024):
    w=rng.normal(size=(40,dimension));w-=w.mean(axis=-1,keepdims=True);w/=np.linalg.norm(w,axis=-1,keepdims=True)
    gamma=rng.uniform(.7,1.3,size=dimension);bias=rng.normal(0,.01,size=dimension)
    fgamma=rng.uniform(.8,1.2,size=dimension);fbias=rng.normal(0,.01,size=dimension);radius=np.sqrt(dimension)*.99
    c,u,certificate=inverse_direction(w,gamma,bias,fgamma,fbias,radius)
    actual=(c-fbias)/fgamma;actual-=actual.mean(axis=-1,keepdims=True);actual/=np.linalg.norm(actual,axis=-1,keepdims=True)
    assert np.max(abs(actual-w))<1e-12 and np.max(abs(np.linalg.norm(u,axis=-1)-radius))<1e-12
    for row in range(40):
        roots=np.roots([certificate['quadratic_squared'][row],2*certificate['quadratic_cross'][row],-certificate['positive_radius_gap']])
        assert abs(max(roots)-certificate['ray_distance'][row])<1e-12
    x=layernorm_preimage(u,1e-6);out=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)
    assert np.max(abs(out-u))<1e-12
checks+=['random7_and1024_dim_actual_affine_direction_and_radius_roundtrip','positive_root_matches_independent_polynomial_eigensolver','finite_epsilon_LayerNorm_preimage_matches_direct_forward']
w=np.array([[1.,-1.,0.,0.]]);w/=np.linalg.norm(w,axis=-1,keepdims=True)
c,u,cert=inverse_direction(w,np.ones(4),np.zeros(4),np.ones(4),np.zeros(4),1.5)
assert np.max(abs(c-1.5*w))<1e-14;checks.append('identical_affine_charts_reduce_to_radius_times_direction')
def reject(call):
    try:call()
    except ValueError:return
    raise AssertionError('Unproven chart must fail closed')
reject(lambda:chart_parameters(np.ones(4),np.array([5.,-5.,0.,0.]),np.ones(4),np.zeros(4),1.5));checks.append('outside_origin_rejects_global_coverage_claim')
reject(lambda:chart_parameters(np.array([1.,1.,-1.,-1.]),np.zeros(4),np.ones(4),np.zeros(4),1.5));checks.append('singular_centered_projection_rejected')
reject(lambda:chart_parameters(np.array([0.,1.,1.,1.]),np.zeros(4),np.ones(4),np.zeros(4),1.5));checks.append('zero_affine_scale_rejected')
reject(lambda:chart_parameters(np.ones(4),np.zeros(4),np.ones(4),np.zeros(4),2.));checks.append('unattainable_exact_unit_variance_radius_rejected')
print(json.dumps(dict(status='PASS_CLASSICAL_AFFINE_LAYERNORM_DIRECTION_INVERSE_CPU',checks=checks,new_GPU_calls=0)))
