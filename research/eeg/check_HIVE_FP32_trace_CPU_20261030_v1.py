"""Positive and deliberately wrong-model/geometry/activation rejection checks."""
import copy,json
import numpy as np
from scipy.special import erf
from brainhive_FP32_trace_audit_CPU_20261030_v1 import verify,merge,bit_equal
def run():
    rng=np.random.default_rng(1729);c=rng.normal(size=(16,1024)).astype(np.float32);g=rng.uniform(.7,1.2,size=c.shape).astype(np.float32);b=rng.normal(scale=.1,size=c.shape).astype(np.float32)
    w={'trunk.0.weight':rng.normal(scale=.05,size=(256,1024)).astype(np.float32),'trunk.0.bias':rng.normal(scale=.1,size=256).astype(np.float32),'direction.weight':rng.normal(scale=.1,size=(1024,256)).astype(np.float32),'direction.bias':rng.normal(scale=.1,size=1024).astype(np.float32)}
    u=(c-b)/g;z=u@w['trunk.0.weight'].T+w['trunk.0.bias'];t=z*np.float32(np.sqrt(.5));h=(z*np.float32(.5))*(np.float32(1)+erf(t.astype(np.float64)).astype(np.float32));raw=h@w['direction.weight'].T+w['direction.bias'];y=c+g*(np.float32(.25)*raw)
    trace=dict(u=u,z=z,h=h,raw=raw,output=y);accepted=verify(trace,w,c,g,b,.25);checks=['independent_correct_FP32_trace']
    def rejected(label,changed=None,weights=None,radius=.25):
        try:verify(changed if changed is not None else trace,weights if weights is not None else w,c,g,b,radius)
        except AssertionError:checks.append(label)
        else:raise AssertionError('Incorrect computation accepted: '+label)
    wrong=copy.deepcopy(w);wrong['trunk.0.weight']=np.roll(wrong['trunk.0.weight'],1,axis=1);rejected('permuted_first_layer_weights_rejected',weights=wrong)
    wrong=copy.deepcopy(w);wrong['direction.weight']=np.roll(wrong['direction.weight'],1,axis=1);rejected('permuted_last_layer_weights_rejected',weights=wrong)
    wrong=copy.deepcopy(trace);wrong['z']+=np.float32(.01);rejected('linear_output_corruption_rejected',changed=wrong)
    wrong=copy.deepcopy(trace);wrong['u']=wrong['u']*np.float32(1.01);rejected('wrong_affine_coordinates_rejected',changed=wrong)
    wrong=copy.deepcopy(trace);wrong['h']=np.float32(.5)*z*(np.float32(1)+np.tanh(np.float32(np.sqrt(2/np.pi))*(z+np.float32(.044715)*z**3)));rejected('tanh_GELU_branch_rejected',changed=wrong)
    wrong=copy.deepcopy(trace);v=(y-b)/g;v*=np.linalg.norm(u,axis=1,keepdims=True)/np.linalg.norm(v,axis=1,keepdims=True);wrong['output']=b+g*v;rejected('unregistered_norm_restoration_rejected',changed=wrong)
    rejected('wrong_residual_strength_rejected',radius=.125)
    wrong=copy.deepcopy(trace);wrong['output'][0,0]=np.nan;rejected('nonfinite_output_rejected',changed=wrong)
    assert not bit_equal(np.array([0.],np.float32),np.array([-0.],np.float32));checks.append('signed_zero_bit_equality_is_strict')
    assert merge([accepted])['queries']==16;checks.append('full_denominator_aggregate_preserved')
    result=dict(status='PASS_TYPED_FP32_TRACE_CHECKS_WITH_WRONG_MODEL_GEOMETRY_AND_ACTIVATION_REJECTIONS',checks=checks,new_GPU_calls=0,trained_models_used=False,original_whole64_gate_reclassified=False)
    print(json.dumps(result));return result
if __name__=='__main__':run()
