"""Critical sampling covariance, matched control and exact-off checks."""
import argparse,json
from pathlib import Path
import torch
import brainhive_conditional_noise_geometry_20261028_v1 as V

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(1729);checks={}
    def check(name,value):assert value,name;checks[name]=True
    z=torch.randn(1,4,32,32);d=torch.randn_like(z)
    check('isotropic_returns_exact_same_tensor',V.transform_ancestral_noise(z,d,'isotropic') is z)
    check('zero_response_returns_exact_same_tensor',V.transform_ancestral_noise(z,torch.zeros_like(d),'response_null') is z)
    v,mask=V.unit_direction(d,'response_null');vv,_=V.unit_direction(d,'orthogonal_control')
    check('matched_control_is_unit_and_perpendicular',((v*vv).sum(1).abs().max()<2e-7) and ((vv.square().sum(1)-1).abs().max()<3e-7))
    got=V.transform_ancestral_noise(z,d,'response_null');check('protected_local_response_dot_is_zero_to_FP32',((got*v).sum(1).abs().max()<1e-6))
    zz=z.half();dd=d.half();mixed=dd.clone();mixed[:,:,:8]=0
    out=V.transform_ancestral_noise(zz,mixed,'response_null');check('mixed_zero_pixels_bitwise_preserved',torch.equal(out[:,:,:8],zz[:,:,:8]))
    q=v[0,:,0,0].double();matrix=(4/3)*(torch.eye(4,dtype=torch.float64)-q[:,None]*q[None,:]);eigen=torch.linalg.eigvalsh(matrix)
    check('covariance_trace_four_and_rank_three',abs(float(matrix.trace())-4)<1e-6 and eigen[0].abs()<1e-6 and torch.allclose(eigen[1:],torch.full((3,),4/3,dtype=torch.float64),atol=1e-6))
    samples=torch.randn(1,4,1,120000);direction=torch.tensor([1.,0.,0.,0.]).reshape(1,4,1,1).expand_as(samples);draw=V.transform_ancestral_noise(samples,direction,'response_null').reshape(4,-1);cov=draw@draw.T/draw.shape[1]
    check('actual_sample_covariance_matches_closed_form',cov[0].abs().max()==0 and torch.allclose(cov[1:,1:],torch.eye(3)*(4/3),atol=.02))
    state=torch.get_rng_state();one=V.transform_ancestral_noise(z,d,'response_null');two=V.transform_ancestral_noise(z,d,'response_null');check('operator_consumes_no_extra_RNG_and_is_deterministic',torch.equal(state,torch.get_rng_state()) and torch.equal(one,two))
    condition=torch.randn(1,1024);copy=condition.clone();_ =V.transform_ancestral_noise(z,d,'orthogonal_control');check('condition_tensor_unmodified_and_no_condition_module_dependency',torch.equal(condition,copy))
    value=dict(status='D25_ALL9_SAMPLING_NOISE_GEOMETRY_CPU_CHECKS_PASSED',checks=checks,GPU_initialized=torch.cuda.is_initialized(),ordinary_changed=False,new_models=0,new_images=0)
    assert not value['GPU_initialized'];a.output.parent.mkdir(parents=True,exist_ok=True);assert not a.output.exists();a.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))

if __name__=='__main__':main()
