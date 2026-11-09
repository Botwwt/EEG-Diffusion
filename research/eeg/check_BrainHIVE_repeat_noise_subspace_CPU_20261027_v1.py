"""Critical mathematical checks of D23 before real training or held readout."""
import argparse,json
from pathlib import Path
import torch
import brainhive_repeat_noise_subspace_20261027_v1 as V
torch.set_num_threads(2);torch.manual_seed(1729)
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
x=torch.randn(100,4,12,dtype=torch.float64);res=x-x.mean(1,keepdim=True)
n=V.shrink_mean4_noise(res.flatten(0,1).T@res.flatten(0,1),100)
other=x[:,[3,0,2,1]];r=other-other.mean(1,keepdim=True);n2=V.shrink_mean4_noise(r.flatten(0,1).T@r.flatten(0,1),100);assert torch.allclose(n,n2,atol=1e-14,rtol=1e-14)
w=torch.randn(5,12);g=V.geometry(w,n);raw=torch.randn(20,12);bias=torch.randn(5);white=V.whiten(raw,g);q=g['Q0'].clone()
native=torch.nn.functional.linear(raw,w,bias);initial=V.residual_preactivation(raw,white,q,g,bias);assert torch.equal(initial,native) and torch.equal(V.folded_weight(q,g),w)
grad=torch.randn_like(q);tangent=V.tangent_gradient(q,grad);assert (tangent@q.T+q@tangent.T).norm()<1e-5
updated,info=V.fixed_step(q,grad,True);assert info['linearized_risk_change']<0 and V.metric_error(V.folded_weight(updated,g),g)<1e-5
fold=torch.nn.functional.linear(raw,V.folded_weight(updated,g),bias);expanded=V.residual_preactivation(raw,white,updated,g,bias);assert torch.allclose(fold,expanded,atol=2e-6,rtol=2e-6)
zero,zi=V.fixed_step(q,torch.zeros_like(q),True);assert torch.equal(zero,q)
perm=torch.randperm(12);pg=V.geometry(w,n[perm][:,perm]);assert V.metric_error(w,pg)<1e-6
qfree,_=V.fixed_step(q,grad,False);assert (qfree-updated).norm()>1e-5
rawmean=(x.mean(1).float());assert V.whiten(rawmean,g).shape==(100,12)
assert not torch.cuda.is_initialized()
value=dict(status='D23_NINE_CRITICAL_MATH_CHECKS_PASSED',checks=9,scope=['four_repeat_covariance_permutation','exact_native_zero_update_and_fold','tangent_constraint','finite_metric_conservation','folded_native_linear_equivalence','zero_gradient_noop','permuted_metric_native_initialization','unconstrained_control_differs','mean4_only_inference'],GPU_jobs=0)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
