"""Small algebra/gradient contracts for a significant new training path."""
import json
import numpy as np
import torch
from brainhive_paired_semantic_effect_20261030_v1 import paired_objective


def main():
    torch.set_num_threads(2);torch.manual_seed(1729);checks=[]
    x=torch.randn(2,9,requires_grad=True);v=torch.randn(2,9)
    loss,r=paired_objective(x,v)
    g=r['features'].detach().numpy().astype(np.float64);t=r['targets'].numpy().astype(np.float64)
    d=float((g[0]-g[1])@(t[0]-t[1]));expected=1-float((g*t).sum(1).mean())+.07*np.logaddexp(0,-d/.07)
    assert abs(float(loss)-expected)<1e-6;checks.append('independent_float64_objective')
    swapped,_=paired_objective(x.flip(0),v.flip(0));assert torch.equal(loss,swapped);checks.append('swap_both_invariant')
    _,wrong=paired_objective(x,v.flip(0));assert torch.allclose(wrong['effect'],-r['effect']);checks.append('wrong_target_reverses_effect')
    dx=torch.autograd.grad(loss,x)[0];assert torch.isfinite(dx).all() and dx.norm()>0;checks.append('nonzero_finite_gradient')
    # Normalization has a known analytic derivative, independent of autograd.
    a=g.copy();dv=t[0]-t[1];sigmoid=1/(1+np.exp(d/.07))
    da=-t/2;da[0]-=sigmoid*dv;da[1]+=sigmoid*dv
    raw=x.detach().numpy().astype(np.float64);norm=np.linalg.norm(raw,axis=1,keepdims=True)
    analytic=(da-a*(da*a).sum(1,keepdims=True))/norm
    assert np.max(np.abs(dx.numpy()-analytic))<1e-6;checks.append('independent_normalization_chain_gradient')
    z=torch.randn(1,9).repeat(2,1).requires_grad_();_,collapsed=paired_objective(z,v)
    gc=torch.autograd.grad(collapsed['pairing'],z)[0]
    assert torch.allclose(gc[0],-gc[1]) and gc.norm()>0;checks.append('collapsed_pair_has_opposite_identifiable_gradients')
    _,identical=paired_objective(x,v[:1].repeat(2,1))
    assert float(identical['effect'])==0 and abs(float(identical['pairing'])-.07*np.log(2))<1e-7;checks.append('identical_visual_targets_have_no_effect_signal')
    _,terms=paired_objective(x,v);common=torch.randn(9,dtype=x.dtype)
    assert torch.allclose(((terms['features'][0]+common)-(terms['features'][1]+common))@(terms['targets'][0]-terms['targets'][1]),terms['effect'],atol=2e-6);checks.append('shared_additive_feature_component_cancels')
    # This cancellation is after features are formed; a nonlinear generator
    # does NOT guarantee its entire random-noise influence is additive.
    pure,p=paired_objective(x,v,effect_enabled=False)
    assert torch.equal(pure,p['absolute']);checks.append('absolute_only_matched_control')
    assert not torch.cuda.is_initialized();checks.append('CPU_only_no_CUDA')
    print(json.dumps(dict(status='PASS_PAIRED_SEMANTIC_ALGEBRA_ONLY',checks=checks,new_GPU_jobs=0,new_models=0,new_images=0)))


if __name__=='__main__':main()
