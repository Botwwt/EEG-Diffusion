"""Meaningful numerical checks: exact fallback, bound, affine support, gradients."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from brainhive_native_models_v1 import *
from brainhive_bounded_component_v1 import BoundedConditionAdapter


def main():
    torch.set_num_threads(2);torch.manual_seed(1729)
    report={}
    u=torch.randn(32,1024);u=F.layer_norm(u,(1024,))
    weight=torch.linspace(.9,1.1,1024).expand(32,-1);bias=torch.linspace(-.02,.02,1024).expand(32,-1)
    c=u*weight+bias
    for name,with_risk in [('ordinary',False),('risk',True)]:
        model=BoundedConditionAdapter(predict_risk=with_risk)
        initial,info=model(c,weight,bias)
        assert torch.equal(initial,c)
        initial.square().mean().backward()
        assert model.direction.weight.grad is not None and model.direction.weight.grad.norm()>0
        with torch.no_grad():model.direction.weight.normal_(0,10);model.direction.bias.normal_(0,10)
        output,info=model(c,weight,bias)
        assert torch.equal(model(c,weight,bias,enabled=False)[0],c)
        assert torch.all(info['standardized_delta_norm']<=.25*info['standardized_input_norm']+1e-5)
        transformed=(output-bias)/weight
        torch.testing.assert_close(transformed.norm(dim=-1),u.norm(dim=-1),rtol=1e-5,atol=1e-5)
        assert transformed.mean(-1).abs().max()<1e-6
        report[name]=dict(parameters=sum(p.numel() for p in model.parameters()),zero_init_exact=True,
            disabled_exact=True,nonzero_initial_direction_gradient=True,large_weights_bound_passed=True,
            output_affine_support_mean_and_norm_passed=True)
    ratio=report['risk']['parameters']/report['ordinary']['parameters']
    assert ratio<1.001
    out=WORK/'audit/bounded_component_CPU_v1.json';assert not out.exists()
    write(out,dict(status='BOUNDED_COMPONENT_EXACT_FALLBACK_BOUND_AND_GRADIENT_CHECKS_PASSED_CPU',
        modules=report,capacity_ratio=ratio,source_sha256=sha(ROOT/'brainhive_bounded_component_v1.py'),
        real_data_fitted=False,scientific_gain_established=False))
    print(report,flush=True)


if __name__=='__main__':main()
