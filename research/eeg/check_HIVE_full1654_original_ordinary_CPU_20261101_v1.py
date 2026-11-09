"""Original-objective scope and independent head/Adam calculus; CPU fixtures only."""
import copy
import io
import math
from pathlib import Path
import torch
import brainhive_full1654_ordinary_engine_20261101_v1 as E
from audit_HIVE_full1654_ordinary_profile_CPU_20261101_v1 import explicit_head,manual_AdamW
from profile_HIVE_full1654_original_ordinary_GPU_20261101_v1 import exact


def run(s):
    E.validate_scope(s);torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    checks=[]
    def reject(name,key,value):
        other=copy.deepcopy(s);other[key]=value
        try:E.validate_scope(other)
        except AssertionError:checks.append(name)
        else:raise AssertionError('Expected rejection '+name)
    reject('no_formal_fit_admission','formal_training_admitted',True)
    reject('no_benchmark_admission','benchmark_admitted',True)
    reject('no_false_exposure','official200_previously_exposed',False)
    reject('no_independent_H','independent_H',True)
    reject('no_R80_payloads_in_profile','R80_payloads_allowed',True)
    reject('no_NEW_GPU','host','new')
    reject('no_future_GPU_exactness_claim','future_GPU_bit_exactness_claimed',True)
    method=copy.deepcopy(E.METHOD);method['old_Dfit_postfit_calibration']=True
    reject('no_recalibration_on_old_Dfit_now_train','method',method)
    method=copy.deepcopy(E.METHOD);method['objective']='F_cosine_only'
    reject('original_epsilon_token_objective_required','method',method)
    order=E.orders(torch);coverage=torch.zeros(3,165400,dtype=torch.bool);updates=exposures=0
    for epoch in range(3):
        assert len(torch.unique(order[epoch]))==165400
        for lo in range(0,165400,16):
            ids=order[epoch][lo:lo+16];assert not coverage[epoch,ids].any();coverage[epoch,ids]=True
            updates+=1;exposures+=len(ids)
        assert len(ids)==8
    assert coverage.all() and (updates,exposures)==(31014,496200)
    checks.append('actual_CPU_full3pass_order_coverage_31014_and496200')
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    torch.manual_seed(1729);model=BoundedConditionAdapter(predict_risk=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    generator=torch.Generator().manual_seed(9182)
    records=[];parameter_max=gradient_max=0.
    for step in range(4):
        c=torch.nn.functional.layer_norm(torch.randn(16,1024,generator=generator),(1024,))
        weight=torch.ones_like(c);bias=torch.zeros_like(c)
        upstream=torch.randn(16,1024,generator=generator)/512
        prior=E.clone_CPU(model.state_dict(),torch) if step==0 else records[-1]['model_after']
        prior_opt=E.clone_CPU(optimizer.state_dict(),torch) if step==0 else records[-1]['optimizer_after']
        optimizer.zero_grad(set_to_none=True);changed=model(c,weight,bias)[0]
        (changed*upstream).sum().backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        grad={name:p.grad.detach().clone() for name,p in model.named_parameters()}
        parameters={name:value.double().requires_grad_(True) for name,value in prior.items()}
        manual=explicit_head(parameters,c.double(),weight.double(),bias.double(),torch)
        assert torch.allclose(manual,changed.double(),**s['numerical_guard'])
        gradients=torch.autograd.grad((manual*upstream.double()).sum(),tuple(parameters.values()))
        raw=float(torch.stack([v.square().sum() for v in gradients]).sum().sqrt())
        assert math.isclose(raw,float(norm),rel_tol=5e-6,abs_tol=1e-5)
        coef=min(1.,1./(raw+1e-6))
        for name,value in zip(parameters,gradients):
            assert torch.allclose(value*coef,grad[name].double(),**s['numerical_guard'])
            gradient_max=max(gradient_max,float((value*coef-grad[name].double()).abs().max()))
        expected,moments=manual_AdamW(prior,prior_opt,grad,torch)
        optimizer.step()
        for name,value in model.state_dict().items():
            assert torch.allclose(value.double(),expected[name],**s['numerical_guard'])
            parameter_max=max(parameter_max,float((value.double()-expected[name]).abs().max()))
        record=dict(model_before=prior,optimizer_before=prior_opt,clipped_parameter_gradients=grad,
                    model_after=E.clone_CPU(model.state_dict(),torch),optimizer_after=E.clone_CPU(optimizer.state_dict(),torch),
                    condition=c,weight=weight,bias=bias,changed_condition=changed.detach(),changed_input_gradient=upstream,
                    predicted=torch.zeros(16,4,64,64),noise=torch.ones(16,4,64,64),
                    tokens=torch.ones(16,4,2048),teacher_tokens=torch.ones(16,4,2048))
        records.append(record)
    checks.append('four_CPU_fixture_head_VJPs_and16_Adam_parameter_transitions_match_independent_algebra')
    stream=io.BytesIO();torch.save(records,stream)
    trace_bytes=stream.getbuffer().nbytes
    # Separate final resume state is below7MiB; reserve2MiB for actual metadata.
    assert trace_bytes+(7<<20)+(2<<20)<64<<20
    checks.append('full_shape_trace_with_shared_snapshots_fits64MiB_planning_cap')
    state=E.clone_CPU(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state()),torch)
    stream=io.BytesIO();torch.save(state,stream);stream.seek(0)
    assert exact(state,torch.load(stream,weights_only=True,map_location='cpu'),torch)
    checks.append('CPU_model_Adam_RNG_serialization_roundtrip_exact_without_future_step_claim')
    assert not torch.cuda.is_initialized()
    return dict(status='PASS_CPU_ORIGINAL_ORDINARY_SCOPE_AND_INDEPENDENT_CALCULUS_FIXTURES',
                checks=checks,CPU_fixture_updates=4,CPU_fixture_formal_training_use_forbidden=True,
                full_shape_trace_fixture_bytes=trace_bytes,head_VJP_maximum_error=gradient_max,
                Adam_parameter_maximum_error=parameter_max,actual_frozen_UNet_loaded=False,
                actual_GPU_profile_performed=False,actual_formal_updates=0,new_GPU_jobs=0,
                source_method=E.METHOD)
