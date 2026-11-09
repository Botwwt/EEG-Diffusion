"""Actual4-step loss/CPU64 head VJP/Adam algebra audit; never claims UNet replay."""
import argparse
from pathlib import Path
import time
import brainhive_full1654_ordinary_engine_20261101_v1 as E
K=E.K


def explicit_head(state,c,weight,bias,torch):
    import torch.nn.functional as F
    u=(c-bias)/weight
    h=F.gelu(F.linear(u,state['trunk.0.weight'],state['trunk.0.bias']))
    d=F.linear(h,state['direction.weight'],state['direction.bias'])
    d=d-d.mean(-1,keepdim=True)
    r2=u.square().sum(-1,keepdim=True)
    d=d-(d*u).sum(-1,keepdim=True)/r2*u
    delta=.25*r2.sqrt()*d/(r2+d.square().sum(-1,keepdim=True)).sqrt()
    scale=r2.sqrt()/(u+delta).norm(dim=-1,keepdim=True)
    return c+weight*((scale-1)*u+scale*delta)


def manual_AdamW(before,optimizer,gradient,torch):
    group=optimizer['param_groups'][0]
    assert (group['lr'],group['weight_decay'],tuple(group['betas']),group['eps'])==(1e-4,.01,(.9,.999),1e-8)
    beta1,beta2=group['betas'];result={};moments={}
    assert len(group['params'])==len(before)==4
    for key,index in zip(before,group['params']):
        p=before[key].double();g=gradient[key].double();old=optimizer['state'].get(index)
        m=old['exp_avg'].double() if old else torch.zeros_like(g)
        v=old['exp_avg_sq'].double() if old else torch.zeros_like(g)
        step=int(old['step'])+1 if old else 1
        m=beta1*m+(1-beta1)*g;v=beta2*v+(1-beta2)*g.square()
        result[key]=p*(1-group['lr']*group['weight_decay'])-group['lr']*(m/(1-beta1**step))/((v/(1-beta2**step)).sqrt()+group['eps'])
        moments[index]=dict(exp_avg=m,exp_avg_sq=v,step=step)
    return result,moments


def run(protocol):
    started=time.monotonic();s=E.setup_profile(protocol)
    import numpy as np
    import torch
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    root=Path(s['output_root'])/'profile';done=K.read(root/'completion.json')
    assert done['status']=='PASS_ACTUAL_FULL1654_ORIGINAL_ORDINARY_ENGINEERING_ONLY' and done['temporary_updates']==4
    trace=torch.load(K.check(done['trace']),weights_only=True,map_location='cpu')
    assert len(trace)==4
    expected_order=E.orders(torch)[0][:64]
    errors=[]
    for step,row in enumerate(trace,1):
        assert torch.equal(row['indices'],expected_order[16*(step-1):16*step])
        assert row['batch']==16 and row['mean_gate']==1.0
        image=float(np.square(row['predicted'].numpy().astype(np.float64)-row['noise'].numpy().astype(np.float64)).mean())
        a,b=row['tokens'].numpy().astype(np.float64),row['teacher_tokens'].numpy().astype(np.float64)
        token=float((1-(a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1))).mean())
        assert np.isclose(image,row['epsilon_loss'],**s['numerical_guard'])
        assert np.isclose(token,row['token_loss'],**s['numerical_guard'])
        assert np.isclose(image+.1*token,row['loss'],**s['numerical_guard'])
        values={key:value.double().requires_grad_(True) for key,value in row['model_before'].items()}
        changed=explicit_head(values,row['condition'].double(),row['weight'].double(),row['bias'].double(),torch)
        forward_error=float((changed-row['changed_condition'].double()).abs().max())
        assert torch.allclose(changed,row['changed_condition'].double(),**s['numerical_guard'])
        gradients=torch.autograd.grad((changed*row['changed_input_gradient'].double()).sum(),tuple(values.values()))
        raw_norm=torch.stack([value.square().sum() for value in gradients]).sum().sqrt()
        assert np.isclose(float(raw_norm),row['gradient_norm_before_clip'],**s['numerical_guard'])
        coefficient=min(1.,1./(float(raw_norm)+1e-6))
        grad_error=0.
        for key,gradient in zip(values,gradients):
            expected=gradient*coefficient;actual=row['clipped_parameter_gradients'][key].double()
            assert torch.allclose(expected,actual,**s['numerical_guard'])
            grad_error=max(grad_error,float((expected-actual).abs().max()))
        after,moments=manual_AdamW(row['model_before'],row['optimizer_before'],row['clipped_parameter_gradients'],torch)
        parameter_error=moment_error=0.
        for key,value in after.items():
            actual=row['model_after'][key].double()
            assert torch.allclose(value,actual,**s['numerical_guard'])
            parameter_error=max(parameter_error,float((value-actual).abs().max()))
        for key,parts in moments.items():
            actual=row['optimizer_after']['state'][key];assert int(actual['step'])==parts['step']==step
            for field in ('exp_avg','exp_avg_sq'):
                assert torch.allclose(parts[field],actual[field].double(),**s['numerical_guard'])
                moment_error=max(moment_error,float((parts[field]-actual[field].double()).abs().max()))
        errors.append(dict(step=step,loss_recalculation_error=abs(image+.1*token-row['loss']),
                           CPU64_condition_forward_maxerror=forward_error,CPU64_head_VJP_maxerror=grad_error,
                           CPU64_Adam_parameter_maxerror=parameter_error,CPU64_Adam_moment_maxerror=moment_error))
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_ACTUAL_FOUR_STEP_ORDINARY_LOSS_HEAD_VJP_AND_ADAM_CPU_AUDIT',
                protocol=K.ref(protocol),profile_completion=K.ref(root/'completion.json'),errors=errors,
                temporary_updates_audited=4,head_parameter_transitions_audited=16,
                frozen_UNet_gradient_replayed_on_CPU=False,future_GPU_bit_exactness_NOT_proved=True,
                formal_training_admitted=False,R80_effect_read=False,seconds=time.monotonic()-started)
    K.write(root/'independent_CPU_audit.json',result)
    print(__import__('json').dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);run(parser.parse_args().protocol)
