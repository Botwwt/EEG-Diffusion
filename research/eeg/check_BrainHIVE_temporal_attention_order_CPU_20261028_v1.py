"""Synthetic algebra only: centered D37 correction starts at order two.

The lambda values below are mathematical checks, not EEG image candidates or
an empirical strength scan. The registered image method remains unchanged.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import torch
import brainhive_temporal_conditional_attention_20261028_v1 as M


def run(output):
    torch.set_num_threads(2);torch.manual_seed(1729)
    q=torch.randn(1,2,3,6,dtype=torch.float64);k=torch.randn(1,2,4,6,dtype=torch.float64)
    dk=torch.randn(1,2,4,5,6,dtype=torch.float64);dv=torch.randn_like(dk)
    dk-=dk.mean(3,keepdim=True);dv-=dv.mean(3,keepdim=True)
    a=(q@k.transpose(-1,-2)/math.sqrt(6)).softmax(-1)
    s=torch.einsum('bhnd,bhktd->bhnkt',q,dk)/math.sqrt(6)
    coefficient=torch.einsum('bhnk,bhnkt,bhktd->bhnd',a,s,dv)/5
    projection=torch.randn_like(coefficient)
    def scalar(lam):return (M.attention_delta(q,k,lam*dk,lam*dv)*projection).sum()
    zero=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    first=torch.autograd.functional.jacobian(scalar,zero)
    second=torch.autograd.functional.hessian(scalar,zero)
    second_expected=2*(coefficient*projection).sum()
    assert first.abs()<1e-12 and (second-second_expected).abs()<1e-12
    small=1e-4;at_small=M.attention_delta(q,k,small*dk,small*dv)
    coefficient_relative_error=float((at_small/small**2-coefficient).norm()/coefficient.norm());assert coefficient_relative_error<1e-3
    samples=[]
    for lam in [.2,.1,.05,.025]:
        current=M.attention_delta(q,k,lam*dk,lam*dv)
        flat=M.attention_delta(q,k,lam*dk,lam*dv,mode='flat_residual')
        samples.append(dict(synthetic_lambda=lam,conditional_norm=float(current.norm()),flat_minus_conditional_norm=float((flat-current).norm())))
    ratio=samples[-2]['flat_minus_conditional_norm']/samples[-1]['flat_minus_conditional_norm'];assert 12<ratio<20
    # Each centered five-vector cross moment has rank at most four; four token
    # contributions together have rank at most sixteen, before later UNet maps.
    matrices=torch.einsum('bhktd,bhkte->bhkde',dv,dk)/(5*math.sqrt(6))
    ranks=torch.linalg.matrix_rank(matrices,atol=1e-10,rtol=0);assert ranks.max()<=4
    # Independent K/V does not guarantee a positive semidefinite cross moment.
    negative=-(dk.transpose(-1,-2)@dk)/5
    eig=torch.linalg.eigvalsh(negative);assert eig.min()<0 and eig.max()<1e-10
    value=dict(status='D37_CENTERED_ATTENTION_SECOND_ORDER_AND_CONTROL_FOURTH_ORDER_CHECKED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        first_derivative_at_zero=float(first),second_derivative_error=float((second-second_expected).abs()),
        conditional_lambda_squared_coefficient_relative_error_at_1e_minus4=coefficient_relative_error,
        flat_minus_conditional_halving_ratio=ratio,synthetic_samples=samples,maximum_per_token_cross_moment_rank=int(ranks.max()),
        cross_moment_need_not_be_PSD=True,actual_EEG_values_or_image_scores_read=False,
        actual_image_strength_changed=False,new_image_candidates=0,new_GPU_jobs=0,SGD_updates=0,
        conclusion='At a fixed actual Q, centered K and V make the first lambda derivative zero; leading correction is a rank<=4 per-token K/V cross moment, and flat-vs-conditional residual difference begins at lambda^4. Finite registered strength need not be in this local regime.')
    with output.open('x',encoding='utf8') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
