"""Check the proposed contrast objective's algebra, not its scientific efficacy."""
import torch
from brainhive_paired_effect_DRAFT_20261025_v1 import paired_effect_loss


def main():
    generator=torch.Generator().manual_seed(1729)
    tp=torch.randn(3,2,4,5,generator=generator,requires_grad=True)
    tn=torch.randn(3,2,4,5,generator=generator,requires_grad=True)
    sp=torch.randn(3,2,4,5,generator=generator,requires_grad=True)
    sn=torch.randn(3,2,4,5,generator=generator,requires_grad=True)
    eps=1e-6
    loss,info=paired_effect_loss(sp,sn,tp,tn,epsilon=eps)
    shift=torch.randn(3,2,4,5,generator=generator)
    moved,_=paired_effect_loss(sp+shift,sn+shift,tp,tn,epsilon=eps)
    assert torch.allclose(loss,moved,atol=2e-6,rtol=2e-6)
    swapped,_=paired_effect_loss(sn,sp,tn,tp,epsilon=eps)
    assert torch.equal(loss,swapped)
    loss.backward()
    assert sp.grad.norm()>0 and sn.grad.norm()>0
    assert torch.equal(sp.grad,-sn.grad)
    assert tp.grad is None and tn.grad is None
    # Zero target contrast stays in the full three-query denominator.
    a=tp.detach().clone();b=tn.detach().clone();b[1]=a[1]
    masked,info=paired_effect_loss(sp.detach(),sn.detach(),a,b,epsilon=eps)
    assert info['valid_teacher'].tolist()==[True,False,True]
    assert info['per_query'][1]==0
    assert torch.equal(masked,info['per_query'].sum()/3)
    # Exact target differences minimize this response-space objective;
    # doubling a correct contrast must be penalized, unlike unbounded ranking.
    zero=torch.zeros_like(a)
    exact,_=paired_effect_loss(a-b,zero,a,b,epsilon=eps)
    over,_=paired_effect_loss(2*(a-b),zero,a,b,epsilon=eps)
    assert exact==0 and over>0
    assert not torch.cuda.is_initialized()
    print('DRAFT_PAIR_LOSS_CPU_ALGEBRA_PASSED_COMMON_SHIFT_SWAP_GRADIENT_MASK_OVERSHOOT_NO_FIT')


if __name__=='__main__':main()
