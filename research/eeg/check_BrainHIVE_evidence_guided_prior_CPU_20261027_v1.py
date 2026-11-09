"""D14 analytic velocity identity, production frozen-head hook and decision checks."""
import argparse
import math
import numpy as np
import torch
from torch import nn
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_evidence_guided_prior_20261027_v1 as V
import brainhive_noisy_EEG_evidence_20261027_v1 as E


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True); a=p.parse_args()
    torch.set_num_threads(2); torch.manual_seed(1729); checks=[]; errors=[]
    # Y|c is Gaussian. Compute both conditional and marginal velocities analytically.
    x=torch.linspace(-3.,3.,19,dtype=torch.float64); mu=1.2; variance=.7; marginal_variance=1.9
    for t in (.000001,.125,.375,.625,.875,.999):
        conditional_var=(1-t)**2+t*t*variance; marginal_var=(1-t)**2+t*t*marginal_variance
        sc=-(x-t*mu)/conditional_var; sm=-x/marginal_var
        vc=x/t+(1-t)/t*sc; vm=x/t+(1-t)/t*sm
        # Closed Gaussian regression of Y-Z given X.
        direct_c=mu+(t*variance-(1-t))/conditional_var*(x-t*mu)
        direct_m=(t*marginal_variance-(1-t))/marginal_var*x
        errors.append(float(((vc-vm)-(direct_c-direct_m)).abs().max()))
        assert torch.allclose(vc-vm,(1-t)/t*(sc-sm),atol=1e-8,rtol=1e-8)
    assert max(errors)<1e-8; checks.append('Gaussian marginal-plus-log-ratio velocity identity on registered path')
    t=1e-8; cv=(1-t)**2+t*t*variance; mv=(1-t)**2+t*t*marginal_variance
    stable=(1-t)*((-(x-t*mu)/cv+x/mv)/t)
    assert torch.allclose(stable,torch.full_like(x,mu),atol=1e-6); checks.append('finite t0 limit without dividing learned logit by zero')
    head=E.Evidence().eval(); nn.init.normal_(head.out.weight,std=.01); head.requires_grad_(False)
    leaf=torch.randn(4,1024,requires_grad=True); c=torch.randn(4,1024); subjects=torch.zeros(4,dtype=torch.long); t=torch.tensor([.125,.375,.625,.875])
    h,r=V.unscaled_evidence(head,leaf,t,c,subjects)
    assert torch.equal(r,t*h); checks.append('production hook captures exact original r=t*h')
    gr=torch.autograd.grad(r.sum(),leaf,retain_graph=True)[0]
    gh=torch.autograd.grad(h.sum(),leaf)[0]
    assert torch.allclose(gr/t[:,None],gh,atol=2e-8,rtol=2e-5)
    assert all(v.grad is None and not v.requires_grad for v in head.parameters()) and gh.norm()>0
    checks.append('frozen parameters retain actual state gradient and scaled-logit identity')
    h0,r0=V.unscaled_evidence(head,leaf,torch.zeros(4),c,subjects)
    assert torch.count_nonzero(r0)==0 and torch.autograd.grad(h0.sum(),leaf)[0].norm()>0
    assert len(head.out._forward_hooks)==0; checks.append('zero-time neutrality is structural and hooks are removed')
    class ConstantPrior(nn.Module):
        def forward(self,x,t): return torch.ones_like(x)*.1
    class ZeroHead(nn.Module):
        def __init__(self): super().__init__(); self.out=nn.Linear(1024,1,bias=False); nn.init.zeros_(self.out.weight)
        def forward(self,x,t,c,b,s,arm): return t*self.out(x).squeeze(-1)
    z=torch.randn(2,1024); args=(z,c[:2],subjects[:2],torch.zeros(1024),torch.tensor(1.),torch.ones(1024),torch.zeros(1024))
    zero=ZeroHead().eval().requires_grad_(False)
    f,x,tr=V.sample(ConstantPrior(),None,*args); fz,xz,trz=V.sample(ConstantPrior(),zero,*args)
    assert torch.equal(f,fz) and torch.equal(x,xz) and torch.allclose(x,z+.1,atol=4e-6)
    assert tr['states'].shape==(2,4,1024) and trz['diagnostics'].shape==(2,32,4)
    checks.append('production 32-step sampler equals analytic integration and zero-head bypass')
    assert sum(v.numel() for v in V.Marginal().parameters())==4483072
    checks.append('true marginal prior has no EEG or subject input and fixed capacity')
    queries=[dict(concept_id=str(i//10),stimulus_id=str(i)) for i in range(400)]
    zeros=np.zeros(80); positive=np.ones(80)
    null={k:zeros for k in ('DINO2AFC','target_cosine','within_concept_Top1')}
    # Use actual scorer key set from its array function.
    gallery=np.eye(400,768,dtype=np.float32); generated=gallery[:80,None].repeat(3,axis=1); baseline=generated.copy()
    out=V.score(generated,gallery,baseline,queries[:80],queries,80)
    assert out['decision']['action']=='EXPAND400'; checks.append('null stage80 cannot claim success')
    from unittest.mock import patch
    negative={key:dict(difference=-.1,descriptive_concept_bootstrap95=[-.12,-.08]) for key in U.S.METRICS}
    with patch.object(U.S,'compare',return_value=negative):
        assert V.score(generated,gallery,baseline,queries[:80],queries,80)['decision']['action']=='STOP_INSTANCE'
    checks.append('clear image harm stops before stage400')
    assert not torch.cuda.is_initialized()
    result=dict(status='D14_ALL_CPU_MATH_AND_PRODUCTION_CHECKS_PASSED',checks=len(checks),passed=checks,
        Gaussian_identity_max_absolute_error=max(errors),CUDA_initialized=False,real_fit_or_images=False,source=U.ref(__file__))
    U.atomic(a.output,result); print(result)


if __name__=='__main__': main()
